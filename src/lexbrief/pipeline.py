"""End-to-end brief pipeline on IN-Ext: load documents, fit on training folds, generate briefs.

Role sources:

* ``predicted`` - coarse roles and confidence (max marginal) from
  ``outputs/results/<brief.roles_run>/preds_inext.jsonl``.
* ``gold`` - IN-Ext has no full gold roles; the segment-inferred roles exist only for sentences
  aligned to a summary, so ``gold`` = segment-inferred role where available, predicted role
  elsewhere (confidence 1.0 for gold-labelled sentences). This leaks summary membership and is
  an optimistic upper bound, not a clean gold-role condition.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lexbrief.brief import baselines
from lexbrief.brief.budget import BudgetStrategy, make_strategy, role_stats
from lexbrief.brief.features import PositionPrior, compute_features
from lexbrief.brief.oracle import greedy_oracle
from lexbrief.brief.rouge2 import sentence_bigrams
from lexbrief.brief.scorer import PreparedDoc, grid_search, score
from lexbrief.brief.selector import select
from lexbrief.config import Config
from lexbrief.labels import INEXT_SEGMENT_TO_CODE, INEXT_TO_COARSE
from lexbrief.utils.io import read_json, read_jsonl

logger = logging.getLogger(__name__)

ANNOTATORS = ("A1", "A2")
_ALPHA_WORD = re.compile(r"[a-z]{2,}", re.IGNORECASE)
ROLE_MODES = ("predicted", "gold")


@dataclass
class BriefDoc:
    """One IN-Ext judgment with everything the brief generator needs."""

    doc_id: str
    texts: list[str]
    words: list[int]
    emb: np.ndarray
    gold_roles: list[str | None]
    pred_roles: list[str | None]
    pred_conf: list[float]
    in_summary: dict[str, list[bool]]
    refs: dict[str, str]
    ref_role_words: dict[str, dict[str, int]]
    _bigrams: list[Counter] | None = field(default=None, repr=False)
    _ref_bigrams: dict[str, Counter] | None = field(default=None, repr=False)

    def roles(self, mode: str) -> tuple[list[str | None], list[float]]:
        """Roles and confidences for a role mode (``predicted`` or ``gold``)."""
        if mode == "predicted":
            return self.pred_roles, self.pred_conf
        if mode == "gold":
            roles = [
                g if g is not None else p
                for g, p in zip(self.gold_roles, self.pred_roles, strict=True)
            ]
            conf = [
                1.0 if g is not None else c
                for g, c in zip(self.gold_roles, self.pred_conf, strict=True)
            ]
            return roles, conf
        raise ValueError(f"Unknown role mode {mode!r}; expected {ROLE_MODES}")

    @property
    def n_words(self) -> int:
        return sum(self.words)

    def eligible(self, min_alpha_words: int) -> list[bool]:
        """Selectable sentences: at least ``min_alpha_words`` alphabetic words (2+ letters).

        Rejects number/citation debris such as ``"a 2 5."`` while keeping short operative
        sentences such as ``"appeal dismissed."``.
        """
        return [len(_ALPHA_WORD.findall(t)) >= min_alpha_words for t in self.texts]

    def ref_words(self) -> float:
        """Mean reference length (words) over annotators."""
        return float(np.mean([len(self.refs[a].split()) for a in self.refs])) if self.refs else 0.0

    def bigrams(self, stemmer: bool = True) -> list[Counter]:
        if self._bigrams is None:
            self._bigrams = sentence_bigrams(self.texts, stemmer)
        return self._bigrams

    def ref_bigrams(self, stemmer: bool = True) -> dict[str, Counter]:
        if self._ref_bigrams is None:
            self._ref_bigrams = {
                a: sum(sentence_bigrams([t], stemmer), Counter()) for a, t in self.refs.items()
            }
        return self._ref_bigrams


def _load_predictions(path: Path) -> dict[str, dict[int, tuple[str, float]]]:
    preds: dict[str, dict[int, tuple[str, float]]] = {}
    for row in read_jsonl(path):
        conf = row.get("confidence")
        if conf is None:
            conf = max(row.get("probs") or [1.0])
        preds.setdefault(row["doc_id"], {})[row["idx"]] = (row["pred_coarse"], float(conf))
    return preds


def load_inext_docs(cfg: Config, require_predictions: bool = True) -> list[BriefDoc]:
    """Join ``inext.jsonl`` with cached embeddings and predicted roles."""
    from lexbrief.models.embed_cache import load_split_embeddings

    bc = cfg.brief
    raw = read_jsonl(Path(cfg.data.processed_dir) / "inext.jsonl")
    emb_data = load_split_embeddings(cfg.paths.emb_dir, bc.emb_run, "inext")
    pool = emb_data[bc.emb_pool].float().numpy()
    offsets = emb_data["offsets"].tolist()
    emb_by_doc = {d: pool[offsets[k] : offsets[k + 1]] for k, d in enumerate(emb_data["doc_ids"])}

    pred_path = Path(cfg.paths.results_dir) / bc.roles_run / "preds_inext.jsonl"
    if pred_path.is_file():
        preds = _load_predictions(pred_path)
    elif require_predictions:
        raise FileNotFoundError(
            f"{pred_path} not found; train and evaluate {bc.roles_run} first (Prompt 4)"
        )
    else:
        preds = {}

    docs = []
    for d in raw:
        sents = d["sentences"]
        p = preds.get(d["doc_id"], {})
        refs, ref_words = {}, {}
        for a in ANNOTATORS:
            summ = d["summaries"][a]
            refs[a] = " ".join(s["text"] for s in summ["full"])
            rw: Counter[str] = Counter()
            for folder, seg_sents in summ["segments"].items():
                coarse = INEXT_TO_COARSE[INEXT_SEGMENT_TO_CODE[folder]]
                rw[coarse] += sum(len(s.split()) for s in seg_sents)
            ref_words[a] = dict(rw)
        emb = emb_by_doc[d["doc_id"]]
        if emb.shape[0] != len(sents):
            raise ValueError(f"Embedding/sentence count mismatch for {d['doc_id']}")
        docs.append(
            BriefDoc(
                doc_id=d["doc_id"],
                texts=[s["text"] for s in sents],
                words=[len(s["text"].split()) for s in sents],
                emb=emb,
                gold_roles=[s.get("role_coarse") for s in sents],
                pred_roles=[p.get(s["idx"], (None, 0.0))[0] for s in sents],
                pred_conf=[p.get(s["idx"], (None, 0.0))[1] for s in sents],
                in_summary={a: [bool(s["in_summary"][a]) for s in sents] for a in ANNOTATORS},
                refs=refs,
                ref_role_words=ref_words,
            )
        )
    logger.info("Loaded %d IN-Ext docs (roles from %s)", len(docs), pred_path)
    return docs


def resolve_length(doc: BriefDoc, budget: float) -> int:
    """``budget`` <= 1 is a fraction of document words, otherwise absolute words."""
    return int(round(budget * doc.n_words)) if budget <= 1 else int(budget)


class BriefSystem:
    """Role-budgeted extractive summariser (A1/A2/A3 budget + linear scorer + selector)."""

    def __init__(self, cfg: Config, strategy: str = "A3", roles: str = "predicted") -> None:
        self.cfg = cfg
        self.bc = cfg.brief
        self.roles_mode = roles
        self.budget: BudgetStrategy = make_strategy(
            strategy,
            self.bc.expert_shares,
            self.bc.ruling_min_sentences,
            self.bc.min_sentences_per_role,
        )
        self.prior = PositionPrior(self.bc.pos_bins, self.bc.pos_laplace)
        self.weights: tuple[float, ...] = tuple(self.bc.weights)
        self.grid_table: list[dict[str, Any]] = []

    # --------------------------------------------------------------- fitting
    def fit(self, train: list[BriefDoc], search_weights: bool = True) -> BriefSystem:
        """Fit A3 shares, position priors and (optionally) the scorer weights on ``train``."""
        self.budget.fit(d.ref_role_words[a] for d in train for a in d.ref_role_words)
        samples = []
        for d in train:
            roles, _ = d.roles(self.roles_mode)
            n = len(d.texts)
            for i, r in enumerate(roles):
                if any(d.in_summary[a][i] for a in d.in_summary):
                    samples.append((r, i / max(1, n - 1)))
        self.prior.fit(samples)
        if search_weights:
            prepared = [self.prepare(d, int(round(d.ref_words()))) for d in train]
            self.weights, self.grid_table = grid_search(
                prepared, self.bc.weight_grid, self.bc.redundancy_threshold
            )
        return self

    # --------------------------------------------------------------- inference
    def prepare(self, doc: BriefDoc, length: int) -> PreparedDoc:
        roles, conf = doc.roles(self.roles_mode)
        # fragments are not candidates (same eligibility rule as the baselines and oracle)
        ok = doc.eligible(self.bc.min_alpha_words)
        roles = [r if e else None for r, e in zip(roles, ok, strict=True)]
        feats = compute_features(
            doc.texts,
            roles,
            conf,
            doc.emb,
            self.prior,
            self.bc.edge_threshold,
            self.bc.pagerank_damping,
        )
        budgets = self.budget.transform(role_stats(roles, doc.words), length)
        return PreparedDoc(
            texts=doc.texts,
            words=doc.words,
            roles=list(roles),
            features=feats.matrix,
            sim=feats.sim,
            budgets=budgets,
            sent_bigrams=doc.bigrams(self.bc.rouge_stemmer),
            ref_bigrams=list(doc.ref_bigrams(self.bc.rouge_stemmer).values()),
        )

    def generate(
        self, doc: BriefDoc, length: int, weights: tuple[float, ...] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Brief as ``{section: [{idx, text, score}]}`` in BRIEF_SECTION_ORDER."""
        p = self.prepare(doc, length)
        s = score(p.features, weights or self.weights)
        return select(p.texts, p.words, p.roles, s, p.budgets, p.sim, self.bc.redundancy_threshold)


def baseline_indices(name: str, doc: BriefDoc, length: int, cfg: Config) -> list[int]:
    """Indices chosen by a role-agnostic baseline (or the oracle against reference A1)."""
    bc = cfg.brief
    mw = doc.eligible(bc.min_alpha_words)
    if name == "lead":
        return baselines.lead(doc.words, length, mw)
    if name == "textrank":
        return baselines.textrank_baseline(
            doc.emb, doc.words, length, bc.edge_threshold, bc.pagerank_damping, mw
        )
    if name == "lexrank":
        return baselines.lexrank_baseline(doc.texts, doc.words, length, mw)
    if name == "mmr":
        return baselines.mmr_baseline(doc.emb, doc.words, length, bc.mmr_lambda, mw)
    if name in ("oracle", "oracle_A1", "oracle_A2"):
        ref = "A2" if name == "oracle_A2" else "A1"
        return greedy_oracle(
            doc.bigrams(bc.rouge_stemmer),
            doc.words,
            doc.ref_bigrams(bc.rouge_stemmer)[ref],
            length,
            mw,
        )
    raise ValueError(f"Unknown baseline {name!r}")


def fold_train_ids(cfg: Config, doc_id: str) -> list[str]:
    """Training document ids of the IN-Ext fold whose test split contains ``doc_id``."""
    folds = read_json(cfg.data.folds_file)["folds"]
    for f in folds:
        if doc_id in f["test"]:
            return list(f["train"])
    raise KeyError(f"{doc_id} is not in any IN-Ext fold ({cfg.data.folds_file})")
