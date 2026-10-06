"""Parse the IN-Ext corpus (Zenodo 7152317, ``dataset/IN-Ext``) into the canonical schema.

Real release layout (verified with ``lexbrief inspect-raw``)::

    IN-Ext/
      IN-EXT-length.txt                  "<file>\\t<int>" per judgment
      judgement/<id>.txt                 50 files, ONE SENTENCE PER LINE, lowercased,
                                         most punctuation removed
      summary/full/{A1,A2}/<id>.txt      headed sections (FACTS, ARGUMENT, ISSUE, STATUTE,
                                         ANALYSIS) followed by one sentence per line
      summary/segment-wise/{A1,A2}/{facts,argument,statute,analysis,judgement}/<id>.txt
                                         one paragraph per file (needs sentence splitting);
                                         a folder may lack a file when the segment is empty

The release has **no per-sentence gold rhetorical roles**. Roles are therefore inferred by
aligning summary sentences back to judgment sentences and voting over the summary segment(s)
they came from (``role_source = "segment_inferred"``). Unaligned sentences get ``None``.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from lexbrief.config import DataConfig
from lexbrief.data.align import align_sentences
from lexbrief.data.sentence_splitter import split_sentences
from lexbrief.labels import INEXT_HEADING_TO_COARSE, INEXT_SEGMENT_TO_CODE, INEXT_TO_COARSE

logger = logging.getLogger(__name__)

ANNOTATORS: tuple[str, ...] = ("A1", "A2")
ROLE_SOURCE = "segment_inferred"

_HEADING_TO_CODE = {
    "FACTS": "FAC",
    "ARGUMENT": "ARG",
    "ISSUE": "ISSUE",
    "STATUTE": "STA",
    "ANALYSIS": "Ratio",
}
# When votes tie, prefer the rarer / more specific role.
_TIE_PRIORITY = ("RULING", "ISSUES", "STATUTE", "ARGUMENTS", "PRECEDENT", "FACTS", "REASONING")


def read_text(path: Path) -> str:
    """Read a text file as UTF-8, falling back to cp1252."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def read_lines(path: Path) -> list[str]:
    """Non-empty stripped lines of a file."""
    return [ln.strip() for ln in read_text(path).splitlines() if ln.strip()]


def parse_full_summary(path: Path) -> list[dict[str, str | None]]:
    """Parse a ``summary/full`` file into ``[{heading, text}]`` (heading may be None)."""
    out: list[dict[str, str | None]] = []
    heading: str | None = None
    for line in read_lines(path):
        if line.upper() == line and line.strip().upper() in INEXT_HEADING_TO_COARSE:
            heading = line.strip().upper()
            continue
        out.append({"heading": heading, "text": line})
    return out


def parse_segment_summaries(seg_dir: Path, doc_id: str, spacy_model: str) -> dict[str, list[str]]:
    """Read ``segment-wise/<A>/<folder>/<doc_id>.txt`` for every known folder that has it."""
    out: dict[str, list[str]] = {}
    for folder in INEXT_SEGMENT_TO_CODE:
        path = seg_dir / folder / f"{doc_id}.txt"
        if path.is_file():
            text = read_text(path).strip()
            if text:
                out[folder] = split_sentences(text, model=spacy_model)
    return out


def _read_lengths(root: Path) -> dict[str, int]:
    path = root / "IN-EXT-length.txt"
    lengths: dict[str, int] = {}
    if path.is_file():
        for line in read_lines(path):
            parts = line.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                lengths[Path(parts[0]).stem] = int(parts[-1])
    return lengths


def _pick_role(votes: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    """Majority vote over (code, coarse) pairs with a deterministic tie-break."""
    if not votes:
        return None, None
    counts = Counter(coarse for _, coarse in votes)
    best = max(counts.values())
    tied = [c for c in counts if counts[c] == best]
    coarse = min(tied, key=lambda c: _TIE_PRIORITY.index(c) if c in _TIE_PRIORITY else 99)
    code = Counter(code for code, c in votes if c == coarse).most_common(1)[0][0]
    return code, coarse


def parse_document(root: Path, doc_id: str, cfg: DataConfig, lengths: dict[str, int]) -> dict:
    """Parse one IN-Ext judgment with its A1/A2 summaries into the canonical schema."""
    doc_sents = read_lines(root / "judgement" / f"{doc_id}.txt")
    n = len(doc_sents)
    in_summary = {a: [False] * n for a in ANNOTATORS}
    segments: dict[str, list[set[str]]] = {a: [set() for _ in range(n)] for a in ANNOTATORS}
    seg_votes: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    full_votes: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    summaries: dict[str, Any] = {}
    alignment: dict[str, Any] = {}

    align_kw = {
        "threshold": cfg.fuzzy_threshold,
        "warn_below": None,
        "partial_threshold": cfg.partial_threshold,
        "partial_min_tokens": cfg.partial_min_tokens,
        "partial_min_len_ratio": cfg.partial_min_len_ratio,
    }
    for a in ANNOTATORS:
        full_path = root / "summary" / "full" / a / f"{doc_id}.txt"
        full = parse_full_summary(full_path) if full_path.is_file() else []
        segs = parse_segment_summaries(
            root / "summary" / "segment-wise" / a, doc_id, cfg.spacy_model
        )
        summaries[a] = {"full": full, "segments": segs}

        res_full = align_sentences([s["text"] or "" for s in full], doc_sents, **align_kw)
        methods: Counter[str] = Counter(res_full.methods)
        for s, m in zip(full, res_full.matches, strict=True):
            if m is None:
                continue
            in_summary[a][m] = True
            h = s["heading"]
            if h is not None:
                full_votes[m].append((_HEADING_TO_CODE[h], INEXT_HEADING_TO_COARSE[h]))

        seg_total = seg_matched = 0
        for folder, sents in segs.items():
            code = INEXT_SEGMENT_TO_CODE[folder]
            res = align_sentences(sents, doc_sents, **align_kw)
            seg_total += res.n
            seg_matched += res.n_matched
            methods.update(res.methods)
            for m in res.matches:
                if m is None:
                    continue
                in_summary[a][m] = True
                segments[a][m].add(folder)
                seg_votes[m].append((code, INEXT_TO_COARSE[code]))

        alignment[a] = {
            "full_n": res_full.n,
            "full_matched": res_full.n_matched,
            "seg_n": seg_total,
            "seg_matched": seg_matched,
            "methods": {m: methods.get(m, 0) for m in ("exact", "fuzzy", "partial", "none")},
        }

    sentences = []
    for i, text in enumerate(doc_sents):
        code, coarse = _pick_role(seg_votes[i] or full_votes[i])
        sentences.append(
            {
                "idx": i,
                "text": text,
                "role_fine": None,
                "role_coarse": coarse,
                "role_inext": code,
                "role_source": ROLE_SOURCE if coarse is not None else None,
                "in_summary": {a: in_summary[a][i] for a in ANNOTATORS},
                "segments": {a: sorted(segments[a][i]) for a in ANNOTATORS},
            }
        )

    year = doc_id.split("_", 1)[0]
    return {
        "doc_id": doc_id,
        "source": "inext",
        "split": "inext",
        "meta": {
            "year": int(year) if year.isdigit() else None,
            "inext_length": lengths.get(doc_id),
            "role_source": ROLE_SOURCE,
        },
        "sentences": sentences,
        "summaries": summaries,
        "alignment": alignment,
    }


def parse_inext(root: Path, cfg: DataConfig) -> list[dict]:
    """Parse every IN-Ext judgment under ``root`` (the ``IN-Ext`` folder)."""
    judg_dir = root / "judgement"
    if not judg_dir.is_dir():
        raise FileNotFoundError(f"IN-Ext judgement folder not found: {judg_dir}")
    lengths = _read_lengths(root)
    doc_ids = sorted(p.stem for p in judg_dir.glob("*.txt"))
    logger.info("Parsing %d IN-Ext judgments from %s", len(doc_ids), root)
    return [parse_document(root, d, cfg, lengths) for d in doc_ids]


def alignment_rate(docs: list[dict]) -> dict[str, float]:
    """Aggregate alignment rates over all IN-Ext docs (full, segment-wise, overall)."""
    tot = {"full_n": 0, "full_matched": 0, "seg_n": 0, "seg_matched": 0}
    methods: Counter[str] = Counter()
    for d in docs:
        for a in ANNOTATORS:
            for k in tot:
                tot[k] += d["alignment"][a][k]
            methods.update(d["alignment"][a]["methods"])
    n = tot["full_n"] + tot["seg_n"]
    return {
        "full": tot["full_matched"] / tot["full_n"] if tot["full_n"] else 0.0,
        "segment_wise": tot["seg_matched"] / tot["seg_n"] if tot["seg_n"] else 0.0,
        "overall": (tot["full_matched"] + tot["seg_matched"]) / n if n else 0.0,
        "strict_exact_fuzzy": (methods["exact"] + methods["fuzzy"]) / n if n else 0.0,
        **{k: float(v) for k, v in tot.items()},
        **{f"n_{m}": float(methods[m]) for m in ("exact", "fuzzy", "partial", "none")},
    }
