"""Sentence features for the role-budgeted scorer.

* ``cent`` - TextRank centrality *within the sentence's role* (``networkx.pagerank`` on the
  cosine-similarity graph of sentence embeddings; edges below ``edge_threshold`` dropped).
* ``pos``  - role-specific position prior: smoothed 10-bin histogram of the relative positions
  of in-summary sentences of that role in training documents.
* ``cue``  - role-specific regex cue count (statute refs, citations, disposal verbs).
* ``conf`` - role confidence (max marginal of the tagger; 1.0 for gold roles).

Every feature is min-max normalised within the document (constant features become 0).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from lexbrief.labels import BRIEF_SECTION_ORDER

logger = logging.getLogger(__name__)

FEATURES: tuple[str, ...] = ("cent", "pos", "cue", "conf")

# IN-Ext judgments are lowercased with most punctuation removed, so patterns tolerate both.
CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "STATUTE": re.compile(
        r"\b(?:act|section|sec|article|art|rule|clause|schedule|ordinance|code)s?\b"
        r"|\bss?\.?\s*\d+"  # "s 302", "ss. 3"; a bare "s" (e.g. "testator s") is not a cue
        r"|\bi\.?p\.?c\b|\bcr\.?p\.?c\b|\bc\.?p\.?c\b",
        re.IGNORECASE,
    ),
    "PRECEDENT": re.compile(
        r"\bvs?\b\.?|\ba\.?i\.?r\b|\bs\.?c\.?c\b|\bs\.?c\.?r\b|\bi\.?l\.?r\b|"
        r"\b(?:19|20)\d{2}\s+(?:\d+\s+)?(?:scr|scc|air)\b",
        re.IGNORECASE,
    ),
    "RULING": re.compile(
        r"\b(?:dismissed|allowed|set\s+aside|acquitted|convicted|disposed\s+of|remanded|"
        r"quashed|upheld|affirmed)\b",
        re.IGNORECASE,
    ),
}


def cue_count(text: str, role: str | None) -> float:
    """Number of role-specific cue matches in ``text`` (0 for roles without cues)."""
    pat = CUE_PATTERNS.get(role or "")
    return float(len(pat.findall(text))) if pat else 0.0


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of row vectors."""
    x = emb.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-8)
    return x @ x.T


def textrank(
    sim: np.ndarray, nodes: Sequence[int], edge_threshold: float = 0.1, damping: float = 0.85
) -> dict[int, float]:
    """PageRank over ``nodes`` with edge weight = cosine (edges < threshold dropped)."""
    nodes = list(nodes)
    if len(nodes) == 1:
        return {nodes[0]: 1.0}
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for a_i, a in enumerate(nodes):
        for b in nodes[a_i + 1 :]:
            w = float(sim[a, b])
            if w >= edge_threshold:
                g.add_edge(a, b, weight=w)
    if g.number_of_edges() == 0:
        return {n: 1.0 / len(nodes) for n in nodes}
    try:
        return nx.pagerank(g, alpha=damping, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge; using weighted degree")
        deg = dict(g.degree(weight="weight"))
        total = sum(deg.values()) or 1.0
        return {n: deg[n] / total for n in nodes}


class PositionPrior:
    """Role-specific histogram of relative positions of in-summary sentences."""

    def __init__(self, bins: int = 10, laplace: float = 1.0) -> None:
        self.bins = bins
        self.laplace = laplace
        self.hist_: dict[str, np.ndarray] = {}

    def _bin(self, rel: float) -> int:
        return min(self.bins - 1, max(0, int(rel * self.bins)))

    def fit(self, samples: Iterable[tuple[str | None, float]]) -> PositionPrior:
        """Fit from ``(role, relative_position)`` of in-summary sentences."""
        counts = {r: np.zeros(self.bins) for r in BRIEF_SECTION_ORDER}
        overall = np.zeros(self.bins)
        for role, rel in samples:
            b = self._bin(rel)
            overall[b] += 1
            if role in counts:
                counts[role][b] += 1
        self.hist_ = {
            r: (c + self.laplace) / (c.sum() + self.laplace * self.bins) for r, c in counts.items()
        }
        self.hist_["_all"] = (overall + self.laplace) / (overall.sum() + self.laplace * self.bins)
        return self

    def score(self, role: str | None, rel: float) -> float:
        if not self.hist_:
            raise RuntimeError("PositionPrior must be fit() first")
        return float(self.hist_.get(role or "_all", self.hist_["_all"])[self._bin(rel)])


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; a constant vector becomes all zeros."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


@dataclass
class DocFeatures:
    """Normalised feature matrix ``[n, 4]`` (cent, pos, cue, conf) plus the cosine matrix."""

    matrix: np.ndarray
    sim: np.ndarray


def compute_features(
    texts: Sequence[str],
    roles: Sequence[str | None],
    conf: Sequence[float],
    emb: np.ndarray,
    prior: PositionPrior,
    edge_threshold: float = 0.1,
    damping: float = 0.85,
) -> DocFeatures:
    """Compute the four normalised features for one document."""
    n = len(texts)
    sim = cosine_matrix(emb)
    cent = np.zeros(n)
    for role in BRIEF_SECTION_ORDER:
        idx = [i for i, r in enumerate(roles) if r == role]
        if idx:
            for i, v in textrank(sim, idx, edge_threshold, damping).items():
                cent[i] = v
    rel = [i / max(1, n - 1) for i in range(n)]
    pos = np.array(
        [
            prior.score(r, p) if r in BRIEF_SECTION_ORDER else 0.0
            for r, p in zip(roles, rel, strict=True)
        ]
    )
    cue = np.array([cue_count(t, r) for t, r in zip(texts, roles, strict=True)])
    mat = np.stack([minmax(cent), minmax(pos), minmax(cue), minmax(np.asarray(conf, float))], 1)
    return DocFeatures(matrix=mat, sim=sim)
