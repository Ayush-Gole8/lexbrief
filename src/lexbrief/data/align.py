"""Align summary sentences to judgment sentences (exact match, then fuzzy fallback)."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
        "–": "-",
        "—": "-",
    }
)
_SPACED_CLITIC = re.compile(r"\s+'(s|t|re|ve|ll|d|m)\b")
_NON_WORD = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise for matching: NFKC, ASCII quotes, lowercase, drop punctuation, squash spaces.

    Punctuation is removed because IN-Ext judgments are released without most punctuation
    while the summaries keep commas and apostrophes (``testator 's`` vs ``testator's``).
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    text = _SPACED_CLITIC.sub(r"'\1", text)
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


@dataclass
class AlignmentResult:
    """Alignment of summary sentences onto document sentences."""

    matches: list[int | None]
    """For each summary sentence, the matched document sentence index (or None)."""
    scores: list[float]
    """Match score per summary sentence (100 for exact, fuzz.ratio otherwise, 0 if none)."""
    methods: list[str] = field(default_factory=list)
    """``exact`` / ``fuzzy`` / ``none`` per summary sentence."""

    @property
    def n(self) -> int:
        return len(self.matches)

    @property
    def n_matched(self) -> int:
        return sum(m is not None for m in self.matches)

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n if self.n else 1.0

    @property
    def matched_doc_indices(self) -> set[int]:
        return {m for m in self.matches if m is not None}


def _partial_match(
    q: str, doc_norm: list[str], threshold: float, min_tokens: int, min_len_ratio: float
) -> tuple[int, float] | None:
    """Best ``fuzz.partial_ratio`` match whose length is comparable to the query."""
    if len(q.split()) < min_tokens:
        return None
    cands = process.extract(q, doc_norm, scorer=fuzz.partial_ratio, score_cutoff=threshold, limit=5)
    for _, score, idx in cands:  # sorted by score desc
        if len(doc_norm[idx]) >= min_len_ratio * len(q):
            return int(idx), float(score)
    return None


def align_sentences(
    summary_sents: list[str],
    doc_sents: list[str],
    threshold: float = 90.0,
    warn_below: float | None = 0.95,
    name: str = "",
    partial_threshold: float | None = None,
    partial_min_tokens: int = 6,
    partial_min_len_ratio: float = 0.6,
) -> AlignmentResult:
    """Align each summary sentence to its best document sentence.

    Strategy: normalise both sides, try exact match, else take the document sentence with the
    highest ``rapidfuzz.fuzz.ratio`` if it is ``>= threshold``. Optionally (``partial_threshold``)
    a third tier uses ``fuzz.partial_ratio`` to catch summary sentences that are sub-spans of a
    longer document sentence; it requires ``partial_min_tokens`` query tokens and a candidate at
    least ``partial_min_len_ratio`` times the query length (guards against short false hits).

    Args:
        summary_sents: Summary sentences.
        doc_sents: Document sentences.
        threshold: Minimum fuzz.ratio (0-100) for a fuzzy match.
        warn_below: Log a warning if the match rate is below this (None disables).
        name: Label used in the warning.
        partial_threshold: Minimum fuzz.partial_ratio for the sub-span tier (None disables).
        partial_min_tokens: Minimum query length (tokens) for the sub-span tier.
        partial_min_len_ratio: Minimum candidate/query character-length ratio for that tier.
    """
    doc_norm = [normalize(s) for s in doc_sents]
    exact: dict[str, int] = {}
    for i, s in enumerate(doc_norm):
        exact.setdefault(s, i)

    matches: list[int | None] = []
    scores: list[float] = []
    methods: list[str] = []
    for sent in summary_sents:
        q = normalize(sent)
        if not q:
            matches.append(None)
            scores.append(0.0)
            methods.append("none")
            continue
        if q in exact:
            matches.append(exact[q])
            scores.append(100.0)
            methods.append("exact")
            continue
        best = process.extractOne(q, doc_norm, scorer=fuzz.ratio, score_cutoff=threshold)
        if best is not None:
            _, score, idx = best
            matches.append(int(idx))
            scores.append(float(score))
            methods.append("fuzzy")
            continue
        part = (
            _partial_match(q, doc_norm, partial_threshold, partial_min_tokens, partial_min_len_ratio)
            if partial_threshold is not None
            else None
        )
        if part is not None:
            matches.append(part[0])
            scores.append(part[1])
            methods.append("partial")
        else:
            matches.append(None)
            scores.append(0.0)
            methods.append("none")

    result = AlignmentResult(matches=matches, scores=scores, methods=methods)
    if warn_below is not None and result.n and result.match_rate < warn_below:
        logger.warning(
            "Low alignment rate %.1f%% (%d/%d) %s",
            100 * result.match_rate,
            result.n_matched,
            result.n,
            name,
        )
    return result
