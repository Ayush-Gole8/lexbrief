"""Role-agnostic extractive baselines under the same word budget L.

All baselines rank sentences and then fill the budget greedily (a sentence that does not fit
is skipped, so shorter later ones may still fit); output indices are in document order.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import numpy as np

from lexbrief.brief.features import cosine_matrix, textrank

logger = logging.getLogger(__name__)


def fill_budget(order: Sequence[int], words: Sequence[int], length: int) -> list[int]:
    """Take sentences in ``order`` while they fit in ``length`` words; return sorted indices."""
    used, out = 0, []
    for i in order:
        if words[i] > 0 and used + words[i] <= length:
            out.append(i)
            used += words[i]
    return sorted(out)


def lead(words: Sequence[int], length: int) -> list[int]:
    """Lead-L: first sentences of the document."""
    return fill_budget(range(len(words)), words, length)


def textrank_baseline(
    emb: np.ndarray,
    words: Sequence[int],
    length: int,
    edge_threshold: float = 0.1,
    damping: float = 0.85,
) -> list[int]:
    """TextRank over the whole document (same embeddings as the role-aware system)."""
    sim = cosine_matrix(emb)
    pr = textrank(sim, range(len(words)), edge_threshold, damping)
    order = sorted(range(len(words)), key=lambda i: (-pr[i], i))
    return fill_budget(order, words, length)


class _WordTokenizer:
    """Minimal sumy-compatible tokenizer (avoids an NLTK punkt download)."""

    _word = re.compile(r"[a-z0-9]+")

    def to_words(self, text: str) -> list[str]:
        return self._word.findall(text.lower())

    def to_sentences(self, text: str) -> list[str]:
        return [text]


def lexrank_scores(texts: Sequence[str]) -> np.ndarray:
    """LexRank centrality from sumy (tf-idf cosine graph + power method)."""
    from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence
    from sumy.nlp.stemmers import Stemmer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.utils import get_stop_words

    tok = _WordTokenizer()
    sents = [Sentence(t, tok) for t in texts]
    doc = ObjectDocumentModel([Paragraph(sents)])
    lr = LexRankSummarizer(Stemmer("english"))
    lr.stop_words = get_stop_words("english")
    words = [lr._to_words_set(s) for s in doc.sentences]
    if not words:
        return np.zeros(0)
    tf = lr._compute_tf(words)
    idf = lr._compute_idf(words)
    matrix = lr._create_matrix(words, lr.threshold, tf, idf)
    return np.asarray(lr.power_method(matrix, lr.epsilon), dtype=np.float64)


def lexrank_baseline(texts: Sequence[str], words: Sequence[int], length: int) -> list[int]:
    s = lexrank_scores(texts)
    order = sorted(range(len(texts)), key=lambda i: (-s[i], i))
    return fill_budget(order, words, length)


def mmr_baseline(emb: np.ndarray, words: Sequence[int], length: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: relevance = cosine to the document centroid."""
    sim = cosine_matrix(emb)
    x = emb.astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    centroid = x.mean(0)
    rel = x @ (centroid / max(np.linalg.norm(centroid), 1e-8))
    remaining = set(range(len(words)))
    chosen: list[int] = []
    used = 0
    while remaining:
        best, best_val = None, -np.inf
        for i in remaining:
            red = float(np.max(sim[i, chosen])) if chosen else 0.0
            val = lam * rel[i] - (1 - lam) * red
            if val > best_val:
                best, best_val = i, val
        remaining.discard(best)
        if words[best] > 0 and used + words[best] <= length:
            chosen.append(best)
            used += words[best]
    return sorted(chosen)
