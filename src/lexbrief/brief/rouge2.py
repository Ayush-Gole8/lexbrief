"""Fast ROUGE-2 for search loops (weight grid search, greedy oracle).

Uses the exact ``rouge_score`` tokenizer + Porter stemmer, but counts bigrams per sentence so
candidate summaries can be scored incrementally. Bigrams spanning two selected sentences are
not counted, so values can differ marginally from ``rouge_score`` on the joined text; reported
results (Prompt 6) always use ``rouge_score`` itself.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache


@lru_cache(maxsize=2)
def _tokenizer(stemmer: bool = True):  # noqa: ANN202 - rouge_score tokenizer
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge2"], use_stemmer=stemmer)._tokenizer


def tokens(text: str, stemmer: bool = True) -> list[str]:
    """rouge_score tokenisation (lowercase, alnum, optional Porter stemming)."""
    return _tokenizer(stemmer).tokenize(text)


def bigrams(toks: list[str]) -> Counter[tuple[str, str]]:
    return Counter(zip(toks, toks[1:], strict=False))


def sentence_bigrams(texts: Iterable[str], stemmer: bool = True) -> list[Counter]:
    """Bigram counters for each sentence."""
    return [bigrams(tokens(t, stemmer)) for t in texts]


def rouge2_f(candidate: Counter, reference: Counter) -> float:
    """ROUGE-2 F1 between two bigram counters."""
    c_total, r_total = sum(candidate.values()), sum(reference.values())
    if c_total == 0 or r_total == 0:
        return 0.0
    overlap = sum((candidate & reference).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / c_total, overlap / r_total
    return 2 * p * r / (p + r)


def summary_bigrams(sent_bigrams: list[Counter], selected: Iterable[int]) -> Counter:
    """Union (sum) of the bigram counters of the selected sentences."""
    out: Counter = Counter()
    for i in selected:
        out.update(sent_bigrams[i])
    return out
