"""Greedy ROUGE-2 oracle: the extractive ceiling against one reference summary."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

from lexbrief.brief.rouge2 import rouge2_f

logger = logging.getLogger(__name__)


def greedy_oracle(
    sent_bigrams: Sequence[Counter],
    words: Sequence[int],
    reference: Counter,
    length: int,
) -> list[int]:
    """Repeatedly add the sentence that most increases ROUGE-2 F1 against ``reference``.

    Stops when no remaining sentence that fits in ``length`` words improves the score.

    Returns:
        Selected sentence indices in document order.
    """
    chosen: list[int] = []
    cand: Counter = Counter()
    best_score, used = 0.0, 0
    remaining = [i for i in range(len(words)) if 0 < words[i] <= length]
    while remaining:
        best_i, best_val = None, best_score
        for i in remaining:
            if used + words[i] > length:
                continue
            val = rouge2_f(cand + sent_bigrams[i], reference)
            if val > best_val + 1e-12:
                best_i, best_val = i, val
        if best_i is None:
            break
        chosen.append(best_i)
        cand += sent_bigrams[best_i]
        used += words[best_i]
        best_score = best_val
        remaining.remove(best_i)
    return sorted(chosen)
