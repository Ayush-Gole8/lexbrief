"""Linear sentence scorer ``alpha*cent + beta*pos + gamma*cue + delta*conf`` and grid search.

The weights are chosen by exhaustive search over ``grid^4`` (all-zero skipped), maximising the
mean ROUGE-2 F1 (fast bigram implementation, averaged over references) of the role-budgeted
briefs on the training documents. Budgets and features do not depend on the weights, so they
are prepared once per document.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lexbrief.brief.rouge2 import rouge2_f, summary_bigrams
from lexbrief.brief.selector import select, selected_indices

logger = logging.getLogger(__name__)


def score(features: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """``features [n, 4] @ weights [4]``."""
    return features @ np.asarray(weights, dtype=np.float64)


@dataclass
class PreparedDoc:
    """Everything the grid search needs for one document (weights excluded)."""

    texts: list[str]
    words: list[int]
    roles: list[str | None]
    features: np.ndarray
    sim: np.ndarray
    budgets: dict[str, int]
    sent_bigrams: list[Counter]
    ref_bigrams: list[Counter]


def doc_rouge2(doc: PreparedDoc, weights: Sequence[float], redundancy: float) -> float:
    """Mean ROUGE-2 over references of the brief produced with ``weights``."""
    brief = select(
        doc.texts,
        doc.words,
        doc.roles,
        score(doc.features, weights),
        doc.budgets,
        doc.sim,
        redundancy,
    )
    cand = summary_bigrams(doc.sent_bigrams, selected_indices(brief))
    return float(np.mean([rouge2_f(cand, r) for r in doc.ref_bigrams]))


def weight_grid(values: Sequence[float]) -> list[tuple[float, ...]]:
    """All 4-tuples over ``values`` except all-zero."""
    return [w for w in itertools.product(values, repeat=4) if any(v > 0 for v in w)]


def grid_search(
    docs: Sequence[PreparedDoc],
    values: Sequence[float],
    redundancy: float = 0.8,
    progress: Callable[[int, int], Any] | None = None,
) -> tuple[tuple[float, ...], list[dict[str, Any]]]:
    """Return the best weights and the full table ``[{weights, rouge2}]`` (sorted, best first).

    Ties are broken towards the earlier grid point, i.e. smaller weights first.
    """
    grid = weight_grid(values)
    table = []
    for k, w in enumerate(grid):
        r2 = float(np.mean([doc_rouge2(d, w, redundancy) for d in docs])) if docs else 0.0
        table.append({"weights": list(w), "rouge2": r2})
        if progress:
            progress(k + 1, len(grid))
    table.sort(key=lambda row: -row["rouge2"])
    best = tuple(table[0]["weights"])
    logger.info(
        "Best weights %s (train ROUGE-2 %.4f over %d docs)", best, table[0]["rouge2"], len(docs)
    )
    return best, table
