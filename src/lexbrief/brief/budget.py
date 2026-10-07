"""Per-role word budgets B_r for a brief of total length L.

Strategies (``shares`` = desired share of L per coarse role):

* **A1 proportional** - shares proportional to each role's word count in this document.
* **A2 expert-priority** - fixed shares (``brief.expert_shares``, DELSumm ordering).
* **A3 learned** - ``w_r`` = mean over training documents of (reference summary words in role
  r / reference summary words); :meth:`LearnedBudget.fit` then :meth:`transform`.

Allocation (:func:`allocate`) is shared: every role present in the document is first guaranteed
room for its shortest sentence (RULING: its ``ruling_min_sentences`` shortest sentences), as far
as L allows and in priority order; the rest of L is split by the shares with water-filling so
no role gets more words than it has; integer rounding never exceeds L.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from lexbrief.labels import BRIEF_SECTION_ORDER

logger = logging.getLogger(__name__)

RULING = "RULING"
# Priority used when L is too small for every guarantee (DELSumm ordering).
GUARANTEE_PRIORITY: tuple[str, ...] = (
    "RULING",
    "ISSUES",
    "FACTS",
    "STATUTE",
    "PRECEDENT",
    "REASONING",
    "ARGUMENTS",
)


@dataclass
class RoleStats:
    """Word counts of the sentences of one role in a document."""

    sentence_words: list[int] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(self.sentence_words)


def role_stats(roles: Iterable[str | None], words: Iterable[int]) -> dict[str, RoleStats]:
    """Group sentence word counts by brief role (DROP / None ignored)."""
    out: dict[str, RoleStats] = {}
    for r, w in zip(roles, words, strict=True):
        if r in BRIEF_SECTION_ORDER and w > 0:
            out.setdefault(r, RoleStats()).sentence_words.append(int(w))
    return out


def _minimums(
    stats: Mapping[str, RoleStats], length: int, ruling_min: int, min_per_role: int
) -> dict[str, int]:
    """Guaranteed words per role, granted in priority order while they fit in L."""
    mins: dict[str, int] = {}
    used = 0
    for role in GUARANTEE_PRIORITY:
        if role not in stats:
            continue
        shortest = sorted(stats[role].sentence_words)
        want = ruling_min if role == RULING else min_per_role
        # try the full guarantee, then fewer sentences (RULING: 3 -> 2 -> 1)
        for k in range(min(want, len(shortest)), 0, -1):
            need = sum(shortest[:k])
            if used + need <= length:
                mins[role] = need
                used += need
                break
    return mins


def allocate(
    shares: Mapping[str, float],
    stats: Mapping[str, RoleStats],
    length: int,
    ruling_min: int = 3,
    min_per_role: int = 1,
) -> dict[str, int]:
    """Integer word budget per present role with ``sum <= length`` and ``B_r <= words_r``.

    Args:
        shares: Non-negative weight per role (renormalised over present roles).
        stats: Sentence word counts per role present in the document.
        length: Total budget L in words.
        ruling_min: Guarantee room for this many (shortest) RULING sentences.
        min_per_role: Guarantee room for this many (shortest) sentences of every other role.
    """
    length = max(0, int(length))
    present = [r for r in BRIEF_SECTION_ORDER if r in stats and stats[r].words > 0]
    if not present or length == 0:
        return {r: 0 for r in present}
    mins = _minimums(stats, length, ruling_min, min_per_role)
    budget = {r: float(mins.get(r, 0)) for r in present}
    cap = {r: float(stats[r].words) for r in present}
    remaining = float(length - sum(mins.values()))

    # water-filling: distribute `remaining` by share, cap at available words, re-spread leftovers
    active = [r for r in present if budget[r] < cap[r]]
    while remaining > 1e-9 and active:
        weights = {r: max(0.0, float(shares.get(r, 0.0))) for r in active}
        total_w = sum(weights.values())
        if total_w <= 0:  # no share information left: spread by remaining headroom
            weights = {r: cap[r] - budget[r] for r in active}
            total_w = sum(weights.values())
        spent = 0.0
        for r in active:
            give = min(remaining * weights[r] / total_w, cap[r] - budget[r])
            budget[r] += give
            spent += give
        remaining -= spent
        active = [r for r in active if budget[r] < cap[r] - 1e-9]
        if spent <= 1e-9:
            break

    # integer rounding: floor, then hand out leftover words by largest remainder within caps
    out = {r: int(math.floor(budget[r] + 1e-9)) for r in present}
    leftover = length - sum(out.values())
    order = sorted(present, key=lambda r: budget[r] - out[r], reverse=True)
    for r in order:
        if leftover <= 0:
            break
        if out[r] < cap[r] and budget[r] - out[r] > 1e-9:
            out[r] += 1
            leftover -= 1
    return out


class BudgetStrategy:
    """Base class: ``fit`` (optional) then ``transform`` into per-role word budgets."""

    name = "base"

    def __init__(self, ruling_min: int = 3, min_per_role: int = 1) -> None:
        self.ruling_min = ruling_min
        self.min_per_role = min_per_role

    def fit(self, docs: Iterable[Mapping[str, float]] = ()) -> BudgetStrategy:  # noqa: B027
        """No-op for non-learned strategies."""
        return self

    def shares(self, stats: Mapping[str, RoleStats]) -> dict[str, float]:
        raise NotImplementedError

    def transform(self, stats: Mapping[str, RoleStats], length: int) -> dict[str, int]:
        """Per-role budgets for one document."""
        return allocate(self.shares(stats), stats, length, self.ruling_min, self.min_per_role)


class ProportionalBudget(BudgetStrategy):
    """A1: share proportional to the role's word count in this document."""

    name = "A1"

    def shares(self, stats: Mapping[str, RoleStats]) -> dict[str, float]:
        return {r: float(s.words) for r, s in stats.items()}


class ExpertBudget(BudgetStrategy):
    """A2: fixed expert-priority shares."""

    name = "A2"

    def __init__(self, expert_shares: Mapping[str, float], **kw: int) -> None:
        super().__init__(**kw)
        unknown = set(expert_shares) - set(BRIEF_SECTION_ORDER)
        if unknown:
            raise ValueError(f"Unknown roles in expert_shares: {sorted(unknown)}")
        self.expert_shares = dict(expert_shares)

    def shares(self, stats: Mapping[str, RoleStats]) -> dict[str, float]:
        return {r: self.expert_shares.get(r, 0.0) for r in stats}


class LearnedBudget(BudgetStrategy):
    """A3: shares learned from reference summaries of training documents."""

    name = "A3"

    def __init__(self, **kw: int) -> None:
        super().__init__(**kw)
        self.weights_: dict[str, float] | None = None

    def fit(self, docs: Iterable[Mapping[str, float]] = ()) -> LearnedBudget:
        """Fit from per-document reference word counts per role.

        Args:
            docs: One mapping per (document, reference) with summary words per role.
        """
        per_doc = []
        for words in docs:
            total = sum(v for k, v in words.items() if k in BRIEF_SECTION_ORDER)
            if total > 0:
                per_doc.append({r: words.get(r, 0) / total for r in BRIEF_SECTION_ORDER})
        if not per_doc:
            raise ValueError("LearnedBudget.fit needs at least one non-empty reference")
        self.weights_ = {r: sum(d[r] for d in per_doc) / len(per_doc) for r in BRIEF_SECTION_ORDER}
        logger.debug("A3 learned shares: %s", self.weights_)
        return self

    def shares(self, stats: Mapping[str, RoleStats]) -> dict[str, float]:
        if self.weights_ is None:
            raise RuntimeError("LearnedBudget must be fit() before transform()")
        return {r: self.weights_.get(r, 0.0) for r in stats}


def make_strategy(
    name: str,
    expert_shares: Mapping[str, float] | None = None,
    ruling_min: int = 3,
    min_per_role: int = 1,
) -> BudgetStrategy:
    """Build ``A1`` / ``A2`` / ``A3``."""
    kw = {"ruling_min": ruling_min, "min_per_role": min_per_role}
    if name == "A1":
        return ProportionalBudget(**kw)
    if name == "A2":
        if expert_shares is None:
            raise ValueError("A2 needs expert_shares")
        return ExpertBudget(expert_shares, **kw)
    if name == "A3":
        return LearnedBudget(**kw)
    raise ValueError(f"Unknown budget strategy {name!r} (A1, A2, A3)")
