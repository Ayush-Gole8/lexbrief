"""Tests for lexbrief.brief.budget."""

from __future__ import annotations

import pytest

from lexbrief.brief.budget import (
    ExpertBudget,
    LearnedBudget,
    ProportionalBudget,
    allocate,
    make_strategy,
    role_stats,
)

EXPERT = {
    "RULING": 0.25,
    "ISSUES": 0.20,
    "FACTS": 0.18,
    "STATUTE": 0.10,
    "PRECEDENT": 0.10,
    "REASONING": 0.10,
    "ARGUMENTS": 0.07,
}


def _stats() -> dict:
    roles = ["FACTS"] * 5 + ["REASONING"] * 10 + ["RULING"] * 4 + ["ISSUES"] + ["DROP", None]
    words = [20] * 5 + [30] * 10 + [10, 12, 15, 40] + [18] + [50, 50]
    return role_stats(roles, words)


@pytest.mark.parametrize("strategy", ["A1", "A2", "A3"])
@pytest.mark.parametrize("length", [0, 5, 37, 100, 250, 10_000])
def test_budgets_sum_le_length_and_respect_caps(strategy: str, length: int) -> None:
    stats = _stats()
    s = make_strategy(strategy, EXPERT)
    if strategy == "A3":
        s.fit([{"FACTS": 30, "REASONING": 50, "RULING": 20}])
    b = s.transform(stats, length)
    assert sum(b.values()) <= length
    assert all(b[r] <= stats[r].words for r in b)
    assert all(v >= 0 for v in b.values())
    assert set(b) == {"FACTS", "REASONING", "RULING", "ISSUES"}  # DROP/None ignored
    if length >= sum(st.words for st in stats.values()):
        assert all(b[r] == stats[r].words for r in b)  # everything fits


def test_guarantees_every_present_role_and_three_ruling_sentences() -> None:
    stats = _stats()
    # A3 shares put nothing on ISSUES/RULING; guarantees must still give them room
    s = LearnedBudget().fit([{"FACTS": 50, "REASONING": 50}])
    b = s.transform(stats, 150)
    assert b["ISSUES"] >= 18  # its only sentence
    assert b["RULING"] >= 10 + 12 + 15  # three shortest ruling sentences
    assert b["FACTS"] >= 20 and b["REASONING"] >= 30
    assert sum(b.values()) <= 150


def test_tiny_length_prioritises_ruling() -> None:
    stats = _stats()
    b = ProportionalBudget().transform(stats, 11)
    assert sum(b.values()) <= 11
    assert b["RULING"] >= 10  # one shortest ruling sentence fits, 3 do not
    b2 = ProportionalBudget().transform(stats, 22)
    assert b2["RULING"] >= 22  # two shortest ruling sentences (10 + 12)


def test_missing_roles_and_single_role_doc() -> None:
    single = role_stats(["FACTS"] * 3, [10, 20, 30])
    b = ExpertBudget(EXPERT).transform(single, 25)
    assert b == {"FACTS": 25}
    assert allocate(EXPERT, {}, 100) == {}
    only_drop = role_stats(["DROP", None], [10, 10])
    assert ProportionalBudget().transform(only_drop, 50) == {}


def test_leftover_is_redistributed_when_capped() -> None:
    stats = role_stats(["RULING", "FACTS", "FACTS"], [5, 100, 100])
    b = ExpertBudget(EXPERT).transform(stats, 150)
    assert b["RULING"] == 5  # capped at its available words
    assert b["FACTS"] == 145  # the rest goes to FACTS
    assert sum(b.values()) == 150


def test_proportional_shares() -> None:
    stats = role_stats(["FACTS", "REASONING"], [100, 300])
    b = ProportionalBudget(ruling_min=0, min_per_role=0).transform(stats, 40)
    assert b == {"FACTS": 10, "REASONING": 30}


def test_learned_fit_averages_per_document_shares() -> None:
    s = LearnedBudget().fit([{"FACTS": 10, "RULING": 10}, {"FACTS": 30, "RULING": 10}])
    assert s.weights_["FACTS"] == pytest.approx((0.5 + 0.75) / 2)
    assert s.weights_["RULING"] == pytest.approx((0.5 + 0.25) / 2)
    with pytest.raises(RuntimeError):
        LearnedBudget().transform(_stats(), 10)
    with pytest.raises(ValueError):
        LearnedBudget().fit([{}])


def test_make_strategy_errors() -> None:
    with pytest.raises(ValueError):
        make_strategy("A9")
    with pytest.raises(ValueError):
        make_strategy("A2")
    with pytest.raises(ValueError):
        ExpertBudget({"BOGUS": 1.0})
