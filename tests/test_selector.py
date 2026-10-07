"""Tests for brief selection, features, baselines, oracle and fast ROUGE-2."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from lexbrief.brief.baselines import fill_budget, lead, lexrank_scores, mmr_baseline
from lexbrief.brief.features import PositionPrior, compute_features, cue_count, minmax
from lexbrief.brief.oracle import greedy_oracle
from lexbrief.brief.rouge2 import rouge2_f, sentence_bigrams, summary_bigrams
from lexbrief.brief.scorer import weight_grid
from lexbrief.brief.selector import display_sections, format_brief, select, selected_indices

TEXTS = [
    "the appellant filed a suit",  # 0 FACTS
    "the suit was decreed",  # 1 FACTS
    "section 14 of the act applies",  # 2 STATUTE
    "we hold the widow took a limited estate",  # 3 REASONING
    "the appeal is dismissed",  # 4 RULING
    "the appeal is dismissed with costs",  # 5 RULING (near-duplicate of 4)
]
ROLES = ["FACTS", "FACTS", "STATUTE", "REASONING", "RULING", "RULING"]
WORDS = [len(t.split()) for t in TEXTS]


def _sim(dup: float = 0.95) -> np.ndarray:
    s = np.eye(len(TEXTS))
    s[4, 5] = s[5, 4] = dup
    return s


def test_select_respects_budgets_order_and_redundancy() -> None:
    scores = [0.9, 0.1, 0.5, 0.7, 0.8, 0.95]
    budgets = {"FACTS": 6, "STATUTE": 6, "REASONING": 3, "RULING": 20}
    out = select(TEXTS, WORDS, ROLES, scores, budgets, _sim(), redundancy_threshold=0.8)
    assert list(out) == ["FACTS", "STATUTE", "RULING"]  # BRIEF_SECTION_ORDER; REASONING too long
    assert [s["idx"] for s in out["FACTS"]] == [0]  # 0 fits (5 words), then 1 does not (4 > 1)
    assert [s["idx"] for s in out["RULING"]] == [5]  # 4 is redundant with 5 (cos 0.95)
    for role, sents in out.items():
        assert sum(WORDS[s["idx"]] for s in sents) <= budgets[role]


def test_select_skips_long_sentence_and_keeps_document_order() -> None:
    scores = [0.1, 0.9, 0, 0, 0, 0]
    out = select(TEXTS, WORDS, ROLES, scores, {"FACTS": 10}, _sim(), 0.8)
    assert [s["idx"] for s in out["FACTS"]] == [0, 1]  # document order, not score order
    assert selected_indices(out) == [0, 1]


def test_display_merges_statute_and_precedent() -> None:
    brief = {
        "STATUTE": [{"idx": 5, "text": "s", "score": 1.0}],
        "PRECEDENT": [{"idx": 2, "text": "p", "score": 1.0}],
        "RULING": [{"idx": 9, "text": "r", "score": 1.0}],
    }
    sections = display_sections(brief)
    assert [t for t, _ in sections] == ["Statute & Precedent", "Ruling"]
    assert [s["idx"] for s in sections[0][1]] == [2, 5]
    assert "## Statute & Precedent" in format_brief(brief, "t")


def test_cues_and_minmax() -> None:
    assert cue_count("under section 14 of the hindu succession act", "STATUTE") >= 2
    assert cue_count("the testator s will", "STATUTE") == 0  # bare 's' is not a section
    assert cue_count("state v ram 1955 scr 123", "PRECEDENT") >= 2
    assert cue_count("the appeal is accordingly dismissed", "RULING") == 1
    assert cue_count("section 14", "FACTS") == 0
    assert minmax(np.array([2.0, 2.0])).tolist() == [0.0, 0.0]
    assert minmax(np.array([1.0, 3.0, 2.0])).tolist() == [0.0, 1.0, 0.5]


def test_position_prior_and_features_shape() -> None:
    prior = PositionPrior(bins=10, laplace=1.0).fit([("RULING", 0.95)] * 5 + [("FACTS", 0.05)])
    assert prior.score("RULING", 0.99) > prior.score("RULING", 0.05)
    assert sum(prior.hist_["RULING"]) == pytest.approx(1.0)
    emb = np.random.default_rng(0).normal(size=(len(TEXTS), 8))
    f = compute_features(TEXTS, ROLES, [0.9] * 6, emb, prior)
    assert f.matrix.shape == (6, 4)
    assert (f.matrix >= 0).all() and (f.matrix <= 1).all()
    assert f.matrix[:, 3].tolist() == [0.0] * 6  # constant confidence -> 0 after min-max


def test_weight_grid_skips_all_zero() -> None:
    grid = weight_grid([0, 0.25, 0.5, 1])
    assert len(grid) == 4**4 - 1
    assert (0, 0, 0, 0) not in grid


def test_baselines_respect_budget() -> None:
    assert lead(WORDS, 10) == [0, 1]
    assert fill_budget([3, 0, 1], WORDS, 9) == [3]  # 3 (8 words) fits; 0 (5) and 1 (4) do not
    assert fill_budget([2, 1, 0], WORDS, 10) == [1, 2]  # 2 (6) fits, 1 (4) fits, 0 skipped
    emb = np.random.default_rng(1).normal(size=(len(TEXTS), 8))
    idx = mmr_baseline(emb, WORDS, 12, lam=0.7)
    assert sum(WORDS[i] for i in idx) <= 12 and idx == sorted(idx)
    s = lexrank_scores(TEXTS)
    assert s.shape == (len(TEXTS),) and np.isfinite(s).all()


def test_rouge2_and_oracle() -> None:
    bg = sentence_bigrams(TEXTS)
    ref = sentence_bigrams(["the appeal is dismissed with costs"])[0]
    assert rouge2_f(bg[5], ref) == pytest.approx(1.0)
    assert rouge2_f(Counter(), ref) == 0.0
    picked = greedy_oracle(bg, WORDS, ref, length=6)
    assert picked == [5]
    assert rouge2_f(summary_bigrams(bg, picked), ref) == pytest.approx(1.0)
    assert greedy_oracle(bg, WORDS, ref, length=2) == []  # nothing fits
