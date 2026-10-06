"""Tests for lexbrief.labels."""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from lexbrief import labels as L

EXPECTED_FINE_TO_COARSE = {
    "PREAMBLE": "DROP",
    "FAC": "FACTS",
    "RLC": "FACTS",
    "ISSUE": "ISSUES",
    "ARG_PETITIONER": "ARGUMENTS",
    "ARG_RESPONDENT": "ARGUMENTS",
    "ANALYSIS": "REASONING",
    "STA": "STATUTE",
    "PRE_RELIED": "PRECEDENT",
    "PRE_NOT_RELIED": "PRECEDENT",
    "RATIO": "REASONING",
    "RPC": "RULING",
    "NONE": "DROP",
}

EXPECTED_INEXT_TO_COARSE = {
    "FAC": "FACTS",
    "ARG": "ARGUMENTS",
    "STA": "STATUTE",
    "PRE": "PRECEDENT",
    "Ratio": "REASONING",
    "RPC": "RULING",
}


def test_fine_labels_fixed_order() -> None:
    assert L.FINE_LABELS == (
        "PREAMBLE",
        "FAC",
        "RLC",
        "ISSUE",
        "ARG_PETITIONER",
        "ARG_RESPONDENT",
        "ANALYSIS",
        "STA",
        "PRE_RELIED",
        "PRE_NOT_RELIED",
        "RATIO",
        "RPC",
        "NONE",
    )
    assert len(set(L.FINE_LABELS)) == 13


@pytest.mark.parametrize(("fine", "coarse"), sorted(EXPECTED_FINE_TO_COARSE.items()))
def test_fine_to_coarse_each(fine: str, coarse: str) -> None:
    assert L.FINE_TO_COARSE[fine] == coarse
    assert L.fine_to_coarse(fine) == coarse


def test_fine_to_coarse_complete() -> None:
    assert set(L.FINE_TO_COARSE) == set(L.FINE_LABELS)
    assert L.FINE_TO_COARSE == EXPECTED_FINE_TO_COARSE
    assert set(L.FINE_TO_COARSE.values()) == set(L.COARSE_LABELS)


@pytest.mark.parametrize(("seg", "coarse"), sorted(EXPECTED_INEXT_TO_COARSE.items()))
def test_inext_to_coarse_each(seg: str, coarse: str) -> None:
    assert L.inext_to_coarse(seg) == coarse


def test_inext_to_coarse_complete() -> None:
    assert L.INEXT_TO_COARSE == EXPECTED_INEXT_TO_COARSE
    assert set(L.INEXT_TO_COARSE.values()) <= set(L.BRIEF_SECTION_ORDER)


def test_inext_segment_and_heading_maps() -> None:
    assert L.INEXT_SEGMENT_TO_CODE == {
        "facts": "FAC",
        "argument": "ARG",
        "statute": "STA",
        "analysis": "Ratio",
        "judgement": "RPC",
    }
    for code in L.INEXT_SEGMENT_TO_CODE.values():
        assert code in L.INEXT_TO_COARSE
    assert L.INEXT_HEADING_TO_COARSE["ISSUE"] == "ISSUES"
    assert set(L.INEXT_HEADING_TO_COARSE.values()) <= set(L.BRIEF_SECTION_ORDER)


def test_id_roundtrip() -> None:
    for i, label in enumerate(L.FINE_LABELS):
        assert L.fine_id(label) == i
        assert L.fine_label(i) == label


def test_brief_section_order() -> None:
    assert L.DROP not in L.BRIEF_SECTION_ORDER
    assert set(L.BRIEF_SECTION_ORDER) == set(L.COARSE_LABELS) - {L.DROP}
    assert len(L.BRIEF_SECTION_ORDER) == 7
    assert L.BRIEF_SECTION_ORDER[0] == "FACTS"
    assert L.BRIEF_SECTION_ORDER[-1] == "RULING"


def test_colours() -> None:
    assert set(L.COARSE_COLORS) == set(L.COARSE_LABELS)
    for coarse in L.COARSE_LABELS:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", L.coarse_color(coarse))
    assert len(set(L.COARSE_COLORS.values())) == len(L.COARSE_COLORS)


def test_is_dropped_and_collapse() -> None:
    assert L.is_dropped("PREAMBLE")
    assert L.is_dropped("NONE")
    assert not L.is_dropped("RATIO")
    assert L.collapse_sequence(["FAC", "RLC", "RPC", "NONE"]) == [
        "FACTS",
        "FACTS",
        "RULING",
        "DROP",
    ]


@pytest.mark.parametrize(
    "func",
    [L.fine_to_coarse, L.inext_to_coarse, L.fine_id, L.coarse_color],
)
def test_unknown_label_raises(func: Callable[[str], object]) -> None:
    with pytest.raises(KeyError):
        func("NOT_A_LABEL")


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        L.fine_label(13)
