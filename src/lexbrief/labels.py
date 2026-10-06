"""Rhetorical-role label maps for LexBrief.

Fine labels follow the BUILD / InRhetoricalRoles 13-role scheme. They are collapsed into
7 coarse brief sections (plus ``DROP``). IN-Ext expert-summary segment names are mapped
onto the same coarse sections so predicted roles and gold summaries can be compared.
"""

from __future__ import annotations

from collections.abc import Iterable

FINE_LABELS: tuple[str, ...] = (
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
"""The 13 fine rhetorical roles in fixed order (index == class id)."""

FINE2ID: dict[str, int] = {label: i for i, label in enumerate(FINE_LABELS)}
ID2FINE: dict[int, str] = dict(enumerate(FINE_LABELS))

DROP = "DROP"

COARSE_LABELS: tuple[str, ...] = (
    "FACTS",
    "ISSUES",
    "ARGUMENTS",
    "STATUTE",
    "PRECEDENT",
    "REASONING",
    "RULING",
    DROP,
)
"""The 7 coarse brief sections plus ``DROP`` (sentences excluded from briefs)."""

FINE_TO_COARSE: dict[str, str] = {
    "PREAMBLE": DROP,
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
    "NONE": DROP,
}

INEXT_TO_COARSE: dict[str, str] = {
    "FAC": "FACTS",
    "ARG": "ARGUMENTS",
    "STA": "STATUTE",
    "PRE": "PRECEDENT",
    "Ratio": "REASONING",
    "RPC": "RULING",
}
"""IN-Ext expert summary segment names -> coarse sections."""

INEXT_SEGMENT_TO_CODE: dict[str, str] = {
    "facts": "FAC",
    "argument": "ARG",
    "statute": "STA",
    "analysis": "Ratio",
    "judgement": "RPC",
}
"""Zenodo IN-Ext ``summary/segment-wise/<A>/<folder>`` names -> IN-Ext codes.

The released data has no separate precedent folder: precedent discussion is part of
``analysis`` (mapped to Ratio / REASONING), so ``PRE`` never occurs in practice.
"""

INEXT_HEADING_TO_COARSE: dict[str, str] = {
    "FACTS": "FACTS",
    "ARGUMENT": "ARGUMENTS",
    "ISSUE": "ISSUES",
    "STATUTE": "STATUTE",
    "ANALYSIS": "REASONING",
}
"""Section headings found in IN-Ext ``summary/full`` files -> coarse sections."""

BRIEF_SECTION_ORDER: tuple[str, ...] = (
    "FACTS",
    "ISSUES",
    "ARGUMENTS",
    "STATUTE",
    "PRECEDENT",
    "REASONING",
    "RULING",
)
"""Order in which sections appear in a generated brief (``DROP`` excluded)."""

COARSE_COLORS: dict[str, str] = {
    "FACTS": "#0072B2",  # blue
    "ISSUES": "#E69F00",  # orange
    "ARGUMENTS": "#56B4E9",  # sky blue
    "STATUTE": "#009E73",  # bluish green
    "PRECEDENT": "#CC79A7",  # reddish purple
    "REASONING": "#D55E00",  # vermillion
    "RULING": "#F0E442",  # yellow
    DROP: "#999999",  # grey
}
"""Okabe-Ito colour-blind-safe palette, one colour per coarse role."""


def _lookup(mapping: dict[str, str], label: str, kind: str) -> str:
    try:
        return mapping[label]
    except KeyError:
        raise KeyError(
            f"Unknown {kind} label {label!r}; expected one of {sorted(mapping)}"
        ) from None


def fine_to_coarse(label: str) -> str:
    """Map a fine rhetorical role to its coarse brief section."""
    return _lookup(FINE_TO_COARSE, label, "fine")


def inext_to_coarse(label: str) -> str:
    """Map an IN-Ext summary segment name to its coarse brief section."""
    return _lookup(INEXT_TO_COARSE, label, "IN-Ext")


def fine_id(label: str) -> int:
    """Return the class id of a fine label."""
    if label not in FINE2ID:
        raise KeyError(f"Unknown fine label {label!r}; expected one of {list(FINE_LABELS)}")
    return FINE2ID[label]


def fine_label(idx: int) -> str:
    """Return the fine label for a class id."""
    if idx not in ID2FINE:
        raise KeyError(f"Fine label id {idx} out of range [0, {len(FINE_LABELS) - 1}]")
    return ID2FINE[idx]


def coarse_color(coarse: str) -> str:
    """Return the hex colour assigned to a coarse role."""
    return _lookup(COARSE_COLORS, coarse, "coarse")


def is_dropped(label: str) -> bool:
    """Return True if a fine label is excluded from briefs."""
    return fine_to_coarse(label) == DROP


def collapse_sequence(labels: Iterable[str]) -> list[str]:
    """Map a sequence of fine labels to coarse labels."""
    return [fine_to_coarse(label) for label in labels]
