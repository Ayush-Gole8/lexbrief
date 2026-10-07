"""Role-budgeted greedy sentence selection with redundancy filtering."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lexbrief.labels import BRIEF_SECTION_ORDER

logger = logging.getLogger(__name__)

DISPLAY_MERGE: dict[str, str] = {
    "STATUTE": "Statute & Precedent",
    "PRECEDENT": "Statute & Precedent",
}


def select(
    texts: Sequence[str],
    words: Sequence[int],
    roles: Sequence[str | None],
    scores: Sequence[float],
    budgets: Mapping[str, int],
    sim: np.ndarray,
    redundancy_threshold: float = 0.8,
) -> dict[str, list[dict[str, Any]]]:
    """Pick sentences per role, best score first, until each role's word budget is used.

    A sentence that does not fit the remaining budget is skipped (smaller ones may still fit);
    a sentence whose cosine similarity to any already-selected sentence exceeds
    ``redundancy_threshold`` is skipped. Roles are filled in BRIEF_SECTION_ORDER and each
    section is returned in document order.

    Returns:
        ``{section: [{"idx", "text", "score"}]}`` for every section with a non-empty selection,
        in BRIEF_SECTION_ORDER.
    """
    chosen: list[int] = []
    out: dict[str, list[dict[str, Any]]] = {}
    for role in BRIEF_SECTION_ORDER:
        budget = int(budgets.get(role, 0))
        if budget <= 0:
            continue
        cands = sorted((i for i, r in enumerate(roles) if r == role), key=lambda i: (-scores[i], i))
        used, picked = 0, []
        for i in cands:
            if used + words[i] > budget:
                continue
            if chosen and float(np.max(sim[i, chosen])) > redundancy_threshold:
                continue
            picked.append(i)
            chosen.append(i)
            used += words[i]
        if picked:
            out[role] = [
                {"idx": i, "text": texts[i], "score": float(scores[i])} for i in sorted(picked)
            ]
    return out


def selected_indices(brief: Mapping[str, list[dict[str, Any]]]) -> list[int]:
    """All selected sentence indices in document order."""
    return sorted(s["idx"] for sents in brief.values() for s in sents)


def brief_words(brief: Mapping[str, list[dict[str, Any]]], words: Sequence[int]) -> int:
    return sum(words[i] for i in selected_indices(brief))


def display_sections(
    brief: Mapping[str, list[dict[str, Any]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Sections for display: STATUTE and PRECEDENT merged into "Statute & Precedent"."""
    merged: dict[str, list[dict[str, Any]]] = {}
    for role in BRIEF_SECTION_ORDER:
        if role not in brief:
            continue
        title = DISPLAY_MERGE.get(role, role.title())
        merged.setdefault(title, []).extend(brief[role])
    return [(t, sorted(s, key=lambda x: x["idx"])) for t, s in merged.items()]


def format_brief(brief: Mapping[str, list[dict[str, Any]]], title: str = "") -> str:
    """Plain-text rendering with section headings."""
    lines = [title, "=" * len(title)] if title else []
    for section, sents in display_sections(brief):
        lines.append(f"\n## {section}")
        lines.extend(f"- [{s['idx']}] {s['text']}" for s in sents)
    return "\n".join(lines).lstrip("\n")
