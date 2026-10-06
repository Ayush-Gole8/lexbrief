"""Parse BUILD / InRhetoricalRoles (Label-Studio export JSON) into the canonical schema.

Each file is a list of documents::

    {"id": ..., "data": {"text": ...}, "meta": {"group": ...},
     "annotations": [{"result": [{"value": {"start", "end", "text", "labels": [ROLE]}}, ...]}]}

Sentences are the ``annotations[0].result[*].value`` spans, kept in file order.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lexbrief.labels import FINE_TO_COARSE

logger = logging.getLogger(__name__)


def has_labels(raw_docs: list[dict[str, Any]]) -> bool:
    """True if any document carries at least one labelled span."""
    for doc in raw_docs:
        for ann in doc.get("annotations") or []:
            for res in ann.get("result") or []:
                if (res.get("value") or {}).get("labels"):
                    return True
    return False


def load_raw(path: str | Path) -> list[dict[str, Any]]:
    """Load a BUILD JSON file (a list of documents)."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate {"data": [...]} wrappers
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return data


def _meta_group(doc: dict[str, Any]) -> str | None:
    meta = doc.get("meta") or {}
    if isinstance(meta, dict):
        for key in ("group", "category", "court"):
            if meta.get(key):
                return str(meta[key])
    return None


def parse_document(doc: dict[str, Any], split: str, min_chars: int = 1) -> dict[str, Any]:
    """Convert one Label-Studio document into the canonical schema."""
    anns = doc.get("annotations") or []
    results = (anns[0].get("result") or []) if anns else []
    sentences: list[dict[str, Any]] = []
    unknown: set[str] = set()
    for res in results:
        value = res.get("value") or {}
        text = " ".join(str(value.get("text", "")).split())
        if len(text) < min_chars:
            continue
        labels = value.get("labels") or []
        fine = labels[0] if labels else None
        if fine is not None and fine not in FINE_TO_COARSE:
            unknown.add(fine)
            fine = None
        sentences.append(
            {
                "idx": len(sentences),
                "text": text,
                "role_fine": fine,
                "role_coarse": FINE_TO_COARSE[fine] if fine else None,
                "in_summary": {"A1": None, "A2": None},
                "start": value.get("start"),
                "end": value.get("end"),
            }
        )
    if unknown:
        logger.warning("Doc %s: unknown labels %s set to None", doc.get("id"), sorted(unknown))
    meta = dict(doc.get("meta") or {}) if isinstance(doc.get("meta"), dict) else {}
    meta["group"] = _meta_group(doc)
    return {
        "doc_id": f"build_{doc.get('id')}",
        "source": "build",
        "split": split,
        "meta": meta,
        "sentences": sentences,
    }


def parse_file(path: str | Path, split: str, min_chars: int = 1) -> list[dict[str, Any]]:
    """Parse every document in a BUILD JSON file."""
    raw = load_raw(path)
    docs = [parse_document(d, split, min_chars) for d in raw]
    docs = [d for d in docs if d["sentences"]]
    logger.info(
        "BUILD %s: %d docs, %d sentences (%s)",
        split,
        len(docs),
        sum(len(d["sentences"]) for d in docs),
        Path(path).name,
    )
    return docs
