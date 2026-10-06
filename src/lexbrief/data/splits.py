"""Document-level splits: BUILD train/val/test and IN-Ext cross-validation folds."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sklearn.model_selection import KFold, train_test_split

logger = logging.getLogger(__name__)


def _strata(docs: list[dict[str, Any]], min_count: int) -> list[str] | None:
    keys = [str((d.get("meta") or {}).get("group") or "unknown") for d in docs]
    counts = Counter(keys)
    keys = [k if counts[k] >= min_count else "other" for k in keys]
    counts = Counter(keys)
    if len(counts) < 2 or min(counts.values()) < min_count:
        return None
    return keys


def split_train_val(
    docs: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split documents into train/val, stratified by ``meta.group`` when possible."""
    n_val = max(1, round(len(docs) * val_fraction))
    strata = _strata(docs, min_count=2)
    if strata is None:
        logger.warning("Too few docs per meta group to stratify; using a random split")
    train, val = train_test_split(
        docs, test_size=n_val, random_state=seed, shuffle=True, stratify=strata
    )
    for d in train:
        d["split"] = "train"
    for d in val:
        d["split"] = "val"
    return list(train), list(val)


def make_build_splits(
    train_docs: list[dict[str, Any]],
    dev_docs: list[dict[str, Any]],
    test_docs: list[dict[str, Any]] | None,
    val_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{"train", "val", "test"}``.

    If ``test_docs`` is labelled (non-empty list passed in), dev becomes val and test stays test.
    Otherwise train is split 90/10 into train/val and dev is used as test.
    """
    if test_docs:
        logger.info("test.json is labelled: train=train, val=dev, test=test")
        for d in train_docs:
            d["split"] = "train"
        for d in dev_docs:
            d["split"] = "val"
        for d in test_docs:
            d["split"] = "test"
        return {"train": train_docs, "val": dev_docs, "test": test_docs}
    logger.info("test.json unlabelled: train -> %.0f/%.0f train/val, dev -> test",
                100 * (1 - val_fraction), 100 * val_fraction)
    train, val = split_train_val(train_docs, val_fraction, seed)
    for d in dev_docs:
        d["split"] = "test"
    return {"train": train, "val": val, "test": dev_docs}


def make_folds(doc_ids: list[str], n_folds: int, seed: int) -> list[dict[str, Any]]:
    """Document-level K-fold split; returns ``[{"fold", "train", "test"}]``."""
    ids = sorted(doc_ids)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        {"fold": k, "train": [ids[i] for i in tr], "test": [ids[i] for i in te]}
        for k, (tr, te) in enumerate(kf.split(ids))
    ]
