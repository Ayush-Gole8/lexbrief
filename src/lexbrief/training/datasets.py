"""Sentence-level datasets and input encoding for M1 (sentence) and M2 (context) classifiers."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lexbrief.labels import FINE2ID, FINE_LABELS
from lexbrief.utils.io import read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class SentenceExample:
    """One sentence with its document context."""

    doc_id: str
    idx: int
    text: str
    prev: str
    next: str
    label: int  # fine label id, -100 if unknown
    gold_coarse: str | None


def load_split(processed_dir: str | Path, name: str, max_docs: int = 0) -> list[dict[str, Any]]:
    """Load ``<processed_dir>/<name>.jsonl`` (e.g. ``build_train``), optionally the first N docs."""
    path = Path(processed_dir) / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `lexbrief prepare-data` first")
    docs = read_jsonl(path)
    return docs[:max_docs] if max_docs > 0 else docs


def flatten(docs: list[dict[str, Any]]) -> list[SentenceExample]:
    """Turn documents into sentence examples with prev/next context (empty at edges)."""
    out: list[SentenceExample] = []
    for d in docs:
        sents = d["sentences"]
        for i, s in enumerate(sents):
            fine = s.get("role_fine")
            out.append(
                SentenceExample(
                    doc_id=d["doc_id"],
                    idx=s["idx"],
                    text=s["text"],
                    prev=sents[i - 1]["text"] if i > 0 else "",
                    next=sents[i + 1]["text"] if i + 1 < len(sents) else "",
                    label=FINE2ID[fine] if fine in FINE2ID else -100,
                    gold_coarse=s.get("role_coarse"),
                )
            )
    return out


def class_weights(labels: list[int], scheme: str = "inv_sqrt") -> torch.Tensor:
    """Per-class CE weights. ``inv_sqrt``: 1/sqrt(count), normalised to mean 1 over classes."""
    n = len(FINE_LABELS)
    if scheme == "none":
        return torch.ones(n)
    if scheme != "inv_sqrt":
        raise ValueError(f"Unknown class_weighting {scheme!r}")
    counts = Counter(lab for lab in labels if lab >= 0)
    c = np.array([counts.get(i, 0) for i in range(n)], dtype=np.float64)
    w = 1.0 / np.sqrt(np.maximum(c, 1.0))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_input(
    target: list[int],
    max_length: int,
    cls_id: int,
    sep_id: int,
    prev: list[int] | None = None,
    nxt: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Assemble wordpiece ids (no special tokens in inputs) into one model input.

    Without context: ``[CLS] target [SEP]``.
    With context:    ``[CLS] prev [SEP] target [SEP] next [SEP]`` where prev/next are truncated
    first (prev keeps its *last* tokens, next its *first*) so the target is never cut unless it
    alone exceeds the budget.

    Returns:
        ``(input_ids, token_type_ids)``; token type 1 marks the target segment in context mode.
    """
    if prev is None and nxt is None:
        t = target[: max_length - 2]
        return [cls_id, *t, sep_id], [0] * (len(t) + 2)

    prev = prev or []
    nxt = nxt or []
    budget = max_length - 4  # [CLS] + 3x [SEP]
    t = target[:budget]
    rest = budget - len(t)
    half = rest // 2
    n_prev = min(len(prev), half)
    n_next = min(len(nxt), rest - n_prev)
    n_prev = min(len(prev), rest - n_next)  # give unused room back to prev
    p = prev[len(prev) - n_prev :] if n_prev else []
    q = nxt[:n_next]
    ids = [cls_id, *p, sep_id, *t, sep_id, *q, sep_id]
    types = [0] * (len(p) + 2) + [1] * (len(t) + 1) + [0] * (len(q) + 1)
    return ids, types


class EncodedDataset(Dataset):
    """Pre-tokenised sentence dataset (tokenises every unique text once)."""

    def __init__(
        self,
        examples: list[SentenceExample],
        tokenizer: Any,
        max_length: int,
        context: bool = False,
    ) -> None:
        self.examples = examples
        unique = {e.text for e in examples}
        if context:
            unique |= {e.prev for e in examples} | {e.next for e in examples}
        texts = sorted(unique)
        enc = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        ids = dict(zip(texts, enc, strict=True))
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.items: list[tuple[list[int], list[int], int]] = []
        for e in examples:
            if context:
                inp, tt = build_input(
                    ids[e.text], max_length, cls_id, sep_id, prev=ids[e.prev], nxt=ids[e.next]
                )
            else:
                inp, tt = build_input(ids[e.text], max_length, cls_id, sep_id)
            self.items.append((inp, tt, e.label))
        self.pad_id = tokenizer.pad_token_id or 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[list[int], list[int], int]:
        return self.items[i]

    def collate(self, batch: list[tuple[list[int], list[int], int]]) -> dict[str, torch.Tensor]:
        """Pad a batch to its longest sequence."""
        width = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        types = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for r, (inp, tt, _) in enumerate(batch):
            ids[r, : len(inp)] = torch.tensor(inp)
            types[r, : len(tt)] = torch.tensor(tt)
            mask[r, : len(inp)] = 1
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return {"input_ids": ids, "token_type_ids": types, "attention_mask": mask, "labels": labels}
