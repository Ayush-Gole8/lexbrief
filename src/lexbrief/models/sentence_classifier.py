"""M1: transformer sentence classifier ([CLS] -> dropout -> Linear(hidden, 13)).

Works with any BERT-style encoder: ``law-ai/InLegalBERT`` (M1) or ``bert-base-uncased``
(M1-ctrl), selected via ``model.encoder`` in the config.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from lexbrief.labels import FINE_LABELS
from lexbrief.utils.io import read_json, write_json

logger = logging.getLogger(__name__)

HEAD_FILE = "head.pt"
LABELS_FILE = "label_map.json"
ENCODER_DIR = "encoder"


class SentenceClassifier(nn.Module):
    """Encoder + linear head over the [CLS] representation."""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = len(FINE_LABELS),
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``[batch, num_labels]``."""
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls))

    def save(self, out_dir: str | Path, tokenizer: object | None = None) -> None:
        """Save encoder (``save_pretrained``), head state_dict and the label map."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out_dir / ENCODER_DIR)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir / ENCODER_DIR)
        torch.save(self.head.state_dict(), out_dir / HEAD_FILE)
        write_json(
            {"labels": list(FINE_LABELS), "num_labels": self.num_labels},
            out_dir / LABELS_FILE,
        )

    @classmethod
    def load(cls, run_dir: str | Path, dropout: float = 0.1) -> SentenceClassifier:
        """Load a model saved with :meth:`save`."""
        run_dir = Path(run_dir)
        meta = read_json(run_dir / LABELS_FILE)
        if meta["labels"] != list(FINE_LABELS):
            raise ValueError(f"Label map in {run_dir} does not match lexbrief.labels")
        model = cls(str(run_dir / ENCODER_DIR), num_labels=meta["num_labels"], dropout=dropout)
        model.head.load_state_dict(torch.load(run_dir / HEAD_FILE, map_location="cpu"))
        return model
