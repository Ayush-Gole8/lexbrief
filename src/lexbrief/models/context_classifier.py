"""M2: context classifier.

Same network as M1, but each input is ``[CLS] prev [SEP] target [SEP] next [SEP]`` with empty
strings at document edges. Prev/next are truncated first so the target sentence is never cut
(see :func:`lexbrief.training.datasets.build_input`); token type 1 marks the target.
"""

from __future__ import annotations

from lexbrief.models.sentence_classifier import SentenceClassifier
from lexbrief.training.datasets import build_input

__all__ = ["ContextClassifier", "build_input"]


class ContextClassifier(SentenceClassifier):
    """[CLS]-pooled classifier over a prev/target/next input (architecture identical to M1)."""

    uses_context: bool = True
