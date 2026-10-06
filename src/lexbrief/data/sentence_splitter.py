"""Legal-aware sentence splitting (spaCy rule-based sentencizer + abbreviation exceptions)."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

LEGAL_ABBREVIATIONS: tuple[str, ...] = (
    "No.",
    "Nos.",
    "vs.",
    "v.",
    "S.",
    "Sec.",
    "Art.",
    "Ltd.",
    "Pvt.",
    "Hon'ble",
    "i.e.",
    "e.g.",
    "viz.",
    "Cr.P.C.",
    "I.P.C.",
    "Ors.",
    "Anr.",
    # common companions in Indian judgments
    "Ss.",
    "Arts.",
    "Co.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Sr.",
    "Jr.",
    "Smt.",
    "Shri.",
    "Govt.",
    "Cl.",
    "Para.",
    "Paras.",
    "p.",
    "pp.",
    "etc.",
    "C.P.C.",
    "A.I.R.",
    "S.C.C.",
    "S.C.R.",
)

_PAGE_LINE = re.compile(
    r"^\s*(?:page\s*\d+(?:\s*of\s*\d+)?|[-–—]?\s*\d{1,4}\s*[-–—]?|\[\s*\d{1,4}\s*\])\s*$",
    re.IGNORECASE,
)
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WS = re.compile(r"[ \t ]+")


def clean_text(text: str) -> str:
    """Remove page-number lines, join hyphenated line breaks and normalise whitespace.

    Paragraph breaks (blank lines) are kept as ``\\n\\n``; single newlines become spaces.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if not _PAGE_LINE.match(ln)]
    text = "\n".join(lines)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [_WS.sub(" ", p.replace("\n", " ")).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def _variants(abbr: str) -> set[str]:
    return {abbr, abbr.lower(), abbr.upper(), abbr[:1].upper() + abbr[1:]}


@lru_cache(maxsize=4)
def get_nlp(model: str = "en_core_web_sm"):  # noqa: ANN201 - spacy.Language
    """Load a lightweight spaCy pipeline: tokenizer + rule-based sentencizer."""
    import spacy

    nlp = spacy.load(
        model,
        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"],
    )
    for abbr in LEGAL_ABBREVIATIONS:
        for form in _variants(abbr):
            nlp.tokenizer.add_special_case(form, [{"ORTH": form}])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000
    return nlp


def split_sentences(
    text: str, model: str = "en_core_web_sm", min_chars: int = 3, clean: bool = True
) -> list[str]:
    """Split legal text into sentences.

    Args:
        text: Raw text.
        model: spaCy model name (only its tokenizer is used).
        min_chars: Drop sentences shorter than this after stripping.
        clean: Apply :func:`clean_text` first.
    """
    if clean:
        text = clean_text(text)
    nlp = get_nlp(model)
    out: list[str] = []
    for para in text.split("\n\n"):
        if not para.strip():
            continue
        for sent in nlp(para).sents:
            s = sent.text.strip()
            if len(s) >= min_chars:
                out.append(s)
    return out
