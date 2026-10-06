"""Tests for lexbrief.data.align and lexbrief.data.sentence_splitter."""

from __future__ import annotations

import logging

import pytest

from lexbrief.data.align import align_sentences, normalize

DOC = [
    "The appellant filed a suit for declaration of title.",
    "The High Court dismissed the second appeal.",
    "We find no merit in this appeal and it is accordingly dismissed.",
    "There shall be no order as to costs.",
]


def test_normalize_quotes_case_space_punct() -> None:
    assert normalize("  The  Testator’s   WILL, dated 1924. ") == "the testator s will dated 1924"
    assert normalize("testator 's will") == normalize("testator's will")
    assert normalize("“quoted”") == "quoted"


def test_exact_match_after_normalisation() -> None:
    res = align_sentences(["the HIGH court dismissed the second appeal"], DOC)
    assert res.matches == [1]
    assert res.methods == ["exact"]
    assert res.match_rate == 1.0


def test_fuzzy_fallback() -> None:
    summ = ["We find no merit in the appeal, and it is accordingly dimissed."]
    res = align_sentences(summ, DOC, threshold=90)
    assert res.matches == [2]
    assert res.methods == ["fuzzy"]
    assert 90 <= res.scores[0] < 100


def test_no_match_below_threshold_and_warning(caplog: pytest.LogCaptureFixture) -> None:
    summ = ["Completely unrelated sentence about tax law.", DOC[0]]
    with caplog.at_level(logging.WARNING):
        res = align_sentences(summ, DOC, threshold=90, warn_below=0.95, name="t")
    assert res.matches == [None, 0]
    assert res.match_rate == pytest.approx(0.5)
    assert res.matched_doc_indices == {0}
    assert "Low alignment rate" in caplog.text


def test_partial_tier_for_sub_spans() -> None:
    doc = [
        "On 19th September 1945 an injunction was issued and inspite of this injunction "
        "she executed two deeds of settlement in favour of the other defendants.",
        "It was dismissed.",
    ]
    summ = ["inspite of this injunction she executed two deeds of settlement in favour"]
    assert align_sentences(summ, doc, warn_below=None).matches == [None]
    res = align_sentences(summ, doc, warn_below=None, partial_threshold=90)
    assert res.matches == [0]
    assert res.methods == ["partial"]
    # short queries never use the partial tier
    short = align_sentences(["it was"], doc, warn_below=None, partial_threshold=90)
    assert short.matches == [None]


def test_empty_inputs() -> None:
    assert align_sentences([], DOC).match_rate == 1.0
    assert align_sentences(["x y z"], []).matches == [None]


def test_sentence_splitter_legal_abbreviations() -> None:
    from lexbrief.data.sentence_splitter import split_sentences

    text = (
        "The appellant was convicted under S. 302 of the I.P.C. read with Sec. 34 by the "
        "Sessions Court. "
        "In State of Punjab vs. Baldev Singh, Civil Appeal No. 12 of 1999, the Hon'ble Court "
        "held that the procedure under the Cr.P.C. was mandatory, i.e. not directory. "
        "The respondents, M/s ABC Pvt. Ltd. and Ors., relied on Art. 14, viz. equality "
        "before law, e.g. as applied in Anr. v. Union of India."
    )
    sents = split_sentences(text)
    assert len(sents) == 3, sents
    assert sents[0].endswith("by the Sessions Court.")
    assert sents[1].startswith("In State of Punjab vs. Baldev Singh")
    assert sents[2].endswith("Anr. v. Union of India.")


def test_clean_text_page_numbers_and_hyphens() -> None:
    from lexbrief.data.sentence_splitter import clean_text

    raw = "The judg-\nment was delivered.\n12\nPage 3 of 10\nIt was affirmed."
    assert clean_text(raw) == "The judgment was delivered. It was affirmed."
