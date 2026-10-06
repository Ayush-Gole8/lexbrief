"""Tests for sentence-classifier data encoding, class weights, metrics and M0 (no downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lexbrief.labels import FINE2ID, FINE_LABELS
from lexbrief.training.datasets import build_input, class_weights, flatten
from lexbrief.training.evaluate_classifier import coarse_metrics, fine_metrics

CLS, SEP = 101, 102


def test_build_input_no_context_truncates_target() -> None:
    ids, types = build_input(list(range(1, 200)), 10, CLS, SEP)
    assert ids == [CLS, 1, 2, 3, 4, 5, 6, 7, 8, SEP]
    assert len(types) == len(ids) and set(types) == {0}


def test_build_input_context_never_cuts_target() -> None:
    prev, target, nxt = list(range(1000, 1100)), list(range(1, 21)), list(range(2000, 2100))
    ids, types = build_input(target, 40, CLS, SEP, prev=prev, nxt=nxt)
    assert len(ids) == 40
    assert ids[0] == CLS and ids[-1] == SEP and ids.count(SEP) == 3
    first, second = ids.index(SEP), ids.index(SEP, ids.index(SEP) + 1)
    assert ids[first + 1 : second] == target  # full target kept
    assert ids[1:first] == prev[-len(ids[1:first]) :]  # prev keeps its tail
    assert ids[second + 1 : -1] == nxt[: len(ids[second + 1 : -1])]  # next keeps its head
    assert [i for i, t in zip(ids, types, strict=True) if t == 1] == [*target, SEP]


def test_build_input_context_edges_and_leftover_room() -> None:
    target = [1, 2, 3]
    ids, _ = build_input(target, 20, CLS, SEP, prev=[], nxt=list(range(50, 80)))
    assert ids[:2] == [CLS, SEP]  # empty prev at document start
    assert len(ids) == 20  # unused prev budget goes to next
    ids2, _ = build_input(target, 20, CLS, SEP, prev=[7, 8], nxt=[])
    assert ids2 == [CLS, 7, 8, SEP, 1, 2, 3, SEP, SEP]


def test_build_input_context_target_longer_than_budget() -> None:
    ids, _ = build_input(list(range(1, 100)), 16, CLS, SEP, prev=[5], nxt=[6])
    assert len(ids) == 16
    assert ids[:2] == [CLS, SEP] and ids[-2:] == [SEP, SEP]


def test_class_weights_inverse_sqrt() -> None:
    labels = [0] * 100 + [1] * 25 + [2] * 4
    w = class_weights(labels)
    assert w.shape == (len(FINE_LABELS),)
    assert torch.isclose(w.mean(), torch.tensor(1.0))
    assert w[0] / w[1] == pytest.approx(0.5, rel=1e-5)  # sqrt(25/100)
    assert w[1] / w[2] == pytest.approx(0.4, rel=1e-5)  # sqrt(4/25)
    assert torch.equal(class_weights(labels, "none"), torch.ones(len(FINE_LABELS)))


def test_flatten_context_and_labels() -> None:
    doc = {
        "doc_id": "d",
        "sentences": [
            {"idx": 0, "text": "a", "role_fine": "FAC", "role_coarse": "FACTS"},
            {"idx": 1, "text": "b", "role_fine": None, "role_coarse": None},
            {"idx": 2, "text": "c", "role_fine": "RPC", "role_coarse": "RULING"},
        ],
    }
    ex = flatten([doc])
    assert [(e.prev, e.text, e.next) for e in ex] == [
        ("", "a", "b"),
        ("a", "b", "c"),
        ("b", "c", ""),
    ]
    assert [e.label for e in ex] == [FINE2ID["FAC"], -100, FINE2ID["RPC"]]


def test_fine_and_coarse_metrics() -> None:
    gold = [0, 1, 1, 11]
    pred = [0, 1, 2, 11]
    m = fine_metrics(gold, pred)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class_f1"]["PREAMBLE"] == 1.0
    assert set(m["per_class_f1"]) == set(FINE_LABELS)
    c = coarse_metrics(["FACTS", "FACTS", "RULING"], ["FACTS", "DROP", "RULING"])
    assert c["labels"] == ["FACTS", "RULING"]
    assert c["per_class_f1"]["FACTS"] == pytest.approx(2 / 3)
    assert c["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)


def test_tfidf_baseline_fit_predict_save(tmp_path: Path) -> None:
    from lexbrief.models.tfidf_baseline import TfidfBaseline, train_tfidf

    texts = ["the appeal is dismissed", "appeal allowed with costs", "facts of the case are"] * 10
    labels = [FINE2ID["RPC"], FINE2ID["RPC"], FINE2ID["FAC"]] * 10
    m = train_tfidf(texts, labels, texts, labels, c_grid=[1.0], min_df=1)
    p = m.predict_proba(["appeal is dismissed"])
    assert p.shape == (1, len(FINE_LABELS))
    assert np.isclose(p.sum(), 1.0)
    assert int(p.argmax()) == FINE2ID["RPC"]
    m.save(tmp_path)
    assert np.allclose(
        TfidfBaseline.load(tmp_path).predict_proba(["facts"]), m.predict_proba(["facts"])
    )
