"""Tests for BUILD parsing, splits and stats helpers (synthetic data)."""

from __future__ import annotations

from lexbrief.data.build_parser import has_labels, parse_document
from lexbrief.data.splits import make_build_splits, make_folds
from lexbrief.data.stats import recommend_max_len, transition_matrix


def _raw(doc_id: int, group: str, labels: list[str]) -> dict:
    return {
        "id": doc_id,
        "meta": {"group": group},
        "data": {"text": "..."},
        "annotations": [
            {
                "result": [
                    {"value": {"start": i, "end": i + 1, "text": f" s{i} ", "labels": [lab]}}
                    for i, lab in enumerate(labels)
                ]
            }
        ],
    }


def test_parse_document_keeps_order_and_maps_coarse() -> None:
    doc = parse_document(_raw(7, "Tax", ["PREAMBLE", "FAC", "RATIO", "RPC"]), "train")
    assert doc["doc_id"] == "build_7"
    assert doc["meta"]["group"] == "Tax"
    assert [s["text"] for s in doc["sentences"]] == ["s0", "s1", "s2", "s3"]
    assert [s["role_coarse"] for s in doc["sentences"]] == ["DROP", "FACTS", "REASONING", "RULING"]
    assert [s["idx"] for s in doc["sentences"]] == [0, 1, 2, 3]


def test_has_labels() -> None:
    assert has_labels([_raw(1, "Tax", ["FAC"])])
    unlabelled = _raw(1, "Tax", ["FAC"])
    unlabelled["annotations"][0]["result"][0]["value"]["labels"] = []
    assert not has_labels([unlabelled])
    assert not has_labels([{"id": 1, "annotations": []}])


def test_build_splits_unlabelled_test_and_determinism() -> None:
    def docs() -> list[dict]:
        return [
            parse_document(_raw(i, "Tax" if i % 2 else "Criminal", ["FAC"]), "train")
            for i in range(40)
        ]

    dev = [parse_document(_raw(100 + i, "Tax", ["FAC"]), "dev") for i in range(5)]
    s1 = make_build_splits(docs(), dev, None, 0.1, seed=42)
    s2 = make_build_splits(docs(), dev, None, 0.1, seed=42)
    assert len(s1["train"]) == 36 and len(s1["val"]) == 4
    assert [d["doc_id"] for d in s1["val"]] == [d["doc_id"] for d in s2["val"]]
    assert {d["split"] for d in s1["test"]} == {"test"}
    assert not {d["doc_id"] for d in s1["train"]} & {d["doc_id"] for d in s1["val"]}
    groups = [d["meta"]["group"] for d in s1["val"]]
    assert groups.count("Tax") == groups.count("Criminal") == 2  # stratified


def test_build_splits_labelled_test() -> None:
    train = [parse_document(_raw(i, "Tax", ["FAC"]), "train") for i in range(3)]
    dev = [parse_document(_raw(10, "Tax", ["FAC"]), "dev")]
    test = [parse_document(_raw(20, "Tax", ["FAC"]), "test")]
    s = make_build_splits(train, dev, test, 0.1, seed=42)
    assert len(s["train"]) == 3 and s["val"] == dev and s["test"] == test


def test_folds_cover_each_doc_once() -> None:
    ids = [f"d{i}" for i in range(50)]
    folds = make_folds(ids, 5, seed=42)
    tested = [d for f in folds for d in f["test"]]
    assert sorted(tested) == sorted(ids)
    assert all(len(f["test"]) == 10 for f in folds)
    assert all(not set(f["train"]) & set(f["test"]) for f in folds)


def test_transition_matrix_rows_normalised() -> None:
    doc = parse_document(_raw(1, "Tax", ["FAC", "FAC", "RATIO", "RPC"]), "train")
    tm = transition_matrix([doc])
    assert abs(tm.sum(axis=1)[1] - 1.0) < 1e-9  # FAC row
    assert tm.sum(axis=1)[0] == 0.0  # PREAMBLE unseen


def test_recommend_max_len() -> None:
    assert recommend_max_len(60.2) == 64
    assert recommend_max_len(94) == 96
