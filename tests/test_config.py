"""Tests for lexbrief.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from lexbrief.config import Config, load_config, save_config


def test_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.train.num_workers == 0
    assert cfg.model.encoder == "law-ai/InLegalBERT"


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == Config()


def test_yaml_and_overrides(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\ntrain:\n  lr: 5e-5\n  epochs: 2\n", encoding="utf-8")
    cfg = load_config(p, overrides=["train.lr=3e-5", "seed=7", "model.use_crf=false"])
    assert cfg.train.lr == pytest.approx(3e-5)
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.epochs == 2
    assert cfg.seed == 7
    assert cfg.model.use_crf is False
    assert cfg.train.batch_size == Config().train.batch_size


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(overrides=["train.not_a_key=1"])


def test_bad_type_raises() -> None:
    with pytest.raises(ValueError, match="cannot convert"):
        load_config(overrides=["train.epochs=abc"])


def test_save_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(overrides=["brief.target_words=250"])
    out = tmp_path / "saved.yaml"
    save_config(cfg, out)
    assert load_config(out) == cfg
