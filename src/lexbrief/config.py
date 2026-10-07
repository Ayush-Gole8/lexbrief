"""Typed configuration loaded from ``configs/*.yaml`` with dotted CLI overrides.

Usage::

    cfg = load_config("configs/m1_inlegalbert.yaml", overrides=["train.lr=3e-5"])
    cfg.train.lr  # 3e-05 (float)

Every YAML key must correspond to a dataclass field; unknown keys raise ``ValueError``.
Missing keys keep their dataclass defaults, so an empty YAML file yields the defaults.
"""

from __future__ import annotations

import dataclasses
import logging
import types
import typing
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Filesystem locations (relative to the repo root)."""

    data_raw: str = "data/raw"
    data_interim: str = "data/interim"
    data_processed: str = "data/processed"
    data_samples: str = "data/samples"
    models_dir: str = "models"
    logs_dir: str = "outputs/logs"
    runs_dir: str = "outputs/runs"
    results_dir: str = "outputs/results"
    figures_dir: str = "outputs/figures"
    emb_dir: str = "data/interim/emb"


@dataclass
class DataConfig:
    """Dataset preparation settings."""

    # Raw sources
    build_repo_id: str = "opennyaiorg/InRhetoricalRoles"
    build_files: list[str] = field(default_factory=lambda: ["train.json", "dev.json", "test.json"])
    build_dir: str = "data/raw/build"
    zenodo_url: str = "https://zenodo.org/records/7152317/files/dataset.zip?download=1"
    zenodo_md5: str = "c77948fe580d26200f5c9fe53b81485d"
    zenodo_dir: str = "data/raw/zenodo"
    inext_subdir: str = "IN-Ext"
    # Outputs
    processed_dir: str = "data/processed"
    folds_file: str = "data/processed/inext_folds.json"
    stats_md: str = "outputs/results/data_stats.md"
    label_dist_figure: str = "outputs/figures/label_distribution.png"
    # Processing
    spacy_model: str = "en_core_web_sm"
    tokenizer: str = "law-ai/InLegalBERT"
    val_fraction: float = 0.1
    n_folds: int = 5
    fuzzy_threshold: float = 90.0
    partial_threshold: float | None = 90.0
    partial_min_tokens: int = 6
    partial_min_len_ratio: float = 0.6
    align_warn_below: float = 0.95
    min_sentence_chars: int = 3


@dataclass
class ModelConfig:
    """Sentence encoder and sequence-model settings."""

    name: str = "m1_inlegalbert"
    kind: str = "sentence"
    """``tfidf`` (M0), ``sentence`` (M1), ``context`` (M2)."""
    encoder: str = "law-ai/InLegalBERT"
    max_length: int = 128
    num_labels: int = 13
    dropout: float = 0.1
    context_window: int = 0
    lstm_hidden: int = 256
    lstm_layers: int = 1
    use_crf: bool = True
    use_position: bool = True
    """M3a: append relative position idx/n to each sentence embedding."""
    emb_run: str = "m1_inlegalbert_s42"
    """M3a: M1 run whose cached sentence embeddings are the input."""
    emb_pool: str = "cls"
    """M3a: which cached embedding to use, ``cls`` or ``mean``."""
    emb_dim: int = 768
    freeze_encoder: bool = False
    # M0 TF-IDF + logistic regression
    tfidf_ngram_max: int = 2
    tfidf_max_features: int = 50000
    tfidf_min_df: int = 2
    lr_c_grid: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0])
    lr_max_iter: int = 2000


@dataclass
class TrainConfig:
    """Optimisation settings (sized for a 6 GB GPU)."""

    lr: float = 2e-5
    head_lr: float = 1e-3
    batch_size: int = 16
    grad_accum_steps: int = 1
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    num_workers: int = 0
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 2
    eval_every_steps: int = 0
    output_dir: str = "models/checkpoints"
    eval_batch_size: int = 64
    seeds: list[int] = field(default_factory=lambda: [13, 42, 1234])
    class_weighting: str = "inv_sqrt"
    """``inv_sqrt`` (1/sqrt(freq), mean-normalised) or ``none``."""
    max_docs: int = 0
    """Smoke mode: use only the first N train/val docs (0 = all)."""
    log_every_steps: int = 50


@dataclass
class BriefConfig:
    """Role-budgeted extractive brief settings (Prompt 5). Section order comes from labels.py."""

    # Inputs
    roles_run: str = "m3a_bilstm_crf_s42"
    """Run whose ``outputs/results/<run>/preds_inext.jsonl`` supplies predicted roles."""
    emb_run: str = "m1_inlegalbert_s42"
    """Cached sentence embeddings (``paths.emb_dir/<run>/inext.pt``)."""
    emb_pool: str = "cls"
    # Budget
    budget_fraction: float = 0.3
    """Default L as a fraction of document words (``--budget`` > 1 means absolute words)."""
    strategy: str = "A3"
    expert_shares: dict[str, float] = field(
        default_factory=lambda: {
            "RULING": 0.25,
            "ISSUES": 0.20,
            "FACTS": 0.18,
            "STATUTE": 0.10,
            "PRECEDENT": 0.10,
            "REASONING": 0.10,
            "ARGUMENTS": 0.07,
        }
    )
    """A2 expert-priority shares (DELSumm ordering RULING > ISSUES > FACTS > STA/PRE/REASONING >
    ARGUMENTS)."""
    ruling_min_sentences: int = 3
    min_sentences_per_role: int = 1
    min_alpha_words: int = 2
    """Sentences with fewer alphabetic words (2+ letters) are never selected, by any system."""
    # Features
    edge_threshold: float = 0.1
    pagerank_damping: float = 0.85
    pos_bins: int = 10
    pos_laplace: float = 1.0
    # Scorer
    weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    """Default (alpha cent, beta pos, gamma cue, delta conf) before grid search."""
    weight_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])
    # Selector / baselines
    redundancy_threshold: float = 0.8
    mmr_lambda: float = 0.7
    rouge_stemmer: bool = True


@dataclass
class DemoConfig:
    """Streamlit demo settings."""

    model_dir: str = "models/best"
    max_upload_mb: int = 20
    port: int = 8501


@dataclass
class Config:
    """Root configuration."""

    seed: int = 42
    device: str = "cuda"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    brief: BriefConfig = field(default_factory=BriefConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)


_TRUE = {"true", "yes", "1", "on"}
_FALSE = {"false", "no", "0", "off"}


def _coerce(value: Any, hint: Any, key: str) -> Any:
    """Coerce a YAML/CLI value to the annotated field type where unambiguous."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        return _coerce(value, non_none[0], key) if len(non_none) == 1 else value
    try:
        if hint is bool:
            if isinstance(value, bool):
                return value
            if str(value).lower() in _TRUE:
                return True
            if str(value).lower() in _FALSE:
                return False
            raise ValueError
        if hint is int:
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            return int(value)
        if hint is float:
            if isinstance(value, bool):
                raise ValueError
            return float(value)  # handles YAML's "3e-5" string
        if hint is str:
            return str(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config key {key!r}: cannot convert {value!r} to {hint}") from None
    return value


def _build(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    """Recursively build dataclass ``cls`` from a (partial) dict, keeping defaults."""
    if not isinstance(data, dict):
        raise ValueError(f"Config section {prefix.rstrip('.') or '<root>'!r} must be a mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(
            f"Unknown config key(s) {sorted(prefix + k for k in unknown)}; "
            f"valid keys under {prefix.rstrip('.') or '<root>'!r}: {sorted(names)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value or {}, f"{prefix}{name}.")
        else:
            kwargs[name] = _coerce(value, hint, prefix + name)
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    """Turn ``["train.lr=3e-5", "seed=7"]`` into a nested dict."""
    nested: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like 'section.key=value'")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        if not all(keys):
            raise ValueError(f"Override {item!r} has an empty key")
        value = yaml.safe_load(raw) if raw.strip() else ""
        node = nested
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override {item!r} conflicts with a scalar override")
        node[keys[-1]] = value
    return nested


def _read_yaml_with_base(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read YAML; a top-level ``base: other.yaml`` (relative path) is merged underneath."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular config base chain: {[str(p) for p in (*_seen, path)]}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    logger.debug("Loaded config %s", path)
    base = data.pop("base", None)
    if base:
        data = _deep_merge(_read_yaml_with_base(path.parent / base, (*_seen, path)), data)
    return data


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config onto the defaults and apply dotted overrides.

    Args:
        path: YAML file; ``None`` uses defaults only.
        overrides: Strings like ``"train.lr=3e-5"``.

    Returns:
        A fully typed :class:`Config`.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = _read_yaml_with_base(Path(path))
    data = _deep_merge(data, parse_overrides(overrides))
    return _build(Config, data)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a config to a plain dict (e.g. for logging or saving with a run)."""
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write a config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
