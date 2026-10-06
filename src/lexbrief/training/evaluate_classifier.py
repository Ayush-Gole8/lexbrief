"""Evaluate a trained sentence classifier run on BUILD test and IN-Ext (cross-dataset).

Outputs in ``outputs/results/<run>/``: ``metrics.json``, ``preds_build_test.jsonl``,
``preds_inext.jsonl`` and ``confusion_build_test.png``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from lexbrief.config import Config, load_config
from lexbrief.labels import BRIEF_SECTION_ORDER, FINE_LABELS, FINE_TO_COARSE
from lexbrief.training.datasets import EncodedDataset, SentenceExample, flatten, load_split
from lexbrief.utils.gpu import autocast_ctx, get_device
from lexbrief.utils.io import write_json, write_jsonl

logger = logging.getLogger(__name__)

RUN_CONFIG = "config.yaml"


# --------------------------------------------------------------------------- metrics
def fine_metrics(gold: list[int], pred: list[int]) -> dict[str, Any]:
    """Macro / weighted F1 and per-class F1 over the 13 fine labels."""
    labels = list(range(len(FINE_LABELS)))
    per = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(gold, pred, labels=labels, average="weighted", zero_division=0)
        ),
        "accuracy": float(np.mean(np.asarray(gold) == np.asarray(pred))) if gold else 0.0,
        "per_class_f1": {FINE_LABELS[i]: float(per[i]) for i in labels},
        "support": {FINE_LABELS[i]: int(np.sum(np.asarray(gold) == i)) for i in labels},
    }


def coarse_metrics(gold: list[str], pred: list[str], labels: list[str] | None = None) -> dict:
    """Macro-F1 over coarse sections (DROP excluded from the average, still counted as errors).

    Args:
        gold: Gold coarse labels.
        pred: Predicted coarse labels (may include DROP).
        labels: Sections to average over; default = the 7 brief sections present in gold.
    """
    if labels is None:
        present = set(gold)
        labels = [c for c in BRIEF_SECTION_ORDER if c in present]
    per = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "per_class_f1": {c: float(f) for c, f in zip(labels, per, strict=True)},
        "n": len(gold),
    }


def plot_confusion(gold: list[int], pred: list[int], path: str | Path, title: str) -> Path:
    """Row-normalised 13x13 confusion matrix PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(range(len(FINE_LABELS)))
    cm = confusion_matrix(gold, pred, labels=labels).astype(np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, FINE_LABELS, rotation=60, ha="right")
    ax.set_yticks(labels, FINE_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    for i in labels:
        for j in labels:
            if cmn[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{cmn[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_neural(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[SentenceExample],
    max_length: int,
    context: bool,
    batch_size: int,
    device: torch.device,
    bf16: bool = True,
) -> np.ndarray:
    """Softmax probabilities ``[n, 13]`` for a list of examples."""
    from torch.utils.data import DataLoader

    ds = EncodedDataset(examples, tokenizer, max_length, context=context)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate, num_workers=0)
    model.eval()
    out: list[np.ndarray] = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with autocast_ctx(device, enabled=bf16):
            logits = model(**batch)
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(FINE_LABELS)), dtype=np.float32)


def load_predictor(run_dir: Path, cfg: Config, device: torch.device):  # noqa: ANN201
    """Return ``predict(examples) -> probs`` for a saved run of any kind."""
    kind = cfg.model.kind
    if kind == "tfidf":
        from lexbrief.models.tfidf_baseline import TfidfBaseline

        m0 = TfidfBaseline.load(run_dir)
        return lambda ex: m0.predict_proba([e.text for e in ex])
    if kind in ("sentence", "context"):
        from transformers import AutoTokenizer

        from lexbrief.models.sentence_classifier import ENCODER_DIR, SentenceClassifier

        model = SentenceClassifier.load(run_dir, dropout=cfg.model.dropout).to(device)
        tok = AutoTokenizer.from_pretrained(run_dir / ENCODER_DIR)
        return lambda ex: predict_neural(
            model,
            tok,
            ex,
            cfg.model.max_length,
            context=kind == "context",
            batch_size=cfg.train.eval_batch_size,
            device=device,
            bf16=cfg.train.bf16,
        )
    raise ValueError(f"Unknown model.kind {kind!r}")


def _pred_rows(examples: list[SentenceExample], probs: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for e, p in zip(examples, probs, strict=True):
        k = int(p.argmax())
        rows.append(
            {
                "doc_id": e.doc_id,
                "idx": e.idx,
                "gold_fine": FINE_LABELS[e.label] if e.label >= 0 else None,
                "pred_fine": FINE_LABELS[k],
                "gold_coarse": e.gold_coarse,
                "pred_coarse": FINE_TO_COARSE[FINE_LABELS[k]],
                "probs": [round(float(x), 5) for x in p],
            }
        )
    return rows


# --------------------------------------------------------------------------- driver
def evaluate_run(
    run_name: str, models_dir: str = "models", results_dir: str = "outputs/results"
) -> dict[str, Any]:
    """Evaluate ``models/<run_name>`` on build_test and inext; write preds and metrics."""
    run_dir = Path(models_dir) / run_name
    if not (run_dir / RUN_CONFIG).is_file():
        raise FileNotFoundError(f"No trained run at {run_dir} (missing {RUN_CONFIG})")
    cfg = load_config(run_dir / RUN_CONFIG)
    out_dir = Path(results_dir) / run_name
    device = get_device(cfg.device)
    predict = load_predictor(run_dir, cfg, device)
    metrics: dict[str, Any] = {
        "run": run_name,
        "model": cfg.model.name,
        "kind": cfg.model.kind,
        "seed": cfg.seed,
    }

    # BUILD test (fine gold)
    test_ex = [e for e in flatten(load_split(cfg.data.processed_dir, "build_test")) if e.label >= 0]
    probs = predict(test_ex)
    rows = _pred_rows(test_ex, probs)
    write_jsonl(rows, out_dir / "preds_build_test.jsonl")
    gold = [e.label for e in test_ex]
    pred = [int(k) for k in probs.argmax(axis=1)]
    metrics["build_test"] = {
        "fine": fine_metrics(gold, pred),
        "coarse": coarse_metrics(
            [r["gold_coarse"] for r in rows],
            [r["pred_coarse"] for r in rows],
            labels=list(BRIEF_SECTION_ORDER),
        ),
    }
    plot_confusion(gold, pred, out_dir / "confusion_build_test.png", f"{run_name} - BUILD test")

    # IN-Ext (coarse, segment-inferred gold; only aligned sentences have a role)
    inext_path = Path(cfg.data.processed_dir) / "inext.jsonl"
    if inext_path.is_file():
        inext_ex = flatten(load_split(cfg.data.processed_dir, "inext"))
        probs_x = predict(inext_ex)
        rows_x = _pred_rows(inext_ex, probs_x)
        write_jsonl(rows_x, out_dir / "preds_inext.jsonl")
        labelled = [r for r in rows_x if r["gold_coarse"]]
        metrics["inext"] = {
            "coarse": coarse_metrics(
                [r["gold_coarse"] for r in labelled], [r["pred_coarse"] for r in labelled]
            ),
            "n_sentences": len(rows_x),
            "n_with_gold": len(labelled),
            "gold_source": "segment_inferred",
        }
    write_json(metrics, out_dir / "metrics.json")
    b = metrics["build_test"]
    logger.info(
        "%s | BUILD test fine macro-F1 %.4f weighted-F1 %.4f | coarse macro-F1 %.4f%s",
        run_name,
        b["fine"]["macro_f1"],
        b["fine"]["weighted_f1"],
        b["coarse"]["macro_f1"],
        f" | IN-Ext coarse macro-F1 {metrics['inext']['coarse']['macro_f1']:.4f}"
        if "inext" in metrics
        else "",
    )
    return metrics
