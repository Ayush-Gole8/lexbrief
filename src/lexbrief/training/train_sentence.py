"""Train sentence-level role classifiers: M0 (TF-IDF), M1 (sentence), M2 (context).

Custom loop (no HF Trainer): AdamW, linear warmup/decay, bf16 autocast, gradient
accumulation, class-weighted CE (inverse sqrt frequency), gradient clipping, early stopping on
val macro-F1, TensorBoard logging and peak-VRAM logging per epoch.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from lexbrief.config import Config, save_config
from lexbrief.labels import FINE_LABELS
from lexbrief.training.datasets import (
    EncodedDataset,
    SentenceExample,
    class_weights,
    flatten,
    load_split,
)
from lexbrief.training.evaluate_classifier import RUN_CONFIG, predict_neural
from lexbrief.utils.gpu import autocast_ctx, get_device, peak_memory, vram_report
from lexbrief.utils.io import write_json
from lexbrief.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_name_for(cfg: Config) -> str:
    """``<model.name>_s<seed>`` (+ ``_smoke`` when ``train.max_docs`` is set)."""
    name = f"{cfg.model.name}_s{cfg.seed}"
    return f"{name}_smoke" if cfg.train.max_docs > 0 else name


def _macro_f1(gold: list[int], pred: list[int]) -> float:
    labels = list(range(len(FINE_LABELS)))
    return float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))


def _load_examples(cfg: Config) -> tuple[list[SentenceExample], list[SentenceExample]]:
    md = cfg.train.max_docs
    train = [
        e for e in flatten(load_split(cfg.data.processed_dir, "build_train", md)) if e.label >= 0
    ]
    val = [e for e in flatten(load_split(cfg.data.processed_dir, "build_val", md)) if e.label >= 0]
    logger.info("Examples: train=%d val=%d (max_docs=%d)", len(train), len(val), md)
    return train, val


# --------------------------------------------------------------------------- M0
def train_tfidf_run(cfg: Config) -> dict[str, Any]:
    """Train M0 and save it to ``models/<run>/``."""
    from lexbrief.models.tfidf_baseline import train_tfidf

    set_seed(cfg.seed)
    run = run_name_for(cfg)
    out_dir = Path(cfg.paths.models_dir) / run
    train, val = _load_examples(cfg)
    t0 = time.time()
    m = cfg.model
    model = train_tfidf(
        [e.text for e in train],
        [e.label for e in train],
        [e.text for e in val],
        [e.label for e in val],
        c_grid=m.lr_c_grid,
        ngram_max=m.tfidf_ngram_max,
        max_features=m.tfidf_max_features,
        min_df=m.tfidf_min_df,
        max_iter=m.lr_max_iter,
        seed=cfg.seed,
    )
    model.save(out_dir)
    save_config(cfg, out_dir / RUN_CONFIG)
    best_c = max(model.c_scores, key=model.c_scores.get)
    summary = {
        "run": run,
        "best_C": best_c,
        "val_macro_f1": model.c_scores[best_c],
        "c_scores": {str(k): v for k, v in model.c_scores.items()},
        "train_seconds": round(time.time() - t0, 1),
    }
    write_json(summary, out_dir / "train_summary.json")
    logger.info("M0 saved to %s (%.1fs)", out_dir, summary["train_seconds"])
    return summary


# --------------------------------------------------------------------------- M1 / M2
def _linear_warmup(optimizer: torch.optim.Optimizer, warmup: int, total: int):  # noqa: ANN202
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "LayerNorm" in n or n.endswith(".bias") else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_neural_run(cfg: Config) -> dict[str, Any]:
    """Train M1 (``model.kind=sentence``) or M2 (``model.kind=context``)."""
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer

    from lexbrief.models.context_classifier import ContextClassifier
    from lexbrief.models.sentence_classifier import SentenceClassifier

    set_seed(cfg.seed)
    tc, mc = cfg.train, cfg.model
    context = mc.kind == "context"
    run = run_name_for(cfg)
    out_dir = Path(cfg.paths.models_dir) / run
    device = get_device(cfg.device)
    use_bf16 = tc.bf16 and device.type == "cuda"

    train_ex, val_ex = _load_examples(cfg)
    tok = AutoTokenizer.from_pretrained(mc.encoder)
    train_ds = EncodedDataset(train_ex, tok, mc.max_length, context=context)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate,
        num_workers=tc.num_workers,
        generator=gen,
    )

    model_cls = ContextClassifier if context else SentenceClassifier
    model = model_cls(
        mc.encoder,
        num_labels=mc.num_labels,
        dropout=mc.dropout,
        gradient_checkpointing=tc.gradient_checkpointing,
    ).to(device)

    weights = class_weights([e.label for e in train_ex], tc.class_weighting).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optim = torch.optim.AdamW(_param_groups(model, tc.weight_decay), lr=tc.lr)
    steps_per_epoch = math.ceil(len(train_dl) / tc.grad_accum_steps)
    total_steps = steps_per_epoch * tc.epochs
    sched = _linear_warmup(optim, int(tc.warmup_ratio * total_steps), total_steps)
    writer = SummaryWriter(log_dir=str(Path(cfg.paths.runs_dir) / run))
    logger.info(
        "Run %s: %s (%s), %d train batches/epoch, effective batch %d, %d optimizer steps",
        run,
        mc.encoder,
        mc.kind,
        len(train_dl),
        tc.batch_size * tc.grad_accum_steps,
        total_steps,
    )

    best_f1, best_epoch, bad_epochs, step = -1.0, -1, 0, 0
    history: list[dict[str, Any]] = []
    t_start = time.time()
    for epoch in range(1, tc.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        with peak_memory(f"{run} epoch {epoch}") as mem:
            optim.zero_grad(set_to_none=True)
            for i, batch in enumerate(train_dl, start=1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                with autocast_ctx(device, enabled=use_bf16):
                    logits = model(**batch)
                loss = loss_fn(logits.float(), labels) / tc.grad_accum_steps
                loss.backward()
                running += loss.item() * tc.grad_accum_steps
                if i % tc.grad_accum_steps == 0 or i == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    if step % tc.log_every_steps == 0:
                        writer.add_scalar("train/loss", running / i, step)
                        writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            probs = predict_neural(
                model, tok, val_ex, mc.max_length, context, tc.eval_batch_size, device, use_bf16
            )
        val_f1 = _macro_f1([e.label for e in val_ex], list(probs.argmax(axis=1)))
        rec = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_dl)),
            "val_macro_f1": val_f1,
            "peak_vram_gib": mem["peak_gib"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        writer.add_scalar("val/macro_f1", val_f1, epoch)
        writer.add_scalar("train/epoch_loss", rec["train_loss"], epoch)
        writer.add_scalar("gpu/peak_vram_gib", mem["peak_gib"], epoch)
        logger.info(
            "epoch %d | loss %.4f | val macro-F1 %.4f | peak VRAM %.2f GiB | %.0fs",
            epoch,
            rec["train_loss"],
            val_f1,
            mem["peak_gib"],
            rec["seconds"],
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            model.save(out_dir, tokenizer=tok)
            save_config(cfg, out_dir / RUN_CONFIG)
            logger.info("New best (val macro-F1 %.4f) saved to %s", val_f1, out_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= tc.early_stopping_patience:
                logger.info(
                    "Early stopping after epoch %d (patience %d)", epoch, tc.early_stopping_patience
                )
                break
    writer.close()

    summary = {
        "run": run,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "peak_vram_gib": max(h["peak_vram_gib"] for h in history),
        "train_seconds": round(time.time() - t_start, 1),
        "history": history,
        "gpu": vram_report() if device.type == "cuda" else {"available": False},
        "effective_batch": tc.batch_size * tc.grad_accum_steps,
    }
    write_json(summary, out_dir / "train_summary.json")
    del model, optim
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def train(cfg: Config) -> dict[str, Any]:
    """Dispatch on ``model.kind``."""
    if cfg.model.kind == "tfidf":
        return train_tfidf_run(cfg)
    if cfg.model.kind in ("sentence", "context"):
        return train_neural_run(cfg)
    if cfg.model.kind == "hierarchical":
        from lexbrief.training.train_hierarchical import train_hierarchical

        return train_hierarchical(cfg)
    raise ValueError(f"Unknown model.kind {cfg.model.kind!r}")
