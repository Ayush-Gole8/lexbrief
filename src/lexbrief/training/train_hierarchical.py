"""Train M3a (BiLSTM-CRF over cached M1 sentence embeddings) and its ablations.

Variants are config flags: ``model.use_crf=false`` (M3a-noCRF), ``model.use_position=false``
(M3a-noPosition). Batches are whole documents; evaluation uses Viterbi decoding.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from lexbrief.config import Config, save_config
from lexbrief.labels import FINE_LABELS
from lexbrief.models.embed_cache import load_split_embeddings
from lexbrief.models.hierarchical import HierarchicalTagger
from lexbrief.utils.gpu import get_device, peak_memory
from lexbrief.utils.io import write_json
from lexbrief.utils.seed import set_seed

logger = logging.getLogger(__name__)

MODEL_FILE = "tagger.pt"
RUN_CONFIG = "config.yaml"


def doc_tensors(data: dict[str, Any], pool: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split a cached split into per-document ``(embeddings [n, d], labels [n])``."""
    emb, labels, off = data[pool], data["labels"], data["offsets"].tolist()
    return [(emb[a:b], labels[a:b]) for a, b in zip(off[:-1], off[1:], strict=True) if b > a]


def pad_docs(
    docs: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of documents to ``[B, T, d]`` embeddings, ``[B, T]`` mask and labels."""
    t = max(e.size(0) for e, _ in docs)
    d = docs[0][0].size(1)
    emb = torch.zeros(len(docs), t, d, dtype=torch.float32)
    mask = torch.zeros(len(docs), t, dtype=torch.bool)
    lab = torch.full((len(docs), t), -100, dtype=torch.long)
    for i, (e, y) in enumerate(docs):
        n = e.size(0)
        emb[i, :n] = e.float()
        mask[i, :n] = True
        lab[i, :n] = y
    return emb.to(device), mask.to(device), lab.to(device)


def build_model(cfg: Config) -> HierarchicalTagger:
    m = cfg.model
    return HierarchicalTagger(
        input_dim=m.emb_dim,
        hidden=m.lstm_hidden,
        num_labels=m.num_labels,
        dropout=m.dropout,
        num_layers=m.lstm_layers,
        use_crf=m.use_crf,
        use_position=m.use_position,
    )


@torch.no_grad()
def predict_docs(
    model: HierarchicalTagger,
    docs: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int = 8,
    with_marginals: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi predictions and (optionally) marginals, concatenated over documents in order.

    With ``with_marginals=False`` the returned probability array is empty (used for the
    per-epoch validation, where only the Viterbi tags are needed).
    """
    model.eval()
    preds: list[int] = []
    margs: list[np.ndarray] = []
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        emb, mask, _ = pad_docs(chunk, device)
        paths = model.decode(emb, mask)
        m = model.marginals(emb, mask).cpu().numpy() if with_marginals else None
        for k, (e, _) in enumerate(chunk):
            preds.extend(paths[k])
            if m is not None:
                margs.append(m[k, : e.size(0)])
    probs = np.concatenate(margs) if margs else np.zeros((0, len(FINE_LABELS)), np.float32)
    return probs.astype(np.float32), np.asarray(preds, dtype=np.int64)


def _macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    keep = gold >= 0
    return float(
        f1_score(
            gold[keep],
            pred[keep],
            labels=list(range(len(FINE_LABELS))),
            average="macro",
            zero_division=0,
        )
    )


def train_hierarchical(cfg: Config) -> dict[str, Any]:
    """Train M3a / ablation and save the best (val macro-F1) checkpoint."""
    from torch.utils.tensorboard import SummaryWriter

    from lexbrief.training.train_sentence import run_name_for

    set_seed(cfg.seed)
    mc, tc = cfg.model, cfg.train
    run = run_name_for(cfg)
    out_dir = Path(cfg.paths.models_dir) / run
    device = get_device(cfg.device)
    emb_dir = cfg.paths.emb_dir

    train_docs = doc_tensors(load_split_embeddings(emb_dir, mc.emb_run, "build_train"), mc.emb_pool)
    val_docs = doc_tensors(load_split_embeddings(emb_dir, mc.emb_run, "build_val"), mc.emb_pool)
    if tc.max_docs > 0:
        train_docs, val_docs = train_docs[: tc.max_docs], val_docs[: tc.max_docs]
    val_gold = torch.cat([y for _, y in val_docs]).numpy()
    logger.info(
        "M3 %s: %d train docs, %d val docs, emb=%s/%s, crf=%s, position=%s",
        run,
        len(train_docs),
        len(val_docs),
        mc.emb_run,
        mc.emb_pool,
        mc.use_crf,
        mc.use_position,
    )

    model = build_model(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    writer = SummaryWriter(log_dir=str(Path(cfg.paths.runs_dir) / run))
    gen = torch.Generator().manual_seed(cfg.seed)

    best, best_epoch, bad, history = -1.0, -1, 0, []
    t_start = time.time()
    for epoch in range(1, tc.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train_docs), generator=gen).tolist()
        total = 0.0
        with peak_memory(f"{run} epoch {epoch}") as mem:
            for i in range(0, len(perm), tc.batch_size):
                emb, mask, lab = pad_docs(
                    [train_docs[j] for j in perm[i : i + tc.batch_size]], device
                )
                loss = model.loss(emb, mask, lab)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                optim.step()
                total += loss.item()
            _, val_pred = predict_docs(model, val_docs, device, with_marginals=False)
        val_f1 = _macro_f1(val_gold, val_pred)
        n_batches = max(1, -(-len(perm) // tc.batch_size))
        rec = {
            "epoch": epoch,
            "train_loss": total / n_batches,
            "val_macro_f1": val_f1,
            "peak_vram_gib": mem["peak_gib"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        writer.add_scalar("train/loss", rec["train_loss"], epoch)
        writer.add_scalar("val/macro_f1", val_f1, epoch)
        logger.info(
            "epoch %d | loss %.4f | val macro-F1 %.4f | %.1fs",
            epoch,
            rec["train_loss"],
            val_f1,
            rec["seconds"],
        )
        if val_f1 > best:
            best, best_epoch, bad = val_f1, epoch, 0
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / MODEL_FILE)
            save_config(cfg, out_dir / RUN_CONFIG)
            write_json({"labels": list(FINE_LABELS)}, out_dir / "label_map.json")
        else:
            bad += 1
            if bad >= tc.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (patience %d)", epoch, tc.early_stopping_patience
                )
                break
    writer.close()
    summary = {
        "run": run,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best,
        "peak_vram_gib": max(h["peak_vram_gib"] for h in history),
        "train_seconds": round(time.time() - t_start, 1),
        "history": history,
    }
    write_json(summary, out_dir / "train_summary.json")
    return summary


def load_tagger(run_dir: Path, cfg: Config, device: torch.device) -> HierarchicalTagger:
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / MODEL_FILE, map_location="cpu", weights_only=True))
    return model.to(device).eval()
