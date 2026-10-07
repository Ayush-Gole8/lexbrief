"""Cache one vector per sentence from a fine-tuned M1 encoder ([CLS] and mean-pool).

Layout ``<emb_dir>/<run>/``::

    index.json            run, encoder dims, and per split: file, n_docs, n_sentences
    <split>.pt            {"cls": fp16 [N, d], "mean": fp16 [N, d], "labels": int64 [N],
                           "coarse": list[str|None], "doc_ids": list[str],
                           "offsets": int64 [D+1]}  (doc k = rows offsets[k]:offsets[k+1])

Rows follow :func:`lexbrief.training.datasets.flatten` order of ``<processed_dir>/<split>.jsonl``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lexbrief.training.datasets import EncodedDataset, flatten, load_split
from lexbrief.utils.gpu import autocast_ctx, get_device, peak_memory
from lexbrief.utils.io import read_json, write_json

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("build_train", "build_val", "build_test", "inext")
INDEX_FILE = "index.json"


@torch.no_grad()
def encode_split(
    encoder: torch.nn.Module,
    tokenizer: Any,
    docs: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: torch.device,
    bf16: bool = True,
) -> dict[str, Any]:
    """Encode every sentence of ``docs``; returns the tensors stored in ``<split>.pt``."""
    examples = flatten(docs)
    ds = EncodedDataset(examples, tokenizer, max_length, context=False)
    # length-sorted batches minimise padding; results are scattered back to original order
    order = sorted(range(len(ds)), key=lambda i: len(ds.items[i][0]))
    dl = DataLoader(
        [ds[i] for i in order], batch_size=batch_size, collate_fn=ds.collate, num_workers=0
    )
    dim = encoder.config.hidden_size
    cls_out = torch.empty(len(ds), dim, dtype=torch.float16)
    mean_out = torch.empty(len(ds), dim, dtype=torch.float16)
    encoder.eval()
    pos = 0
    for batch in dl:
        mask = batch["attention_mask"].to(device)
        with autocast_ctx(device, enabled=bf16):
            hs = encoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=mask,
                token_type_ids=batch["token_type_ids"].to(device),
            ).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        mean = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        idx = torch.tensor(order[pos : pos + hs.size(0)])
        cls_out[idx] = hs[:, 0].half().cpu()
        mean_out[idx] = mean.half().cpu()
        pos += hs.size(0)

    offsets = [0]
    for d in docs:
        offsets.append(offsets[-1] + len(d["sentences"]))
    return {
        "cls": cls_out,
        "mean": mean_out,
        "labels": torch.tensor([e.label for e in examples], dtype=torch.long),
        "coarse": [e.gold_coarse for e in examples],
        "doc_ids": [d["doc_id"] for d in docs],
        "offsets": torch.tensor(offsets, dtype=torch.long),
    }


def embed_run(
    run: str,
    models_dir: str | Path,
    processed_dir: str | Path,
    emb_dir: str | Path,
    batch_size: int = 64,
    splits: tuple[str, ...] = SPLITS,
    device_name: str = "cuda",
) -> Path:
    """Embed all available splits with the encoder of ``models/<run>``."""
    from transformers import AutoTokenizer

    from lexbrief.config import load_config
    from lexbrief.models.sentence_classifier import ENCODER_DIR, SentenceClassifier
    from lexbrief.training.evaluate_classifier import RUN_CONFIG

    run_dir = Path(models_dir) / run
    run_cfg = load_config(run_dir / RUN_CONFIG)
    if run_cfg.model.kind != "sentence":
        raise ValueError(f"{run} is a {run_cfg.model.kind!r} run; embeddings need an M1 run")
    device = get_device(device_name)
    model = SentenceClassifier.load(run_dir).to(device)
    tok = AutoTokenizer.from_pretrained(run_dir / ENCODER_DIR)
    out_dir = Path(emb_dir) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "run": run,
        "encoder": run_cfg.model.encoder,
        "dim": model.encoder.config.hidden_size,
        "max_length": run_cfg.model.max_length,
        "splits": {},
    }
    for split in splits:
        path = Path(processed_dir) / f"{split}.jsonl"
        if not path.is_file():
            logger.warning("Skipping %s (no %s)", split, path)
            continue
        docs = load_split(processed_dir, split)
        with peak_memory(f"embed {split}"):
            data = encode_split(
                model.encoder, tok, docs, run_cfg.model.max_length, batch_size, device,
                run_cfg.train.bf16,
            )
        torch.save(data, out_dir / f"{split}.pt")
        index["splits"][split] = {
            "file": f"{split}.pt",
            "n_docs": len(docs),
            "n_sentences": int(data["offsets"][-1]),
        }
        logger.info("Embedded %s: %d docs, %d sentences", split, len(docs), data["offsets"][-1])
    write_json(index, out_dir / INDEX_FILE)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def load_split_embeddings(emb_dir: str | Path, run: str, split: str) -> dict[str, Any]:
    """Load ``<emb_dir>/<run>/<split>.pt`` (checked against the index)."""
    base = Path(emb_dir) / run
    index = read_json(base / INDEX_FILE)
    if split not in index["splits"]:
        raise FileNotFoundError(f"No cached embeddings for {split} in {base}; run `lexbrief embed`")
    return torch.load(base / index["splits"][split]["file"], weights_only=True)
