"""``lexbrief prepare-data``: parse -> align -> split -> stats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lexbrief.config import Config
from lexbrief.data import build_parser, inext_parser, stats
from lexbrief.data.download import find_inext_root
from lexbrief.data.splits import make_build_splits, make_folds
from lexbrief.utils.io import write_json, write_jsonl

logger = logging.getLogger(__name__)


def _build_paths(cfg: Config) -> dict[str, Path]:
    base = Path(cfg.data.build_dir)
    paths = {Path(f).stem: base / f for f in cfg.data.build_files}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BUILD files {missing}; run `lexbrief download` first")
    return paths


def prepare_data(cfg: Config) -> dict[str, Any]:
    """Run the full data pipeline and return a summary dict."""
    dcfg = cfg.data
    out_dir = Path(dcfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. BUILD
    paths = _build_paths(cfg)
    test_raw = build_parser.load_raw(paths["test"]) if "test" in paths else []
    test_labelled = bool(test_raw) and build_parser.has_labels(test_raw)
    logger.info("BUILD test.json labelled: %s", test_labelled)
    train = build_parser.parse_file(paths["train"], "train", dcfg.min_sentence_chars)
    dev = build_parser.parse_file(paths["dev"], "dev", dcfg.min_sentence_chars)
    test = (
        build_parser.parse_file(paths["test"], "test", dcfg.min_sentence_chars)
        if test_labelled
        else None
    )
    splits = make_build_splits(train, dev, test, dcfg.val_fraction, cfg.seed)
    for name, docs in splits.items():
        n = write_jsonl(docs, out_dir / f"build_{name}.jsonl")
        logger.info("Wrote %s (%d docs)", out_dir / f"build_{name}.jsonl", n)

    # 2. IN-Ext (parse + align)
    inext_docs = inext_parser.parse_inext(find_inext_root(dcfg), dcfg)
    write_jsonl(inext_docs, out_dir / "inext.jsonl")
    align = inext_parser.alignment_rate(inext_docs)
    level = logging.WARNING if align["overall"] < dcfg.align_warn_below else logging.INFO
    logger.log(
        level,
        "IN-Ext alignment rate: %.1f%% overall (full %.1f%%, segment-wise %.1f%%, "
        "strict exact+fuzzy %.1f%%)%s",
        100 * align["overall"],
        100 * align["full"],
        100 * align["segment_wise"],
        100 * align["strict_exact_fuzzy"],
        f"; below {100 * dcfg.align_warn_below:.0f}% target" if level == logging.WARNING else "",
    )

    # 3. IN-Ext folds
    folds = make_folds([d["doc_id"] for d in inext_docs], dcfg.n_folds, cfg.seed)
    write_json({"seed": cfg.seed, "n_folds": dcfg.n_folds, "folds": folds}, dcfg.folds_file)

    # 4. Stats
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(dcfg.tokenizer)
    token_stats = stats.length_percentiles(stats.token_lengths(splits["train"], tok))
    figure = stats.plot_label_distribution(splits["train"], dcfg.label_dist_figure)
    report = stats.write_report(
        dcfg.stats_md,
        splits,
        inext_docs,
        token_stats,
        dcfg.tokenizer,
        align,
        stats.role_word_shares(inext_docs),
        figure,
        test_labelled,
        dcfg.fuzzy_threshold,
        dcfg.partial_threshold,
    )
    logger.info("Wrote %s and %s", report, figure)
    return {
        "sizes": stats.split_sizes(
            {**{f"build_{k}": v for k, v in splits.items()}, "inext": inext_docs}
        ),
        "alignment": align,
        "token_stats": token_stats,
        "recommended_max_len": stats.recommend_max_len(token_stats.get("p95", 0.0)),
        "test_labelled": test_labelled,
    }
