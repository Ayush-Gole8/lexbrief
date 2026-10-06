"""Dataset statistics, ``data_stats.md`` report and label-distribution figure."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lexbrief.labels import (
    BRIEF_SECTION_ORDER,
    COARSE_COLORS,
    COARSE_LABELS,
    FINE_LABELS,
    FINE_TO_COARSE,
    INEXT_SEGMENT_TO_CODE,
    INEXT_TO_COARSE,
)

logger = logging.getLogger(__name__)

Docs = list[dict[str, Any]]
PERCENTILES = (50, 75, 90, 95, 99)


def split_sizes(splits: dict[str, Docs]) -> dict[str, dict[str, int]]:
    """Documents and sentences per split."""
    return {
        name: {"docs": len(docs), "sentences": sum(len(d["sentences"]) for d in docs)}
        for name, docs in splits.items()
    }


def token_lengths(docs: Docs, tokenizer: Any, batch_size: int = 2048) -> np.ndarray:
    """Wordpiece tokens per sentence (excluding [CLS]/[SEP])."""
    texts = [s["text"] for d in docs for s in d["sentences"]]
    lengths: list[int] = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], add_special_tokens=False)["input_ids"]
        lengths.extend(len(x) for x in enc)
    return np.asarray(lengths, dtype=np.int64)


def length_percentiles(lengths: np.ndarray) -> dict[str, float]:
    """Percentiles, mean and max of a length array."""
    if lengths.size == 0:
        return {}
    out = {f"p{p}": float(np.percentile(lengths, p)) for p in PERCENTILES}
    out["mean"] = float(lengths.mean())
    out["max"] = float(lengths.max())
    return out


def recommend_max_len(p95: float, multiple: int = 8) -> int:
    """95th-percentile token length + 2 special tokens, rounded up to ``multiple``."""
    n = int(np.ceil(p95)) + 2
    return int(np.ceil(n / multiple) * multiple)


def label_counts(docs: Docs, key: str) -> Counter[str]:
    """Count ``sentence[key]`` values (None counted as 'UNLABELLED')."""
    return Counter(s.get(key) or "UNLABELLED" for d in docs for s in d["sentences"])


def transition_matrix(docs: Docs, labels: tuple[str, ...] = FINE_LABELS) -> np.ndarray:
    """Row-normalised role -> next-role transition probabilities within documents."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for d in docs:
        seq = [s["role_fine"] for s in d["sentences"] if s.get("role_fine") in idx]
        for a, b in zip(seq, seq[1:], strict=False):
            m[idx[a], idx[b]] += 1
    rows = m.sum(axis=1, keepdims=True)
    return np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)


def role_word_shares(inext_docs: Docs) -> dict[str, dict[str, float]]:
    """Share of summary words per coarse role, from IN-Ext segment-wise A1/A2 summaries.

    Returns ``{"A1": {...}, "A2": {...}, "mean": {...}}``; each maps every brief section to its
    pooled share of summary words (sections absent from the release get 0.0).
    """
    out: dict[str, dict[str, float]] = {}
    for a in ("A1", "A2"):
        words: Counter[str] = Counter()
        for d in inext_docs:
            for folder, sents in d["summaries"][a]["segments"].items():
                coarse = INEXT_TO_COARSE[INEXT_SEGMENT_TO_CODE[folder]]
                words[coarse] += sum(len(s.split()) for s in sents)
        total = sum(words.values()) or 1
        out[a] = {c: words[c] / total for c in BRIEF_SECTION_ORDER}
    out["mean"] = {c: (out["A1"][c] + out["A2"][c]) / 2 for c in BRIEF_SECTION_ORDER}
    return out


def plot_label_distribution(train_docs: Docs, path: str | Path) -> Path:
    """Bar charts of fine (coloured by coarse section) and coarse label shares in BUILD train."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fine = label_counts(train_docs, "role_fine")
    coarse = label_counts(train_docs, "role_coarse")
    n = sum(fine.values()) or 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
    fl = list(FINE_LABELS)
    ax1.bar(
        fl,
        [100 * fine[x] / n for x in fl],
        color=[COARSE_COLORS[FINE_TO_COARSE[x]] for x in fl],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax1.set_ylabel("% of sentences")
    ax1.set_title("BUILD train: fine rhetorical roles (colour = coarse section)")
    ax1.tick_params(axis="x", rotation=60)
    cl = list(COARSE_LABELS)
    ax2.bar(
        cl,
        [100 * coarse[x] / n for x in cl],
        color=[COARSE_COLORS[x] for x in cl],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax2.set_title("Coarse sections")
    ax2.tick_params(axis="x", rotation=60)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _md_table(header: list[str], rows: list[list[Any]]) -> str:
    def fmt(x: Any) -> str:
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def _dist_table(splits: dict[str, Docs], key: str, labels: list[str]) -> str:
    counts = {name: label_counts(docs, key) for name, docs in splits.items()}
    totals = {name: sum(c.values()) or 1 for name, c in counts.items()}
    extra = sorted({k for c in counts.values() for k in c} - set(labels))
    rows = []
    for lab in labels + extra:
        row: list[Any] = [lab]
        for name in splits:
            c = counts[name][lab]
            row.append(f"{c} ({100 * c / totals[name]:.1f}%)")
        rows.append(row)
    return _md_table(["label", *splits.keys()], rows)


def write_report(
    path: str | Path,
    build_splits: dict[str, Docs],
    inext_docs: Docs,
    token_stats: dict[str, float],
    tokenizer_name: str,
    align: dict[str, float],
    shares: dict[str, dict[str, float]],
    figure: Path | None,
    test_labelled: bool,
    fuzzy_threshold: float,
    partial_threshold: float | None,
) -> Path:
    """Write ``data_stats.md``."""
    all_splits = {**{f"build_{k}": v for k, v in build_splits.items()}, "inext": inext_docs}
    sizes = split_sizes(all_splits)
    p95 = token_stats.get("p95", 0.0)
    md: list[str] = ["# LexBrief data statistics", ""]
    md += [
        "Generated by `lexbrief prepare-data`. BUILD = InRhetoricalRoles (sentence-level gold "
        "roles); IN-Ext = 50 judgments with A1/A2 expert extractive summaries.",
        "",
        f"- BUILD `test.json` labelled: **{test_labelled}** -> "
        + (
            "train/dev/test used as train/val/test."
            if test_labelled
            else "train split 90/10 into train/val; `dev.json` used as test."
        ),
        "- IN-Ext roles are **segment_inferred** (no per-sentence gold roles in the release): "
        "a judgment sentence takes the role of the summary segment(s) it aligns to; unaligned "
        "sentences are UNLABELLED.",
        "",
        "## Documents and sentences per split",
        "",
        _md_table(
            ["split", "documents", "sentences", "sent/doc"],
            [
                [k, v["docs"], v["sentences"], round(v["sentences"] / max(v["docs"], 1), 1)]
                for k, v in sizes.items()
            ],
        ),
        "",
        f"## Tokens per sentence ({tokenizer_name}, BUILD train, no special tokens)",
        "",
        _md_table(list(token_stats.keys()), [[round(v, 1) for v in token_stats.values()]]),
        "",
        f"**Recommended `max_length` = {recommend_max_len(p95)}** "
        f"(95th percentile {p95:.0f} tokens + [CLS]/[SEP], rounded up to a multiple of 8).",
        "",
        "## Fine label distribution",
        "",
        _dist_table(all_splits, "role_fine", list(FINE_LABELS)),
        "",
        "## Coarse label distribution",
        "",
        _dist_table(all_splits, "role_coarse", list(COARSE_LABELS)),
        "",
    ]
    if figure is not None:
        md += [f"![label distribution](../figures/{Path(figure).name})", ""]

    tm = transition_matrix(build_splits["train"])
    short = [lab[:8] for lab in FINE_LABELS]
    md += [
        "## Role transition matrix (BUILD train, P(next | current), rows = current)",
        "",
        _md_table(
            ["from \\ to", *short],
            [[FINE_LABELS[i], *[round(float(x), 2) for x in tm[i]]] for i in range(len(tm))],
        ),
        "",
        f"Self-transition (diagonal) mean: {float(np.mean(np.diag(tm))):.2f}. "
        "High diagonal values motivate the document-level BiLSTM-CRF.",
        "",
        "## IN-Ext alignment (summary sentences -> judgment sentences)",
        "",
        f"Method: normalise -> exact -> `fuzz.ratio >= {fuzzy_threshold:g}`"
        + (
            f" -> sub-span `fuzz.partial_ratio >= {partial_threshold:g}`"
            if partial_threshold is not None
            else ""
        ),
        "",
        _md_table(
            ["metric", "value"],
            [
                ["overall alignment rate", f"{100 * align['overall']:.1f}%"],
                ["full summaries", f"{100 * align['full']:.1f}%"],
                ["segment-wise summaries", f"{100 * align['segment_wise']:.1f}%"],
                ["strict (exact + fuzz.ratio only)", f"{100 * align['strict_exact_fuzzy']:.1f}%"],
                ["exact / fuzzy / partial / unmatched",
                 f"{align['n_exact']:.0f} / {align['n_fuzzy']:.0f} / "
                 f"{align['n_partial']:.0f} / {align['n_none']:.0f}"],
            ],
        ),
        "",
    ]
    if align["overall"] < 0.95:
        md += [
            "> **Below the 95% target.** IN-Ext summaries are lightly *edited* extracts "
            "(words dropped, sentences shortened or merged), so some summary sentences have no "
            "close judgment counterpart.",
            "",
        ]
    md += [
        "## Learned role word shares (IN-Ext segment-wise summaries)",
        "",
        "Share of summary words per brief section; these seed the per-role word budget.",
        "",
        _md_table(
            ["section", "A1", "A2", "mean"],
            [[c, shares["A1"][c], shares["A2"][c], shares["mean"][c]] for c in BRIEF_SECTION_ORDER],
        ),
        "",
        "ISSUES and PRECEDENT have no segment folder in the release (issues appear only as a "
        "heading in the full summaries; precedent discussion is part of `analysis`), so their "
        "learned share is 0.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")
    return path
