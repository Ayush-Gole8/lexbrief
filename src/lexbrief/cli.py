"""LexBrief command-line interface (``lexbrief --help``)."""

from __future__ import annotations

import logging
import platform

import typer

from lexbrief import __version__

app = typer.Typer(
    name="lexbrief",
    help="Rhetorical-role-aware extractive briefs for Indian court judgments.",
    no_args_is_help=True,
    add_completion=False,
)
logger = logging.getLogger("lexbrief.cli")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    """LexBrief CLI."""
    from lexbrief.utils.logging import setup_logging

    setup_logging("cli", log_dir=None, level=logging.DEBUG if verbose else logging.INFO)


@app.command("check-env")
def check_env() -> None:
    """Report Python, torch, CUDA, GPU, VRAM, bf16 and transformers versions."""
    rows: list[tuple[str, str]] = [
        ("lexbrief", __version__),
        ("python", f"{platform.python_version()} ({platform.system()} {platform.release()})"),
    ]
    ok = True
    try:
        import torch

        from lexbrief.utils.gpu import bf16_supported, vram_report

        rows.append(("torch", torch.__version__))
        rows.append(("torch CUDA build", str(torch.version.cuda)))
        cuda = torch.cuda.is_available()
        rows.append(("CUDA available", str(cuda)))
        if cuda:
            rep = vram_report()
            rows.append(("GPU", f"{rep['name']} (compute {rep['capability']})"))
            rows.append(("total VRAM", f"{rep['total_gib']:.2f} GiB"))
            rows.append(("bf16", str(bf16_supported())))
        else:
            ok = False
            rows.append(("GPU", "none"))
            rows.append(("bf16", "False"))
    except ImportError:
        ok = False
        rows.append(("torch", "NOT INSTALLED (run scripts/setup_env.ps1)"))
    try:
        import transformers

        rows.append(("transformers", transformers.__version__))
    except ImportError:
        ok = False
        rows.append(("transformers", "NOT INSTALLED"))

    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        typer.echo(f"{key:<{width}} : {value}")
    if not ok:
        logger.warning("Environment incomplete: GPU training will not work.")
        raise typer.Exit(code=1)


def _stub(prompt: int) -> None:
    typer.echo(f"Not available yet: implemented in Prompt {prompt}.")
    raise typer.Exit(code=0)


DATA_CONFIG = "configs/data.yaml"
_CONFIG_OPT = typer.Option(DATA_CONFIG, "--config", "-c", help="YAML config file.")
_OVERRIDES_ARG = typer.Argument(None, help="Overrides like data.val_fraction=0.2")


def _load(config: str, overrides: list[str] | None):  # noqa: ANN202 - Config
    from lexbrief.config import load_config
    from lexbrief.utils.seed import set_seed

    cfg = load_config(config, overrides or [])
    set_seed(cfg.seed)
    return cfg


@app.command()
def download(
    config: str = _CONFIG_OPT,
    force: bool = typer.Option(False, "--force", help="Re-download even if present."),
    overrides: list[str] = _OVERRIDES_ARG,
) -> None:
    """Download BUILD (Hugging Face) and IN-Ext/IN-Abs (Zenodo) into data/raw/."""
    from lexbrief.data.download import download_all

    cfg = _load(config, overrides)
    if download_all(cfg.data, force=force):
        typer.echo("All raw data downloaded.")
    else:
        typer.echo("Some downloads failed; see the instructions above.")
        raise typer.Exit(code=1)


@app.command("inspect-raw")
def inspect_raw(config: str = _CONFIG_OPT, overrides: list[str] = _OVERRIDES_ARG) -> None:
    """Show the raw data layout, file counts, samples and whether test.json is labelled."""
    from lexbrief.data.download import inspect_raw as _inspect

    cfg = _load(config, overrides)
    for line in _inspect(cfg.data):
        typer.echo(line)


@app.command("prepare-data")
def prepare_data(config: str = _CONFIG_OPT, overrides: list[str] = _OVERRIDES_ARG) -> None:
    """Parse BUILD + IN-Ext, align summaries, split, and write data_stats.md."""
    from lexbrief.data.prepare import prepare_data as _prepare
    from lexbrief.utils.logging import setup_logging

    cfg = _load(config, overrides)
    setup_logging("prepare_data", log_dir=cfg.paths.logs_dir)
    try:
        summary = _prepare(cfg)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(code=1) from None
    width = max(len(k) for k in summary["sizes"])
    for name, v in summary["sizes"].items():
        typer.echo(f"{name:<{width}} : {v['docs']:>5} docs  {v['sentences']:>7} sentences")
    a = summary["alignment"]
    typer.echo(
        f"IN-Ext alignment rate: {100 * a['overall']:.1f}% "
        f"(strict exact+fuzzy {100 * a['strict_exact_fuzzy']:.1f}%)"
    )
    typer.echo(f"Recommended max_length: {summary['recommended_max_len']}")
    typer.echo(f"Report: {cfg.data.stats_md}")


@app.command()
def train(
    config: str = typer.Option(
        ..., "--config", "-c", help="Model YAML, e.g. configs/m1_inlegalbert.yaml"
    ),
    seed: int | None = typer.Option(None, "--seed", help="Override the seed."),
    all_seeds: bool = typer.Option(False, "--all-seeds", help="Train once per train.seeds."),
    max_docs: int | None = typer.Option(None, "--max-docs", help="Smoke mode: first N docs."),
    epochs: int | None = typer.Option(None, "--epochs", help="Override train.epochs."),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip test/IN-Ext evaluation."),
    overrides: list[str] = _OVERRIDES_ARG,
) -> None:
    """Train a sentence-role classifier (M0 / M1 / M2) and evaluate the best checkpoint."""
    from lexbrief.config import load_config, read_config_dict
    from lexbrief.training.evaluate_classifier import evaluate_run
    from lexbrief.training.train_sentence import train as _train
    from lexbrief.utils.logging import setup_logging

    # An empty/placeholder YAML would silently fall back to the defaults (model m1_inlegalbert)
    # and overwrite that run's checkpoint, so training configs must name the model explicitly.
    try:
        model_section = read_config_dict(config).get("model") or {}
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(code=1) from None
    missing = [k for k in ("name", "kind") if not model_section.get(k)]
    if missing:
        typer.echo(f"ERROR: {config} must set model.{' and model.'.join(missing)} explicitly")
        raise typer.Exit(code=1)

    extra = list(overrides or [])
    if max_docs is not None:
        extra.append(f"train.max_docs={max_docs}")
    if epochs is not None:
        extra.append(f"train.epochs={epochs}")
    base = load_config(config, extra)
    seeds = base.train.seeds if all_seeds else [seed if seed is not None else base.seed]
    setup_logging(f"train_{base.model.name}", log_dir=base.paths.logs_dir)
    for s in seeds:
        cfg = load_config(config, [*extra, f"seed={s}"])
        try:
            summary = _train(cfg)
        except FileNotFoundError as e:
            typer.echo(f"ERROR: {e}")
            raise typer.Exit(code=1) from None
        typer.echo(
            f"Trained {summary['run']}: "
            + ", ".join(
                f"{k}={v}" for k, v in summary.items() if k not in ("history", "gpu", "c_scores")
            )
        )
        if not no_eval:
            m = evaluate_run(summary["run"], cfg.paths.models_dir, cfg.paths.results_dir)
            _echo_metrics(m)


def _echo_metrics(m: dict) -> None:
    b = m["build_test"]
    typer.echo(
        f"[{m['run']}] BUILD test: fine macro-F1 {b['fine']['macro_f1']:.4f} | "
        f"weighted-F1 {b['fine']['weighted_f1']:.4f} | "
        f"coarse macro-F1 {b['coarse']['macro_f1']:.4f}"
    )
    if "inext" in m:
        typer.echo(f"[{m['run']}] IN-Ext coarse macro-F1 {m['inext']['coarse']['macro_f1']:.4f}")


@app.command()
def embed(
    run: str = typer.Option(..., "--run", help="Fine-tuned M1 run, e.g. m1_inlegalbert_s42"),
    config: str = typer.Option("configs/m3a_bilstm_crf.yaml", "--config", "-c"),
    overrides: list[str] = _OVERRIDES_ARG,
) -> None:
    """Cache [CLS] and mean-pooled sentence vectors from an M1 encoder for all splits."""
    from lexbrief.models.embed_cache import embed_run
    from lexbrief.utils.logging import setup_logging

    cfg = _load(config, overrides)
    setup_logging(f"embed_{run}", log_dir=cfg.paths.logs_dir)
    try:
        out = embed_run(
            run,
            cfg.paths.models_dir,
            cfg.data.processed_dir,
            cfg.paths.emb_dir,
            batch_size=cfg.train.eval_batch_size,
            device_name=cfg.device,
        )
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(code=1) from None
    typer.echo(f"Embeddings written to {out}")


@app.command()
def brief(
    doc_id: str = typer.Option(..., "--doc-id", help="IN-Ext document id, e.g. 1953_L_1"),
    strategy: str = typer.Option("A3", "--strategy", help="Budget strategy: A1, A2 or A3"),
    roles: str = typer.Option("predicted", "--roles", help="predicted | gold"),
    budget: float = typer.Option(None, "--budget", help="<=1: fraction of doc words; >1: words"),
    compare: str = typer.Option(
        "none", "--compare", help="Also print a baseline: textrank, lexrank, lead, mmr, oracle"
    ),
    no_search: bool = typer.Option(False, "--no-search", help="Skip the weight grid search."),
    config: str = typer.Option("configs/brief.yaml", "--config", "-c"),
    overrides: list[str] = _OVERRIDES_ARG,
) -> None:
    """Print a role-budgeted extractive brief for one IN-Ext judgment.

    A3 shares, position priors and score weights are fitted on the training folds of the
    document's IN-Ext fold, so the document itself is never seen during fitting.
    """
    from lexbrief.brief.selector import brief_words, format_brief
    from lexbrief.pipeline import (
        BriefSystem,
        baseline_indices,
        fold_train_ids,
        load_inext_docs,
        resolve_length,
    )
    from lexbrief.utils.logging import setup_logging

    cfg = _load(config, overrides)
    setup_logging(f"brief_{doc_id}", log_dir=cfg.paths.logs_dir, level=logging.WARNING)
    try:
        docs = {d.doc_id: d for d in load_inext_docs(cfg)}
        train_ids = fold_train_ids(cfg, doc_id)
    except (FileNotFoundError, KeyError) as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(code=1) from None
    if doc_id not in docs:
        typer.echo(f"ERROR: unknown doc id {doc_id!r}")
        raise typer.Exit(code=1)
    doc = docs[doc_id]
    length = resolve_length(doc, budget if budget is not None else cfg.brief.budget_fraction)
    system = BriefSystem(cfg, strategy=strategy, roles=roles).fit(
        [docs[i] for i in train_ids], search_weights=not no_search
    )
    out = system.generate(doc, length)
    w = ", ".join(
        f"{k}={v:g}" for k, v in zip(("cent", "pos", "cue", "conf"), system.weights, strict=True)
    )
    title = (
        f"{doc_id} | {strategy}-{roles} | L={length} words "
        f"({brief_words(out, doc.words)} used of {doc.n_words}) | weights {w}"
    )
    typer.echo(format_brief(out, title))
    if compare != "none":
        idx = baseline_indices(compare, doc, length, cfg)
        used = sum(doc.words[i] for i in idx)
        head = f"{doc_id} | {compare} baseline | L={length} words ({used} used)"
        typer.echo("\n\n" + head + "\n" + "=" * len(head))
        pred_roles, _ = doc.roles("predicted")
        for i in idx:
            typer.echo(f"- [{i}] ({pred_roles[i] or '-'}) {doc.texts[i]}")


@app.command()
def evaluate(
    run: str = typer.Option(..., "--run", help="Run name under models/, e.g. m1_inlegalbert_s42"),
    models_dir: str = typer.Option("models", "--models-dir"),
    results_dir: str = typer.Option("outputs/results", "--results-dir"),
) -> None:
    """Evaluate a trained classifier run on BUILD test and IN-Ext (briefs: Prompt 6)."""
    from lexbrief.training.evaluate_classifier import evaluate_run
    from lexbrief.utils.logging import setup_logging

    setup_logging(f"evaluate_{run}", log_dir="outputs/logs")
    try:
        m = evaluate_run(run, models_dir, results_dir)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(code=1) from None
    _echo_metrics(m)
    typer.echo(f"Wrote {results_dir}/{run}/metrics.json")


@app.command()
def experiments() -> None:
    """Run the full experiment grid and write result tables."""
    _stub(7)


@app.command()
def demo() -> None:
    """Launch the Streamlit demo."""
    _stub(8)


if __name__ == "__main__":
    app()
