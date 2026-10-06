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
    from lexbrief.config import load_config
    from lexbrief.training.evaluate_classifier import evaluate_run
    from lexbrief.training.train_sentence import train as _train
    from lexbrief.utils.logging import setup_logging

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
def embed() -> None:
    """Cache sentence embeddings for the hierarchical model."""
    _stub(4)


@app.command()
def brief() -> None:
    """Generate an extractive brief for a judgment."""
    _stub(5)


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
