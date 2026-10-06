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


@app.command("prepare-data")
def prepare_data() -> None:
    """Download, sentence-split, align and split datasets."""
    _stub(2)


@app.command()
def train() -> None:
    """Train a sentence-role classifier or hierarchical model."""
    _stub(3)


@app.command()
def embed() -> None:
    """Cache sentence embeddings for the hierarchical model."""
    _stub(4)


@app.command()
def brief() -> None:
    """Generate an extractive brief for a judgment."""
    _stub(5)


@app.command()
def evaluate() -> None:
    """Evaluate classifiers and briefs (ROUGE, per-role ROUGE, coverage)."""
    _stub(6)


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
