"""Logging setup: console + timestamped file under outputs/logs/."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(
    name: str = "lexbrief",
    log_dir: str | Path | None = "outputs/logs",
    level: int | str = logging.INFO,
) -> Path | None:
    """Configure the root logger with a console handler and an optional file handler.

    Safe to call more than once: existing handlers installed by this function are replaced.

    Args:
        name: Prefix of the log file name (``<name>_<timestamp>.log``).
        log_dir: Directory for the log file; ``None`` disables file logging.
        level: Logging level.

    Returns:
        Path of the log file, or ``None`` if file logging is disabled.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_lexbrief", False):
            root.removeHandler(h)
            h.close()

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console._lexbrief = True  # type: ignore[attr-defined]
    root.addHandler(console)

    log_path: Path | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{name}_{stamp}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        fh._lexbrief = True  # type: ignore[attr-defined]
        root.addHandler(fh)

    for noisy in ("urllib3", "filelock", "httpx", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (thin wrapper for symmetry)."""
    return logging.getLogger(name)
