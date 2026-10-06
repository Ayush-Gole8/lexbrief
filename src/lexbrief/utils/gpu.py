"""GPU helpers: device selection, bf16 support, VRAM reporting, peak-memory tracking."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

_GIB = 1024**3


def get_device(prefer: str = "cuda") -> torch.device:
    """Return ``cuda`` if requested and available, else ``cpu``."""
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    if prefer.startswith("cuda"):
        logger.warning("CUDA requested but not available; falling back to CPU")
    return torch.device("cpu")


def bf16_supported() -> bool:
    """True if the current CUDA device supports bfloat16 (Ampere or newer)."""
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def autocast_ctx(
    device: torch.device | str, enabled: bool = True
) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA when supported, otherwise a no-op context."""
    device = torch.device(device)
    if enabled and device.type == "cuda" and bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def vram_report(device: int | None = None) -> dict[str, Any]:
    """Return a dict describing the CUDA device and its memory usage (GiB)."""
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device() if device is None else device
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "index": idx,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_gib": round(props.total_memory / _GIB, 2),
        "allocated_gib": round(torch.cuda.memory_allocated(idx) / _GIB, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(idx) / _GIB, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(idx) / _GIB, 3),
    }


@contextlib.contextmanager
def peak_memory(tag: str = "block") -> Iterator[dict[str, float]]:
    """Track peak CUDA memory allocated inside the ``with`` block.

    Yields a dict that is filled with ``peak_gib`` on exit and logs the result.
    """
    result: dict[str, float] = {"peak_gib": 0.0}
    if not torch.cuda.is_available():
        yield result
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        result["peak_gib"] = round(torch.cuda.max_memory_allocated() / _GIB, 3)
        logger.info("[%s] peak VRAM allocated: %.3f GiB", tag, result["peak_gib"])
