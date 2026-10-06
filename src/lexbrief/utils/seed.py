"""Global seeding for reproducible runs."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, torch and CUDA RNGs.

    Args:
        seed: The seed value.
        deterministic: If True, also force deterministic cuDNN kernels (slower).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for lightweight tooling
        logger.debug("torch not installed; seeded python and numpy only")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Seeded all RNGs with %d (deterministic=%s)", seed, deterministic)
