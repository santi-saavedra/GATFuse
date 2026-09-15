"""Device selection, thread limits and seeding."""

import os
import random

import numpy as np
import torch

from .errors import UsageError
from .logging_utils import get_logger

log = get_logger("runtime")

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(preference="auto"):
    """Return the torch device to run on."""
    if preference not in DEVICE_CHOICES:
        raise UsageError(f"--device must be one of: {', '.join(DEVICE_CHOICES)}")

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise UsageError("--device cuda requested but no CUDA device is visible to PyTorch.")
        device = torch.device("cuda")
    elif preference == "mps":
        if not torch.backends.mps.is_available():
            raise UsageError("--device mps requested but Apple MPS is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log.info("Compute device: %s", device)
    return device


def set_torch_threads(threads):
    """Limit intra-op parallelism; None leaves PyTorch's own default in place."""
    if threads is None:
        log.debug("PyTorch intra-op threads: %d (auto)", torch.get_num_threads())
        return
    if threads < 1:
        raise UsageError("--threads must be >= 1")
    torch.set_num_threads(threads)
    log.debug("PyTorch intra-op threads: %d", threads)


def set_seed(seed):
    """Seed every source of randomness GATFuse uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    log.debug("Random seed: %d", seed)
