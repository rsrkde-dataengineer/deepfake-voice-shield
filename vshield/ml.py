"""Shared helpers for the optional ML backends (torch is imported lazily, never by the demo path)."""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def device() -> str:
    """VSHIELD_DEVICE overrides; otherwise CUDA, then Apple-silicon MPS, then CPU."""
    import torch
    if os.getenv("VSHIELD_DEVICE"):
        return os.environ["VSHIELD_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
