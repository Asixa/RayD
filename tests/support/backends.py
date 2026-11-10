# Copyright Xingyu Chen.
# Detects which CUDA backends are importable so cross-backend tests can skip cleanly.

from __future__ import annotations

import functools
import importlib
from types import SimpleNamespace


@functools.lru_cache(maxsize=None)
def torch_cuda_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        importlib.import_module("rayd.torch")
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=None)
def drjit_modules() -> SimpleNamespace | None:
    """Return the Dr.Jit backend modules (`pj`, `dr`, `cuda`, `ad`), or None when they cannot be used."""
    try:
        pj = importlib.import_module("rayd.drjit")
        if int(pj.device_count()) < 1:
            return None
        dr = importlib.import_module("dr" + "jit")
        cuda = importlib.import_module("dr" + "jit.cuda")
        ad = importlib.import_module("dr" + "jit.cuda.ad")
    except Exception:
        return None
    return SimpleNamespace(pj=pj, dr=dr, cuda=cuda, ad=ad)


def drjit_available() -> bool:
    return drjit_modules() is not None


def both_backends_available() -> bool:
    """True when CUDA Torch, `rayd.torch` and a CUDA-capable `rayd.drjit` all import in this process."""
    return torch_cuda_available() and drjit_available()


def cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0
