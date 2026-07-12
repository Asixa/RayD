"""Load the Python/LibTorch-ABI-bound RayD dispatcher library."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch


def _library_name() -> str:
    if sys.platform == "win32":
        return "_legacy_ops.dll"
    if sys.platform == "darwin":
        return "_legacy_ops.dylib"
    return "_legacy_ops.so"


def _candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("RAYD_TORCH_LEGACY_LIBRARY")
    if override:
        candidates.append(Path(override).expanduser().resolve())
    candidates.append(Path(__file__).resolve().with_name(_library_name()))
    spec = importlib.util.find_spec("rayd.torch._C")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().with_name(_library_name()))
    return list(dict.fromkeys(candidates))


def _registered() -> bool:
    required = (
        "intersect_forward_t",
        "nearest_edge_forward",
        "visibility_forward",
        "diffraction_accumulation_forward",
    )
    if not all(hasattr(torch.ops.rayd_torch, name) for name in required):
        return False
    try:
        getattr(torch.classes.rayd_torch, "Scene")
    except RuntimeError:
        return False
    return True


def _load() -> tuple[bool, Exception | None, Path | None]:
    if _registered():
        return True, None, None
    first_error: Exception | None = None
    for path in _candidates():
        if not path.is_file():
            continue
        try:
            torch.ops.load_library(str(path))
        except Exception as exc:  # retain the first loader error for diagnostics
            if first_error is None:
                first_error = exc
            continue
        if _registered():
            return True, None, path
        if first_error is None:
            first_error = RuntimeError(
                f"Legacy dispatcher registrations were not provided by {path}"
            )
    if first_error is not None:
        return False, first_error, None
    return False, FileNotFoundError(
        f"RayD legacy dispatcher library {_library_name()} was not found"
    ), None


AVAILABLE, LOAD_ERROR, LOADED_PATH = _load()


def is_registered() -> bool:
    return _registered()
