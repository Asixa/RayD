from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch


def _library_name() -> str:
    if sys.platform == "win32":
        return "_stable_ops.dll"
    if sys.platform == "darwin":
        return "_stable_ops.dylib"
    return "_stable_ops.so"


def _candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("RAYD_TORCH_STABLE_LIBRARY")
    if override:
        candidates.append(Path(override).expanduser().resolve())
    candidates.append(Path(__file__).resolve().with_name(_library_name()))
    spec = importlib.util.find_spec("rayd.torch._C")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().with_name(_library_name()))
    return list(dict.fromkeys(candidates))


def _load() -> tuple[bool, Exception | None]:
    required = (
        "camera_sample_to_world",
        "camera_sample_to_world_backward",
        "camera_world_to_sample",
        "camera_world_to_sample_backward",
        "camera_sample_ray",
        "camera_sample_ray_backward",
        "intersection_valid",
    )
    if all(hasattr(torch.ops.rayd_torch_stable, name) for name in required):
        return True, None
    for path in _candidates():
        if not path.is_file():
            continue
        try:
            torch.ops.load_library(str(path))
        except Exception as exc:  # preserve the actual loader error for diagnostics
            return False, exc
        if all(hasattr(torch.ops.rayd_torch_stable, name) for name in required):
            return True, None
        return False, RuntimeError(f"Stable ABI operators were not registered by {path}")
    return False, FileNotFoundError(f"RayD stable ABI library {_library_name()} was not found")


AVAILABLE, LOAD_ERROR = _load()


def camera_ops():
    return torch.ops.rayd_torch_stable if AVAILABLE else torch.ops.rayd_torch


def core_ops():
    return torch.ops.rayd_torch_stable if AVAILABLE else torch.ops.rayd_torch
