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
        if all(hasattr(torch.ops.rayd_torch_stable, name) for name in required):
            return True, None
        if first_error is None:
            first_error = RuntimeError(
                f"Stable ABI operators were not registered by {path}"
            )
    if first_error is not None:
        return False, first_error
    return False, FileNotFoundError(f"RayD stable ABI library {_library_name()} was not found")


AVAILABLE, LOAD_ERROR = _load()


def _stable_ops():
    # RayD has no legacy-dispatch fallback: a failed stable ABI load is a hard
    # error so a broken build cannot silently run a different code path.
    if not AVAILABLE:
        raise RuntimeError(
            f"RayD Torch stable ABI operators are unavailable: {_library_name()} did not load."
        ) from LOAD_ERROR
    return torch.ops.rayd_torch_stable


def camera_ops():
    return _stable_ops()


def core_ops():
    return _stable_ops()
