# Copyright Xingyu Chen.
# Implements shared Python support for runtime.

"""Torch native loading, compile registration, and optional warm-up runtime."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


def _library_name(stem: str) -> str:
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"{stem}.dylib"
    return f"{stem}.so"


def _candidates(stem: str, environment: str) -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get(environment)
    if override:
        candidates.append(Path(override).expanduser().resolve())
    spec = importlib.util.find_spec("rayd.torch._C")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().with_name(_library_name(stem)))
    return list(dict.fromkeys(candidates))


_STABLE_REQUIRED = (
    "camera_sample_to_world",
    "camera_sample_to_world_backward",
    "camera_world_to_sample",
    "camera_world_to_sample_backward",
    "camera_sample_ray",
    "camera_sample_ray_backward",
    "intersection_valid",
)


def _stable_registered() -> bool:
    return all(hasattr(torch.ops.rayd_torch_stable, name) for name in _STABLE_REQUIRED)


def _load_stable() -> tuple[bool, Exception | None]:
    if _stable_registered():
        return True, None
    first_error: Exception | None = None
    for path in _candidates("_stable_ops", "RAYD_TORCH_STABLE_LIBRARY"):
        if not path.is_file():
            continue
        try:
            torch.ops.load_library(str(path))
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        if _stable_registered():
            return True, None
        if first_error is None:
            first_error = RuntimeError(f"Stable ABI operators were not registered by {path}")
    return False, first_error or FileNotFoundError(
        f"RayD stable ABI library {_library_name('_stable_ops')} was not found"
    )


_STABLE_AVAILABLE, _STABLE_LOAD_ERROR = _load_stable()


def _stable_ops():
    if not _STABLE_AVAILABLE:
        raise RuntimeError(
            f"RayD Torch stable ABI operators are unavailable: {_library_name('_stable_ops')} did not load."
        ) from _STABLE_LOAD_ERROR
    return torch.ops.rayd_torch_stable


def camera_ops():
    return _stable_ops()


def core_ops():
    return _stable_ops()


_LEGACY_REQUIRED = (
    "intersect_forward_t",
    "nearest_edge_forward",
    "visibility_forward",
    "diffraction_accumulation_forward",
)


def _registered() -> bool:
    if not all(hasattr(torch.ops.rayd_torch, name) for name in _LEGACY_REQUIRED):
        return False
    try:
        getattr(torch.classes.rayd_torch, "Scene")
    except RuntimeError:
        return False
    return True


def _load_legacy() -> tuple[bool, Exception | None, Path | None]:
    if _registered():
        return True, None, None
    first_error: Exception | None = None
    for path in _candidates("_legacy_ops", "RAYD_TORCH_LEGACY_LIBRARY"):
        if not path.is_file():
            continue
        try:
            torch.ops.load_library(str(path))
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        if _registered():
            return True, None, path
        if first_error is None:
            first_error = RuntimeError(f"Legacy dispatcher registrations were not provided by {path}")
    return (
        False,
        first_error
        or FileNotFoundError(f"RayD legacy dispatcher library {_library_name('_legacy_ops')} was not found"),
        None,
    )


# Stable loads before legacy; both precede every operation facade.
AVAILABLE, LOAD_ERROR, LOADED_PATH = _load_legacy()


def is_registered() -> bool:
    return _registered()


try:
    from rayd.torch import _C as _extension
except ImportError:
    _extension = None

NATIVE_AVAILABLE = AVAILABLE or is_registered()
C = _extension if NATIVE_AVAILABLE else None
EXTENSION_IMPORT_ERROR = None if NATIVE_AVAILABLE else LOAD_ERROR


def _fake_intersect_forward_tape_h(scene_handle, vertices, ray_o, ray_d, ray_tmax):
    count = ray_o.shape[0]
    return (ray_o.new_empty((count,)), ray_o.new_empty((count,), dtype=torch.int32))


def _fake_intersect_backward_t_h(scene_handle, vertices, ray_o, ray_d, tape_prim_id, grad_t):
    return vertices.new_empty(vertices.shape)


def _setup_context(ctx, inputs, output):
    scene_handle, vertices, ray_o, ray_d, _ray_tmax = inputs
    _t, tape_prim_id = output
    ctx.scene_handle = scene_handle
    ctx.save_for_backward(vertices, ray_o, ray_d, tape_prim_id)


def _backward(ctx, grad_t, _grad_tape_prim_id):
    vertices, ray_o, ray_d, tape_prim_id = ctx.saved_tensors
    grad_vertices = (
        torch.zeros_like(vertices)
        if grad_t is None
        else torch.ops.rayd_torch.intersect_backward_t_h(ctx.scene_handle, vertices, ray_o, ray_d, tape_prim_id, grad_t)
    )
    return None, grad_vertices, None, None, None


def _register_compile_support() -> None:
    torch.library.register_fake("rayd_torch::intersect_forward_tape_h")(_fake_intersect_forward_tape_h)
    torch.library.register_fake("rayd_torch::intersect_backward_t_h")(_fake_intersect_backward_t_h)
    torch.library.register_autograd("rayd_torch::intersect_forward_tape_h", _backward, setup_context=_setup_context)


if NATIVE_AVAILABLE:
    _register_compile_support()


# ---- optional multi-device warm-up ----

"""Private per-device OptiX pipeline warm-up.

The first call of an OptiX-backed op on a device pays that device's module
JIT and pipeline link cost. With one scene replica per device (the multi-GPU
plan's D1) that cost is paid `len(devices)` times, and paying it serially
delays the first real launch by the sum instead of the maximum.

`warm_up_devices` builds a throwaway single-triangle `Scene` per device on a
worker thread and issues one 1-ray call of each requested op, so the
pipelines every replica will need are already linked and cached when the
caller's own scene runs.

One worker thread per device is the intended shape, and the workers run their
device work concurrently: the native layer tolerates concurrent host threads
since the `destroy_scene` lock-order fix of 2026-07-27 (see
`docs/dev/multi_gpu_operations.md`).

This module is private. It is not exported from `rayd.torch`, and Phase 2 is
its only intended caller (from `Scene(devices=...)`).
Importing it does no CUDA work, so it stays importable on a CPU-only machine.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch


# The three OptiX pipelines a replicated scene reaches on its first launches:
# the scene traversal pipeline, the multipath pipeline, and the edge pipeline.
DEFAULT_OPS: tuple[str, ...] = ("intersect", "trace_reflections", "nearest_edge")

_SUPPORTED_OPS: frozenset[str] = frozenset(DEFAULT_OPS)

# A single triangle spanning the origin: the smallest mesh that still gives
# every warm-up ray a real hit, so the closest-hit programs are linked too.
_TRIANGLE_VERTICES = ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))
_TRIANGLE_FACES = ((0, 1, 2),)
_WARM_UP_RAY_O = ((0.0, 0.0, -1.0),)
_WARM_UP_RAY_D = ((0.0, 0.0, 1.0),)
_WARM_UP_POINT = ((0.0, 0.0, 0.5),)


def _normalize_devices(devices: Sequence[int]) -> list[int]:
    """Validate `devices` against the visible CUDA devices, preserving order."""
    if isinstance(devices, (int, str, torch.device)):
        raise TypeError(
            f"warm_up_devices() expects a sequence of device indices; pass [{devices!r}] instead of {devices!r}."
        )

    indices: list[int] = []
    for device in devices:
        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError(f"warm_up_devices() only warms CUDA devices, got {device!r}.")
            index = 0 if device.index is None else device.index
        elif isinstance(device, int) and not isinstance(device, bool):
            index = device
        else:
            raise TypeError(
                f"warm_up_devices() device entries must be int or torch.device, got {type(device).__name__}."
            )
        indices.append(index)

    if not indices:
        return indices

    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise ValueError(
            f"warm_up_devices() received duplicate devices {duplicates}; each device is warmed exactly once."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("warm_up_devices() needs CUDA, but torch.cuda.is_available() is False.")
    count = torch.cuda.device_count()
    for index in indices:
        if index < 0 or index >= count:
            raise ValueError(
                f"warm_up_devices() got device index {index}, but only {count} CUDA device(s) are visible."
            )
    return indices


def _normalize_ops(ops: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ops, str):
        raise TypeError(f"warm_up_devices() expects a sequence of op names; pass ({ops!r},) instead of {ops!r}.")
    names = tuple(ops)
    for name in names:
        if name not in _SUPPORTED_OPS:
            raise ValueError(
                f"warm_up_devices() cannot warm unknown op {name!r}; supported ops are {sorted(_SUPPORTED_OPS)}."
            )
    return names


def _throwaway_scene(device: torch.device):
    """A one-triangle scene owned by this call alone; the caller never sees it.

    Imported lazily so the runtime helper stays independent of the public
    package's import order.
    """
    from .scene import Mesh
    from .scene import Scene

    vertices = torch.tensor(_TRIANGLE_VERTICES, dtype=torch.float32, device=device)
    faces = torch.tensor(_TRIANGLE_FACES, dtype=torch.int32, device=device)
    scene = Scene()
    scene.add_mesh(Mesh(vertices, faces))
    scene.build()
    return scene


def _run_op(scene, name: str, device: torch.device) -> None:
    """Issue one 1-ray call of `name`, touching a result field to force it."""
    from .geometry import Ray

    if name == "intersect":
        ray = Ray(
            torch.tensor(_WARM_UP_RAY_O, dtype=torch.float32, device=device),
            torch.tensor(_WARM_UP_RAY_D, dtype=torch.float32, device=device),
        )
        _ = scene.intersect(ray).t
    elif name == "trace_reflections":
        ray = Ray(
            torch.tensor(_WARM_UP_RAY_O, dtype=torch.float32, device=device),
            torch.tensor(_WARM_UP_RAY_D, dtype=torch.float32, device=device),
        )
        _ = scene.trace_reflections(ray, max_bounces=1).valid
    elif name == "nearest_edge":
        point = torch.tensor(_WARM_UP_POINT, dtype=torch.float32, device=device)
        _ = scene.nearest_edge(point).distance
    else:  # pragma: no cover - _normalize_ops rejects everything else
        raise ValueError(f"unknown warm-up op {name!r}")


def _warm_up_device(index: int, ops: tuple[str, ...]) -> float:
    """Warm one device on the calling (worker) thread; returns its wall time.

    Nothing here serializes against the other devices' workers, so the returned
    times overlap. The current device is thread-local in Torch, so setting it
    here does not disturb the thread that called `warm_up_devices`.
    """
    device = torch.device("cuda", index)
    start = time.perf_counter()
    with torch.cuda.device(index):
        scene = _throwaway_scene(device)
        for name in ops:
            _run_op(scene, name, device)
        # The JIT itself is host-side, but synchronizing makes the reported
        # time cover the launches it enabled rather than just their
        # submission. Only this worker's own stream is waited on: a
        # device-wide sync would reach across into whatever else the caller
        # has running.
        torch.cuda.current_stream(device).synchronize()
        del scene
    return time.perf_counter() - start


def warm_up_devices(devices: Sequence[int], *, ops: Sequence[str] = DEFAULT_OPS) -> dict[int, float]:
    """Pre-link the OptiX pipelines for `ops` on every device in `devices`.

    Returns wall time in seconds per device index, in the order given. Each
    device gets its own worker thread and the workers run concurrently, so the
    returned times overlap and their sum exceeds the call's wall time. An empty
    `devices` is a no-op and touches no CUDA state at all.
    """
    indices = _normalize_devices(devices)
    op_names = _normalize_ops(ops)
    if not indices or not op_names:
        return {}

    # Initialize CUDA once here rather than racing lazy initialization from
    # every worker thread at once.
    torch.cuda.init()

    with ThreadPoolExecutor(max_workers=len(indices), thread_name_prefix="rayd-warmup") as pool:
        futures = {index: pool.submit(_warm_up_device, index, op_names) for index in indices}
        elapsed: dict[int, float] = {}
        failures: list[tuple[int, BaseException]] = []
        for index, future in futures.items():
            try:
                elapsed[index] = future.result()
            except BaseException as error:  # noqa: BLE001 - re-raised below
                failures.append((index, error))

    if failures:
        index, error = failures[0]
        raise RuntimeError(f"RayD warm-up failed on cuda:{index}: {error}") from error
    return elapsed
