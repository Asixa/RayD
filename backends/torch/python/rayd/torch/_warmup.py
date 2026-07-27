"""Private per-device OptiX pipeline warm-up.

The first call of an OptiX-backed op on a device pays that device's module
JIT and pipeline link cost. With one scene replica per device (the multi-GPU
plan's D1) that cost is paid `len(devices)` times, and paying it serially
delays the first real launch by the sum instead of the maximum.

`warm_up_devices` builds a throwaway single-triangle `Scene` per device on a
worker thread and issues one 1-ray call of each requested op, so the
pipelines every replica will need are already linked and cached when the
caller's own scene runs.

One worker thread per device is the intended shape, but the device work
itself is currently serialized by `_DEVICE_WORK_LOCK` — see that comment. Once
the native layer tolerates concurrent host threads, dropping the lock is the
only change needed here and the per-device JIT starts overlapping.

This module is private. It is not exported from `rayd.torch`, carries no
`.pyi`, and Phase 2 is its only intended caller (from `Scene(devices=...)`).
Importing it does no CUDA work, so it stays importable on a CPU-only machine.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch


# Two host threads issuing RayD Torch ops concurrently deadlock in the current
# native layer: one thread blocks inside the op while the other never sees its
# stream drain. Measured 2026-07-27 on 2x RTX A6000 at roughly one run in three,
# and reproduced with both threads on a *single* device, so it is a host-thread
# defect rather than anything multi-device. Until it is fixed, this helper takes
# the loss and serializes each device's work; a warm-up that intermittently
# hangs the caller is worse than one that does not overlap. Remove this lock —
# and nothing else here — once concurrent host threads are supported.
_DEVICE_WORK_LOCK = threading.Lock()


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
            "warm_up_devices() expects a sequence of device indices; "
            f"pass [{devices!r}] instead of {devices!r}."
        )

    indices: list[int] = []
    for device in devices:
        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError(
                    f"warm_up_devices() only warms CUDA devices, got {device!r}."
                )
            index = 0 if device.index is None else device.index
        elif isinstance(device, int) and not isinstance(device, bool):
            index = device
        else:
            raise TypeError(
                "warm_up_devices() device entries must be int or torch.device, "
                f"got {type(device).__name__}."
            )
        indices.append(index)

    if not indices:
        return indices

    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise ValueError(
            f"warm_up_devices() received duplicate devices {duplicates}; "
            "each device is warmed exactly once."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "warm_up_devices() needs CUDA, but torch.cuda.is_available() is False."
        )
    count = torch.cuda.device_count()
    for index in indices:
        if index < 0 or index >= count:
            raise ValueError(
                f"warm_up_devices() got device index {index}, but only "
                f"{count} CUDA device(s) are visible."
            )
    return indices


def _normalize_ops(ops: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ops, str):
        raise TypeError(
            "warm_up_devices() expects a sequence of op names; "
            f"pass ({ops!r},) instead of {ops!r}."
        )
    names = tuple(ops)
    for name in names:
        if name not in _SUPPORTED_OPS:
            raise ValueError(
                f"warm_up_devices() cannot warm unknown op {name!r}; "
                f"supported ops are {sorted(_SUPPORTED_OPS)}."
            )
    return names


def _throwaway_scene(device: torch.device):
    """A one-triangle scene owned by this call alone; the caller never sees it.

    Imported lazily so `rayd.torch._warmup` stays importable before (and
    independently of) the package's own public modules.
    """
    from .mesh import Mesh
    from .scene import Scene

    vertices = torch.tensor(_TRIANGLE_VERTICES, dtype=torch.float32, device=device)
    faces = torch.tensor(_TRIANGLE_FACES, dtype=torch.int32, device=device)
    scene = Scene()
    scene.add_mesh(Mesh(vertices, faces))
    scene.build()
    return scene


def _run_op(scene, name: str, device: torch.device) -> None:
    """Issue one 1-ray call of `name`, touching a result field to force it."""
    from .types import Ray

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

    The clock starts after `_DEVICE_WORK_LOCK` is held, so the reported time is
    this device's own warm-up and not how long it waited behind another device.
    The current device is thread-local in Torch, so setting it here does not
    disturb the thread that called `warm_up_devices`.
    """
    device = torch.device("cuda", index)
    with _DEVICE_WORK_LOCK:
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


def warm_up_devices(
    devices: Sequence[int],
    *,
    ops: Sequence[str] = DEFAULT_OPS,
) -> dict[int, float]:
    """Pre-link the OptiX pipelines for `ops` on every device in `devices`.

    Returns wall time in seconds per device index, in the order given. Each
    device gets its own worker thread, but the device work is serialized by
    `_DEVICE_WORK_LOCK`, so the returned times do not overlap today. An empty
    `devices` is a no-op and touches no CUDA state at all.
    """
    indices = _normalize_devices(devices)
    op_names = _normalize_ops(ops)
    if not indices or not op_names:
        return {}

    # Initialize CUDA once here rather than racing lazy initialization from
    # every worker thread at once.
    torch.cuda.init()

    with ThreadPoolExecutor(
        max_workers=len(indices), thread_name_prefix="rayd-warmup"
    ) as pool:
        futures = {
            index: pool.submit(_warm_up_device, index, op_names)
            for index in indices
        }
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
