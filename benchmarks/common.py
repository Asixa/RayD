# Copyright Xingyu Chen.
# Provides backend-neutral benchmark data, timing, cleanup, and output helpers.

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable


def make_grid_mesh_data(
    resolution: int, x_offset: float = 0.0, z_offset: float = 0.0
) -> dict[str, list[float] | list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for y in range(resolution + 1):
        fy = y / resolution
        for x in range(resolution + 1):
            fx = x / resolution
            xs.append(x_offset + fx)
            ys.append(fy)
            zs.append(z_offset)

    i0: list[int] = []
    i1: list[int] = []
    i2: list[int] = []
    stride = resolution + 1
    for y in range(resolution):
        for x in range(resolution):
            v00 = y * stride + x
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            i0.extend([v00, v00])
            i1.extend([v10, v11])
            i2.extend([v11, v01])
    return {"x": xs, "y": ys, "z": zs, "i0": i0, "i1": i1, "i2": i2}


def summarize_timings(times_s: list[float], query_count: int) -> dict[str, float]:
    average = statistics.fmean(times_s)
    return {"min_ms": min(times_s) * 1000.0, "avg_ms": average * 1000.0, "qps_m": query_count / average / 1.0e6}


def summarize_samples_ms(samples_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples_ms)
    return {
        "samples_ms": samples_ms,
        "min_ms": min(samples_ms),
        "avg_ms": statistics.fmean(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
    }


def to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def diffraction_path_output_bytes(capacity: int) -> int:
    per_path = 1 + 6 * 4 + 4 + 3 * 2 * 4 + 3 * 3 * 4
    return 4 + capacity * per_path


def materialize_reflection_slots(dr: Any, value: Any) -> None:
    times, primitive_ids, valid_count, slot_count, checksum = value
    dr.eval(*times, *primitive_ids, valid_count, slot_count, checksum)


def time_build(fn: Callable[[], Any], sync: Callable[[], None]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    sync()
    return value, (time.perf_counter() - start) * 1000.0


def cleanup_drjit(dr: Any) -> None:
    gc.collect()
    dr.sync_thread()
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()
    dr.sync_thread()


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
