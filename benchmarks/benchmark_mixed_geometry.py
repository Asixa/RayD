# Copyright Xingyu Chen.
# Benchmarks unified mesh, SDF, and surfel ray queries against equivalent manual composition.

"""Measure mixed-geometry tracing without host reads in the timed region.

Run from the repository root:

    python -m benchmarks.benchmark_mixed_geometry --backend torch --rays 65536 262144
    python -m benchmarks.benchmark_mixed_geometry --backend drjit --rays 65536 262144

Before ``MixedScene`` is available, the script records the manual resident
composition baseline. Afterward it additionally measures the unified query and
the exact mesh-only forwarding path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _summary(samples_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {"samples_ms": samples_ms, "p50_ms": statistics.median(samples_ms), "p95_ms": ordered[p95_index]}


def _attach_gates(result: dict[str, Any]) -> None:
    if "unified_mixed" not in result:
        return
    manual_ms = result["manual_mixed"]["p50_ms"]
    unified_ms = result["unified_mixed"]["p50_ms"]
    native_mesh_ms = result["native_mesh"]["p50_ms"]
    unified_mesh_ms = result["unified_mesh_only"]["p50_ms"]
    mixed_delta = unified_ms - manual_ms
    mesh_delta = unified_mesh_ms - native_mesh_ms
    result["performance_gates"] = {
        "mixed": {
            "ratio": unified_ms / manual_ms,
            "absolute_delta_ms": mixed_delta,
            "passed": unified_ms <= 1.10 * manual_ms or mixed_delta <= 0.1,
            "limit": "ratio <= 1.10 or absolute_delta_ms <= 0.1",
        },
        "mesh_only": {
            "ratio": unified_mesh_ms / native_mesh_ms,
            "absolute_delta_ms": mesh_delta,
            "passed": unified_mesh_ms <= 1.05 * native_mesh_ms or mesh_delta <= 0.01,
            "limit": "ratio <= 1.05 or absolute_delta_ms <= 0.01",
        },
    }


def _measure(
    call: Callable[[], Any],
    materialize: Callable[[Any], None],
    synchronize: Callable[[], None],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        materialize(call())
        synchronize()
    samples_ms = []
    for _ in range(repeat):
        synchronize()
        begin = time.perf_counter()
        value = call()
        materialize(value)
        synchronize()
        samples_ms.append((time.perf_counter() - begin) * 1.0e3)
    return _summary(samples_ms)


def _measure_pair(
    first_call: Callable[[], Any],
    first_materialize: Callable[[Any], None],
    second_call: Callable[[], Any],
    second_materialize: Callable[[Any], None],
    synchronize: Callable[[], None],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def sample(call: Callable[[], Any], materialize: Callable[[Any], None]) -> float:
        synchronize()
        begin = time.perf_counter()
        materialize(call())
        synchronize()
        return (time.perf_counter() - begin) * 1.0e3

    for index in range(warmup):
        if index % 2 == 0:
            first_materialize(first_call())
            second_materialize(second_call())
        else:
            second_materialize(second_call())
            first_materialize(first_call())
        synchronize()
    first_samples = []
    second_samples = []
    for index in range(repeat):
        if index % 2 == 0:
            first_samples.append(sample(first_call, first_materialize))
            second_samples.append(sample(second_call, second_materialize))
        else:
            second_samples.append(sample(second_call, second_materialize))
            first_samples.append(sample(first_call, first_materialize))
    return _summary(first_samples), _summary(second_samples)


def _torch_case(ray_count: int, warmup: int, repeat: int) -> dict[str, Any]:
    import torch
    import rayd.torch as rt

    device = torch.device("cuda")
    vertices = torch.tensor(((-4.0, -4.0, 2.0), (4.0, -4.0, 2.0), (0.0, 4.0, 2.0)), device=device)
    faces = torch.tensor(((0, 1, 2),), dtype=torch.int32, device=device)
    mesh_scene = rt.Scene()
    mesh_scene.add_mesh(rt.Mesh(vertices, faces, edges_enabled=False))
    mesh_scene.build()

    resolution = 32
    axis = torch.linspace(-1.0, 1.0, resolution, dtype=torch.float32, device=device)
    _x, _y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    sdf = rt.SdfGrid(
        z.contiguous(),
        torch.zeros((3,), dtype=torch.float32, device=device),
        torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=device),
        torch.tensor((2.0, 2.0, 2.0), dtype=torch.float32, device=device),
    )
    cloud = rt.SurfelCloud(
        torch.tensor(((0.0, 0.0, -1.0),), dtype=torch.float32, device=device),
        torch.tensor(((2.0, 0.0, 0.0),), dtype=torch.float32, device=device),
        torch.tensor(((0.0, 2.0, 0.0),), dtype=torch.float32, device=device),
        torch.tensor((0.5,), dtype=torch.float32, device=device),
        torch.ones((1,), dtype=torch.float32, device=device),
    )
    surfel = rt.SurfelScene(cloud, rt.SurfelTraceOptions(max_candidate_hits=1))
    surfel.build()
    x = torch.linspace(-0.5, 0.5, ray_count, dtype=torch.float32, device=device)
    origins = torch.stack((x, torch.zeros_like(x), torch.full_like(x, -3.0)), dim=1).contiguous()
    directions = torch.zeros_like(origins)
    directions[:, 2] = 1.0
    ray = rt.Ray(origins, directions.contiguous())

    def manual() -> tuple[Any, Any, Any]:
        return mesh_scene.intersect(ray), sdf.intersect(ray), surfel.intersect(ray)

    def manual_materialize(value: tuple[Any, Any, Any]) -> None:
        _ = tuple(part.t for part in value)

    result = {}
    if hasattr(rt, "MixedScene"):
        mixed = rt.MixedScene()
        mixed.add_mesh(rt.Mesh(vertices, faces, edges_enabled=False))
        mixed.add_sdf(sdf)
        mixed.add_surfel(cloud, rt.SurfelTraceOptions(max_candidate_hits=1))
        mixed.build()
        mesh_only = rt.MixedScene()
        mesh_only.add_mesh(rt.Mesh(vertices, faces, edges_enabled=False))
        mesh_only.build()
        result["manual_mixed"], result["unified_mixed"] = _measure_pair(
            manual,
            manual_materialize,
            lambda: mixed.intersect(ray),
            lambda value: value.t,
            torch.cuda.synchronize,
            warmup,
            repeat,
        )
        result["native_mesh"], result["unified_mesh_only"] = _measure_pair(
            lambda: mesh_scene.intersect(ray),
            lambda value: value.t,
            lambda: mesh_only.intersect(ray),
            lambda value: value.t,
            torch.cuda.synchronize,
            warmup,
            repeat,
        )
        hit = mixed.intersect(ray)
        result["stress_sanity"] = {
            "ray_count": ray_count,
            "all_expected_closest_surfel": bool(torch.all(torch.abs(hit.t - 2.0) <= 1.0e-4).item()),
        }
        _attach_gates(result)
    else:
        result["manual_mixed"] = _measure(manual, manual_materialize, torch.cuda.synchronize, warmup, repeat)
    return result


def _drjit_case(ray_count: int, warmup: int, repeat: int) -> dict[str, Any]:
    import drjit as dr
    import drjit.cuda as cuda
    import rayd.drjit as rt

    vertices = cuda.Array3f(cuda.Float([-4.0, 4.0, 0.0]), cuda.Float([-4.0, -4.0, 4.0]), cuda.Float([2.0, 2.0, 2.0]))
    faces = cuda.Array3i(cuda.Int([0]), cuda.Int([1]), cuda.Int([2]))
    mesh = rt.Mesh(vertices, faces)
    mesh.edges_enabled = False
    mesh_scene = rt.Scene()
    mesh_scene.add_mesh(mesh)
    mesh_scene.build()

    resolution = 32
    values = cuda.Float(
        [
            -1.0 + 2.0 * k / (resolution - 1)
            for _i in range(resolution)
            for _j in range(resolution)
            for k in range(resolution)
        ]
    )
    sdf = rt.SdfGrid(
        values,
        resolution,
        resolution,
        resolution,
        cuda.Array3f(0.0),
        cuda.Float([1.0, 0.0, 0.0, 0.0]),
        cuda.Array3f(2.0),
    )
    cloud = rt.SurfelCloud(
        cuda.Array3f(cuda.Float([0.0]), cuda.Float([0.0]), cuda.Float([-1.0])),
        cuda.Array3f(cuda.Float([2.0]), cuda.Float([0.0]), cuda.Float([0.0])),
        cuda.Array3f(cuda.Float([0.0]), cuda.Float([2.0]), cuda.Float([0.0])),
        cuda.Float([0.5]),
        cuda.Float([1.0]),
    )
    surfel_options = rt.SurfelTraceOptions()
    surfel_options.max_candidate_hits = 1
    surfel = rt.SurfelScene(cloud, surfel_options)
    surfel.build()
    lane = dr.arange(cuda.Float, ray_count)
    denominator = max(ray_count - 1, 1)
    x = -0.5 + lane / denominator
    ray = rt.Ray(cuda.Array3f(x, 0.0, -3.0), cuda.Array3f(0.0, 0.0, 1.0))

    def manual() -> tuple[Any, Any, Any]:
        return mesh_scene.intersect(ray), sdf.intersect(ray), surfel.intersect(ray)

    def materialize(parts: tuple[Any, ...]) -> None:
        dr.eval(*(part.t for part in parts))

    result = {}
    if hasattr(rt, "MixedScene"):
        mixed = rt.MixedScene()
        mixed.add_mesh(mesh)
        mixed.add_sdf(sdf)
        mixed.add_surfel(cloud, surfel_options)
        mixed.build()
        mesh_only = rt.MixedScene()
        mesh_only.add_mesh(mesh)
        mesh_only.build()
        result["manual_mixed"], result["unified_mixed"] = _measure_pair(
            manual,
            materialize,
            lambda: mixed.intersect(ray),
            lambda value: dr.eval(value.t),
            dr.sync_thread,
            warmup,
            repeat,
        )
        result["native_mesh"], result["unified_mesh_only"] = _measure_pair(
            lambda: mesh_scene.intersect(ray),
            lambda value: dr.eval(value.t),
            lambda: mesh_only.intersect(ray),
            lambda value: dr.eval(value.t),
            dr.sync_thread,
            warmup,
            repeat,
        )
        hit = mixed.intersect(ray)
        result["stress_sanity"] = {
            "ray_count": ray_count,
            "all_expected_closest_surfel": bool(dr.all(dr.abs(hit.t - 2.0) <= 1.0e-4)),
        }
        _attach_gates(result)
    else:
        result["manual_mixed"] = _measure(manual, materialize, dr.sync_thread, warmup, repeat)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "drjit", "all"), default="all")
    parser.add_argument("--rays", type=int, nargs="+", default=(65_536, 262_144))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "rayd_mixed_geometry",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cases": {},
    }
    for backend, runner in (("torch", _torch_case), ("drjit", _drjit_case)):
        if args.backend not in (backend, "all"):
            continue
        results["cases"][backend] = {
            str(ray_count): runner(ray_count, args.warmup, args.repeat) for ray_count in args.rays
        }
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
