import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
TESTS_DIR = os.path.normcase(str(THIS_FILE.parent))
REPO_ROOT = THIS_FILE.parent.parent
CWD = os.path.normcase(os.path.abspath(os.getcwd()))
sys.path = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or CWD)) != TESTS_DIR
]
sys.path.insert(0, str(REPO_ROOT))

import drjit as dr
import drjit.cuda as cuda
import matplotlib.pyplot as plt
import rayd.drjit as rd


def grid_shape(count: int) -> tuple[int, int]:
    width = math.ceil(math.sqrt(count))
    height = math.ceil(count / width)
    return width, height


def make_surfel_grid(count: int, spacing: float) -> rd.SurfelCloud:
    width, height = grid_shape(count)
    idx = dr.arange(cuda.UInt, count)
    ix = idx % width
    iy = idx // width
    x = (cuda.Float(ix) - 0.5 * (width - 1)) * spacing
    y = (cuda.Float(iy) - 0.5 * (height - 1)) * spacing
    zeros = dr.zeros(cuda.Float, count)
    radius = dr.full(cuda.Float, spacing * 0.48, count)
    ones = dr.full(cuda.Float, 1.0, count)
    return rd.SurfelCloud(
        cuda.Array3f(x, y, zeros),
        cuda.Array3f(radius, zeros, zeros),
        cuda.Array3f(zeros, radius, zeros),
        ones,
    )


def make_ray_grid(count: int, extent_x: float, extent_y: float) -> rd.Ray:
    width, height = grid_shape(count)
    idx = dr.arange(cuda.UInt, count)
    ix = idx % width
    iy = idx // width
    x = -extent_x + 2.0 * extent_x * ((cuda.Float(ix) + 0.5) / width)
    y = -extent_y + 2.0 * extent_y * ((cuda.Float(iy) + 0.5) / height)
    zeros = dr.zeros(cuda.Float, count)
    return rd.Ray(
        cuda.Array3f(x, y, dr.full(cuda.Float, 2.0, count)),
        cuda.Array3f(zeros, zeros, dr.full(cuda.Float, -1.0, count)),
    )


def make_options(mode_name: str, single_launch: bool = True) -> rd.SurfelTraceOptions:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.single_launch = single_launch
    if mode_name == "ico":
        opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    elif mode_name == "quad":
        opts.primitive_mode = rd.SurfelPrimitiveMode.QuadTriangles
    else:
        raise ValueError(f"Unknown mode: {mode_name}")
    return opts


def materialize(its: rd.SurfelIntersection) -> None:
    dr.eval(its.t, its.surfel_id, its.triangle_id, its.gaussian_weight)


def summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "avg_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
    }


def trace_once(scene: rd.SurfelScene, rays: rd.Ray, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        materialize(scene.intersect(rays))
        dr.sync_thread()

    samples: list[float] = []
    last = None
    for _ in range(repeats):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            last = scene.intersect(rays)
            materialize(last)
            dr.sync_thread()
            samples.append((time.perf_counter() - start) * 1000.0)

    trace = summarize(samples)
    if last is not None:
        valid = last.is_valid()
        dr.eval(valid)
        trace["valid_count"] = sum(1 for i in range(len(valid)) if bool(valid[i]))
    return trace


def estimate_proxy_bytes(surfel_count: int, mode_name: str) -> dict[str, float]:
    vertices_per_surfel = 12 if mode_name == "ico" else 4
    triangles_per_surfel = 20 if mode_name == "ico" else 2
    vertex_bytes = surfel_count * vertices_per_surfel * 3 * 4
    index_bytes = surfel_count * triangles_per_surfel * 3 * 4
    triangle_to_surfel_bytes = surfel_count * triangles_per_surfel * 4
    cloud_bytes = surfel_count * 10 * 4
    known_bytes = vertex_bytes + index_bytes + triangle_to_surfel_bytes + cloud_bytes
    return {
        "known_proxy_gib": known_bytes / (1024**3),
        "vertex_gib": vertex_bytes / (1024**3),
        "index_gib": index_bytes / (1024**3),
        "triangle_to_surfel_gib": triangle_to_surfel_bytes / (1024**3),
        "cloud_gib": cloud_bytes / (1024**3),
    }


def run_case(
    surfel_count: int,
    ray_count: int,
    mode_name: str,
    spacing: float,
    repeats: int,
    warmup: int,
    single_launch: bool = True,
) -> dict[str, Any]:
    width, height = grid_shape(surfel_count)
    extent_x = max(1.0, width * spacing * 0.55)
    extent_y = max(1.0, height * spacing * 0.55)
    record: dict[str, Any] = {
        "mode": mode_name,
        "surfel_count": surfel_count,
        "ray_count": ray_count,
        "trace_backend": "single_launch" if single_launch else "legacy_retrace",
        "grid": {"width": width, "height": height},
        "estimate": estimate_proxy_bytes(surfel_count, mode_name),
    }

    start = time.perf_counter()
    cloud = make_surfel_grid(surfel_count, spacing)
    dr.sync_thread()
    record["cloud_ms"] = (time.perf_counter() - start) * 1000.0

    opts = make_options(mode_name, single_launch)
    start = time.perf_counter()
    scene = rd.SurfelScene(cloud, opts)
    scene.build()
    dr.sync_thread()
    record["build_ms"] = (time.perf_counter() - start) * 1000.0
    record["triangle_count"] = scene.triangle_count

    start = time.perf_counter()
    rays = make_ray_grid(ray_count, extent_x, extent_y)
    dr.eval(rays.o, rays.d)
    dr.sync_thread()
    record["ray_gen_ms"] = (time.perf_counter() - start) * 1000.0
    record["trace"] = trace_once(scene, rays, repeats, warmup)
    return record


def run_surfel_count(
    surfel_count: int,
    ray_counts: list[int],
    mode_name: str,
    spacing: float,
    repeats: int,
    warmup: int,
    single_launch: bool = True,
) -> list[dict[str, Any]]:
    width, height = grid_shape(surfel_count)
    extent_x = max(1.0, width * spacing * 0.55)
    extent_y = max(1.0, height * spacing * 0.55)
    base: dict[str, Any] = {
        "mode": mode_name,
        "surfel_count": surfel_count,
        "trace_backend": "single_launch" if single_launch else "legacy_retrace",
        "grid": {"width": width, "height": height},
        "estimate": estimate_proxy_bytes(surfel_count, mode_name),
    }

    try:
        start = time.perf_counter()
        cloud = make_surfel_grid(surfel_count, spacing)
        dr.sync_thread()
        base["cloud_ms"] = (time.perf_counter() - start) * 1000.0

        opts = make_options(mode_name, single_launch)
        start = time.perf_counter()
        scene = rd.SurfelScene(cloud, opts)
        scene.build()
        dr.sync_thread()
        base["build_ms"] = (time.perf_counter() - start) * 1000.0
        base["triangle_count"] = scene.triangle_count
    except Exception as exc:  # noqa: BLE001 - benchmark must preserve partial results.
        return [
            {
                **base,
                "ray_count": ray_count,
                "status": "failed",
                "error": repr(exc),
            }
            for ray_count in ray_counts
        ]

    records: list[dict[str, Any]] = []
    for ray_count in ray_counts:
        record = {**base, "ray_count": ray_count}
        try:
            start = time.perf_counter()
            rays = make_ray_grid(ray_count, extent_x, extent_y)
            dr.eval(rays.o, rays.d)
            dr.sync_thread()
            record["ray_gen_ms"] = (time.perf_counter() - start) * 1000.0
            record["trace"] = trace_once(scene, rays, repeats, warmup)
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - benchmark must preserve partial results.
            record["status"] = "failed"
            record["error"] = repr(exc)
        records.append(record)
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    records = [r for r in data["records"] if r.get("status") == "ok"]
    plots: dict[str, str] = {}
    if not records:
        return plots

    for ray_count in sorted({r["ray_count"] for r in records}):
        subset = [r for r in records if r["ray_count"] == ray_count]
        subset.sort(key=lambda r: r["surfel_count"])
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(
            [r["surfel_count"] for r in subset],
            [r["trace"]["avg_ms"] for r in subset],
            marker="o",
        )
        plt.xscale("log", base=10)
        plt.yscale("log")
        plt.xlabel("surfel count")
        plt.ylabel("trace avg time (ms)")
        backend = records[0].get("trace_backend", "single_launch")
        plt.title(f"RayD surfel trace scale at {ray_count:,} rays ({backend})")
        plt.grid(True, which="both", alpha=0.25)
        path = output_dir / f"surfel_trace_scale_ray_{ray_count}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plots[f"trace_surfel_scale_ray_{ray_count}"] = str(path)

    build_subset = {}
    for record in records:
        build_subset.setdefault(record["surfel_count"], record)
    ordered = [build_subset[k] for k in sorted(build_subset)]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(
        [r["surfel_count"] for r in ordered],
        [r["build_ms"] for r in ordered],
        marker="o",
    )
    plt.xscale("log", base=10)
    plt.yscale("log")
    plt.xlabel("surfel count")
    plt.ylabel("build time (ms)")
    plt.title("RayD surfel GAS build scale")
    plt.grid(True, which="both", alpha=0.25)
    path = output_dir / "surfel_build_scale_base10.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots["build_scale"] = str(path)
    return plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-10 RayD surfel scale benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surfel_scale_base10"))
    parser.add_argument("--surfel-counts", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--ray-counts", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--mode", choices=["ico", "quad"], default="ico")
    parser.add_argument("--trace-backend", choices=["single-launch", "legacy-retrace"], default="single-launch")
    parser.add_argument("--spacing", type=float, default=0.08)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "mode": args.mode,
        "trace_backend": args.trace_backend,
        "spacing": args.spacing,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "records": [],
    }
    json_path = args.output_dir / "surfel_scale_base10_results.json"

    for surfel_count in args.surfel_counts:
        result["records"].extend(
            run_surfel_count(
                surfel_count,
                args.ray_counts,
                args.mode,
                args.spacing,
                args.repeats,
                args.warmup,
                args.trace_backend == "single-launch",
            )
        )
        write_json(json_path, result)

    result["plots"] = plot(result, args.output_dir)
    result["json"] = str(json_path)
    write_json(json_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
