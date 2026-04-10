import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark nearest_edge throughput and BVH quality for a single "
            "treelet optimize-root cutoff value."
        )
    )
    parser.add_argument("--mesh-resolution", type=int, default=192)
    parser.add_argument("--query-grid-side", type=int, default=256)
    parser.add_argument("--point-z", type=float, default=0.25)
    parser.add_argument("--finite-ray-origin-z", type=float, default=1.0)
    parser.add_argument("--finite-ray-tmax", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


ARGS = _parse_args()
os.environ["RAYD_EDGE_BVH_TREELET_MIN_OPTIMIZE_ROOTS"] = str(ARGS.cutoff)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
THIS_DIR_NORM = os.path.normcase(os.path.abspath(THIS_DIR))
sys.path = [
    entry for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or ".")) != THIS_DIR_NORM
]
if os.path.normcase(os.path.abspath(REPO_ROOT)) not in {
    os.path.normcase(os.path.abspath(entry or "."))
    for entry in sys.path
}:
    sys.path.insert(0, str(REPO_ROOT))

import drjit as dr
import drjit.cuda as cuda
import rayd as pj

from tests.benchmark_support import _cleanup_drjit
from tests.benchmark_support import _make_grid_mesh_data
from tests.benchmark_support import _make_ray_data


def _make_scene_mesh(mesh_data: dict[str, list[float] | list[int]]) -> Any:
    return pj.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )


def _make_point_queries(side: int, z: float) -> cuda.Array3f:
    ray_data = _make_ray_data(side, z_origin=z)
    return cuda.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"])


def _make_downward_rays(
    side: int,
    z_origin: float,
    *,
    tmax: float | None,
) -> Any:
    ray_data = _make_ray_data(side, z_origin=z_origin)
    count = len(ray_data["ox"])
    ray = pj.RayDetached(
        cuda.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
        cuda.Array3f([0.0] * count, [0.0] * count, [-1.0] * count),
    )
    if tmax is not None:
        ray.tmax = cuda.Float([tmax] * count)
    return ray


def _eval_point_result(result: Any) -> None:
    dr.eval(
        result.is_valid(),
        result.shape_id,
        result.edge_id,
        result.distance,
        result.edge_t,
        result.point,
        result.edge_point,
        result.is_boundary,
    )


def _eval_ray_result(result: Any) -> None:
    dr.eval(
        result.is_valid(),
        result.shape_id,
        result.edge_id,
        result.distance,
        result.ray_t,
        result.edge_t,
        result.point,
        result.edge_point,
        result.is_boundary,
    )


def _measure(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
        dr.sync_thread()

    result: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        dr.sync_thread()
        result.append(time.perf_counter() - start)
    return result


def _summarize_query_timings(times_s: list[float], query_count: int) -> dict[str, float]:
    avg_s = statistics.fmean(times_s)
    min_s = min(times_s)
    return {
        "min_ms": min_s * 1000.0,
        "avg_ms": avg_s * 1000.0,
        "qps_m": query_count / avg_s / 1.0e6,
    }


def _to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _bvh_stats_to_json(scene: Any) -> dict[str, Any]:
    stats = scene.edge_bvh_stats()
    return {
        "primitive_count": int(stats.primitive_count),
        "node_count": int(stats.node_count),
        "internal_node_count": int(stats.internal_node_count),
        "leaf_node_count": int(stats.leaf_node_count),
        "max_height": int(stats.max_height),
        "refit_level_count": int(stats.refit_level_count),
        "min_leaf_size": int(stats.min_leaf_size),
        "max_leaf_size": int(stats.max_leaf_size),
        "avg_leaf_size": float(stats.avg_leaf_size),
        "root_surface_area": float(stats.root_surface_area),
        "internal_surface_area_sum": float(stats.internal_surface_area_sum),
        "sibling_overlap_surface_area_sum": float(stats.sibling_overlap_surface_area_sum),
        "sibling_overlap_surface_area_avg": float(stats.sibling_overlap_surface_area_avg),
        "normalized_sibling_overlap": float(stats.normalized_sibling_overlap),
    }


def main() -> int:
    _cleanup_drjit()
    mesh_data = _make_grid_mesh_data(ARGS.mesh_resolution)
    point_queries = _make_point_queries(ARGS.query_grid_side, ARGS.point_z)
    finite_rays = _make_downward_rays(
        ARGS.query_grid_side,
        ARGS.finite_ray_origin_z,
        tmax=ARGS.finite_ray_tmax,
    )
    infinite_rays = _make_downward_rays(
        ARGS.query_grid_side,
        ARGS.finite_ray_origin_z,
        tmax=None,
    )
    query_count = len(point_queries[0])

    scene = pj.Scene()
    scene.add_mesh(_make_scene_mesh(mesh_data))

    build_times = _measure(scene.build, ARGS.repeats, ARGS.warmup)

    point_result = scene.nearest_edge(point_queries)
    finite_result = scene.nearest_edge(finite_rays)
    infinite_result = scene.nearest_edge(infinite_rays)
    _eval_point_result(point_result)
    _eval_ray_result(finite_result)
    _eval_ray_result(infinite_result)
    dr.sync_thread()

    point_times = _measure(
        lambda: _eval_point_result(scene.nearest_edge(point_queries)),
        ARGS.repeats,
        ARGS.warmup,
    )
    finite_ray_times = _measure(
        lambda: _eval_ray_result(scene.nearest_edge(finite_rays)),
        ARGS.repeats,
        ARGS.warmup,
    )
    infinite_ray_times = _measure(
        lambda: _eval_ray_result(scene.nearest_edge(infinite_rays)),
        ARGS.repeats,
        ARGS.warmup,
    )

    payload = {
        "benchmark": "rayd_edge_bvh_cutoff_compare",
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "drjit_version": getattr(dr, "__version__", "unknown"),
        },
        "config": {
            "cutoff": ARGS.cutoff,
            "mesh_resolution": ARGS.mesh_resolution,
            "query_grid_side": ARGS.query_grid_side,
            "query_count": query_count,
            "point_z": ARGS.point_z,
            "finite_ray_origin_z": ARGS.finite_ray_origin_z,
            "finite_ray_tmax": ARGS.finite_ray_tmax,
            "repeats": ARGS.repeats,
            "warmup": ARGS.warmup,
        },
        "build": {
            "min_ms": min(build_times) * 1000.0,
            "avg_ms": statistics.fmean(build_times) * 1000.0,
        },
        "bvh_stats": _bvh_stats_to_json(scene),
        "throughput": {
            "nearest_edge_point": _summarize_query_timings(point_times, query_count),
            "nearest_edge_ray_finite": _summarize_query_timings(finite_ray_times, query_count),
            "nearest_edge_ray_infinite": _summarize_query_timings(infinite_ray_times, query_count),
        },
        "sanity": {
            "point_valid_count": int(_to_scalar(dr.count(point_result.is_valid()))),
            "point_distance_sum": _to_scalar(dr.sum(point_result.distance)),
            "finite_ray_valid_count": int(_to_scalar(dr.count(finite_result.is_valid()))),
            "finite_ray_distance_sum": _to_scalar(dr.sum(finite_result.distance)),
            "finite_ray_t_sum": _to_scalar(dr.sum(finite_result.ray_t)),
            "infinite_ray_valid_count": int(_to_scalar(dr.count(infinite_result.is_valid()))),
            "infinite_ray_distance_sum": _to_scalar(dr.sum(infinite_result.distance)),
            "infinite_ray_t_sum": _to_scalar(dr.sum(infinite_result.ray_t)),
        },
    }

    text = json.dumps(payload, indent=2)
    if ARGS.json_output is not None:
        ARGS.json_output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.json_output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
