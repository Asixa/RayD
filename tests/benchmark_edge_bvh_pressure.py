import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time
from typing import Any


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
from tests.benchmark_support import _measure
from tests.benchmark_support import _summarize_timings


@dataclass(frozen=True)
class MaskScenario:
    name: str
    description: str
    mask: Any
    active_edge_count: int

    @property
    def keep_ratio(self) -> float:
        edge_count = int(dr.width(self.mask))
        return (
            0.0
            if edge_count == 0
            else float(self.active_edge_count) / float(edge_count)
        )


def _shift_mesh_data(
    mesh_data: dict[str, list[float] | list[int]],
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
) -> dict[str, list[float] | list[int]]:
    return {
        "x": [float(value) + x_offset for value in mesh_data["x"]],
        "y": [float(value) + y_offset for value in mesh_data["y"]],
        "z": [float(value) + z_offset for value in mesh_data["z"]],
        "i0": list(mesh_data["i0"]),
        "i1": list(mesh_data["i1"]),
        "i2": list(mesh_data["i2"]),
    }


def _make_scene_mesh(mesh_data: dict[str, list[float] | list[int]]) -> Any:
    return pj.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )


def _build_tiled_scene(
    *,
    mesh_resolution: int,
    tiles_x: int,
    tiles_y: int,
    tile_spacing: float,
    row_z_stride: float,
) -> tuple[Any, dict[str, float]]:
    scene = pj.Scene()
    base_mesh = _make_grid_mesh_data(mesh_resolution)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            mesh_data = _shift_mesh_data(
                base_mesh,
                x_offset=tile_x * tile_spacing,
                y_offset=tile_y * tile_spacing,
                z_offset=tile_y * row_z_stride,
            )
            scene.add_mesh(_make_scene_mesh(mesh_data))

    scene.build()
    bounds = {
        "min_x": 0.0,
        "max_x": (tiles_x - 1) * tile_spacing + 1.0,
        "min_y": 0.0,
        "max_y": (tiles_y - 1) * tile_spacing + 1.0,
        "min_z": 0.0,
        "max_z": max(0.0, (tiles_y - 1) * row_z_stride),
    }
    return scene, bounds


def _make_world_query_grid(
    side: int,
    bounds: dict[str, float],
    z_value: float,
) -> cuda.Array3f:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    span_x = bounds["max_x"] - bounds["min_x"]
    span_y = bounds["max_y"] - bounds["min_y"]
    for iy in range(side):
        fy = (iy + 0.5) / side
        for ix in range(side):
            fx = (ix + 0.5) / side
            xs.append(bounds["min_x"] + fx * span_x)
            ys.append(bounds["min_y"] + fy * span_y)
            zs.append(z_value)
    return cuda.Array3f(xs, ys, zs)


def _make_downward_rays(
    side: int,
    bounds: dict[str, float],
    z_origin: float,
    *,
    tmax: float | None,
) -> Any:
    origins = _make_world_query_grid(side, bounds, z_origin)
    count = len(origins[0])
    ray = pj.RayDetached(
        origins,
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
        result.global_edge_id,
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
        result.global_edge_id,
        result.distance,
        result.ray_t,
        result.edge_t,
        result.point,
        result.edge_point,
        result.is_boundary,
    )


def _to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _to_json_scalar(value: Any) -> float | None:
    scalar = _to_scalar(value)
    return scalar if math.isfinite(scalar) else None


def _edge_count(scene: Any) -> int:
    return int(dr.width(scene.edge_info().global_edge_id))


def _summarize_latency(times_s: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(times_s) * 1000.0,
        "avg_ms": statistics.fmean(times_s) * 1000.0,
    }


def _profile_to_sample(profile: Any) -> dict[str, float | int]:
    return {
        "mesh_update_ms": float(profile.mesh_update_ms),
        "triangle_scatter_ms": float(profile.triangle_scatter_ms),
        "triangle_eval_ms": float(profile.triangle_eval_ms),
        "edge_scatter_ms": float(profile.edge_scatter_ms),
        "edge_refit_ms": float(profile.edge_refit_ms),
        "optix_sync_ms": float(profile.optix_sync_ms),
        "total_ms": float(profile.total_ms),
        "optix_gas_update_ms": float(profile.optix_gas_update_ms),
        "optix_ias_update_ms": float(profile.optix_ias_update_ms),
        "updated_meshes": int(profile.updated_meshes),
        "updated_vertex_meshes": int(profile.updated_vertex_meshes),
        "updated_transform_meshes": int(profile.updated_transform_meshes),
        "updated_edge_meshes": int(profile.updated_edge_meshes),
        "updated_edges": int(profile.updated_edges),
    }


def _summarize_profile_samples(
    samples: list[dict[str, float | int]],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {}
    if not samples:
        return summary

    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key.startswith("updated_"):
            summary[key] = int(round(statistics.fmean(float(value) for value in values)))
        else:
            summary[key] = statistics.fmean(float(value) for value in values)
    return summary


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
        "leaf_size_histogram": [int(value) for value in list(stats.leaf_size_histogram)],
    }


def _build_mask_scenarios(
    scene: Any,
    *,
    tiles_x: int,
    keep_stride: int,
) -> tuple[Any, list[MaskScenario]]:
    edge_count = _edge_count(scene)
    edge_info = scene.edge_info()
    shape_ids = [int(value) for value in list(edge_info.shape_id)]
    boundary_flags = [bool(value) for value in list(edge_info.is_boundary)]

    def make_mask(
        name: str,
        description: str,
        values: list[bool],
    ) -> MaskScenario:
        return MaskScenario(
            name=name,
            description=description,
            mask=cuda.Bool(values),
            active_edge_count=sum(1 for value in values if value),
        )

    full_values = [True] * edge_count
    empty_values = [False] * edge_count
    stride_sparse_values = [(edge_idx % keep_stride) == 0 for edge_idx in range(edge_count)]
    checker_tile_values = [
        (((shape_id % tiles_x) + (shape_id // tiles_x)) % 2) == 0
        for shape_id in shape_ids
    ]
    boundary_only_values = list(boundary_flags)
    interior_only_values = [not value for value in boundary_flags]

    full_mask = cuda.Bool(full_values)
    scenarios = [
        make_mask(
            "full",
            "All edges active. Baseline traversal and post-query gather cost.",
            full_values,
        ),
        make_mask(
            "checker_tiles",
            "Spatially coherent 50% mesh checkerboard mask.",
            checker_tile_values,
        ),
        make_mask(
            "boundary_only",
            "Only boundary edges kept.",
            boundary_only_values,
        ),
        make_mask(
            "interior_only",
            "Only non-boundary edges kept.",
            interior_only_values,
        ),
        make_mask(
            "stride_sparse",
            f"Index-strided sparse mask keeping every {keep_stride}-th edge.",
            stride_sparse_values,
        ),
        make_mask(
            "empty",
            "No edges active. Measures query fixed overhead after a degenerate rebuild.",
            empty_values,
        ),
    ]
    return full_mask, scenarios


def _measure_build(
    *,
    mesh_resolution: int,
    tiles_x: int,
    tiles_y: int,
    tile_spacing: float,
    row_z_stride: float,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    def run() -> None:
        scene, _ = _build_tiled_scene(
            mesh_resolution=mesh_resolution,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            tile_spacing=tile_spacing,
            row_z_stride=row_z_stride,
        )
        dr.eval(scene.edge_info().global_edge_id)

    return _summarize_latency(_measure(run, repeats, warmup))


def _benchmark_queries(
    scene: Any,
    *,
    point_queries: cuda.Array3f,
    finite_rays: Any,
    infinite_rays: Any,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float | int]]:
    point_result = scene.nearest_edge(point_queries)
    finite_result = scene.nearest_edge(finite_rays)
    infinite_result = scene.nearest_edge(infinite_rays)
    _eval_point_result(point_result)
    _eval_ray_result(finite_result)
    _eval_ray_result(infinite_result)
    dr.sync_thread()

    def run_point() -> None:
        result = scene.nearest_edge(point_queries)
        _eval_point_result(result)

    def run_finite_ray() -> None:
        result = scene.nearest_edge(finite_rays)
        _eval_ray_result(result)

    def run_infinite_ray() -> None:
        result = scene.nearest_edge(infinite_rays)
        _eval_ray_result(result)

    query_count = len(point_queries[0])
    timings = {
        "point_query": _summarize_timings(_measure(run_point, repeats, warmup), query_count),
        "finite_ray_query": _summarize_timings(
            _measure(run_finite_ray, repeats, warmup),
            query_count,
        ),
        "infinite_ray_query": _summarize_timings(
            _measure(run_infinite_ray, repeats, warmup),
            query_count,
        ),
    }

    sanity = {
        "query_count": int(query_count),
        "point_valid_count": int(_to_scalar(dr.count(point_result.is_valid()))),
        "finite_ray_valid_count": int(_to_scalar(dr.count(finite_result.is_valid()))),
        "infinite_ray_valid_count": int(_to_scalar(dr.count(infinite_result.is_valid()))),
        "point_distance_sum": _to_json_scalar(dr.sum(point_result.distance)),
        "finite_ray_distance_sum": _to_json_scalar(dr.sum(finite_result.distance)),
        "infinite_ray_distance_sum": _to_json_scalar(dr.sum(infinite_result.distance)),
    }
    return timings, sanity


def _ensure_mask(scene: Any, mask: Any) -> None:
    scene.set_edge_mask(mask)
    scene.sync()
    dr.sync_thread()


def _measure_mask_transition(
    scene: Any,
    *,
    target_mask: Any,
    baseline_mask: Any,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    def run_once() -> tuple[float, dict[str, float | int]]:
        _ensure_mask(scene, baseline_mask)
        start = time.perf_counter()
        scene.set_edge_mask(target_mask)
        scene.sync()
        dr.sync_thread()
        elapsed_s = time.perf_counter() - start
        return elapsed_s, _profile_to_sample(scene.last_sync_profile)

    for _ in range(warmup):
        run_once()

    wall_times_s: list[float] = []
    profile_samples: list[dict[str, float | int]] = []
    for _ in range(repeats):
        elapsed_s, profile_sample = run_once()
        wall_times_s.append(elapsed_s)
        profile_samples.append(profile_sample)

    _ensure_mask(scene, baseline_mask)
    return {
        "wall_ms": _summarize_latency(wall_times_s),
        "profile_avg": _summarize_profile_samples(profile_samples),
    }


def _benchmark_mask_scenario(
    scene: Any,
    *,
    scenario: MaskScenario,
    full_mask: Any,
    point_queries: cuda.Array3f,
    finite_rays: Any,
    infinite_rays: Any,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    sync_to_mask = None
    restore_full = None
    if scenario.name != "full":
        sync_to_mask = _measure_mask_transition(
            scene,
            target_mask=scenario.mask,
            baseline_mask=full_mask,
            repeats=repeats,
            warmup=warmup,
        )

    _ensure_mask(scene, scenario.mask)
    query_timings, sanity = _benchmark_queries(
        scene,
        point_queries=point_queries,
        finite_rays=finite_rays,
        infinite_rays=infinite_rays,
        repeats=repeats,
        warmup=warmup,
    )

    if scenario.name != "full":
        restore_full = _measure_mask_transition(
            scene,
            target_mask=full_mask,
            baseline_mask=scenario.mask,
            repeats=repeats,
            warmup=warmup,
        )
    else:
        _ensure_mask(scene, full_mask)

    return {
        "name": scenario.name,
        "description": scenario.description,
        "active_edge_count": int(scenario.active_edge_count),
        "keep_ratio": float(scenario.keep_ratio),
        "bvh_stats": _bvh_stats_to_json(scene),
        "sync_to_mask": sync_to_mask,
        "restore_full": restore_full,
        "queries": query_timings,
        "sanity": sanity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stress benchmark RayD secondary-edge BVH on a large tiled scene, "
            "including multiple edge-mask rebuild patterns."
        )
    )
    parser.add_argument("--mesh-resolution", type=int, default=64)
    parser.add_argument("--tiles-x", type=int, default=16)
    parser.add_argument("--tiles-y", type=int, default=16)
    parser.add_argument("--tile-spacing", type=float, default=1.25)
    parser.add_argument("--row-z-stride", type=float, default=0.02)
    parser.add_argument("--query-grid-side", type=int, default=256)
    parser.add_argument("--point-z", type=float, default=0.05)
    parser.add_argument("--ray-z-origin", type=float, default=1.5)
    parser.add_argument("--finite-ray-tmax", type=float, default=3.0)
    parser.add_argument("--mask-keep-stride", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    if args.mesh_resolution <= 0:
        raise ValueError("--mesh-resolution must be positive.")
    if args.tiles_x <= 0 or args.tiles_y <= 0:
        raise ValueError("--tiles-x and --tiles-y must be positive.")
    if args.query_grid_side <= 0:
        raise ValueError("--query-grid-side must be positive.")
    if args.mask_keep_stride <= 0:
        raise ValueError("--mask-keep-stride must be positive.")

    _cleanup_drjit()
    scene, bounds = _build_tiled_scene(
        mesh_resolution=args.mesh_resolution,
        tiles_x=args.tiles_x,
        tiles_y=args.tiles_y,
        tile_spacing=args.tile_spacing,
        row_z_stride=args.row_z_stride,
    )
    edge_count = _edge_count(scene)
    point_queries = _make_world_query_grid(args.query_grid_side, bounds, args.point_z)
    finite_rays = _make_downward_rays(
        args.query_grid_side,
        bounds,
        args.ray_z_origin,
        tmax=args.finite_ray_tmax,
    )
    infinite_rays = _make_downward_rays(
        args.query_grid_side,
        bounds,
        args.ray_z_origin,
        tmax=None,
    )
    full_mask, mask_scenarios = _build_mask_scenarios(
        scene,
        tiles_x=args.tiles_x,
        keep_stride=args.mask_keep_stride,
    )
    _ensure_mask(scene, full_mask)

    build_timings = _measure_build(
        mesh_resolution=args.mesh_resolution,
        tiles_x=args.tiles_x,
        tiles_y=args.tiles_y,
        tile_spacing=args.tile_spacing,
        row_z_stride=args.row_z_stride,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    mask_results = [
        _benchmark_mask_scenario(
            scene,
            scenario=scenario,
            full_mask=full_mask,
            point_queries=point_queries,
            finite_rays=finite_rays,
            infinite_rays=infinite_rays,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        for scenario in mask_scenarios
    ]
    _ensure_mask(scene, full_mask)
    full_result = next(result for result in mask_results if result["name"] == "full")

    payload = {
        "scene": {
            "mesh_resolution": int(args.mesh_resolution),
            "tiles_x": int(args.tiles_x),
            "tiles_y": int(args.tiles_y),
            "tile_count": int(args.tiles_x * args.tiles_y),
            "tile_spacing": float(args.tile_spacing),
            "row_z_stride": float(args.row_z_stride),
            "edge_count": int(edge_count),
        },
        "queries": {
            "query_grid_side": int(args.query_grid_side),
            "query_count": int(full_result["sanity"]["query_count"]),
            "point_z": float(args.point_z),
            "ray_z_origin": float(args.ray_z_origin),
            "finite_ray_tmax": float(args.finite_ray_tmax),
            "mask_keep_stride": int(args.mask_keep_stride),
        },
        "build": build_timings,
        "bvh_stats": _bvh_stats_to_json(scene),
        "timings": full_result["queries"],
        "sanity": full_result["sanity"],
        "mask_scenarios": mask_results,
    }

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
