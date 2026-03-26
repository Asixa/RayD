import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import platform
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
import drjit.cuda.ad as ad
import rayd as pj

from tests.benchmark_support import _cleanup_drjit
from tests.benchmark_support import _make_grid_mesh_data
from tests.benchmark_support import _make_ray_data
from tests.benchmark_support import _measure
from tests.benchmark_support import _summarize_timings


def _grid_edge_count(resolution: int) -> int:
    return resolution * (3 * resolution + 2)


def _make_scene_mesh(mesh_data: dict[str, list[float] | list[int]]) -> Any:
    return pj.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )


def _make_point_queries(side: int, z: float, x_offset: float = 0.0) -> cuda.Array3f:
    ray_data = _make_ray_data(side, x_offset=x_offset, z_origin=z)
    return cuda.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"])


def _make_downward_rays(
    side: int,
    z_origin: float,
    x_offset: float = 0.0,
    tmax: float | None = None,
    ad_mode: bool = False,
) -> Any:
    ray_data = _make_ray_data(side, x_offset=x_offset, z_origin=z_origin)
    count = len(ray_data["ox"])

    array3_type = ad.Array3f if ad_mode else cuda.Array3f
    float_type = ad.Float if ad_mode else cuda.Float
    ray_type = pj.Ray if ad_mode else pj.RayDetached

    ray = ray_type(
        array3_type(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
        array3_type([0.0] * count, [0.0] * count, [-1.0] * count),
    )
    if tmax is not None:
        ray.tmax = float_type([tmax] * count)
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
    )


def _to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _summarize_latency(times_s: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(times_s) * 1000.0,
        "avg_ms": statistics.fmean(times_s) * 1000.0,
    }


def _measure_commit_updates(
    scene: Any,
    mesh_id: int,
    base_positions: cuda.Array3f,
    updated_positions: cuda.Array3f,
    repeats: int,
    warmup: int,
) -> list[float]:
    def measure_once(use_updated: bool) -> float:
        positions = updated_positions if use_updated else base_positions
        scene.update_mesh_vertices(mesh_id, positions)
        dr.sync_thread()
        start = time.perf_counter()
        scene.commit_updates()
        dr.sync_thread()
        return time.perf_counter() - start

    use_updated = False
    for _ in range(warmup):
        use_updated = not use_updated
        measure_once(use_updated)

    times_s: list[float] = []
    for _ in range(repeats):
        use_updated = not use_updated
        times_s.append(measure_once(use_updated))
    return times_s


def _measure_configure(
    mesh_data: dict[str, list[float] | list[int]],
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    def run() -> None:
        scene = pj.Scene()
        scene.add_mesh(_make_scene_mesh(mesh_data))
        scene.configure()

    return _summarize_latency(_measure(run, repeats, warmup))


def _build_static_scene(mesh_data: dict[str, list[float] | list[int]]) -> Any:
    scene = pj.Scene()
    scene.add_mesh(_make_scene_mesh(mesh_data))
    scene.configure()
    return scene


def _build_dynamic_scene(mesh_data: dict[str, list[float] | list[int]]) -> tuple[Any, int]:
    scene = pj.Scene()
    mesh_id = scene.add_mesh(_make_scene_mesh(mesh_data), dynamic=True)
    scene.configure()
    return scene, mesh_id


def _benchmark_forward_queries(
    scene: Any,
    point_queries: cuda.Array3f,
    finite_rays: Any,
    infinite_rays: Any,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def run_point() -> None:
        result = scene.nearest_edge(point_queries)
        _eval_point_result(result)

    def run_finite_ray() -> None:
        result = scene.nearest_edge(finite_rays)
        _eval_ray_result(result)

    def run_infinite_ray() -> None:
        result = scene.nearest_edge(infinite_rays)
        _eval_ray_result(result)

    point_result = scene.nearest_edge(point_queries)
    _eval_point_result(point_result)
    finite_result = scene.nearest_edge(finite_rays)
    _eval_ray_result(finite_result)
    infinite_result = scene.nearest_edge(infinite_rays)
    _eval_ray_result(infinite_result)
    dr.sync_thread()

    query_count = len(point_queries[0])
    sanity = {
        "point_valid_count": int(_to_scalar(dr.count(point_result.is_valid()))),
        "point_distance_sum": _to_scalar(dr.sum(point_result.distance)),
        "finite_ray_valid_count": int(_to_scalar(dr.count(finite_result.is_valid()))),
        "finite_ray_distance_sum": _to_scalar(dr.sum(finite_result.distance)),
        "finite_ray_t_sum": _to_scalar(dr.sum(finite_result.ray_t)),
        "infinite_ray_valid_count": int(_to_scalar(dr.count(infinite_result.is_valid()))),
        "infinite_ray_distance_sum": _to_scalar(dr.sum(infinite_result.distance)),
        "infinite_ray_t_sum": _to_scalar(dr.sum(infinite_result.ray_t)),
    }

    timings = {
        "point_query": _summarize_timings(_measure(run_point, repeats, warmup), query_count),
        "finite_ray_query": _summarize_timings(_measure(run_finite_ray, repeats, warmup), query_count),
        "infinite_ray_query": _summarize_timings(_measure(run_infinite_ray, repeats, warmup), query_count),
    }
    return timings, sanity


def _benchmark_commit_updates(
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    scene, mesh_id = _build_dynamic_scene(mesh_data)
    base_positions = cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"])
    updated_positions = cuda.Array3f(
        updated_mesh_data["x"],
        updated_mesh_data["y"],
        updated_mesh_data["z"],
    )
    return _summarize_latency(
        _measure_commit_updates(scene, mesh_id, base_positions, updated_positions, repeats, warmup)
    )


def _benchmark_point_gradient(
    mesh_data: dict[str, list[float] | list[int]],
    point_side: int,
    point_z: float,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    mesh = pj.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )
    verts = ad.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"])
    point_data = _make_ray_data(point_side, z_origin=point_z)
    points = ad.Array3f(
        point_data["ox"],
        point_data["oy"],
        point_data["oz"],
    )
    dr.enable_grad(verts)
    dr.enable_grad(points)
    mesh.vertex_positions = verts

    scene = pj.Scene()
    scene.add_mesh(mesh)
    scene.configure()
    query_count = len(points[0])

    def run() -> None:
        dr.set_grad(verts, 0)
        dr.set_grad(points, 0)
        result = scene.nearest_edge(points)
        loss = dr.sum(result.distance)
        dr.backward(loss)
        dr.eval(dr.grad(verts), dr.grad(points))

    return _summarize_timings(_measure(run, repeats, warmup), query_count)


def _benchmark_finite_ray_gradient(
    mesh_data: dict[str, list[float] | list[int]],
    ray_side: int,
    ray_origin_z: float,
    ray_tmax: float,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    scene = _build_static_scene(mesh_data)
    ray_data = _make_ray_data(ray_side, z_origin=ray_origin_z)
    query_count = len(ray_data["ox"])
    origins = ad.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"])
    directions = ad.Array3f([0.0] * query_count, [0.0] * query_count, [-1.0] * query_count)
    dr.enable_grad(origins)
    rays = pj.Ray(origins, directions)
    rays.tmax = ad.Float([ray_tmax] * query_count)

    def run() -> None:
        dr.set_grad(origins, 0)
        result = scene.nearest_edge(rays)
        loss = dr.sum(result.ray_t)
        dr.backward(loss)
        dr.eval(dr.grad(origins))

    return _summarize_timings(_measure(run, repeats, warmup), query_count)


@dataclass(frozen=True)
class EdgeBenchmarkScenario:
    label: str
    mesh_resolution: int
    query_grid_side: int

    def config(
        self,
        point_z: float,
        finite_ray_tmax: float,
        dynamic_x_offset: float,
    ) -> dict[str, Any]:
        return {
            "label": self.label,
            "mesh_resolution": self.mesh_resolution,
            "triangle_count": self.mesh_resolution * self.mesh_resolution * 2,
            "vertex_count": (self.mesh_resolution + 1) * (self.mesh_resolution + 1),
            "edge_count": _grid_edge_count(self.mesh_resolution),
            "query_grid_side": self.query_grid_side,
            "query_count": self.query_grid_side * self.query_grid_side,
            "point_z": point_z,
            "finite_ray_origin_z": 1.0,
            "finite_ray_tmax": finite_ray_tmax,
            "dynamic_x_offset": dynamic_x_offset,
        }


def _default_scenario_label(mesh_resolution: int, query_grid_side: int) -> str:
    return f"{mesh_resolution}x{mesh_resolution} mesh / {query_grid_side}x{query_grid_side} queries"


def _parse_scenario_spec(spec: str) -> EdgeBenchmarkScenario:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if len(parts) == 2:
        mesh_resolution = int(parts[0])
        query_grid_side = int(parts[1])
        label = _default_scenario_label(mesh_resolution, query_grid_side)
    elif len(parts) == 3:
        label = parts[0]
        mesh_resolution = int(parts[1])
        query_grid_side = int(parts[2])
    else:
        raise ValueError(
            "Invalid --scenario spec. Use 'mesh_resolution:query_grid_side' "
            "or 'label:mesh_resolution:query_grid_side'."
        )

    if mesh_resolution <= 0 or query_grid_side <= 0:
        raise ValueError("Scenario dimensions must be positive.")

    return EdgeBenchmarkScenario(label, mesh_resolution, query_grid_side)


def run_edge_benchmark_case(
    scenario: EdgeBenchmarkScenario,
    repeats: int,
    warmup: int,
    dynamic_x_offset: float,
    point_z: float,
    finite_ray_tmax: float,
    include_gradients: bool,
) -> dict[str, Any]:
    base_mesh = _make_grid_mesh_data(scenario.mesh_resolution)
    updated_mesh = _make_grid_mesh_data(scenario.mesh_resolution, x_offset=dynamic_x_offset)
    point_queries = _make_point_queries(scenario.query_grid_side, z=point_z)
    finite_rays = _make_downward_rays(
        scenario.query_grid_side,
        z_origin=1.0,
        tmax=finite_ray_tmax,
    )
    infinite_rays = _make_downward_rays(
        scenario.query_grid_side,
        z_origin=1.0,
        tmax=None,
    )

    results: dict[str, Any] = {
        "config": scenario.config(point_z, finite_ray_tmax, dynamic_x_offset),
        "sanity": {},
        "performance": {},
    }

    results["performance"]["configure"] = _measure_configure(base_mesh, repeats, warmup)
    _cleanup_drjit()

    static_scene = _build_static_scene(base_mesh)
    forward_timings, sanity = _benchmark_forward_queries(
        static_scene,
        point_queries,
        finite_rays,
        infinite_rays,
        repeats,
        warmup,
    )
    results["sanity"] = sanity
    results["performance"].update(forward_timings)
    _cleanup_drjit()

    results["performance"]["commit_updates"] = _benchmark_commit_updates(
        base_mesh,
        updated_mesh,
        repeats,
        warmup,
    )
    _cleanup_drjit()

    if include_gradients:
        results["performance"]["point_gradient"] = _benchmark_point_gradient(
            base_mesh,
            scenario.query_grid_side,
            point_z,
            repeats,
            warmup,
        )
        _cleanup_drjit()
        results["performance"]["finite_ray_gradient"] = _benchmark_finite_ray_gradient(
            base_mesh,
            scenario.query_grid_side,
            1.0,
            finite_ray_tmax,
            repeats,
            warmup,
        )
        _cleanup_drjit()

    return results


def _write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RayDi edge-query performance for configure(), "
            "point/ray nearest-edge queries, and commit_updates()."
        )
    )
    parser.add_argument("--mesh-resolution", type=int, default=192)
    parser.add_argument("--query-grid-side", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dynamic-x-offset", type=float, default=2.0)
    parser.add_argument("--point-z", type=float, default=0.25)
    parser.add_argument("--finite-ray-tmax", type=float, default=2.0)
    parser.add_argument(
        "--include-gradients",
        action="store_true",
        help="Also benchmark point-distance and finite-ray gradient paths.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Scenario spec to run. Use 'mesh_resolution:query_grid_side' "
            "or 'label:mesh_resolution:query_grid_side'. Repeat to benchmark multiple sizes."
        ),
    )
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    scenario_specs = args.scenario or [f"{args.mesh_resolution}:{args.query_grid_side}"]
    scenarios = [_parse_scenario_spec(spec) for spec in scenario_specs]

    payload = {
        "benchmark": "rayd_edge_queries",
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "drjit_version": getattr(dr, "__version__", "unknown"),
        },
        "suite_config": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "dynamic_x_offset": args.dynamic_x_offset,
            "point_z": args.point_z,
            "finite_ray_tmax": args.finite_ray_tmax,
            "include_gradients": args.include_gradients,
        },
        "scenarios": [
            run_edge_benchmark_case(
                scenario,
                repeats=args.repeats,
                warmup=args.warmup,
                dynamic_x_offset=args.dynamic_x_offset,
                point_z=args.point_z,
                finite_ray_tmax=args.finite_ray_tmax,
                include_gradients=args.include_gradients,
            )
            for scenario in scenarios
        ],
    }

    if args.json_output:
        _write_json(args.json_output, payload)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
