import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import drjit as dr

_THIS_FILE = Path(__file__).resolve()
_TESTS_DIR = os.path.normcase(str(_THIS_FILE.parent))
_REPO_ROOT = os.path.normcase(str(_THIS_FILE.parent.parent))
_CWD = os.path.normcase(os.path.abspath(os.getcwd()))
sys.path = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or _CWD)) != _TESTS_DIR
]
sys.path.insert(0, str(_THIS_FILE.parent.parent))
import rayd as rd


def make_grid_mesh(resolution: int) -> rd.Mesh:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for y in range(resolution + 1):
        fy = y / resolution
        for x in range(resolution + 1):
            fx = x / resolution
            xs.append(fx)
            ys.append(fy)
            zs.append(0.0)

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

    return rd.Mesh(
        dr.cuda.Array3f(xs, ys, zs),
        dr.cuda.Array3i(i0, i1, i2),
    )


def make_ray_grid(side: int, z_origin: float = -1.0) -> rd.RayDetached:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for iy in range(side):
        for ix in range(side):
            xs.append((ix + 0.5) / side)
            ys.append((iy + 0.5) / side)
            zs.append(z_origin)

    return rd.RayDetached(
        dr.cuda.Array3f(xs, ys, zs),
        dr.cuda.Array3f(
            [0.0] * len(xs),
            [0.0] * len(xs),
            [1.0] * len(xs),
        ),
    )


def make_sync_positions(resolution: int, amplitude: float) -> dr.cuda.Array3f:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for y in range(resolution + 1):
        fy = y / resolution
        for x in range(resolution + 1):
            fx = x / resolution
            xs.append(fx)
            ys.append(fy)
            zs.append(amplitude * math.sin(math.pi * fx) * math.sin(math.pi * fy))
    return dr.cuda.Array3f(xs, ys, zs)


def summarize_kernel_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [int(entry.get("size", 0)) for entry in history]
    execution_ms = [float(entry.get("execution_time", 0.0)) for entry in history]
    codegen_ms = [float(entry.get("codegen_time", 0.0)) for entry in history]
    backend_ms = [float(entry.get("backend_time", 0.0)) for entry in history]

    type_histogram: dict[str, int] = {}
    for entry in history:
        key = str(entry["type"])
        type_histogram[key] = type_histogram.get(key, 0) + 1

    top_by_exec = sorted(
        history,
        key=lambda entry: float(entry.get("execution_time", 0.0)),
        reverse=True,
    )[:8]

    return {
        "jit_kernel_count": len(history),
        "jit_optix_kernel_count": sum(int(entry.get("uses_optix", 0)) for entry in history),
        "jit_cache_hit_count": sum(int(entry.get("cache_hit", 0)) for entry in history),
        "jit_total_launch_size": sum(sizes),
        "jit_size_min": min(sizes) if sizes else 0,
        "jit_size_max": max(sizes) if sizes else 0,
        "jit_exec_ms": sum(execution_ms),
        "jit_codegen_ms": sum(codegen_ms),
        "jit_backend_compile_ms": sum(backend_ms),
        "jit_type_histogram": type_histogram,
        "top_kernels_by_execution_ms": [
            {
                "type": str(entry["type"]),
                "size": int(entry.get("size", 0)),
                "uses_optix": bool(entry.get("uses_optix", 0)),
                "operation_count": int(entry.get("operation_count", 0)),
                "execution_time_ms": float(entry.get("execution_time", 0.0)),
                "codegen_time_ms": float(entry.get("codegen_time", 0.0)),
                "backend_time_ms": float(entry.get("backend_time", 0.0)),
                "cache_hit": bool(entry.get("cache_hit", 0)),
            }
            for entry in top_by_exec
        ],
    }


def profile_stage(name: str, fn, materialize=None) -> tuple[dict[str, Any], Any]:
    dr.flush_malloc_cache()
    dr.kernel_history_clear()
    rd.native_launch_audit_clear()

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            start = time.perf_counter()
            value = fn()
            if materialize is not None:
                materialize(value)
            dr.sync_thread()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

    history = dr.kernel_history()
    summary = summarize_kernel_history(history)
    native_audit = rd.native_launch_audit()
    summary["stage"] = name
    summary["wall_ms"] = elapsed_ms
    native_stage = "trace_reflections" if name.startswith("trace_reflections") else name
    if native_stage in native_audit:
        summary["native"] = native_audit[native_stage]
    return summary, value


def materialize_trace_result(result: rd.ReflectionChainDetached) -> None:
    dr.eval(
        result.bounce_count,
        result.discovery_count,
        result.representative_ray_index,
        result.t,
        result.shape_ids,
        result.prim_ids,
    )


def build_probe_scene(resolution: int) -> tuple[rd.Scene, int]:
    scene = rd.Scene()
    mesh_id = scene.add_mesh(make_grid_mesh(resolution), dynamic=True)
    return scene, mesh_id


def source_audit() -> dict[str, Any]:
    return {
        "build": {
            "explicit_eval_sync_pairs": [],
            "extra_sync_sites": [],
            "native_launch_notes": [
                "Scene::build() also enters OptiX GAS/IAS build in src/scene/scene_optix.cpp.",
                "SceneEdge::build() enters build_edge_bvh_gpu() in src/scene/edge_bvh.cu.",
                "Current build path no longer inserts an explicit Dr.Jit eval/sync fence before the edge BVH build.",
            ],
        },
        "sync": {
            "conditional_eval_sync_pairs": [],
            "conditional_extra_sync_sites": [],
            "native_launch_notes": [
                "Scene::sync() also enters OptiX update build in src/scene/scene_optix.cpp.",
                "SceneEdge::refit() may enter dirty-ancestor kernels in src/scene/edge_bvh.cu.",
                "Current sync path no longer forces a trailing Dr.Jit eval/sync on edge_info_ or refit buffers.",
            ],
        },
        "trace_reflections": {
            "explicit_eval_sync_pairs": [
                "src/scene/scene.cpp:1412",
                "src/scene/scene.cpp:1422",
            ],
            "native_launch_notes": [
                "trace_reflections() performs one native optixLaunch() per call in src/multipath/reflection_trace_host.cpp:218.",
                "With deduplicate=true, reflection_dedup_gpu() adds native CUDA work in src/multipath/reflection_dedup.cu.",
            ],
        },
        "caveat": (
            "Dr.Jit kernel history does not capture raw CUDA launches, cudaMemset/cudaMemcpy, "
            "CUB internals, OptiX optixAccelBuild(), or the native optixLaunch() used by "
            "ReflectionTracePipeline::launch()."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Dr.Jit kernel launches around Scene.build/sync/trace_reflections."
    )
    parser.add_argument("--mesh-resolution", type=int, default=192)
    parser.add_argument("--ray-grid-side", type=int, default=256)
    parser.add_argument("--max-bounces", type=int, default=1)
    parser.add_argument("--sync-amplitude", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene, mesh_id = build_probe_scene(args.mesh_resolution)
    rays = make_ray_grid(args.ray_grid_side)
    updated_positions = make_sync_positions(args.mesh_resolution, args.sync_amplitude)

    build_summary, _ = profile_stage("build", scene.build)

    sync_summary, _ = profile_stage(
        "sync",
        lambda: (
            scene.update_mesh_vertices(mesh_id, updated_positions),
            scene.sync(),
            scene.last_sync_profile,
        ),
    )

    trace_cold_summary, _ = profile_stage(
        "trace_reflections_cold_call_only",
        lambda: scene.trace_reflections(rays, args.max_bounces, True, False),
    )

    trace_hot_summary, _ = profile_stage(
        "trace_reflections_hot_call_only",
        lambda: scene.trace_reflections(rays, args.max_bounces, True, False),
    )

    trace_hot_materialized_summary, trace_result = profile_stage(
        "trace_reflections_hot_materialized",
        lambda: scene.trace_reflections(rays, args.max_bounces, True, False),
        materialize=materialize_trace_result,
    )

    payload = {
        "scenario": {
            "mesh_resolution": args.mesh_resolution,
            "ray_grid_side": args.ray_grid_side,
            "ray_count": args.ray_grid_side * args.ray_grid_side,
            "max_bounces": args.max_bounces,
            "sync_amplitude": args.sync_amplitude,
        },
        "source_audit": source_audit(),
        "runtime": {
            "build": build_summary,
            "sync": sync_summary,
            "trace_reflections_cold_call_only": trace_cold_summary,
            "trace_reflections_hot_call_only": trace_hot_summary,
            "trace_reflections_hot_materialized": trace_hot_materialized_summary,
        },
        "sync_profile": {
            "mesh_update_ms": scene.last_sync_profile.mesh_update_ms,
            "triangle_scatter_ms": scene.last_sync_profile.triangle_scatter_ms,
            "triangle_eval_ms": scene.last_sync_profile.triangle_eval_ms,
            "edge_scatter_ms": scene.last_sync_profile.edge_scatter_ms,
            "edge_refit_ms": scene.last_sync_profile.edge_refit_ms,
            "optix_sync_ms": scene.last_sync_profile.optix_sync_ms,
            "optix_gas_update_ms": scene.last_sync_profile.optix_gas_update_ms,
            "optix_ias_update_ms": scene.last_sync_profile.optix_ias_update_ms,
            "updated_meshes": scene.last_sync_profile.updated_meshes,
            "updated_vertex_meshes": scene.last_sync_profile.updated_vertex_meshes,
            "updated_transform_meshes": scene.last_sync_profile.updated_transform_meshes,
            "updated_edge_meshes": scene.last_sync_profile.updated_edge_meshes,
            "updated_edges": scene.last_sync_profile.updated_edges,
            "total_ms": scene.last_sync_profile.total_ms,
        },
        "trace_result": {
            "ray_count": trace_result.ray_count,
            "max_bounces": trace_result.max_bounces,
        },
    }

    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
