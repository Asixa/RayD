"""Run the reproducible BVH-0 edge benchmark matrix.

The public mode orchestrates the one-factor coverage suite declared in
``shared/benchmarks/edge_bvh_matrix.json``. Each case is measured in a fresh
worker process so process-cached BVH strategy settings cannot leak between
cases. Workers return measured samples only; missing GPU capabilities raise a
ContractError instead of emitting placeholder values.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
THIS_DIR_NORM = os.path.normcase(os.path.abspath(THIS_DIR))
sys.path = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or ".")) != THIS_DIR_NORM
]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.performance.edge_bvh_gate import (  # noqa: E402
    ContractError,
    evaluate_gate,
    expected_case_dimensions,
    load_matrix,
    validate_result,
)


MATRIX_PATH = ROOT / "shared" / "benchmarks" / "edge_bvh_matrix.json"
WORKER_PREFIX = "RAYD_EDGE_BVH_WORKER="
COLD_PREFIX = "RAYD_EDGE_BVH_COLD="


def case_id(dimensions: dict[str, Any]) -> str:
    return "-".join(
        str(dimensions[key])
        for key in (
            "edge_count", "query_count", "query_kind", "top_k",
            "update_mode", "mask", "distribution",
        )
    )


def summarize(samples: list[float], unit: str) -> dict[str, Any]:
    if len(samples) < 5:
        raise ContractError("each metric requires at least five measured samples")
    ordered = sorted(samples)
    return {
        "unit": unit,
        "samples": samples,
        "median": statistics.median(samples),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def _run_command(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ContractError(result.stderr.strip() or result.stdout.strip() or "command failed")
    return result.stdout.strip()


def collect_environment() -> dict[str, str]:
    gpu_lines = _run_command([
        "nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader",
    ]).splitlines()
    if not gpu_lines:
        raise ContractError("nvidia-smi returned no GPU")
    gpu_name, compute_capability, driver_version = [part.strip() for part in gpu_lines[0].split(",")]

    optix_header = (ROOT / "backends" / "drjit" / "include" / "rayd" / "optix.h").read_text(encoding="utf-8")
    optix_match = re.search(r"RAYD_OPTIX_TARGET_VERSION\s+([0-9]+)", optix_header)
    if optix_match is None:
        raise ContractError("could not determine the RayD OptiX target version")
    optix_value = int(optix_match.group(1))
    optix_version = f"{optix_value // 10000}.{(optix_value // 100) % 100}.{optix_value % 100}"

    compiler_id, compiler_version = _native_compiler()
    commit = _run_command(["git", "rev-parse", "HEAD"])
    return {
        "gpu_name": gpu_name,
        "gpu_compute_capability": compute_capability,
        "cuda_runtime_version": _cuda_runtime_version(),
        "cuda_driver_version": driver_version,
        "optix_version": optix_version,
        "compiler_id": os.environ.get("RAYD_BENCHMARK_COMPILER_ID", compiler_id),
        "compiler_version": os.environ.get("RAYD_BENCHMARK_COMPILER_VERSION", compiler_version),
        "build_type": os.environ.get("RAYD_BENCHMARK_BUILD_TYPE", "Release"),
        "git_commit": commit,
    }


def _native_compiler() -> tuple[str, str]:
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    build_root = ROOT / "backends" / "drjit" / "build"
    for compiler_file in build_root.glob(f"{tag}-*/CMakeFiles/*/CMakeCXXCompiler.cmake"):
        text = compiler_file.read_text(encoding="utf-8", errors="replace")
        compiler_id = re.search(r'CMAKE_CXX_COMPILER_ID "([^"]+)"', text)
        compiler_version = re.search(r'CMAKE_CXX_COMPILER_VERSION "([^"]+)"', text)
        if compiler_id and compiler_version:
            return compiler_id.group(1), compiler_version.group(1)
    compiler = platform.python_compiler()
    compiler_match = re.search(r"(MSC|GCC|Clang)[^0-9]*([0-9.]+)", compiler)
    return (
        compiler_match.group(1) if compiler_match else compiler.split()[0],
        compiler_match.group(2) if compiler_match else compiler,
    )


def _worker_command(dimensions: dict[str, Any], matrix_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--matrix", str(matrix_path),
        "--worker-case", json.dumps(dimensions, separators=(",", ":")),
    ]


def run_worker(dimensions: dict[str, Any], matrix_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        _worker_command(dimensions, matrix_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(WORKER_PREFIX):
            payload = json.loads(line[len(WORKER_PREFIX):])
            if result.returncode != 0 or "error" in payload:
                raise ContractError(payload.get("error", "worker failed"))
            return payload
    raise ContractError(result.stderr.strip() or "worker returned no machine-readable payload")


def aggregate_case(dimensions: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    performance = {}
    units = {
        "hot_query_ms": "ms", "build_ms": "ms", "refit_ms": "ms",
        "peak_device_memory_bytes": "bytes", "cold_create_ms": "ms",
    }
    for metric, unit in units.items():
        performance[metric] = summarize(
            [float(value) for value in sample["performance"][metric]], unit
        )
    return {
        "case_id": case_id(dimensions),
        "dimensions": dimensions,
        "performance": performance,
        "correctness": sample["correctness"],
        "ad": sample["ad"],
    }


def _load_backend() -> tuple[Any, Any, Any, Any]:
    import drjit as dr
    import drjit.cuda as cuda
    import drjit.cuda.ad as ad
    import rayd.drjit as rayd

    return dr, cuda, ad, rayd


def _component_edge_counts(edge_count: int, component_count: int = 100) -> list[int]:
    if edge_count % 2 != 0 or edge_count < component_count * 3 or component_count % 2 != 0:
        raise ContractError("edge_count must be even and support 100 odd-edge strip components")
    base = edge_count // component_count
    if base % 2 == 0:
        base -= 1
    counts = [base] * component_count
    remaining = edge_count - sum(counts)
    for index in range(remaining // 2):
        counts[index % component_count] += 2
    if sum(counts) != edge_count or any(value % 2 != 1 for value in counts):
        raise ContractError("failed to partition the requested edge count")
    return counts


def _mesh_components(edge_count: int, distribution: str, seed: int) -> list[dict[str, list[Any]]]:
    components = []
    for component, component_edges in enumerate(_component_edge_counts(edge_count)):
        triangle_count = (component_edges - 1) // 2
        vertex_count = triangle_count + 2
        rng = random.Random(seed + component * 104729)
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for vertex in range(vertex_count):
            step = vertex // 2
            side = vertex & 1
            if distribution == "grid":
                xs.append(float(step % 64) / 63.0)
                ys.append(component * 1000.0 + float(step // 64) * 2.0 + float(side))
                zs.append(0.0)
            elif distribution == "random":
                xs.append(rng.random())
                ys.append(component * 1000.0 + rng.random())
                zs.append(rng.random() * 0.01)
            elif distribution == "long_thin":
                xs.append(float(step) / max(triangle_count, 1))
                ys.append(component * 0.01 + float(side) * 1.0e-4)
                zs.append(0.0)
            else:
                raise ContractError(f"unsupported distribution: {distribution}")
        components.append({
            "x": xs,
            "y": ys,
            "z": zs,
            "i0": list(range(triangle_count)),
            "i1": list(range(1, triangle_count + 1)),
            "i2": list(range(2, triangle_count + 2)),
        })
    return components


def _query_data(query_count: int, distribution: str, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed + 99991)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for index in range(query_count):
        component = index % 100
        if distribution == "grid":
            xs.append(float((index * 17) % 64) / 63.0)
            ys.append(component * 1000.0 + float((index // 100) % 8) * 2.0 + 0.5)
            zs.append(0.05)
        elif distribution == "random":
            xs.append(rng.random())
            ys.append(component * 1000.0 + rng.random())
            zs.append(0.05)
        else:
            xs.append(float((index * 17) % 1024) / 1023.0)
            ys.append(component * 0.01 + 5.0e-5)
            zs.append(0.05)
    return {"x": xs, "y": ys, "z": zs}


def _configure_refit(dimensions: dict[str, Any]) -> None:
    mode = dimensions["update_mode"]
    if mode == "full_refit":
        os.environ["RAYD_EDGE_BVH_REFIT_STRATEGY"] = "full"
    elif mode.startswith("dirty_refit_"):
        os.environ["RAYD_EDGE_BVH_REFIT_STRATEGY"] = "dirty_ancestors"
    else:
        os.environ.pop("RAYD_EDGE_BVH_REFIT_STRATEGY", None)


def _build_scene(components: list[dict[str, list[Any]]], dynamic: bool, cuda: Any, rayd: Any) -> tuple[Any, list[int]]:
    scene = rayd.Scene(edge_bvh_backend="drjit")
    mesh_ids = []
    for data in components:
        mesh = rayd.Mesh(
            cuda.Array3f(data["x"], data["y"], data["z"]),
            cuda.Array3i(data["i0"], data["i1"], data["i2"]),
        )
        mesh_ids.append(scene.add_mesh(mesh, dynamic=dynamic))
    scene.build()
    return scene, mesh_ids


def _edge_mask(edge_count: int, name: str, seed: int, cuda: Any) -> Any:
    modulus = 10
    values = [((index * 7 + seed) % modulus) == 0 for index in range(edge_count)]
    if name == "dense":
        values = [not value for value in values]
    return cuda.Bool(values)


def _make_query(dimensions: dict[str, Any], query_data: dict[str, list[float]], cuda: Any, rayd: Any, shift: float = 0.0) -> Any:
    tangent = (0.37, -0.23, 0.11)
    points = cuda.Array3f(
        [value + shift * tangent[0] for value in query_data["x"]],
        [value + shift * tangent[1] for value in query_data["y"]],
        [value + shift * tangent[2] for value in query_data["z"]],
    )
    if dimensions["query_kind"] == "point":
        return points
    count = dimensions["query_count"]
    rays = rayd.Ray(points, cuda.Array3f([0.0] * count, [0.0] * count, [-1.0] * count))
    if dimensions["query_kind"] == "finite_ray":
        rays.tmax = cuda.Float([2.0] * count)
    return rays


def _query(scene: Any, dimensions: dict[str, Any], query: Any) -> Any:
    if dimensions["query_kind"] == "point":
        return scene.nearest_edges(query, dimensions["top_k"])
    return scene.nearest_edge(query)


def _materialize(result: Any, dimensions: dict[str, Any], dr: Any) -> None:
    if dimensions["query_kind"] == "point":
        dr.eval(result.is_valid, result.distances, result.points, result.edge_points, result.global_edge_ids)
    else:
        dr.eval(result.is_valid(), result.distance, result.point, result.edge_point, result.global_edge_id)


def _correctness(result: Any, dimensions: dict[str, Any], dr: Any) -> dict[str, float]:
    if dimensions["query_kind"] == "point":
        valid, distance, point, edge_point = result.is_valid, result.distances, result.points, result.edge_points
    else:
        valid, distance, point, edge_point = result.is_valid(), result.distance, result.point, result.edge_point
    derived = dr.sqrt(dr.squared_norm(point - edge_point))
    absolute = dr.select(valid, dr.abs(derived - distance), 0.0)
    relative = dr.select(valid, absolute / dr.maximum(dr.abs(distance), 1.0e-8), 0.0)
    dr.eval(absolute, relative)
    return {"max_abs_error": _scalar(dr.max(absolute)), "max_rel_error": _scalar(dr.max(relative))}


def _loss(scene: Any, dimensions: dict[str, Any], query: Any, dr: Any) -> Any:
    result = _query(scene, dimensions, query)
    if dimensions["query_kind"] == "point":
        values = dr.select(result.is_valid, result.distances, 0.0)
    else:
        values = dr.select(result.is_valid(), result.distance, 0.0)
    return dr.sum(values) / max(dr.width(values), 1)


def _scalar(value: Any) -> float:
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _ad_errors(scene: Any, dimensions: dict[str, Any], query_data: dict[str, list[float]], dr: Any, cuda: Any, ad: Any, rayd: Any) -> dict[str, float]:
    tangent = (0.37, -0.23, 0.11)
    fixed = _query(scene, dimensions, _make_query(dimensions, query_data, cuda, rayd))
    if dimensions["query_kind"] == "point":
        valid, distance, point, edge_point = fixed.is_valid, fixed.distances, fixed.points, fixed.edge_points
    else:
        valid, distance, point, edge_point = fixed.is_valid(), fixed.distance, fixed.point, fixed.edge_point
    delta = point - edge_point
    directional = dr.dot(delta, cuda.Array3f(*tangent)) / dr.maximum(distance, 1.0e-8)
    reference_values = dr.select(valid & (distance > 1.0e-8), directional, 0.0)
    reference = _scalar(dr.sum(reference_values) / max(dr.width(reference_values), 1))

    def ad_loss() -> tuple[Any, Any]:
        shift = ad.Float([0.0])
        dr.enable_grad(shift)
        points = ad.Array3f(
            ad.Float(query_data["x"]) + shift * tangent[0],
            ad.Float(query_data["y"]) + shift * tangent[1],
            ad.Float(query_data["z"]) + shift * tangent[2],
        )
        if dimensions["query_kind"] == "point":
            query = points
        else:
            count = dimensions["query_count"]
            query = rayd.RayAD(points, ad.Array3f([0.0] * count, [0.0] * count, [-1.0] * count))
            if dimensions["query_kind"] == "finite_ray":
                query.tmax = ad.Float([2.0] * count)
        return shift, _loss(scene, dimensions, query, dr)

    shift, loss = ad_loss()
    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    vjp = _scalar(dr.grad(shift))
    shift, loss = ad_loss()
    dr.set_grad(shift, 1.0)
    dr.forward(shift)
    jvp = _scalar(dr.grad(loss))

    # Mixed absolute/relative tolerance: near zero, the absolute scale is the
    # meaningful one and should not be amplified into an artificial relative failure.
    denominator = max(abs(reference), 1.0)
    return {
        "vjp_max_abs_error": abs(vjp - reference),
        "vjp_max_rel_error": abs(vjp - reference) / denominator,
        "jvp_max_abs_error": abs(jvp - reference),
        "jvp_max_rel_error": abs(jvp - reference) / denominator,
    }


def _cuda_runtime_candidates() -> list[str]:
    if os.name != "nt":
        return [name for name in (ctypes.util.find_library("cudart"), "libcudart.so.12", "libcudart.so") if name]
    candidates: list[Path] = []
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(variable)
        if root:
            candidates.extend(Path(root).glob("bin/cudart64_*.dll"))
    candidates.extend((Path(sys.prefix) / "Library" / "bin").glob("cudart64_*.dll"))
    candidates.extend(
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*/bin/cudart64_*.dll")
    )
    names = [os.fspath(path) for path in sorted(set(candidates), reverse=True)]
    names.append("cudart64_12.dll")
    return names


def _cuda_mem_free() -> int:
    for name in _cuda_runtime_candidates():
        try:
            runtime = ctypes.CDLL(name)
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            if runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
                return int(free.value)
        except OSError:
            continue
    raise ContractError("cudaMemGetInfo is unavailable; refusing to fabricate memory data")


def _cuda_runtime_version() -> str:
    for name in _cuda_runtime_candidates():
        try:
            runtime = ctypes.CDLL(name)
            version = ctypes.c_int()
            if runtime.cudaRuntimeGetVersion(ctypes.byref(version)) == 0:
                value = int(version.value)
                return f"{value // 1000}.{(value % 1000) // 10}"
        except OSError:
            continue
    raise ContractError("cudaRuntimeGetVersion is unavailable")


def _apply_mask(scene: Any, dimensions: dict[str, Any], matrix: dict[str, Any], cuda: Any, dr: Any) -> None:
    scene.set_edge_mask(_edge_mask(dimensions["edge_count"], dimensions["mask"], matrix["seed"], cuda))
    scene.sync()
    dr.sync_thread()


def _measure_refit(scene: Any, mesh_ids: list[int], components: list[dict[str, list[Any]]], dimensions: dict[str, Any], cuda: Any, dr: Any) -> float:
    mode = dimensions["update_mode"]
    fraction = {
        "static": 0.0, "full_refit": 1.0, "dirty_refit_1pct": 0.01,
        "dirty_refit_10pct": 0.10, "dirty_refit_100pct": 1.0,
    }[mode]
    selected = int(round(len(mesh_ids) * fraction))
    for index in range(selected):
        data = components[index]
        scene.update_mesh_vertices(
            mesh_ids[index],
            cuda.Array3f(data["x"], data["y"], [value + 0.001 for value in data["z"]]),
        )
    start = time.perf_counter()
    scene.sync()
    dr.sync_thread()
    return (time.perf_counter() - start) * 1000.0


def _cold_once(dimensions: dict[str, Any], matrix: dict[str, Any]) -> dict[str, float]:
    _configure_refit(dimensions)
    before = _cuda_mem_free()
    peak_bytes = 0
    dr, cuda, _, rayd = _load_backend()
    components = _mesh_components(dimensions["edge_count"], dimensions["distribution"], matrix["seed"])
    query_data = _query_data(dimensions["query_count"], dimensions["distribution"], matrix["seed"])
    start = time.perf_counter()
    scene, _ = _build_scene(components, dimensions["update_mode"] != "static", cuda, rayd)
    peak_bytes = max(peak_bytes, before - _cuda_mem_free())
    _apply_mask(scene, dimensions, matrix, cuda, dr)
    peak_bytes = max(peak_bytes, before - _cuda_mem_free())
    result = _query(scene, dimensions, _make_query(dimensions, query_data, cuda, rayd))
    _materialize(result, dimensions, dr)
    dr.sync_thread()
    peak_bytes = max(peak_bytes, before - _cuda_mem_free())
    return {
        "cold_create_ms": (time.perf_counter() - start) * 1000.0,
        "peak_device_memory_bytes": float(max(0, peak_bytes)),
    }


def _cold_samples(dimensions: dict[str, Any], matrix_path: Path, count: int) -> tuple[list[float], list[float]]:
    cold_samples = []
    memory_samples = []
    for _ in range(count):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--matrix", str(matrix_path),
            "--cold-case", json.dumps(dimensions, separators=(",", ":")),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(COLD_PREFIX):
                payload = json.loads(line[len(COLD_PREFIX):])
                if "error" in payload:
                    raise ContractError(payload["error"])
                cold_samples.append(float(payload["cold_create_ms"]))
                memory_samples.append(float(payload["peak_device_memory_bytes"]))
                break
        else:
            raise ContractError(result.stderr.strip() or "cold worker returned no result")
    return cold_samples, memory_samples


def collect_case_sample(
    dimensions: dict[str, Any], matrix: dict[str, Any], matrix_path: Path = MATRIX_PATH,
) -> dict[str, Any]:
    _configure_refit(dimensions)
    dr, cuda, ad, rayd = _load_backend()
    repeats = matrix["measurement"]["minimum_timed_runs"]
    warmups = matrix["measurement"]["warmup_runs"]
    components = _mesh_components(dimensions["edge_count"], dimensions["distribution"], matrix["seed"])
    query_data = _query_data(dimensions["query_count"], dimensions["distribution"], matrix["seed"])

    build_samples: list[float] = []
    hot_samples: list[float] = []
    refit_samples: list[float] = []
    scene = None
    mesh_ids: list[int] = []
    result = None
    for _ in range(repeats):
        dr.flush_malloc_cache()
        dr.sync_thread()
        start = time.perf_counter()
        scene, mesh_ids = _build_scene(components, dimensions["update_mode"] != "static", cuda, rayd)
        dr.sync_thread()
        build_samples.append((time.perf_counter() - start) * 1000.0)
        actual_edge_count = int(dr.width(scene.edge_info().global_edge_id))
        if actual_edge_count != dimensions["edge_count"]:
            raise ContractError(f"generated {actual_edge_count} edges, expected {dimensions['edge_count']}")

        _apply_mask(scene, dimensions, matrix, cuda, dr)
        query = _make_query(dimensions, query_data, cuda, rayd)
        for _ in range(warmups):
            _materialize(_query(scene, dimensions, query), dimensions, dr)
            dr.sync_thread()
        start = time.perf_counter()
        result = _query(scene, dimensions, query)
        _materialize(result, dimensions, dr)
        dr.sync_thread()
        hot_samples.append((time.perf_counter() - start) * 1000.0)
        refit_samples.append(_measure_refit(scene, mesh_ids, components, dimensions, cuda, dr))
    if scene is None or result is None:
        raise ContractError("case produced no scene or query result")

    cold_samples, memory_samples = _cold_samples(dimensions, matrix_path, repeats)
    return {
        "performance": {
            "hot_query_ms": hot_samples,
            "build_ms": build_samples,
            "refit_ms": refit_samples,
            "peak_device_memory_bytes": memory_samples,
            "cold_create_ms": cold_samples,
        },
        "correctness": _correctness(result, dimensions, dr),
        "ad": _ad_errors(scene, dimensions, query_data, dr, cuda, ad, rayd),
    }


def orchestrate(profile: str, matrix_path: Path) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    cases = []
    for dimensions in expected_case_dimensions(matrix, profile):
        cases.append(aggregate_case(dimensions, run_worker(dimensions, matrix_path)))
    payload = {
        "schema_version": matrix["schema_version"],
        "matrix_id": matrix["matrix_id"],
        "benchmark": matrix["benchmark"],
        "seed": matrix["seed"],
        "profile": profile,
        "environment": collect_environment(),
        "tolerances": matrix["tolerances"],
        "cases": cases,
    }
    validate_result(payload, matrix)
    gate = evaluate_gate(payload, payload, matrix)
    if not gate["passed"]:
        raise ContractError(f"benchmark result exceeds frozen tolerances: {gate['failures']}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RayD edge BVH benchmark matrix.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--cold-case", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
        if args.cold_case is not None:
            value = _cold_once(json.loads(args.cold_case), matrix)
            print(COLD_PREFIX + json.dumps(value, separators=(",", ":")))
            return 0
        if args.worker_case is not None:
            payload = collect_case_sample(json.loads(args.worker_case), matrix, args.matrix)
            print(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")))
            return 0
        payload = orchestrate(args.profile, args.matrix)
        text = json.dumps(payload, indent=2)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(text, encoding="utf-8")
        print(text)
        return 0
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        if args.cold_case is not None:
            print(COLD_PREFIX + json.dumps({"error": str(exc)}, separators=(",", ":")))
        elif args.worker_case is not None:
            print(WORKER_PREFIX + json.dumps({"error": str(exc)}, separators=(",", ":")))
        else:
            print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
