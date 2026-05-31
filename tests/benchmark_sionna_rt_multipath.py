import argparse
import importlib.metadata
import json
import math
import os
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)

import drjit as dr
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIONNA_ROOT = Path(
    r"E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1"
)
SPEED_OF_LIGHT = 299_792_458.0


def _prepare_imports(sionna_root: Path):
    src = sionna_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "jit_kernel_count": len(history),
        "jit_optix_kernel_count": sum(int(k.get("uses_optix", 0)) for k in history),
        "jit_exec_ms": sum(float(k.get("execution_time", 0.0)) for k in history),
        "jit_optix_exec_ms": sum(
            float(k.get("execution_time", 0.0))
            for k in history
            if int(k.get("uses_optix", 0))
        ),
        "jit_codegen_ms": sum(float(k.get("codegen_time", 0.0)) for k in history),
        "jit_backend_ms": sum(float(k.get("backend_time", 0.0)) for k in history),
    }


def summarize_numeric_samples(samples: list[float]) -> dict[str, float]:
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "avg_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, math.ceil(0.95 * len(samples)) - 1)],
    }


def measure(
    fn: Callable[[], Any],
    materialize: Callable[[Any], None],
    repeats: int,
    warmup: int,
    after_sample: Callable[[Any], None] | None = None,
):
    for _ in range(warmup):
        value = fn()
        materialize(value)
        dr.sync_thread()

    samples = []
    histories = []
    last_value = None
    for _ in range(repeats):
        dr.kernel_history_clear()
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
                dr.sync_thread()
                start = time.perf_counter()
                last_value = fn()
                materialize(last_value)
                dr.sync_thread()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed_ms)
        histories.append(summarize_history(dr.kernel_history()))
        if after_sample is not None:
            after_sample(last_value)

    kernel_timing = {
        key: summarize_numeric_samples([h[key] for h in histories])
        for key in ("jit_exec_ms", "jit_optix_exec_ms", "jit_codegen_ms", "jit_backend_ms")
    } if histories else {}

    return {
        **summarize_numeric_samples(samples),
        "kernel_history_avg": {
            key: statistics.fmean(h[key] for h in histories)
            for key in histories[0]
        }
        if histories
        else {},
        "kernel_history_timing": kernel_timing,
        "last_value": last_value,
    }


def summarize_native_optix_launch_timing(
    audit_samples: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    launch_counts = [
        int(sample.get(stage, {}).get("optix_launch", 0))
        for sample in audit_samples
    ]
    durations = [
        float(sample.get(stage, {}).get("optix_launch_time_ms", 0.0))
        for sample in audit_samples
    ]
    result = {
        "stage": stage,
        "samples_ms": durations,
        "launch_count_samples": launch_counts,
        "launch_count_avg": statistics.fmean(launch_counts) if launch_counts else 0.0,
    }
    timed = [duration for duration in durations if duration > 0.0]
    if timed:
        result.update({
            "min_ms": min(timed),
            "avg_ms": statistics.fmean(timed),
            "p50_ms": statistics.median(timed),
            "p95_ms": sorted(timed)[max(0, math.ceil(0.95 * len(timed)) - 1)],
        })
    else:
        result.update({
            "min_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        })
    return result


def load_binary_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    vertex_count = 0
    face_count = 0
    for line in header:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])

    offset = header_end
    vertices = []
    for _ in range(vertex_count):
        x, y, z, _u, _v = struct.unpack_from("<5f", data, offset)
        offset += 20
        vertices.append((x, y, z))

    faces = []
    for _ in range(face_count):
        count = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        indices = struct.unpack_from("<" + "i" * count, data, offset)
        offset += 4 * count
        if count == 3:
            faces.append((indices[0], indices[1], indices[2]))
    return vertices, faces


def reflector_mesh_path(sionna_root: Path) -> Path:
    return sionna_root / "src" / "sionna" / "rt" / "scenes" / "simple_reflector" / "meshes" / "reflector.ply"


def wedge_mesh_path(sionna_root: Path) -> Path:
    return sionna_root / "src" / "sionna" / "rt" / "scenes" / "simple_wedge" / "meshes" / "wedge.ply"


def reflection_rx_positions(rx_side: int) -> list[tuple[float, float, float]]:
    positions = []
    for iy in range(rx_side):
        y = 0.2 * (iy - (rx_side - 1) * 0.5)
        for ix in range(rx_side):
            x = 10.0 + 0.2 * (ix - (rx_side - 1) * 0.5)
            positions.append((x, y, 10.0))
    return positions


def diffraction_rx_positions(rx_side: int) -> list[tuple[float, float, float]]:
    positions = []
    for iy in range(rx_side):
        y = -2.5 + 0.1 * (iy - (rx_side - 1) * 0.5)
        for ix in range(rx_side):
            x = -1.0 + 0.1 * (ix - (rx_side - 1) * 0.5)
            positions.append((x, y, -10.0))
    return positions


def cycle_positions(positions: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
    return [positions[i % len(positions)] for i in range(count)]


def make_sionna_scene(workload: str, sionna_root: Path, rx_side: int):
    import mitsuba as mi
    import sionna.rt as rt
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

    if workload in ("los", "reflection"):
        scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)
        scene.add(Transmitter("tx", [-10.0, 0.0, 10.0]))
        rx_positions = reflection_rx_positions(rx_side)
    else:
        scene = load_scene(rt.scene.simple_wedge, merge_shapes=False)
        scene.add(Transmitter("tx", [1.0, 1.0, 0.0], orientation=[0.0, 0.0, 0.0]))
        rx_positions = diffraction_rx_positions(rx_side)

    for i, pos in enumerate(rx_positions):
        scene.add(Receiver(f"rx-{i}", list(pos), orientation=mi.Point3f(0.0, 0.0, 0.0)))

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="VH",
    )
    return scene


def make_mitsuba_scene(workload: str):
    import sionna.rt as rt
    from sionna.rt import load_scene

    scene_file = rt.scene.simple_reflector if workload in ("los", "reflection") else rt.scene.simple_wedge
    return load_scene(scene_file, merge_shapes=False).mi_scene


def mi_points(positions: list[tuple[float, float, float]]):
    import mitsuba as mi

    xs, ys, zs = zip(*positions)
    return mi.Point3f(list(xs), list(ys), list(zs))


def mi_repeat_point(pos: tuple[float, float, float], count: int):
    import mitsuba as mi

    return mi.Point3f([pos[0]] * count, [pos[1]] * count, [pos[2]] * count)


def mi_segment_visible(mi_scene, start, end, eps: float = 1e-4):
    import mitsuba as mi

    delta = end - start
    dist = dr.norm(delta)
    direction = delta / dist
    ray = mi.Ray3f(start + direction * eps, direction)
    ray.maxt = dr.maximum(dist - 2.0 * eps, 0.0)
    active = dist > (2.0 * eps)
    return ~mi_scene.ray_test(ray, active=active)


def run_mitsuba_minimal_los(args: argparse.Namespace) -> dict[str, Any]:
    mi_scene = make_mitsuba_scene("los")
    rx_positions = reflection_rx_positions(args.rx_side)
    n = len(rx_positions)
    tx = mi_repeat_point((-10.0, 0.0, 10.0), n)
    rx = mi_points(rx_positions)

    def call_kernel():
        visible = mi_segment_visible(mi_scene, tx, rx)
        dist = dr.norm(rx - tx)
        checksum = dr.sum(dr.select(visible, dist, 0.0))
        count = dr.sum(dr.select(visible, 1.0, 0.0))
        return visible, checksum, count

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    visible, checksum, count = measured.pop("last_value")
    return {
        "backend": "mitsuba_minimal",
        "workload": "los",
        "tier": 1,
        "scene": "simple_reflector",
        "rx_count": n,
        "valid_path_count": int(count[0]),
        "path_length_checksum": float(checksum[0]),
        "timing": measured,
    }


def run_mitsuba_minimal_reflection(args: argparse.Namespace) -> dict[str, Any]:
    import mitsuba as mi

    mi_scene = make_mitsuba_scene("reflection")
    rx_positions = reflection_rx_positions(args.rx_side)
    n = len(rx_positions)
    tx = mi_repeat_point((-10.0, 0.0, 10.0), n)
    rx = mi_points(rx_positions)
    tx_image = mi_repeat_point((-10.0, 0.0, -10.0), n)

    def call_kernel():
        image_to_rx = rx - tx_image
        alpha = -tx_image.z / image_to_rx.z
        point = tx_image + image_to_rx * alpha
        in_bounds = (dr.abs(point.x) <= 0.5) & (dr.abs(point.y) <= 0.5)

        to_point = point - tx
        dist0 = dr.norm(to_point)
        ray = mi.Ray3f(tx, to_point / dist0)
        ray.maxt = dist0 + 1e-4
        si = mi_scene.ray_intersect(
            ray,
            ray_flags=mi.RayFlags.Minimal,
            coherent=True,
            active=in_bounds,
        )
        hit_reflector = si.is_valid() & (dr.abs(si.t - dist0) < 1e-3)
        visible1 = hit_reflector
        visible2 = mi_segment_visible(mi_scene, point, rx)
        valid = in_bounds & visible1 & visible2
        total_dist = dist0 + dr.norm(rx - point)
        checksum = dr.sum(dr.select(valid, total_dist, 0.0))
        count = dr.sum(dr.select(valid, 1.0, 0.0))
        return valid, checksum, count, si.t

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    valid, checksum, count, _t = measured.pop("last_value")
    return {
        "backend": "mitsuba_minimal",
        "workload": "reflection",
        "tier": 1,
        "scene": "simple_reflector",
        "rx_count": n,
        "valid_path_count": int(count[0]),
        "path_length_checksum": float(checksum[0]),
        "timing": measured,
    }


def run_mitsuba_minimal_diffraction(args: argparse.Namespace) -> dict[str, Any]:
    mi_scene = make_mitsuba_scene("diffraction")
    n = args.state_count
    rx_positions = cycle_positions(diffraction_rx_positions(args.rx_side), n)
    tx = mi_repeat_point((1.0, 1.0, 0.0), n)
    rx = mi_points(rx_positions)
    if n == 1:
        z_values = [-5.0]
    else:
        z_values = [-10.0 + 10.0 * i / (n - 1) for i in range(n)]
    edge_points = mi_points([(0.0, 0.0, z) for z in z_values])

    def call_kernel():
        visible0 = mi_segment_visible(mi_scene, tx, edge_points)
        visible1 = mi_segment_visible(mi_scene, edge_points, rx)
        valid = visible0 & visible1
        total_dist = dr.norm(edge_points - tx) + dr.norm(rx - edge_points)
        checksum = dr.sum(dr.select(valid, total_dist, 0.0))
        count = dr.sum(dr.select(valid, 1.0, 0.0))
        return valid, checksum, count

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    valid, checksum, count = measured.pop("last_value")
    return {
        "backend": "mitsuba_minimal",
        "workload": "diffraction",
        "tier": 1,
        "scene": "simple_wedge",
        "state_count": n,
        "rx_count": args.rx_side * args.rx_side,
        "valid_path_count": int(count[0]),
        "path_length_checksum": float(checksum[0]),
        "timing": measured,
    }


def run_mitsuba_minimal(args: argparse.Namespace, workload: str) -> dict[str, Any]:
    if workload == "los":
        return run_mitsuba_minimal_los(args)
    if workload == "reflection":
        return run_mitsuba_minimal_reflection(args)
    return run_mitsuba_minimal_diffraction(args)


def metric_with_throughput(
    result: dict[str, Any],
    input_count: int,
    valid_count: int,
    output_bytes: int,
    optix_launch_count: float,
) -> dict[str, Any]:
    timing = result["timing"]
    p50_s = max(timing["p50_ms"] / 1000.0, 1e-12)
    min_s = max(timing["min_ms"] / 1000.0, 1e-12)
    result["input_count"] = input_count
    result["valid_path_count"] = valid_count
    result["estimated_output_bytes"] = output_bytes
    result["estimated_output_mib"] = output_bytes / (1024.0 * 1024.0)
    result["optix_launch_count_avg"] = optix_launch_count
    result["input_throughput_p50_per_s"] = input_count / p50_s
    result["input_throughput_min_per_s"] = input_count / min_s
    result["valid_path_throughput_p50_per_s"] = valid_count / p50_s
    result["valid_path_throughput_min_per_s"] = valid_count / min_s
    return result


def reflection_trace_output_bytes(ray_count: int, max_bounces: int, export_mode: str = "full") -> int:
    slot_count = ray_count * max_bounces
    if export_mode == "count_only":
        return ray_count * 4
    if export_mode == "minimal":
        return ray_count * 4 + slot_count * (4 + 4 + 4)
    per_ray = 3 * 4 + 4 + 4 + 3 * 4 + 3 * 4
    per_slot = 4 + 5 * 3 * 4 + 4 * 4
    return ray_count * per_ray + slot_count * per_slot


def dfr_path_output_bytes(capacity: int) -> int:
    per_path = 1 + 6 * 4 + 4 + 3 * 2 * 4 + 3 * 3 * 4
    return 4 + capacity * per_path


def make_reflection_trace_rays_mi(ray_count: int):
    import mitsuba as mi

    grid_side = int(math.ceil(math.sqrt(ray_count)))
    idx = dr.arange(mi.UInt, ray_count)
    ix = idx % grid_side
    iy = idx // grid_side
    denom = max(1, grid_side - 1)
    x = -0.9 + 1.8 * mi.Float(ix) / denom
    y = -0.9 + 1.8 * mi.Float(iy) / denom
    origin = mi.Point3f(x, y, dr.full(mi.Float, 0.5, ray_count))
    direction = mi.Vector3f(
        dr.zeros(mi.Float, ray_count),
        dr.zeros(mi.Float, ray_count),
        dr.full(mi.Float, 1.0, ray_count),
    )
    return mi.Ray3f(origin, direction)


def make_reflection_trace_rays_cuda(ray_count: int):
    import drjit.cuda as cuda
    import rayd as rd

    grid_side = int(math.ceil(math.sqrt(ray_count)))
    idx = dr.arange(cuda.UInt, ray_count)
    ix = idx % grid_side
    iy = idx // grid_side
    denom = max(1, grid_side - 1)
    x = -0.9 + 1.8 * cuda.Float(ix) / denom
    y = -0.9 + 1.8 * cuda.Float(iy) / denom
    origin = cuda.Array3f(x, y, dr.full(cuda.Float, 0.5, ray_count))
    direction = cuda.Array3f(
        dr.zeros(cuda.Float, ray_count),
        dr.zeros(cuda.Float, ray_count),
        dr.full(cuda.Float, 1.0, ray_count),
    )
    return rd.Ray(origin, direction)


def make_rayd_parallel_reflector_scene():
    import drjit.cuda as cuda
    import rayd as rd

    vertices = cuda.Array3f(
        [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    faces = cuda.Array3i([0, 0, 4, 4], [1, 2, 5, 6], [2, 3, 6, 7])
    scene = rd.Scene()
    scene.add_mesh(rd.Mesh(vertices, faces))
    scene.build()
    return scene


def make_mitsuba_parallel_reflector_scene():
    import mitsuba as mi

    transform = mi.ScalarTransform4f
    return mi.load_dict({
        "type": "scene",
        "lower": {
            "type": "rectangle",
            "to_world": transform.translate([0.0, 0.0, 0.0]),
        },
        "upper": {
            "type": "rectangle",
            "to_world": transform.translate([0.0, 0.0, 1.0]),
        },
    })


def run_mitsuba_path_reflection_trace(args: argparse.Namespace) -> dict[str, Any]:
    import mitsuba as mi

    mi_scene = make_mitsuba_parallel_reflector_scene()
    ray = make_reflection_trace_rays_mi(args.ray_count)

    def call_kernel():
        active = dr.full(mi.Bool, True, args.ray_count)
        origin = ray.o
        direction = ray.d
        ts = []
        prim_ids = []
        for _ in range(args.max_bounces):
            bounce_ray = mi.Ray3f(origin, direction)
            if args.mitsuba_ray_api == "preliminary":
                pi = mi_scene.ray_intersect_preliminary(
                    bounce_ray,
                    coherent=True,
                    active=active,
                )
                hit = active & pi.is_valid()
                si = pi.compute_surface_interaction(
                    bounce_ray,
                    mi.RayFlags.Minimal,
                    hit,
                )
                hit_t = pi.t
                prim_index = pi.prim_index
                hit_point = origin + direction * pi.t
                normal = si.n
            else:
                si = mi_scene.ray_intersect(
                    bounce_ray,
                    ray_flags=mi.RayFlags.Minimal,
                    coherent=True,
                    active=active,
                )
                hit = active & si.is_valid()
                hit_t = si.t
                prim_index = si.prim_index
                hit_point = si.p
                normal = si.n
            ts.append(dr.select(hit, hit_t, 0.0))
            prim_ids.append(dr.select(hit, prim_index, mi.UInt(0xFFFFFFFF)))
            direction = direction - 2.0 * dr.dot(direction, normal) * normal
            origin = hit_point + direction * 1e-4
            active &= hit
        valid_count = dr.sum(dr.select(active, 1.0, 0.0))
        slot_count = dr.sum(dr.select(dr.concat(ts) > 0.0, 1.0, 0.0))
        checksum = dr.sum(dr.concat(ts))
        return ts, prim_ids, valid_count, slot_count, checksum

    def materialize(value):
        ts, prim_ids, valid_count, slot_count, checksum = value
        dr.eval(*ts, *prim_ids, valid_count, slot_count, checksum)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    _ts, _prim_ids, valid_count, slot_count, checksum = measured.pop("last_value")
    result = {
        "backend": "mitsuba_path",
        "workload": "reflection_trace",
        "tier": 1,
        "scene": "parallel_reflectors",
        "mitsuba_ray_api": args.mitsuba_ray_api,
        "ray_count": args.ray_count,
        "max_bounces": args.max_bounces,
        "valid_full_depth_path_count": int(valid_count[0]),
        "valid_bounce_slot_count": int(slot_count[0]),
        "path_length_checksum": float(checksum[0]),
        "timing": measured,
    }
    return metric_with_throughput(
        result,
        args.ray_count,
        int(slot_count[0]),
        reflection_trace_output_bytes(args.ray_count, args.max_bounces, "minimal"),
        measured["kernel_history_avg"].get("jit_optix_kernel_count", 0.0),
    )


def run_rayd_path_reflection_trace(args: argparse.Namespace) -> dict[str, Any]:
    import rayd as rd

    scene = make_rayd_parallel_reflector_scene()
    ray = make_reflection_trace_rays_cuda(args.ray_count)
    options = rd.ReflectionTraceOptions()
    options.export_mode = rd.REFLECTION_EXPORT_MINIMAL
    options.return_trailing = False

    def call_kernel():
        rd.native_launch_audit_clear()
        return scene.trace_reflections(ray, args.max_bounces, options, True, False)

    def materialize(result):
        dr.eval(
            result.bounce_count,
            result.t,
            result.prim_ids,
            result.global_prim_ids,
        )

    audit_samples: list[dict[str, Any]] = []
    measured = measure(
        call_kernel,
        materialize,
        args.repeats,
        args.warmup,
        after_sample=lambda _result: audit_samples.append(rd.native_launch_audit()),
    )
    trace = measured.pop("last_value")
    slot_count = int(dr.sum(dr.select(trace.is_valid(), 1.0, 0.0))[0])
    full_depth_count = int(dr.sum(dr.select(trace.bounce_count == args.max_bounces, 1.0, 0.0))[0])
    checksum = float(dr.sum(dr.select(trace.is_valid(), trace.t, 0.0))[0])
    audit = audit_samples[-1] if audit_samples else rd.native_launch_audit()
    result = {
        "backend": "rayd_path",
        "workload": "reflection_trace",
        "tier": 1,
        "scene": "parallel_reflectors",
        "ray_count": args.ray_count,
        "max_bounces": args.max_bounces,
        "valid_full_depth_path_count": full_depth_count,
        "valid_bounce_slot_count": slot_count,
        "path_length_checksum": checksum,
        "native_audit": audit,
        "native_audit_samples": audit_samples,
        "native_optix_launch_timing": summarize_native_optix_launch_timing(
            audit_samples,
            "trace_reflections",
        ),
        "timing": measured,
    }
    return metric_with_throughput(
        result,
        args.ray_count,
        slot_count,
        reflection_trace_output_bytes(args.ray_count, args.max_bounces, "minimal"),
        float(audit["trace_reflections"]["optix_launch"]),
    )


def make_mitsuba_dfr_export_inputs(state_count: int):
    import mitsuba as mi

    tx = mi_repeat_point((0.0, 0.0, 1.0), state_count)
    rx = mi_repeat_point((0.0, 0.0, -1.0), state_count)
    edge_pos = mi_repeat_point((0.0, 0.0, 0.0), state_count)
    return tx, rx, edge_pos


def run_mitsuba_path_diffraction_export(args: argparse.Namespace) -> dict[str, Any]:
    mi_scene = make_mitsuba_scene("diffraction")
    tx, rx, edge_pos = make_mitsuba_dfr_export_inputs(args.state_count)

    def call_kernel():
        visible0 = mi_segment_visible(mi_scene, tx, edge_pos)
        visible1 = mi_segment_visible(mi_scene, edge_pos, rx)
        valid = visible0 & visible1
        delay = (dr.norm(edge_pos - tx) + dr.norm(rx - edge_pos)) / SPEED_OF_LIGHT
        checksum = dr.sum(dr.select(valid, delay, 0.0))
        count = dr.sum(dr.select(valid, 1.0, 0.0))
        return valid, delay, edge_pos, count, checksum

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    _valid, _delay, _edge_pos, count, checksum = measured.pop("last_value")
    valid_count = int(count[0])
    result = {
        "backend": "mitsuba_path",
        "workload": "diffraction_export",
        "tier": 1,
        "scene": "simple_wedge",
        "state_count": args.state_count,
        "path_capacity": args.state_count,
        "path_length_checksum": float(checksum[0]),
        "timing": measured,
    }
    return metric_with_throughput(
        result,
        args.state_count,
        valid_count,
        dfr_path_output_bytes(args.state_count),
        measured["kernel_history_avg"].get("jit_optix_kernel_count", 0.0),
    )


def run_rayd_path_diffraction_export(args: argparse.Namespace) -> dict[str, Any]:
    import drjit.cuda as cuda
    import rayd as rd

    scene = rd.Scene()
    vertices = cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0])
    scene.add_mesh(rd.Mesh(vertices, cuda.Array3i([0], [1], [2])))
    scene.build()
    states = make_rayd_dfr_states(args.state_count)

    material = rd.DfrMaterial()
    material.eta_r = cuda.Float([4.0])
    material.sigma = cuda.Float([0.0])
    material.mu_r = cuda.Float([1.0])
    material.gain = cuda.Float([1.0])
    material.valid = cuda.Bool([True])

    options = rd.DfrPathOptions()
    options.wavelength = 0.125
    options.k = 50.26548245743669
    options.seed = args.seed
    options.max_order = 1
    options.max_paths = args.state_count
    options.max_rx = 1
    options.strategy_mask = rd.RAYD_DFR_DIRECT
    options.sample_count = 1
    options.return_geom = 1
    options.receiver_model = rd.RAYD_DFR_MATCHED_ISO

    tx = cuda.Array3f([0.0], [0.0], [1.0])
    rx = cuda.Array3f([0.0], [0.0], [-1.0])

    def call_kernel():
        rd.native_launch_audit_clear()
        return scene.trace_dfr_paths(tx, rx, states, material, options, cuda.Bool([True]))

    def materialize(result):
        dr.eval(
            result.count,
            result.valid,
            result.rx_id,
            result.edge0,
            result.delay,
            result.field_x.real,
            result.field_x.imag,
            result.p0,
            result.p1,
            result.p2,
        )

    audit_samples: list[dict[str, Any]] = []
    measured = measure(
        call_kernel,
        materialize,
        args.repeats,
        args.warmup,
        after_sample=lambda _result: audit_samples.append(rd.native_launch_audit()),
    )
    paths = measured.pop("last_value")
    valid_count = int(paths.count[0])
    checksum = float(dr.sum(dr.select(paths.valid, paths.delay, 0.0))[0])
    audit = audit_samples[-1] if audit_samples else rd.native_launch_audit()
    result = {
        "backend": "rayd_path",
        "workload": "diffraction_export",
        "tier": 1,
        "scene": "synthetic_single_edge_state",
        "state_count": args.state_count,
        "path_capacity": paths.capacity,
        "path_length_checksum": checksum,
        "native_audit": audit,
        "native_audit_samples": audit_samples,
        "native_optix_launch_timing": summarize_native_optix_launch_timing(
            audit_samples,
            "accum_dfr",
        ),
        "timing": measured,
    }
    return metric_with_throughput(
        result,
        args.state_count,
        valid_count,
        dfr_path_output_bytes(paths.capacity),
        float(audit["accum_dfr"]["optix_launch"]),
    )


def run_mitsuba_path(args: argparse.Namespace, workload: str) -> dict[str, Any]:
    if workload == "reflection_trace":
        return run_mitsuba_path_reflection_trace(args)
    if workload == "diffraction_export":
        return run_mitsuba_path_diffraction_export(args)
    raise ValueError(f"mitsuba_path backend does not support workload {workload!r}")


def run_rayd_path(args: argparse.Namespace, workload: str) -> dict[str, Any]:
    if workload == "reflection_trace":
        return run_rayd_path_reflection_trace(args)
    if workload == "diffraction_export":
        return run_rayd_path_diffraction_export(args)
    raise ValueError(f"rayd_path backend does not support workload {workload!r}")


def run_sionna(args: argparse.Namespace, workload: str) -> dict[str, Any]:
    from sionna.rt import PathSolver

    scene = make_sionna_scene(workload, args.sionna_root, args.rx_side)
    solver = PathSolver()
    solver.loop_mode = args.loop_mode

    def call_solver():
        return solver(
            scene,
            max_depth=args.max_bounces,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=(workload == "los") or (workload == "reflection" and args.include_los),
            specular_reflection=(workload == "reflection"),
            diffuse_reflection=False,
            refraction=False,
            diffraction=(workload == "diffraction"),
            edge_diffraction=args.edge_diffraction,
            diffraction_lit_region=True,
            seed=args.seed,
        )

    def materialize(paths):
        dr.eval(paths.valid, paths.tau, paths.a[0], paths.a[1])

    measured = measure(call_solver, materialize, args.repeats, args.warmup)
    paths = measured.pop("last_value")
    valid = paths.valid.numpy()
    tau = paths.tau.numpy()
    a0 = paths.a[0].numpy()
    a1 = paths.a[1].numpy()
    return {
        "backend": "sionna_rt",
        "workload": workload,
        "tier": 2,
        "scene": "simple_reflector" if workload in ("los", "reflection") else "simple_wedge",
        "loop_mode": args.loop_mode,
        "rx_count": args.rx_side * args.rx_side,
        "max_depth": args.max_bounces,
        "samples_per_src": args.samples_per_src,
        "max_num_paths_per_src": args.max_num_paths_per_src,
        "valid_path_count": int(np.count_nonzero(valid)),
        "tau_shape": list(tau.shape),
        "a_shape": list(a0.shape),
        "field_abs_sum": float(np.sum(np.abs(a0 + 1j * a1))),
        "timing": measured,
    }


def rayd_reflector_scene(sionna_root: Path):
    import drjit.cuda as cuda
    import rayd as rd

    ply = reflector_mesh_path(sionna_root)
    vertices, faces = load_binary_ply(ply)
    xs, ys, zs = zip(*vertices)
    i0, i1, i2 = zip(*faces)
    scene = rd.Scene()
    scene.add_mesh(
        rd.Mesh(
            cuda.Array3f(list(xs), list(ys), list(zs)),
            cuda.Array3i(list(i0), list(i1), list(i2)),
        )
    )
    scene.build()
    return scene, len(faces)


def rayd_scene_from_ply(path: Path):
    import drjit.cuda as cuda
    import rayd as rd

    vertices, faces = load_binary_ply(path)
    xs, ys, zs = zip(*vertices)
    i0, i1, i2 = zip(*faces)
    scene = rd.Scene()
    scene.add_mesh(
        rd.Mesh(
            cuda.Array3f(list(xs), list(ys), list(zs)),
            cuda.Array3i(list(i0), list(i1), list(i2)),
        )
    )
    scene.build()
    return scene, len(faces)


def cuda_points(positions: list[tuple[float, float, float]]):
    import drjit.cuda as cuda

    xs, ys, zs = zip(*positions)
    return cuda.Array3f(list(xs), list(ys), list(zs))


def cuda_repeat_point(pos: tuple[float, float, float], count: int):
    import drjit.cuda as cuda

    return cuda.Array3f([pos[0]] * count, [pos[1]] * count, [pos[2]] * count)


def rayd_segment_visible(scene, start, end, eps: float = 1e-4):
    delta = end - start
    dist = dr.norm(delta)
    direction = delta / dist
    return scene.visible(start + direction * eps, end - direction * eps).visible


def run_rayd_minimal_los(args: argparse.Namespace) -> dict[str, Any]:
    import rayd as rd

    scene, _prim_count = rayd_reflector_scene(args.sionna_root)
    rx_positions = reflection_rx_positions(args.rx_side)
    n = len(rx_positions)
    tx = cuda_repeat_point((-10.0, 0.0, 10.0), n)
    rx = cuda_points(rx_positions)

    def call_kernel():
        rd.native_launch_audit_clear()
        visible = rayd_segment_visible(scene, tx, rx)
        dist = dr.norm(rx - tx)
        checksum = dr.sum(dr.select(visible, dist, 0.0))
        count = dr.sum(dr.select(visible, 1.0, 0.0))
        return visible, checksum, count

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    visible, checksum, count = measured.pop("last_value")
    return {
        "backend": "rayd_minimal",
        "workload": "los",
        "tier": 1,
        "scene": "simple_reflector_ply",
        "rx_count": n,
        "valid_path_count": int(count[0]),
        "path_length_checksum": float(checksum[0]),
        "native_audit": rd.native_launch_audit(),
        "timing": measured,
    }


def run_rayd_minimal_reflection(args: argparse.Namespace) -> dict[str, Any]:
    import drjit.cuda as cuda
    import rayd as rd

    scene, _prim_count = rayd_reflector_scene(args.sionna_root)
    rx_positions = reflection_rx_positions(args.rx_side)
    n = len(rx_positions)
    tx = cuda_repeat_point((-10.0, 0.0, 10.0), n)
    rx = cuda_points(rx_positions)
    tx_image = cuda_repeat_point((-10.0, 0.0, -10.0), n)

    def call_kernel():
        rd.native_launch_audit_clear()
        image_to_rx = rx - tx_image
        alpha = -tx_image.z / image_to_rx.z
        point = tx_image + image_to_rx * alpha
        in_bounds = (dr.abs(point.x) <= 0.5) & (dr.abs(point.y) <= 0.5)

        to_point = point - tx
        dist0 = dr.norm(to_point)
        ray = rd.Ray(tx, to_point / dist0)
        ray.tmax = dist0 + 1e-4
        its = scene.intersect(ray)
        hit_reflector = its.is_valid() & (dr.abs(its.t - dist0) < 1e-3)
        visible2 = rayd_segment_visible(scene, point, rx)
        valid = in_bounds & hit_reflector & visible2
        total_dist = dist0 + dr.norm(rx - point)
        checksum = dr.sum(dr.select(valid, total_dist, 0.0))
        count = dr.sum(dr.select(valid, 1.0, 0.0))
        return valid, checksum, count, its.t

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    valid, checksum, count, _t = measured.pop("last_value")
    return {
        "backend": "rayd_minimal",
        "workload": "reflection",
        "tier": 1,
        "scene": "simple_reflector_ply",
        "rx_count": n,
        "valid_path_count": int(count[0]),
        "path_length_checksum": float(checksum[0]),
        "native_audit": rd.native_launch_audit(),
        "timing": measured,
    }


def run_rayd_minimal_diffraction(args: argparse.Namespace) -> dict[str, Any]:
    import rayd as rd

    scene, _prim_count = rayd_scene_from_ply(wedge_mesh_path(args.sionna_root))
    n = args.state_count
    rx_positions = cycle_positions(diffraction_rx_positions(args.rx_side), n)
    tx = cuda_repeat_point((1.0, 1.0, 0.0), n)
    rx = cuda_points(rx_positions)
    if n == 1:
        z_values = [-5.0]
    else:
        z_values = [-10.0 + 10.0 * i / (n - 1) for i in range(n)]
    edge_points = cuda_points([(0.0, 0.0, z) for z in z_values])

    def call_kernel():
        rd.native_launch_audit_clear()
        visible0 = rayd_segment_visible(scene, tx, edge_points)
        visible1 = rayd_segment_visible(scene, edge_points, rx)
        valid = visible0 & visible1
        total_dist = dr.norm(edge_points - tx) + dr.norm(rx - edge_points)
        checksum = dr.sum(dr.select(valid, total_dist, 0.0))
        count = dr.sum(dr.select(valid, 1.0, 0.0))
        return valid, checksum, count

    def materialize(value):
        dr.eval(*value)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    valid, checksum, count = measured.pop("last_value")
    return {
        "backend": "rayd_minimal",
        "workload": "diffraction",
        "tier": 1,
        "scene": "simple_wedge_ply",
        "state_count": n,
        "rx_count": args.rx_side * args.rx_side,
        "valid_path_count": int(count[0]),
        "path_length_checksum": float(checksum[0]),
        "native_audit": rd.native_launch_audit(),
        "timing": measured,
    }


def run_rayd_minimal(args: argparse.Namespace, workload: str) -> dict[str, Any]:
    if workload == "los":
        return run_rayd_minimal_los(args)
    if workload == "reflection":
        return run_rayd_minimal_reflection(args)
    return run_rayd_minimal_diffraction(args)


def run_rayd_reflection(args: argparse.Namespace) -> dict[str, Any]:
    import drjit.cuda as cuda
    import rayd as rd

    scene, prim_count = rayd_reflector_scene(args.sionna_root)
    n = args.ray_count
    zero = dr.full(cuda.Float, 0.0, n)
    ray = rd.Ray(
        cuda.Array3f(zero, zero, dr.full(cuda.Float, -1.0, n)),
        cuda.Array3f(zero, zero, dr.full(cuda.Float, 1.0, n)),
    )
    tx = cuda.Array3f([0.0], [0.0], [-1.0])

    grid = rd.AccumGrid()
    grid.axis = 2
    grid.position = -2.0
    grid.coord0_min = -1.0
    grid.coord0_max = 1.0
    grid.coord1_min = -1.0
    grid.coord1_max = 1.0
    grid.resolution0 = args.grid_side
    grid.resolution1 = args.grid_side

    material = rd.Material()
    material.eta_r = dr.full(cuda.Float, 4.0, prim_count)
    material.sigma = dr.full(cuda.Float, 0.0, prim_count)
    material.gain = dr.full(cuda.Float, 1.0, prim_count)
    material.mu_r = dr.full(cuda.Float, 1.0, prim_count)
    material.valid = dr.full(cuda.Bool, True, prim_count)

    options = rd.AccumOptions()
    options.wavelength = 12.566370614359172
    options.k = 0.5
    options.solid_angle_per_ray = 1.0 / max(1, n)
    options.cell_area = 4.0 / max(1, args.grid_side * args.grid_side)
    options.seed = args.seed
    options.rr_depth = 0
    options.rr_prob = 1.0
    options.stop_threshold = 0.0

    def call_kernel():
        rd.native_launch_audit_clear()
        return scene.accumulate_reflections(ray, tx, grid, material, args.max_bounces, options)

    def materialize(result):
        dr.eval(result.reflection_power, result.reflection_count)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    result = measured.pop("last_value")
    audit = rd.native_launch_audit()
    return {
        "backend": "rayd",
        "workload": "reflection",
        "tier": 3,
        "scene": "simple_reflector_ply",
        "ray_count": n,
        "grid_cell_count": args.grid_side * args.grid_side,
        "max_depth": args.max_bounces,
        "reflection_count": int(result.reflection_count[0]),
        "power_sum": float(dr.sum(result.reflection_power)[0]),
        "native_audit": audit,
        "timing": measured,
    }


def make_rayd_dfr_states(state_count: int):
    import drjit.cuda as cuda
    import rayd as rd

    states = rd.DfrStates()
    states.count = state_count
    states.edge_index = dr.arange(cuda.Int, state_count)
    states.edge_pos = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.edge_dir = cuda.Array3f(
        dr.full(cuda.Float, 1.0, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.edge_t_min = dr.full(cuda.Float, -0.5, state_count)
    states.edge_t_max = dr.full(cuda.Float, 0.5, state_count)
    states.n0 = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, 1.0, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.n1 = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, -1.0, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.prim0 = dr.full(cuda.Int, -1, state_count)
    states.prim1 = dr.full(cuda.Int, -1, state_count)
    states.exterior_angle = dr.full(cuda.Float, 1.5 * math.pi, state_count)
    states.src = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, 1.0, state_count),
    )
    states.src_power = dr.full(cuda.Float, 2.0, state_count)
    states.wi = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, -1.0, state_count),
    )
    states.d0 = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, -1.0, state_count),
    )
    states.prefix_depth = dr.full(cuda.Int, 0, state_count)
    return states


def run_rayd_diffraction(args: argparse.Namespace) -> dict[str, Any]:
    import drjit.cuda as cuda
    import rayd as rd

    scene = rd.Scene()
    vertices = cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0])
    scene.add_mesh(rd.Mesh(vertices, cuda.Array3i([0], [1], [2])))
    scene.build()

    states = make_rayd_dfr_states(args.state_count)
    grid = rd.DfrGrid()
    grid.axis = 2
    grid.position = -1.0
    grid.coord0_min = -1.0
    grid.coord0_max = 1.0
    grid.coord1_min = -1.0
    grid.coord1_max = 1.0
    grid.resolution0 = args.grid_side
    grid.resolution1 = args.grid_side
    grid.cell_area = 4.0 / max(1, args.grid_side * args.grid_side)

    material = rd.DfrMaterial()
    material.eta_r = cuda.Float([4.0])
    material.sigma = cuda.Float([0.0])
    material.mu_r = cuda.Float([1.0])
    material.gain = cuda.Float([1.0])
    material.valid = cuda.Bool([True])

    options = rd.DfrOptions()
    options.wavelength = 0.125
    options.k = 50.26548245743669
    options.seed = args.seed
    options.samples = args.dfr_samples
    options.max_order = 1
    options.direct_samples = args.dfr_samples
    options.keller_samples = 0
    options.suffix_samples = 0
    options.strategy_mask = rd.RAYD_DFR_DIRECT
    options.sample_sequence = rd.RAYD_DFR_HASH
    options.receiver_model = rd.RAYD_DFR_MATCHED_ISO
    options.collect_edge_use = True
    options.collect_debug_counts = True

    def call_kernel():
        rd.native_launch_audit_clear()
        return scene.accum_dfr_direct(states, grid, material, options, True)

    def materialize(result):
        dr.eval(result.power, result.direct_count, result.vis_rejects, result.edge_uses)

    measured = measure(call_kernel, materialize, args.repeats, args.warmup)
    result = measured.pop("last_value")
    audit = rd.native_launch_audit()
    return {
        "backend": "rayd",
        "workload": "diffraction",
        "tier": 3,
        "scene": "synthetic_single_edge_state",
        "state_count": args.state_count,
        "samples": args.dfr_samples,
        "grid_cell_count": args.grid_side * args.grid_side,
        "direct_count": int(result.direct_count[0]),
        "vis_rejects": int(result.vis_rejects[0]),
        "edge_uses": int(result.edge_uses[0]),
        "power_sum": float(dr.sum(result.power)[0]),
        "native_audit": audit,
        "timing": measured,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RayD native multipath kernels against Sionna RT PathSolver tiers."
    )
    parser.add_argument(
        "--backend",
        choices=[
            "sionna",
            "sionna_rt",
            "mitsuba_minimal",
            "rayd_minimal",
            "mitsuba_path",
            "rayd_path",
            "rayd",
            "both",
            "all",
        ],
        default="both",
    )
    parser.add_argument(
        "--workload",
        choices=[
            "los",
            "reflection",
            "diffraction",
            "reflection_trace",
            "diffraction_export",
            "both",
            "path_scaling",
            "all",
        ],
        default="both",
    )
    parser.add_argument("--sionna-root", type=Path, default=DEFAULT_SIONNA_ROOT)
    parser.add_argument("--loop-mode", choices=["symbolic", "evaluated"], default="symbolic")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rx-side", type=int, default=4)
    parser.add_argument("--samples-per-src", type=int, default=10000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=10000)
    parser.add_argument("--max-depth", "--max-bounces", dest="max_bounces", type=int, default=1)
    parser.add_argument(
        "--mitsuba-ray-api",
        choices=["preliminary", "surface"],
        default="preliminary",
        help="Mitsuba reflection path API: preliminary uses ray_intersect_preliminary + compute_surface_interaction; surface uses ray_intersect.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-los", action="store_true")
    parser.add_argument("--edge-diffraction", action="store_true")
    parser.add_argument("--ray-count", type=int, default=65536)
    parser.add_argument("--state-count", type=int, default=1024)
    parser.add_argument("--dfr-samples", type=int, default=1024)
    parser.add_argument("--grid-side", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _prepare_imports(args.sionna_root)
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono_polarized")

    results: list[dict[str, Any]] = []
    workloads = ["reflection", "diffraction"] if args.workload == "both" else (
        ["reflection_trace", "diffraction_export"] if args.workload == "path_scaling" else
        ["los", "reflection", "diffraction"] if args.workload == "all" else [args.workload]
    )
    backends = {
        "both": ["sionna", "rayd"],
        "all": ["sionna", "mitsuba_minimal", "rayd_minimal", "rayd"],
        "sionna_rt": ["sionna"],
    }.get(args.backend, [args.backend])

    if "sionna" in backends:
        for workload in workloads:
            results.append(run_sionna(args, workload))

    if "mitsuba_minimal" in backends:
        for workload in workloads:
            results.append(run_mitsuba_minimal(args, workload))

    if "rayd_minimal" in backends:
        for workload in workloads:
            results.append(run_rayd_minimal(args, workload))

    if "mitsuba_path" in backends:
        for workload in workloads:
            results.append(run_mitsuba_path(args, workload))

    if "rayd_path" in backends:
        for workload in workloads:
            results.append(run_rayd_path(args, workload))

    if "rayd" in backends:
        if "reflection" in workloads:
            results.append(run_rayd_reflection(args))
        if "diffraction" in workloads:
            results.append(run_rayd_diffraction(args))

    payload = {
        "config": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "sionna_root": str(args.sionna_root),
            "mitsuba_variant": mi.variant(),
            "python": sys.executable,
            "versions": {
                "drjit": package_version("drjit"),
                "mitsuba": package_version("mitsuba"),
                "sionna-rt": package_version("sionna-rt"),
                "rayd": package_version("rayd"),
            },
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()
