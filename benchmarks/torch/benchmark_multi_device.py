# Copyright Xingyu Chen.
# Benchmarks benchmark multi device.

"""Multi-GPU throughput benchmark for `Scene(devices=[...])` (plan Phase 2d).

Run it from the repository root, the same way as `benchmark_torch_native.py`:

```bash
python -m benchmarks.torch.benchmark_multi_device            # both configs
python -m benchmarks.torch.benchmark_multi_device --config light
```

It runs on one GPU as well as on several: with fewer than two visible CUDA
devices every multi-device column is simply absent and the single-device
numbers are still printed, so the same command is a baseline collector on a
one-GPU machine and a scaling report on a two-GPU one.

What it measures, per configuration:

- `intersect` and `trace_reflections` -- the `per_ray` class, sharded along the
  batch axis -- at the shipped defaults, and again at the weights
  `Scene.calibrate_devices()` chose for that operation.
- `accum_dfr_direct` -- the `grid_reduce` class, sharded along the Monte-Carlo
  lane axis -- at the configuration's sample count.
- a chunked streaming case: the same `intersect` under an `offload` hook, which
  is how a batch whose output does not fit one GPU is executed.

The calibrated rows are decisions, not measurements, so each one is printed
with the margin between its best sharded rung and running on the master alone.
A near-crossover operation (its per-row bytes and its per-row compute within a
few percent) decides differently between runs, and the benchmark says so
instead of letting one run's `1.00x` read as a guarantee.

Every chunked row also gets an overlap line: one chunk's round trip at the link
speed this run measured, the cost with no overlap, the cost with perfect
overlap, and where the measurement landed between them. The executor does not
expose per-chunk timestamps, so that is arithmetic over the chunk plan rather
than an instrumented breakdown, and the residual it reports is honestly the
pipeline's fixed cost and its unhidden copies together.

Two configurations are shipped, because the interesting result is the
difference between them:

- `light` -- the 192-vertex grid of `benchmark_torch_native.py`, one bounce.
  A ray costs almost nothing to trace and 24 bytes to send plus up to 76 bytes
  to bring back, so this configuration is *transfer-bound*: it is what the
  small-batch floor exists for, and -- because its sharded and master-only
  rungs are nearly tied -- it is also where `calibrate_devices()` is least
  able to answer reliably, which the near-crossover flag reports.
- `compute` -- a 2.1M-triangle cloud with incoherent rays and four bounces, the
  configuration the pipelined dispatch was validated on. A ray costs enough
  traversal to pay for its own bytes, and this is where two devices are worth
  having. Triangle *count* is not what makes a scene compute-bound: a plane
  grid of the same 2.1M triangles is answered in 0.6 ns per ray, a cloud of
  them in 4.5 ns.

Timing is interleaved (every variant runs once per round, in the same round)
and reduced with a minimum over the rounds, so a neighbouring tenant's spike on
a shared machine inflates one sample rather than one variant. Nothing here
changes a result: the benchmark only reads the public API plus the two debug
attributes the executor already exposes (`_multi.last_dispatch` and
`_multi.last_chunk_plan`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import rayd.torch as rt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One benchmark point: a scene shape, a batch, and a sample count."""

    name: str
    kind: str
    size: int
    extent: float
    rays: int
    max_bounces: int
    accum_samples: int
    chunk_rays: int
    note: str

    @property
    def triangles(self) -> int:
        return 2 * self.size * self.size if self.kind == "grid" else self.size


LIGHT = Config(
    name="light",
    kind="grid",
    # 191 cells is the 192x192 vertex grid of `benchmark_torch_native.py`.
    size=191,
    extent=2.0,
    rays=1 << 22,
    max_bounces=1,
    accum_samples=1 << 13,
    chunk_rays=1 << 19,
    note="192-vertex grid (72,962 triangles), 1 bounce, small sample count -- transfer-bound",
)

COMPUTE = Config(
    name="compute",
    kind="cloud",
    # The pipelined dispatch's validation scene: a 2.1M-triangle cloud, whose
    # BVH a ray actually has to descend, rather than a plane it hits at once.
    size=1 << 21,
    extent=0.005,
    rays=1 << 22,
    max_bounces=4,
    accum_samples=1 << 26,
    chunk_rays=1 << 19,
    note="2.1M-triangle cloud, incoherent rays, 4 bounces, 67M samples -- compute-bound",
)

CONFIGS = {config.name: config for config in (LIGHT, COMPUTE)}


def build_mesh(config: Config, device: torch.device):
    """This configuration's geometry, bit-identical on every device."""
    if config.kind == "grid":
        return grid_mesh(device, config.size, config.extent)
    return triangle_cloud(device, config.size, config.extent)


def build_rays(config: Config, device: torch.device, count: int) -> rt.Ray:
    """This configuration's batch: plane-facing for a grid, volumetric for a cloud."""
    if config.kind == "grid":
        return plane_rays(device, count)
    return volume_rays(device, count)


def grid_mesh(device: torch.device, cells: int, span: float = 2.0):
    """Deterministic `z = 0` triangle grid; identical bits on every device."""
    axis = torch.linspace(-0.5 * span, 0.5 * span, cells + 1, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    flat_x = x.reshape(-1)
    vertices = torch.stack((flat_x, y.reshape(-1), torch.zeros_like(flat_x)), dim=1)
    index = torch.arange((cells + 1) * (cells + 1), dtype=torch.int32).reshape(cells + 1, cells + 1)
    a = index[:-1, :-1].reshape(-1)
    b = index[:-1, 1:].reshape(-1)
    c = index[1:, :-1].reshape(-1)
    d = index[1:, 1:].reshape(-1)
    faces = torch.cat((torch.stack((a, b, c), dim=1), torch.stack((b, d, c), dim=1)))
    return vertices.contiguous().to(device), faces.contiguous().to(device)


def triangle_cloud(device: torch.device, triangles: int, edge: float, span: float = 2.0):
    """`triangles` small independent triangles scattered through a cube.

    This is the compute-bound geometry. A plane grid of the same triangle count
    is *not* compute-bound however many triangles it has: its BVH is flat, and
    a ray from outside hits at the first leaf it touches. A cloud makes a ray
    descend a deep, overlapping BVH and test many leaves, which at this size
    costs 4-5 ns per ray -- about eight times a grid of the same triangle
    count, and enough that a ray's compute outweighs its bytes.
    """
    generator = torch.Generator().manual_seed(20260727)
    centers = (torch.rand((triangles, 3), generator=generator) - 0.5) * span
    offsets = (torch.rand((triangles, 3, 3), generator=generator) * 2.0 - 1.0) * edge
    vertices = (centers[:, None, :] + offsets).reshape(-1, 3).contiguous()
    faces = torch.arange(triangles * 3, dtype=torch.int32).reshape(-1, 3).contiguous()
    return vertices.to(device), faces.to(device)


def plane_rays(device: torch.device, count: int) -> rt.Ray:
    """`count` rays from below the grid, aimed at it from scattered directions."""
    generator = torch.Generator().manual_seed(20260727)
    origins = torch.rand((count, 3), generator=generator) * 1.8 - 0.9
    origins[:, 2] = -1.0
    directions = torch.randn((count, 3), generator=generator)
    directions[:, 2] = directions[:, 2].abs() + 0.25
    directions = directions / directions.norm(dim=1, keepdim=True)
    return rt.Ray(origins.contiguous().to(device), directions.contiguous().to(device))


def volume_rays(device: torch.device, count: int) -> rt.Ray:
    """`count` incoherent rays: random origins in the cloud, random directions.

    Incoherent on purpose -- neighbouring lanes take different BVH paths, which
    is what makes traversal, rather than the cache, the cost of a ray.
    """
    generator = torch.Generator().manual_seed(20260728)
    origins = torch.rand((count, 3), generator=generator) * 2.0 - 1.0
    directions = torch.randn((count, 3), generator=generator)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return rt.Ray(origins.contiguous().to(device), directions.contiguous().to(device))


def accum_fixture(device: torch.device, **kwargs):
    """The order-1 diffraction accumulation fixture: one triangle, two states."""

    def f32(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    vertices = f32([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    faces = i32([[0, 1, 2]])
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()

    states = rt.DfrStates(
        edge_index=i32([0, 1]),
        edge_pos=f32([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        edge_dir=f32([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_t_min=f32([-1.0, -1.0]),
        edge_t_max=f32([1.0, 1.0]),
        n0=f32([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        n1=f32([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        prim0=i32([0, 0]),
        prim1=i32([0, 0]),
        exterior_angle=f32([torch.pi, torch.pi]),
        src=f32([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25]]),
        src_power=f32([1.0, 1.0]),
    )
    material = rt.DfrMaterial(
        eta_r=torch.ones((1,), device=device),
        sigma=torch.zeros((1,), device=device),
        mu_r=torch.ones((1,), device=device),
        gain=torch.ones((1,), device=device),
        valid=torch.ones((1,), device=device, dtype=torch.bool),
    )
    grid = rt.DfrGrid(axis=2, position=0.0, resolution0=4, resolution1=4)
    return scene, states, material, grid


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def sync(devices) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def interleaved_min_ms(
    variants: dict[str, Callable[[], object]], devices, *, warmup: int, repeat: int
) -> dict[str, float]:
    """Run every variant once per round; keep each variant's fastest round.

    Interleaving is the point: on a shared machine the alternative -- all of
    one variant, then all of the other -- hands whichever variant ran during a
    neighbour's quiet minute a result that is about the neighbour.
    """
    for _ in range(max(warmup, 0)):
        for run in variants.values():
            run()
        sync(devices)
    best = {name: math.inf for name in variants}
    for _ in range(max(repeat, 1)):
        for name, run in variants.items():
            sync(devices)
            start = time.perf_counter()
            run()
            sync(devices)
            best[name] = min(best[name], (time.perf_counter() - start) * 1e3)
    return best


def device_to_device_gbps(src: torch.device, dst: torch.device, *, mib: int = 256):
    """One-direction D2D copy bandwidth, which is what a scatter or gather pays.

    Reported next to the byte counts below so the transfer-bound/compute-bound
    crossover in the documentation is arithmetic rather than an assertion.
    """
    count = mib * 1024 * 1024 // 4
    source = torch.empty((count,), dtype=torch.float32, device=src)
    target = torch.empty((count,), dtype=torch.float32, device=dst)
    best = math.inf
    for index in range(6):
        sync((src, dst))
        start = time.perf_counter()
        target.copy_(source)
        sync((src, dst))
        elapsed = time.perf_counter() - start
        if index:
            best = min(best, elapsed)
    del source, target
    return (mib / 1024.0) / best


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def chunk_record(scene, dispatch: str | None = None) -> dict:
    """The executor's own account of what it just ran: mode, chunks, sizes.

    `last_dispatch` is a `per_ray` notion (it names which of the four routing
    decisions the batch axis took); the lane-windowed `grid_reduce` path has
    only one route, so its caller passes the name in.
    """
    layer = getattr(scene, "_multi", None)
    if layer is None:
        return {"dispatch": "single-device"}
    plan = layer.last_chunk_plan
    record = {"dispatch": layer.last_dispatch or dispatch}
    if record["dispatch"] == "master":
        # A master-only dispatch is one single-device launch and leaves the
        # previous call's plan behind; reporting it would invent chunks.
        return record
    if plan is not None:
        record.update(
            {
                "chunk_source": plan.source,
                "chunk_rows": plan.chunk_rays,
                "chunk_count": plan.chunk_count,
                "row_bytes": plan.row_bytes,
                "measured_row_bytes": plan.measured_row_bytes,
            }
        )
    return record


# What a sharded row costs to send: `Ray` is two float3 columns, and it is the
# input of both `per_ray` operations this benchmark shards.
_RAY_ROW_BYTES = 24


# The band in which the ladder's shard/no-shard answer is not a property of
# the workload. It is the executor's own refinement tolerance (3%) plus the
# run-to-run spread of a calibration probe on this class of machine (~2%): a
# decision inside it was measured to be a tie and resolved by the tie-break.
_CROSSOVER_BAND_PCT = 5.0


def calibration_margin(record) -> dict:
    """How far the shard/no-shard decision was from flipping.

    The refinement ladder keeps the *largest* remote share within
    `_REFINE_TOLERANCE` (3%) of the fastest rung, so a split that calibration
    itself timed as slower than the master alone is kept whenever it lost by
    less than that. The number that says whether the run's answer will hold is
    therefore not the chosen rung's time but the gap between the best *sharded*
    rung and the master-only rung:

    - `margin_pct` well above +3: sharding is a measurement, and it will
      reproduce.
    - `margin_pct` well below 0: the master alone is a measurement.
    - `|margin_pct|` inside `_CROSSOVER_BAND_PCT`: near-crossover. The two rungs
      swap places between runs, the ladder resolves the tie towards the split,
      and the kept split then loses at run time by more than it lost during
      calibration. A single run's weights are not a property of the workload
      here; pin them.
    """
    seconds = list(record.candidate_seconds)
    weights = list(record.weights)
    # `refine=False` records no ladder at all, and a future ladder that reports
    # its rungs and its times separately must keep them the same length before
    # this can pair them up.
    if not seconds or len(record.candidates) != len(seconds):
        return {}
    chosen = next(
        (index for index, candidate in enumerate(record.candidates) if list(candidate) == weights),
        seconds.index(min(seconds)),
    )
    chosen_ms = seconds[chosen] * 1e3
    master_only_ms = seconds[-1] * 1e3
    sharded_rungs = [
        value * 1e3
        for index, value in enumerate(seconds)
        if any(share != 0.0 for share in list(record.candidates[index])[1:])
    ]
    best_sharded_ms = min(sharded_rungs) if sharded_rungs else float("nan")
    sharded = any(value != 0.0 for value in weights[1:])
    margin_pct = (
        (master_only_ms - best_sharded_ms) / master_only_ms * 100.0 if master_only_ms > 0.0 and sharded_rungs else 0.0
    )
    return {
        "calibration_chosen_ms": chosen_ms,
        "calibration_master_only_ms": master_only_ms,
        "calibration_best_sharded_ms": best_sharded_ms,
        "calibration_margin_pct": margin_pct,
        "calibration_sharded": sharded,
        # True when a split was kept although calibration timed the master
        # alone as at least as fast: the tolerance, not a measurement, chose.
        "calibration_kept_on_tolerance": bool(sharded and margin_pct <= 0.0),
        # True when the decision sits inside the band where it flips between
        # runs, whichever way this run happened to fall.
        "calibration_near_crossover": bool(abs(margin_pct) <= _CROSSOVER_BAND_PCT),
    }


def compare(single_ms: float, multi_ms: float | None) -> dict:
    record = {"single_ms": single_ms}
    if multi_ms is None:
        return record
    record["multi_ms"] = multi_ms
    record["speedup"] = single_ms / multi_ms if multi_ms > 0.0 else 0.0
    # Two devices can at best halve the compute; whatever is left is the
    # pipeline's own cost (first scatter, last gather, per-chunk host time).
    record["overhead_ms"] = multi_ms - single_ms / 2.0
    return record


def overlap_breakdown(record: dict, total_rows: int, gbps: float | None) -> dict:
    """Where a chunked run sits between "no overlap" and "perfect overlap".

    The executor does not expose per-chunk timestamps, so this is arithmetic
    over what it does expose (the chunk plan, the measured row size) plus the
    link bandwidth this run measured. Two bounds frame the measurement:

    - `serial_ms` -- half the single-device compute plus *every* remote row's
      round trip (its ray in, its result out), i.e. what a chunked run costs if
      nothing is hidden.
    - `pipelined_ms` -- half the compute plus *one* chunk's round trip, i.e.
      the best a double-buffered pipeline can do, since the first chunk's
      scatter and the last chunk's gather have nothing to hide behind.

    The gap between `pipelined_ms` and the measured time is the pipeline's
    fixed cost *plus* whatever copy time did not overlap. Those two are not
    separable from outside the executor, so this deliberately does not report
    an "overlap fraction": it reports the bounds and leaves the residual named
    for what it is.
    """
    # `per_ray` only. A `grid_reduce` chunk is a window of Monte-Carlo *lanes*,
    # and what comes back per device is one grid, not one row per lane -- so
    # neither "bytes per row" nor "remote rows" means anything for it, and
    # computing them anyway would print a confidently wrong number.
    if record.get("dispatch") not in ("pipelined", "chunked", "sharded"):
        return {}
    multi_ms = record.get("multi_ms")
    rows = record.get("chunk_rows")
    chunks = record.get("chunk_count")
    row_bytes = record.get("measured_row_bytes") or record.get("row_bytes")
    if not (gbps and multi_ms and rows and chunks and row_bytes and total_rows):
        return {}
    # Half the batch is remote in the balanced case this benchmark runs; the
    # master's own rows are copied within the device and are not on the link.
    remote_rows = total_rows / 2.0
    per_byte_ms = 1e3 / (float(gbps) * 1e9)
    trip_bytes = _RAY_ROW_BYTES + row_bytes
    chunk_trip_ms = rows * trip_bytes * per_byte_ms
    compute_ms = record["single_ms"] / 2.0
    serial_ms = compute_ms + remote_rows * trip_bytes * per_byte_ms
    pipelined_ms = compute_ms + chunk_trip_ms
    return {
        "overlap_row_trip_bytes": trip_bytes,
        "overlap_chunk_trip_ms": chunk_trip_ms,
        "overlap_serial_bound_ms": serial_ms,
        "overlap_pipelined_bound_ms": pipelined_ms,
        "overlap_residual_ms": multi_ms - pipelined_ms,
    }


def agreement(single, multi) -> float:
    """Fraction of rows where the two runs agree bitwise; a cheap sanity check."""
    left = single.view(torch.int32) if single.dtype == torch.float32 else single
    right = multi.view(torch.int32) if multi.dtype == torch.float32 else multi
    return float((left == right.to(left.device)).float().mean().item())


def measure_per_ray(config: Config, devices, args) -> dict:
    """`intersect` and `trace_reflections`, single vs multi, then calibrated."""
    master = devices[0]
    vertices, faces = build_mesh(config, master)
    ray = build_rays(config, master, config.rays)

    single = rt.Scene()
    single.add_mesh(rt.Mesh(vertices, faces, edges_enabled=False))
    single.build()
    sync(devices[:1])

    multi = None
    if len(devices) > 1:
        multi = rt.Scene(devices=[d.index for d in devices])
        multi.add_mesh(rt.Mesh(vertices.clone(), faces, edges_enabled=False))
        multi.build()
        sync(devices)

    results: dict = {"triangles": config.triangles, "rays": config.rays, "max_bounces": config.max_bounces}

    def run_intersect(scene):
        return lambda: scene.intersect(ray, flags=rt.RayFlags.All).t

    def run_reflections(scene):
        return lambda: scene.trace_reflections(ray, max_bounces=config.max_bounces).t

    for name, build in (("intersect", run_intersect), ("trace_reflections", run_reflections)):
        variants = {"single": build(single)}
        if multi is not None:
            variants["multi"] = build(multi)
        timings = interleaved_min_ms(variants, devices, warmup=args.warmup, repeat=args.repeat)
        record = compare(timings["single"], timings.get("multi"))
        record["batch"] = f"{config.rays} rays"
        record["ns_per_ray_single"] = timings["single"] * 1e6 / config.rays
        if multi is not None:
            record.update(chunk_record(multi))
            reference = build(single)()
            sharded = build(multi)()
            record["bitwise_agreement"] = agreement(reference, sharded)
            del reference, sharded
        results[name] = record

    # Calibration: measured, then re-timed at the weights it chose. On a
    # transfer-bound configuration the answer that wins is a zero remote
    # weight, which the dispatcher runs as the single-device call it is -- but
    # calibration only gets there when the master-only rung is *measurably*
    # faster than every larger share, so `calibration_margin()` records how
    # close the decision was and the row is honest about being a decision.
    if multi is not None:
        for name, bounces, build in (
            ("intersect", 0, run_intersect),
            ("trace_reflections", config.max_bounces, run_reflections),
        ):
            # Calibrate at the batch size the workload actually uses. The
            # refinement stage times the real dispatch, and the dispatch's
            # fixed cost (one launch and one copy per output field per chunk)
            # is a different fraction of a 1M-row batch than of a 4M-row one:
            # a probe smaller than the workload can and does pick a split that
            # loses at the workload's size.
            record = multi.calibrate_devices(
                rays=args.calibration_rays or config.rays, max_bounces=bounces, repeats=args.calibration_repeats
            )
            timings = interleaved_min_ms(
                {"single": build(single), "multi": build(multi)}, devices, warmup=args.warmup, repeat=args.repeat
            )
            calibrated = compare(timings["single"], timings["multi"])
            calibrated["batch"] = f"{config.rays} rays"
            calibrated["weights"] = list(record.weights)
            calibrated["probe_seconds"] = list(record.seconds)
            calibrated["candidate_weights"] = [list(c) for c in record.candidates]
            calibrated["candidate_ms"] = [s * 1e3 for s in record.candidate_seconds]
            calibrated["describe"] = record.describe()
            calibrated.update(calibration_margin(record))
            calibrated.update(chunk_record(multi))
            results[name + "_calibrated"] = calibrated

    # The streaming case: the same batch, chunked, with the results handed to a
    # hook instead of being concatenated. It is a memory story, so the peak
    # allocation on the master is the number that matters next to the time.
    if multi is not None:
        results["intersect_offload"] = measure_offload(config, devices, single, ray, args)

    del single, multi, ray, vertices, faces
    torch.cuda.empty_cache()
    return results


def measure_offload(config: Config, devices, single, ray, args) -> dict:
    """`intersect` streamed through the `offload` hook, chunk by chunk."""
    consumed = {"rows": 0, "chunks": 0}

    def hook(start: int, chunk) -> None:
        # A realistic consumer: reduce the chunk and drop it, so no chunk's
        # output outlives the chunk. Keeping `chunk` here would defeat the hook.
        consumed["rows"] += int(chunk.t.shape[0])
        consumed["chunks"] += 1
        chunk.t.isfinite().sum()

    vertices, faces = build_mesh(config, devices[0])
    scene = rt.Scene(
        devices=[d.index for d in devices], options=rt.MultiDeviceOptions(chunk_rays=config.chunk_rays, offload=hook)
    )
    scene.add_mesh(rt.Mesh(vertices, faces, edges_enabled=False))
    scene.build()
    sync(devices)

    def streamed():
        consumed["rows"] = 0
        consumed["chunks"] = 0
        return scene.intersect(ray, flags=rt.RayFlags.All)

    def concatenated():
        return single.intersect(ray, flags=rt.RayFlags.All).t

    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    streamed()
    sync(devices)
    streamed_peaks = [
        {"device_index": int(device.index), "bytes": int(torch.cuda.max_memory_allocated(device))} for device in devices
    ]
    streamed_peak = streamed_peaks[0]["bytes"]
    torch.cuda.reset_peak_memory_stats(devices[0])
    concatenated()
    sync(devices)
    concatenated_peak = torch.cuda.max_memory_allocated(devices[0])

    timings = interleaved_min_ms(
        {"single": concatenated, "multi": streamed}, devices, warmup=args.warmup, repeat=args.repeat
    )
    record = compare(timings["single"], timings["multi"])
    record.update(chunk_record(scene))
    record["batch"] = f"{config.rays} rays"
    record["rows_offloaded"] = consumed["rows"]
    record["chunks_offloaded"] = consumed["chunks"]
    record["master_peak_bytes_streamed"] = streamed_peak
    record["master_peak_bytes_concatenated"] = concatenated_peak
    record["peak_memory"] = {
        "status": "measured",
        "master_device_index": int(devices[0].index),
        "streamed_bytes": int(streamed_peak),
        "concatenated_bytes": int(concatenated_peak),
        "per_device_streamed_bytes": streamed_peaks,
    }
    del scene, vertices, faces
    torch.cuda.empty_cache()
    return record


def measure_accum(config: Config, devices, args) -> dict:
    """`accum_dfr_direct`: sharded over Monte-Carlo lanes, merged on the master."""
    master = devices[0]
    single, states, material, grid = accum_fixture(master)
    multi = None
    if len(devices) > 1:
        multi, _s, _m, _g = accum_fixture(master, devices=[d.index for d in devices])

    def run(scene):
        return lambda: (
            scene.accum_dfr_direct(
                states=states,
                grid=grid,
                material=material,
                wavelength=1.0,
                direct_samples=config.accum_samples,
                seed=17,
            ).power
        )

    variants = {"single": run(single)}
    if multi is not None:
        variants["multi"] = run(multi)
    timings = interleaved_min_ms(variants, devices, warmup=args.warmup, repeat=args.repeat)
    record = compare(timings["single"], timings.get("multi"))
    record["direct_samples"] = config.accum_samples
    record["batch"] = f"{config.accum_samples} samples"
    if multi is not None:
        record.update(chunk_record(multi, "lane-sharded"))
        reference = run(single)()
        merged = run(multi)()
        # A merged grid is the single-device grid up to float32 summation
        # order, so this is a relative deviation, not an agreement fraction.
        denominator = float(reference.abs().sum().item())
        record["relative_grid_deviation"] = (
            float((merged.to(master) - reference).abs().sum().item()) / denominator if denominator > 0.0 else 0.0
        )
        del reference, merged
    del single, multi
    torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_ROWS = (
    ("intersect", "intersect"),
    ("trace_reflections", "trace_reflections"),
    ("intersect_calibrated", "intersect (calibrated)"),
    ("trace_reflections_calibrated", "trace_reflections (calibrated)"),
    ("accum_dfr_direct", "accum_dfr_direct"),
    ("intersect_offload", "intersect (chunked + offload)"),
)


def markdown(results: dict) -> str:
    """The table this benchmark exists to produce, ready to paste into the docs."""
    count = len(results["machine"]["devices"])
    # With one visible device there is no second column to fill, and calling it
    # "1 GPUs" next to the baseline reads as two measurements of the same run.
    multi_label = f"{count} GPUs" if count > 1 else "multi (n/a)"
    lines = [
        f"| Configuration | Operation | Batch | 1 GPU | {multi_label} | speedup | dispatch | chunks | weights |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for name, config in results["configs"].items():
        for key, label in _ROWS:
            record = config.get(key)
            if record is None:
                continue
            multi = record.get("multi_ms")
            weights = record.get("weights")
            lines.append(
                "| {config} | {label} | {batch} | {single:.2f} ms | {multi} | "
                "{speedup} | {dispatch} | {chunks} | {weights} |".format(
                    config=name,
                    label=label,
                    batch=record.get("batch", "--"),
                    single=record["single_ms"],
                    multi="--" if multi is None else f"{multi:.2f} ms",
                    speedup="--" if multi is None else f"{record['speedup']:.2f}x",
                    dispatch=record.get("dispatch", "--"),
                    chunks=record.get("chunk_count", "--"),
                    weights="--" if weights is None else ", ".join(f"{value:.3f}" for value in weights),
                )
            )
    notes = overlap_notes(results) + calibration_notes(results)
    if notes:
        lines.append("")
        lines.extend(notes)
    return "\n".join(lines)


def overlap_notes(results: dict) -> list[str]:
    """One line per chunked row: the two bounds and where the run landed."""
    lines: list[str] = []
    for name, config in results["configs"].items():
        for key, label in _ROWS:
            record = config.get(key)
            if record is None or "overlap_pipelined_bound_ms" not in record:
                continue
            lines.append(
                "- overlap, {config} {label}: {chunks} chunks, {bytes} B per "
                "row round trip = {trip:.2f} ms per chunk; no overlap would "
                "cost {serial:.2f} ms, perfect overlap {pipelined:.2f} ms, "
                "measured {measured:.2f} ms (residual {residual:+.2f} ms = "
                "fixed cost + unhidden copies)".format(
                    config=name,
                    label=label,
                    chunks=record["chunk_count"],
                    bytes=record["overlap_row_trip_bytes"],
                    trip=record["overlap_chunk_trip_ms"],
                    serial=record["overlap_serial_bound_ms"],
                    pipelined=record["overlap_pipelined_bound_ms"],
                    measured=record["multi_ms"],
                    residual=record["overlap_residual_ms"],
                )
            )
    return lines


def calibration_notes(results: dict) -> list[str]:
    """One line per calibrated row saying how close its decision was.

    The calibrated rows are the only ones in the table that are a *decision*
    rather than a measurement, and on a near-crossover operation the decision
    is not stable between runs. Printing the ladder's own margin next to the
    re-timed speedup is what keeps a reader from copying a 1.00x into a
    guarantee: a row whose margin is a fraction of a percent will not be that
    row on the next run.
    """
    lines: list[str] = []
    for name, config in results["configs"].items():
        for key, label in _ROWS:
            record = config.get(key)
            if record is None or "calibration_margin_pct" not in record:
                continue
            verdict = (
                "kept a remote share on tolerance alone"
                if record["calibration_kept_on_tolerance"]
                else ("sharded with a measured margin" if record["calibration_sharded"] else "chose the master alone")
            )
            flags = ""
            if record["calibration_near_crossover"]:
                flags += " -- NEAR-CROSSOVER, decision flips between runs"
            if record.get("speedup", 1.0) < 0.98:
                flags += " -- SLOWER THAN ONE GPU"
            lines.append(
                "- calibration, {config} {label}: {verdict}; best sharded rung "
                "{sharded:.2f} ms vs master-only {master:.2f} ms "
                "({margin:+.1f}% for sharding), re-timed at {speedup:.2f}x{flags}".format(
                    config=name,
                    label=label,
                    verdict=verdict,
                    sharded=record["calibration_best_sharded_ms"],
                    master=record["calibration_master_only_ms"],
                    margin=record["calibration_margin_pct"],
                    speedup=record.get("speedup", float("nan")),
                    flags=flags,
                )
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--devices", default=None, help="comma-separated CUDA indices")
    parser.add_argument("--rays", type=int, default=None, help="override the batch size")
    parser.add_argument("--accum-samples", type=int, default=None, help="override the accum_dfr_direct sample count")
    parser.add_argument(
        "--accum-only", action="store_true", help="skip the per_ray operations (for sweeping the sample count cheaply)"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument(
        "--calibration-rays",
        type=int,
        default=0,
        help="probe size for calibrate_devices(); 0 means the batch size itself",
    )
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None, help="write the schema-versioned benchmark record here")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmark_multi_device needs at least one CUDA device.")

    if args.devices:
        indices = [int(value) for value in args.devices.split(",") if value != ""]
    else:
        indices = list(range(torch.cuda.device_count()))
    devices = [torch.device("cuda", index) for index in indices]
    torch.cuda.set_device(devices[0])

    machine = {
        "device_count": len(devices),
        "devices": [
            {
                "index": device.index,
                "name": torch.cuda.get_device_name(device),
                "total_bytes": torch.cuda.get_device_properties(device).total_memory,
                "compute_capability": [
                    int(torch.cuda.get_device_properties(device).major),
                    int(torch.cuda.get_device_properties(device).minor),
                ],
            }
            for device in devices
        ],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    peer_pairs = [
        {
            "source": int(source.index),
            "destination": int(destination.index),
            "can_access": bool(torch.cuda.can_device_access_peer(source, destination)),
        }
        for source in devices
        for destination in devices
        if source != destination
    ]
    machine["peer_access"] = {
        "status": "measured" if len(devices) > 1 else "not_applicable",
        "all_pairs_accessible": bool(peer_pairs) and all(pair["can_access"] for pair in peer_pairs),
        "pairs": peer_pairs,
    }
    if len(devices) > 1:
        machine["d2d_gbps"] = device_to_device_gbps(devices[0], devices[1])
    else:
        print("Only one CUDA device is visible: reporting single-device times with no multi-device column.")

    selected = list(CONFIGS.values()) if args.config == "all" else [CONFIGS[args.config]]
    results = {
        "schema_version": 1,
        "benchmark": "rayd_multi_device",
        "provenance": {
            "kind": "live_measurement",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "parameters": {
            "selected_configs": [config.name for config in selected],
            "warmup": args.warmup,
            "repeat": args.repeat,
            "calibration_rays": args.calibration_rays,
            "calibration_repeats": args.calibration_repeats,
            "accum_only": args.accum_only,
        },
        "machine": machine,
        "configs": {},
    }
    for config in selected:
        overrides = {}
        if args.rays is not None:
            overrides["rays"] = args.rays
        if args.accum_samples is not None:
            overrides["accum_samples"] = args.accum_samples
        if overrides:
            config = Config(**{**config.__dict__, **overrides})
        print(f"[{config.name}] {config.note} ...", flush=True)
        record = {} if args.accum_only else measure_per_ray(config, devices, args)
        record["accum_dfr_direct"] = measure_accum(config, devices, args)
        record["note"] = config.note
        for key, _label in _ROWS:
            row = record.get(key)
            if isinstance(row, dict):
                row.update(overlap_breakdown(row, record.get("rays", 0), machine.get("d2d_gbps")))
        results["configs"][config.name] = record

    print(json.dumps(results, indent=2, sort_keys=True))
    print()
    print(markdown(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
