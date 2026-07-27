"""Private replicated multi-device orchestration for `Scene(devices=[...])`.

This is Phases 2a, 2b, 2d and the grid-reduce half of 2c of
`docs/dev/multi_gpu_plan.md`: decision D1 (one full
scene replica per device, work sharded along the batch axis), D4 (replica
vertices are `master.to(device_k)`, so torch autograd reduces every replica
gradient back onto the master leaf), D7 (a large batch is executed as a stream
of chunks per device), D8 (no parallel public surface -- `Scene` gains
`devices=` and `MultiDeviceOptions`, and everything else stays where it was)
and D9 (a `Scene` that asks for neither several devices nor chunking never
reaches this module at all: `Scene` imports it only when `devices=` is passed,
and it still declines to orchestrate a one-device scene that wants nothing
from the chunked executor).

Two of the plan's shardability classes are wrapped here. The `per_ray`
operations shard the batch axis and their sharded result is field-for-field the
single-device result. The `grid_reduce` diffraction accumulation operations
shard the Monte-Carlo *sample* axis instead (see below), and their merged grid
is the single-device grid up to float32 summation order. Everything else --
`trace_dfr_paths`, whose exporter row placement is batch-coupled, and
`accum_dfr_coherent_direct`, which has no lane window to shard -- still raises
`NotImplementedError`, because it needs the explicit per-shard contract of D6.

Execution is deliberately single-threaded: two host threads issuing RayD Torch
ops concurrently can deadlock in the current native layer (see the comment on
`_warmup._DEVICE_WORK_LOCK`). Overlap comes from streams instead of threads.

Pipelined dispatch (Phase 2d)
-----------------------------

Torch runs a cross-device copy on the *source* device's current stream and
barriers it against the *destination* device's current stream. Left alone,
that puts both the scatter to the other devices and the gather back on the
master's own stream, in front of the master's compute: measured on this
repository's 2x RTX A6000 (NV4, 52.6 GB/s per direction, 101 GB/s both ways at
once), a 4M-ray `intersect` landed at 0.21x of one device and a 4M-ray
`trace_reflections` at 0.48x -- the layer was correct and slower than not
having it.

So a multi-device `per_ray` operation now runs through the chunked executor by
default, with a chunk size chosen for overlap rather than for a memory budget,
and every copy on a stream of its own: one per device *pair direction*, plus an
otherwise unused stream on each destination so the copy's two-way barrier never
lands on the caller's stream or on a compute stream. The master's shard is one
launch -- it has no copy to overlap -- and every other shard is cut into
`pipeline_chunks_per_device` chunks, so chunk `k`'s scatter and chunk `k-1`'s
gather run while chunk `k` computes. The operation's whole output is allocated
once on the master and each chunk copies its rows straight into it, which is
what keeps the host (one native launch plus one copy per output field per
chunk, ~0.3 ms whatever the chunk's size) from becoming the bound.

What that buys, on the same machine, at default options, min-of-9 and
interleaved against the single-device run (4M rays each):

| Configuration                                  |  1 GPU |  2 GPUs | ratio |
| ---------------------------------------------- | -----: | ------: | ----: |
| `intersect`, 2M-triangle cloud, incoherent     | 25.3 ms | 14.8 ms | 1.71x |
| `trace_reflections`, 4 bounces, 3.1M triangles | 18.2 ms | 10.4 ms | 1.75x |
| `intersect`, 192-grid mesh (transfer-bound)    |  1.18 ms |  1.21 ms | 0.97x |

The first two were 1.42x and 1.52x on the Phase 2a path. The third was 0.21x
there and is 0.27x through the pipeline on its own.

That third row is the point of `calibrate()` and of `min_rays_per_device`, not
a failure of the pipeline: two devices can only beat one when a row's compute
costs more than its bytes cost to move. A full `Intersection` row is 76 bytes,
which is 1.44 ns of one NVLink direction, so an `intersect` cheaper than that
per ray is faster on the master alone however well the copies overlap.
Calibration measures exactly that and answers with a zero remote weight, which
the dispatcher runs as the single-device call it is -- which is what puts the
third row at parity rather than at 0.31x.

Where the remaining distance to 2x goes, for the first row: half the batch is
12.6 ms of compute per device, the pipeline finishes in 14.8 ms, and the 2.2 ms
of difference is the first chunk's scatter (500k rows x 24 B = 0.23 ms), the
last chunk's gather (500k x 76 B = 0.72 ms), the master's copy of its own 2M
rows into the output (~0.4 ms), and the interconnect and the traversal kernels
competing for the same memory system for the rest. The second row's gather is
36 B per row instead of 76, and it lands correspondingly closer.

Nothing in the pipeline changes a result: it is the same launches on the same
per-shard inputs, with the ordering expressed as events instead of as stream
serialization, so a pipelined result is bitwise the unpipelined one and a fixed
(devices, weights, chunking) reproduces itself.

Private streams mean every edge to the caller's streams has to be written down,
on *every* device rather than only on the master. A replica's own state is
mutated on its own device: `build()` ends with a stream-ordered acceleration
structure build, `sync()` enqueues the triangle GAS refit, the IAS rebuild and
the edge-GAS build on the scene's stream and returns without a host
synchronization, and `set_edge_mask` is the same shape. So the executor enters
by making each device's compute stream wait on an event recorded on *that*
device's current stream (as well as on the master's, where the inputs were
produced), and leaves by making each device's current stream wait on an event
recorded on our compute stream. Without the first edge a query issued straight
after `update_mesh_vertices()` + `sync()` traverses a half-rebuilt structure
and answers, silently, from partly stale geometry; without the second a
mutation can overwrite geometry a shard is still traversing. Both are covered
by `PipelinedStreamOrderingTests`. The copy streams need neither edge: they
touch caller and chunk tensors, whose cross-stream lifetime the caching
allocator already tracks through `record_stream`.

Chunked execution (Phase 2b)
----------------------------

`MultiDeviceOptions.chunk_rays`, `.tape_memory_budget_bytes` and `.offload`
ask the same executor for a chunk size that fits a memory budget instead of one
that overlaps well, and are honoured at every batch size -- a memory bound is a
contract, and the small-batch floor does not apply to it. Ordering is the
pipelined one described above: chunk `k`'s gather runs on that pair's copy
stream while chunk `k+1`'s compute already runs on the compute stream, tied by
events, never by a device or host synchronization. The master's stream is made
to wait on a chunk's gather event before anything reads it, which is the only
ordering a caller can observe.

Chunking is engaged for a one-device scene too (`Scene(devices=[d],
options=MultiDeviceOptions(chunk_rays=...))`): at extreme N the binding
constraint is tape and output memory rather than scene memory (D7), so
splitting one device's batch is the memory story even when there is nothing to
shard. A `Scene` that asks for chunking on neither axis keeps the untouched
single-device path.

With `offload` set, per-ray results are streamed rather than concatenated: the
hook is called once per chunk as `offload(chunk_start_row, chunk_result)` with
the chunk's fields already on the master device, and the operation itself
returns `None`. Without it, the chunks are concatenated on the master, which
is bitwise the unchunked result because every wrapped operation is `per_ray`.

Chunking for memory buys memory, not speed: every chunk is one more native
call, and a native call's host cost does not shrink with its batch (measured on
one RTX A6000, a 4M-ray 3-bounce `trace_reflections` costs 0.31 ms of host time
and 0.92 ms of wall time; split into 16 chunks the same batch costs 3.8 ms,
host bound). That is what `calibrate_chunk_size()` is for -- the right chunk is
the largest one the budget allows, never the smallest one that fits -- and it
is why the pipelined path counts its chunks per shard instead of asking for a
size.

Chunking reaches exactly the operations sharding reaches, for the same reason:
a chunk of a `grid_reduce` operation is a partial grid, and merging those needs
the per-shard semantics of Phase 2c.

Training at extreme N pairs chunking with a *per-chunk* backward: a chunked
forward keeps one tape per chunk, so the memory bound only holds if the caller
reduces and backpropagates each chunk inside the `offload` hook instead of
holding the whole batch's graph. That is ordinary gradient accumulation and is
exact for RayD geometry gradients, which land in `grad_vertices` by summation
(D4); only float32 summation order differs from the unchunked backward.

Diffraction accumulation (Phase 2c, `grid_reduce`)
--------------------------------------------------

`accum_dfr_direct` and `accum_dfr` have no batch axis to shard: their cost is
the Monte-Carlo lane space of `direct_samples + keller_samples +
suffix_samples` samples, and their output is a grid whose size is independent
of it. They are therefore sharded along the *sample* axis, using the
`lane_offset` / `lane_count` window of D5: device `k` runs the sub-window
`(begin_k, count_k)` of the window the caller asked for, the windows are
contiguous and disjoint and cover it exactly, and every launch keeps the
caller's `direct/keller/suffix` counts so a local lane still runs the global
lane the single launch would have run. The per-shard result is a partial grid,
never a slice of one.

Merge-layer semantics (D3/D6). Chunks accumulate into their device's running
partial grid, in ascending lane order, on that device; the per-device partials
are then moved to the master and summed in `devices` order. Both the float and
the integer counters are summed the same way, so the sample counts merge
exactly while the grids merge in float32: a merged grid equals the
single-launch grid only up to float32 summation order, and the deviation grows
with the number of shards and chunks, not with the sample count. Nothing here
re-associates a *within-launch* reduction, which stays exactly the atomics the
single-device kernel has always run (D3). A run with the same devices, weights,
chunk size and inputs merges in a fixed order and therefore reproduces itself
as exactly as one device does -- which for the order-1 direct path is bitwise
and for the atomics-reducing order-2 chain path is to within the last ULP.

Shard and chunk boundaries are aligned to the 32-lane warp, relative to the
caller's window. Grid accumulation aggregates a warp's contributions before
the atomic, and a partially filled warp already loses contributions on one
device (a plain `direct_samples=20` launch accumulates fewer samples than it
tapes -- a defect that predates sharding and is not corrected here), so an
unaligned split would drop a fraction of a warp per shard *in addition* to
that. Aligned windows give every shard and chunk the same warp partition the
single launch has, including its trailing partial warp, which is why the
merged grid reproduces it. A window narrower than one warp per device is legal
and simply leaves the leading devices idle.

Gradients follow the same route as the `per_ray` ops (D4): a shard's states
and material are `master.to(device_k)`, so every shard's backward reduces onto
the caller's master leaves, and the merged grid's backward runs one native
backward per (device, chunk) launch against that launch's own tape.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import torch

from .types import (
    AxialEdgeVisibility,
    DfrAccum,
    DfrMaterial,
    DfrStates,
    Intersection,
    NearestEdgesTopK,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    ReflEpcField,
    ReflectionChain,
    SegmentChainVisibility,
    SegmentPairVisibility,
    _ReducedIntersection,
)


# Everything the multi-device layer cannot do yet points at the phase that owns
# it, so a caller who hits the wall knows what would have to land first.
_PHASE_2C = "docs/dev/multi_gpu_plan.md Phase 2c"


@dataclass(frozen=True)
class MultiDeviceOptions:
    """Tuning knobs for `Scene(devices=[...])`; every field is optional.

    `weights` is the static shard split of decision D1: one non-negative
    weight per device, in `devices` order, defaulting to an equal split. It is
    a ratio, not a count, so `[9.0, 1.0]` and `[0.9, 0.1]` mean the same thing,
    and a zero weight is legal and simply leaves that device idle.

    `warm_up` pre-links each device's OptiX pipelines through
    `rayd.torch._warmup` while the scene is being built, so the first real
    launch does not pay `len(devices)` module JITs in a row.

    `chunk_rays`, `tape_memory_budget_bytes` and `offload` are the chunked
    executor of D7. `chunk_rays` is the number of batch rows per launch and
    wins over everything else; leaving it `None` and setting
    `tape_memory_budget_bytes` instead asks `calibrate_chunk_size()` for the
    largest chunk whose tape and outputs fit that budget. `offload` is called
    once per chunk as `offload(chunk_start_row, chunk_result)` with the chunk's
    fields already on the master device; setting it means per-ray results are
    streamed instead of concatenated and the operation returns `None`. Any of
    the three engages the chunked path, including on a one-device scene.

    `pipeline`, `pipeline_chunks_per_device` and `min_rays_per_device` are the
    throughput half of the executor (Phase 2d), and only ever apply to a scene
    with more than one device. `pipeline` (on by default) runs every `per_ray`
    operation through the double-buffered executor so scatter, compute and
    gather overlap; `pipeline=False` keeps the Phase 2a one-launch-per-shard
    path, which is the reference the pipelined results are compared against.
    `pipeline_chunks_per_device` is how many chunks the *widest* remote shard
    is cut into: more chunks hide more of the pipeline's first scatter and last
    gather, fewer chunks pay less per-launch host cost. The master's shard is
    always one launch, because it has no copy to overlap.
    `min_rays_per_device` is the small-batch floor: a batch with fewer than
    `min_rays_per_device * len(devices)` rows runs entirely on the master
    replica, bitwise as a single-device `Scene` would run it, because a shard
    that small cannot pay for its own copies. An explicit chunking knob is a
    memory contract and is honoured whatever the batch size.
    """

    weights: Sequence[float] | None = None
    warm_up: bool = True
    chunk_rays: int | None = None
    offload: Callable[[int, object], None] | None = None
    tape_memory_budget_bytes: int | None = None
    pipeline: bool = True
    pipeline_chunks_per_device: int = 4
    min_rays_per_device: int = 262144


@dataclass
class ChunkPlan:
    """What the chunked executor decided, and what it then saw.

    This is a debug surface, not API: it is what a test (or a caller sizing a
    budget) reads off `Scene._multi.last_chunk_plan` to check that a memory
    budget really did shrink the launches. `chunk_rays` is the effective size
    after clamping to the batch, `row_bytes` the estimate that drove the
    decision, and `measured_row_bytes` the output bytes per row the first
    retired chunk actually produced -- the estimate covers the tape too, so the
    two are only expected to agree in order of magnitude.

    The lane-windowed `grid_reduce` operations reuse it for their sample-axis
    chunking, where a row is a Monte-Carlo lane rather than a batch row, and
    where `measured_row_bytes` stays `None` because a chunk produces a whole
    grid rather than a slab of rows.
    """

    operation: str
    total_rows: int
    chunk_rays: int
    source: str
    row_bytes: int
    budget_bytes: int | None = None
    chunk_count: int = 0
    measured_row_bytes: float | None = None


@dataclass
class DeviceCalibration:
    """What `calibrate()` measured, and the weights it derived from it.

    `seconds` is the best of the timed runs per device, in `devices` order, for
    a probe of `rows` rows, and `throughput_weights` is the split those timings
    imply. `candidates` and `candidate_seconds` are the refinement stage's
    ladder of remote shares and what the real dispatch cost at each of them;
    `weights` is the rung it kept, and is what the scene has been using since
    the call returned. It is a readable record on purpose -- calibration is the
    one thing in this module that is allowed to depend on the machine's mood,
    so a caller (or a bug report) can see exactly which numbers produced the
    split.
    """

    operation: str
    rows: int
    devices: tuple[int, ...] = ()
    seconds: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    samples: tuple[tuple[float, ...], ...] = ()
    throughput_weights: tuple[float, ...] = ()
    candidates: tuple[tuple[float, ...], ...] = ()
    candidate_seconds: tuple[float, ...] = ()

    @property
    def rows_per_second(self) -> tuple[float, ...]:
        return tuple(self.rows / value if value > 0.0 else 0.0 for value in self.seconds)

    def describe(self) -> str:
        """A human-readable block, for logging: one line per device, then the ladder."""
        lines = [
            f"{self.operation} probe, {self.rows} rows",
            *(
                f"  cuda:{index}  {seconds * 1e3:8.3f} ms  "
                f"{self.rows / seconds / 1e6:8.2f} Mrow/s  weight {weight:.4f}"
                if seconds > 0.0
                else f"  cuda:{index}  (no timing)  weight {weight:.4f}"
                for index, seconds, weight in zip(
                    self.devices,
                    self.seconds,
                    self.throughput_weights or self.weights,
                )
            ),
        ]
        for weights, seconds in zip(self.candidates, self.candidate_seconds):
            chosen = "  <-- chosen" if weights == self.weights else ""
            split = ", ".join(f"{weight:.4f}" for weight in weights)
            lines.append(f"  dispatch [{split}]  {seconds * 1e3:8.3f} ms{chosen}")
        return "\n".join(lines)


# `docs/dev/multi_gpu_plan.md` D7 sizes the reflection tape at 40-50 bytes per
# ray per bounce; the executor takes the pessimistic end so a budget is a bound
# rather than a hope. The output row of a reflection chain is separate: one
# `valid` byte, one float `t`, one int `prim_id` and three floats of image
# source, per bounce.
_REFLECTION_TAPE_BYTES_PER_RAY_BOUNCE = 50
_REFLECTION_OUTPUT_BYTES_PER_RAY_BOUNCE = 21

# Per-row output sizes of the remaining wrapped operations, summed field by
# field from their result schemas (float32 and int32 are 4 bytes, bool is 1).
_INTERSECT_REDUCED_ROW_BYTES = 4
_INTERSECT_FULL_ROW_BYTES = 76
_NEAREST_POINT_EDGE_ROW_BYTES = 32
_NEAREST_RAY_EDGE_ROW_BYTES = 48
_NEAREST_EDGES_ROW_BYTES_PER_K = 46
_VISIBLE_ROW_BYTES = 1
_VISIBLE_PAIR_ROW_BYTES = 2
_VISIBLE_EDGE_ROW_BYTES = 1
_VISIBLE_CHAIN_ROW_BYTES = 9
_REFL_EPC_FIELD_ROW_BYTES = 13

# Diffraction accumulation's per-row cost is per Monte-Carlo lane, not per ray:
# the AD tape (1 + 4 + 4 + 4 + 4 bytes) plus the visibility scratch byte, or the
# no-AD staging pair (4 + 16 bytes) plus that same byte, whichever is larger.
_DFR_ACCUM_LANE_BYTES = 21

# Grid accumulation aggregates within a warp before its atomic, so every lane
# window this module cuts is a whole number of warps (see the module docstring).
_LANE_ALIGNMENT = 32

# The pipelined dispatch of Phase 2d. Both numbers are measured, not guessed;
# `docs/dev/multi_gpu_plan.md` Phase 2d and the module docstring record the
# sweep they come from (2x RTX A6000, NV4 at 52.6 GB/s per direction).
_PIPELINE_CHUNKS_PER_DEVICE = 4
_MIN_RAYS_PER_DEVICE = 262144

# The probe `calibrate()` runs when the caller does not supply one: a batch big
# enough to be dominated by the device rather than by the launch, small enough
# that calibrating a scene costs milliseconds.
_CALIBRATION_ROWS = 1 << 20

# What share of its throughput-implied weight each non-master device is offered
# in the refinement stage. The rungs are coarse on purpose -- the curve between
# them is flat compared with the difference between "shard it" and "do not" --
# and the last one is the master alone.
_REFINE_SHARES = (1.0, 0.5, 0.25, 0.1, 0.0)

# How much slower than the best rung a larger share may be and still be kept.
# The ladder is walked from the largest share down, so a tie -- or a
# neighbouring tenant's spike during one candidate's turn -- resolves towards
# using the devices rather than towards giving up on them.
_REFINE_TOLERANCE = 0.03


def calibrate_chunk_size(
    operation: str,
    total_rows: int,
    *,
    row_bytes: int,
    chunk_rays: int | None = None,
    budget_bytes: int | None = None,
) -> ChunkPlan:
    """Pick the largest chunk that fits the tape budget, and record why.

    An explicit `chunk_rays` is honoured verbatim (clamped to the batch, so a
    chunk larger than the batch is simply one launch). Otherwise a
    `budget_bytes` is divided by the operation's per-row cost -- the D7 tape
    estimate for `trace_reflections`, the result schema's own row size for the
    operations whose tape is not the binding term -- and never falls below a
    single row, because a batch has to make progress even under an absurd
    budget. With neither, a chunk is a whole shard and the executor's only job
    is the double-buffered gather.
    """
    rows = max(int(total_rows), 0)
    cost = max(int(row_bytes), 1)
    if chunk_rays is not None:
        size = int(chunk_rays)
        source = "requested"
    elif budget_bytes is not None:
        size = max(int(budget_bytes) // cost, 1)
        source = "budget"
    else:
        size = max(rows, 1)
        source = "shard"
    if rows:
        size = min(size, rows)
    return ChunkPlan(
        operation=operation,
        total_rows=rows,
        chunk_rays=max(size, 1),
        source=source,
        row_bytes=cost,
        budget_bytes=None if budget_bytes is None else int(budget_bytes),
    )


def _resolve_lane_window(
    lane_offset: int, lane_count: int, total_samples: int
) -> tuple[int, int]:
    """The `(begin, count)` window a diffraction accumulation call asks for.

    This is the host-side twin of `resolve_lane_window()` in
    `diffraction/ops.cpp`, including its messages: the orchestrator has to know
    the width of the window before it can split it, and a caller must see the
    same rejection on a multi-device scene as on a single-device one.
    """
    total = int(total_samples)
    begin = int(lane_offset)
    count = int(lane_count)
    if begin < 0:
        raise RuntimeError("lane_offset must be non-negative.")
    if begin > total:
        raise RuntimeError("lane_offset must not exceed the total sample count.")
    remaining = total - begin
    if count < 0:
        return begin, remaining
    if count > remaining:
        raise RuntimeError(
            "lane_offset + lane_count must not exceed the total sample count."
        )
    return begin, count


def _pick_candidate(seconds: Sequence[float]) -> int:
    """The first rung within `_REFINE_TOLERANCE` of the fastest one.

    "First" is the largest remote share, because the ladder is built that way:
    shrinking a device's shard is only worth doing when it is *measurably*
    faster, and a run whose candidates all land within a few percent of each
    other has not measured a reason to.
    """
    best = min(seconds)
    threshold = best * (1.0 + _REFINE_TOLERANCE)
    for index, value in enumerate(seconds):
        if value <= threshold:
            return index
    return int(seconds.index(best))


def _align_lanes(value: int) -> int:
    """Round a lane count to the nearest whole warp, halves upwards."""
    return ((value + _LANE_ALIGNMENT // 2) // _LANE_ALIGNMENT) * _LANE_ALIGNMENT


def _lane_chunk_size(chunk_rays: int, count: int) -> int:
    """A chunk of lanes: at least one warp, never wider than the window.

    A requested chunk is rounded *up* to a whole warp so that no chunk boundary
    falls inside one, and the trailing chunk is simply short -- exactly as the
    single launch's trailing warp is.
    """
    size = max(int(chunk_rays), 1)
    if size % _LANE_ALIGNMENT:
        size += _LANE_ALIGNMENT - size % _LANE_ALIGNMENT
    if count > 0:
        size = min(size, count)
    return size


def _to(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    """Replicate one whole (unsharded) input onto `device`."""
    if value is None or value.device == device:
        return value
    return value.to(device, non_blocking=True)


def _states_to(states: DfrStates, device: torch.device) -> DfrStates:
    """One replica's view of caller-owned diffraction states.

    Every field is whole: the lane split is over Monte-Carlo samples, not over
    states, so a shard sees the same states the single launch sees. The copy is
    autograd-recorded, which is what reduces a shard's state gradients back onto
    the caller's master leaves (D4).
    """
    if states.edge_pos.device == device:
        return states
    return DfrStates(
        edge_index=_to(states.edge_index, device),
        edge_pos=_to(states.edge_pos, device),
        edge_dir=_to(states.edge_dir, device),
        edge_t_min=_to(states.edge_t_min, device),
        edge_t_max=_to(states.edge_t_max, device),
        n0=_to(states.n0, device),
        n1=_to(states.n1, device),
        prim0=_to(states.prim0, device),
        prim1=_to(states.prim1, device),
        exterior_angle=_to(states.exterior_angle, device),
        src=_to(states.src, device),
        src_power=_to(states.src_power, device),
        wi=_to(states.wi, device),
        d0=_to(states.d0, device),
        count=states.count,
    )


def _material_to(material: DfrMaterial, device: torch.device) -> DfrMaterial:
    if material.eta_r.device == device:
        return material
    return DfrMaterial(
        eta_r=_to(material.eta_r, device),
        sigma=_to(material.sigma, device),
        mu_r=_to(material.mu_r, device),
        gain=_to(material.gain, device),
        valid=_to(material.valid, device),
    )


def _add_accum(left: DfrAccum, right: DfrAccum) -> DfrAccum:
    """Sum two partial accumulation results field by field, on their device."""
    return DfrAccum(
        left.grid_cell_count,
        *(getattr(left, name) + getattr(right, name) for name in _DFR_ACCUM_FIELDS),
    )


def _accum_to(result: DfrAccum, device: torch.device) -> DfrAccum:
    if result.power.device == device:
        return result
    return DfrAccum(
        result.grid_cell_count,
        *(
            getattr(result, name).to(device, non_blocking=True)
            for name in _DFR_ACCUM_FIELDS
        ),
    )


def _device_index(value: object, position: int) -> int:
    """One `devices` entry as a plain CUDA device index."""
    if isinstance(value, torch.device):
        device = value
    elif isinstance(value, str):
        device = torch.device(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        device = torch.device("cuda", value)
    else:
        raise TypeError(
            "Scene(devices=...) entries must be int, str, or torch.device, got "
            f"{type(value).__name__} at position {position}."
        )
    if device.type != "cuda":
        raise ValueError(
            f"Scene(devices=...) only accepts CUDA devices, got {device!r} "
            f"at position {position}."
        )
    return 0 if device.index is None else device.index


def _normalize_devices(devices: Sequence[object]) -> list[int]:
    if isinstance(devices, (int, str, torch.device)):
        raise TypeError(
            "Scene(devices=...) expects a sequence of devices; "
            f"pass [{devices!r}] instead of {devices!r}."
        )
    indices = [_device_index(value, position) for position, value in enumerate(devices)]
    if not indices:
        raise ValueError("Scene(devices=...) needs at least one device.")
    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise ValueError(
            f"Scene(devices=...) received duplicate devices {duplicates}; "
            "each device holds exactly one replica."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Scene(devices=...) needs CUDA, but torch.cuda.is_available() is False."
        )
    count = torch.cuda.device_count()
    for index in indices:
        if index < 0 or index >= count:
            raise ValueError(
                f"Scene(devices=...) got device index {index}, but only "
                f"{count} CUDA device(s) are visible."
            )
    return indices


def _normalize_weights(
    options: MultiDeviceOptions, devices: Sequence[int]
) -> tuple[float, ...]:
    if options.weights is None:
        return (1.0,) * len(devices)
    weights = tuple(float(value) for value in options.weights)
    if len(weights) != len(devices):
        raise ValueError(
            f"MultiDeviceOptions.weights has {len(weights)} entries but "
            f"{len(devices)} devices were given."
        )
    for weight in weights:
        if not (weight >= 0.0) or weight == float("inf"):
            raise ValueError(
                f"MultiDeviceOptions.weights must be finite and non-negative, got {weights}."
            )
    if sum(weights) <= 0.0:
        raise ValueError("MultiDeviceOptions.weights must not sum to zero.")
    return weights


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"MultiDeviceOptions.{name} must be an int, got {type(value).__name__}."
        )
    if value < 1:
        raise ValueError(f"MultiDeviceOptions.{name} must be at least 1, got {value}.")
    return int(value)


def _validate_chunking(options: MultiDeviceOptions) -> bool:
    """Check the chunked executor's knobs; report whether any of them is set."""
    if options.chunk_rays is not None:
        _positive_int(options.chunk_rays, "chunk_rays")
    if options.tape_memory_budget_bytes is not None:
        _positive_int(options.tape_memory_budget_bytes, "tape_memory_budget_bytes")
    if options.offload is not None and not callable(options.offload):
        raise TypeError(
            "MultiDeviceOptions.offload must be callable as offload(chunk_start, "
            f"result), got {type(options.offload).__name__}."
        )
    return (
        options.chunk_rays is not None
        or options.tape_memory_budget_bytes is not None
        or options.offload is not None
    )


def _validate_pipeline(options: MultiDeviceOptions) -> None:
    """Check the throughput knobs of Phase 2d."""
    if not isinstance(options.pipeline, bool):
        raise TypeError(
            "MultiDeviceOptions.pipeline must be a bool, got "
            f"{type(options.pipeline).__name__}."
        )
    chunks = _positive_int(
        options.pipeline_chunks_per_device, "pipeline_chunks_per_device"
    )
    if chunks < 2:
        raise ValueError(
            "MultiDeviceOptions.pipeline_chunks_per_device must be at least 2 -- "
            "one chunk per device has nothing to overlap with. Pass "
            "pipeline=False for the one-launch-per-shard path."
        )
    _positive_int(options.min_rays_per_device, "min_rays_per_device")


def plan(
    devices: Sequence[object] | None,
    options: MultiDeviceOptions | None,
    *,
    trace_backend: str,
    edge_bvh_backend: str,
) -> "_ReplicatedScene | None":
    """Return the orchestrator for `devices`, or `None` for the single-device path.

    A one-device `Scene(devices=[d])` is not a degenerate replica set: it is
    the pre-existing path verbatim (D9), so it gets no orchestrator at all --
    unless it asked for the chunked executor, which is a per-device memory
    story and therefore has something to do on one device as well (D7).
    """
    if devices is None:
        if options is not None:
            raise TypeError("Scene(options=...) also requires devices=[...].")
        return None
    indices = _normalize_devices(devices)
    resolved = MultiDeviceOptions() if options is None else options
    if not isinstance(resolved, MultiDeviceOptions):
        raise TypeError(
            "Scene(options=...) expects rayd.torch.MultiDeviceOptions, got "
            f"{type(resolved).__name__}."
        )
    weights = _normalize_weights(resolved, indices)
    chunking = _validate_chunking(resolved)
    _validate_pipeline(resolved)
    if len(indices) == 1 and not chunking:
        return None
    return _ReplicatedScene(
        indices, resolved, weights, trace_backend, edge_bvh_backend, chunking
    )


class _PipeStreams(NamedTuple):
    """One device's private streams: compute, plus one per pair direction.

    `scatter_src` and `gather_dst` live on the *master*, `compute`,
    `scatter_dst` and `gather_src` on the device itself. The master's own entry
    has no copies to make, so its four copy streams are `None`.
    """

    compute: "torch.cuda.Stream"
    scatter_src: "torch.cuda.Stream | None"
    scatter_dst: "torch.cuda.Stream | None"
    gather_src: "torch.cuda.Stream"
    gather_dst: "torch.cuda.Stream | None"


class _ReplicatedScene:
    """One ordinary single-device `Scene` per device, plus the batch sharder."""

    __slots__ = (
        "devices",
        "options",
        "weights",
        "chunked",
        "pipelined",
        "min_rays_per_device",
        "last_chunk_plan",
        "last_dispatch",
        "last_calibration",
        "_trace_backend",
        "_edge_bvh_backend",
        "_replicas",
        "_streams",
        "_active_stream",
    )

    def __init__(
        self,
        devices: Sequence[int],
        options: MultiDeviceOptions,
        weights: Sequence[float],
        trace_backend: str,
        edge_bvh_backend: str,
        chunked: bool = False,
    ) -> None:
        self.devices = tuple(torch.device("cuda", index) for index in devices)
        self.options = options
        self.weights = tuple(weights)
        self.chunked = bool(chunked)
        # Pipelining is a sharding story: a one-device scene has no scatter and
        # no gather to overlap, so it stays on whatever the chunking knobs said.
        self.pipelined = bool(options.pipeline) and len(self.devices) > 1
        self.min_rays_per_device = int(options.min_rays_per_device)
        self.last_chunk_plan: ChunkPlan | None = None
        # Which of the four dispatch paths the last `per_ray` operation took;
        # a debug surface for tests and for anyone wondering why a batch was
        # fast or slow. Not API.
        self.last_dispatch: str | None = None
        self.last_calibration: DeviceCalibration | None = None
        self._trace_backend = trace_backend
        self._edge_bvh_backend = edge_bvh_backend
        self._replicas: tuple = ()
        self._streams: dict[int, "_PipeStreams"] = {}
        self._active_stream = None

    # -- replica lifecycle -------------------------------------------------

    @property
    def master_device(self) -> torch.device:
        return self.devices[0]

    def discard(self) -> None:
        """Drop the replicas; the next `build()` rebuilds them from scratch."""
        self._replicas = ()

    def _require_replicas(self) -> tuple:
        if not self._replicas:
            raise RuntimeError("Scene is not ready. Call build() before querying.")
        return self._replicas

    def master(self):
        return self._require_replicas()[0]

    def master_native_scene(self):
        """The master replica's native scene, which the public `Scene` reads.

        Scene-level metadata (`version`, `num_meshes`, `edge_mask`,
        `global_geometry`) is replica-invariant by construction, so the public
        object answers those from the master instead of asking every device.
        """
        return self.master()._native_scene

    def build(self, meshes: Sequence[tuple]) -> None:
        from .mesh import Mesh
        from .scene import Scene

        self.discard()
        master = self.master_device
        for position, (mesh, _dynamic) in enumerate(meshes):
            for name in (
                "vertices",
                "faces",
                "uv",
                "face_uv",
                "to_world_left",
                "to_world_right",
            ):
                value = getattr(mesh, name)
                if value.device != master:
                    raise ValueError(
                        f"Scene(devices=...) requires every mesh tensor on the master "
                        f"device {master}, but mesh {position} has {name} on "
                        f"{value.device}. Replicas are built from the master tensors."
                    )

        if self.options.warm_up:
            from ._warmup import warm_up_devices

            warm_up_devices([device.index for device in self.devices])

        replicas = []
        for device in self.devices:
            replica = Scene(
                trace_backend=self._trace_backend,
                edge_bvh_backend=self._edge_bvh_backend,
            )
            for mesh, dynamic in meshes:
                replica.add_mesh(
                    Mesh(
                        # `.to()` on the master device is the identity, so the
                        # master replica shares the caller's tensors and runs
                        # exactly what a single-device Scene would run. On the
                        # other devices this is a recorded autograd copy, which
                        # is what reduces every replica's vertex gradient back
                        # onto the master leaf (D4).
                        vertices=mesh.vertices.to(device),
                        faces=mesh.faces.to(device),
                        uv=mesh.uv.to(device),
                        face_uv=mesh.face_uv.to(device),
                        use_face_normals=mesh.use_face_normals,
                        edges_enabled=mesh.edges_enabled,
                        to_world_left=mesh.to_world_left.to(device),
                        to_world_right=mesh.to_world_right.to(device),
                    ),
                    dynamic,
                )
            # Building an acceleration structure is the one step that still
            # wants its own device current.
            with torch.cuda.device(device):
                replica.build()
            replicas.append(replica)
        self._replicas = tuple(replicas)
        self._require_version_lockstep()

    def _require_version_lockstep(self) -> None:
        """Every replica must agree on the scene version after any mutation."""
        versions = [replica.version for replica in self._require_replicas()]
        if len(set(versions)) != 1:
            raise RuntimeError(
                "RayD multi-device replicas diverged: scene versions "
                f"{versions} for devices {[device.index for device in self.devices]}."
            )

    # -- broadcast mutation ------------------------------------------------

    def update_mesh_vertices(self, mesh_id: int, positions: torch.Tensor) -> None:
        for replica, device in zip(self._require_replicas(), self.devices):
            with torch.cuda.device(device):
                replica.update_mesh_vertices(mesh_id, positions.to(device))
        self._require_version_lockstep()

    def sync(self) -> None:
        for replica, device in zip(self._require_replicas(), self.devices):
            with torch.cuda.device(device):
                replica.sync()
        self._require_version_lockstep()

    def set_edge_mask(self, mask: torch.Tensor) -> None:
        for replica, device in zip(self._require_replicas(), self.devices):
            replica.set_edge_mask(mask.to(device))
        self._require_version_lockstep()

    def unsupported(self, operation: str) -> None:
        raise NotImplementedError(
            f"Scene.{operation}() has no multi-device semantics yet; it is neither "
            f"a per_ray operation nor a lane-windowed grid_reduce one, so sharding "
            f"it needs the explicit per-shard contract of {_PHASE_2C}. Build the "
            "Scene without devices=[...] to run it on one device."
        )

    # -- sharding ----------------------------------------------------------

    def _shards(self, total: int) -> list[tuple]:
        """Contiguous weighted `[start, stop)` slices, one per device.

        Zero-length shards are dropped rather than launched: several native
        ops reject an empty batch, and a device with nothing to do contributes
        nothing to the concatenation anyway. An all-empty batch is handled by
        the callers, which fall back to the master replica so that an empty
        query behaves exactly as it does on one device.
        """
        weights = self.weights
        total_weight = sum(weights)
        shards = []
        start = 0
        carried = 0.0
        for position, (replica, device) in enumerate(zip(self._require_replicas(), self.devices)):
            if position + 1 == len(weights):
                stop = total
            else:
                carried += weights[position]
                stop = int(total * carried / total_weight)
            if stop > start:
                shards.append((replica, device, start, stop))
            start = stop
        return shards

    def _lane_shards(self, begin: int, count: int) -> list[tuple]:
        """Weighted, warp-aligned `(replica, device, lane_begin, lane_count)` windows.

        The windows are contiguous, disjoint, and cover `[begin, begin + count)`
        exactly, and every interior boundary is `begin` plus a whole number of
        warps, so each shard inherits the warp partition the single launch has
        (see the module docstring). Rounding a boundary can therefore move up to
        half a warp of samples between neighbours, which is why a window
        narrower than a warp per device leaves the leading devices idle rather
        than splitting one. A boundary that has reached the end of the window --
        a device weighted to zero, for instance -- stays there instead of being
        rounded back inside it.
        """
        weights = self.weights
        total_weight = sum(weights)
        shards = []
        start = 0
        carried = 0.0
        for position, (replica, device) in enumerate(
            zip(self._require_replicas(), self.devices)
        ):
            if position + 1 == len(weights):
                stop = count
            else:
                carried += weights[position]
                raw = int(count * carried / total_weight)
                if raw >= count:
                    stop = count
                else:
                    stop = min(max(_align_lanes(raw), start), count)
            if stop > start:
                shards.append((replica, device, begin + start, stop - start))
            start = stop
        return shards

    def _retain(self, value: torch.Tensor) -> torch.Tensor:
        """Keep `value` alive for the side stream that is about to read it.

        Only the chunked executor runs on a stream other than the one the
        caller's tensors were allocated on, so this is a no-op everywhere else.
        There it matters: a chunk's inputs may be the caller's own tensors
        (the master shard copies nothing), and the caching allocator would be
        free to hand their memory to the next allocation the moment the caller
        drops its reference, while our compute stream has not run yet.
        """
        stream = self._active_stream
        if stream is not None and value.device == stream.device:
            value.record_stream(stream)
        return value

    def _slice(
        self, value: torch.Tensor | None, start: int, stop: int, device: torch.device
    ) -> torch.Tensor | None:
        """One shard of a batched input, on `device`.

        A shard covering the whole batch passes the caller's tensor through
        untouched, so a degenerate split (one device weighted to zero) issues
        literally the single-device call.
        """
        if value is None:
            return None
        if start != 0 or stop != value.shape[0]:
            value = value[start:stop]
        return self._retain(value.to(device, non_blocking=True))

    def _slice_rows(
        self,
        value: torch.Tensor | None,
        name: str,
        total: int,
        start: int,
        stop: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Shard an optional per-row input whose batch axis must be explicit."""
        if value is None:
            return None
        if value.ndim < 1 or value.shape[0] != total:
            raise ValueError(
                f"Scene(devices=...) can only shard {name} when its first dimension "
                f"is the batch axis; expected {total} rows, got shape "
                f"{tuple(value.shape)}."
            )
        return self._slice(value, start, stop, device)

    def _shard_ray(
        self, ray: Ray, start: int, stop: int, device: torch.device
    ) -> Ray:
        tmax = ray.tmax
        if tmax.numel() == 0:
            shard_tmax = self._retain(tmax.to(device, non_blocking=True))
        else:
            shard_tmax = self._slice(tmax, start, stop, device)
        return Ray(
            self._slice(ray.o, start, stop, device),
            self._slice(ray.d, start, stop, device),
            shard_tmax,
        )

    def _gather(self, parts: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate one output field back onto the master device."""
        master = self.master_device
        return self._concat([value.to(master, non_blocking=True) for value in parts])

    @staticmethod
    def _concat(parts: list[torch.Tensor]) -> torch.Tensor:
        """Join chunk outputs that already live on the master device."""
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    # -- dispatch ----------------------------------------------------------

    def _run_shards(self, total: int, shard, call) -> list | None:
        """One launch per shard: scatter every shard's inputs, then enqueue.

        Scattering first and launching second is what lets the devices overlap
        on this path: the whole batch is on its way before the first replica's
        op is enqueued. `None` means the batch is empty, which the callers
        answer from the master replica so that an empty query behaves exactly
        as it does on one device.
        """
        shards = self._shards(total)
        if not shards:
            return None
        inputs = [
            (replica, shard(device, start, stop))
            for replica, device, start, stop in shards
        ]
        return [call(replica, arguments) for replica, arguments in inputs]

    def _pipe_streams(self, device: torch.device) -> "_PipeStreams":
        """This device's private streams, created once and reused.

        One stream per device *pair direction* plus the device's own compute
        stream, which is what keeps a copy off the master's compute stream and
        off the caller's (D7/Phase 2d). The two `_dst` streams look redundant
        -- nothing is ever enqueued on them -- but they are load-bearing:
        torch runs a cross-device copy on the *source* device's current stream
        and barriers it against the *destination* device's current stream in
        both directions, so if the destination's current stream were the
        caller's, chunk `k`'s copy would wait for whatever the caller had
        queued, and if it were the compute stream, chunk `k`'s scatter would
        wait for chunk `k-1`'s compute -- which is exactly the overlap the
        pipeline exists to buy. Pointing the barrier at an otherwise unused
        stream costs two event records and keeps the copy independent.
        """
        streams = self._streams.get(device.index)
        if streams is None:
            master = self.master_device
            with torch.cuda.device(device):
                compute = torch.cuda.Stream(device=device)
                gather_src = torch.cuda.Stream(device=device)
                scatter_dst = (
                    None if device == master else torch.cuda.Stream(device=device)
                )
            scatter_src = gather_dst = None
            if device != master:
                with torch.cuda.device(master):
                    scatter_src = torch.cuda.Stream(device=master)
                    gather_dst = torch.cuda.Stream(device=master)
            streams = _PipeStreams(
                compute, scatter_src, scatter_dst, gather_src, gather_dst
            )
            self._streams[device.index] = streams
        return streams

    def _scatter_chunk(self, streams: "_PipeStreams", device, shard, start, stop):
        """One chunk's inputs on `device`, and the event its compute must await.

        The master's shard copies nothing -- it is the caller's own memory, so
        there is no event either, only the `entry` edge every stream already
        took. Every other device's inputs are copied on that pair's own master
        side stream, so the scatter overlaps the master's compute instead of
        queueing in front of it.
        """
        if streams.scatter_src is None:
            self._active_stream = streams.compute
            try:
                return shard(device, start, stop), None
            finally:
                self._active_stream = None
        master = self.master_device
        with torch.cuda.device(master), torch.cuda.stream(streams.scatter_src):
            with torch.cuda.stream(streams.scatter_dst):
                # `_retain` records the *destination* against the compute
                # stream, which is the stream that will read it.
                self._active_stream = streams.compute
                try:
                    arguments = shard(device, start, stop)
                finally:
                    self._active_stream = None
        copied = torch.cuda.Event()
        copied.record(streams.scatter_src)
        return arguments, copied

    def _gather_chunk(self, streams: "_PipeStreams", device, columns, sink, rows):
        """One chunk's outputs on their way to the master, off every hot stream.

        `sink` is either `None` -- every chunk allocates its own master-side
        copy, which is then concatenated, the mode the streaming hook and the
        autograd path need -- or the operation's preallocated master buffers,
        into which every chunk copies its own row window directly. The second
        mode is what a large pipelined batch runs: it removes one master-side
        allocation and one `record_stream` per chunk *per field*, and it
        removes the final concatenation altogether, which at these batch sizes
        is the difference between being bound by the devices and being bound by
        the host.
        """
        computed = torch.cuda.Event()
        computed.record(streams.compute)
        streams.gather_src.wait_event(computed)
        start, stop = rows
        if streams.gather_dst is None:
            # The master's own chunk needs no interconnect, only the event
            # below -- and, in buffer mode, its row window filled in.
            if sink is None:
                moved = list(columns)
            else:
                with torch.cuda.device(device), torch.cuda.stream(streams.gather_src):
                    moved = [buffer[start:stop] for buffer in sink]
                    for target, column in zip(moved, columns):
                        # The chunk's own output was allocated on the compute
                        # stream and is being read here on another one; without
                        # this the allocator may hand it to the next chunk's
                        # compute while this copy is still in flight.
                        column.record_stream(streams.gather_src)
                        target.copy_(column, non_blocking=True)
        else:
            master = self.master_device
            with torch.cuda.device(device), torch.cuda.stream(streams.gather_src):
                with torch.cuda.stream(streams.gather_dst):
                    if sink is None:
                        moved = []
                        for column in columns:
                            column.record_stream(streams.gather_src)
                            moved.append(column.to(master, non_blocking=True))
                    else:
                        moved = [buffer[start:stop] for buffer in sink]
                        for target, column in zip(moved, columns):
                            column.record_stream(streams.gather_src)
                            target.copy_(column, non_blocking=True)
        gathered = torch.cuda.Event()
        gathered.record(streams.gather_src)
        return moved, gathered

    def _master_buffers(self, total: int, rows: int, columns) -> list[torch.Tensor] | None:
        """The operation's whole output, allocated once on the master.

        `None` means the operation cannot use one, and the chunks are
        concatenated instead. A column that carries gradient always declines:
        filling one buffer slice by slice would make every chunk's backward
        walk the whole buffer's `CopySlices` chain. So does a column that is
        not a plain contiguous row block of the chunk it came from, because
        then "this chunk's rows" is not a slice of the result. The values are
        the same either way.
        """
        for column in columns:
            if (
                column.requires_grad
                or not column.is_contiguous()
                or column.dim() < 1
                or column.shape[0] != rows
            ):
                return None
        master = self.master_device
        return [
            torch.empty(
                (total,) + tuple(column.shape[1:]),
                dtype=column.dtype,
                device=master,
            )
            for column in columns
        ]

    def _run_chunked(
        self,
        operation: str,
        total: int,
        row_bytes: int,
        shard,
        call,
        extract,
        assemble,
        chunk_rows: int | None = None,
        master_chunk_rows: int | None = None,
    ):
        """One launch per chunk, double-buffered per device (D7).

        `extract` reads a chunk's result into its canonical field order and
        `assemble` rebuilds the public result from gathered fields, so the same
        loop serves the concatenating and the streaming mode. Ordering is
        event-only: the copy streams wait for the chunk's compute, the master
        stream waits for the chunk's gather, and nothing waits for the host.

        `chunk_rows` is the pipelined dispatch of Phase 2d: the chunk size is
        chosen for overlap rather than for a memory budget, and the caller's
        chunking knobs -- which are a memory contract -- are not consulted.
        """
        if chunk_rows is None:
            plan = calibrate_chunk_size(
                operation,
                total,
                row_bytes=row_bytes,
                chunk_rays=self.options.chunk_rays,
                budget_bytes=self.options.tape_memory_budget_bytes,
            )
        else:
            plan = ChunkPlan(
                operation=operation,
                total_rows=max(int(total), 0),
                chunk_rays=max(int(chunk_rows), 1),
                source="pipeline",
                row_bytes=max(int(row_bytes), 1),
            )
        self.last_chunk_plan = plan
        offload = self.options.offload
        master_rows = plan.chunk_rays if master_chunk_rows is None else max(
            int(master_chunk_rows), 1
        )
        queues = []
        for replica, device, start, stop in self._shards(total):
            rows = master_rows if device == self.master_device else plan.chunk_rays
            queues.append(
                (
                    replica,
                    device,
                    [
                        (chunk_start, min(chunk_start + rows, stop))
                        for chunk_start in range(start, stop, rows)
                    ],
                )
            )
        plan.chunk_count = sum(len(chunks) for _replica, _device, chunks in queues)
        if not queues:
            # An empty batch is one empty launch on the master, which is what
            # the unchunked path does too; the hook sees no chunk at all.
            if offload is not None:
                return None
            master = self.master_device
            return assemble(0, list(extract(call(self.master(), shard(master, 0, 0)))))

        # Round-robin over the devices so every device has a chunk in flight
        # before any device gets its second one.
        order = []
        for index in range(max(len(chunks) for _replica, _device, chunks in queues)):
            for replica, device, chunks in queues:
                if index < len(chunks):
                    order.append((replica, device, chunks[index]))

        master = self.master_device
        master_stream = torch.cuda.current_stream(master)
        # The executor's streams have to pick up where the caller's streams
        # left off on *every* device it is about to touch, not only on the
        # master. The master's current stream is where the batch's inputs were
        # produced, and every stream waits for it. But a replica's own state is
        # mutated on that replica's device: `build()`, `sync()` -- whose
        # triangle GAS refit, IAS rebuild and edge-GAS build are stream-ordered
        # on the scene's stream, with no host synchronization -- and
        # `set_edge_mask` all enqueue there. A compute stream that waited only
        # on the master's event would be free to traverse a half-rebuilt
        # acceleration structure, so a query issued right after
        # `update_mesh_vertices()` + `sync()` could silently answer from stale
        # geometry. The compute stream is the only one that reads replica
        # state, so it is the only one that takes the second edge.
        entry: dict[int, "torch.cuda.Event"] = {}
        for index in (master.index, *(device.index for _r, device, _c in order)):
            if index not in entry:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(torch.device("cuda", index)))
                entry[index] = event

        # Two chunks per device in flight is the double buffer: the chunk being
        # gathered plus the chunk being computed.
        depth = 2 * len(queues)
        pipeline: deque = deque()
        collected: list = []
        started: set = set()
        sink: list | None = None
        first = True
        for replica, device, (chunk_start, chunk_stop) in order:
            streams = self._pipe_streams(device)
            if device.index not in started:
                for stream in streams:
                    if stream is not None:
                        stream.wait_event(entry[master.index])
                if device.index != master.index:
                    streams.compute.wait_event(entry[device.index])
                started.add(device.index)
            arguments, copied = self._scatter_chunk(
                streams, device, shard, chunk_start, chunk_stop
            )
            if copied is not None:
                streams.compute.wait_event(copied)
            with torch.cuda.device(device), torch.cuda.stream(streams.compute):
                self._active_stream = streams.compute
                try:
                    columns = list(extract(call(replica, arguments)))
                finally:
                    self._active_stream = None
            del arguments
            if first:
                # The first chunk is what says how wide a row is, so the whole
                # output can be allocated once -- and whether it may be, which
                # only autograd and the streaming hook can veto.
                first = False
                if offload is None:
                    sink = self._master_buffers(
                        total, chunk_stop - chunk_start, columns
                    )
            moved, gathered = self._gather_chunk(
                streams, device, columns, sink, (chunk_start, chunk_stop)
            )
            pipeline.append((chunk_start, chunk_stop, moved, gathered))
            del columns, moved
            if len(pipeline) > depth:
                self._retire(
                    pipeline.popleft(), plan, master_stream, assemble, collected, sink
                )
        while pipeline:
            self._retire(
                pipeline.popleft(), plan, master_stream, assemble, collected, sink
            )
        self._exit(started)

        if offload is not None:
            return None
        if sink is not None:
            return assemble(total, sink)
        collected.sort(key=lambda piece: piece[0])
        width = len(collected[0][1])
        return assemble(
            total,
            [
                self._concat([piece[1][index] for piece in collected])
                for index in range(width)
            ],
        )

    def _exit(self, started: set) -> None:
        """Hand each device back to the stream the caller will use next.

        The other half of the entry edge, and the same hazard read backwards:
        the executor's compute stream is still traversing this replica's
        acceleration structure when the call returns, while the caller's next
        `update_mesh_vertices()` + `sync()` enqueues a GAS refit and an IAS
        rebuild on that device's *current* stream, which knows nothing about
        our private one. Retiring the chunks only ordered the *master's* stream
        after the gathers; every other device needs the same edge, or a
        mutation can overwrite geometry a live traversal is still reading. Only
        the compute stream needs it: the copy streams touch caller and chunk
        tensors, whose lifetime the caching allocator already tracks through
        `record_stream`.
        """
        for index in started:
            done = torch.cuda.Event()
            done.record(self._streams[index].compute)
            torch.cuda.current_stream(torch.device("cuda", index)).wait_event(done)

    def _retire(
        self, piece, plan: ChunkPlan, master_stream, assemble, collected, sink=None
    ) -> None:
        """Hand one finished chunk to the hook, or keep it for the gather."""
        chunk_start, chunk_stop, moved, gathered = piece
        # The chunk ran on a side stream; whoever reads it next reads it from
        # the master's stream, so that is the one edge that has to exist.
        master_stream.wait_event(gathered)
        rows = chunk_stop - chunk_start
        if plan.measured_row_bytes is None and rows > 0:
            plan.measured_row_bytes = (
                sum(value.numel() * value.element_size() for value in moved) / rows
            )
        if sink is not None:
            # The rows are already in the operation's own output, which the
            # caller's stream owns and which nothing frees until the caller
            # does; there is no per-chunk tensor left to keep alive.
            return
        for value in moved:
            value.record_stream(master_stream)
        if self.options.offload is None:
            collected.append((chunk_start, moved))
            return
        self.options.offload(chunk_start, assemble(rows, moved))

    # -- dispatch policy ---------------------------------------------------

    def _dispatch_mode(self, total: int) -> str:
        """Which of the four paths a `per_ray` batch of `total` rows takes.

        `chunked` -- the caller set a chunking knob, which is a memory
        contract: it is honoured at every batch size and its chunk size, not
        the pipeline's, decides the launches.
        `master` -- the batch is too small for sharding to pay for its own
        copies, so it runs on the master replica exactly as a single-device
        `Scene` would run it (see `min_rays_per_device`).
        `pipelined` -- the default multi-device path: scatter, compute and
        gather overlap chunk by chunk and device by device.
        `sharded` -- `pipeline=False`: one launch per shard, which is the
        Phase 2a path the pipelined results are checked against.
        """
        if self.chunked:
            return "chunked"
        if len(self.devices) > 1 and total < self.min_rays_per_device * len(self.devices):
            return "master"
        if not self.pipelined:
            return "sharded"
        return "pipelined"

    def _master_takes_everything(self, total: int) -> bool:
        """Do the weights leave the master with the whole batch?

        `weights=[1, 0]` -- which is what a calibration answers with when an
        operation moves more bytes per row than it spends compute on -- is the
        single-device call written as a split, and is dispatched as one. The
        test is on the weights rather than on `_shards()` because it runs on
        every query: a non-zero weight always takes at least one row of a
        non-empty batch, since `int(total * carried / sum) < total` whenever
        anything is left over.
        """
        return total > 0 and not any(weight > 0.0 for weight in self.weights[1:])

    def _pipeline_rows(self, total: int) -> tuple[int, int]:
        """The auto chunk sizes: `(remote_rows, master_rows)`.

        Sizing from the widest shard rather than the narrowest bounds the total
        launch count at roughly `chunks * len(devices)` however skewed the
        weights are, which matters because a chunk's host cost does not shrink
        with its batch (measured on this machine: ~0.3 ms per chunk, of which
        ~0.12 ms is the native launch itself, whatever the chunk's size). A
        device weighted far below its neighbours therefore gets fewer chunks --
        it also has proportionally less to transfer, so there is less to hide.

        The master's shard is one launch. It has nothing to overlap: its inputs
        are already there and its rows are already on the right device, so
        splitting it buys only the chance to spread its copy into the
        operation's output -- and pays a launch per chunk for it, which at
        these sizes is the more expensive half.
        """
        chunks = max(int(self.options.pipeline_chunks_per_device), 2)
        master = self.master_device
        widest = 0
        master_rows = 0
        for _replica, device, start, stop in self._shards(total):
            widest = max(widest, stop - start)
            if device == master:
                master_rows = stop - start
        if widest <= 0:
            return 1, 1
        return max(-(-widest // chunks), 1), max(master_rows, 1)

    def _dispatch(
        self,
        operation: str,
        total: int,
        row_bytes: int,
        shard,
        call,
        extract,
        assemble,
        master,
    ):
        """Route one `per_ray` operation, then run it.

        `master()` is the operation as the master replica alone would run it,
        with the caller's own tensors -- the small-batch answer, and the empty
        batch's answer, both of which are bitwise a single-device result
        because they *are* a single-device call.
        """
        mode = self._dispatch_mode(total)
        self.last_dispatch = mode
        if mode == "master":
            return master()
        if mode == "sharded":
            results = self._run_shards(total, shard, call)
            if results is None:
                self.last_dispatch = "master"
                return master()
            columns = [extract(result) for result in results]
            return assemble(
                total, [self._gather(list(parts)) for parts in zip(*columns)]
            )
        if mode == "chunked":
            return self._run_chunked(
                operation, total, row_bytes, shard, call, extract, assemble
            )
        if self._master_takes_everything(total):
            # A split that leaves the other devices empty is a single-device
            # call, and running it as one is bitwise what the master alone
            # would produce -- without the pipeline's copy into a fresh output.
            self.last_dispatch = "master"
            return master()
        remote_rows, master_rows = self._pipeline_rows(total)
        return self._run_chunked(
            operation,
            total,
            row_bytes,
            shard,
            call,
            extract,
            assemble,
            chunk_rows=remote_rows,
            master_chunk_rows=master_rows,
        )

    # -- calibration -------------------------------------------------------

    def calibrate(
        self,
        *,
        rows: int = _CALIBRATION_ROWS,
        max_bounces: int = 0,
        probe: Callable[[object, torch.device], object] | None = None,
        repeats: int = 3,
        warm_up: int = 1,
        refine: bool = True,
    ) -> DeviceCalibration:
        """Measure each device on the same work, and split the batch by it.

        The plan's Sharder gets its weights from "a one-time `calibrate()`
        micro-benchmark" (D1). This is that, in two stages.

        The *throughput* stage runs the same probe on every replica, on its own
        device, with its inputs already resident there -- so what is timed is
        the device, not the interconnect -- and makes the weights inversely
        proportional to the times. On a matched pair that lands within noise of
        the equal split; on a mixed pair it is the difference between the two
        devices.

        The *refinement* stage then times the real multi-device dispatch of the
        same probe while scaling everything but the master down through
        `1, 1/2, 1/4, 1/10, 0`, and keeps the fastest. This is the stage that
        knows about the interconnect: an operation whose per-row transfer costs
        more than its per-row compute (a cheap query with a wide result, say)
        is faster with a *small* remote shard, and at the far end a share of
        zero is the single-device path, which is why calibration can never
        leave a scene slower than the master alone on the probe it was given.
        Pass `refine=False` for the throughput stage on its own.

        The default probe is the scene's own geometry: `rows` rays drawn from a
        fixed seed inside the master mesh's bounding box, run through
        `intersect` (or through `trace_reflections` when `max_bounces` is set).
        It is a *shape*, not the caller's workload -- pass `probe(scene,
        device)` to time the operation that actually matters, for instance the
        exact call the training loop makes. The probe is handed a replica and
        that replica's device in the throughput stage, and the multi-device
        layer itself and the master device in the refinement stage, so write it
        to build its inputs on the device it is given and to call the operation
        positionally; both objects take the same operation signatures.

        Timing is interleaved device by device (and candidate by candidate) and
        reduced with a minimum, so a neighbouring tenant's spike inflates a
        sample rather than the split. The measurement is written to
        `last_calibration` and the weights it chose are readable there and on
        `weights`; nothing else about execution changes, so a run at the
        resulting weights is as reproducible as any other run at fixed weights.
        """
        replicas = self._require_replicas()
        rows = _positive_int(int(rows), "calibrate(rows=...)")
        repeats = _positive_int(int(repeats), "calibrate(repeats=...)")
        if warm_up < 0:
            raise ValueError("calibrate(warm_up=...) must not be negative.")
        if probe is None:
            probe = self._default_probe(rows, int(max_bounces))
        samples: list[list[float]] = [[] for _ in replicas]
        for round_index in range(int(warm_up) + repeats):
            for index, (replica, device) in enumerate(zip(replicas, self.devices)):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.cuda.device(device):
                    probe(replica, device)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                if round_index >= int(warm_up):
                    samples[index].append(elapsed)
        seconds = tuple(min(values) for values in samples)
        rates = [1.0 / value if value > 0.0 else 0.0 for value in seconds]
        total_rate = sum(rates)
        if total_rate <= 0.0:
            raise RuntimeError(
                "Scene.calibrate_devices() measured no time on any device; the probe "
                "did not run."
            )
        throughput = tuple(rate * len(rates) / total_rate for rate in rates)
        weights = throughput
        candidates: tuple[tuple[float, ...], ...] = ()
        candidate_seconds: tuple[float, ...] = ()
        if refine and len(self.devices) > 1:
            candidates, candidate_seconds = self._time_weight_candidates(
                weights, probe, repeats, int(warm_up)
            )
            weights = candidates[_pick_candidate(candidate_seconds)]
        self.weights = weights
        self.last_calibration = DeviceCalibration(
            operation="trace_reflections" if int(max_bounces) > 0 else "intersect",
            rows=rows,
            devices=tuple(device.index for device in self.devices),
            seconds=seconds,
            weights=weights,
            samples=tuple(tuple(values) for values in samples),
            throughput_weights=throughput,
            candidates=candidates,
            candidate_seconds=candidate_seconds,
        )
        return self.last_calibration

    def _time_weight_candidates(self, base, probe, repeats: int, warm_up: int):
        """Time the real dispatch at a ladder of remote shares, smallest last.

        The ladder scales every non-master weight by the same factor, so the
        relative standing the throughput stage measured between the non-master
        devices is preserved and only their share of the batch moves. The last
        rung is zero -- the master alone -- which is what makes this stage a
        floor rather than a gamble.
        """
        master = self.master_device
        candidates: list[tuple[float, ...]] = []
        for share in _REFINE_SHARES:
            scaled = tuple(
                weight if device == master else weight * share
                for weight, device in zip(base, self.devices)
            )
            if sum(scaled) > 0.0 and scaled not in candidates:
                candidates.append(scaled)
        samples: list[list[float]] = [[] for _ in candidates]
        saved_weights = self.weights
        saved_floor = self.min_rays_per_device
        # The probe is deliberately small; the small-batch floor would answer
        # every candidate with the same master-only run and measure nothing.
        self.min_rays_per_device = 1
        try:
            for round_index in range(warm_up + repeats):
                for index, weights in enumerate(candidates):
                    self.weights = weights
                    for device in self.devices:
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    probe(self, master)
                    for device in self.devices:
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    if round_index >= warm_up:
                        samples[index].append(elapsed)
        finally:
            self.min_rays_per_device = saved_floor
            self.weights = saved_weights
        return tuple(candidates), tuple(min(values) for values in samples)

    def _default_probe(self, rows: int, max_bounces: int):
        """A self-contained probe: this scene's own bounding box, filled with rays.

        The rays are drawn on the host from a fixed seed and moved to each
        device, so every device is timed on bit-identical work.
        """
        master_mesh = self.master()._meshes[0][0].vertices.detach()
        low = master_mesh.amin(dim=0).cpu()
        high = master_mesh.amax(dim=0).cpu()
        span = (high - low).clamp_min(1e-3)
        generator = torch.Generator().manual_seed(0x2026072D)
        origins = torch.rand((rows, 3), generator=generator) * span + low
        directions = torch.randn((rows, 3), generator=generator)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        origins = origins.contiguous()
        directions = directions.contiguous()
        resident: dict[int, Ray] = {}

        def probe(replica, device: torch.device) -> None:
            # The rays are staged once per device and reused: a probe that
            # copied 24 MB from the host on every timed run would be measuring
            # the PCIe bus, not the device.
            ray = resident.get(device.index)
            if ray is None:
                ray = Ray(origins.to(device), directions.to(device))
                torch.cuda.synchronize(device)
                resident[device.index] = ray
            if max_bounces > 0:
                replica.trace_reflections(ray, max_bounces, None).valid
            else:
                replica.intersect(ray, None, 0).t

        return probe

    # -- per_ray operations ------------------------------------------------

    def intersect(self, ray: Ray, active, flags: int):
        total = int(ray.o.shape[0])

        def shard(device, start, stop):
            return (
                self._shard_ray(ray, start, stop, device),
                self._slice_rows(active, "active", total, start, stop, device),
                flags,
            )

        def call(replica, arguments):
            return replica.intersect(*arguments)

        def master():
            return self.master().intersect(ray, active, flags)

        if flags == 0:
            # The single-device path answers `flags=0` with the reduced result
            # whose remaining fields are the scene's empty tensors; rebuilding
            # it on the master keeps both the type and that laziness.
            return self._dispatch(
                "intersect",
                total,
                _INTERSECT_REDUCED_ROW_BYTES,
                shard,
                call,
                lambda result: (result.t,),
                lambda rows, columns: _ReducedIntersection(
                    self.master_native_scene(), columns[0]
                ),
                master,
            )
        return self._dispatch(
            "intersect",
            total,
            _INTERSECT_FULL_ROW_BYTES,
            shard,
            call,
            _intersection_fields,
            lambda rows, columns: Intersection(*columns),
            master,
        )

    def nearest_edge(self, point):
        if isinstance(point, Ray):
            total = int(point.o.shape[0])

            def shard_ray(device, start, stop):
                return (self._shard_ray(point, start, stop, device),)

            def call_ray(replica, arguments):
                return replica.nearest_edge(*arguments)

            return self._dispatch(
                "nearest_edge_ray",
                total,
                _NEAREST_RAY_EDGE_ROW_BYTES,
                shard_ray,
                call_ray,
                _field_reader(_NEAREST_RAY_EDGE_FIELDS),
                lambda rows, columns: NearestRayEdge(*columns),
                lambda: self.master().nearest_edge(point),
            )

        total = int(point.shape[0])

        def shard(device, start, stop):
            return (self._slice(point, start, stop, device),)

        def call(replica, arguments):
            return replica.nearest_edge(*arguments)

        return self._dispatch(
            "nearest_edge",
            total,
            _NEAREST_POINT_EDGE_ROW_BYTES,
            shard,
            call,
            _field_reader(_NEAREST_POINT_EDGE_FIELDS),
            lambda rows, columns: NearestPointEdge(*columns),
            lambda: self.master().nearest_edge(point),
        )

    def nearest_edges(self, point: torch.Tensor, k: int, active):
        total = int(point.shape[0])

        def shard(device, start, stop):
            return (
                self._slice(point, start, stop, device),
                k,
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.nearest_edges(*arguments)

        return self._dispatch(
            "nearest_edges",
            total,
            _NEAREST_EDGES_ROW_BYTES_PER_K * max(int(k), 1),
            shard,
            call,
            _field_reader(_NEAREST_EDGES_TOPK_FIELDS),
            lambda rows, columns: NearestEdgesTopK(rows, int(k), *columns),
            lambda: self.master().nearest_edges(point, k, active),
        )

    def visible(self, start_points: torch.Tensor, end_points: torch.Tensor, active):
        total = int(start_points.shape[0])

        def shard(device, start, stop):
            return (
                self._slice(start_points, start, stop, device),
                self._slice(end_points, start, stop, device),
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.visible(*arguments)

        return self._dispatch(
            "visible",
            total,
            _VISIBLE_ROW_BYTES,
            shard,
            call,
            lambda result: (result,),
            lambda rows, columns: columns[0],
            lambda: self.master().visible(start_points, end_points, active),
        )

    def visible_pair(self, start_points, end_a, end_b, ignore_prim_ids, active):
        total = int(start_points.shape[0])

        def shard(device, start, stop):
            return (
                self._slice(start_points, start, stop, device),
                self._slice(end_a, start, stop, device),
                self._slice(end_b, start, stop, device),
                self._slice_rows(
                    ignore_prim_ids, "ignore_prim_ids", total, start, stop, device
                ),
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.visible_pair(*arguments)

        return self._dispatch(
            "visible_pair",
            total,
            _VISIBLE_PAIR_ROW_BYTES,
            shard,
            call,
            _field_reader(_SEGMENT_PAIR_FIELDS),
            lambda rows, columns: SegmentPairVisibility(rows, *columns),
            lambda: self.master().visible_pair(
                start_points, end_a, end_b, ignore_prim_ids, active
            ),
        )

    def visible_edge(
        self,
        source,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        sample_fractions,
        active,
    ):
        total = int(source.shape[0])

        def shard(device, start, stop):
            return (
                self._slice(source, start, stop, device),
                self._slice(edge_position, start, stop, device),
                self._slice(edge_direction, start, stop, device),
                self._slice(edge_t_min, start, stop, device),
                self._slice(edge_t_max, start, stop, device),
                sample_fractions,
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.visible_edge(*arguments)

        return self._dispatch(
            "visible_edge",
            total,
            _VISIBLE_EDGE_ROW_BYTES,
            shard,
            call,
            _field_reader(_AXIAL_EDGE_FIELDS),
            lambda rows, columns: AxialEdgeVisibility(rows, *columns),
            lambda: self.master().visible_edge(
                source,
                edge_position,
                edge_direction,
                edge_t_min,
                edge_t_max,
                sample_fractions,
                active,
            ),
        )

    def visible_chain(self, points, chain_length, ignore_prim_per_segment, active):
        total = int(points.shape[0])
        segments = int(points.shape[1]) - 1

        def shard(device, start, stop):
            return (
                self._slice(points, start, stop, device),
                self._slice(chain_length, start, stop, device),
                self._slice_rows(
                    ignore_prim_per_segment,
                    "ignore_prim_per_segment",
                    total,
                    start,
                    stop,
                    device,
                ),
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.visible_chain(*arguments)

        return self._dispatch(
            "visible_chain",
            total,
            _VISIBLE_CHAIN_ROW_BYTES,
            shard,
            call,
            _field_reader(_SEGMENT_CHAIN_FIELDS),
            lambda rows, columns: SegmentChainVisibility(rows, segments, *columns),
            lambda: self.master().visible_chain(
                points, chain_length, ignore_prim_per_segment, active
            ),
        )

    def trace_reflections(self, ray: Ray, max_bounces: int, active):
        total = int(ray.o.shape[0])
        row_bytes = max_bounces * (
            _REFLECTION_TAPE_BYTES_PER_RAY_BOUNCE
            + _REFLECTION_OUTPUT_BYTES_PER_RAY_BOUNCE
        )

        def shard(device, start, stop):
            return (
                self._shard_ray(ray, start, stop, device),
                max_bounces,
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.trace_reflections(*arguments)

        mode = self._dispatch_mode(total)
        self.last_dispatch = mode
        if mode == "master":
            return self.master().trace_reflections(ray, max_bounces, active)
        if mode == "chunked":
            # A chunk cannot stay lazy under a memory budget: keeping every
            # chunk's device-side trace alive until someone reads
            # `image_sources` is exactly the memory the chunked executor exists
            # to bound, so a chunked chain is materialized in full, one chunk
            # at a time.
            return self._run_chunked(
                "trace_reflections",
                total,
                row_bytes,
                shard,
                call,
                _reflection_fields,
                lambda rows, columns: ReflectionChain(*columns),
            )
        if mode == "pipelined":
            if self._master_takes_everything(total):
                self.last_dispatch = "master"
                return self.master().trace_reflections(ray, max_bounces, active)
            remote_rows, master_rows = self._pipeline_rows(total)

            def load(full: bool):
                # `ReflectionChain` is lazy on one device -- the reduced trace
                # and the image-source trace are separate launches -- and stays
                # lazy here, so a caller who never reads `image_sources` never
                # pays for it on any device. Upgrading a reduced chain re-runs
                # the trace, which is what the single-device loader does too.
                columns = self._run_chunked(
                    "trace_reflections",
                    total,
                    row_bytes,
                    shard,
                    call,
                    _reflection_fields if full else _reduced_reflection_fields,
                    lambda rows, values: values,
                    chunk_rows=remote_rows,
                    master_chunk_rows=master_rows,
                )
                if full:
                    return columns[0], columns[1], columns[2], columns[3]
                return columns[0], columns[1], None, columns[2]

            return ReflectionChain(loader=load)
        results = self._run_shards(total, shard, call)
        if results is None:
            self.last_dispatch = "master"
            return self.master().trace_reflections(ray, max_bounces, active)

        def load(full: bool):
            # Reading the shards in device order keeps both devices enqueued
            # before either is waited on.
            image_sources = None
            if full:
                image_sources = self._gather(
                    [result.image_sources for result in results]
                )
            valid = self._gather([result.valid for result in results])
            t = self._gather([result.t for result in results])
            prim_ids = self._gather([result.prim_ids for result in results])
            return valid, t, image_sources, prim_ids

        return ReflectionChain(loader=load)

    def trace_refl_epc_field(self, source, receiver, max_bounces: int, active):
        total = int(source.shape[0])

        def shard(device, start, stop):
            return (
                self._slice(source, start, stop, device),
                self._slice(receiver, start, stop, device),
                max_bounces,
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.trace_refl_epc_field(*arguments)

        return self._dispatch(
            "trace_refl_epc_field",
            total,
            _REFL_EPC_FIELD_ROW_BYTES + 4 * max_bounces,
            shard,
            call,
            _field_reader(_REFL_EPC_FIELD_FIELDS),
            lambda rows, columns: ReflEpcField(*columns),
            lambda: self.master().trace_refl_epc_field(
                source, receiver, max_bounces, active
            ),
        )

    # -- grid_reduce operations --------------------------------------------

    def _run_lane_shards(
        self,
        operation: str,
        total_samples: int,
        lane_offset: int,
        lane_count: int,
        scatter,
        call,
    ) -> DfrAccum:
        """One launch per (device, chunk) lane window, merged on the master.

        `scatter` replicates the whole-batch inputs onto one device once, and
        `call(replica, inputs, lane_begin, lane_count)` runs that replica's
        accumulation over one window of the lane space. Chunks fold into their
        device's partial grid in ascending lane order before the partials are
        moved and summed in device order, which is the merge order the module
        docstring pins down.
        """
        begin, count = _resolve_lane_window(lane_offset, lane_count, total_samples)
        plan = calibrate_chunk_size(
            operation,
            count,
            row_bytes=_DFR_ACCUM_LANE_BYTES,
            chunk_rays=self.options.chunk_rays,
            budget_bytes=self.options.tape_memory_budget_bytes,
        )
        plan.chunk_rays = _lane_chunk_size(plan.chunk_rays, count)
        self.last_chunk_plan = plan
        shards = self._lane_shards(begin, count)
        if not shards:
            # An empty window is one empty launch on the master, which is what
            # the single-device path does with the very same arguments.
            plan.chunk_count = 1
            return call(self.master(), scatter(self.master_device), begin, count)

        # Scatter every shard's inputs before launching anything, so the devices
        # overlap for the same reason the `per_ray` path's devices do.
        replicas = [replica for replica, _device, _begin, _count in shards]
        inputs = [scatter(device) for _replica, device, _begin, _count in shards]
        queues = []
        for _replica, _device, shard_begin, shard_count in shards:
            shard_stop = shard_begin + shard_count
            queues.append(
                [
                    (chunk_begin, min(plan.chunk_rays, shard_stop - chunk_begin))
                    for chunk_begin in range(shard_begin, shard_stop, plan.chunk_rays)
                ]
            )
        plan.chunk_count = sum(len(chunks) for chunks in queues)

        # Round-robin over the devices so every device has a chunk in flight
        # before any device gets its second one. Each device still folds its own
        # chunks in ascending lane order, which is the merge order that makes a
        # fixed split reproduce itself.
        partials: list = [None] * len(shards)
        for step in range(max(len(chunks) for chunks in queues)):
            for index, chunks in enumerate(queues):
                if step >= len(chunks):
                    continue
                chunk_begin, chunk_count = chunks[step]
                result = call(replicas[index], inputs[index], chunk_begin, chunk_count)
                partials[index] = (
                    result
                    if partials[index] is None
                    else _add_accum(partials[index], result)
                )

        master = self.master_device
        merged = None
        for partial in partials:
            hosted = _accum_to(partial, master)
            merged = hosted if merged is None else _add_accum(merged, hosted)
        return merged

    def accum_dfr_direct(
        self,
        *,
        states: DfrStates,
        grid,
        material: DfrMaterial,
        active,
        wavelength: float,
        direct_samples: int,
        keller_samples: int,
        suffix_samples: int,
        seed: int,
        lane_offset: int,
        lane_count: int,
    ) -> DfrAccum:
        def scatter(device):
            return (
                _states_to(states, device),
                _material_to(material, device),
                _to(active, device),
            )

        def call(replica, moved, begin, count):
            shard_states, shard_material, shard_active = moved
            return replica.accum_dfr_direct(
                states=shard_states,
                grid=grid,
                material=shard_material,
                active=shard_active,
                wavelength=wavelength,
                direct_samples=direct_samples,
                keller_samples=keller_samples,
                suffix_samples=suffix_samples,
                seed=seed,
                lane_offset=begin,
                lane_count=count,
            )

        return self._run_lane_shards(
            "accum_dfr_direct",
            int(direct_samples) + int(keller_samples) + int(suffix_samples),
            lane_offset,
            lane_count,
            scatter,
            call,
        )

    def accum_dfr(
        self,
        *,
        initial_states: DfrStates,
        recursive_states: DfrStates,
        grid,
        material: DfrMaterial,
        active,
        recursive_active,
        wavelength: float,
        direct_samples: int,
        keller_samples: int,
        suffix_samples: int,
        seed: int,
        max_order: int,
        lane_offset: int,
        lane_count: int,
    ) -> DfrAccum:
        def scatter(device):
            return (
                _states_to(initial_states, device),
                _states_to(recursive_states, device),
                _material_to(material, device),
                _to(active, device),
                _to(recursive_active, device),
            )

        def call(replica, moved, begin, count):
            initial, recursive, shard_material, shard_active, shard_recursive = moved
            return replica.accum_dfr(
                initial_states=initial,
                recursive_states=recursive,
                grid=grid,
                material=shard_material,
                active=shard_active,
                recursive_active=shard_recursive,
                wavelength=wavelength,
                direct_samples=direct_samples,
                keller_samples=keller_samples,
                suffix_samples=suffix_samples,
                seed=seed,
                max_order=max_order,
                lane_offset=begin,
                lane_count=count,
            )

        return self._run_lane_shards(
            "accum_dfr",
            int(direct_samples) + int(keller_samples) + int(suffix_samples),
            lane_offset,
            lane_count,
            scatter,
            call,
        )


def _field_reader(names: Sequence[str]):
    """Read one chunk's result into `names` order, for the chunked executor."""

    def read(result) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(result, name) for name in names)

    return read


def _reflection_fields(result) -> tuple[torch.Tensor, ...]:
    """One chunk's reflection chain, in `ReflectionChain.__init__` order.

    `image_sources` is read first because it is the field that forces the full
    trace; the other three come out of the same load.
    """
    image_sources = result.image_sources
    return (result.valid, result.t, image_sources, result.prim_ids)


def _reduced_reflection_fields(result) -> tuple[torch.Tensor, ...]:
    """One chunk's reflection chain without its image sources.

    The reduced trace is a different, cheaper launch than the full one, so a
    pipelined chain that nobody asks image sources of never runs the full one
    -- exactly as on a single device.
    """
    return (result.valid, result.t, result.prim_ids)


def _intersection_fields(result) -> tuple[torch.Tensor, ...]:
    """The ten intersection fields of one shard, in canonical order.

    `p` is read first on purpose: an AD shard hands back the lazy intersection,
    whose `t` is a second launch unless the full record has already been
    materialized.
    """
    p = result.p
    return (
        result.t,
        p,
        result.n,
        result.geo_n,
        result.uv,
        result.barycentric,
        result.shape_id,
        result.prim_id,
        result.local_prim_id,
        result.global_prim_id,
    )


_NEAREST_POINT_EDGE_FIELDS = (
    "distance",
    "edge_point",
    "edge_t",
    "shape_id",
    "edge_id",
    "global_edge_id",
)
_NEAREST_RAY_EDGE_FIELDS = (
    "distance",
    "ray_t",
    "point",
    "edge_t",
    "edge_point",
    "shape_id",
    "edge_id",
    "global_edge_id",
)
_NEAREST_EDGES_TOPK_FIELDS = (
    "is_valid",
    "distances",
    "points",
    "edge_t",
    "edge_points",
    "shape_ids",
    "edge_ids",
    "global_edge_ids",
    "is_boundary",
)
_SEGMENT_PAIR_FIELDS = ("visible_a", "visible_b")
_AXIAL_EDGE_FIELDS = ("any_visible",)
_SEGMENT_CHAIN_FIELDS = (
    "all_visible",
    "first_blocked_segment",
    "first_blocked_prim",
)
_REFL_EPC_FIELD_FIELDS = (
    "field_real",
    "field_imag",
    "path_length",
    "valid",
    "resolved_prim_ids",
)

# Every `DfrAccum` payload is a partial: the float grids merge by float32
# summation, the integer counters merge exactly.
_DFR_ACCUM_FIELDS = (
    "power",
    "field_x_re",
    "field_x_im",
    "field_y_re",
    "field_y_im",
    "field_z_re",
    "field_z_im",
    "direct_count",
    "keller_count",
    "suffix_count",
    "vis_rejects",
    "edge_vis_rejects",
    "utd_rejects",
    "edge_uses",
)
