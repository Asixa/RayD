"""Private replicated multi-device orchestration for `Scene(devices=[...])`.

This is Phases 2a, 2b and the grid-reduce half of 2c of
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
`_warmup._DEVICE_WORK_LOCK`). Overlap instead comes from the asynchrony that
is already there -- every shard's inputs are scattered first, then each
replica's op is enqueued in device order on that device's current stream, so
the devices run concurrently while the host runs ahead.

That is correctness-shaped, not yet throughput-shaped. Torch runs a
cross-device copy on the *source* device's current stream, so on the unchunked
path both the scatter to the other devices and the gather back sit on the
master's stream, in front of the master's own compute; measured on 2x RTX A6000
an 8.4M-ray `trace_reflections` lands at 0.77x of one device and a bare
`intersect`, whose compute is smaller than its transfer, at 0.27x. The
throughput gate itself is Phase 2d; nothing here should be read as a speed-up
claim.

Chunked execution (Phase 2b)
----------------------------

`MultiDeviceOptions.chunk_rays`, `.tape_memory_budget_bytes` and `.offload`
turn one launch per shard into a stream of chunks per shard. Each device gets
two streams: chunk `k`'s gather runs on the copy stream while chunk `k+1`'s
compute already runs on the compute stream, and the two are ordered by events,
never by a device or host synchronization. The master's stream is made to wait
on a chunk's gather event before anything reads it, which is the only ordering
a caller can observe.

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

Chunking buys memory, not speed: every chunk is one more native call, and a
native call's host cost does not shrink with its batch (measured on one RTX
A6000, a 4M-ray 3-bounce `trace_reflections` costs 0.31 ms of host time and
0.92 ms of wall time; split into 16 chunks the same batch costs 3.8 ms, host
bound). That is what `calibrate_chunk_size()` is for -- the right chunk is the
largest one the budget allows, never the smallest one that fits.

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

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

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
    """

    weights: Sequence[float] | None = None
    warm_up: bool = True
    chunk_rays: int | None = None
    offload: Callable[[int, object], None] | None = None
    tape_memory_budget_bytes: int | None = None


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
    if len(indices) == 1 and not chunking:
        return None
    return _ReplicatedScene(
        indices, resolved, weights, trace_backend, edge_bvh_backend, chunking
    )


class _ReplicatedScene:
    """One ordinary single-device `Scene` per device, plus the batch sharder."""

    __slots__ = (
        "devices",
        "options",
        "weights",
        "chunked",
        "last_chunk_plan",
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
        self.last_chunk_plan: ChunkPlan | None = None
        self._trace_backend = trace_backend
        self._edge_bvh_backend = edge_bvh_backend
        self._replicas: tuple = ()
        self._streams: dict[int, tuple] = {}
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

    def _gather_fields(self, results: list, names: Sequence[str]) -> list[torch.Tensor]:
        return [self._gather([getattr(result, name) for result in results]) for name in names]

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

    def _chunk_streams(self, device: torch.device) -> tuple:
        """This device's (compute, copy) pair, created once and reused."""
        pair = self._streams.get(device.index)
        if pair is None:
            with torch.cuda.device(device):
                pair = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
            self._streams[device.index] = pair
        return pair

    def _run_chunked(
        self,
        operation: str,
        total: int,
        row_bytes: int,
        shard,
        call,
        extract,
        assemble,
    ):
        """One launch per chunk, double-buffered per device (D7).

        `extract` reads a chunk's result into its canonical field order and
        `assemble` rebuilds the public result from gathered fields, so the same
        loop serves the concatenating and the streaming mode. Ordering is
        event-only: the copy stream waits for the chunk's compute, the master
        stream waits for the chunk's gather, and nothing waits for the host.
        """
        plan = calibrate_chunk_size(
            operation,
            total,
            row_bytes=row_bytes,
            chunk_rays=self.options.chunk_rays,
            budget_bytes=self.options.tape_memory_budget_bytes,
        )
        self.last_chunk_plan = plan
        offload = self.options.offload
        queues = [
            (
                replica,
                device,
                [
                    (chunk_start, min(chunk_start + plan.chunk_rays, stop))
                    for chunk_start in range(start, stop, plan.chunk_rays)
                ],
            )
            for replica, device, start, stop in self._shards(total)
        ]
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
        entry = torch.cuda.Event()
        entry.record(master_stream)

        # Two chunks per device in flight is the double buffer: the chunk being
        # gathered plus the chunk being computed.
        depth = 2 * len(queues)
        pipeline: deque = deque()
        collected: list = []
        started: set = set()
        for replica, device, (chunk_start, chunk_stop) in order:
            compute, copy = self._chunk_streams(device)
            if device.index not in started:
                # Our streams pick up where the caller's stream left off: the
                # inputs were produced there.
                compute.wait_event(entry)
                copy.wait_event(entry)
                started.add(device.index)
            with torch.cuda.device(device), torch.cuda.stream(compute):
                self._active_stream = compute
                try:
                    columns = list(
                        extract(call(replica, shard(device, chunk_start, chunk_stop)))
                    )
                finally:
                    self._active_stream = None
            computed = torch.cuda.Event()
            computed.record(compute)
            copy.wait_event(computed)
            with torch.cuda.device(device), torch.cuda.stream(copy):
                moved = []
                for column in columns:
                    column.record_stream(copy)
                    moved.append(column.to(master, non_blocking=True))
            gathered = torch.cuda.Event()
            gathered.record(copy)
            pipeline.append((chunk_start, chunk_stop, moved, gathered))
            del columns, moved
            if len(pipeline) > depth:
                self._retire(pipeline.popleft(), plan, master_stream, assemble, collected)
        while pipeline:
            self._retire(pipeline.popleft(), plan, master_stream, assemble, collected)

        if offload is not None:
            return None
        collected.sort(key=lambda piece: piece[0])
        width = len(collected[0][1])
        return assemble(
            total,
            [
                self._concat([piece[1][index] for piece in collected])
                for index in range(width)
            ],
        )

    def _retire(self, piece, plan: ChunkPlan, master_stream, assemble, collected) -> None:
        """Hand one finished chunk to the hook, or keep it for the gather."""
        chunk_start, chunk_stop, moved, gathered = piece
        # The chunk ran on a side stream; whoever reads it next reads it from
        # the master's stream, so that is the one edge that has to exist.
        master_stream.wait_event(gathered)
        for value in moved:
            value.record_stream(master_stream)
        rows = chunk_stop - chunk_start
        if plan.measured_row_bytes is None and rows > 0:
            plan.measured_row_bytes = (
                sum(value.numel() * value.element_size() for value in moved) / rows
            )
        if self.options.offload is None:
            collected.append((chunk_start, moved))
            return
        self.options.offload(chunk_start, assemble(rows, moved))

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

        if self.chunked:
            if flags == 0:
                return self._run_chunked(
                    "intersect",
                    total,
                    _INTERSECT_REDUCED_ROW_BYTES,
                    shard,
                    call,
                    lambda result: (result.t,),
                    lambda rows, columns: _ReducedIntersection(
                        self.master_native_scene(), columns[0]
                    ),
                )
            return self._run_chunked(
                "intersect",
                total,
                _INTERSECT_FULL_ROW_BYTES,
                shard,
                call,
                _intersection_fields,
                lambda rows, columns: Intersection(*columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().intersect(ray, active, flags)
        if flags == 0:
            # The single-device path answers `flags=0` with the reduced result
            # whose remaining fields are the scene's empty tensors; rebuilding
            # it on the master keeps both the type and that laziness.
            return _ReducedIntersection(
                self.master_native_scene(), self._gather([result.t for result in results])
            )
        columns = [_intersection_fields(result) for result in results]
        return Intersection(*(self._gather(list(parts)) for parts in zip(*columns)))

    def nearest_edge(self, point):
        if isinstance(point, Ray):
            total = int(point.o.shape[0])

            def shard_ray(device, start, stop):
                return (self._shard_ray(point, start, stop, device),)

            def call_ray(replica, arguments):
                return replica.nearest_edge(*arguments)

            if self.chunked:
                return self._run_chunked(
                    "nearest_edge_ray",
                    total,
                    _NEAREST_RAY_EDGE_ROW_BYTES,
                    shard_ray,
                    call_ray,
                    _field_reader(_NEAREST_RAY_EDGE_FIELDS),
                    lambda rows, columns: NearestRayEdge(*columns),
                )
            results = self._run_shards(total, shard_ray, call_ray)
            if results is None:
                return self.master().nearest_edge(point)
            return NearestRayEdge(*self._gather_fields(results, _NEAREST_RAY_EDGE_FIELDS))

        total = int(point.shape[0])

        def shard(device, start, stop):
            return (self._slice(point, start, stop, device),)

        def call(replica, arguments):
            return replica.nearest_edge(*arguments)

        if self.chunked:
            return self._run_chunked(
                "nearest_edge",
                total,
                _NEAREST_POINT_EDGE_ROW_BYTES,
                shard,
                call,
                _field_reader(_NEAREST_POINT_EDGE_FIELDS),
                lambda rows, columns: NearestPointEdge(*columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().nearest_edge(point)
        return NearestPointEdge(*self._gather_fields(results, _NEAREST_POINT_EDGE_FIELDS))

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

        if self.chunked:
            return self._run_chunked(
                "nearest_edges",
                total,
                _NEAREST_EDGES_ROW_BYTES_PER_K * max(int(k), 1),
                shard,
                call,
                _field_reader(_NEAREST_EDGES_TOPK_FIELDS),
                lambda rows, columns: NearestEdgesTopK(rows, int(k), *columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().nearest_edges(point, k, active)
        return NearestEdgesTopK(
            total, int(k), *self._gather_fields(results, _NEAREST_EDGES_TOPK_FIELDS)
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

        if self.chunked:
            return self._run_chunked(
                "visible",
                total,
                _VISIBLE_ROW_BYTES,
                shard,
                call,
                lambda result: (result,),
                lambda rows, columns: columns[0],
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().visible(start_points, end_points, active)
        return self._gather(results)

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

        if self.chunked:
            return self._run_chunked(
                "visible_pair",
                total,
                _VISIBLE_PAIR_ROW_BYTES,
                shard,
                call,
                _field_reader(_SEGMENT_PAIR_FIELDS),
                lambda rows, columns: SegmentPairVisibility(rows, *columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().visible_pair(
                start_points, end_a, end_b, ignore_prim_ids, active
            )
        return SegmentPairVisibility(
            total, *self._gather_fields(results, _SEGMENT_PAIR_FIELDS)
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

        if self.chunked:
            return self._run_chunked(
                "visible_edge",
                total,
                _VISIBLE_EDGE_ROW_BYTES,
                shard,
                call,
                _field_reader(_AXIAL_EDGE_FIELDS),
                lambda rows, columns: AxialEdgeVisibility(rows, *columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().visible_edge(
                source,
                edge_position,
                edge_direction,
                edge_t_min,
                edge_t_max,
                sample_fractions,
                active,
            )
        return AxialEdgeVisibility(
            total, *self._gather_fields(results, _AXIAL_EDGE_FIELDS)
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

        if self.chunked:
            return self._run_chunked(
                "visible_chain",
                total,
                _VISIBLE_CHAIN_ROW_BYTES,
                shard,
                call,
                _field_reader(_SEGMENT_CHAIN_FIELDS),
                lambda rows, columns: SegmentChainVisibility(rows, segments, *columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().visible_chain(
                points, chain_length, ignore_prim_per_segment, active
            )
        return SegmentChainVisibility(
            total,
            segments,
            *self._gather_fields(results, _SEGMENT_CHAIN_FIELDS),
        )

    def trace_reflections(self, ray: Ray, max_bounces: int, active):
        total = int(ray.o.shape[0])

        def shard(device, start, stop):
            return (
                self._shard_ray(ray, start, stop, device),
                max_bounces,
                self._slice_rows(active, "active", total, start, stop, device),
            )

        def call(replica, arguments):
            return replica.trace_reflections(*arguments)

        if self.chunked:
            # A chunk cannot stay lazy: keeping every chunk's device-side trace
            # alive until someone reads `image_sources` is exactly the memory
            # the chunked executor exists to bound, so a chunked chain is
            # materialized in full, one chunk at a time.
            return self._run_chunked(
                "trace_reflections",
                total,
                max_bounces
                * (
                    _REFLECTION_TAPE_BYTES_PER_RAY_BOUNCE
                    + _REFLECTION_OUTPUT_BYTES_PER_RAY_BOUNCE
                ),
                shard,
                call,
                _reflection_fields,
                lambda rows, columns: ReflectionChain(*columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().trace_reflections(ray, max_bounces, active)

        def load(full: bool):
            # `ReflectionChain` is lazy on one device -- the reduced trace and
            # the image-source trace are separate launches -- and stays lazy
            # here, so a caller that never reads `image_sources` never pays for
            # it on any device. Reading the shards in device order keeps both
            # devices enqueued before either is waited on.
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

        if self.chunked:
            return self._run_chunked(
                "trace_refl_epc_field",
                total,
                _REFL_EPC_FIELD_ROW_BYTES + 4 * max_bounces,
                shard,
                call,
                _field_reader(_REFL_EPC_FIELD_FIELDS),
                lambda rows, columns: ReflEpcField(*columns),
            )
        results = self._run_shards(total, shard, call)
        if results is None:
            return self.master().trace_refl_epc_field(
                source, receiver, max_bounces, active
            )
        return ReflEpcField(*self._gather_fields(results, _REFL_EPC_FIELD_FIELDS))

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
