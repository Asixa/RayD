# ADR-0038: Replicated multi-device and chunked execution

- Status: Accepted; Python source-owner paths superseded by ADR-0040
- Date: 2026-07-27
- Decision ID: `replicated-multi-device-execution`
- Scope: how RayD uses more than one GPU -- scene replication and batch
  sharding, the Torch backend's single-process multi-device layer, the
  Dr.Jit backend's process-per-GPU route, the merge layer's float semantics,
  the Monte-Carlo lane window, chunked execution, and the invariance of
  single-device execution

## Context

RayD's target multi-GPU workload is extremely large ray and sample batches
against scenes that fit in one GPU's memory. That workload selects its own
regime: every device holds a full scene replica and work shards along the batch
axis. Sharing one acceleration structure across devices over NVLink traverses
several times slower than a replica, and the scenes in question do not need the
memory, so partitioning geometry would buy nothing and cost a ray-forwarding
engine.

Before this work RayD had no multi-GPU story at all, and two separate reasons
for it. The first was correctness: roughly thirty Torch CUDA translation units
launched on the ambient device rather than on the scene's, several RF families
checked `is_cuda()` without checking the device index, and the Dr.Jit backend
kept no record of the device a scene or pipeline was built on, so a query after
`set_device()` corrupted state instead of failing. The second was that even
once every entry point was device-correct, splitting a batch across devices,
gathering it, reducing gradients, and keeping replicas consistent was caller
code that RayD gave no help with -- and two of RayD's operation families are
not shardable by slicing at all: diffraction accumulation has no batch axis,
only a Monte-Carlo lane space, and the path exporters place rows by a rule that
depends on the whole launch.

The implementation plan this record governs is
[`docs/dev/multi_gpu_plan.md`](../dev/multi_gpu_plan.md), and the operational
contract and the measured performance are in
[`docs/dev/multi_gpu_operations.md`](../dev/multi_gpu_operations.md). Those
documents own the phase breakdown, the recipes, and the numbers; this record
owns the decisions and the invariants. It is written after the implementation
landed (Phases 0 through 3, commits `f643336`..`2ee6e67`), so every claim below
describes shipped code rather than an intention.

## Decision

### 1. Scene replication and batch sharding (D1)

Every device holds a full scene replica: its own GAS/IAS, edge BVH, compact
BVH, SoA buffers and edge mask. Work shards along the batch axis, never along
the geometry. A replica is an ordinary single-device `Scene` object, built from
the master's mesh tensors; the orchestration layer owns a tuple of them and
nothing else.

The split is static and weighted: one non-negative weight per device in
`devices` order, defaulting to an equal split. A weight is a ratio rather than
a count, so `[9.0, 1.0]` and `[0.9, 0.1]` mean the same thing, and a zero
weight is legal and leaves that device idle. There is no dynamic rebalancing:
a re-split mid-batch would change which lanes a device runs, and reproducibility
at fixed inputs is worth more than the last few percent of load balance.

`devices[0]` is the **master**. Every mesh tensor a replicated `Scene` is built
from must live on it -- replicas are `master.to(device_k)` (section 4) -- and
every result and every gradient is returned there, so the caller sees one
device.

Replica mutation is broadcast-only: `update_mesh_vertices`, `sync` and
`set_edge_mask` run on every replica, and the scene versions are checked in
lockstep afterwards. Divergence raises rather than being repaired, because a
replica set that disagrees about its geometry answers different questions on
different devices and there is no safe way to guess which one the caller meant.

### 2. Backend split: Torch in-process, Dr.Jit process-per-GPU (D2)

The **Torch backend** gets first-class single-process multi-device execution
through `Scene(devices=[...])`, plus the process-per-GPU recipe.

The **Dr.Jit backend** is process-per-GPU only. Dr.Jit is single-device per
process, so a second device needs a second process with
`CUDA_VISIBLE_DEVICES` pinned to it. RayD does not work around that, and no
in-process Dr.Jit multi-device layer exists or is planned here. What Phase 0
added on that side is a hard failure instead of corruption: `Scene` records its
build device and every public query asserts the current device matches, and
`OptixLaunchPipeline` records its build device and asserts it at launch.

The capability is therefore declared `drjit: false, torch: true` in
`shared/contracts/public_api.json`. `false` means "no in-process multi-device
API"; it does not mean the Dr.Jit backend cannot be run on many GPUs, which the
process-per-GPU recipe does for both backends.

### 3. Single-device numerics are untouched; merge-layer float semantics (D3)

Multi-device execution is a **composition layer above unchanged kernels**. No
kernel launch shape, no reduction order, no atomic, and no compile flag changes
because a scene has several devices. The frozen guarantees of ADR-0026,
ADR-0030, ADR-0032 and ADR-0035 are untouched, and the only kernel-visible
change in the whole of this work is the lane window of section 5, which this
record governs.

Everything cross-device or cross-chunk happens in the orchestration layer, with
its own stated float32 order:

- On the default homogeneous topology, a `per_ray` operation's shards and
  chunks are **concatenated**, never summed.
  Row `i` of the gathered result is bitwise the row a single-device call
  produces, on every field, at any weighting and any chunk count, because it is
  the same kernel on the same rows. Explicit heterogeneous opt-in carries no
  cross-device bitwise guarantee.
- A `grid_reduce` operation's per-(device, chunk) partial grids are summed:
  chunks accumulate into their device's running partial in ascending lane order
  on that device, and the per-device partials are then moved to the master and
  summed there in `devices` order. Integer counters merge exactly; the float
  fields merge in float32, so a merged grid equals the single-launch grid only
  **up to float32 summation order**. The deviation grows with the number of
  shards and chunks, not with the sample count: measured at 6.5e-08 (light) and
  2.9e-07 (compute) relative deviation on the benchmark configurations.
- Nothing in the merge re-associates a *within-launch* reduction. A launch
  still runs exactly the atomics the single-device kernel has always run.

At fixed `(devices, weights, chunking, inputs)` the merge order is fixed, so a
multi-device run reproduces itself as exactly as a single-device run does.

### 4. Gradient reduction through recorded `.to()` edges (D4)

Replica vertices are produced as `master.to(device_k)`, which is an ordinary
autograd-recorded copy. Torch therefore reduces every replica's
`grad_vertices` back onto the master leaf with no reduction code of RayD's own,
and a distributed run gets the same thing through DDP's NCCL reduction. On the
master device `.to()` is the identity, so the master replica shares the
caller's tensors and runs exactly what a single-device `Scene` runs.

Chunked execution accumulates gradients additively across chunks. That is
ordinary gradient accumulation and it is exact for RayD geometry gradients,
which land in `grad_vertices` by summation; only the float32 summation order
differs from the unchunked backward. The same holds for the sharded
`grid_reduce` backward, which runs one native backward per (device, chunk)
launch against that launch's own tape.

No gradient is ever detached, zeroed or approximated to make a shard work. An
operation whose derivative cannot be reduced this way is not sharded at all
(section 6).

### 5. Diffraction lane windows (D5)

Diffraction accumulation has no batch axis. Its cost is a Monte-Carlo lane
space of `direct_samples + keller_samples + suffix_samples` lanes and its
output is a grid whose size is independent of that count, so it shards along
the **lane axis**. Sharding it by re-seeding, or by launching each shard with a
smaller sample count, would make every shard redraw lanes `0..M/K-1` of the same
sequence; that is a different estimator, and it is forbidden.

The mechanism is a lane window, and it is the only kernel-visible change this
record introduces:

- `DfrAccumParams` carries `lane_offset`. `n_rays` keeps its meaning -- the size
  of the **global** lane space -- and local lane `l` of a launch runs global
  lane `lane_offset + l`. Per-lane buffers (`sample_*`, `temp_visibility`,
  `tape_*`, `stage_*`) stay local: the host rebases their pointers by
  `-lane_offset` so a global lane addresses the local slot, and only lanes in
  `[lane_offset, lane_offset + launch width)` run, so a rebased pointer is never
  dereferenced outside its buffer.
- The public parameters are `lane_offset` (default `0`) and `lane_count`
  (default `-1`, meaning every remaining lane), on `Scene.accum_dfr_direct`,
  `Scene.accum_dfr`, `Scene.accum_dfr_coherent_direct`, their applicable
  `autograd` entry points, and the
  `diffraction_accumulation_forward` dispatcher schema. The four AD ops
  (`diffraction_accumulation_direct_backward`, `_direct_jvp`,
  `_chain_backward`, `_chain_jvp`) take `lane_offset` alone, because their
  launch width is the tape they were handed.
- `resolve_lane_window(lane_offset, lane_count, total_samples)` is the one
  definition of the window, and the Python orchestrator carries a host-side twin
  of it with identical messages so a multi-device scene rejects exactly what a
  single-device scene rejects. A negative `lane_offset`, a `lane_offset` past
  the total, and a `lane_offset + lane_count` past the total are errors.
- `direct_samples`, `keller_samples` and `suffix_samples` always describe the
  global lane space. A shard passes the caller's counts unchanged and selects a
  window of them, so a local lane runs the global lane the single launch would
  have run.
- Coherent direct accumulation is forward-only and uses the same window rule
  over its deterministic `state_count * grid_cell_count` lane space. Each
  shard keeps the full states, material, and grid description; partial complex
  fields and integer diagnostics are summed on the master. Its hard-budget
  estimate charges the staged route's complete 36 bytes per lane (one int32
  key plus eight float32 values); smaller non-staged launches retain that
  conservative charge.
- **`lane_offset = 0` with the default `lane_count` is bitwise the pre-ADR
  launch.** The pointer rebase is a no-op at zero offset and the window resolves
  to the whole space, so the unsharded path executes the code it always did.
- `lane_offset != 0` requires the OptiX trace backend and raises on the
  pure-CUDA one, which has no windowed launch.

Replicated accumulation with the CUDA trace backend therefore executes a
zero-offset caller window once on the master. A requested chunk or memory
budget that would split that window fails before launch; the orchestrator never
submits a first shard and then discovers that the next offset is unsupported.

**Contract.** A K-way split into contiguous, disjoint windows that covers the
caller's window exactly draws exactly the samples the single launch draws; the
multiset of drawn samples is independent of the split. Merging remains
float-order-dependent under section 3.

**Warp-multiple caveat.** Diffraction grid accumulation aggregates a warp's contributions
before its atomic, and a partially filled warp already drops contributions on a
single device -- a plain `direct_samples = 20` launch accumulates fewer samples
than it tapes, which predates sharding and is not corrected here. An unaligned
split would drop a fraction of a warp per shard *in addition* to that. Every
shard and chunk boundary the orchestrator cuts is therefore aligned to the
32-lane warp relative to the caller's window: diffraction shard boundaries round to the
nearest warp, explicitly requested chunk sizes round up to a whole warp, and
budget-derived chunk sizes round down so the hard memory limit is not exceeded.
The post-alignment size is checked against the budget; a budget that cannot hold
one safe warp fails loudly. Every shard therefore sees the same warp partition
the single launch sees, including its trailing partial warp. A window narrower
than one warp per device is legal and simply leaves the
leading devices idle. **Merged-grid equality with a single launch is claimed
only for warp-multiple windows.**

### 6. Batch-coupled operations get explicit semantics or they fail (D6)

An operation whose result depends on the *whole* batch rather than on each row
is not sharded by slicing. RayD's rule is that such an operation either gets a
written per-shard semantics or refuses to run on a multi-device scene; it never
silently changes meaning.

In the shipped layer:

- `trace_dfr_paths` requires `SourceLane` on a multi-device scene and shards
  whole transmitter blocks while retaining the full receiver and state axes.
  Each shard therefore emits one contiguous slice of the global
  `((tx * rx_count + rx) * state_limit) + state` row space. Concatenation in
  transmitter order is row-for-row identical to a single SourceLane launch;
  valid local transmitter IDs are offset to their global IDs, and device counts
  are summed without a host read or compaction. Explicit `Compact` requests,
  chunking, and offload fail loudly instead of changing placement.
- `accum_dfr_coherent_direct` uses the deterministic lane window from section 5
  and the ordinary `grid_reduce` merge order from section 3.
- `accumulate_reflections` shards warp-aligned ray-batch windows. Each launch
  retains the unsplit batch's atomic/staged strategy, and the seven float grids
  plus `reflection_count` are summed on the master in device order.
  `WedgeEvents` is deliberately not reduced: its bounded atomic buffer has one
  global capacity, overflow count, slot order, and ray-index space. Enabling
  wedge collection therefore runs one complete master launch; combining it
  with chunking, a memory budget, or offload fails loudly.
- Multi-device `Scene.trace_refl_epc` calls still raise `NotImplementedError`.
- Reflection-chain deduplication is a separate batch-coupled Torch op
  (`reflection_dedup_forward`) over a batch of chains, and the Dr.Jit
  `ReflectionTraceOptions.deduplicate` path is process-per-GPU only. The
  replicated layer wraps neither, so dedup keeps its single-launch meaning.
- The ADR-0033 segment-penetration family has no Python binding and is not
  reachable from a `Scene` at all; its failure transaction stays per-launch and
  is untouched here.

Extending any of these to several devices is a decision, not an implementation
detail; the deferred list below records them as such.

### 7. Chunked execution and offload (D7)

At extreme batch sizes the binding constraint is tape and output memory rather
than scene memory -- the reflection tape alone runs 40-50 bytes per ray per
bounce -- so chunked execution is a first-class component rather than a fallback.

- `MultiDeviceOptions.chunk_rays` sets the rows per launch verbatim for
  `per_ray` operations and reflection ray-batch accumulation. Diffraction
  grid-reduce operations round it up to a 32-lane sample boundary so a shard
  does not split their native warp reduction;
  `tape_memory_budget_bytes` asks for the largest chunk whose tape and outputs
  fit a budget; `offload` streams per-ray results instead of concatenating them.
  Any of the three engages the chunked executor, **including on a one-device
  scene**, because a per-device memory bound is real with one device too.
- A chunking knob is a **memory contract** and is honoured at every batch size.
  It outranks the small-batch floor of section 10, which is only a throughput
  heuristic.
- `tape_memory_budget_bytes` is a per-device peak-increment contract. The
  estimator accounts for the executor's maximum of three resident chunks and,
  when results are concatenated, first reserves the complete returned output.
  Per-row copied inputs, outputs and frozen tape are all charged; accumulation
  also reserves replicated state/material inputs and its fixed-size grid
  partials. CUDA allocator granularity, the already-resident scene/caller
  tensors, and allocations made by a user `offload` hook are outside that
  estimate.
  A request whose fixed output already exceeds the budget fails and points the
  caller to `offload`; it does not launch with a knowingly false bound.
- Inference `grid_reduce` chunks reuse one fixed-size partial grid, so their
  value-buffer memory is O(1) in the sample count. That statement does **not**
  apply to autograd: ordinary additions retain every chunk's frozen native tape
  until backward, so an AD grid request that would need several chunks under a
  budget fails loudly. Likewise, a chunked `per_ray` call that would retain a
  multi-chunk graph must use `offload` and perform backward per chunk.
  `per_ray` inference chunks are concatenated on the master, or handed to
  `offload(chunk_start_row, chunk_result)` with the chunk's fields already on
  the master, in which case the operation returns `None`. Chunks arrive in
  issue order -- ascending rows per device, interleaved across devices -- so a
  hook must use `chunk_start_row` rather than assume a front-to-back walk.
- Ordering is expressed with events on private streams, never with a device or
  host synchronization: chunk `k`'s gather runs while chunk `k+1` computes. The
  executor enters by making each device's compute stream wait on an event
  recorded on that device's current stream *and* on the master's, and leaves by
  making each device's current stream wait on its compute stream, so a query
  issued straight after `sync()` cannot traverse a half-rebuilt structure and a
  mutation cannot overwrite geometry a shard is still traversing.
- A chunked `per_ray` result is bitwise the unchunked one at any chunk size,
  because chunking a `per_ray` operation only changes how its rows are grouped.

Chunking buys memory, not speed: every chunk is one more native call, and a
native call's host cost does not shrink with its batch. The right chunk is the
largest one the budget allows, which is what `calibrate_chunk_size()` returns.

### 8. Multi-GPU is invisible at the top-level API (D8)

There is no parallel public surface, no `MultiScene`, and no per-operation
multi-device variant. `Scene` gains one optional `devices=` argument and one
optional `options=` record; every existing operation shards and chunks
transparently when the scene was built with several devices.

The orchestration machinery -- replicas, sharder, chunked executor, calibration
-- lives in the private module `python/rayd/_impl/multi.py` and is not public
API. The public
surface this record adds is exactly:

- `Scene(devices=[...], options=...)`;
- `rayd.torch.MultiDeviceOptions`, an additive frozen dataclass whose every
  field is defaulted;
- `Scene.calibrate_devices(...)`, the latest/base `Scene.device_weights`
  property, and `Scene.device_weights_for(operation)` for the effective
  operation-local split;
- `lane_offset` / `lane_count` on the three accumulation entry points (section 5).

`MultiDeviceOptions` defaults are `weights=None` (equal split),
`operation_weights=None`, `require_peer_access=True`,
`require_homogeneous_devices=True`, `warm_up=True`,
`chunk_rays=None`, `offload=None`,
`tape_memory_budget_bytes=None`, `pipeline=True`,
`pipeline_chunks_per_device=4`, `min_rays_per_device=262144` and
`min_lanes_per_device=262144`. The throughput numbers are measured, not
guessed (section 10).

Process-per-GPU and distributed execution need no API change at all: each rank
builds a rank-local single-device scene.

### 9. Zero single-GPU regression (D9)

A single-device `Scene` executes today's code path, not a degenerate case of the
multi-device one:

- `rayd.torch._multi` is imported **only** when `devices=` is passed. A default
  `Scene()` never imports it, which a subprocess probe asserts.
- `plan()` returns `None` for a one-device `Scene(devices=[d])` that wants
  nothing from the chunked executor, so such a scene is the pre-existing path
  verbatim and is bitwise equal to `Scene()` on every wrapped operation.
- Every operation's multi-device branch is one `if self._multi is not None`
  comparison. There is no wrapper indirection, no extra synchronization and no
  allocation on the single-device hot path.

**The Phase 0 device guards are the only change this work makes to the
single-GPU path.** A `c10::cuda::CUDAGuard` at each op entry, a same-device-index
check in the RF families that previously checked only `is_cuda()`, occupancy
queried under a guard for the requested device, and device-explicit stream and
AABB helpers. A guard whose device is already current is a cheap no-op, kernels
and launch counts are unchanged, and the ADR-0026 pinned RF source hashes were
refreshed as an intentional identity update with the numerics unchanged.

The single-GPU parity gate is the `benchmark_torch_native.py` set against the
pre-Phase-2 baseline (`cc5f0f9`), six interleaved runs of each, minimum per
metric: six of eight metrics within +/-1.6%, with `nearest_edge` at -3.8%
(faster, on a metric whose per-run spread on this host is +/-10%) and
`diffraction_direct` at +2.8% of a 0.19 ms measurement whose samples spanned
0.20-1.70 ms. Since a single-device scene never reaches the layer at all, that
is a measurement of unchanged code and the residual is the machine.

### 10. Small-batch fallback and calibration semantics

Two mechanisms keep the layer from being slower than not having it, and they
carry **different strengths**. The difference is part of the contract.

**The work floor is a guarantee.** The layer computes the target operation's
weighted remote shard, then compares that actual shard with a transfer-aware
floor derived from `min_rays_per_device` and the operation's copied input plus
returned output bytes per row. A highly skewed split therefore cannot launch
a token remote shard merely because the total batch is large. `grid_reduce`
applies the analogous
`min_lanes_per_device` floor to the actual remote lane windows. Below either
floor the operation runs on the master replica, through the same code a
single-device `Scene` runs. An explicit chunking knob outranks the per-ray
throughput floor (section 7).

`operation_weights` may override the base `weights` for named operations.
Exact keys such as `trace_reflections:4` win over family keys. Calibration
updates only the operation it measured; a custom probe therefore requires an
explicit `operation=...`, and the built-in probe cannot be relabelled as a
different operation. It cannot silently change the split used by unrelated
query or accumulation families.

**Calibration is a measurement, not a guarantee.** `calibrate_devices()` runs a
throughput stage (the same probe on every replica with resident inputs, so the
device is timed rather than the interconnect) and then a refinement stage that
times the real multi-device dispatch while scaling every non-master weight
through `1, 1/2, 1/4, 1/10, 0` and keeps the **largest** remote share within 3%
of the fastest rung. Weights of `[1.0, 0.0]` are dispatched as the single-device
call they are, at any batch size, unless a chunking knob asked otherwise. The
exact claim is:

> Calibration will not knowingly keep a split that it measured as more than the
> refinement tolerance (3%) slower than the master alone.

It is **not** "calibration cannot leave you slower than one GPU". On an
operation whose sharded and master-only rungs sit inside that band the tie-break
decides, and run-to-run noise puts the answer on either side: light 1-bounce
`trace_reflections` (0.67 ns/ray of transfer against 0.27 ns/ray of compute)
kept a quarter remote share in 3 of 20 consecutive benchmark runs and ran at
0.86x, 0.85x and 0.38x, and in 5 of 6 back-to-back calibrations at 0.79-0.89x.
A contended throughput stage is the other failure shape: two of those 20 runs
weighted an identical `cuda:1` at 0.72 and 0.81 and ran a compute-bound
`intersect` at 1.42x and 1.54x instead of 1.61x. The remedy is a configuration
value, not a code path: pin `MultiDeviceOptions(weights=[1.0, 0.0])` where the
benchmark prints `NEAR-CROSSOVER`, and calibrate on a quiet machine. Calibration
only chooses weights; at fixed weights execution stays exactly as reproducible
as it was.

### 11. Shardability classification

Every operation or explicitly enumerated operation variant belongs to exactly
one concrete class, and that class decides how (and whether) it shards. The
classes are declared in `contracts/operations.json` as
`shardability_classes`, and every operation entry carries a `shardability`
block naming its family class and what the Torch replicated layer does with it.
Families whose variants differ use `variant_specific` plus a complete
`variant_shardability` map.

| Class | Multi-device semantics |
| --- | --- |
| `per_ray` | shard the batch axis; the gathered result is bitwise the single-device result |
| `grid_reduce` | shard the lane axis; per-shard partial grids merge on the master in float32 (section 3) |
| `source_lane` | shard complete transmitter blocks and concatenate fixed `(tx, rx, state)` rows without compaction |
| `variant_specific` | read the family's complete `variant_shardability` map |
| `batch_coupled` | no slice-based semantics; explicit per-shard contract or a loud refusal (section 6) |

The Torch dispositions are `sharded` (the replicated layer wraps it),
`unsupported` (it raises on a multi-device scene) and `single_device` (it is
outside the `Scene` surface the layer wraps, so it neither shards nor refuses).
`variant_specific` means the family disposition must be resolved from the same
variant map.

| Operation | Class | Torch multi-device |
| --- | --- | --- |
| `intersect` | `per_ray` | `sharded` |
| `nearest_edge_point` | `per_ray` | `sharded` |
| `nearest_edge_ray` | `per_ray` | `sharded` |
| `nearest_edges_topk` | `per_ray` | `sharded` |
| `visibility` | `per_ray` | `sharded` |
| `visibility_pair` | `per_ray` | `sharded` |
| `visibility_edge` | `per_ray` | `sharded` |
| `visibility_chain` | `per_ray` | `sharded` |
| `reflection_trace` | `variant_specific` | `variant_specific` |
| `reflection_accumulation` | `grid_reduce` | `sharded` |
| `diffraction_direct` | `variant_specific` | `sharded` |
| `diffraction_chain` | `grid_reduce` | `sharded` |
| `sdf_intersect` | `per_ray` | `single_device` |
| `mixed_scene` | `per_ray` | `single_device` |

`reflection_trace` resolves `trace_reflections` and `trace_refl_epc_field` to
`per_ray`/`sharded`, while `trace_refl_epc` is `per_ray`/`unsupported`.
`diffraction_direct` resolves `trace_dfr_paths` to
`source_lane`/`sharded`, and both accumulation variants to
`grid_reduce`/`sharded`. `reflection_accumulation` is reached through the new
  Torch `Scene.accumulate_reflections()` method and shards its ray batch; wedge
  collection retains one master launch because its bounded event buffer is not
  a reducible grid.
`sdf_intersect` and `mixed_scene` remain `single_device`: the former is a
standalone primitive with no `Scene` membership (ADR-0037), and the latter is
the explicitly single-device `MixedScene` surface (ADR-0043).

## Measured results

The verification machine for this repository is a single Linux node with 2x
NVIDIA RTX A6000 (48 GB each), peer access enabled, measured device-to-device
copy bandwidth 49.1 GB/s in one direction, Torch 2.13.0+cu130, conda env
`maxwell`. Timings are interleaved (single- and multi-device runs alternate
inside one round) and reduced with a minimum over rounds; the machine was
shared, so every number carries roughly +/-5%. The full tables, the twenty-run
spread, and the crossover derivation are in
[`multi_gpu_operations.md`](../dev/multi_gpu_operations.md) section 5.
The machine-readable run-of-record transcription is
[`shared/benchmarks/baselines/multi_device_2xa6000_20260727.json`](../../shared/benchmarks/baselines/multi_device_2xa6000_20260727.json),
pinned with its schema by
[`shared/benchmarks/multi_device_manifest.json`](../../shared/benchmarks/multi_device_manifest.json).
Its provenance says `historical_documentation_import`: it is a structured copy
of the measurements already recorded here, not a newly executed run.

| Configuration | Operation | 1 GPU | 2 GPUs | speedup |
| --- | --- | ---: | ---: | ---: |
| compute (2.1M-triangle cloud, incoherent, 4.19M rays) | `intersect` | 19.09 ms | 11.83 ms | 1.61x |
| compute (4 bounces) | `trace_reflections` | 53.33 ms | 28.38 ms | 1.88x |
| compute (67.1M samples) | `accum_dfr_direct` | 34.76 ms | 18.83 ms | 1.85x |
| light (192-vertex grid, 4.19M rays) | `intersect` | 1.27 ms | 4.63 ms | 0.27x |
| light, after `calibrate_devices()` | `intersect` | 1.22 ms | 1.22 ms | 1.00x |

Medians over 20 consecutive runs: 1.62x, 1.87x, 1.85x for the three compute
rows. Every sharded `per_ray` result in that table was bitwise the single-device
result (agreement 1.0 on every row); the merged grids matched to 6.5e-08 and
2.9e-07 relative deviation, which is the float32 merge order of section 3.

The light rows are the point, not a defect: a sharded row travels twice, so a
full `Intersection` row's 100 B round trip is 2.04 ns of interconnect per ray
against 0.31 ns/ray of compute on that configuration. Two devices cannot help
however well the copies overlap, which is what sections 10's floor and
calibration exist to detect.

## Platform note (observation)

Everything above was verified on Linux with 2x RTX A6000; this repository's
historical baselines were taken on Windows with an RTX 5080. One divergence
between those platforms is known and is recorded here as an observation, not as
a decision of this record:
`backends.drjit.tests.drjit.test_cuda_multipath.test_diffraction_paths_parity`
differs on Linux/A6000 from the Windows/RTX 5080 baseline. The evidence that it
is pre-existing rather than caused by this work is that it differs
**bit-identically before and after** the Phase 3 change (`2ee6e67`), on a
change that touched host and object translation units only and left every
committed-PTX closure and digest untouched. No decision in this record depends
on that divergence, and nothing in this record may be read as accepting it;
diagnosing it is separate work.

## Contract impact

Phase 4 of the plan lands the following, as one change:

1. `shared/contracts/public_api.json`: `capability_keys` gains
   `multi_device_replicated`; `apis.multi_device_replicated` is added with
   `category: "core"` and `stability: "provisional"`;
   `backends.drjit.capabilities.multi_device_replicated` is `false` and
   `backends.torch.capabilities.multi_device_replicated` is `true`. The
   category is `core` because the capability is a property of the existing core
   `Scene` surface rather than a new API family. No schema change:
   `capability_keys` is an open string array and both enum values already
   exist in `public_api.schema.json`.
2. `shared/contracts/operations.json`: `required_capability_keys` gains
   `multi_device_replicated`, which `tests/test_public_api_manifest.py`
   requires to stay equal to the manifest's keys plus `backend`. The file gains
   a top-level `shardability_classes` block and a per-operation `shardability`
   block (section 11). `operations.json` has no JSON-Schema file and its
   governing test does not forbid additional keys, so the annotation is
   declarable there; had it been forbidden, the classification would have lived
   in section 11 alone. No operation is added: this capability is an execution
   property of the existing operations, not a new one, so `operations` keeps its
   fourteen entries.
3. `backends/drjit/python/rayd/drjit/_capabilities.py` and
   `backends/torch/python/rayd/torch/_capabilities.py`: both gain the key in
   the same change, and both repin `_SCHEMA_SHA256` to the new EOL-normalized
   SHA-256 of `public_api.json`. This makes the divergence between the two
   copies **five** lines, not the four ADR-0037 left it at. ADR-0036's
   enforcement test counts no lines, so nothing fails, but its prose is a
   factual claim about the repository and is amended in the same change rather
   than left false.
4. `tests/test_shared_operation_contract.py`: the per-operation shardability
   table, hard-coded the way the integration matrix is, so that a class change
   in either direction fails.
5. `tests/test_adr0038_multi_device.py`: the guard suite for this record. It
   cross-checks the record against the contracts, both capability copies, the
   `MultiDeviceOptions` defaults, the lane-window defaults in the Python
   surface and in the dispatcher schema, and the plan.
6. `CLAUDE.md` and its byte-identical `AGENTS.md`: a multi-GPU section, and
   `README.md`: the capability row and its pointer to this record.
7. No compile-flag change: `shared/contracts/compile_policy.json` is untouched,
   because this work added no translation unit and moved none between profiles.
   ADR-0035 stands unchanged.
8. No PTX change: `backends/drjit/ptx_sources.json` is untouched. The lane
   window lives in the Torch backend, and the Dr.Jit Phase 3 hardening touched
   host and object translation units only, so every committed-PTX closure and
   digest is unchanged.

`tests/test_ptx_source_digest.py` and `tests/test_compile_flag_policy_contract.py`
must stay green throughout, not be repaired afterwards.

## Consequences

- A caller with two GPUs and a compute-bound batch gets 1.6-1.9x by passing
  `devices=[0, 1]`, and gets it without changing a single call site.
- A caller with a transfer-bound batch gets no speedup, and the layer's job is
  to detect that rather than to hide it. `min_rays_per_device` guarantees the
  small case; calibration measures the rest and can be wrong inside its 3%
  band, which is why pinned weights are the recommendation there.
- Merged accumulation grids differ from single-launch grids in the last ULPs.
  A training loop that compares against a stored single-device reference has to
  compare with a tolerance, and a loop that changes its device count or chunk
  count changes its float order.
- Replication costs one full copy of every acceleration structure per device
  and pays N module JITs on first touch (partly overlapped by the warm-up
  helper). A scene that does not fit twice does not fit this design at all.
- `trace_refl_epc` keeps single-device semantics on a multi-device scene by
  raising. Reflection accumulation now shards its reducible grid path while
  preserving bounded wedge collection through a master launch.
- Nothing about single-device execution changed except the Phase 0 guards, so a
  downstream that never passes `devices=` cannot be affected by any of this.

## Non-goals

Each of these is excluded deliberately, not by omission.

- Geometry partitioning and ray forwarding. The replicated design assumes the
  scene fits; Appendix A of the plan records what a partitioned route would
  require and is not authorized here.
- NVLink-shared acceleration structures.
- Dynamic load rebalancing.
- Dr.Jit single-process multi-device execution.
- Wavefront restructuring of `trace_reflections`.
- A collective library inside RayD. RayD launches no collectives and holds no
  communicator; reductions are the caller's, on the caller's tensors.
- A CPU or single-device fallback for an operation that refuses to shard. It
  raises instead.
- Any change to kernels, launch shapes, reduction order, atomics or compile
  flags in the name of multi-device execution.

## Deferred

Recorded so the boundary of this record is legible. None is authorized here;
each needs its own decision when it is picked up.

1. `deduplicate = true` cross-shard semantics: shard-local dedup equals
   single-device dedup only when shards align with dedup key groups, and the
   general case needs an explicit flag and a merge pass.
2. ADR-0033 failure-bit merging across shards, including the inertness rule
   that would have to hold per shard.
3. Torch `trace_refl_epc` per-ray sharding.
4. Geometry partitioning and ray forwarding (plan Appendix A).
5. A heterogeneous-device calibration claim. The verification machine's two
   devices are identical, so the calibrated weights answer 1.00/1.00 on
   compute-bound probes and nothing here is evidence about unequal GPUs.
   `require_homogeneous_devices=False` permits such an experiment explicitly,
   but neither bitwise parity nor calibration quality is claimed for it.

## Stop conditions

Stop and reopen this record before:

- changing any kernel launch shape, reduction order, atomic, or CUDA numeric
  compile flag for a multi-device reason, or letting a scene's device count
  reach a kernel at all beyond the lane window of section 5;
- making single-device execution take the orchestration path, importing
  `_multi` for a scene that did not ask for it, or losing the bitwise
  equality of a one-device `Scene(devices=[d])` with a default `Scene()`;
- giving `lane_offset = 0` with the default `lane_count` any behavior other
  than the pre-ADR single launch, or claiming merged-grid equality for windows
  that are not warp multiples;
- sharding a `batch_coupled` operation without writing its per-shard semantics
  down first, or replacing one of the current loud refusals with a silent
  partial result;
- detaching, zeroing, or approximating a gradient so that a shard or a chunk
  can proceed, or reducing replica gradients by any route other than the
  recorded `.to()` edges;
- weakening the row floor from a guarantee to a heuristic, or restating
  calibration as a guarantee that a split cannot be slower than the master
  alone;
- adding a parallel public multi-device API surface, a dispatcher, a
  `Scene`-like wrapper, or a public name under `rayd.torch.multi`;
- adding a collective, a communicator, or a process-group dependency to RayD
  itself;
- partitioning geometry, sharing an acceleration structure across devices, or
  rebalancing shards dynamically.
