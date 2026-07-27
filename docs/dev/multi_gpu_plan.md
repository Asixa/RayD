# Multi-GPU Ray Tracing — Implementation Plan

Status: draft proposal (Phase 4 produces the accepted ADR).

Target workload (agreed 2026-07-27): **extremely large ray/sample batches
against scenes that fit in a single GPU's memory.** This selects the
scene-replication + batch-sharding regime for single-node multi-GPU and makes
the cluster tier embarrassingly parallel (per-node replicas, global ray
sharding, all-reduce only for small outputs: accumulation grids and vertex
gradients). Geometry partitioning + ray forwarding is deferred to Appendix A.

## 1. Headline decisions

**D1 — Scene replication + batch sharding.** Every device holds a full scene
replica (GAS/IAS, edge BVH, compact BVH, SoA, edge mask); work shards along
the batch axis. Rationale: scenes fit in VRAM; NVLink-shared acceleration
structures traverse 2–5x slower than replicas; Cycles/V-Ray/OWL/OptiX_Apps
and Sionna's own guidance all replicate.

**D2 — Backend split.** Dr.Jit backend: process-per-GPU only
(`CUDA_VISIBLE_DEVICES` per rank; Dr.Jit is single-device per process,
drjit#359 / mitsuba3#808). Torch backend: first-class single-process
multi-device plus the process-per-GPU recipe.

**D3 — Single-device numerics are untouched.** Multi-GPU is a composition
layer above the existing kernels: no kernel launch shape, reduction order,
atomics, or compile flag changes. Cross-device/chunk merging happens in the
orchestration layer with its own documented float32 summation order. The
frozen guarantees of ADR-0026/0030/0032/0035 stay intact; the new ADR defines
only the merge-layer semantics and the invariant that single-device,
single-chunk execution stays bitwise identical.

**D4 — Autograd-native gradient reduction.** Replica vertices are
`master.to(device_k)`; torch autograd reduces per-replica `grad_vertices`
back onto the master leaf automatically. DDP mode gets NCCL reduction free.
Chunked execution accumulates gradients additively across chunks (standard
gradient accumulation; valid because all geometry gradients land in
`grad_vertices` by summation).

**D5 — RNG lane-offset parameter (core requirement).** `DfrAccumParams`
gains an explicit `lane_offset` so that a K-device × M-chunk split is a
partition of the single-launch lane space: `uniform01(lane_offset + lane,
stream, seed)`. Contract: the multiset of drawn samples is independent of the
split (merging remains float-order-dependent). Same-seed sharding without the
offset reuses lanes `0..M/K-1` per shard and is forbidden.

**D6 — Batch-coupled features get explicit per-shard semantics.**
`deduplicate=true` dedups within a shard/chunk (optional merge-layer second
pass); the diffraction exporter prefers `SourceLane` layout under sharding
(row placement is `(tx,rx,state)`-determined), `Compact` falls back to
per-shard compaction + concatenation; the ADR-0033 failure transaction stays
per-shard, merge layer ORs bits and applies inertness per shard.

**D7 — Chunked execution is a first-class component.** At extreme N the
binding constraint is tape/output memory, not scene memory (reflection tape
≈ 40–50 B/ray/bounce ⇒ 10⁹ rays × 3 bounces ≈ 150 GB). The orchestration
layer therefore executes every large batch as a stream of chunks per device:
`grid_reduce` outputs accumulate in place (O(1) memory in N); `per_ray`
outputs either concatenate on the master device or stream through a
caller-supplied offload hook; forward+backward run per chunk with gradient
accumulation. Chunk size is calibrated (as large as tape memory allows) to
amortize per-launch overhead.

**D8 — Multi-GPU is invisible at the top-level API.** No parallel public
surface: `Scene` gains one optional `devices=` argument (default: today's
single-device behavior, derived from the mesh tensors), and every existing
op (`scene.intersect`, `trace_reflections`, accumulation, ...) transparently
shards/chunks when the scene was built with multiple devices. The
orchestration machinery (replicas, sharder, chunked executor) lives in a
private module (`backends/torch/python/rayd/torch/_multi.py`) and is not
public API. Process-per-GPU / distributed mode needs no API change at all
(rank-local scenes). Tuning knobs (shard weights, chunk memory budget,
offload hook) ride on an optional `MultiDeviceOptions` dataclass — additive
and defaulted.

**D9 — Zero single-GPU regression.** A single-device `Scene` executes the
identical code path as today: the multi-device layer is a Python-level
branch engaged only when `len(devices) > 1` — no wrapper indirection, no
extra synchronization, no allocations on the hot path. The only
single-GPU-path change in the whole plan is Phase 0's device guards
(no-op-cheap when the device is already current). Gate: single-GPU
benchmarks (`benchmark_torch_native.py` set) must stay within noise (±2%)
of the pre-change baseline before each phase merges.

## 2. Current state (2026-07 audit, abridged)

Already multi-device-shaped: Torch OptiX contexts keyed
`(device_index, CUcontext)`; pipeline caches keyed by `OptixDeviceContext`
with per-instance device + guard; `SceneCache` carries `device_index` with
loud cross-device failures in scene-coupled ops; all `shared/` kernels are
device-neutral (raw pointers + explicit stream); `stable/camera.cu` is the
reference for guard/stream handling; Dr.Jit has
`set_device/current_device/device_count` and a `(device, context)`-keyed
capability cache.

To fix (27 audit findings): ~30 Torch CUDA TUs launch without `CUDAGuard`;
scattering table/ensemble/patch (+ transmission/layer-stack/wedge) families
check `is_cuda()` only, not the device index; `geometry_backward.cu` queries
cooperative occupancy on the ambient device; `native_compat.h` and
`compute_edge_optix_aabbs_cuda` use the ambient device; Dr.Jit pure-CUDA TUs
(`edge_bvh.cu`, `triangle_bvh.cu`, `cuda_multipath.cu`) use raw `cudaMalloc`
on the runtime-current device, private per-call streams,
`cudaDeviceSynchronize`, a default-stream blocking copy, a racy file-scope
`__constant__` params symbol, and a device/stream-less public ABI; the Dr.Jit
multipath pipeline cache object records no device.

Shardability classes:

| Class | Ops | Multi-GPU semantics |
|---|---|---|
| `per_ray` | intersect, nearest_edge(s), visible*, trace_reflections (no dedup), trace_refl_epc_field, sdf forward, scattering primal, camera | shard result bitwise ≡ single device |
| `grid_reduce` | reflection/diffraction accumulation, SDF value/box grads, scene-vertex grads | per-shard partials + merge-layer sum; float order changes (already atomic-nondeterministic on one device) |
| `batch_coupled` | dedup, Compact exporter count, ADR-0033 failure bit, MC RNG | explicit per-shard semantics (D5/D6) |

## 3. Phases with acceptance criteria

### Phase 0 — Device-correctness hardening (~1–2 weeks; single-GPU-safe)

Work items:

1. `c10::cuda::CUDAGuard` (scene or input device) at every Torch op entry
   that launches kernels/CUB: `scene/ops_intersect.cpp`, `edge/ops_edge.cpp`
   + `edge_topk.cu`/`edge_backward.cu`, `scene/geometry_backward.cu`,
   `diffraction/ops.cpp`, `sdf/ops_sdf.cpp`, `common/ops_stats.cpp`, all 13
   `rf/*` TUs (model: `stable/camera.cu`).
2. Same-device-index check in `common/tensor_check`, applied to the
   scattering table/ensemble/patch, transmission, layer-stack, and wedge
   families (mirroring `scattering_chain_checks.h`).
3. `coop_launch_config_for_device` queries occupancy under a guard for the
   requested device.
4. Replace ambient-device use in `native_compat.h` and
   `compute_edge_optix_aabbs_cuda` with scene-derived device/stream.
5. Dr.Jit: store the device in `OptixLaunchPipeline` and assert at launch;
   assert build-device vs current-device in `Scene` queries (loud failure
   after `set_device`).

Acceptance:

- [ ] Full existing test suite passes unchanged (no numerical drift; golden
      tests bitwise).
- [ ] New cross-device rejection tests for the families that lacked them
      (table/ensemble/patch scattering, transmission, layer stack, wedge).
- [ ] 2-GPU smoke suite (skipped when `torch.cuda.device_count() < 2`):
      same mesh built on dev0 and dev1, `intersect` results bitwise equal;
      every public op runs correctly on a non-zero device while device 0 is
      current.
- [ ] Dr.Jit: querying a scene after `set_device` to another device raises
      instead of corrupting.

### Phase 1 — Torch multi-device correctness (~1 week; manual orchestration works)

Work items:

1. `penetration/ops.cpp` resolves context/pipeline from the validated scene
   device (drop ambient-context coupling).
2. Serialize OptiX context/module creation across devices (in-process mutex
   exists); document per-process `OPTIX_CACHE_PATH` for process-parallel
   launches (Blender disk-cache race).
3. Optional per-device pipeline warm-up helper (worker threads) to hide N×
   module JIT.

Acceptance:

- [ ] Two scenes on two devices, driven concurrently from two host threads on
      non-default streams, produce per-device results bitwise equal to
      single-device runs (stress test in `backends/torch/tests`).
- [ ] Per-device OptiX cold-create passes (generalization of
      `test_optix_pipeline_cold_create` to a non-zero device).
- [ ] No cross-device serialization beyond context creation (verified by
      overlapping-launch timing check, coarse threshold).

### Phase 2 — `rayd.torch.multi` orchestration layer (~3–5 weeks; the feature)

Private module `backends/torch/python/rayd/torch/_multi.py`; public exposure
is only `Scene(devices=[...])` plus the optional `MultiDeviceOptions`
dataclass (D8). With `devices` absent or singular, `Scene` takes the
pre-existing code path unchanged (D9).

**2a — Replicated state + Sharder + per_ray dispatch (~1–1.5 weeks)**

- Internal `_ReplicatedScene(meshes, devices, master_device=devices[0])`: replica
  vertices are `master.to(dev_k)` (autograd-recorded); `update_mesh_vertices`
  / `sync()` / `set_edge_mask` broadcast to all replicas; version counters
  verified in lockstep, divergence fails loudly.
- `Sharder`: static weighted split; weights from a one-time `calibrate()`
  micro-benchmark; no dynamic rebalancing.
- `per_ray` wrappers: non-blocking scatter, launch on each device's current
  stream, event-ordered gather to master, concatenate. P2P enabled when
  `torch.cuda.can_device_access_peer` allows.

Acceptance:

- [ ] Single-device `Scene` (no `devices=` or one device) provably takes the
      pre-existing code path: a unit test asserts the multi layer is never
      engaged, and single-GPU benchmarks stay within ±2% of baseline (D9).
- [ ] 1-device `Scene(devices=[d])` ≡ native `Scene` path bitwise for every
      wrapped op (runs in single-GPU CI).
- [ ] 2-device: every `per_ray` op bitwise ≡ single-device result after
      gather, for several shard ratios including degenerate (0-length) shards.
- [ ] Broadcast mutation: `update_mesh_vertices` + `sync()` on the replica
      set keeps per-device results bitwise equal; injected divergence is
      detected and raises.
- [ ] Autograd: `master_vertices.grad` from a 2-device `per_ray` backward
      matches single-device grad within atomics tolerance; JVP paths covered.

**2b — Chunked executor (~1–1.5 weeks)**

- Per-device chunk queue with double-buffered streams (chunk k+1 compute
  overlaps chunk k gather/D2H).
- `grid_reduce` ops: in-place accumulation of chunk partials on each device,
  single cross-device merge at the end.
- `per_ray` ops: concatenate on master, or stream through a caller-supplied
  offload hook (`on_chunk(result_slice)`), for outputs too large for one GPU.
- Chunked forward+backward: per-chunk backward with gradient accumulation
  into `master_vertices.grad`.
- `calibrate_chunk_size()`: pick the largest chunk fitting a tape-memory
  budget; expose per-launch overhead measurement.

Acceptance:

- [ ] Chunked `per_ray` execution ≡ unchunked bitwise (same device, any chunk
      size).
- [ ] Chunked `grid_reduce` matches unchunked within float tolerance and is
      run-to-run reproducible at fixed chunking.
- [ ] Chunked backward gradient matches unchunked within atomics tolerance.
- [ ] A batch whose outputs exceed single-GPU memory completes via the
      offload hook (synthetic test with a capped allocator budget).
- [ ] Overlap measured: with ≥4 chunks, end-to-end time <
      (sum of compute) + 1.15 × (one chunk's D2H), on the benchmark scene.

**2c — RNG lane offset + batch-coupled semantics (~1 week, includes ADR text)**

- Add `lane_offset` to `DfrAccumParams` (and the Torch op schema /
  `autograd.py` plumbing); default 0 preserves current behavior bitwise.
- Wire shard/chunk lane offsets in the executor so device × chunk splits
  partition the lane space.
- Per-shard dedup semantics flag; `SourceLane` selection for the exporter
  under sharding; `Compact` per-shard + concatenation; ADR-0033 failure-bit
  OR-merge.

Acceptance:

- [ ] `lane_offset = 0` is bitwise identical to today (existing accumulation
      golden tests unchanged).
- [ ] Lane-partition test: the multiset of `(tape_state_idx, tape_cell,
      tape_edge_u)` rows from a K×M split equals the single-launch multiset
      (exact comparison after sort).
- [ ] Sharded exporter (`SourceLane`): row-for-row identical to single-device
      for the successful lanes; `Compact` concatenation preserves the
      per-shard row sets.
- [ ] Dedup: per-shard semantics documented and tested (shard-local dedup
      equals single-device dedup when shards align with dedup key groups; the
      general case is asserted to differ and gated behind an explicit flag).

**2d — Throughput validation (~0.5 week)**

Acceptance:

- [ ] ≥1.8× on 2 GPUs (same model) vs 1 GPU for: 65k+ ray `intersect`,
      `trace_reflections`, and a large-sample `accum_dfr_direct`, at
      calibrated shard weights and chunk size (benchmark added to
      `backends/torch/tests/benchmark_*`).
- [ ] Heterogeneous pair (if available): calibrated weights beat naive 50/50
      by a recorded margin (informational, no hard threshold).
- [ ] Single-GPU parity gate (D9): the full `benchmark_torch_native.py` set
      within ±2% of the pre-Phase-2 baseline.

### Phase 3 — Process-per-GPU and multi-node recipes (~1–2 weeks)

Work items:

1. Single-node recipe: `torchrun` one rank per GPU, `CUDA_VISIBLE_DEVICES`
   per rank, rank-local scene build, NCCL all-reduce of grids/vertex grads
   (automatic under DDP when vertices are module parameters; manual
   `all_reduce` example otherwise). Works for both backends; this is the only
   Dr.Jit path.
2. Multi-node recipe: same script under multi-node `torchrun`; document that
   traffic is only grids + gradients (scene-sized, N-independent).
3. Example scripts under `examples/` (or `backends/*/examples/`) with a
   smoke-test harness.
4. Dr.Jit pure-CUDA hardening (quality items, not blocking for
   process-per-GPU): device recorded in `CudaBuffer`, `cudaDeviceSynchronize`
   → stream sync, treelet H2D off the default stream, `DfrAccumParams` as a
   kernel argument instead of file-scope `__constant__`, device/stream
   parameters in the public builder ABI (`edge_bvh.h`).

Acceptance:

- [ ] 2-rank single-node example produces grids/grads matching single-process
      2-device execution within float tolerance, and runs in the multi-GPU CI
      job.
- [ ] Dr.Jit 2-rank example (each rank pinned via `CUDA_VISIBLE_DEVICES`)
      runs the reflection benchmark with ≥1.8× aggregate throughput.
- [ ] Multi-node invocation documented and exercised at least manually
      (recorded in the PR); no code path differs from single-node beyond the
      rendezvous.
- [ ] Hardening items each keep the Dr.Jit test suite bitwise-green.

### Phase 4 — ADR, contracts, CI (runs alongside Phases 2–3)

Work items:

1. ADR "replicated multi-device and chunked execution": D1–D7, merge-layer
   float semantics, lane-offset contract, per-shard dedup/failure semantics,
   SourceLane recommendation, and the bitwise-invariance guarantee for
   single-device single-chunk execution.
2. Contracts (template: the SDF capability commits): `multi_device_replicated`
   capability key in `public_api.json` (torch=true, drjit=false),
   `shardability` annotation per operation in `operations.json`, both
   `_capabilities.py` twins + `_SCHEMA_SHA256` repin, ADR-0036
   divergence-count amendment, `.pyi` stubs for `rayd.torch.multi`,
   `library.cpp` schema entries for any new/extended ops (lane offset).
3. CI: multi-GPU runner job gated on `device_count() >= 2`; single-GPU CI
   runs the 1-device-equivalence subset; contract tests extended
   (`test_shared_operation_contract.py`, `test_public_api_manifest.py`).
4. If any `.cu` reachable from committed Dr.Jit PTX changes (lane offset in
   shared headers): regenerate PTX and refresh `ptx_sources.json` per the
   committed-PTX policy.

Acceptance:

- [ ] ADR accepted; contract tests green in both directions (declaration ↔
      build).
- [ ] Capability visible from both backends' `_capabilities.py` with correct
      values; manifest/typing tests pass.
- [ ] Multi-GPU CI job green on the 2-device matrix; single-GPU CI unchanged.
- [ ] PTX digest test green (regenerated if touched).

## 4. Risks / non-goals

Non-goals (this plan): geometry partitioning & ray forwarding, NVLink-shared
GAS, dynamic load rebalancing, Dr.Jit single-process multi-device, wavefront
restructuring of `trace_reflections`.

Risks: merge-layer float-order drift in training loops (documented +
tolerance-tested); replica divergence (broadcast-only mutation + version
lockstep checks); N× first-build module JIT (warm-up + disk cache);
per-launch overhead at large chunk counts (mitigated by chunk-size
calibration; measure before considering CUDA Graphs); heterogeneous-GPU
static weights (explicitly re-calibratable).

## Appendix A — Deferred: geometry partitioning + ray forwarding

Revisit trigger: a target scene whose replicated footprint (after IAS
instancing and selective replication of small structures) exceeds
single-GPU memory. Cheap capacity levers to exhaust first: per-mesh IAS
instancing (already built), a per-structure memory profiler, selective
replication (edge BVH / material tables are far smaller than triangle
geometry and can stay replicated even when triangles shard).

The partitioned route then requires, in dependency order: (P1) wavefront
restructuring of `trace_reflections` (bounce-level launches with materialized
ray state; conflicts with the OptiX pipeline guardrail and needs its own
ADR; the Dr.Jit symbolic path is a limited prototype); (P2) partitioned
GAS/IAS with proxy routing and partition-local → global ID mapping; (P3) a
ray-forwarding engine (NVLink ray-queue cycling intra-node; NCCL
`alltoallv` / MPI + GPUDirect inter-node) — closest-hit merges are a
deterministic min-reduce with `(t, global_prim_id)` tie-break, visibility is
boolean OR with early-out; (P4) `nearest_edge` under partitioning — prefer
full edge-structure replication; else bound-pruned multi-round top-k merge;
(P5) distributed differentiation — RayD's fixed-winner tape AD is unusually
distribution-friendly (backward needs tape rows + per-hit vertex gathers,
never BVH re-traversal): route tape rows to owning partitions, accumulate
locally, reduce by owner; reflection-chain VJPs need per-ray tape gathering;
no published prior art; (P6) distributed dedup/exporter semantics (extends
D6); (P7) cluster runtime + partition loader boundary (RayD provides the
partitioner and partition-scene build API; file formats stay caller-owned);
(P8) Torch-first, `shared/`-neutral engine — the Dr.Jit backend does not
participate in-process.
