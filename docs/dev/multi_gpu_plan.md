# Multi-GPU Ray Tracing — Implementation Plan

Status: **executed and accepted** (2026-07-27). Phases 0–3 landed and the
Phase 4 contract surface landed with them; the decisions this plan proposed are
the accepted record
[`ADR-0038 — Replicated multi-device and chunked execution`](../adr/0038-replicated-multi-device-execution.md).
The plan stays as the phase-by-phase account; ADR-0038 is what governs the
shipped layer. The acceptance boxes below are ticked against the recorded
verifications, and what is still open is listed in the execution record and
left unticked.

Target workload (agreed 2026-07-27): **extremely large ray/sample batches
against scenes that fit in a single GPU's memory.** This selects the
scene-replication + batch-sharding regime for single-node multi-GPU and makes
the cluster tier embarrassingly parallel (per-node replicas, global ray
sharding, all-reduce only for small outputs: accumulation grids and vertex
gradients). Geometry partitioning + ray forwarding is deferred to Appendix A.

## 0. Execution record

Landed as five commits, all on 2026-07-27, all verified on this repository's
Linux verification host: **2× NVIDIA RTX A6000** (48 GB each, peer access
enabled, measured D2D copy bandwidth **49.1 GB/s** one direction), Torch
`2.13.0+cu130`, conda env `maxwell`. The Windows/RTX 5080 machine that carries
this repository's historical baselines was not used for any of it.

| Commit | Phase | Landed | Verification recorded with it |
| --- | --- | --- | --- |
| `f643336` | 0 — device-correctness hardening | 2026-07-27 | Torch suite 198 OK (19 skipped); new 2-GPU smoke suite (10 tests); Dr.Jit device-binding test |
| `cc5f0f9` | 1 — Torch multi-device correctness, docs, warm-up | 2026-07-27 | Torch suite 221 OK (19 skipped); governance green; no product C++ changed |
| `c2c50ce` | 2a–2c — replicated `Scene`, chunked executor, lane window | 2026-07-27 | Torch suite 282 OK (19 skipped); single-GPU parity within the D9 gate against `cc5f0f9`; GIL×registry deadlock fixed, 40/40 cold-JIT threaded trials clean |
| `869eb62` | 2d — pipelined dispatch, calibration, row floor | 2026-07-27 | Torch suite 306 OK; the measured table below; bitwise agreement 1.0 on every sharded `per_ray` row |
| `2ee6e67` | 3 — distributed recipes, Dr.Jit pure-CUDA hardening | 2026-07-27 | Torch suite 310 OK; Dr.Jit on Linux: `test_geometry` 63/63, `test_device_binding` 4/4, edge-BVH modules 23/23; governance 46/46 |
| Phase 4 (this closeout) | ADR, contracts, CI | 2026-07-27 | ADR-0038 accepted; contract/capability/PTX/compile-flag suites green (90 tests); Torch suite re-run 310 OK (19 skipped); the multi-device modules, the distributed recipe and the governance suite re-run per `multi_gpu_operations.md` §8 |

Headline measured numbers, from
[`multi_gpu_operations.md`](multi_gpu_operations.md) §5 (interleaved runs,
minimum over 7 rounds, ±5% on a shared machine; medians over 20 consecutive
runs in brackets):

| Configuration | Operation | 1 GPU | 2 GPUs | speedup |
| --- | --- | ---: | ---: | ---: |
| compute (2.1M-triangle cloud, incoherent, 4.19M rays) | `intersect` | 19.09 ms | 11.83 ms | 1.61× [1.62×] |
| compute (4 bounces, 4.19M rays) | `trace_reflections` | 53.33 ms | 28.38 ms | 1.88× [1.87×] |
| compute (67.1M samples) | `accum_dfr_direct` | 34.76 ms | 18.83 ms | 1.85× [1.85×] |
| light (192-vertex grid, 4.19M rays) | `intersect` | 1.27 ms | 4.63 ms | 0.27× |
| light, after `calibrate_devices()` | `intersect` | 1.22 ms | 1.22 ms | 1.00× |

Every sharded `per_ray` result in that table was bitwise the single-device
result (agreement 1.0 on every row); merged accumulation grids matched the
single-launch grids to 6.5e-08 (`light`) and 2.9e-07 (`compute`) relative
deviation, which is the float32 merge order D3 allows. The chunked+offload row
peaked at 2.10 GB on the master against 2.29 GB concatenated at 4.19M rows. The
row floor is 262,144 rows per device (524,288 on two).

**Still open.** Ticked boxes below are the ones the recorded verifications
actually satisfy; these are not, and are left unticked where they appear:

1. Dr.Jit 2-rank aggregate-throughput benchmark (Phase 3): never measured.
2. Multi-node execution (Phase 3): documented and structurally argued, not
   executed — this repository has one node.
3. A Dr.Jit run on the Windows/RTX 5080 baseline machine after the Phase 3
   pure-CUDA hardening. The Linux modules listed above are green; the known
   pre-existing `test_diffraction_paths_parity` platform divergence is recorded
   in ADR-0038's platform note and is not accepted by it.
4. The D6 batch-coupled half of 2c: sharded `trace_dfr_paths` (both `SourceLane`
   and `Compact`), cross-shard `deduplicate=true` semantics, ADR-0033
   failure-bit merging, and `accum_dfr_coherent_direct` under sharding. The
   layer refuses these loudly rather than guessing; they are the Deferred list
   of ADR-0038.
5. A heterogeneous-device calibration claim (2d): both devices here are
   identical.
6. The multi-GPU CI job actually running.
   [`.github/workflows/multi_gpu.yml`](../../.github/workflows/multi_gpu.yml)
   declares it — the multi-device modules fresh-process, the distributed recipe
   and the governance suite — pinned to a `self-hosted, linux, x64, cuda,
   multi-gpu` runner, and it is **inert**: no such runner is registered, and its
   only trigger is `workflow_dispatch`. Every hosted job in this repository
   (`ci.yml`, `stable-abi-ci.yml`, `pypi.yml`) builds and checks packaging
   without a CUDA device, so "single-GPU CI" wherever it appears below means the
   single-device subset of the suite run on a developer machine, not a hosted
   job. Until a runner exists the acceptance set is run by hand:
   [`multi_gpu_operations.md`](multi_gpu_operations.md) §8.
7. A defect this closeout found rather than fixed: Phase 0 (`f643336`)
   falsified `tests/test_bvh4_shared_aabb.py::SharedEdgeAabbSourceTests::`
   `test_torch_adapter_uses_shared_api_on_current_stream`. That test asserts the
   Torch edge-AABB adapter calls `current_torch_cuda_context()` and launches on
   `torch_ctx.stream`, which is exactly the ambient-device coupling Phase 0
   work item 4 removed in favour of a `c10::cuda::CUDAGuard` plus
   `getCurrentCUDAStream(out_aabbs.get_device())`. The shipped code is what the
   plan asked for and the test states the pre-Phase-0 shape; the BVH-4 contract
   owner has to decide how it should read now. Nothing in this closeout touched
   either side, so Phase 0's first acceptance box stays unticked.

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
layer therefore executes every large batch as a stream of chunks per device.
The budget is a per-device peak-increment bound and includes copied inputs,
outputs and tape for as many as three resident pipeline chunks plus any
complete returned output; accumulation also reserves replicated
state/material payloads and its fixed grid partials. Inference
`grid_reduce` reuses a fixed-size value buffer, but its autograd graph is not
O(1): each chunk retains a frozen native tape until backward. A budgeted
multi-chunk AD grid therefore fails loudly. `per_ray` outputs either concatenate
on the master device when the complete result fits, or stream through a
caller-supplied offload hook; bounded forward+backward uses the hook to run
backward per chunk. Chunk size is calibrated as large as the complete peak
model allows, to amortize per-launch overhead.

**D8 — Multi-GPU is invisible at the top-level API.** No parallel public
surface: `Scene` gains one optional `devices=` argument (default: today's
single-device behavior, derived from the mesh tensors), and every existing
op (`scene.intersect`, `trace_reflections`, accumulation, ...) transparently
shards/chunks when the scene was built with multiple devices. The
orchestration machinery (replicas, sharder, chunked executor) lives in a
private module (`python/rayd/_impl/multi.py`) and is not
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
      tests bitwise). **Open on one test.** The Torch suite passed at every
      phase (198 → 310 OK) with no numerical drift and golden tests bitwise,
      but work item 4 falsified
      `tests/test_bvh4_shared_aabb.py::SharedEdgeAabbSourceTests::`
      `test_torch_adapter_uses_shared_api_on_current_stream`, which asserts the
      `current_torch_cuda_context()` shape the device-explicit rewrite replaced
      (execution record item 7). It is a source-shape contract, not a numerical
      one, and it is still red.
- [x] New cross-device rejection tests for the families that lacked them
      (table/ensemble/patch scattering, transmission, layer stack, wedge) —
      `tests/scattering/scattering_test.cpp`, extended to all six
      families in `f643336`.
- [x] 2-GPU smoke suite (skipped when `torch.cuda.device_count() < 2`):
      same mesh built on dev0 and dev1, `intersect` results bitwise equal;
      every public op runs correctly on a non-zero device while device 0 is
      current — `test_multi_device_smoke.py`, 10 tests, green.
- [x] Dr.Jit: querying a scene after `set_device` to another device raises
      instead of corrupting — `backends.drjit.tests.drjit.test_device_binding`,
      4/4 green on Linux.

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

- [x] Two scenes on two devices, driven concurrently from two host threads on
      non-default streams, produce per-device results bitwise equal to
      single-device runs (stress test in `tests/scene`) —
      `test_multi_device_stress.py::test_two_threads_on_two_devices_`
      `reproduce_single_device_results`.
- [x] Per-device OptiX cold-create passes (generalization of
      `test_optix_pipeline_cold_create` to a non-zero device) —
      `test_multi_device_stress.py::test_cold_create_on_a_non_zero_device_`
      `in_a_fresh_process`.
- [x] No cross-device serialization beyond context creation (verified by
      overlapping-launch timing check, coarse threshold) —
      `test_multi_device_stress.py::test_two_devices_overlap_instead_of_`
      `serializing`. What does serialize process-wide (device-context creation,
      the multipath launch-pipeline cache, and the GIL over an op body) is
      written up in [`multi_gpu_operations.md`](multi_gpu_operations.md) §2.

### Phase 2 — `rayd.torch.multi` orchestration layer (~3–5 weeks; the feature)

Private module `python/rayd/_impl/multi.py`; public exposure
is only `Scene(devices=[...])` plus the optional `MultiDeviceOptions`
dataclass (D8). With `devices` absent or singular, `Scene` takes the
pre-existing code path unchanged (D9).

**2a — Replicated state + Sharder + per_ray dispatch (~1–1.5 weeks)**

- Internal `_ReplicatedScene(meshes, devices, master_device=devices[0])`: replica
  vertices are `master.to(dev_k)` (autograd-recorded); `update_mesh_vertices`
  / `sync()` / `set_edge_mask` broadcast to all replicas; version counters
  verified in lockstep, divergence fails loudly.
- `Sharder`: static operation-local weighted splits from configuration or an
  explicit `calibrate_devices()` micro-benchmark; no dynamic rebalancing.
- `per_ray` wrappers: non-blocking scatter, launch on each device's current
  stream, event-ordered gather to master, concatenate. Bidirectional P2P and
  homogeneous model/capability are required by default; host staging and
  heterogeneous execution are explicit opt-ins with reduced guarantees.

Acceptance:

- [x] Single-device `Scene` (no `devices=` or one device) provably takes the
      pre-existing code path: a unit test asserts the multi layer is never
      engaged, and single-GPU benchmarks stay within ±2% of baseline (D9).
- [x] 1-device `Scene(devices=[d])` ≡ native `Scene` path bitwise for every
      wrapped op (runs in single-GPU CI).
- [x] 2-device: every `per_ray` op bitwise ≡ single-device result after
      gather, for several shard ratios including degenerate (0-length) shards.
- [x] Broadcast mutation: `update_mesh_vertices` + `sync()` on the replica
      set keeps per-device results bitwise equal; injected divergence is
      detected and raises.
- [x] Autograd: `master_vertices.grad` from a 2-device `per_ray` backward
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

- [x] Chunked `per_ray` execution ≡ unchunked bitwise (same device, any chunk
      size).
- [x] Chunked `grid_reduce` matches unchunked within float tolerance and is
      run-to-run reproducible at fixed chunking.
- [x] Chunked backward gradient matches unchunked within atomics tolerance.
- [x] A batch whose outputs exceed single-GPU memory completes via the
      offload hook (the synthetic case is a tape-memory budget that sizes the
      chunk plus a streamed hook, rather than a capped allocator; measured
      effect: 2.10 GB master peak streamed against 2.29 GB concatenated at
      4M rows).
- [x] Overlap measured: with ≥4 chunks, end-to-end time <
      (sum of compute) + 1.15 × (one chunk's D2H), on the benchmark scene.
      Measured at 8 chunks: 11.71 ms against 19.15 + 1.15 × 0.81 = 20.08 ms.

**2c — RNG lane offset + batch-coupled semantics (~1 week, includes ADR text)**

- Add `lane_offset` to `DfrAccumParams` (and the Torch op schema /
  `autograd.py` plumbing); default 0 preserves current behavior bitwise.
- Wire shard/chunk lane offsets in the executor so device × chunk splits
  partition the lane space.
- Per-shard dedup semantics flag; `SourceLane` selection for the exporter
  under sharding; `Compact` per-shard + concatenation; ADR-0033 failure-bit
  OR-merge.

Acceptance:

- [x] `lane_offset = 0` is bitwise identical to today (existing accumulation
      golden tests unchanged).
- [x] Lane-partition test: the multiset of `(tape_state_idx, tape_cell,
      tape_edge_u)` rows from a K×M split equals the single-launch multiset
      (exact comparison after sort). Covered at the reachable boundary: the
      lane windows are asserted to be a contiguous, disjoint, warp-aligned
      partition of the caller's window, and the merged grid reproduces the
      single-launch grid; the tape rows themselves are internal.
- [ ] Sharded exporter (`SourceLane`): row-for-row identical to single-device
      for the successful lanes; `Compact` concatenation preserves the
      per-shard row sets. **Open** — `trace_dfr_paths` still raises
      `NotImplementedError` on a multi-device scene.
- [ ] Dedup: per-shard semantics documented and tested (shard-local dedup
      equals single-device dedup when shards align with dedup key groups; the
      general case is asserted to differ and gated behind an explicit flag).
      **Open** — the batch-coupled half of 2c is not landed.

**2d — Throughput validation (~0.5 week)**

The original criterion — "≥1.8× for 65k+ ray `intersect`" — asked for the
wrong thing, and measuring it is what showed why. A sharded row travels twice
(inputs out, outputs back), so two devices can only beat one when a row's
compute costs more than its bytes cost to move. On the verification machine
(2× RTX A6000, 49.1 GB/s measured one-direction D2D) a full `Intersection` row
is 100 B in and out, i.e. 2.04 ns of interconnect per ray: an `intersect`
cheaper than that per ray is faster on one GPU at *any* batch size, and 65k
rays are below the small-batch floor besides. The revised criteria therefore
separate the two regimes instead of demanding one number from both.

Acceptance:

- [x] `grid_reduce` ≥1.8× on 2 GPUs (same model) vs 1 GPU for a large-sample
      `accum_dfr_direct`: **1.85×** median at 67.1M samples (34.76 → 18.83 ms
      in the recorded run), 1.81–1.89× over 20 consecutive runs. No per-row
      data crosses the link, so the speedup is `T / (T/2 + ~1.4 ms merge)` and
      rises monotonically with the sample count (1.08× at 4.2M, 1.60× at
      16.8M, 1.75× at 33.6M).
- [x] `per_ray` ≥1.6–1.8× on 2 GPUs vs 1 GPU on compute-bound configurations
      through the pipelined path: **1.62×** median for `intersect` and
      **1.87×** median for `trace_reflections` (4 bounces) at 4.19M rays
      against a 2.1M-triangle cloud with incoherent rays, with a total spread
      of 1.60–1.63× and 1.85–1.90× over 20 consecutive runs. The residual is
      the pipeline's fixed ~2 ms (first scatter, last gather, the master's copy
      into the output, ~0.3 ms of host time per chunk), not the interconnect:
      both configurations are
      compute-bound by 2.2× and 10× respectively. Calibration reaches the same
      weights (`[1.0, 1.0]` on identical devices) when its throughput stage
      runs on quiet devices, matching the uncalibrated rows in 18 of 20 runs —
      but it is a measurement of a shared machine: two runs weighted a busy
      `cuda:1` at 0.72 and 0.81 (30.1 ms against 17.0 ms for `cuda:0` on
      identical hardware) and ran `intersect` at 1.42× and 1.54×, and one
      demoted the reflection remote share to 1/2 and ran at 1.47×. All stayed
      above 1.4×, because the operation is compute-bound by a wide margin. The
      criterion is met by the dispatch, not by the calibrator.
- [x] Transfer-bound configurations are covered by the fallbacks rather than
      by a speedup, with the two fallbacks carrying different strengths:
      - The **row floor is a guarantee.** Below `min_rays_per_device ×
        len(devices)` rows (524,288 by default) the batch runs on the master
        bitwise as a single-device `Scene` would, verified at the
        524,287/524,288 boundary.
      - **Calibration is a measurement, not a guarantee.** A light
        configuration (192-vertex grid, 0.31 ns/ray) shards at 0.27× and
        `calibrate_devices()` usually answers it with a zero remote weight,
        which the dispatcher runs as the single-device call it is, at
        **1.00×** — that is what the recorded light rows show. But the
        refinement ladder keeps the *largest* remote share within its 3%
        tolerance of the fastest rung, so on a near-crossover operation the
        tie-break can keep a split that then loses at run time. For light
        1-bounce `trace_reflections` (0.67 ns/ray transfer against 0.27 ns/ray
        compute) that happened in 3 of 20 consecutive full benchmark runs
        (0.86×, 0.85×, 0.38×), in 5 of 6 back-to-back calibrations
        (0.79–0.89×), and in 2 of 3 in a third study (0.71–0.84×). The claim is
        therefore bounded:
        *calibration will not knowingly keep a split it measured as more than
        the refinement tolerance slower than the master alone*, and inside
        that band the answer flips between runs. Ship pinned weights
        (`MultiDeviceOptions(weights=[1.0, 0.0])`) for such operations; the
        benchmark labels the rows `NEAR-CROSSOVER`. Full evidence:
        [`multi_gpu_operations.md`](multi_gpu_operations.md) §5.4.
- [x] Sharded `per_ray` results stay bitwise the single-device results at the
      benchmark's sizes (agreement 1.0 on every row of the recorded table), and
      merged accumulation grids stay within float32 merge order.
- [x] Benchmark added: `benchmarks/torch/benchmark_multi_device.py`, which
      runs both configurations, reports single vs multi times, speedups,
      calibrated weights, the per-chunk plan and each calibration's margin over
      running master-only (flagging the near-crossover decisions that do not
      reproduce), and runs on one GPU as a baseline collector. Recorded table,
      machine, cross-run spread and contention caveat:
      [`multi_gpu_operations.md`](multi_gpu_operations.md) §5.
- [ ] Heterogeneous pair (if available): calibrated weights beat naive 50/50
      by a recorded margin (informational, no hard threshold). **Not
      available** — the verification machine's two devices are identical, and
      calibration accordingly answers 1.00/1.00 on compute-bound probes.
- [x] Single-GPU parity gate (D9): the full `benchmark_torch_native.py` set
      against the pre-Phase-2 baseline (`cc5f0f9`), six interleaved runs of
      each, min per metric: six of eight metrics within ±1.6%; the two
      exceptions are `nearest_edge` at −3.8% (i.e. faster after the change,
      on a metric whose per-run spread on this contended host is ±10%) and
      `diffraction_direct` at +2.8% on a 0.19 ms measurement whose samples
      spanned 0.20–1.70 ms. A single-device `Scene` never imports the
      orchestration layer (asserted by a subprocess probe), so this is a
      measurement of unchanged code and the residual is the machine.

### Phase 3 — Process-per-GPU and multi-node recipes (~1–2 weeks)

Work items:

1. Single-node recipe: `torchrun` one rank per GPU, `CUDA_VISIBLE_DEVICES`
   per rank, rank-local scene build, NCCL all-reduce of grids/vertex grads
   (automatic under DDP when vertices are module parameters; manual
   `all_reduce` example otherwise). Works for both backends; this is the only
   Dr.Jit path.
2. Multi-node recipe: same script under multi-node `torchrun`; document that
   traffic is only grids + gradients (scene-sized, N-independent).
3. Example scripts under `examples/torch/distributed/` with a
   smoke-test harness.
4. Dr.Jit pure-CUDA hardening (quality items, not blocking for
   process-per-GPU): device recorded in `CudaBuffer`, `cudaDeviceSynchronize`
   → stream sync, treelet H2D off the default stream, `DfrAccumParams` as a
   kernel argument instead of file-scope `__constant__`, device/stream
   parameters in the public builder ABI (`edge_bvh.h`).

Acceptance:

- [x] 2-rank single-node example produces grids/grads matching single-process
      execution within float tolerance —
      `test_distributed_recipe.py`, 4 tests, green: both examples run under a
      real `torchrun --nproc_per_node=2`, the ranks' final parameters are
      bitwise equal, and the all-reduced grid matches a single-process
      single-device launch of the full sample count.
- [ ] …and runs in the multi-GPU CI job. **Open** — the job is declared in
      [`.github/workflows/multi_gpu.yml`](../../.github/workflows/multi_gpu.yml)
      and is inert until a self-hosted 2-GPU runner exists (execution record
      item 6). Run by hand meanwhile:
      [`multi_gpu_operations.md`](multi_gpu_operations.md) §8.
- [ ] Dr.Jit 2-rank example (each rank pinned via `CUDA_VISIBLE_DEVICES`)
      runs the reflection benchmark with ≥1.8× aggregate throughput. **Open** —
      the recipe and its Dr.Jit variant are documented and the route works, but
      no aggregate-throughput measurement was taken.
- [ ] Multi-node invocation documented and exercised at least manually
      (recorded in the PR); no code path differs from single-node beyond the
      rendezvous. **Half done, honestly.** The invocation is documented
      ([`multi_gpu_operations.md`](multi_gpu_operations.md) §6) and the "no
      differing code path" half is checked by reading: the examples consume
      only `RANK`, `LOCAL_RANK` and `WORLD_SIZE`, and per-step traffic is one
      `[V, 3]` gradient or a fixed set of grids. It has **not** been executed —
      this repository has a single node.
- [ ] Hardening items each keep the Dr.Jit test suite bitwise-green.
      **Partial.** Numerics were verified bitwise-unchanged and the committed
      PTX closures were untouched (digest test green), with `test_geometry`
      63/63, `test_device_binding` 4/4 and the edge-BVH modules 23/23 green on
      the Linux verification host. Outstanding: a full Dr.Jit suite run on the
      Windows/RTX 5080 baseline machine, where the pre-existing
      `test_diffraction_paths_parity` platform divergence recorded in ADR-0038's
      platform note also has to be re-checked.

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

As landed, two of those items read differently from the proposal:

- Item 2's typing work is on the **private** module (`_multi.py`, annotated
  inline; the `_multi.pyi` it briefly carried went when the Torch package
  dropped every stub it had). There is no `rayd.torch.multi`: the public
  surface is
  `Scene(devices=..., options=...)`, `MultiDeviceOptions`,
  `Scene.calibrate_devices()`, `Scene.device_weights`,
  `Scene.device_weights_for(operation)` and the two lane-window parameters,
  and ADR-0038's stop conditions forbid adding a public name under that path.
- Item 4 was not needed: no `.cu` reachable from a committed PTX module
  changed, so `ptx_sources.json` is untouched and its digest test stayed green
  throughout rather than being repaired afterwards.

Acceptance:

- [x] ADR accepted; contract tests green in both directions (declaration ↔
      build) —
      [`ADR-0038`](../adr/0038-replicated-multi-device-execution.md) is
      Accepted (2026-07-27), `tests/test_adr0038_multi_device.py` guards it
      against the contracts, both capability copies, the `MultiDeviceOptions`
      defaults, the lane-window defaults in Python and in the dispatcher
      schema, and this plan; `tests/test_shared_operation_contract.py` carries
      the hard-coded per-operation shardability table.
- [x] Capability visible from both backends' `_capabilities.py` with correct
      values; manifest/typing tests pass — `multi_device_replicated` is
      `torch: true`, `drjit: false`, both copies repinned `_SCHEMA_SHA256`, and
      `tests/test_public_api_manifest.py` is green.
- [ ] Multi-GPU CI job green on the 2-device matrix. **Open** — the job exists
      (`.github/workflows/multi_gpu.yml`) but is inert: no self-hosted
      2-GPU runner is registered, and `workflow_dispatch` is its only trigger.
      The set it declares — the multi-device modules fresh-process, the
      distributed recipe, the governance suite — was run by hand on the
      verification host instead
      ([`multi_gpu_operations.md`](multi_gpu_operations.md) §8).
- [x] Single-GPU CI unchanged — `ci.yml`, `stable-abi-ci.yml` and `pypi.yml`
      are untouched by this work. None of them has a CUDA device, so they were
      never running the GPU subset in the first place.
- [x] PTX digest test green (regenerated if touched) —
      `tests/test_ptx_source_digest.py` green with
      `drjit/ptx_sources.json` untouched: the lane window lives in the
      Torch backend and the Dr.Jit Phase 3 hardening changed host and object
      translation units only.

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
