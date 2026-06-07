# RayDTorch CUDA / OptiX Performance Analysis

> Static source analysis by Claude (cuda-optimize). No profiler run yet — items are
> labelled `[verified]` (provable from source) or `[needs Nsight]` (impact needs
> measurement). Every change below should be re-measured with `nsys`/`ncu` against the
> repository benchmark before being accepted.
>
> - Target GPU: NVIDIA GeForce RTX 5080 (Blackwell, compute_120), Torch CUDA 12.8, Windows.
> - Benchmark of record: `tests/benchmark_raydtorch_native.py` and
>   `tests/benchmark_rayd_vs_raydtorch.py`.
> - Known-slow paths per [raydtorch_native_performance.md](raydtorch_native_performance.md):
>   **scene build > reflection trace > nearest-edge**.

## Current implementation status (2026-06-07)

Implemented in the current worktree:

- CUDA build defaults now target modern GPUs instead of stale `sm_52`:
  `75-real;80-real;86-real;89-real;120-real;120-virtual`.
- Single-config native builds default to `Release` when the caller did not set
  `CMAKE_BUILD_TYPE`.
- OptiX PTX builds use fast math behind `RAYDTORCH_OPTIX_FAST_MATH` and now
  compile explicit `compute_75` PTX instead of relying on nvcc's old default
  virtual architecture.
- `Scene.intersect(ray, flags=...)` now exposes the RayD-compatible
  `RayFlags.None/Geometric/ShadingN/UV/All` contract. The t-only/minimal
  path is reached through `flags=getattr(rt.RayFlags, "None")`.
- No-AD `intersect` uses an on-demand native path:
  - `None`: launches OptiX and returns only `t`.
  - `Geometric`: returns `t`, `p`, barycentric, ids, and `geo_n`.
  - `ShadingN`: returns `t` and `n`.
  - `UV`: returns `t` and `uv`.
  - `All`: returns the full legacy intersection fields.
  Reverse-mode and forward-mode AD keep the full existing autograd path.
- Edge-search radii no longer copy six full edge SoA tensors back to CPU;
  a CUDA reduction returns only bbox/max-length partials.
- The edge topology path is explicitly named `build_edge_topology_cpu_fallback`.
  It still uses CPU fallback semantics, but now uses `unordered_map` plus sorted
  keys for deterministic edge ids.
- Reflection no-AD tracing avoids allocating/writing autograd tape-only arrays.
- Reflection trace uses bounce-major internal output storage when
  `max_bounces > 1`, then returns the existing public `[ray, bounce]` tensors.
  This targets coalesced per-bounce stores without changing the Python or AD
  contract. `max_bounces == 1` keeps the original layout to avoid a pointless
  transpose in the RayD comparison benchmark.
- Visibility-style OptiX traces use first-hit termination where ignore lists are
  not required.
- Reflection and diffraction accumulation now use warp-aggregated same-cell
  atomics for the hot complex/power field scatter paths.
- Diffraction path export uses warp-aggregated path-slot reservation and a
  single primary-scene launch for order-1 export. It also avoids redundant
  zero-field stores and skips unused p1/p2 component writes for order-1 paths.
- Diffraction direct no-AD accumulation bypasses autograd tape export; AD/JVP/VJP
  still uses the full tape-producing path.
- Dynamic edge sync reuses compatible edge GAS/AABB/temp buffers and rebuilds
  in place. A direct `OPTIX_BUILD_OPERATION_UPDATE` refit was tested but caused
  a severe post-sync nearest-edge traversal regression on the benchmark shape,
  so it is not retained.

Fallbacks that remain intentionally present:

- Edge topology construction is still a CPU fallback because the full GPU
  sort/segmented-reduction implementation is not yet in this patch.
- Intersect AD/JVP/VJP still uses the full legacy field path; the selective
  RayFlags path is no-AD only.
- Diffraction direct/chain AD paths keep their existing full-output contract.
  The no-tape direct path is only selected when neither reverse-mode nor
  forward-mode AD is active.
- Edge GAS true refit remains a guarded backlog item. Current sync uses
  rebuild-in-place when compatible, otherwise full `build_edge_accel` fallback.

Latest verification:

- Incremental native build succeeded via `scripts/dev_build_native.ps1`.
- `python -m unittest tests.raydtorch_native.test_multipath -v`: 20 passed.
- `python -m unittest discover tests.raydtorch_native -v`: 62 passed, 12 skipped.
- Opt-in RayD parity:
  `RAYDTORCH_RUN_DR_JIT_PARITY=1 python -m unittest tests.raydtorch_native.test_drjit_parity -v`:
  12 passed. The known `jitc_llvm_init()` warning appeared and did not affect assertions.
- `git diff --check`: no whitespace errors; Git only reported existing LF/CRLF
  conversion warnings for touched files.

Latest RayD vs RayDTorch static benchmark, `grid=64`, `queries=4096`,
`warmup=8`, `repeat=60`, RayD package resolved to `E:\Code\RayDi\rayd`:

| Operation | RayD ms | RayDTorch ms | Status |
|---|---:|---:|---|
| build | 2287.357 | 1527.668 | RayDTorch faster |
| intersect `RayFlags.None` | 0.1194 | 0.0387 | RayDTorch faster |
| intersect `RayFlags.All` | 0.1257 | 0.0469 | RayDTorch faster |
| nearest edge | 1.1977 | 1.0324 | RayDTorch faster |
| reflection trace | 0.2248 | 0.1738 | RayDTorch faster |
| diffraction direct | 0.3921 | 0.2734 | RayDTorch faster |
| diffraction paths | 0.3042 | 0.2366 | RayDTorch faster |

Latest dynamic benchmark, `grid=64`, `queries=4096`, `warmup=8`, `repeat=60`:

| Operation | RayD ms | RayDTorch ms | Status |
|---|---:|---:|---|
| build | 2298.370 | 1521.933 | RayDTorch faster |
| intersect `RayFlags.None` | 0.1274 | 0.0463 | RayDTorch faster |
| intersect `RayFlags.All` | 0.1372 | 0.0505 | RayDTorch faster |
| nearest edge | 1.2131 | 1.0659 | RayDTorch faster |
| reflection trace | 0.2249 | 0.1944 | RayDTorch faster |
| diffraction direct | 0.5026 | 0.3033 | RayDTorch faster |
| diffraction paths | 0.2601 | 0.2250 | RayDTorch faster |

Latest RayDTorch-native multi-bounce check, `grid=64`, `queries=4096`,
`warmup=8`, `repeat=60`, `max_bounces=4`:

| Operation | RayDTorch ms |
|---|---:|
| build | 1522.405 |
| dynamic sync | 0.882 |
| reflection trace | 0.2250 |
| diffraction direct | 0.2331 |
| diffraction paths | 0.2805 |

The native benchmark's `nearest_edge` uses random 3D points after a dynamic sync,
which drives many queries to the full-radius edge tier; it is not the same
near-surface edge-query shape as the RayD comparison benchmark above.

For this benchmark shape, RayDTorch now meets the current RayD-comparison target:
intersect and reflection are substantially faster; nearest-edge, direct
diffraction, order-1 diffraction paths, and scene build are faster or at parity.
Keep release-size and Nsight-backed benchmarks before claiming broad superiority
across all scenes and multipath workloads.

## Contents

- [Executive summary](#executive-summary)
- [Priority table](#priority-table)
- [P0 — Build configuration](#p0--build-configuration)
- [P1 — Scene build](#p1--scene-build)
- [P1 — Nearest-edge query](#p1--nearest-edge-query)
- [P1 — Reflection trace](#p1--reflection-trace)
- [P2 — Reflection accumulation / EPC / dedup](#p2--reflection-accumulation--epc--dedup)
- [P2 — Diffraction](#p2--diffraction)
- [Cross-cutting — host glue, allocations, syncs](#cross-cutting--host-glue-allocations-syncs)
- [Measurement plan](#measurement-plan)
- [Suggested execution order](#suggested-execution-order)

---

## Executive summary

The three slowest paths each have a clear primary suspect that is provable from source:

1. **Scene build (~1550 ms @ grid 192)** is dominated by **host-side, single-threaded edge
   topology extraction** using `std::map` plus a synchronous `faces.cpu()` round-trip, and
   by a **6× D2H copy + serial CPU bounding-box reduction** to derive edge search radii.
   Almost none of the build work that *could* run on the GPU does.

2. **Reflection trace** originally wrote per-bounce outputs in a **ray-major layout**
   (`slot = ray*B + bounce`) across ~17–24 separate SoA arrays. The current worktree now
   uses bounce-major internal storage for `max_bounces > 1`, while returning the same public
   `[ray, bounce]` tensors. Nsight should still verify store coalescing and transpose cost.

3. **Nearest-edge query** issues **one OptiX launch + one finalize kernel + one params
   H2D copy per radius tier** (up to 3), always over the full query width, and the shared
   edge pipeline reserves **16 payload registers** (needed only by top-k) for the point/ray
   paths too, depressing occupancy.

Above all of these sits a **build-configuration defect**: the cached build targets
`sm_52` (Maxwell) on a Blackwell GPU. Fixing the architecture is zero-risk and may lift
every kernel before any code is touched.

This document is a backlog, not a patch set. Confirm impact ordering with the
[measurement plan](#measurement-plan) before investing in the larger refactors (items 4, 5,
15).

---

## Priority table

| Level | Items | One-liner |
|---|---|---|
| **P0** | 1, 2, 3 | Architecture, build type, fast-math/PTX arch fixes landed |
| **P1 build** | 4, 5, 6, 7 | CPU edge topology + D2H radii fixed; edge sync partially improved; no compaction |
| **P1 edge** | 11, 12 | Per-tier launches + 16-payload occupancy hit |
| **P1 refl** | 15, 16, 17 | Bounce-major trace writes landed for B>1; hit gather and split traces remain |
| **P2 accum** | 19, 20, 25, 26 | Same-cell warp atomics and path-counter aggregation landed; deeper reductions remain |

Impact legend: **High** = likely measurable on the benchmark; **Med** = real but
secondary; **Low** = correctness-neutral cleanup / small constant.

---

## P0 — Build configuration

Highest leverage, smallest change. Do these first; they may shift the whole ranking.

### 1. CUDA architecture compiled as `sm_52` on a Blackwell GPU `[verified] — High`

- Evidence: `artifacts/skbuild/CMakeCache.txt` → `CMAKE_CUDA_ARCHITECTURES:STRING=52`.
  Neither [CMakeLists.txt](../CMakeLists.txt) nor [pyproject.toml](../pyproject.toml) pins an
  architecture, so CMake fell back to the legacy default of 52.
- Impact: every non-OptiX `.cu` kernel (`cache_kernels`, `geometry_*`, `dedup`,
  `epc_field`, `backward`, `accum_ad`, …) is generated for Maxwell and JIT-recompiled from
  PTX at runtime, using old-arch scheduling/occupancy heuristics, plus first-run JIT latency.
- Fix direction depends on whether the build is for **local dev** or a **published wheel**:
  - **Local dev (single known GPU):** `set(CMAKE_CUDA_ARCHITECTURES native)` (builds only for
    the build machine's GPU → `sm_120` here) or explicit `120`. Simplest and fastest to
    compile; do **not** ship this.
  - **Published wheel (others install it):** use a **multi-architecture list with a low
    virtual/PTX baseline**, e.g.
    `set(CMAKE_CUDA_ARCHITECTURES 75-real;80-real;86-real;89-real;120-real;120-virtual)`
    (or via pyproject:
    `[tool.scikit-build.cmake.define] CMAKE_CUDA_ARCHITECTURES = "75-real;80-real;86-real;89-real;120-real;120-virtual"`).
    The `-real` entries embed optimized SASS for each target GPU (no JIT); the trailing
    `120-virtual` keeps PTX for forward-compat with future GPUs.
- Compatibility note (why this matters for distribution): PTX JIT is **forward-only** —
  `compute_XX` PTX runs on compute capability ≥ XX, never below; `sm_XX` SASS is arch-bound.
  - The **lowest virtual/PTX entry sets the minimum supported GPU.** To keep supporting old
    cards, lower the baseline (e.g. add `52-virtual` for Maxwell, `61`/`75` otherwise).
  - Adding `120` does **not** drop old-GPU support — only *replacing* the broad baseline with
    a single high arch (`120` or `native`) does. A `120`-only or `native` wheel will **fail to
    load on any pre-Blackwell GPU**, so never publish those.
  - The current `52` build embeds `sm_52` SASS + `compute_52` PTX, so it runs everywhere from
    Maxwell up — but on this RTX 5080 it runs *only* via `compute_52` PTX JIT (slow, plus
    first-run JIT latency), which is exactly the defect this item fixes.
  - Trade-off: each `-real` arch adds a cubin → larger wheel and longer compile time. Pick the
    `-real` set to match the GPUs users actually have (Turing 75 / Ampere 80,86 / Ada 89 /
    Blackwell 120 above).
- Risk: none for correctness. Re-measure the whole benchmark — this can move every number.

### 2. `CMAKE_BUILD_TYPE` absent / empty in the cache `[implemented] — Med`

- Evidence: `CMAKE_BUILD_TYPE` does not appear in `artifacts/skbuild/CMakeCache.txt`.
- Impact: host glue (`scene_cache.cpp`, the `ops.cpp` files — heavy STL + tensor code) may
  compile without `/O2`. This directly touches the build path, which is CPU-bound.
- Implemented: single-config native builds now default to `Release` when
  `CMAKE_BUILD_TYPE` was not explicitly provided. Multi-config Visual Studio builds still
  use the requested configuration.

### 3. OptiX PTX compiled without `--use_fast_math` (and no arch flag) `[implemented] — Med`

- Evidence: every `--ptx` custom command in [CMakeLists.txt:75-310](../CMakeLists.txt#L75)
  passes only `--std=c++17`.
- Impact: the diffraction / accumulation kernels do many `sincosf` / `sqrtf` / divides (see
  items 28, 19) at full precision. OptiX re-optimizes the PTX, but fast-math semantics must be
  set at the source compile.
- Implemented: PTX custom commands use `RAYDTORCH_OPTIX_NVCC_FLAGS`, which includes
  `--gpu-architecture=compute_75` and, by default, `--use_fast_math`. The explicit PTX
  architecture was required for warp intrinsics such as `__match_any_sync`.

---

## P1 — Scene build

Target of record: `build_ms` ≈ 1550 (native, grid 192) / ≈ 142 (grid 64 vs RayD 95).

### 4. Edge topology built on the CPU, single-threaded, via `std::map` `[verified] — High`

- Location: [scene_cache.cpp:305-381](../src/torch_ext/scene/scene_cache.cpp#L305)
  `build_edge_topology`.
- Problem: `mesh.faces.cpu()` forces a synchronous D2H; then every triangle inserts its 3
  undirected edges into a `std::map<pair<int,int>, vector<pair<int,int>>>` (red-black tree +
  a heap allocation/`vector` growth per edge). At grid 192 the triangle count is large, so
  this is the most probable build hotspot.
- Fix direction: move edge extraction to the GPU — emit all 3 edges per face with a
  canonical (min,max) key, `cub`/`thrust` sort by key, then a segmented reduction to pair
  faces and detect boundary (count==1) vs manifold (count==2) edges. At minimum, drop the
  `.cpu()` round-trip and switch to `unordered_map` with `reserve`.

### 5. Edge search radii via 6× `.cpu()` D2H + serial CPU bbox reduction `[verified] — High`

- Location: [scene_cache.cpp:484-497](../src/torch_ext/scene/scene_cache.cpp#L484)
  (six `scene.edge_*.cpu()` copies) feeding
  [compute_edge_search_radii:383-449](../src/torch_ext/scene/scene_cache.cpp#L383).
- Problem: the edge SoA was just produced on the GPU, then copied back to the host as six
  independent synchronous transfers, only to run a serial min/max + max-edge-length loop.
- Fix direction: compute the bounding box and max edge length on-device (a reduction kernel
  or `at::aminmax`/`cub`), returning a handful of scalars instead of full arrays.

### 6. Up to 3 edge GAS rebuilt on every build *and* every sync `[partial] — Med-High`

- Location: [scene_cache.cpp:498-556](../src/torch_ext/scene/scene_cache.cpp#L498) — loops
  `radii.size()` times, each iteration `optixAccelComputeMemoryUsage` + `optixAccelBuild`.
- Problem: three custom-primitive GAS are sized and built serially. Historically
  `sync_scene` called `build_edge_accel` again, reallocating and rebuilding every tier.
- Fix direction: (a) reassess whether 3 radius tiers are needed or whether one GAS with
  raygen-side tiered `tmax` suffices; (b) for the dynamic path, refit with `ALLOW_UPDATE` +
  `OPTIX_BUILD_OPERATION_UPDATE` instead of full rebuilds.
- Current state: compatible dynamic syncs reuse edge AABB/GAS/temp buffers and rebuild
  the GAS in place. A direct `OPTIX_BUILD_OPERATION_UPDATE` refit was tested and rejected
  because it made post-sync nearest-edge traversal much slower on the native benchmark.
  Single-GAS/tier-collapse work remains open.

### 7. No acceleration-structure compaction anywhere `[verified] — Med`

- Location: triangle GAS [scene_cache.cpp:65-66](../src/torch_ext/scene/scene_cache.cpp#L65),
  IAS [:145](../src/torch_ext/scene/scene_cache.cpp#L145), edge GAS
  [:530](../src/torch_ext/scene/scene_cache.cpp#L530) — all `PREFER_FAST_TRACE` only.
- Problem: GAS output buffers are left uncompacted: larger VRAM footprint and worse
  traversal cache locality. The cuda-optimize checklist explicitly calls this out.
- Fix direction: add `OPTIX_BUILD_FLAG_ALLOW_COMPACTION` for static geometry and run a
  compaction pass (one extra build-time cost for a smaller, faster AS).

### 8. `refresh_global_geometry` does many tiny tensor ops + 12 separate SoA buffers `[verified] — Med`

- Location: [scene_cache.cpp:172-241](../src/torch_ext/scene/scene_cache.cpp#L172).
- Problem: per-mesh `at::full` / `at::arange` / `(faces + vertex_offset)` / `at::cat` spawn
  many small kernels and temporaries; then 12 distinct `tri_*` tensors
  (`tri_p0_x … tri_fn_z`) are allocated.
- Fix direction: write a packed layout directly from one `compute_*` kernel; generate
  shape/local ids in a single kernel; minimize `cat` calls.

### 9. Triangle GAS built serially per mesh `[verified] — Med`

- Location: [scene_cache.cpp:584-586](../src/torch_ext/scene/scene_cache.cpp#L584).
- Fix direction: for multi-mesh scenes, batch into a single build with multiple build
  inputs, or build on concurrent streams.

### 10. IAS instances constructed on host and re-copied every sync `[verified] — Low-Med`

- Location: [build_triangle_ias:115-134](../src/torch_ext/scene/scene_cache.cpp#L115).
- Problem: the identity transforms never change, yet the instance buffer is rebuilt and
  re-`cudaMemcpyAsync`'d on every sync.
- Fix direction: reuse the instance buffer and refit the IAS (`ALLOW_UPDATE`).

---

## P1 — Nearest-edge query

### 11. Tiered query = N launches + N finalize kernels + N params H2D `[verified] — Med-High`

- Location: [edge_forward.cu:325-372](../src/torch_ext/edge/edge_forward.cu#L325) (point) and
  [:431-483](../src/torch_ext/edge/edge_forward.cu#L431) (ray) loop over `scene.edge_accels`.
- Problem: each radius tier issues a `cudaMemcpyAsync(params)`
  ([edge_forward.cu:257](../src/torch_ext/edge/edge_forward.cu#L257)) + an `optixLaunch` + a
  finalize kernel, and the launch width is always the full query count even after most
  queries resolve in tier 0.
- Fix direction: collapse into a single GAS / single launch (raygen grows the radius and
  early-outs), or stream-compact the unresolved set before later tiers to shrink the launch.

### 12. Edge pipeline reserves 16 payload registers for all raygens `[verified] — Med`

- Location: [optix_context.cpp:248](../src/torch_ext/scene/optix_context.cpp#L248)
  (`numPayloadValues = 16`). Top-k (k≤8) needs 16; point uses 4, ray uses 5.
- Problem: payload count is per-pipeline, so the point/ray raygens pay the full 16-register
  reservation, lowering megakernel occupancy.
- Fix direction: split top-k into its own pipeline (payload ≤ 6) or use payload semantics;
  keep the point/ray pipeline at minimum payload. Confirm with `launch__registers_per_thread`.

### 13. AoS point/edge coordinates loaded as 3 scalar loads `[verified] — Low`

- Location: [edge_forward.cu:17](../src/torch_ext/edge/edge_forward.cu#L17) `make_aos_f3`;
  [edge_optix.cu:31-51](../src/torch_ext/edge/edge_optix.cu#L31).
- Fix direction: with 16-byte alignment, vectorize via `float4`/reinterpret (the edge SoA is
  already split into component arrays, so this mainly helps the AoS query points).

### 14. Ray query anyhit calls `optixIgnoreIntersection()` for every candidate `[verified] — Low-Med`

- Location: [edge_optix.cu:380-391](../src/torch_ext/edge/edge_optix.cu#L380). The bigger the
  search radius, the more candidates, the more anyhit invocations.
- Fix direction: couple with items 6/11 to tighten the radius tiers; ensure custom-primitive
  AABBs are not over-inflated.

---

## P1 — Reflection trace

### 15. Ray-major output layout → strided, uncoalesced stores `[implemented for B>1] — High`

- Location: [trace_optix.cu:187-216](../src/torch_ext/reflection/trace_optix.cu#L187), feeding
  ~24 independent SoA output arrays defined in
  [trace_params.h:44-67](../include/raydtorch/reflection/trace_params.h#L44).
- Original problem: outputs were indexed `slot = ray_index * B + bounce`. For a fixed
  bounce, adjacent threads wrote addresses that differed by `B`, so each warp store could
  degenerate into many transactions.
- Implemented: for `max_bounces > 1`, raygen writes bounce-major storage
  (`bounce * n_rays + ray`) and host glue transposes back to public ray-major tensors.
  `max_bounces == 1` keeps ray-major storage because both layouts are contiguous and a
  transpose would only add overhead.
- Confirm with `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio` on
  `__raygen__reflection_trace`.

### 16. Each hit re-gathers 12 scattered `tri_*[global_prim]` to recompute hit point/normal `[verified] — Med`

- Location: [trace_optix.cu:154-173](../src/torch_ext/reflection/trace_optix.cu#L154).
  `global_prim` is effectively random across a warp → 12 scattered global loads per bounce.
- Fix direction: pack the triangle SoA into aligned groups (`float4`) to cut transactions, or
  evaluate using the OptiX barycentrics + vertex buffer directly.

### 17. `split_mode` traces twice per bounce (primary + secondary) `[verified] — Med`

- Location: [trace_optix.cu:128-136](../src/torch_ext/reflection/trace_optix.cu#L128) and the
  trailing segment [:238-246](../src/torch_ext/reflection/trace_optix.cu#L238).
- Fix direction: evaluate merging into a single IAS with multiple instances so one traversal
  replaces the second per-bounce trace.

### 18. Shared multipath pipeline uses a hardcoded oversized stack `[verified] — Low`

- Location: [optix_pipeline.cpp:231](../src/torch_ext/common/optix_pipeline.cpp#L231) —
  `optixPipelineSetStackSize(pipeline_, 0, 0, 4096, 2)`. The context pipelines, by contrast,
  compute exact sizes via `optixUtilComputeStackSizes`.
- Fix direction: compute the multipath stack precisely too, reducing VRAM and potential spill.

---

## P2 — Reflection accumulation / EPC / dedup

> Mostly from a focused sub-agent read of the accumulation modules; treat impact as
> `[needs Nsight]` unless noted.

### 19. Complex-field accumulation via up to 7 `atomicAdd` into hashed cells `[implemented warp aggregation] — High`

- Location: `accum_optix.cu:377-384`, coherent branch `:445-460`.
- Problem: many threads atomically add into the same cell — serialized, complex-valued
  (re/im split into separate atomics per component).
- Implemented: hot reflection and diffraction field/power scatter paths now use same-cell
  warp aggregation before the global atomic. This changes same-warp floating-point addition
  order, so parity tests are the correctness guard. A sort/segmented-reduction design is
  still a possible larger follow-up for heavy cell contention.

### 20. Audit occlusion/visibility ray flags for `TERMINATE_ON_FIRST_HIT | DISABLE_ANYHIT` `[needs check] — High if confirmed`

- Already correct: `visibility_optix.cu` uses `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`.
- To verify: the `DISABLE_ANYHIT`-only traces in `accum_optix.cu` / `epc_optix.cu`. **If a
  trace is an occlusion/shadow test**, add `TERMINATE_ON_FIRST_HIT` (and resolve in miss /
  `optixHitObject`). **If it is a nearest-reflection-point search**, the current closest-hit
  semantics are correct — do not change. Decide per call site.

### 21. EPC raygen holds ~500 B of stack-local arrays per thread `[needs Nsight] — Med`

- Location: `epc_optix.cu:437-550` — five `[ReflEpcMaxBounces=8]` float3 arrays.
- Problem: high register / local-memory footprint depresses occupancy.
- Fix direction: shrink live state, process bounces in chunks, check for local-memory spills
  (`launch__registers_per_thread`, local load/store metrics).

### 22. dedup blocks the host with `cudaStreamSynchronize` to return `unique_count` `[needs Nsight] — Med`

- Location: dedup.cu host path (count copy + sync).
- Fix direction: keep the count device-resident and consume it from a follow-up kernel;
  avoid the host stall that serializes the next op.

### 23. EPC field / backward recompute forward state per bounce `[needs Nsight] — Med`

- Location: `epc_field.cu:186-259`, `backward.cu:132-215`.
- Problem: origins / directions / normals are re-gathered and recomputed each bounce instead
  of being cached or passed down.
- Fix direction: cache the forward intermediates to cut repeated global traffic.

### 24. dedup compact writes 13 scattered fields per bounce `[needs Nsight] — Med`

- Location: `dedup.cu:285-302`.
- Fix direction: structured / merged writes.

---

## P2 — Diffraction

### 25. Path counter is a single global `atomicAdd(out_count, 1)` `[implemented warp aggregation] — High`

- Location: `paths_optix.cu:250` / `:394`.
- Problem: a global atomic serializes all hit-writes.
- Implemented: path export reserves output slots per warp using one `atomicAdd` per active
  warp group. Prefix-sum allocation remains a larger alternative.

### 26. Coherent UTD atomics per cell (6 complex components + counters) `[implemented warp aggregation] — High`

- Location: `accum_optix.cu:445-460` (same class of issue as item 19).
- Implemented: coherent direct/multi field outputs now use same-cell warp aggregation for
  the field components and per-cell counts. Block-level or sorted reductions remain open
  for very high contention workloads.

### 27. Path output scatters 12 SoA complex components via `out_idx` `[needs Nsight] — Med`

- Location: `paths_optix.cu:261-280`.
- Fix direction: merged / packed writes.

### 28. `sincosf` / `sqrtf` / phase math at full precision `[needs Nsight] — Med`

- Location: `paths_optix.cu:256-259` and similar.
- Fix direction: pairs with P0-3 (`--use_fast_math`); validate against parity tests.

### 29. AD unit-JVP loop: 36+ serial `add_unit_vjp` + scattered `atomicAdd(ptr+index)` `[needs Nsight] — Med (backward only)`

- Location: `accum_ad.cu:1518-1627`, `:1774-1775`.
- Problem: many small atomics + per-call `nullptr` branches + large intermediate structs
  (register pressure).
- Fix direction: batch gradient accumulation, prune the nullptr branches, shrink live state.

---

## Cross-cutting — host glue, allocations, syncs

### 30. Many small `at::zeros` / `at::full` / `at::empty` allocations before launches `[verified] — Low-Med`

- Location: `reflection/ops.cpp:252-283`, `diffraction/ops.cpp:255-278`, and similar.
- Fix direction: batch-allocate output buffers / reuse scratch buffers across calls.

### 31. Every OptiX launch re-copies the full params struct to device `[verified] — Low`

- Location: [optix_pipeline.cpp:306](../src/torch_ext/common/optix_pipeline.cpp#L306),
  [edge_forward.cu:257](../src/torch_ext/edge/edge_forward.cu#L257), and each `ops.cpp`.
- Problem: large struct re-transferred per launch.
- Fix direction: cache the invariant portion, or stage from a pinned host buffer.

### 32. `hitgroup_record_capacity` rounds up to a minimum of 64 SBT records `[verified] — Low`

- Location: [optix_pipeline.cpp:107](../src/torch_ext/common/optix_pipeline.cpp#L107).
- Problem: most pipelines need a single record; an oversized SBT hurts cache locality.
- Fix direction: verify the minimum is justified; size to the actual record count.

### 33. intersect raygen loads ray origin/dir as 3 scalar loads `[verified] — Low`

- Location: [optix_intersect.cu:22-29](../src/torch_ext/scene/optix_intersect.cu#L22).
- Fix direction: vectorize after guaranteeing alignment.

---

## Measurement plan

Measure before and after each change; confirm the ranking before the large refactors
(items 4, 5, 15). On Windows, Nsight Compute counters need elevated rights or the GPU
performance-counter restriction lifted, else `ERR_NVGPUCTRPERM`.

```powershell
# 0) Repository benchmark of record
conda run -n witwin2 python -m tests.benchmark_raydtorch_native --grid 192 --queries 65536

# 1) System timeline — is scene build CPU-bound? where are the D2H copies / AS builds?
nsys profile -o prof --stats=true --force-overwrite=true `
  python -m tests.benchmark_raydtorch_native --grid 192 --queries 65536

# 2) Reflection trace — store coalescing + occupancy (item 15)
ncu --set full -k "raygen__reflection_trace" -c 5 -o refl <app>
ncu --metrics `
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,`
  launch__registers_per_thread,launch__occupancy_limit_registers `
  -k "raygen__reflection_trace" -c 5 <app>

# 3) Nearest-edge — payload register pressure / occupancy (item 12)
ncu --metrics `
  launch__registers_per_thread,launch__occupancy_limit_registers,`
  sm__warps_active.avg.pct_of_peak_sustained_active `
  -k "raygen__edge" -c 5 <app>
```

What to look for:

- Build: in the `nsys` timeline, a large CPU gap with no GPU work during build confirms
  items 4/5 dominate (the `std::map` loop + the six `.cpu()` syncs).
- Reflection trace: compare `...op_st.ratio` for `max_bounces > 1` before/after the
  bounce-major internal layout to verify the expected store coalescing and transpose cost.
- Edge: `launch__occupancy_limit_registers` being the limiter confirms the 16-payload cost
  of item 12.

---

## Suggested execution order

1. **Nsight validate landed changes**: PTX fast-math/arch, bounce-major reflection
   trace (`max_bounces > 1`), and warp-aggregated accumulation atomics.
2. **Build path**: GPU-ize edge topology (item 4) and consider AS compaction (item 7)
   only after measuring build/traversal trade-offs.
3. **Edge query**: split the top-k pipeline (item 12) and collapse or compact the tiered
   launches (item 11).
4. **Reflection remaining work**: pack triangle data or otherwise reduce the 12 scattered
   hit-gather loads (item 16), then revisit split-scene double traces (item 17).
5. **Accumulation next level**: if Nsight still shows atomic pressure after warp
   aggregation, evaluate block-level or sort/segmented reductions for items 19/26.

Accuracy guard: items 3, 19, 25, 26, 28 (fast-math, reassociated/atomic reductions) change
the floating-point contract. Keep the native AD and opt-in RayD parity tests green after each.
