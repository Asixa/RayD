# Multi-backend rearchitecture — execution log

Execution log for `docs/RAY_TRACING_BACKEND_ARCHITECTURE.md` (P0 → P6), maintained per phase.
Branch: `wt/ray-tracing-backend`. Desktop verification hardware: RTX 5080 (sm_120), Windows 11, conda `witwin3`.

## Code-audit deltas vs the plan document

Findings from the pre-P0 recon sweep that correct or sharpen claims in the plan document.
The plan remains authoritative for goals and gates; the facts below are authoritative for code reality.

1. **`optixTrace` call sites: 7, not 1.** The plan (§2.2, §6.1) treats `trace_handle()`
   (`reflection_trace_device.cuh:88`) as the single choke point. Reality: 7 `optixTrace` calls across
   5 files, and `trace_handle` is an independently defined file-local static in 3 of them
   (`reflection_trace_device.cuh:98`, `reflection_accumulation_device.cuh:132`,
   `diffraction_accumulation_device.cuh:151` and `:194`, `reflection_epc_device.cuh:148` and `:218`,
   `segment_visibility_device.cuh:129`). P4's traverser refactor must migrate 7 sites / 5 files.
2. **`DISABLE_ANYHIT` is not universal.** Closest-hit traces always disable anyhit, but the two
   ignore-list occlusion paths (`segment_visibility_device.cuh:123-127`, `reflection_epc_device.cuh:209-216`)
   run with anyhit **enabled** when an ignore list is present (anyhit program filters ignored prims).
   The P3/P4 CUDA traversal needs an ignore-aware occlusion mode, not just a boolean occluded query.
3. **Edge top-k tie-break is asymmetric across backends.** `(distance_squared, global_edge_id)` is
   implemented only in the shared CUDA BVH query (`shared/src/edge/bvh_query.cu:168-174`) and the Torch
   brute-force top-k (`edge_topk.cu:57-64`). Every Dr.Jit edge path (OptiX anyhit/topk, native BVH) uses
   strict distance-only comparison — equal-distance winners are traversal-order-dependent. P0 freezes
   this asymmetry as a recorded deviation (`operations.json` → `edge_topk_tie_break`); unification is a
   behavior change deferred to ≥ P3.
4. **Miss-distance sentinel is not uniformly `+inf`.** Primary intersect, segment visibility, and EPC
   use `+inf`; the reflection-trace family uses `kTraceTMax = 1e8f` (`reflection_trace_device.cuh:94,140`).
   Frozen as-is (`miss_sentinels.reflection_trace_distance`).
5. **No JSON-Schema validation machinery existed.** `public_api.schema.json` was loaded only to compare
   its `required` list against a hardcoded set. P0 adds a minimal dependency-free validator
   (`tests/_schema_validate.py`) so "passes schema validation" is a real gate.
6. **P0 item 6 (fix CLAUDE.md) was already done** by commit `1c8f9a2`: the tracked agent doc is
   `AGENTS.md` and already describes the dual-backend workspace. No action needed.
7. **Hash-pin EOL landmines (pre-existing, environment-dependent test failures).**
   `_capabilities.py::_SCHEMA_SHA256` was pinned over LF bytes (fails on CRLF checkouts);
   `backends/torch/abi_audit.json::source_sha256` was pinned over a *mixed-EOL* working tree
   (7 files CRLF + `src/stable/camera.cu` LF) and was unreproducible from **any** clean checkout.
   P0 makes both hashes EOL-normalized (`\r\n` → `\n` before hashing) so the full suite can be green on
   every checkout. This is test-infrastructure robustness, not a semantic change.
8. **Torch scene-level `shadow_test` does not exist**; the Torch `ray_tmin` (1e-6) vs Dr.Jit (1e-3)
   divergence was already recorded in `operations.json` `backend_overrides` and is now also frozen in
   `NumericPolicy` legacy profiles.

## P0 — freeze public semantics

### Design decisions (supervisor)

- **Freeze, don't unify.** P0's zero-behavior-change gate forbids changing either backend's `ray_tmin`.
  `NumericPolicy` (doc §7 shape: `ray_tmin`, `shadow_tmin`, `endpoint_offset`, `parallel_epsilon`,
  `watertight_triangles`) is defined with `kDrJitLegacyProfile` / `kTorchLegacyProfile` recording today's
  values; the *unified* profile is deliberately not chosen until a phase where behavior change is allowed
  and measurable (≥ P3).
- **Compile-time anti-drift locks.** The scattered per-family constants inside device headers
  (`kTraceTMin`, `RayBias`, `kEpcTolerance`, …) get `static_assert`s tying them to
  `shared/rt/numeric_policy.h`, so the frozen contract and the kernels can never silently diverge.
- **SoA batch views.** `RayBatchView` / `SegmentBatchView` are SoA pointer views (doc §7 sketches AoS
  `Float3*`). Repo precedent (`ReflectionTraceParams`, `RaySoAView`, `EdgeQuerySoAView`) and the Dr.Jit
  frontend are SoA-native; Torch `(N,3)` tensors can view as SoA via stride tricks at the dispatch layer.
- **Golden scenes: strict vs informative fields.** Rays exactly through a shared edge/vertex and
  equal-distance edge ties have implementation-defined winners (OptiX traversal order; Dr.Jit edge paths
  have no id tie-break). These are recorded in baselines but tagged `informative` and excluded from the
  strict gate. Consequence for P3/P5: the "discrete results bit-identical across backends" gate applies
  to the strict field set; exact-tie cases cannot be part of it until tie-breaks are unified.
- **Instancing golden case: N/A.** RayD's public API has no instancing; recorded in the golden coverage
  manifest with rationale instead of a scene.

### Baseline (pre-P0, worktree @ `main`)

- drjit GPU suite: **158/158 green** (`test_baseline` must run with `cwd=backends/drjit`:
  `python -m unittest tests.drjit.test_baseline`).
- Workspace suite: 166 tests, 163 green, 3 skips (packaging CI-provided), 2 environmental failures
  (item 7 above) — both eliminated by the P0 EOL-normalization fix.
- Worktree is LF-materialized (`extensions.worktreeConfig` + per-worktree `core.autocrlf=false`);
  repo objects store LF.

### Implementation notes

- `operations.json` v3 → v4: `numeric_policy` (shared multipath constants + per-backend legacy
  profiles), `miss_sentinels`, `edge_topk_tie_break`, per-operation `integration` arrays,
  `result_contracts.raw_hit`.
- `public_api.json` v1 → v2: additive `trace` axis (backends/integration_modes/frontend_support);
  schema extended; `_capabilities.py` × 2 mirror it; sha pins now EOL-normalized; Torch ABI audit
  hashing EOL-normalized and `abi_audit.json` regenerated (reproducible from any clean checkout).
- Golden scenes: 13 scenes, `tests/golden/` (defs + baselines + coverage manifest) +
  `backends/drjit/tests/drjit/test_golden_scenes.py` (double-collect bit-identical determinism +
  baseline compare). Frozen surprises: ray exactly through a shared **vertex** misses entirely;
  ray through the shared-edge midpoint hits prim 0 (informative); `geo_n` is not oriented toward
  the ray (front and back hits both report +z); 0-size batches raise `RuntimeError` (jit size
  mismatch — only the exception type is frozen); drjit edge top-k tie order `[3,0,5,4]` confirms
  the missing id tie-break.
- Newly spotted, not yet frozen: Torch-only `kDfrRayBias = 1e-4f`
  (`backends/torch/include/rayd/torch/common/math.cuh:15`) — a diffraction ray bias with no
  audited Dr.Jit counterpart. Follow-up: inventory both diffraction paths and add to
  `numeric_policy` in a later phase.

### Incidents

- One implementation agent applied its edits to the **main checkout** instead of the worktree.
  Detected via its report paths + CRLF observations; changes were ported into the worktree with LF
  normalization (hash-verified), tests re-run in the worktree (23/23), and the main checkout was
  restored (`git checkout --` + stray-file removal), preserving the user's pre-existing local state.

### Status

- [x] Recon + baseline established
- [x] NumericPolicy + rt/ contract headers + operations.json v4
- [x] Capability manifest `trace` section + schema validator + EOL-normalized pins
- [x] Golden scenes + OptiX baselines + determinism gate
- [ ] Supervisor diff audit (done for native headers/contracts), native rebuild, full-suite gate, commit

## P1 — OptiX adopted as a formal backend + `Scene::build()` decoupled

### Design decisions

- **`TraceBackend` is host-lifecycle only.** The abstract base
  (`backends/drjit/include/rayd/trace/trace_backend.h`) carries just `kind()`,
  `capabilities()`, `is_ready()` — three per-*batch* host virtuals. The POD batch
  trace methods from doc §5 are deliberately **not** added yet (they are the
  eager-axis P3 concern); adding them now would risk a virtual landing in a
  per-ray loop. Per-ray work still reaches the concrete `OptixScene` through one
  extra non-virtual pointer hop per batch (`Scene` → `OptixTraceBackend` →
  `OptixScene`), never a virtual call. Verified by inspecting the call chain:
  `Scene::intersect` → `optix_scene()` (non-virtual) → `optix_backend().primary()`
  (non-virtual inline) → `OptixScene::intersect<Detached>()` (non-virtual
  template); the only per-batch virtual is `Scene::is_ready()`'s
  `trace_backend_->is_ready()`.
- **Split static/dynamic scene logic sunk into `OptixTraceBackend`.** The three
  `OptixScene` unique_ptrs, `split_active`, and the static/dynamic mesh-index
  bookkeeping moved out of `Scene` into `OptixTraceBackend`
  (`src/trace/optix_trace_backend.{h,cpp}`), along with `should_split_optix_scene`
  / `active_optix_split_mode`. The build/sync bodies are a **verbatim** transplant
  of the old `scene.cpp` blocks, so results stay bit-identical (confirmed by the
  golden + baseline gates below).
- **`Scene::build()`'s unconditional OptiX GAS build is gone.** A triangle trace
  backend is constructed only when the resolved plan asks for one
  (`triangle_kind_ == Optix`); `trace_backend='none'` (or auto-resolved None on an
  OptiX-less machine) leaves `trace_backend_` null and builds only the edge BVH.
  `build()` validates the plan first: `trace_backend='optix'` without OptiX raises
  "OptiX driver library is unavailable"; an OptiX edge backend without OptiX raises
  naming `edge_bvh_backend="drjit"` as the software alternative.
- **Clean, non-throwing OptiX probe.** `optix_available()` (`optix.cpp`/`optix.h`)
  is `noexcept`, never calls `jit_optix_context()`, and caches its result per
  process. `Scene(trace_backend="auto")` resolves to Optix when available and to
  None otherwise — capability discovery, not exception catching. It also does not
  eagerly initialize the OptiX context, so `set_device(initialize_optix=...)`
  semantics are unchanged.

### RAYD_DISABLE_OPTIX kill-switch

`optix_available()` returns false immediately when the environment variable
`RAYD_DISABLE_OPTIX` is `1`/`true` (case-insensitive), before touching any driver
library. This lets an OptiX-capable desktop exercise the OptiX-less paths (the P1
key gate). It is honored process-wide and cached on first call, so set it before
constructing any Scene.

### Gate results (RTX 5080, sm_120, Windows 11, conda `witwin3`)

- **Key gate (OptiX blocked):** with `RAYD_DISABLE_OPTIX=1`,
  `rd.optix_available()` is False, `Scene(edge_bvh_backend="drjit")` builds and
  answers `nearest_edge`/`nearest_edges` with discrete fields **bit-identical** to
  `tests/golden/baselines/optix/edge_queries.json`; `capabilities()` reports
  `trace_backend="none"`, `optix_available=False`; `intersect` raises a clear
  "requires a triangle trace backend" error; `Scene(trace_backend="optix")` and
  the default `Scene()` raise the expected build-time errors. New test:
  `backends/drjit/tests/drjit/test_trace_backend_gate.py` (8 cases, all green).
- **OptiX-available regression:** all green — `test_geometry`,
  `test_golden_scenes`, `test_visibility_topk`, `test_reflection_epc`,
  `test_reflection_accumulation`, `test_diffraction_accumulation`, `test_surfel`,
  `test_optix_pipeline_cold_create` (158 tests), `tests.drjit.test_baseline`
  (bit-identical, 2), workspace suite `tests/` (195 tests, 3 pre-existing CI
  skips), and the new gate (8). No baseline drift.
- **Perf (< 3% gate):** controlled A/B, both builds freshly compiled and measured
  back-to-back on the same idle GPU (`p1_perf_probe.py`, 262144 rays, median of 12
  process-medians):

  | op | pre-P1 median (ms) | P1 median (ms) | Δ median |
  | --- | ---: | ---: | ---: |
  | intersect | 0.1630 | 0.1580 | **−3.0%** |
  | shadow_test | 0.1625 | 0.1505 | **−7.4%** |
  | visible | 0.0139 | 0.0135 | **−2.9%** |

  P1 is at parity or slightly faster — comfortably inside the < 3% gate. Note the
  pre-P1 median (0.1630) is itself ~10% above the P0-recorded single-run baseline
  (0.1486): sub-ms kernels are boost-clock/scheduling noise dominated, and that
  drift moves both builds equally, so the controlled A/B — not the raw vs-P0
  numbers — is the valid comparison. Naive back-to-back P1-only runs transiently
  showed +15–45% purely from that drift; the A/B removes it.

### Deviations from the P1 spec

1. **Windows OptiX probe needed the driver-store lookup.** The spec's Windows
   recipe (`GetModuleHandleW` else `LoadLibraryW("nvoptix.dll")`) returns null on
   this machine: `nvoptix.dll` lives in the NVIDIA driver store
   (`C:\WINDOWS\System32\DriverStore\FileRepository\nvmdi.inf_...\nvoptix.dll`),
   not on the DLL search path, so a bare load fails with ERROR_MOD_NOT_FOUND (126)
   and the probe wrongly reported OptiX unavailable — which broke every default
   scene build. Fixed by extending the Windows path with the OptiX SDK's
   `optixLoadWindowsDll` algorithm: enumerate display adapters via cfgmgr32, read
   each device's `OpenGLDriverName`, and load `nvoptix.dll` from that directory.
   Added `cfgmgr32`/`advapi32` to the Windows link libs in `CMakeLists.txt`. The
   Linux/Jetson path uses the spec's `dlopen("libnvoptix.so.1", ...)` recipe
   unchanged (the actual P2 target).
2. **rt/ host-safety test for `backend.h`.** `TraceBackendKind::Optix` lowercases
   to the existing FORBIDDEN token "optix", so `backend.h` cannot be added to the
   shared `test_headers_are_host_safe` list as-is. A dedicated
   `test_backend_header_is_host_safe` checks the real host-safety tokens
   (`__device__`/`__host__`/`float3`/`cuda_runtime`) plus a guard against
   `#include`ing any optix/cuda/drjit header, while allowing the enum-enumerator
   identifier. `test_trace_capabilities_struct_field_order` freezes the field
   order.
3. **`OptixSceneSelection` promoted to namespace scope.** It was a private nested
   `Scene::OptixSceneSelection`; it now lives at `rayd::OptixSceneSelection` in
   `optix_trace_backend.h` so the backend can return it. The ~13 call sites and
   the `test_project_metadata` source anchor
   (`const OptixSceneSelection scenes = select_optix_scenes();`) are textually
   unchanged; `Scene::select_optix_scenes()` remains a private method that
   delegates to the backend.

## P3 Stage A — extract the generic BVH core into shared/bvh/

Behavior-preserving extraction of the primitive-agnostic BVH machinery out of the
edge BVH (plan doc §15 P3 deliverable 1, §8.1). Zero behavior change: identical
kernels, identical launch sequences, identical results.

### What moved where

- **`shared/include/rayd/shared/bvh/topology.h`** (new, namespace `rayd::shared::bvh`):
  the generic `BvhFloat3`/`BvhBounds3`, `AabbSoAView`/`MutableAabbSoAView`, the raw
  and compact topology views (with the literal `left_child[node] = -leaf_begin - 1`
  leaf-encoding contract), `DeviceScratchView`, all eight treelet/leaf/stack/top-k
  constants, and POD asserts (`RAYD_SHARED_BVH_ASSERT_POD`). This is now the single
  home of these definitions.
- **`shared/include/rayd/shared/bvh/build.h`** and **`refit.h`** (new): the seven
  generic build param structs + launcher decls, and the three refit param structs +
  launcher decls, all parameterized on `AabbSoAView`/topology rather than `EdgeSoAView`.
- **`shared/src/bvh/build.cu`** (new, namespace `rayd::shared::bvh`): the ten generic
  kernels (Morton, Karras radix tree, leaf/bounds finalize, SAH leaf/internal costs,
  treelet DP optimizer, dirty-ancestor mark, dirty-level compact, internal-node refit)
  and their ten launchers. The device code is byte-for-byte identical to the previous
  `shared/src/edge/bvh_build.cu` (verified by per-function diff against `HEAD`).
- **`shared/include/rayd/shared/bvh/traversal_common.cuh`** (new): the depth-major
  `stack_push`/`stack_load` helpers (templated on the scratch view) and the near/far
  tie-break `near_child_is_left`, shared for any BVH consumer.
- **`shared/include/rayd/shared/bvh/host_topology.h`** (new, header-only, pure C++, no
  Dr.Jit/Torch types): the primitive-agnostic host algorithms `compute_subtree_leaf_count`,
  `compute_subtree_primitive_count`, `compute_node_height`, `collect_subtree_primitives`,
  and the `HostCompactedBvh<Vec3>` / `emit_compacted_preorder<Vec3>` compaction, templated
  on the caller's scalar vector type so bounds are copied byte-identically.

### Edge layer keeps only edge-specific parts and delegates

- `shared/edge/bvh_types.h` now includes `bvh/topology.h` and re-exports every generic
  name via `using bvh::...` (types + constants), keeping `struct EdgeSoAView` as the only
  edge-specific definition. All `rayd::shared::edge::` names still resolve unchanged.
- `shared/edge/bvh_build.h` keeps the edge `PrimitiveBoundsParams` (+ launcher) and the
  `BvhBuildParams`/`BvhRefitSelection`/`BvhRefitParams` contract structs (they carry
  `EdgeSoAView`), re-exports the ten generic param structs, and declares thin edge
  forwarding launchers.
- `shared/edge/bvh_build.cu` keeps only the edge `compute_primitive_bounds` kernel and
  defines ten thin forwarders (`shared::edge::launch_* -> shared::bvh::launch_*`). The
  generic kernels exist exactly once, in the core.
- `shared/edge/bvh_query.cu` now includes `bvh/traversal_common.cuh` and calls
  `bvh::stack_push`/`bvh::stack_load`/`bvh::near_child_is_left` (local copies deleted);
  the query kernels are otherwise unchanged.
- `backends/drjit/src/edge/scene_edge.cpp` deletes its local copies of the four subtree/
  height utilities and the compaction/`CompactedEdgeBVH`, and now calls the shared
  `bvh::` host algorithms (`CompactedEdgeBVH = bvh::HostCompactedBvh<ScalarVector3f>`). The
  dead host treelet optimizer and dead `build_preorder_mapping` were removed.

### Deviations / scope decisions

- **Torch left alone (as instructed).** Torch's `scene_cache.cpp::build_treelet_schedule`
  is a structurally different iterative schedule (not byte-equivalent to the drjit host
  optimizer), so it was not migrated. Torch native code was otherwise untouched; only its
  CMakeLists source list gained `shared/src/bvh/build.cu` (and `backends/torch/abi_audit.json`
  was regenerated because the audit hashes `CMakeLists.txt` — only its `source_sha256`
  changed).
- **Dead host treelet optimizer not extracted.** `rebuild_treelet_branch` /
  `optimize_treelet_at_node` / `optimize_treelets_recursive` (and `build_preorder_mapping`)
  had no live caller in `scene_edge.cpp` (GPU treelet path is used) and depend on Dr.Jit
  `ScalarVector3f` + `require()`; they were removed rather than migrated. The device treelet
  DP optimizer is fully shared in `bvh/build.cu`.
- **`edge_bvh.obj` DEPENDS fix.** Aliasing the generic types under `bvh::` changes the
  mangled names of the edge forwarders' signatures, so the `edge_bvh.cu` custom-command
  object must recompile in lockstep. Its DEPENDS list gained the moved headers (both build
  branches) so incremental/CI builds stay correct.

### Build system

`shared/src/bvh/build.cu` is compiled as a new object in both backends
(`RAYD_SHARED_BVH_CORE_BUILD_OBJECT` in `backends/drjit/CMakeLists.txt`, and a new source
entry in `backends/torch/CMakeLists.txt`), next to the existing `shared/src/edge/*` units.

### Lockstep structure-test updates (equivalent-or-stronger pins on the new locations)

- `tests/test_share5_edge_bvh_core.py`: topology struct defs / leaf-encoding comment /
  treelet constants pinned in `bvh/topology.h` (plus edge re-export `using` checks); the
  no-resources / `params.stream` and forbidden-strategy scans extended to `bvh/build.cu`
  and the bvh core headers; the build-stage test now also asserts the generic launchers
  live once in `bvh/build.cu` and the edge unit forwards to `bvh::`; the depth-major stack
  indexing pin moved to `traversal_common.cuh` with a check that the query includes/uses it;
  both-backends-compile check extended to `shared/src/bvh/build.cu`.
- `tests/test_bvh4_shared_edge_core.py`: raw-pointer/caller-owned and enqueue-only contracts
  extended to cover the bvh core headers and `bvh/build.cu`.
- `tests/test_bvh1_removal.py`: the `finalize_leaves_and_bounds_kernel` atomic
  publish-before-arrival invariant now pinned in `shared/src/bvh/build.cu`.

### Verification (RTX 5080, sm_120, conda `witwin3`)

- Device code byte-identical: per-function diff of every moved helper/kernel/launcher against
  `HEAD` — 0 mismatches; the retained edge `compute_primitive_bounds` kernel/launcher identical.
- Full suites green: `test_share5`/`test_bvh4`/`test_bvh1..3`/`test_edge_bvh_benchmark_*`,
  drjit `test_geometry`/`test_golden_scenes`/`test_visibility_topk`/`test_trace_backend_gate`
  (85 tests), drjit `tests.drjit.test_baseline`, and `unittest discover -s tests` (195 tests,
  3 pre-existing skips) — all OK.
- **Edge BVH gate — launch counts identical.** Pre vs post smoke: 330/330 launch-audit metric
  comparisons `increase = 0`; 0 per-case launch-count mismatches across 11 cases (e.g. build
  = 1921 launches pre and post). No launch-count regressions.
- **Timing thresholds are noise-limited on the smoke profile, not a real regression.** The
  pre-vs-post gate flags 8 timing failures (`build_ms`/`hot_query_ms`/`refit_ms`), but gating
  the *byte-identical* post binary against itself (post-vs-postB and postB-vs-postC) flags
  6-9 of the same timing metrics per run, with equal or larger deltas (e.g. build 132→162 ms
  = +23 %, refit 1.09→1.61 ms = +48 %). `refit_ms` regresses on refit code this change does not
  touch at all. Conclusion: the timing deltas are run-to-run measurement variance on this
  desktop at the 3 %/5 % smoke thresholds; the deterministic launch-count invariant (the real
  regression check) is identical, and the behavior suites confirm bit-identical edge-query
  results.

## P3 Stage B+C — the pure-CUDA triangle TraceBackend

A second concrete `TraceBackend` (`trace_backend='cuda'`) that answers closest-hit
and occlusion queries with raw CUDA kernels over a scene-level triangle BVH, with
no OptiX driver dependency (plan doc §5 eager-native axis, §8.2). Multipath stays
OptiX-only until the CUDA fused executor (P4).

### Design (locked in the supervisor spec)

- **Single scene-level BVH over world-space triangles, no BLAS/TLAS.** Source is
  `Scene::triangle_info_detached_.{p0,e1,e2}` (SoA, world space, transforms already
  baked, indexed by global primitive id, refreshed by `sync()`). Unlike OptiX we do
  not need an IAS because the detached triangles are already in world space.
- **Watertight intersector** (`shared/include/rayd/shared/bvh/triangle_intersect.h`,
  host/device dual, pure): Woop-Benthin-Wald 2013 ray-centric shear + 2D edge
  functions, no backface culling, boundary-inclusive. The returned `(u, v)` match
  the Möller-Trumbore convention of `utils.h ray_intersect_triangle`
  (`P = p0 + u·e1 + v·e2`), verified against it. The edge functions use an
  FMA-contraction-proof difference of products (Kahan) so a mathematically zero
  edge function is exactly `0.0f` under any `--fmad` setting — without this a
  diagonal-crossing ray gets a tiny-residual edge function of arbitrary sign and is
  spuriously rejected (this was the initial watertight-grid failure: 8/1600 gaps,
  all on the shared diagonal).
- **Three separate traversal kernels** (`shared/src/bvh/triangle_query.cu`,
  allocation-free, stream-parametered, POD params): `closest_hit` (near/far slab
  ordering, `(t, global_prim_id)` tie-break), `occluded` (early-exit any-hit), and
  `first_blocker` (closest blocker + per-ray ignore list). Each uses a 64-deep
  depth-major caller-owned stack (`traversal_common.cuh` `stack_push`/`stack_load`)
  plus a per-ray overflow flag; a build-time guard (`max_height + 1 ≤ 64`) makes
  overflow structurally impossible, and brute-force repair kernels are wired as the
  fallback the overflow flag triggers.
- **`CudaTraceBackend`** (`backends/drjit/{include,src}/rayd/trace/cuda_trace_backend.*`
  + `src/trace/triangle_bvh.cu`): clones the edge-BVH integration shape — persistent
  Dr.Jit member buffers, `dr.eval` + `sync_thread` before exposing `.data()`, the
  `.cu` orchestrator owns its own non-blocking streams + `CudaBuffer` scratch and
  drives the shared `bvh/build.h` launchers, host compaction via `host_topology.h`,
  and refit level-by-level in ascending height. Build is pure LBVH (see Deviations).
  Every op records `audit_*` hooks under a new `NativeLaunchStage::Intersect`.
- **Seam**: `CudaTraceBackend::intersect<Detached>`/`shadow_test<Detached>` return the
  same detached `OptixIntersection`-shaped result as `OptixScene`, so
  `scene_intersect.cpp:72+` (winner derivation + AD recompute) is reused verbatim.
  A `triangle_kind_ == Cuda` arm gates the broad phase in `scene_intersect.cpp` and
  `scene_multipath.cpp shadow_test`, both guarded against Dr.Jit symbolic recording
  (`jit_flag(Recording)` ⇒ clear error). `optix_backend()` now kind-checks and
  `optix_split_active()` is guarded so a CUDA scene never reinterprets the backend
  pointer; multipath entry points raise "requires the OptiX trace backend; CUDA
  multipath arrives with the CudaFusedExecutor (P4)".
- **Numerics** match OptiX exactly: `tmin = RayEpsilon` (1e-3), `tmax = isfinite ?
  tmax : 1e8`, miss ⇒ `t = +inf`, ids `= -1`.
- **Python/contract**: `resolve_trace_backend_kind` maps `"cuda"` to the backend;
  `capabilities()` reports `trace_backend="cuda"`, `integration=["eager_native"]`,
  `intersect`/`shadow_test` true and multipath false; `public_api.json` trace section
  gains the `cuda` backend + `frontend_support.drjit.cuda=["eager_native"]`, with
  both `_capabilities.py` copies, the EOL-normalized SHA pin, and the manifest test
  updated in lockstep.

### Gate results (RTX 5080, sm_120, CUDA 12.9, conda `witwin3`)

All green:

- **`test_cuda_trace_backend`** (new, 10 tests): golden cross-backend parity
  (discrete bit-identical vs `baselines/optix`, continuous within `operations.json`
  tolerances) across every CUDA-servable query in `single_tri`, `shared_edge_quad`,
  `degenerate_tri`, `large_coordinates`, `self_intersection`, `multi_mesh_ids`,
  `dynamic_refit`, `inactive_lanes`, `batch_sizes`, `finite_tmax_visibility`
  (shadow), plus `edge_queries`/`edge_tie`; watertight 40×40 grid over the
  shared-edge quad = 1600/1600 hits with 40 exact-diagonal rays, all occluded, zero
  gaps; exact-diagonal `t ≈ 1.0`; degenerate zero-area triangle never hit and no
  NaN; 1e6-offset scene hits; AD vertex + transform gradients identical to the OptiX
  backend; dynamic vertex update refits (hit → miss → hit at the moved location);
  launch audit shows exactly 1 `triangle_closest_hit_kernel` per query and a stable
  50 launches over 50 queries (no per-call cudaMalloc-driven kernels / no overflow
  repair); recording guard raises; first-blocker self-test returns prim 0, then
  prim 1 with `ignore=[0]`, then -1 with `ignore=[0,1]`.
- **OptiX regression (default path untouched, bit-identical):** `test_golden_scenes`
  (2), `test_geometry` (63), `test_trace_backend_gate` (9, now incl. the CUDA-
  constructs case), `test_visibility_topk` (12), `test_reflection_epc` (12),
  `test_reflection_accumulation` (6), `test_diffraction_accumulation` (30),
  `test_surfel` (32), `test_optix_pipeline_cold_create` (1), `tests.drjit.test_baseline`
  (2), and workspace `unittest discover -s tests` (195, 3 pre-existing skips,
  including the updated `test_public_api_manifest`).

### Perf snapshot (informational, not a gate)

RTX 5080, 18,432 triangles, 262,144 rays, median ms:

| op | OptiX | CUDA |
| --- | ---: | ---: |
| `intersect` | `0.18 ms` | `2.21 ms` |
| `shadow_test` | `0.15 ms` | `1.43 ms` |

CUDA is ~9-12× slower than OptiX, as expected: it is the driver-independent Orin
fallback path (pure LBVH, no RT cores).

### Deviations from the spec

- **Pure LBVH build, no treelet pass.** The build reuses the shared Morton/radix/
  finalize launchers but omits the GPU treelet optimizer that the edge BVH runs at
  ≥65 k primitives. Rationale: treelet is a query-throughput optimization that is
  correctness-neutral (it never changes primitive membership), never triggers on the
  golden scenes, and only affects the informational perf number. This keeps the
  single-pass build tractable and low-risk; the treelet pass can be lifted from the
  shared launchers later behind the same size gate.
- **`shared/src/bvh/triangle_query.cu` is registered in both backends' CMakeLists**
  (and `backends/torch/abi_audit.json` regenerated) to keep the shared source set
  symmetric, even though the Torch backend does not consume the triangle trace
  kernels until P4.

## P4 Stage A — Traverser infrastructure + reflection-trace pipeline migration

The pattern-setter for every later multipath migration: a backend-neutral
Traverser concept, the extraction of the P3 triangle traversal cores into a
reusable device header, and the reflection-trace pipeline lifted end-to-end into a
host-compilable, traverser-templated algorithm body. Reflection is the smallest
multipath pipeline (one ray-cast site, one payload shape), so it establishes the
shape the rest of P4 follows.

### What moved

- **`shared/rt/qualifiers.h`** (new): `RAYD_DEVICE` / `RAYD_HOST_DEVICE` behind the
  `__CUDACC__` toggle (`inline` under a host compiler). `shared/math/vec3.h` now
  spells its inline qualifier through `RAYD_HOST_DEVICE` — behavior-identical to the
  former local `RAYD_SHARED_MATH_INLINE`.
- **`shared/rt/traverser.h`** (new): `struct TriangleHit` (decoded mirror of the
  OptiX `TriangleHitPayload`), the C++17 `is_traverser` traits + `static_assert`
  concept (`trace_closest`, `trace_occluded`, `trace_occluded_ignore`,
  `trace_first_blocker`), and `TraceConfig<Layout, Traverser>` merging the two axes
  (audit A3). Instantiation matrix is documented in-header: DrJit × {Optix, CudaBvh},
  Torch × {Optix} only. Host-safe (no device qualifier / SDK-include tokens; checked
  by `test_rt_contract_headers`).
- **`shared/bvh/triangle_query_device.cuh`** (new): the P3 triangle traversal cores
  (`traverse_closest`, `traverse_any_hit`, `traverse_first_blocker`, brute-force
  repair, `safe_rcp`, `intersect_node_bounds`, …) extracted verbatim from
  `shared/src/bvh/triangle_query.cu`. The kernels become thin `__global__` wrappers
  calling the cores — behavior and the one-launch-per-query contract unchanged.
- **`shared/bvh/cuda_bvh_traverser.h`** (new): `CudaBvhTraverser` over a `CudaBvhView`,
  implementing the Traverser concept on those cores. Compiled and concept-checked via
  `triangle_query.cu` (both backends); not yet wired to a pipeline (that is the P4d
  `CudaFusedExecutor`).
- **`shared/optix/optix_traverser.h`** (new): `OptixTraverser`, the sole home of
  `optixTrace` + the six-register payload codec, decoding to `TriangleHit`. Holds ONE
  traversable handle; the dual-handle "choose nearest" logic stays in the algorithm
  (it is pipeline semantics, not traversal), which owns a primary and a secondary
  traverser.
- **`shared/multipath/reflection_trace_algo.h`** (new): the de-CUDA-ised algorithm.
  `math::Vec3f` throughout (mirroring the exact op order of the old local `float3`
  helpers so device codegen stays bit-identical), all ray casts routed through the
  Traverser, the lane index a plain `uint32_t` parameter, templated over `TraceConfig`.
  The P0 numeric-policy `static_assert` locks moved here with the constants. Contains
  no `optixTrace` / `optixGet/SetPayload` / `optixGetLaunchIndex` / `float3` token
  (grep-gated) and compiles under a pure host C++ compiler.
- **`shared/optix/reflection_trace_device.cuh`** (refactored): keeps only the OptiX
  layer — raygen/closesthit/miss entries, `ReflectionTracePolicy` (now the Layout axis)
  with the DrJit/Torch aliases, the `OptixTraverser` instantiation, and params
  adaptation. Produces bit-identical behavior.

### Risk protocols

- **A (FP bit-identity):** the observable reflection output is bit-identical across the
  refactor — the two-bounce fingerprint (`t = [1.4142135381698608, 0.7070969939231873,
  inf]`, `hit = [1.0, 0.0, 1.4999998807907104] / [0.4999997615814209, 0.0, 2.0]`,
  image sources `[2,0,0.5] / [2,0,3.5]`) matches to the last ULP before and after.
  `test_baseline` (continuous within tolerance, discrete exact), the golden suite, and
  a `test_golden_scenes` determinism double-run are all green.
- **B (PTX regen):** the committed DrJit reflection-trace PTX was regenerated via
  `-DRAYD_REGENERATE_REFLECTION_TRACE_PTX=ON` (passed through scikit-build with
  `SKBUILD_CMAKE_DEFINE`, OptiX SDK 9.1.0). `.version 8.8` / `.target sm_70` are
  unchanged (local nvcc 12.9.41 matches the committed header's compiler); no test pins
  PTX properties. The regenerated PTX differs only structurally — register renumbering,
  the two traversable-handle loads hoisted earlier (up-front traverser construction),
  and a `setp.ge`↔`setp.le` operand swap on the identical `ray_index >= n_rays` guard —
  with no change to the floating-point instruction stream. The multipath pipeline
  guardrail holds (`test_optix_pipeline_cold_create` green after regen).
- **C (Torch):** the full Torch build succeeds and `test_multipath` (32) is green with
  `TorchReflectionTracePolicy` on the same shared headers. The initial build attempt
  failed on `No space left on device` (system `C:` at 100%) in unrelated diffraction /
  edge / common `.cu` files — my two P4a surfaces (`reflection_trace_optix.ptx` from
  `trace_optix.cu`, and `triangle_query.cu.obj`) had already compiled cleanly.
  Redirecting the nvcc temp directory to a drive with free space completed the build.
- **D (perf, < 3% gate):** `trace_reflections` (4-bounce, 4096 rays) median ≈ 0.24 ms /
  best ≈ 0.19 ms, versus the pre-refactor baseline median 0.28 ms / best 0.22 ms — no
  regression (the numerically identical PTX runs the same math).

### Gate results (RTX 5080, sm_120, CUDA 12.9, conda `witwin3`)

All green: `test_geometry` (63), `test_golden_scenes` (2, plus determinism re-run),
`test_trace_backend_gate` (9), `test_reflection_accumulation` (6), `test_reflection_epc`
(12), `test_diffraction_accumulation` (30), `test_visibility_topk` (12), `test_surfel`
(32), `test_optix_pipeline_cold_create` (1, critical after PTX regen),
`tests.drjit.test_baseline` (2, bit-identical), workspace `unittest discover -s tests`
(200, 3 pre-existing/environmental skips), the new host-compile gate
(`test_rt_host_compile`: the algorithm header compiled host-only by `cl.exe` located via
`vswhere`, plus the token grep-gate), and Torch `test_multipath` (32). `test_cuda_trace_backend`
passes on non-racing runs (10 tests) — see the known issue below.

### Known issue (pre-existing, out of scope)

`test_cuda_trace_backend`'s full-collection setup is intermittently flaky under the
CUDA trace backend: the `batch_sizes` intersect-grid occasionally reports zero hits
(all rays miss), which surfaces as a `min() arg is an empty sequence` collection error.
An A/B rebuild confirms this predates P4a and lives in the unchanged P3 CUDA
orchestration (`cuda_trace_backend.cpp` / `triangle_bvh.cu`, a separate-stream eager
launch), not the traversal cores: HEAD `a06487c`'s triangle_query.cu failed 6/30 full
collections; the P4a extraction failed 9/30 (statistically comparable at a ~25% base
rate). The extraction is verbatim code motion and every non-racing run matches the
OptiX baselines bit-for-bit. Left as-is rather than patching P3 orchestration outside
this stage.
