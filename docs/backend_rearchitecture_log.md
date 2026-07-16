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
