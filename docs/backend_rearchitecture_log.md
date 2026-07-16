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
