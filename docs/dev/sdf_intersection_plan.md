# Differentiable SDF Intersection — Implementation Plan

Status: COMPLETED (Phases 0-4; Phase 5 remains a recorded backlog, DO NOT implement).
Branch: `wt/sdf-intersection`. Owner: RayD Torch backend.

## 1. Goal

Add a differentiable SDF (signed distance field) ray-intersection primitive to the
RayD **Torch backend**. The primitive sphere-traces caller-owned dense SDF grid
tensors placed in the world by an oriented bounding box (position / rotation /
scale), and provides analytic VJP/JVP so gradients flow to the grid values, the
bbox transform, and the rays.

## 2. Scope decisions (fixed — do not relitigate)

- **Representation**: caller-owned dense grid `values: float32 CUDA [Nx, Ny, Nz]`,
  vertex-centered samples, trilinear interpolation. Sign convention: **negative
  inside, positive outside** (matches `witwin.core`). Values are world-metric
  distances baked by the caller; RayD never bakes.
- **Bounding box**: the grid spans an oriented box defined by
  `position: float32 [3]` (world center), `rotation: float32 [4]` scalar-first
  unit quaternion (matches `witwin.core` convention), `scale: float32 [3]`
  (full side lengths of the box in world units). Local frame: box centered at
  origin, axis-aligned, spanning `[-scale/2, +scale/2]` per axis.
  World→local: `x_l = R(q)^T (x_w - position)`.
  Grid coordinate: `u_i = (x_l_i / scale_i + 0.5) * (N_i - 1)`.
- **Backend**: Torch only. Dr.Jit port is Phase 5. In `public_api.json` the
  capability is declared for both backends with `drjit: false, torch: true`.
- **No Mesh + SDF coexistence in v1**: the SDF primitive is standalone
  (`SdfGrid` object + functional entry), NOT integrated into `Scene`, no OptiX,
  no mixing with triangle geometry. Scene-level composition is Phase 5.
- **No OptiX**: pure CUDA kernels. New code must NOT include any header listed in
  `drjit/ptx_sources.json` include closures (would stale committed PTX).
- **Algorithm**: sphere tracing with relaxation
  `t_{k+1} = t_k + lambda * step_scale * d(x_k)`, where
  `step_scale = min(scale_i / (N_i - 1)) / value_scale_safety` is NOT the design —
  see below. The traced interval is clipped to the ray/OBB overlap
  `[t_enter, t_exit]` (slab test in local frame); ray misses if no overlap.
  Convergence: `|d| < eps_hit`; divergence: `t > t_exit` or `steps == max_steps`.
  The forward march is fully detached.
- **Conservative stepping**: interpolated grid values are only approximately
  eikonal (baking resolution, and stale after transform-only `scale` edits), so
  the step uses a relaxation factor `lambda` (default `0.9`, user-tunable) and
  the march must be robust to overshoot: if the sign of `d` flips between
  consecutive samples, fall back to bisection on that bracket for a bounded
  number of iterations to land `|d| < eps_hit`.
- **Differentiability contract (implicit function theorem at the frozen hit)**:
  the hit satisfies `d(x_l(o, w, t*, theta)) = 0` with the discrete winner
  (hit voxel neighborhood / final bracket) frozen. Derivatives:
  `dt*/dtheta = -(∂d/∂theta) / (∇_w d · w)` where `∇_w d` is the world-space SDF
  gradient obtained by chain rule through the bbox transform. Supported
  gradient/tangent inputs: `values`, `position`, `rotation`, `scale`,
  `origins`, `directions`. Grad wrt `values` scatters through the 8 trilinear
  weights of the frozen hit point (atomicAdd). Normal output
  `n = ∇_w d / |∇_w d|` is recomputed differentiably at the frozen hit.
- **Grazing-ray guard**: when `|∇_w d · w| < eps_graze` the IFT denominator is
  clamped (`sign(x) * max(|x|, eps_graze)`); this is part of the numeric
  contract and must be documented in `operations.json` and tested (no NaN/Inf
  may ever leave the op).
- **Miss semantics**: missed rays report `t = miss sentinel` consistent with
  `shared/contracts/operations.json` `miss_sentinels`, `hit_mask = false`, zero
  gradients (no atomics contributed) — mirror the existing capacity-shaped-row
  validity philosophy: invalid/missed lanes are bitwise inert.
- **Silhouette / visibility discontinuity gradients**: explicit NON-GOAL for v1
  (recorded in ADR + Phase 5).
- **Compile flags**: all new CUDA TUs use the `nvcc_default` profile. No new
  numeric flags, no profile moves (would require its own ADR per ADR-0035).
- **Dispatcher**: the new op is registered GIL-free from the start (pure C++
  `_impl` pattern like the intersect hot path in
  `src/bindings/library.cpp`), NOT the `py::tuple` +
  `gil_scoped_acquire` legacy pattern.
- **Typed integration boundary**: v1 does NOT touch
  `include/rayd/integration/torch.h`; ADR-0040 later advances the repository-wide boundary to API version 7.
  Typed same-graph exposure for Channel is Phase 5.

## 3. Public API (v1)

```python
from rayd.torch import SdfGrid, sdf_intersect

grid = SdfGrid(values, position=..., rotation=..., scale=...)  # thin tensor holder, validates inputs
res = sdf_intersect(grid, origins, directions,
                    tmax=..., max_steps=64, relaxation=0.9, eps_hit=None)
# res: SdfIntersection(t, hit_mask, position, normal, steps) — position/normal differentiable,
# lazy/consistent with existing result-type conventions in types.py
```

`eps_hit` defaults to a resolution-derived value (`0.5 * min voxel edge` scaled —
exact default decided in ADR). All tensors CUDA float32, batched rays `[N, 3]`.

## 4. Phases

### Phase 0 — ADR
`docs/adr/0037-differentiable-sdf-intersection.md` (follow the style of ADR-0031..0035):
representation, bbox transform math, AD contract incl. grazing clamp, miss
sentinels, sign convention, conservative stepping + bisection fallback,
non-goals, contract-file impact list, Phase 5 backlog. Plus a guard test
`tests/test_adr0037_sdf_intersection.py` asserting the documented constants /
contract entries exist and match code (pattern: existing `test_adr003*.py`).

### Phase 1 — Pure-PyTorch reference implementation (tests only)
`tests/sdf/_reference.py` (or similar under tests/):
detached march loop + differentiable last-step reattachment using
`torch.nn.functional.grid_sample` (mind align_corners=True to match
vertex-centered samples) or manual trilinear gather. Golden tests against
analytic sphere/box SDFs with closed-form `t*` (baked onto grids), gradcheck
(float64 variant allowed in reference), finite-difference checks for all six
gradient inputs, inside-start, miss, grazing clamp. This reference is the
oracle for the CUDA kernels; it must NOT ship in the product package.

### Phase 2 — Shared device math (new files only)
- `shared/include/rayd/shared/sdf/grid_sdf.cuh`: trilinear sample + analytic
  local gradient, `RAYD_HOST_DEVICE`, host-compilable (no CUDA-only intrinsics
  outside `__CUDA_ARCH__` guards), following `shared/math/vec3.h` style.
- `shared/include/rayd/shared/sdf/sphere_trace.h`: backend-neutral march loop
  body (templated on the sampler), plus OBB slab clip and quaternion
  world↔local helpers (reuse existing shared math where present).
- Host-compile contract coverage: extend the `tests/test_rt_host_compile.py` /
  `tests/native/` pattern with an SDF smoke TU; add header contract test in the
  root suite style (`tests/test_shared_headers.py` conventions).
- Do NOT modify any existing shared header.

### Phase 3a — Torch backend native (CUDA + C++ + registration + build)
- `src/sdf/kernels.h`: `SdfIntersectForwardOutputs`
  (t, hit_mask, steps, tape: frozen hit t / bracket data), launcher decls
  forward/backward/jvp.
- `src/sdf/sdf.cu` (forward, VJP, and JVP kernels) (VJP + JVP; JVP
  may live in backward.cu or its own TU).
- `src/sdf/sdf.cpp`: validation (`tensor_check.h`),
  op bodies, GIL-free `_impl` functions.
- `src/bindings/library.cpp`: schema `m.def` +
  `TORCH_LIBRARY_IMPL(rayd_torch, CUDA, ...)` impls (`sdf_intersect_forward`,
  `sdf_intersect_backward`, `sdf_intersect_jvp`; naming consistent with existing
  ops).
- `torch/CMakeLists.txt`: add TUs to `RAYD_TORCH_NATIVE_CORE_SOURCES`.
  No PTX step (no OptiX).
- Must build cleanly: `.\scripts\build_local.cmd -Backend torch` and the ops
  must be callable from `torch.ops.rayd_torch`.

### Phase 3b — Torch backend Python (autograd + types + entry + tests)
- `python/rayd/_impl/geometry.py`: `SdfIntersection`.
- `python/rayd/_impl/sdf.py`: `SdfGrid`,
  `sdf_intersect` dispatch (no-grad fast path / reverse-mode / forward-mode dual
  path), input validation with actionable messages.
- `python/rayd/_impl/multipath.py`: `_SdfIntersectFunction`
  (forward/setup_context/backward/jvp) following the `_make_intersect_function`
  conventions (named autograd function, optional-grad returns).
- `python/rayd/torch/__init__.py`: export.
- Tests `tests/sdf/test_intersect.py`: parity vs
  Phase 1 reference (forward t/normal within tolerance; gradients within FD
  tolerance), gradcheck on small grids, forward-mode jvp checks, miss/inside/
  grazing cases, non-contiguous & wrong-device/dtype rejection, determinism of
  forward, zero-grad inertness of missed lanes.

### Phase 4 — Contracts + sweep
- `shared/contracts/public_api.json`: capability key `sdf_intersect` + API
  entries + per-backend capability booleans (drjit false, torch true).
- `shared/contracts/operations.json`: `required_capability_keys` + full
  `operations.sdf_intersect` entry (inputs, result, ad, numeric policy incl.
  grazing clamp + miss sentinel).
- `tests/test_shared_operation_contract.py`: update the hard-coded operation-set
  literal.
- `python/rayd/_impl/capabilities_jit.py` AND
  `python/rayd/_impl/capabilities.py`: same-change update incl.
  pinned `_SCHEMA_SHA256` (ADR-0036: only the three allowed lines differ).
- `contracts/compile_policy.json`: new TUs under `nvcc_default`.
- Typing: the touched torch modules annotated inline (the Torch package ships no
  `.pyi` at all); `tests/test_public_api_manifest.py` green.
- `FEATURE_LIST.md` (repo root or backend-level, follow existing location).
- Full relevant sweep green: root contract tests (`tests/`), torch backend tests
  touched by this work, ADR-0037 guard test.

### Phase 5 — Backlog only (record in ADR; DO NOT implement)
- Dr.Jit backend port reusing the Phase 2 shared math.
- Scene-level Mesh + SDF coexistence (OptiX custom-AABB primitive per SDF bbox,
  shared closest-hit resolution with triangles).
- Silhouette/visibility discontinuity gradients (reparameterization).
- Typed C++ integration boundary exposure (`integration.h`) for Channel
  same-graph consumption.
- Analytic-SDF fast path (primitive params direct to kernel, no baking).
- Multi-grid batching / instancing.

## 5. Environment & verification

- Conda env: **witwin3** (`C:\Users\Asixa\miniconda3\envs\witwin3\python.exe`).
- Build: `.\scripts\build_local.cmd -Backend torch` from the worktree root
  (long-running; run in background and poll).
- GPU present (RTX 5080); CUDA tests must actually run, not skip.
- Follow karpathy-guidelines and python-code-standard skills for all code.
- Commit per phase on `wt/sdf-intersection`, English commit messages, no
  AI co-author trailers.
