# ADR-0037: Differentiable SDF ray intersection

- Status: Accepted; integration include and API-version clauses superseded by ADR-0040
- Date: 2026-07-26
- Decision ID: `differentiable-sdf-intersection`
- Scope: the RayD Torch backend's standalone signed-distance-field intersection
  primitive, its grid and oriented-box representation, its sphere-trace
  algorithm, its numeric constants, and its fixed-winner AD contract

> **ADR-0040 supersession.** Historical integration-path and API-6 statements
> below remain evidence; the current typed boundary is
> `rayd/integration/torch.h` at API version 7.

## Context

RayD's only surface representation is the triangle mesh behind OptiX. A caller
that already owns a dense signed distance field has no way to intersect it: the
only route is to extract an isosurface, upload it as a `Mesh`, and trace that.
Isosurface extraction is not differentiable in any useful sense, so the gradient
path from a ray query back to the field values is destroyed before RayD ever
sees the geometry. Every downstream that optimizes an SDF today therefore
reimplements sphere tracing in Python, which is slow, allocates one tensor per
march step, and has no analytic derivative of the hit distance.

The primitive that closes this gap is small and self-contained: a march over a
caller-owned dense grid placed by an oriented box, with the hit distance
differentiated by the implicit function theorem. It shares nothing with the
OptiX traversal stack and needs nothing from it. This record fixes its
representation, its algorithm, and its derivative contract before any code
exists, because the parts that are easy to get subtly wrong (the transform
convention, the frozen-winner definition, the grazing denominator, and what a
missed lane contributes) are exactly the parts that are expensive to change once
a downstream depends on them.

The implementation plan this record governs is
`docs/dev/sdf_intersection_plan.md`. That document owns the phase breakdown and
the file layout; this record owns the numerical and contractual decisions.

## Decision

### 1. Field representation

The field is a caller-owned dense grid `values`: contiguous CUDA float32 of rank
3 and shape `[Nx, Ny, Nz]`, with `N_i >= 2` on every axis. Samples are
**vertex-centered**: `values[i, j, k]` sits exactly on the grid vertex indexed by
`(i, j, k)`, so the corner samples lie on the faces of the bounding box and the
sampled domain is the closed box, not the box inset by half a voxel. This is the
`align_corners=True` convention of `torch.nn.functional.grid_sample`, and the
Phase 1 reference implementation must use that setting to be an oracle.

Interpolation is trilinear. For a grid coordinate `u` the base index is
`b_i = clamp(floor(u_i), 0, N_i - 2)`, the fractional part is `f_i = u_i - b_i`,
and

```
D(u) = sum_{m in {0,1}^3} c_m * values[b + m],
c_m  = prod_i ( m_i * f_i + (1 - m_i) * (1 - f_i) ).
```

The grid coordinate is clamped to the sampled domain, `u_i := clamp(u_i, 0,
N_i - 1)`, before that base/fraction split, so `f_i` always lies in `[0, 1]` and
`D` is never evaluated at a `u` outside `[0, N_i - 1]`. The interpolant has no
extrapolated branch: it is defined on the closed box and nowhere else. Every
sample site in this record is already restricted to the traced interval of
section 3, so the clamp only absorbs float32 rounding at the box faces rather
than covering a real out-of-domain query.

The sign convention is **negative inside, positive outside**, matching
`witwin.core`. Values are world-metric distances. RayD never bakes a field,
never rescales one, and never checks that one is eikonal.

### 2. Placement: the oriented bounding box

The grid spans an oriented box given by three caller tensors, each contiguous
CUDA float32 on the same device as `values`:

- `position` `[3]`, the world-space center of the box;
- `rotation` `[4]`, a **scalar-first** quaternion `(qw, qx, qy, qz)` matching the
  `witwin.core` convention;
- `scale` `[3]`, the **full** side lengths of the box in world units.

The local frame is the box centered at the origin, axis-aligned, spanning
`[-scale_i / 2, +scale_i / 2]` per axis. The quaternion is normalized inside the
operation, `qh = q / max(||q||, eps_norm)`, and the rotation matrix
`R = R(qh)` maps local to world:

```
R = [ 1-2(y^2+z^2)   2(xy - wz)    2(xz + wy) ]
    [ 2(xy + wz)   1-2(x^2+z^2)    2(yz - wx) ]
    [ 2(xz - wy)     2(yz + wx)  1-2(x^2+y^2) ]     with (w, x, y, z) = qh.
```

World to local is `x_l = R^T (x_w - position)`. This map is **rigid**: scale
enters only the grid coordinate mapping, never the geometry. Two consequences
are load-bearing and are relied on everywhere below. First, a ray parameter `t`
measures the same metric distance in both frames, so a world-metric field can be
sphere-traced directly with no Lipschitz correction. Second, `R^T = R^{-1}`, so
the world gradient of the field is `grad_w D = R * grad_l D` with no inverse
transpose.

The grid coordinate of a local point is

```
u_i = (x_l_i / scale_i + 0.5) * (N_i - 1),
```

so `u_i` runs over `[0, N_i - 1]` inside the box. The world-unit voxel edge is
`h_i = scale_i / (N_i - 1)` and `h_min = min_i h_i`. The local gradient follows
from the index-space gradient by the chain rule, componentwise:

```
(grad_l D)_i = (dD/du_i) * (N_i - 1) / scale_i.
```

Direction tensors are normalized inside the operation as well,
`wh = w / max(||w||, eps_norm)`, so `t` is a metric distance regardless of the
caller's normalization. Both normalizations are differentiated through
(section 6); neither is a validation.

### 3. Traced interval

The traced interval is the overlap of the ray with the box, computed by a slab
test in the local frame from `o_l = R^T (o - position)` and `w_l = R^T wh`.
Starting from `t_lo = 0` and `t_hi = tmax`, each axis with
`|w_l_i| > eps_parallel` contributes

```
t_a = (-scale_i/2 - o_l_i) / w_l_i,   t_b = (+scale_i/2 - o_l_i) / w_l_i,
t_lo = max(t_lo, min(t_a, t_b)),      t_hi = min(t_hi, max(t_a, t_b)),
```

and an axis with `|w_l_i| <= eps_parallel` either constrains nothing, when
`|o_l_i| <= scale_i / 2`, or forces a miss. `t_lo > t_hi` is a miss. The
operation has no ray-tmin of its own: a ray whose origin is inside the box
starts at `t_lo = 0`, and starting inside is a supported case, not an error.

### 4. Sphere trace with relaxation and sign-flip bisection

Interpolated grid values are only approximately eikonal. Baking resolution,
trilinear interpolation, and transform-only edits of `scale` all break the unit
gradient assumption that makes unrelaxed sphere tracing conservative. The march
therefore takes a relaxed step and treats overshoot as an expected event with a
defined recovery, not as a failure.

The entry sample fixes the marching sign. With `d_0 = D(u(x_l(t_lo)))`,

```
sigma   = +1 if d_0 >= 0 else -1,
t_0     = t_lo,
t_raw_k = t_k + lambda * sigma * d_k,
t_k+1   = min(t_raw_k, t_hi),
```

so a ray starting outside marches forward on a positive field and a ray starting
inside marches forward on a negative one. `lambda` is the caller's relaxation
factor in `(0, 1]`, default `0.9`. **The step is clamped to `t_hi` before it is
sampled.** The march therefore evaluates the field only on `[t_lo, t_hi]`, where
section 1's interpolant is defined, and never on the extrapolated continuation of
a step that overshoots the box. Each iteration is ordered:

1. `|d_k| < eps_hit` terminates as a hit with `t* = t_k`;
2. `sigma * d_k+1 < 0`, with `d_k+1 = D(u(x_l(t_k+1)))` sampled at the clamped
   `t_k+1`, terminates the march and enters bisection on the bracket
   `[t_k, t_k+1]`;
3. `t_raw_k > t_hi` terminates as a miss;
4. exhausting `max_steps` iterations terminates as a miss.

The sign-flip rule is tested before the exit rule deliberately: a step whose raw
target leaves the box may still cross the level set inside it, and clamping the
sample to `t_hi` is what makes that crossing decidable without leaving the
sampled domain. Rule 3 fires on the unclamped `t_raw_k`, not on `t_k+1`, which by
construction never exceeds `t_hi`. Because `sigma * d_k >= 0` holds on entry to
every iteration, the bracket handed to bisection satisfies
`t_lo <= t_k < t_k+1 <= t_hi`, so every reported `t*` satisfies
`t_lo <= t* <= t_hi <= tmax`. A hit outside the box, or beyond `tmax`, is
unreachable rather than merely unlikely.

Bisection maintains the invariant `sigma * D(a) >= 0 > sigma * D(b)` on
`[a, b] = [t_k, t_k+1]`. It performs at most `kSdfBisectionSteps = 32`
halvings; each evaluates the midpoint `m = 0.5 * (a + b)`, returns a hit at `m`
when `|D(m)| < eps_hit`, and otherwise replaces `a` or `b` by `m` according to
the sign of `sigma * D(m)`. **Exhausting the bisection budget still reports a
hit**, at the final midpoint. The sign change is a proof that the interpolant
crosses zero inside the bracket; `eps_hit` is only a tolerance, and refusing the
hit would discard a proven crossing. Thirty-two halvings shrink the bracket by
`2^-32`, which is below float32 resolution of `t` for any box this primitive is
usable on, so budget exhaustion without `|D(m)| < eps_hit` implies a
discontinuity in the sampled field rather than a slow bisection.

A stalled march, the classic near-tangential sphere-trace failure, is bounded by
`max_steps` and reported as a miss. This is a defined outcome, not an error: the
primitive has no way to distinguish a grazing near-miss from a grazing hit
within its step budget, and the caller controls `max_steps`.

### 5. Outputs and miss semantics

`sdf_intersect` returns, for `N` rays:

| field | dtype / shape | on hit | on miss |
| --- | --- | --- | --- |
| `t` | float32 `[N]` | `t*` | `+inf` |
| `hit_mask` | bool `[N]` | `true` | `false` |
| `position` | float32 `[N, 3]` | `o + t* * wh` | `+0.0` |
| `normal` | float32 `[N, 3]` | `grad_w D / max(||grad_w D||, eps_norm)` | `+0.0` |
| `steps` | int32 `[N]` | field evaluations performed | field evaluations performed |

The `t` miss value is the `distance` miss sentinel already recorded in
`shared/contracts/operations.json` (`miss_sentinels.distance = "inf"`); this
operation introduces no new sentinel. `steps` counts every trilinear evaluation
the lane performed, including the entry sample and every bisection probe. It is
a cost diagnostic, is not differentiable, and carries no contract beyond being
non-negative.

Missed lanes are **bitwise inert**, mirroring the capacity-row validity
philosophy of ADR-0030 and ADR-0031: `position` and `normal` are exact positive
zero, every gradient and tangent of a missed lane is exact positive zero, and a
missed lane contributes no atomic to any shared gradient buffer.

`t = +inf` is the only non-finite value any output may ever contain. A lane whose
interval arithmetic or field sample is non-finite, for any reason including a
degenerate or non-finite `scale`, a zero-length direction, or NaN in `values`, is
reported as a miss. No NaN and no negative infinity leaves the operation, and
this is a tested property, not an aspiration.

### 6. Differentiability: frozen-winner implicit function theorem

The hit satisfies `F(theta) = D(u(x_l(o, w, t*, theta))) = 0`. Derivatives are
taken with the discrete decisions **frozen**: the hit distance `t*`, the boolean
`hit_mask`, the interval endpoints, the bisection branch history, and the base
voxel index are constants in the backward and JVP passes. The tape is exactly
`t*` (float32 `[N]`), `hit_mask` (bool `[N]`), and `base_index` (int32
`[N, 3]`). The base index is stored rather than recomputed so that the forward
and the derivative translation units cannot select different neighborhoods
through different FMA contraction of the same expression.

The IFT denominator is the directional derivative of the field along the ray,

```
g = grad_w D . wh,
```

evaluated at the frozen hit on the frozen voxel, and the derivative of the hit
distance with respect to any input `theta` is

```
dt*/dtheta = -(dF/dtheta) / g_clamped.
```

All six supported gradient and tangent inputs are differentiated through the
partials of `F` at the frozen hit:

| input | `dF/dtheta` |
| --- | --- |
| `values[b + m]` | `c_m`, the trilinear weight of corner `m` (8 nonzero) |
| `origins` | `grad_w D` |
| `position` | `-grad_w D` |
| `directions` | `t* * (I - wh wh^T) / max(||w||, eps_norm) * grad_w D` |
| `scale_i` | `-(grad_l D)_i * x_l_i / scale_i` |
| `rotation_a` | `J_q^T r`, with `r_a = (grad_l D)^T (dR(qh)^T/dqh_a) (x_w - position)` and `J_q = (I - qh qh^T) / max(||q||, eps_norm)` |

The `rotation` and `directions` rows differentiate through the internal
normalizations, so the gradient is with respect to the raw caller tensor and an
optimizer is free to hold an unnormalized quaternion. The `scale` row is exact
and closed-form because `scale` enters only the grid coordinate mapping.

The output derivatives follow:

- `t*` by the table above;
- `position = o + t* * wh` differentiates through `t*`, `origins`, and the
  normalized direction;
- `normal = grad_w D / ||grad_w D||` is **recomputed differentiably** at the
  frozen hit and differentiated in full, including its dependence on the hit
  point and therefore on `t*`. The trilinear interpolant's second derivative is
  needed for this; it is available in closed form, with zero diagonal and
  nonzero mixed terms only. The interpolant's gradient is C0-discontinuous
  across voxel faces, so `normal` and its derivative are those of the piecewise
  expression on the frozen voxel. That discontinuity belongs to the trilinear
  representation, not to the freezing.
- `hit_mask` and `steps` carry no derivative.

Forward mode uses the same frozen decisions and the same partials pushed
forward; JVP and VJP are exact duals on every lane.

**Grazing clamp.** A ray that grazes the level set has `g` near zero and an
unbounded IFT derivative. The denominator is clamped,

```
g_clamped = sign(g) * max(|g|, eps_graze),   with sign(0) := +1,
```

which bounds every derivative of `t*` by `|dF/dtheta| / eps_graze`. This is part
of the numeric contract, not an implementation detail: it is declared in
`shared/contracts/operations.json` and it is tested. No NaN and no infinity may
leave the operation through any gradient or tangent, on any input, ever.

**Determinism.** The forward pass is bitwise deterministic and uses no atomics.
The `values`, `position`, `rotation`, and `scale` gradients accumulate across
rays with float32 atomics and are therefore deterministic only up to float32
addition order. The `origins` and `directions` gradients are per-ray and exact.
This asymmetry is documented rather than removed; a deterministic reduction
would cost a sort or a second pass for no numerical benefit at the tolerances
this primitive is used at.

### 7. Numeric constants

| name | value | origin |
| --- | --- | --- |
| `eps_hit` default | `kSdfEpsHitVoxelFraction * h_min`, `kSdfEpsHitVoxelFraction = 1e-3` | new |
| `relaxation` default | `0.9` | new |
| `max_steps` default | `64` | new |
| `kSdfBisectionSteps` | `32` | new |
| `eps_graze` | `1e-6` | `operations.json` `constants.epsilon.small` |
| `eps_norm` | `1e-12` | `operations.json` `numeric_policy.shared_multipath.normalize_floor` |
| `eps_parallel` | `1e-7` | `operations.json` `numeric_policy.backend_profiles.torch.parallel_epsilon` |
| miss `t` | `+inf` | `operations.json` `miss_sentinels.distance` |

The default `eps_hit` is derived from the smallest world-unit voxel edge
`h_min = min_i(scale_i / (N_i - 1))` because that is the only length scale the
representation actually has. One thousandth of it is three orders inside the
trilinear interpolant's own resolution, so it converges to the interpolant's
zero level set rather than to a fraction of a voxel, which matters because the
interpolant's zero level set is the geometry the derivatives are taken on. It is
reached in a small number of extra iterations near a transversal crossing:
relaxed marching with `lambda = 0.9` leaves a residual factor of `0.1` per step
against a locally planar surface. It also stays comfortably above float32
rounding of the sampled field for boxes up to roughly `1e3` world units.

`eps_hit` is **derived on the device**, not on the host. `scale` is a resident
CUDA tensor, so computing `h_min` on the host would require a device-to-host
read. The host scalar passed to the kernel is a non-positive sentinel
(canonically `-1.0`) meaning "derive from the resident `scale` and the grid
extents", and the derivation is a deterministic function of inputs every lane
computes identically. **The operation performs no device-to-host copy, no
stream synchronization, and no host read of any device tensor, anywhere.**

`eps_graze`, `eps_norm`, and `eps_parallel` are contract constants and are not
caller parameters. `tmax`, `max_steps`, `relaxation`, and `eps_hit` are caller
parameters and are host scalars; none of them is differentiable.

### 8. Validation

Host validation covers structure only, because value validation of a resident
tensor would require a synchronization. It rejects, with an actionable message:
a non-CUDA, non-float32, non-contiguous, or wrong-rank `values`, `position`,
`rotation`, `scale`, `origins`, or `directions`; any `N_i < 2`; a device
mismatch between the grid tensors and the ray tensors; a ray batch whose
`origins` and `directions` disagree in `N`; `max_steps < 1`; `relaxation`
outside `(0, 1]`; a non-positive `tmax`; and a non-positive explicit `eps_hit`.
Value conditions that cannot be checked without a sync, notably `scale_i > 0`
and the finiteness of `values`, are handled by the device path as misses under
the rule in section 5. `SdfGrid` construction validates structure and nothing
else, and never synchronizes.

### 9. Structural constraints

- **Torch backend only.** The capability is declared for both backends with
  `drjit: false, torch: true`. The Dr.Jit port is Phase 5 backlog.
- **No OptiX and no `Scene`.** The primitive is standalone: an `SdfGrid` holder
  plus a functional entry point. It does not participate in acceleration
  structure builds and does not mix with triangle geometry.
- **No committed-PTX exposure.** No SDF header or translation unit may be
  reachable from any `backends/drjit/ptx_sources.json` module include closure,
  and no existing shared header may be modified. New shared device math lives
  in new files under `shared/include/rayd/shared/sdf/`; those files may include
  closure-listed leaf headers such as `rayd/shared/math/vec3.h` and
  `rayd/shared/rt/qualifiers.h` read-only, because inclusion in that direction
  leaves every PTX closure and digest unchanged. This keeps
  `tests/test_ptx_source_digest.py` green by construction rather than by repair.
- **`nvcc_default` only.** Every new CUDA translation unit takes the
  `nvcc_default` profile. No new numeric flag, no profile move, no new profile,
  and no global or target-wide CUDA numeric flag. Any of those is an ADR-0035
  decision and needs its own record.
- **No typed integration boundary.** v1 does not touch
  `backends/torch/include/rayd/torch/integration.h` and does not bump
  `kIntegrationApiVersion`, which stays at `6` under ADR-0028.
- **GIL-free dispatch from the start.** The three operations
  (`sdf_intersect_forward`, `sdf_intersect_backward`, `sdf_intersect_jvp`) are
  registered through the pure C++ `_impl` pattern used by the intersect hot
  path, never the `py::tuple` plus `gil_scoped_acquire` legacy pattern.

## Contract impact

Phase 4 of the plan lands the following, as one change:

1. `shared/contracts/public_api.json`: `capability_keys` gains `sdf_intersect`;
   `apis.sdf_intersect` is added with `category: "core"` and
   `stability: "provisional"`; `backends.drjit.capabilities.sdf_intersect` is
   `false` and `backends.torch.capabilities.sdf_intersect` is `true`. No schema
   change: `capability_keys` is an open string array, and both `core` and
   `provisional` are existing enum members of `public_api.schema.json`.
2. `shared/contracts/operations.json`: `required_capability_keys` gains
   `sdf_intersect`; `operations.sdf_intersect` declares its inputs, its result
   contract, its `eager_native` Torch integration, its fixed-winner AD, the
   grazing clamp, and the `inf` miss sentinel; `result_contracts` gains the
   `t / hit_mask / position / normal / steps` schema with its miss block.
3. `tests/test_shared_operation_contract.py`: the hard-coded operation-name set
   and the per-operation expectation tables.
4. `backends/drjit/python/rayd/drjit/_capabilities.py` and
   `backends/torch/python/rayd/torch/_capabilities.py`: both gain the key in the
   same change, and both repin `_SCHEMA_SHA256` to the new EOL-normalized
   SHA-256 of `public_api.json`. This makes the divergence between the two
   copies **four** lines, not the three that ADR-0036 enumerates. ADR-0036's
   enforcement test counts no lines, so nothing fails, but its prose is a
   factual claim about the repository and must be amended in the same change
   rather than left false. ADR-0036 carries a forward reference to this item so
   that an implementer reading only that record sees the pending amendment.
5. `shared/contracts/compile_policy.json`: the new Torch translation units enter
   under `nvcc_default`, raising the Torch object-unit count and its
   `nvcc_default` membership by the number of new units; the recomputed
   `shared_header_exposure` list gains the new `shared/include/rayd/shared/sdf/`
   headers at a single profile. No profile, flag, frozen divergence, or
   uncontracted entry changes.
6. `backends/torch/CMakeLists.txt`: the new sources join
   `RAYD_TORCH_NATIVE_CORE_SOURCES` with no `EXTRA_FLAGS` and no
   `set_source_files_properties(... COMPILE_OPTIONS ...)` block. There is no PTX
   step, because there is no OptiX module.
7. Torch Python surface: `types.py`, `sdf.py`, `autograd.py`, `__init__.py`,
   each typed inline (the shadow `.pyi` files these modules once carried were
   removed; the Torch package ships no stub at all, and the only stubs left in
   the repository are `backends/drjit/python/rayd/drjit/_C.pyi` and
   `backends/drjit/python/rayd/drjit/__init__.pyi`).
   `tests/test_public_api_manifest.py::test_torch_top_level_reexports_match_runtime_all`
   requires `__all__` and the module's actual re-exports to agree exactly, and
   `test_public_python_modules_are_annotated_inline` iterates a fixed module
   list that must gain `sdf`.
8. Host-compile coverage: the shared SDF headers join the `tests/native/` smoke
   translation unit and the `tests/test_shared_headers.py` inventory.
9. The ADR-0034 source bundle is generated from directory inputs
   (`backends/torch/include`, `backends/torch/src`, `shared/include`,
   `shared/src`), so the new files enter its manifest automatically. No manual
   manifest edit is correct; regenerating the bundle before a release is.
10. Documentation: RayD carries no `FEATURE_LIST.md`, so the user-visible
    surface is recorded in `README.md` and `backends/torch/README.md`, and this
    record is indexed in `docs/adr/README.md`.

`tests/test_ptx_source_digest.py` and `tests/test_compile_flag_policy_contract.py`
must stay green throughout, not be repaired afterwards. A drifted PTX digest
caused by this work means a header rule in section 9 was violated.

## Consequences

- A caller can optimize a resident SDF, its placement, or the rays through it
  without leaving the GPU and without a Python march loop.
- The primitive is unusable for scenes that mix an SDF with a mesh, by design.
  A caller that needs both must intersect twice and resolve the nearer hit
  itself in v1.
- Gradients are the interior IFT term only. A perturbation that moves only the
  silhouette of the field produces a zero gradient, because `hit_mask` is frozen
  and carries no derivative. This is a known bias, is stated in the non-goals,
  and is the reason silhouette gradients are Phase 5 rather than "later".
- The `values` gradient is a scatter through eight trilinear weights per hit ray.
  Many rays converging on a small region produce heavy atomic contention on a
  few voxels. That is inherent to the scatter and is not mitigated in v1.
- Nothing in the existing OptiX, multipath, edge, or scattering surfaces changes.
  The primitive adds translation units and contract rows and touches no existing
  device code.

## Non-goals

Each of these is excluded from v1 deliberately, not by omission.

- Silhouette and visibility discontinuity gradients. No reparameterization, no
  boundary term, no derivative through `hit_mask`.
- Coexistence with triangle geometry, `Scene` membership, and any OptiX
  involvement.
- Baking. RayD consumes distances; it never produces them, never rescales them,
  and never repairs a non-eikonal field.
- Analytic SDF primitives. The kernel takes a grid, not a shape description.
- Multi-grid batching or instancing. One grid and one box per call.
- A Dr.Jit implementation.
- A CPU path. There is no host fallback and none will be added.
- Typed same-graph C++ exposure through `integration.h`.
- Derivatives with respect to `tmax`, `max_steps`, `relaxation`, or `eps_hit`.
- A BSDF, material, emitter, or integrator framework. The standing RayD
  exclusion is unchanged; this primitive is a geometric query.

## Phase 5 backlog

Recorded so the boundary of v1 is legible. None of these is authorized by this
record; each needs its own decision when it is picked up.

1. Dr.Jit backend port, reusing the Phase 2 shared device math unchanged and
   flipping `backends.drjit.capabilities.sdf_intersect` to `true`.
2. Scene-level Mesh and SDF coexistence: an OptiX custom-AABB primitive per SDF
   box with closest-hit resolution shared against triangles.
3. Silhouette and visibility discontinuity gradients by reparameterization.
4. Typed C++ integration boundary exposure through `integration.h` for Channel
   same-graph consumption, which is where a `kIntegrationApiVersion` bump would
   belong.
5. An analytic-SDF fast path passing primitive parameters straight to the kernel
   with no baking.
6. Multi-grid batching and instancing.

## Stop conditions

Stop and reopen this record before:

- adding a CPU, Torch-expression, finite-difference, or detached-gradient
  fallback for any part of the forward or derivative path;
- introducing a device-to-host copy, a stream synchronization, or a host read of
  a device tensor into the operation, including for `eps_hit` derivation or
  input validation;
- letting any output or derivative carry a NaN, a negative infinity, or an
  infinity other than the `t` miss sentinel;
- differentiating a frozen discrete decision, or unfreezing the winner so that
  backward re-marches;
- changing the sign convention, the sample centering, the quaternion order, the
  meaning of `scale`, or the grid coordinate mapping;
- removing the grazing clamp, making it a caller parameter, or giving `sign(0)`
  a different value;
- giving a missed lane a nonzero output, a nonzero derivative, or a single
  atomic contribution;
- assigning a new CUDA translation unit anything other than `nvcc_default`, or
  adding a target-wide CUDA numeric flag;
- making any SDF header or translation unit reachable from a
  `backends/drjit/ptx_sources.json` module closure, or editing an existing
  shared header;
- adding a second implementation, a dispatcher, a compatibility alias, or a
  `sdf_intersect_v2` name for any of it.
