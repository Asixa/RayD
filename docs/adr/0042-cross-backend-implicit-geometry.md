# ADR-0042: Cross-backend implicit and surfel geometry

- Status: Accepted; standalone-only composition clauses superseded by ADR-0043
- Date: 2026-07-29
- Decision ID: `cross-backend-implicit-geometry`
- Scope: standalone SDF and surfel geometry in the Dr.Jit and Torch backends

## Context

RayD previously exposed surfels only through Dr.Jit (ADR-0001) and dense SDF
intersection only through Torch (ADR-0037). The required product scope is now
explicitly operation-specific: an SDF participates in line of sight and
specular reflection, while a surfel participates in line of sight, specular
reflection, and alpha transmission. Neither geometry contributes an edge or a
diffraction path.

The standalone geometry owners remain the numerical and acceleration owners.
ADR-0043 later adds a unified composition layer above them without changing the
triangle `Scene` kernels, diffraction exporter, UTD state, reduction order, or
multipath compile profiles.

## Decision

Both backends advertise `surfel = true` and `sdf_intersect = true`.

The operation matrix is closed:

| Geometry | Intersection | LOS | Reflection | Transmission | Diffraction |
| --- | --- | --- | --- | --- | --- |
| dense SDF | yes | yes | yes | no | no |
| surfel | yes | yes | yes | yes | no |

These remain standalone geometry owners and do not become triangle `Scene`
primitives. The original prohibition on RayD-owned mixed precedence and
composition is superseded by ADR-0043. This decision still does not create a
material, BSDF, emitter, or integrator framework.

### SDF

`SdfGrid.intersect`, `SdfGrid.visible`, and
`SdfGrid.trace_reflections` are present in both backends. Both use the sphere
march, bracket refinement, miss sentinels, frozen cell, and implicit-function
derivative contract of ADR-0037.

Torch retains its existing CUDA dispatcher. Dr.Jit adds one ordinary CUDA
object unit, `sdf_jit`, launched on the current Dr.Jit CUDA stream. It shares
the device march implementation with Torch and re-evaluates the frozen cell in
Dr.Jit arrays for reverse and forward AD. The unit uses `nvcc_default`; this is
a new assignment under ADR-0035, not a change to an existing numeric profile.

SDF reflection reports the standalone SDF as primitive ID zero. Query origins
after a bounce are biased by the larger of `RayEpsilon` and twice the resolved
hit tolerance. No SDF function imports or calls a diffraction operation.

### Surfel

`SurfelScene.intersect`, `SurfelScene.visible`,
`SurfelScene.trace_reflections`, and `SurfelScene.composite_alpha` are present
in both backends. `transmittance` is the final transmitted fraction of the
alpha composite.

Dr.Jit retains its existing OptiX candidate pipeline and analytic AD
re-evaluation; reflection is a bounce loop over that same accepted hit.

Torch builds detached quad proxies in the existing accelerated triangle
`Scene`. Proxy intersections choose candidates only. Final depth, point,
normal, Gaussian coordinates, opacity, alpha, value, and every continuous
derivative are re-evaluated from the caller-owned surfel tensors. Transmission
keeps a bounded, depth-and-ID ordered candidate set in resident Torch tensors,
so coplanar translucent surfels are not lost by an origin-advance traversal.

Surfel IDs are the reflection primitive IDs. LOS uses `ShadowEpsilon` at both
segment endpoints on both backends. No surfel path is converted to an edge or
passed to diffraction.

## API and contract changes

- `contracts/public_api.json` marks both capabilities true for both backends.
- `contracts/operations.json` gives the SDF result record and eager integration
  to both backends and records the cross-backend surfel endpoint offset.
- `contracts/surfel_backend_decision.json` now records this decision and marks
  ADR-0001 as superseded.
- `rayd.torch` exports `SurfelCloud`, `SurfelTraceOptions`, `SurfelScene`,
  `SurfelIntersection`, and `SurfelComposite`.
- `rayd.drjit` exports `SdfGrid`, `SdfTraceOptions`, SDF result records, and the
  new surfel reflection method through its existing native module.

The new Dr.Jit public header is `rayd/jit/sdf.h`. No compatibility forwarding
header or alternate dispatcher name is introduced.

## Consequences

- Callers can use either geometry family with either AD frontend.
- Triangle-only scene queries and every diffraction implementation stay
  unchanged.
- Torch surfel transmission is bounded by `max_candidate_hits`; capacity
  saturation is exposed through `candidate_buffer_full` rather than silently
  changing the configured capacity.
- Torch surfel proxy rebuild is explicit through `build()`, matching the
  Dr.Jit lifecycle.

## Superseded decisions

This record supersedes ADR-0001 in full. It supersedes only the Torch-only and
Dr.Jit-backlog clauses of ADR-0037; ADR-0037 continues to own the SDF
representation, numerical march, derivative, sentinel, and failure contracts.

## Rejected alternatives

- Adding SDF/surfel branches to diffraction was rejected because the required
  scope explicitly excludes diffraction.
- Replacing the triangle scene's fused visibility or reflection kernels with a
  generic mixed-geometry dispatcher was rejected here. ADR-0043 supersedes the
  standalone-only conclusion with a composition layer that leaves those
  kernels unchanged.
- Treating a Torch proxy triangle as the differentiable surfel was rejected;
  the proxy is only a detached broad phase and cannot own Gaussian hit math.
