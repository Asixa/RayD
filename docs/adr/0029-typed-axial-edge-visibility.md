# ADR-0029: Typed axial-edge visibility

- Status: Accepted
- Date: 2026-07-20
- Decision ID: `typed-axial-edge-visibility`
- Scope: RayD Torch same-graph native source integration

## Context

Channel ADR-028 selects four axial-edge sample points while building a
device-resident diffraction state mask. Its accepted option 2 moves the exact
sample construction and visibility reduction into one RayD OptiX operation,
without routing through the existing public segment-visibility dispatcher.

## Decision

RayD provides the dormant typed operation `axial_edge_visibility_forward`
through `rayd/torch/integration.h`. The request contains a broadcast contiguous
CUDA float32 transmitter `(3,)`, contiguous CUDA float32 AoS edge position and
direction `(N, 3)`, contiguous CUDA float32 `t_min`/`t_max` `(N,)`, and an
optional contiguous CUDA bool active mask `(N,)`. The result is a contiguous
CUDA bool `any_visible` tensor `(N,)` on the same device.

The four default sample fractions are the exact binary32 bit patterns
`0x3ca3d70a`, `0x3eaaaaab`, `0x3f2aaaab`, and `0x3f7ae148`. For every component,
the device computes `span = t_max - t_min`, then
`t = t_min + fraction * span`, then
`point = edge_position + t * edge_direction`. Inline PTX
`sub.rn.f32`, `mul.rn.f32`, and `add.rn.f32` instructions preserve that order
without FTZ or FMA. Only point construction has this local exactness contract.
The dedicated PTX translation unit otherwise inherits
`${RAYD_TORCH_OPTIX_NVCC_FLAGS}`, including the default legacy visibility
fast-math policy, so `trace_segment` retains its established arithmetic.

One separate Params layout, PTX module, and OptiX pipeline owns this primitive.
One call makes exactly one OptiX launch on the caller's current CUDA stream,
reducing four public launch-parameter staging checks to one. It adds no explicit
synchronization beyond the existing common launch staging implementation,
which currently waits on a reusable staging-slot CUDA event.
`N == 0` returns without a launch. Inactive rows and rows containing any
non-finite transmitter, edge, bound, derived parameter, or sample value produce
`false`. Invalid rank, shape, dtype, contiguity, device, scene, or fraction bits
fail before launch.

This candidate is source-linked and direct-tested but dormant. It has no Python
binding, dispatcher registration, dynamic lookup, compatibility alias, or
fallback. The existing segment-visibility PTX, pipeline, and dispatcher retain
their behavior and compile policy.

## Ownership and activation

RayD owns only the exact numerical primitive and its traversal. Channel retains
diffraction topology, state capacity/mask/count policy, packing, solver policy,
and result assembly. Production activation requires an atomic Channel commit
that pins the exact RayD revision and header identity, switches every intended
caller, proves exact mask parity and performance/resource acceptance, and
deletes the superseded Channel numerical implementation. Until then Channel is
the sole production numerical owner; no runtime selector or fallback is
permitted.

## Acceptance gates

1. Release native-core build and the direct integration CTest pass on CUDA.
2. Direct coverage includes empty, all-visible/all-blocked/partial, active-mask,
   non-finite, every fraction boundary, validation failure, and current-stream
   cases.
3. Generated PTX proves point construction uses non-FTZ `sub.rn.f32`,
   `mul.rn.f32`, and `add.rn.f32` instructions without a point FMA, while the
   rest of the module retains the legacy fast-math compile policy.
4. Governance verifies one launch, empty-before-launch, exact fraction bits,
   shared legacy compile flags, locally exact point instructions, no
   Python/dispatcher exposure, and unchanged legacy visibility source ownership.
5. ABI audit records the operation as a dormant same-graph typed candidate.

## Stop conditions

Stop if implementation requires host geometry/reduction, Torch numerical
expressions, a second launch, new explicit synchronization, a legacy-dispatch
change, contraction or arithmetic-order drift, a fallback, or activation
before the atomic downstream pin/switch/delete gate.

Removing `cudaEventSynchronize` from the common launch-parameter staging ring
is a separate Phase 12 optimization. It is not part of this ownership move.
