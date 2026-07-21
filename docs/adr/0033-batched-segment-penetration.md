# ADR-0033: Batched segment-penetration geometry

- Status: Accepted
- Date: 2026-07-21
- Decision ID: `batched-segment-penetration`
- Scope: RayD Torch stable typed integration, OptiX traversal, fixed-capacity
  geometry, and fixed-winner AD

## Context

Channel previously discovered straight transmission crossings through a Python
depth march around RayD's single-hit query. That route repeatedly compacted
device rows, launched one traversal per depth, and used host Boolean decisions.
It also carried two intentionally different numerical policies for deterministic
enumeration and Monte Carlo target-inset traversal.

Channel ADR-027 accepts a single solver-neutral RayD geometry owner and freezes
those two policies without moving material eligibility or RF wall-product policy
into RayD. This RayD decision records the dormant producer that Channel may pin
and activate atomically.

## Decision

The stable `rayd/torch/integration.h` identity remains
`rayd.torch.integration` and advances to numeric API version `6`. It adds the
complete typed family:

- `SegmentPenetrationPolicy`, with exactly `EnumeratedFullDistance` and
  `MonteCarloTargetInset`;
- `SegmentPenetrationRequest`, `SegmentPenetrationResult`, and
  `SegmentPenetrationTapeResult`;
- `SegmentPenetrationBackwardRequest` and result;
- `SegmentPenetrationJvpRequest` and result; and
- `segment_penetration_forward`, `segment_penetration_forward_tape`,
  `segment_penetration_backward`, and `segment_penetration_jvp`.

The request owns no raw handle. It holds a typed `SceneResource` reference,
contiguous CUDA float32 `[N,3]` origins and targets, optional contiguous CUDA
bool `[N]` input validity, explicit host-known `input_active_any`, non-negative
host-known hit capacity `D`, one explicit policy, a frozen finite non-negative
float32 scene diagonal, the caller's contiguous CUDA int32 `[1]` capacity
failure state, and one caller-assigned non-zero single `failure_bit`. RayD
validates all shape, dtype, device, contiguity, enum, capacity, bit, and int32
indexing conditions before allocation or launch. RayD never clears the shared
failure state.

### Traversal and capacity

For `N>0` and `input_active_any=true`, each forward entry submits exactly one
OptiX launch. A raygen lane performs the ordered closest-hit march internally
and evaluates at most `D+1` probes. The extra probe distinguishes exactly `D`
hits followed by a clear tail from overflow. `D=0` is valid. `N=0` submits no
traversal.

When `input_active_any=false`, RayD submits no OptiX launch. Its fixed
initialization/status CUDA operation checks the device active mask on the
caller's current stream. A true row contradicts the host-known declaration,
atomically ORs `failure_bit`, and leaves the entire output inert. For `N>0`, an
absent mask with `input_active_any=false` is invalid. No device mask or count is
copied to the host and no stream synchronization is introduced.

The output capacity is fixed `[N,D]`. Successful rows publish ordered validity,
counts, reach state, distance, direction, hit distance, position, selected
normal, geometric normal, and global primitive id. Unused and input-inactive
slots are canonical inert. Active degenerate full-distance rows and active
zero-inset Monte Carlo rows complete with `reached_target=true`; input-inactive
rows remain false.

An accepted `D+1` hit atomically ORs `failure_bit`. A same-stream finalization
kernel then makes every result and tape row in the batch inert before the
caller can observe the shared failure. The per-row `overflow` tensor alone may
retain diagnostic bits. The operation never traps, truncates, returns partial
geometry, reads a device scalar on the host, or creates a private failure flag.

### Frozen policies

`EnumeratedFullDistance` uses the complete endpoint distance, ordered
`sqrt((x*x+y*y)+z*z)` normalization with denominator floor `1e-9`, a finite
strict `t < remaining` hit test, normalized geometric normals, L2 restart
epsilon `max(||p||2*1e-6, scene_diagonal*1e-6, 1e-6)`, and tracked
`clamp_min(distance-traveled,0)` remaining distance.

`MonteCarloTargetInset` uses denominator floor `1e-6`, initial remaining
distance `clamp_min(distance-epsilon_inf(target),0)`, the inclusive
`0 < t <= remaining` hit test, the current RayD shading normal without an
additional caller normalization, L-infinity restart epsilon
`max(||p||inf*1e-6, scene_diagonal*1e-6, 1e-6)`, and subtractive remaining
distance. Its OptiX traversal bound is the next representable float above
`remaining` solely so a hardware-open `tmax` can publish an exact endpoint;
the original `0 < t <= remaining` test remains authoritative and rejects every
larger hit. Both use intersection flags `7` and the established RayD scene
intersection t-min. The family OptiX PTX and CUDA AD translation unit compile
with explicit non-FTZ, precise division, and precise square-root flags while
retaining established default FMA contraction; the global OptiX fast-math
option is not inherited. The restart point alone preserves the former two-op
Torch multiply-then-add float32 rounding boundary explicitly, and both
remaining-distance updates retain their former left-associated order.
Non-finite delta components and a non-finite ordered squared-norm result OR the
assigned failure bit and publish only the canonical inert batch/tape.

### Tape and AD

The tape freezes validity, primitive identity, barycentric coordinates,
restart epsilon and branch, L-infinity tie mask, and direction-denominator
branch. Backward and JVP consume those decisions and never call `optixTrace`,
reselect a winner, or differentiate a capacity decision. Continuous distance,
direction, hit distance, hit position, selected normal, and geometric normal
support optional cotangents/tangents for origins, targets, and scene vertices.
Invalid, input-inactive, and failed rows produce exact positive-zero
derivatives. Shared scene-vertex VJP uses native CUDA atomics; there is no Torch
expression, finite-difference, CPU, or detached fallback.

## Ownership and activation

RayD owns traversal, both policies, fixed hit/tape construction, overflow
detection, and the complete native AD family. Channel retains policy choice,
scene-diagonal construction, material and thin-sheet eligibility, topology,
the Monte Carlo polarized wall product, solver results, and the terminal
capacity failure observer.

This implementation is dormant. It has no RayD Python binding and no legacy
dispatcher entry. Activation requires Channel to pin the reviewed RayD commit,
add its sole facade, switch all target callers, delete the old march in the same
commits, and pass the exactness, AD, launch, stream, memory, performance, and
packaging gates in Channel ADR-027.

## Acceptance

- direct typed tests cover both policies, empty and all-inactive batches,
  `D=0`, clear, one-hit, exact-`D`, `D+1`, mixed overflow, degenerate and
  zero-inset rows, poison state, non-default stream, and VJP/JVP duality;
- plain and tape forward primal fields are bitwise equal;
- policy tests compare first-hit fields against the established typed
  intersection and lock strict/inclusive endpoint plus L2/L-infinity restart;
- static inspection proves one host OptiX submission, one raygen depth loop,
  zero traversal for `input_active_any=false`, and no traversal in AD;
- static inspection finds no device-to-host count/mask copy, stream
  synchronization, host scalar extraction, fallback, Python binding, or
  temporary generation name in the family; and
- the stable integration direct test and dedicated penetration direct test pass
  on the supported CUDA/OptiX build.

## Stop conditions

Stop before activation if parity requires a host-visible count, dynamic result
shape, more than one traversal launch per forward batch, an inferred or merged
policy, changed comparison/epsilon/normal semantics, silent truncation, partial
results, retracing AD, a Python dispatcher, a second numerical owner, or a
compatibility alias.
