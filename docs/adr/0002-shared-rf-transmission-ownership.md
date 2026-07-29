# ADR-0002: Shared RF primitives and transmission ownership

- Status: Accepted; source path and ownership-namespace clauses superseded by ADR-0039 and ADR-0040
- Date: 2026-07-19
- Decision ID: `shared-rf-transmission-ownership`
- Scope: Channel Native direct-RayD integration, phases 5 and 6

> **ADR-0039/0040 supersession.** The `rayd/{shared,torch}/rf/` paths and generic
> `rf` ownership namespace below are historical. Canonical transmission owners
> are `src/transmission_device.cuh` and `include/rayd/transmission.h`; cross-concept field transport
> is owned by `include/rayd/field_transport.cuh` and `src/field_transport_ad.cuh`. Numerical ownership, fusion,
> stream, activation, rollback, and downstream atomic-switch clauses remain in
> force.
## Context

Channel Native currently owns both the public Channel contracts and the CUDA
implementation of RF transmission. Its transmission, coupled diffraction,
BDPT, and scattering kernels also share a private device-helper closure for
complex arithmetic, media, Fresnel terms, layer stacks, and Jones transport.
Keeping that general-purpose closure in Channel would make a later RayD-owned
transmission or scattering operation depend on downstream implementation code
or require a copied second implementation.

Ownership is therefore decided by operation family and numerical dependency,
not by Python package layout. RayD can own solver-neutral RF device math and a
complete native operation family without taking ownership of Channel's
material encoding, topology, estimators, or result policy.

The audited migration closure contains 129 helpers. This decision moves 112
solver-neutral numerical helpers to RayD: 4 complex helpers, 2 medium helpers,
5 Fresnel helpers, 2 layer-stack helpers, 21 field-transport helpers, 76
field-transport AD helpers, and the two shared output-chain AD helpers
`fold_output_cotangents` and `write_output_tangents`. Ten Channel
tensor-conversion, validation, allocation, pointer, and launch adapters remain
in Channel. Seven scattering-table interpolation helpers are outside this
decision and are reserved for the separate scattering ownership ADR.

## Decision

### Public RF device headers

RayD becomes the unique source owner of the solver-neutral RF dependency
closure. The public, source-level device headers are:

- `shared/include/rayd/shared/rf/complex.cuh`
- `shared/include/rayd/shared/rf/medium.cuh`
- `shared/include/rayd/shared/rf/fresnel.cuh`
- `shared/include/rayd/shared/rf/layer_stack.cuh`
- `shared/include/rayd/shared/rf/field_transport.cuh`

Torch-specific AD helpers that require `c10::complex` belong in
`backends/torch/include/rayd/torch/rf/field_transport_ad.cuh`; they are not a
backend-neutral shared ABI. Channel-owned fused kernels may include the public
RayD headers. RayD must never include Channel implementation headers.

Migration is by helper closure, with normalized-body, compile-attribute,
generated-code, and primal/dual lockstep evidence. After a Channel consumer
switches, the corresponding private Channel helper definitions are deleted;
copying them into both projects is not an accepted end state.

### Typed transmission operations

RayD owns the native numerical implementation of these complete families:

- `em_layer_stack_eval`, `em_layer_stack_backward`, and
  `em_layer_stack_jvp`
- `field_transmission_sequence`, `field_transmission_sequence_backward`, and
  `field_transmission_sequence_jvp`

The typed C++ declarations live in
`backends/torch/include/rayd/torch/rf/transmission.h`, which is included by
`rayd/torch/integration.h`. They use `at::Tensor`,
`std::optional<at::Tensor>`, and named request/result structures. They are
source-level interfaces for consumers built in the same CMake/LibTorch graph,
not a cross-build stable binary ABI and not a second Python extension surface.

The implementation translation units are
`backends/torch/src/torch_ext/rf/layer_stack.cu` and
`backends/torch/src/torch_ext/rf/transmission_sequence.cu`. File placement does
not split ownership: primal, backward/VJP, and JVP move and remain validated as
one operation family.

### Numerical, fusion, and runtime contract

The move is numerical-identity preserving:

- each row traverses its resident CSR layer chain in one operation; per-layer
  state is not materialized into an inter-launch tensor;
- backward and JVP recompute the current layer chain and add no persistent
  tape;
- shared layer gradients retain the existing atomic accumulation and
  evaluation order;
- complete-row transmission retains layer traversal, Jones transport,
  carrier, receiver projection, length, delay, and gain fusion;
- the seven field outputs, row identity/order, tensor dtype/device/stride,
  projection, phase, delay, and gain schemas remain unchanged;
- transmission translation units use precise math and must not inherit the
  pure-wedge `--use_fast_math` contract or scattering's `--fmad=false`
  contract.

Every entry validates shape, dtype, contiguity, device, optional-tensor, and
API/build compatibility before launch. CUDA work uses the caller's current
Torch CUDA stream. Unsupported devices, invalid contracts, missing capability,
ABI mismatch, or CUDA failure raise an error; there is no CPU, Torch-expression,
finite-difference, legacy-dispatch, reduced-algorithm, or detached-gradient
fallback.

### Channel-owned boundary

RayD does not own Channel policy or public result assembly. Channel continues
to own:

- material models, material ABI and CSR encoding, caches, validation facades,
  and Python-facing typed row/material contracts;
- topology pairs and winners, thin-sheet eligibility, and component-5 packing;
- `_channel_native` bindings and Channel domain facades for all six contracts;
- `bdpt_transmitted_light_subpath_state` primal/backward/JVP as one fused
  19-field state operation, including event masks/types, PDFs, depth, lateral
  exit, and phase compensation;
- BDPT standalone orchestration, event probability, MIS, RNG, and estimator
  policy;
- MC Basic incident-polarization and power-domain estimator semantics;
- component accumulation, metadata, solver results, and final channel
  representation.

Channel-owned fused operations call RayD shared RF device primitives in place;
they are not split into extra kernels merely to move a subexpression. Straight
segment penetration batching and MC event glue are also excluded because they
would change the fusion/launch boundary and require a separate decision with
profiling evidence.

### Cross-repository activation

RayD may merge and test a complete family as a dormant candidate before the
Channel repository changes its pin. While dormant, the candidate is not
compiled into or called by `_channel_native`; Channel remains the authoritative
production numerical owner. Dormancy is a migration state, not a supported
dual-owner mode and not a reason to publish a second production API.

Activation requires a Channel commit that pins the reviewed RayD revision,
links and calls the typed family, proves exact/AD/current-stream/negative
contracts, and deletes the local implementation in the same switch. RayD then
becomes the sole production numerical owner. Rollback changes the Channel pin
to the previous accepted RayD revision; it does not add runtime dispatch or a
fallback.

## Consequences

Positive consequences:

- transmission, coupled diffraction, BDPT, and later scattering kernels share
  one RF device-math source;
- each transmission family has one complete primal/backward/JVP owner;
- Channel preserves its domain contracts and fused solver operations;
- direct source linkage retains one `_channel_native` Python extension, one
  Torch allocator/stream model, and fail-loud behavior.

Costs and limitations:

- RayD public source headers must preserve the accepted numerical and compile
  contract for source-linked consumers;
- the two repositories require an ordered candidate/pin/switch sequence;
- this decision does not authorize a new Python API, a stable cross-DSO ABI,
  a transmission optimization, or any solver-policy migration.

## Acceptance gates

The ownership switch is complete only when both repositories record:

1. exact primal parity and frozen evaluation order for both families;
2. backward/JVP parity, duality, and material/frequency gradient coverage;
3. current-stream and multi-device negative coverage with no synchronization or
   host transfer regression;
4. unchanged launch count, precise-math flags, atomic behavior, persistent-tape
   count, and complete-row fusion;
5. no Channel private duplicate of the migrated helper or CUDA operation;
6. no RayD dependency on Channel headers, schemas, solver policy, or Python
   runtime;
7. fail-loud ABI, dtype, shape, device, optional-input, empty-input, and CUDA
   error tests through the typed integration entry and the Channel end-to-end
   caller.
