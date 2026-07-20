# ADR-0025: Diffraction operation-family ownership

- Status: Accepted
- Date: 2026-07-19
- Decision ID: `diffraction-operation-family-ownership`
- Scope: Channel Native direct-RayD integration, phases 7 and 8

## Context

RayD already owns solver-neutral diffraction discovery, visibility, path export,
and accumulation primitives. Channel Native separately owns a fixed-winner
pure-wedge field evaluator and several composed diffraction operations. Their
similar names do not make them one operation family: they have different tapes,
fusion boundaries, estimators, random-number consumption, and solver policy.

The direct-integration migration needs one numerical owner for the pure-wedge
primal/backward/JVP family without moving Channel's Monte Carlo Sionna
estimator, coupled reflection-diffraction or double-diffraction operations, or
BDPT policy into RayD. It also must preserve the pure-wedge fast-math boundary
that currently matches RayD's OptiX order-1 exporter.

## Decision

### Operation-family boundary

Ownership is frozen by complete operation family:

| Operation family | Numerical owner after activation |
| --- | --- |
| order-1 diffraction export and visibility | RayD |
| pure-wedge field primal, backward/VJP, and JVP | RayD |
| Monte Carlo Sionna diffraction tape accumulation and AD | Channel |
| coupled reflection-diffraction field and AD | Channel |
| coupled double-diffraction field and AD | Channel |
| coupled reflection-diffraction preparation and AD | Channel |
| composed coupled RD/DD geometry | Channel operation using RayD primitives |
| solver packing, accumulation, metadata, and results | Channel |
| BDPT proposal, PDF, MIS, storage, and estimator policy | Channel |

An operation does not move merely because it invokes a RayD primitive. The
fusion/launch contract, tape lifetime, device primitive, numerical order, and
estimator policy determine the owner.

### Typed pure-wedge operation family

RayD owns one complete typed C++ family:

- `field_diffraction_wedge`
- `field_diffraction_wedge_backward`
- `field_diffraction_wedge_jvp`

The declarations belong under
`backends/torch/include/rayd/torch/rf/diffraction.h` and are included by
`rayd/torch/integration.h`. They use `at::Tensor`,
`std::optional<at::Tensor>`, and named request/result structures. They are
source-level interfaces for consumers built in the same CMake/LibTorch graph,
not a cross-build stable binary ABI and not a second Python extension surface.

The primal request preserves the existing row-aligned CUDA contract for source,
target, edge position/direction and finite-edge bounds, the two face normals,
exterior angle, face validity and material values, transmit power, frequency,
and boundary taper state. Winner reconstruction remains one optional all-or-none
five-tensor bundle: `vertex_v0`, `vertex_v1`, `vertex_opp0`, `vertex_opp1`, and
`edge_boundary`. The primal result remains complex64 field vector `[N, 3]` plus
float32 outgoing direction `[N, 3]`, with the existing device, stride, row
identity/order, and empty-row semantics.

Backward accepts optional field and direction cotangents and preserves the
existing requested-gradient flags. It returns only the currently supported
continuous gradients: source, target, both faces' permittivity, conductivity,
and gain, frequency, and the four optional winner vertices. Face validity,
relative permeability, edge-boundary classification, taper width, and all
discrete winners remain non-differentiable. JVP accepts the corresponding
continuous tangents and returns field and direction tangents with the primal
row schema. Vertex tangents require the matching winner-vertex primal inputs.

This is a fixed-winner contract. The forward-selected topology, edge, faces,
visibility, finite-edge regime, and boundary class are frozen; native backward
and JVP differentiate the live continuous geometry and electromagnetic values.
Production finite differences are forbidden.

### Numerical, fusion, and runtime contract

The move is numerical-identity preserving:

- `wedge_row_eval<T>` retains its current evaluation order and branch behavior;
- optional winner vertices and fixed-winner geometry AD retain their current
  semantics;
- primal, backward, and JVP remain three separate native launches, one launch
  for each invoked family entry, with no new persistent tape or materialized
  inter-launch intermediate;
- row/output schemas, dtype, device, stride, empty-input behavior, and gradient
  state remain unchanged;
- the order-1 exporter and fixed-winner wedge reevaluator remain separate ABI
  operations; fusing or deduplicating them requires a separate numerical ADR;
- all CUDA work uses the caller's current Torch CUDA stream and keeps device
  data resident.

Every entry validates shape, dtype, contiguity, device, optional-tensor, and
API/build compatibility before launch. Unsupported devices, invalid contracts,
missing capability, ABI mismatch, or CUDA failure raise immediately. There is
no CPU, Torch-expression, finite-difference, legacy-dispatch,
reduced-algorithm, zero-result, or detached-gradient fallback.

### Fast-math boundary

The pure-wedge translation unit retains its current `--use_fast_math`
compilation because that boundary is required for parity with RayD's OptiX
order-1 diffraction exporter. Acceptance evidence must compare compiler flags,
PTX/SASS, register usage, launch count, and numerical parity before and after
the move.

This decision does not authorize fast math for transmission, coupled RD/DD,
Monte Carlo Sionna accumulation, scattering, or any other precise-math family.
If the flag cannot remain isolated to the pure-wedge family, the migration
stops pending a separate numerical decision.

### RayD tape producers and Channel estimators

RayD diffraction accumulation may produce a fused sampling/visibility tape;
Channel's Monte Carlo Sionna family consumes that tape and remains its numerical
and estimator owner. A misleading Channel facade such as
`bdpt_diffraction_accumulation_forward` or a historical `raydn_*` name must be
renamed to describe the RayD-produced sample tape, for example
`rayd_diffraction_sample_tape_forward`.

That rename is semantic only. On the RayD producer side it preserves the full
fused output, including currently unconsumed map columns, as well as its launch
and fusion boundary, sampling/visibility decisions, RNG consumption, row order,
and output exactness. Cropping the producer to a tape-only kernel, moving
estimator math into RayD, or changing its fusion boundary requires a separate
performance and numerical ADR.

Separately, the Channel consumer keeps the complete fixed-tape Sionna family:
primal, backward, and JVP; proposal and Jacobian terms; finite-thickness slab
semantics; cell atomics; seed and RNG interpretation; row order; and output
exactness. These consumer invariants are not properties of the RayD tape
producer and do not move during its facade rename.

### Channel-owned diffraction and policy

Channel remains the sole owner of Monte Carlo Sionna primal/AD, coupled RD and
DD primal/AD, coupled RD preparation for this migration, and the composition of
RD/DD geometry. These families may call public RayD primitives and shared RF
device headers in place; they are not split into extra kernels or intermediate
tensors to mirror repository layout.

Channel also retains its fields kernel facade, autograd dispatch, and stable
`_channel_native` binding names for the pure-wedge family. Activation replaces
the numerical implementation behind those facades with the typed RayD calls;
it does not delete or bypass Channel's owning domain boundary.

BDPT standalone diffraction continues to use enumerated deterministic discrete
paths. Channel owns BDPT proposal generation, PDFs, MIS, storage, RNG, and
result policy whenever those paths are live. RayD does not acquire a BDPT
solver dependency through this decision.

### Legacy naming and deletion sequence

Cleanup follows usage evidence rather than historical names:

1. rename live Channel discovery facades from
   `bdpt_diffraction_discover_edges*` to `mc_diffraction_discover_edges*`;
2. audit `mc_diffraction_edge_geometry` against historical BDPT geometry and
   delete a dead wrapper instead of copying or preserving it;
3. for every legacy BDPT binding, prove a static caller, dynamic binding use,
   public import, and real BDPT end-to-end caller; if any required evidence is
   absent, delete the dead symbol, test, manifest entry, and maintenance-budget
   entry together;
4. if `_tx_visible_diffraction_states` remains live, replace the complete path
   with one native Channel planning/selection operation. That operation owns
   point planning, the ordered four visibility queries at exact fractions
   `(0.02, 1/3, 2/3, 0.98)`, native any-visible reduction, and stable selected
   row identity/order. It may invoke RayD batched visibility as its primitive,
   but Python loops, Torch geometry or reduction, scalar extraction, host bools,
   and host synchronization are forbidden;
5. remove historical `RayDN/raydn` facades after all callers use the typed RayD
   owner name; delete RayD legacy C/Python surfaces only after a repository-wide
   consumer audit in the later legacy-removal phase.

No compatibility re-export or runtime fallback is added during cleanup.

### Cross-repository activation

This ADR accepts the ownership boundary but does not itself move code or change
an API. Activation is ordered:

1. RayD merges the complete typed primal/backward/JVP family as a dormant
   candidate with direct contract tests;
2. Channel pins that reviewed RayD revision, switches every pure-wedge caller,
   proves parity and end-to-end coverage, and deletes its local pure-wedge CUDA
   implementation in the same activation commit;
3. Channel performs the semantic rename and evidence-based dead legacy cleanup
   without changing numerical behavior;
4. later deletion of RayD legacy surfaces occurs only after all consumers have
   migrated and the repository-wide audit passes.

While the RayD candidate is dormant, Channel remains the authoritative
production numerical owner. Dormancy is not an accepted long-term dual-owner
mode. Rollback changes the Channel pin to the previous accepted RayD revision;
it never introduces runtime dispatch or a fallback.

## Consequences

Positive consequences:

- RayD owns the solver-neutral order-1 export/visibility and fixed-winner
  pure-wedge field families as one coherent diffraction capability;
- Channel retains estimator, coupled-operation, and BDPT policy ownership;
- primal, backward, and JVP have one complete native numerical owner;
- the typed, current-stream, fail-loud integration boundary remains the only
  production path.

Costs and limitations:

- both repositories require an ordered candidate/pin/switch/delete sequence;
- pure-wedge fast-math isolation becomes an explicit release contract;
- this decision does not authorize a new Python API, stable cross-DSO ABI,
  fusion change, numerical optimization, or solver-policy migration.

## Acceptance gates

The ownership switch is complete only when both repositories record:

1. exact or frozen-baseline parity for order-1 export and pure-wedge forward
   outputs, including ISB/RSB, finite-edge, stationary-external-incident, and
   the Channel ADR-012/ADR-013 regression cases;
2. material, frequency, source, target, and winner-vertex JVP/VJP coverage,
   adjoint dot-product checks, and test-only finite-difference oracles;
3. fixed-tape Monte Carlo AD, seed, RNG-consumption, proposal, Jacobian, slab,
   atomic, row-order, and output parity;
4. coupled RD/DD primal/AD lockstep with no new launch, synchronization,
   persistent tape, or materialized intermediate;
5. unchanged pure-wedge fast-math isolation, PTX/SASS behavior, register usage,
   launch count, and caller-current-stream behavior;
6. direct typed contract, invalid-shape/dtype/device/optional-input, empty-input,
   ABI/capability, CUDA-failure, and no-fallback tests;
7. zero live historical source facades for the activated pure-wedge family and
   no duplicate Channel CUDA numerical owner;
8. a real BDPT end-to-end caller for every retained BDPT-labelled binding, with
   dead tests, manifests, and budgets removed alongside dead symbols.

## Stop conditions

The migration stops and leaves the current owner intact if:

- generated code, evaluation order, or baseline drift cannot be explained;
- pure-wedge fast math spreads into any precise-math family;
- an implementation treats all diffraction operations as one family or moves
  Channel estimator/coupled/BDPT policy into RayD;
- parity requires a fallback, duplicate production owner, new host transfer,
  synchronization, launch, persistent tape, or detached gradient;
- the complete typed primal/backward/JVP candidate and its direct tests are not
  available before the Channel switch;
- a sample-tape rename retains a historical alias, crops or reorders producer
  columns, or changes the producer's launch, RNG, visibility, or fusion contract;
- a retained legacy BDPT binding lacks any one of its static caller, dynamic
  binding, public import, or real BDPT end-to-end evidence axes;
- native visibility planning changes any of the four fixed fractions,
  any-visible decision, selected row identity/order, or introduces a Python
  loop, Torch geometry/reduction, scalar extraction, host bool, or host sync.

Any numerical or fusion-boundary change requires its own accepted ADR and
evidence before implementation.
