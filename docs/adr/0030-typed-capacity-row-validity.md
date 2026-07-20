# ADR-0030: Typed capacity-row validity

- Status: Accepted
- Date: 2026-07-20
- Decision ID: `typed-capacity-row-validity`
- Scope: RayD Torch pure-wedge, transmission, and generic scattering rows

The numeric version `3` below records this decision's accepted boundary.
ADR-0031 subsequently advances the current stable integration API to version
`4` for required diffraction path-export validity without changing these row
validity contracts.

## Context

Device-selected path cardinality cannot require a host count transfer merely to
shape tensors. Capacity-shaped requests therefore carry inactive rows. Some RF
operations previously inferred validity from interaction-local state or assumed
every allocated row was live, making poisoned inactive storage unsafe and
preventing a uniform device-resident contract.

## Decision

The stable typed integration API advances to numeric version `3`; its header
path, CMake target, and identity remain unchanged. The following top-level
primal requests require a contiguous CUDA boolean tensor with one element per
row:

- `DiffractionWedgeRequest.valid`;
- `TransmissionSequenceRequest.path_valid`, distinct from per-interaction
  `interaction_valid`;
- `ScatteringTableEvalRequest.valid`;
- `ScatteringTableSampleRequest.valid`;
- `ScatteringTablePdfRequest.valid`;
- `ScatteringEnsembleEvalRequest.valid`;
- `ScatteringPatchIntegralEvalRequest.valid`;
- `ScatteringChainEnsembleEvalRequest.valid`;
- `ScatteringChainRealizationEvalRequest.valid`.

Every relevant CUDA kernel tests row validity before reading row payloads,
indices, depths, or material/table IDs. Invalid primal and JVP outputs are
bitwise zero. Supported row-local backward outputs are bitwise zero, and invalid
rows perform no atomics into shared table, material, layer, height, coefficient,
frequency, or other shared gradients. Total reductions consume the zero row
value without changing reduction topology or order.

Backward and JVP requests inherit the required tensor through their nested
primal request. The mask is never optional; there is no implicit all-valid
compatibility path or generation-suffixed API. Empty requests carry an empty
validity tensor and preserve existing empty schemas. Launches use the caller's
current CUDA stream.

## Consequences

Callers may pass fixed-capacity device-resident storage without compaction or a
device-to-host count transfer. Poisoned invalid rows are safe, and invalid rows
cannot contaminate reductions or shared gradients. Existing callers must supply
the new required tensor and reject API versions other than `3`.

This is a contract change only. It does not alter valid-row mathematics,
fusion, launch count, random-number consumption, reduction order, or the common
OptiX launch-parameter staging synchronization addressed separately by
ADR-0029 follow-up work.

## Acceptance gates

1. Direct CUDA tests cover empty, all-invalid poisoned, sparse valid/invalid,
   backward/JVP zero semantics, and caller-current-stream execution.
2. Static governance verifies required request fields, numeric API version `3`,
   validity-before-payload source order, and identical repository guardrails.
3. The stable header has one identity and no forwarding or runtime selector.
4. Valid-only direct contracts remain unchanged and pass with all-one masks.

## Stop conditions

Stop if an operation reads an invalid row before the gate, makes validity
optional, performs invalid shared-gradient atomics, compacts through a host
count, changes a valid-row numerical result, or adds a synchronization or
fallback path.
