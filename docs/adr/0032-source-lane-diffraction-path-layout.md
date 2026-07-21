# ADR-0032: Source-lane diffraction path layout

- Status: Accepted
- Date: 2026-07-20
- Decision ID: `source-lane-diffraction-path-layout`
- Scope: RayD Torch typed order-1 diffraction path export

## Context

The order-1 exporter launches one logical lane for every transmitter,
receiver, and diffraction-state tuple. Its historical output is compact: each
successful lane reserves a row through a device counter. That layout remains
appropriate for existing RayD consumers, but it does not preserve logical
source-lane identity for a downstream deterministic, state-ordered reduction.

Channel Native ADR-030 accepts a deterministic pair reduction whose input must
retain the source-state order without a device-to-host count, host compaction,
or a second full field workspace. RayD therefore needs an explicit storage
layout without creating a second traversal or UTD implementation.

## Decision

The stable typed integration API advances to numeric version `5`; the header,
identity, target, operation, and result type names remain unchanged.
`DiffractionPathConfig` gains `DiffractionPathLayout` with exactly two stable
values:

- `Compact`, the default for existing callers; and
- `SourceLane`, requested explicitly by same-graph consumers that require
  logical lane identity.

For `SourceLane`, with `R = rx_count` and `M = state_limit`, logical lane

```text
((tx * R + rx) * M) + state
```

writes only that output row. A rejected or inactive lane leaves the row in the
canonical inert representation initialized by the existing output initializer.
The contiguous CUDA `int32[1]` count is still incremented for each successful
path and remains the actual valid-row count. It is not a storage ordinal,
result shape, or numerical-order input in source-lane mode.

Both layouts execute the same existing Torch order-1 traversal, visibility,
stationary-point, UTD, and field-export body. Its layout-aware reservation
helper chooses either the historical warp-aggregated compact row or the launch
lane while the diagnostic count keeps the existing warp-aggregated integer
reservation. The generic shared algorithm has the equivalent layout choice and treats
parameter records without an `output_layout` field as `Compact`, preserving
the Dr.Jit and host-smoke contracts. The RayD dispatcher and Python surface
continue to request `Compact`; this decision adds no Python argument or
compatibility path.

The existing capacity rule remains `capacity >= tx_count * rx_count *
state_limit`. Invalid layout values fail before initialization or launch.
Validity must be checked before state payload, identifiers, or material data in
both layouts. Empty, all-inactive, poisoned-inactive, current-stream, dtype,
shape, device, ABI, and CUDA failures keep their established fail-loud
behavior. No host count read, synchronization, floating-point reduction,
physics change, or output-schema change is authorized.

## Consequences

Source-lane consumers can perform deterministic state-ordered work directly on
the exporter storage. They pay the already-declared full logical capacity and
may have a sparse validity mask. Existing compact consumers retain their
storage behavior and default source contract.

RayD owns only the generic layout choice and per-path export. Downstream pair
reduction, solver accumulation, capacity policy, and result assembly remain
outside RayD. A downstream consumer activates source-lane mode only after it
pins a reviewed RayD revision and removes its former compaction/reduction path
atomically.

## Acceptance gates

1. Direct typed tests prove default compact behavior and the exact source-lane
   row for sparse, multi-state input, including poisoned inactive rows.
2. Multi-transmitter/receiver tests prove the formula, and empty plus
   non-default-stream tests preserve their contracts.
3. Static host compilation proves parameter records without the new field keep
   compact behavior.
4. Valid compact rows retain the existing numerical output and count contract;
   source-lane valid rows contain the same per-path bytes at their fixed rows.
5. The stable boundary has numeric API version `5`, one identity, and no
   generation-suffixed alias or alternate dispatcher.

## Stop conditions

Stop if the implementation duplicates exporter physics, changes traversal or
UTD math, makes source-lane the Python default, reads inactive payload, uses the
count as a host-visible shape, adds synchronization or a host transfer, changes
compact output semantics, performs a floating-point reduction, or introduces a
generation-suffixed name or compatibility shim.
