# ADR-0031: Required diffraction path validity

- Status: Accepted
- Date: 2026-07-20
- Decision ID: `required-diffraction-path-validity`
- Scope: RayD Torch order-1 diffraction path export

## Context

ADR-0032 subsequently advances the stable numeric integration API to version
`5` for an explicit diffraction path storage layout without changing this
required-validity contract.

The order-1 exporter accepts fixed-capacity diffraction states selected on the
device. Its former optional `active` mask allowed an implicit all-valid path,
scalar broadcast, and strided masks. Those forms cannot safely represent an
inactive capacity suffix whose payload may be uninitialized or intentionally
poisoned.

## Decision

The stable typed integration API advances to numeric version `4`; the public
header, exact identity, CMake target, and operation name remain unchanged.
`DiffractionPathConfig.active` is a required `at::Tensor`. It must be a
contiguous CUDA boolean tensor on the scene device with exact shape
`[state_limit]`. A zero state limit requires a defined empty CUDA boolean
tensor.

The dispatcher schema and Python `Scene.trace_dfr_paths(...)` entry require the
same tensor. Omitted, `None`, scalar-broadcast, wider-than-limit, strided, CPU,
and cross-device masks fail before launch. There is no synthesized all-one or
defined-empty compatibility sentinel.

Every OptiX and shared CUDA path tests the state-validity element before reading
the state ID, geometric/RF payload, material ID, or poisonable row data.
Inactive states cannot append a path and leave the initialized fixed-capacity
output canonical and inert. Launches remain on the caller's current CUDA stream.

Diffraction accumulation, coherent accumulation, recursive states, sampling,
RNG, UTD physics, output layout, numerical order, launch topology, and valid-row
mathematics are unchanged.

## Consequences

Downstream fixed-capacity callers can pass selected state storage without host
compaction or a device-to-host count read. The numeric version change makes old
typed consumers fail at compile-time until they adopt required validity.

## Acceptance gates

1. Direct typed and dispatcher tests cover missing/`None`, shape, dtype,
   contiguity, CPU/cross-device, current-stream, poisoned invalid rows, and
   `state_limit == 0` with an empty mask.
2. Static governance proves the field and dispatcher schema are required and
   device helpers contain no implicit all-valid branch.
3. Valid rows preserve the existing direct integration and Torch backend
   numerical tests.
4. The stable header identity/name stay fixed and its normalized hash is pinned.

## Stop conditions

Stop if any export path reads state payload before validity, creates an all-one
mask, accepts an optional/broadcast/strided validity form, changes valid-row
physics or output order, changes accumulation/coherent contracts, adds a
synchronization, or introduces a compatibility dispatcher.
