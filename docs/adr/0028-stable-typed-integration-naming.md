# ADR-0028: Stable typed integration naming

- Status: Accepted
- Date: 2026-07-20
- Decision ID: `stable-typed-integration-naming`
- Scope: RayD Torch same-graph native source integration

## Context

RayD's typed C++ integration surface reached its accepted production operation
set while its file, internal helper, direct-test target, and identity still
carried the migration label `v2`. The number was useful during construction but
is not a capability name and would make every later operation addition look like
a new parallel integration generation.

The numeric API version remains useful for compile-time compatibility checks.
Mathematical names such as `vertex_v2`, serialized schema version `2`, and
immutable historical evidence are unrelated and must not be renamed.

## Decision

The sole current typed source-integration boundary is:

- public header: `rayd/torch/integration.h`;
- internal access header: `integration_internal.h`;
- direct test source: `integration_test.cpp`;
- CMake executable and CTest names: `rayd_torch_integration_test` and
  `rayd_torch_integration`;
- exact header identity: `rayd.torch.integration`;
- current numeric compile-time version: `kIntegrationApiVersion = 6` under
  ADR-0033. The stable name and identity are unchanged.

There is no forwarding `integration_v2.h`, internal alias, old CMake target,
alternate identity, runtime selector, or compatibility shim. Same-graph native
consumers include the stable header directly and still link
`rayd_torch_native_core`. This rename does not add a Python surface, dispatcher,
dynamic lookup, or stable cross-LibTorch binary ABI.

The normalized-LF SHA-256 of the accepted stable header after ADR-0033 is
`57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`.

## Historical evidence

Accepted ADR-0026 and Phase 10B records retain the former long-form identity,
`integration_v2.h` path, and hash as immutable evidence of the dormant candidate
that Channel reviewed and activated. Those records are not current aliases or
valid include paths. Current governance checks pin this ADR's stable header.

## Consequences

The source boundary now has one durable name independent of capability growth,
while callers can continue to reject an incompatible numeric API version at
compile time. Downstream pins must change path, identity, and normalized header
hash atomically. Rollback changes the locked RayD revision; it never restores a
second live integration name.

## Acceptance gates

1. No production, build, or test path contains `integration_v2`, the former
   long-form identity, or a `v2` operation/family label.
2. `integration.h` is the only typed aggregate header, exposes the same request,
   result, RAII scene, RF, AD, stream, and failure contracts, and retains numeric
   current API version `6`.
3. The stable CMake target builds and the stable CTest direct contract passes.
4. ABI governance and ownership tests pin the stable path, identity, and header
   bytes; no legacy extern-C or Python integration surface reappears.
5. Historical evidence remains clearly marked and unchanged.

## Stop conditions

Stop if the rename changes an operation signature, numerical behavior, launch
or synchronization boundary, exception behavior, ownership, or packaged Python
ABI; requires a forwarding alias; or cannot be activated atomically by a locked
same-graph consumer.
