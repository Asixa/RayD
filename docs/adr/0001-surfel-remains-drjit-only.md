# ADR-0001: Keep surfel support Dr.Jit-only

- Status: Accepted
- Date: 2026-07-11
- Decision ID: `F5-surfel-backend-scope`
- Scope: Phase F5 of `docs/archived/rayd_improvement_plan.md`

## Context

Phase F5 asks whether RayD should port the existing surfel subsystem to the
Torch backend. A port is not required for core backend parity: surfel is an
experimental extension, while intersection, edge queries, visibility, and
multipath primitives define the dual-backend core.

The repository provides strong evidence that the Dr.Jit surfel implementation
is real and maintained:

- 16 tracked Dr.Jit surfel implementation/example/test files contain 10,248
  lines and 437,721 bytes.
- The native implementation alone has six C++/CUDA headers or sources with
  3,178 lines and 152,923 bytes, plus a 72,068-byte embedded OptiX PTX header.
- The focused suite contains 57 tests across the native surfel suite, benchmark
  script contract, and multiview example contract.
- Ten surfel-focused commits cover tracing, hardening, appearance, examples,
  candidate-buffer performance, and IRGS-inspired reconstruction.

The same repository does not provide evidence for a Torch port. There is no
Torch surfel implementation, test, example, CMake source, or named Torch user
workflow. The only Torch surfel references are capability metadata declaring
the feature unavailable. This does not prove that no external user exists; it
means the evidence needed to justify a second backend implementation is absent.

## Reuse assessment

Some of the device layer is a plausible future shared core. `SurfelTraceParams`
is a raw-pointer POD and `surfel_trace.cu` depends only on OptiX plus that
contract. The analytic plane hit, Gaussian alpha, spherical-harmonic basis,
candidate-buffer ordering, and front-to-back compositing could be extracted
after their contracts are frozen.

That opportunity is not a low-cost Torch port today. Scene ownership, proxy
generation, OptiX GAS/pipeline ownership, Dr.Jit array materialization, fixed
candidate replay, forward/reverse AD, and nanobind exposure remain in the
Dr.Jit backend. The surfel headers and host sources contain about 189 explicit
Dr.Jit AD/type references. A Torch implementation would additionally need
Torch allocator and stream ownership, dispatcher schemas, eager and
`torch.compile` behavior, autograd/JVP replay, cold-create coverage, packaging,
and a second backend parity matrix.

## Decision

Keep surfel support Dr.Jit-only. Classify it as category `surfel` with
`experimental` stability. Publish the backend capability asymmetry explicitly:

- Dr.Jit: `surfel = true`
- Torch: `surfel = false`

Do not add Torch stubs that appear callable, do not copy the Dr.Jit host owner,
and do not move surfel into the shared directory merely to make the source tree
look symmetric.

Core edge and visibility parity remains higher priority. Those operations are
already part of both public backends and their grouped native validation must
not be displaced by an extension port without a confirmed Torch workflow.

## Consequences

Positive consequences:

- Torch wheels avoid another OptiX pipeline, embedded PTX payload, native host
  owner, dispatcher/autograd surface, and backend-specific tests.
- Dr.Jit users keep the existing implementation, examples, and AD behavior.
- Capability discovery reports the asymmetry instead of failing at import or
  presenting an incomplete compatibility shim.
- Future sharing can begin at the POD/device boundary without committing to a
  premature public Torch API.

Costs and limitations:

- Torch users cannot consume surfel tracing through RayD today.
- Surfel remains part of the Dr.Jit wheel and its build/test maintenance.
- Backend-neutral surfel records are deferred until a real cross-backend use
  case defines what must be exchanged.

## Reconsideration gates

Reopen this decision only when all of the following are satisfied:

1. A named Torch user or maintained repository workflow supplies an acceptance
   fixture and an owner for the requested surfel behavior.
2. A backend-neutral input/result/option contract is written before the port.
3. The params and useful device logic are extracted into a shared core used by
   both backends; a full copied CUDA/OptiX implementation is not acceptable.
4. Torch scene ownership, stream semantics, fixed-candidate reverse AD and JVP,
   eager behavior, and `torch.compile` behavior have an explicit design.
5. Forward and AD parity tests cover invalid, degenerate, candidate saturation,
   continuation, RGB/SH/features, normals, and dynamic geometry updates.
6. Incremental build time, wheel-size delta, cold pipeline creation, and runtime
   memory are measured for both backends and accepted in the release matrix.
7. The core edge/visibility grouped acceptance gates are passing.

## Minimal manifest merge

No manifest edit is required if the Phase F2 manifest keeps its current surfel
entries. If parallel work overwrites them, the main thread should restore only:

```json
{
  "apis.surfel.category": "surfel",
  "apis.surfel.stability": "experimental",
  "backends.drjit.capabilities.surfel": true,
  "backends.torch.capabilities.surfel": false
}
```

Runtime capability copies must mirror those four facts through the existing F2
generation/validation path; F5 introduces no new public capability key.
