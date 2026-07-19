# ADR-0026: Generic scattering runtime ownership

- Status: Accepted
- Date: 2026-07-19
- Decision ID: `generic-scattering-runtime-ownership`
- Scope: Channel Native direct-RayD integration, phases 9 and 10

## Context

Channel Native currently owns seventeen CUDA bindings that evaluate or sample
resident RF scattering data. The operations are solver-neutral numerical
primitives, but their table and phase-screen resources, topology, estimator
policy, random-number generation, accumulation, and result assembly belong to
Channel. Keeping the numerical implementations downstream would also make the
RayD-owned transmission and RF device-math closure depend on private Channel
headers or require a copied implementation.

RayD's exclusion of a high-level BSDF/material/light/integrator framework does
not exclude low-level, solver-neutral RF scattering primitives. This decision
adds no renderer, material model, scene loader, solver, or Python API. It
accepts a source-linked Torch C++ ownership boundary for six complete operation
families while preserving Channel's public facades and resource policy.

## Decision

### Complete operation families

After activation, RayD is the sole numerical implementation owner of exactly
these seventeen typed operations in six families:

| Family | Complete typed operations | Count |
| --- | --- | ---: |
| resident table evaluation AD | `scattering_table_eval`, `scattering_table_eval_backward`, `scattering_table_eval_jvp` | 3 |
| resident table sampling | `scattering_table_sample`, `scattering_table_pdf` | 2 |
| single-bounce ensemble | `scattering_ensemble_eval`, `scattering_ensemble_eval_backward`, `scattering_ensemble_eval_jvp` | 3 |
| phase-screen patch integral | `scattering_patch_integral_eval`, `scattering_patch_integral_eval_backward`, `scattering_patch_integral_eval_jvp` | 3 |
| v2 chain ensemble | `scattering_chain_ensemble_eval`, `scattering_chain_ensemble_eval_backward`, `scattering_chain_ensemble_eval_jvp` | 3 |
| v2 chain realization | `scattering_chain_realization_eval`, `scattering_chain_realization_eval_backward`, `scattering_chain_realization_eval_jvp` | 3 |

Primal, backward/VJP, and JVP companions move together where they exist. Table
sampling and PDF evaluation are the complete two-operation sampling family;
this ADR does not invent derivative entries for discrete sampling decisions.
`scattering_event_probabilities` is estimator/event policy and does not move.

### Unique typed and device-header owners

The seventeen declarations belong in the single public source-level typed
header `backends/torch/include/rayd/torch/rf/scattering.h`, included by
`rayd/torch/integration_v2.h`. Requests and named results use `at::Tensor`,
`std::optional<at::Tensor>` where applicable, requested-gradient flags, and the
caller-current Torch CUDA stream. This is a same-CMake/LibTorch-graph C++
surface, not a cross-build stable binary ABI and not a second Python extension.

Shared table interpolation/device math, including the current seven-helper
closure, has one source owner in
`shared/include/rayd/shared/rf/scattering_table.cuh`. Torch-specific operation
and AD helpers remain under `rayd/torch/rf/`. RayD must not include Channel
private headers. After activation Channel must not retain copied declarations,
device helpers, or CUDA implementations; its `_channel_native` bindings and
domain facades call the typed RayD owner directly.

### Resident tensor resource boundary

RayD consumes validated, device-resident tensors supplied by Channel:

- table values and sampling density/CDF tensors for table evaluation and
  sampling;
- phase-screen height tensors and patch/quadrature tensors for realization
  integrals;
- row-aligned chain geometry, material CSR tensors, table metadata, and field
  inputs for the two row-fused chain families.

RayD does not build, version, validate semantically, cache, seed, assign, or
own the lifetime of `ScatteringTableRuntime`, Kirchhoff table builders, or
`PhaseScreenRuntime`. Channel retains table construction and build AD, cache
and version policy, phase-screen generation and structure assignment, seeds and
reproducibility, and its typed resource facades. Tensors remain resident across
the operation; the boundary adds no scalar extraction, host iteration, host
synchronization, or avoidable host/device copy.

### Family-specific automatic differentiation

The migration preserves the implemented AD surface rather than normalizing it:

- table evaluation, single-bounce ensemble, patch integral, and both chain
  families retain their complete existing primal/backward/JVP contracts;
- structural and discrete inputs remain fixed and any unsupported requested
  derivative fails loudly rather than detaching or returning zeros;
- chain-ensemble reverse mode supports the existing material, table,
  coefficient, and frequency gradients, but a VJP request for continuous chain
  geometry fails loudly; its JVP continues to support those geometry tangents;
- chain-realization backward/VJP and JVP continue to support their existing
  live continuous geometry, phase-screen height, chain material, layer, `k0`,
  and frequency inputs; fixed patch mesh, quadrature, row selection, endpoints,
  depths, and other discrete/structural inputs remain non-differentiable.

The ensemble restriction must not be generalized to realization, and the
realization support must not be used to silently broaden ensemble reverse mode.
Production finite differences or Torch reconstruction are forbidden.

### Fusion, launch, tape, and atomic contract

This is an ownership move with unchanged numerical behavior:

- single-bounce operations retain their existing launch granularity and
  evaluation/reduction order;
- chain ensemble remains one complete row-fused
  `C1 transport -> diffuse scatter -> C2 transport -> receiver projection`
  operation with `Dmax=8`; it is not split into per-leg or per-stage launches;
- chain realization remains complete chain transport plus patch integral and
  coherent output formation; no patch or chain intermediate crosses a new
  launch boundary;
- primal/JVP outputs and reductions retain deterministic order; backward
  shared gradients retain the existing `atomicAdd` targets and order;
- backward/JVP recomputation and existing saved-input/autograd state retain
  their current tape lifetime; no persistent native tape or materialized
  inter-launch intermediate is added;
- output schema, row identity/order, aliasing, stride, dtype, device, gradient
  state, empty-input behavior, default-off behavior, phase/Jones conventions,
  CSR interpretation, weights, and exception behavior remain unchanged;
- solver RNG consumption, visibility, topology, PDFs, MIS, event decisions,
  accumulation, metadata, and result schemas remain unchanged and Channel-owned.

Changing fusion, launches, synchronization, tape lifetime, atomics, reduction
order, schemas, or numerical order requires a separate accepted numerical ADR.

### Translation-unit compile policy

Compile flags remain family- and translation-unit-specific:

- the table primal/sample/PDF implementation corresponding to
  `scattering.cu` uses the RayD target's default CUDA flags and must not gain
  `--fmad=false`;
- the implementations corresponding to `scattering_table_eval_ad.cu`,
  `scattering_ensemble.cu`, `scattering_ensemble_ad.cu`,
  `scattering_patch_integral.cu`, `scattering_patch_integral_ad.cu`,
  `scattering_chain_ensemble.cu`, `scattering_chain_ensemble_ad.cu`,
  `scattering_chain_realization.cu`, and
  `scattering_chain_realization_ad.cu` retain `--fmad=false`;
- neither policy may spread to transmission, pure-wedge, coupled diffraction,
  or another operation family.

Activation evidence must compare the effective NVCC command, generated code,
register/shared-memory use, and numerical outputs for each translation unit.

### Channel-owned policy and resources

Channel remains the sole owner of:

- event probabilities, MC/BDPT event selection, continuation, NEE, PDFs, MIS,
  RNG, and estimator policy;
- Kirchhoff table construction/build AD, cache/version validation, and its
  independent CPU/NumPy test oracle;
- scattering-table and phase-screen runtime lifecycles;
- rough-reflection `C_r` composition;
- chain discovery, join, row budgets, C1/C2 packing, topology, visibility, and
  winners;
- coherent combination, deterministic/Monte Carlo accumulation, solver
  orchestration, public results, capabilities, and metadata.

Channel-owned fused operations may include public RayD RF device headers in
place. They are not split into extra kernels merely to mirror repository
layout. This ADR does not move BDPT state, Monte Carlo Basic's estimator, a
material model, or a high-level BSDF framework into RayD.

### Dormant candidate and atomic activation

Acceptance does not itself move code or change production dispatch. Activation
is ordered by complete family:

1. RayD merges a complete, direct-tested typed candidate and unique source/header
   closure. The candidate is dormant: Channel has not pinned or called it and
   remains the authoritative production numerical owner.
2. Channel pins the reviewed RayD commit and header hash, switches every caller,
   proves direct-contract, AD, CUDA, end-to-end, numerical, launch, and resource
   parity, and deletes the corresponding local `.cu/.cuh` implementation in the
   same activation commit.
3. RayD then becomes the sole production numerical owner. Dormancy ends; no
   production build compiles two copies and no compatibility shim remains.

The eleven table/sampling/single-bounce/patch operations may activate before
the six chain operations, but each row of the family table above is atomic.
Primal and AD companions are never split between repositories.

Rollback changes Channel's lock to the prior complete, accepted RayD commit.
It never selects an owner at runtime and never adds CPU, Torch-expression,
finite-difference, legacy-dispatch, reduced-algorithm, zero-result, or detached-
gradient fallback. A dormant candidate may be removed before activation; after
activation rollback still follows the lock and complete-owner rule.

## Consequences

Positive consequences:

- generic RF scattering evaluation joins RayD's source-linked, solver-neutral
  runtime primitive surface;
- all seventeen operations have one implementation owner while Channel keeps
  resource, estimator, and public-domain ownership;
- resident tensors, fusion, AD asymmetry, compile flags, atomics, and numerical
  order become explicit cross-repository release contracts.

Costs and limitations:

- the repositories require two ordered RayD-candidate/Channel-activation waves;
- the typed header and shared device header become source-level compatibility
  contracts for locked consumers;
- this decision adds no Python API, stable cross-DSO ABI, table builder, solver,
  renderer framework, new derivative, optimization, or numerical change.

## Acceptance gates

An ownership wave is complete only when both repositories record:

1. exact or frozen-baseline parity for all activated forward outputs, table
   boundaries/normalization, ensemble energy/reciprocity/Jones basis, patch
   phase convention, and chain depth `0/1/8` cases;
2. backward/JVP lockstep, adjoint dot-product checks, test-only finite-difference
   oracles, loud fixed-input rejection, chain-ensemble geometry-VJP rejection,
   and chain-realization geometry-VJP/JVP coverage;
3. unchanged row fusion, launch count, synchronization, memcpy, resident/tape
   memory, deterministic reductions, atomics, output schemas, and caller-current
   stream behavior;
4. per-translation-unit default/`--fmad=false` command, generated-code, and
   resource parity;
5. no Channel private duplicate and no RayD dependency on Channel headers,
   runtime objects, topology, estimator, RNG/MIS, or result policy;
6. direct typed invalid-shape/dtype/device/optional-input/empty-input,
   ABI/capability/CUDA-failure, missing-symbol, and no-fallback tests;
7. representative Path, Deterministic, Monte Carlo Basic, and BDPT end-to-end
   parity, including unchanged default-off behavior and stochastic quantities;
8. clean locked checkouts, a pinned RayD commit and typed-header hash, and
   manifests/inventories identifying RayD as implementation owner and Channel
   as binding/resource/policy owner.

## Stop conditions

The migration stops and leaves the current complete owner intact if:

- any family is missing a required primal or AD companion, or a production
  build would contain duplicate implementations;
- parity needs a fallback, copied physics, compatibility alias, detached
  gradient, host transfer/sync, new launch, persistent tape, or materialized
  intermediate;
- a chain is split, `Dmax`, row/tape/output semantics, reduction order, atomics,
  RNG consumption, or any solver-owned decision changes;
- ensemble geometry reverse mode no longer rejects loudly, realization geometry
  VJP/JVP regresses, or an unsupported derivative silently succeeds;
- effective compile flags or generated code drift without accepted numerical
  evidence, including default flags gaining `--fmad=false` or that flag escaping
  the enumerated lockstep translation units;
- RayD acquires table/phase-screen lifecycle, Channel topology/estimator policy,
  a high-level BSDF framework, or a dependency on Channel private headers;
- the candidate, pin, direct tests, end-to-end caller, clean-build identity, or
  deletion evidence is incomplete.

Any numerical, AD-surface, fusion-boundary, resource-lifecycle, or solver-policy
change requires its own accepted ADR before implementation.
