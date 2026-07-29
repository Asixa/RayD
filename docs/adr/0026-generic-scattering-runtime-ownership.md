# ADR-0026: Generic scattering runtime ownership

- Status: Accepted; source path and ownership-namespace clauses superseded by ADR-0039 and ADR-0040
- Date: 2026-07-19
- Decision ID: `generic-scattering-runtime-ownership`
- Scope: Channel Native direct-RayD integration, phases 9 and 10

> **ADR-0039/0040 supersession.** The `rayd/{shared,torch}/rf/` paths and generic
> `rf` ownership namespace below are historical. Canonical scattering owners
> are `include/rayd/detail/scattering_table.cuh`, `include/rayd/scattering.h`, and `src/scattering/`. The 17
> operation contracts, compile profiles, fusion, stream, derivative, failure,
> activation, rollback, and downstream atomic-switch clauses remain in force.
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
| chain ensemble | `scattering_chain_ensemble_eval`, `scattering_chain_ensemble_eval_backward`, `scattering_chain_ensemble_eval_jvp` | 3 |
| chain realization | `scattering_chain_realization_eval`, `scattering_chain_realization_eval_backward`, `scattering_chain_realization_eval_jvp` | 3 |

Primal, backward/VJP, and JVP companions move together where they exist. Table
sampling and PDF evaluation are the complete two-operation sampling family;
this ADR does not invent derivative entries for discrete sampling decisions.
`scattering_event_probabilities` is estimator/event policy and does not move.

### Unique typed and device-header owners

The seventeen declarations belong in the single public source-level typed
header `backends/torch/include/rayd/torch/rf/scattering.h`, included by
`rayd/torch/integration.h`. Requests and named results use `at::Tensor`,
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

The repository's Phase 10A implementation supplied the first eleven entries:
table evaluation/backward/JVP, table sample/PDF, single-bounce ensemble
primal/backward/JVP, and patch-integral primal/backward/JVP. Channel activated
that complete wave through its atomic pin/switch/delete commit. Phase 10B adds
the remaining six chain operations as a compiled and direct-tested dormant
candidate with no Python binding. Channel remains the production numerical
owner of both chain families until it pins this exact revision and atomically
switches and deletes all four local chain translation units. No event-policy
entry moves in either wave.

The exact dormant-candidate identity and hashes below are historical Phase 10B
activation evidence. They intentionally retain the then-current `integration_v2`
name and bytes; ADR-0028 governs the current stable boundary and pin.

The exact dormant candidate identity is
`rayd.torch.integration.v2.20260719.rf-transmission-sequence.pure-wedge-diffraction.scattering-table-single-bounce.scattering-chains`.
The source-level pin records are SHA-256 over source bytes after CRLF and CR
line endings are normalized to LF:
`ac95c418860d109aeaa96623131592e4df8887992e5fc25ecab71b4ddbf1f55b`
for `rayd/torch/rf/scattering.h`,
`38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`
for `rayd/detail/rf/scattering_table.cuh`, and
`0608bfbaf022379bc03442f9baa777ec05cfe3f6ab9b964e2385ec12a7b6c654`
for `rayd/torch/integration_v2.h`. The shared chain-AD host/device helper pin is
`2551c33533dc7ea0a0c1680d67e5432587f8c2f77833d5a717fcb2d20597b507`
for `scattering_chain_ad_common.cuh`, and the shared static typed-contract
validator pin is
`4f61082059d08112d675613e2e0ff0d8b7489753ffb96aec152aa17ac2409b73`
for `scattering_chain_checks.h`. The ten implementation-source pins are
`72fb84a4158652a70c5f4f17e5d1ce61371773cdd54db6835148ee065e474c50`
for `scattering.cu`,
`e09cb3992737b028222e205318baea1aa070d300f0126def9759edaa17ad5b7c`
for `scattering_table_eval_ad.cu`,
`be38ff966dd06afe3f1df46d2eb16094c97111c76534e22d5f3fec6685f1f1fc`
for `scattering_ensemble.cu`,
`8c094b3a6542b1da26e662e38c405ec1d90cf53aaf8934147b0549f66a8fb0ea`
for `scattering_ensemble_ad.cu`,
`e1d8555874a1832067e92e9f1973cee38d9ce2f18dac230b56bb1c6504c0c08b`
for `scattering_patch_integral.cu`, and
`0d3bffe34ecd22656f1c5bdb10a6fe903ad059803547e29ccb95f5fd390858aa`
for `scattering_patch_integral_ad.cu`,
`6293c9238fa5c251d23408493fffd0b88cc557f50de84c90519ec1115ca7d9fd`
for `scattering_chain_ensemble.cu`,
`a207dbf58b62286b8a58d7f22535900b198f187c7d0bffb2bacce728eaae306e`
for `scattering_chain_ensemble_ad.cu`,
`be9601740ad1dce283708446ebc596b5fd5aca1da8f12421cc077d0dac99d424`
for `scattering_chain_realization.cu`, and
`970c579cc9d0c384d28e7aaa8f32200800a1de159de9a0338b2f0bad75f7fa93`
for `scattering_chain_realization_ad.cu`. Governance tests recompute these
hashes and reject pybind/map shims, event/solver scope leaks, dynamic lookup,
explicit synchronization, and accidental fast-math so a source-contract
change requires an intentional identity and pin update.

ADR-0030 advances the live typed capacity-row contract to API version 3 while
preserving the operation families and numerical owners above. Its normalized-LF
successor pins are
`7a29ff216f11a08256ee271ef5dcad817e4b8379d88bc07772685fa3da439aa9`
for `scattering.h`,
`aae7d33ae78d8886c8d1d1a665336e9027d35a3b2ba180a85b7211cd3ee22e21`
for `scattering.cu`,
`fe2046c2a3ba45bb073cb43272a89282593c0c2a1659f4cc7e85d2ebd5335039`
for `scattering_table_eval_ad.cu`,
`ed6f9225e1d987b8624062dafce617755a793d5646f8866bc6106637e8c4d492`
for `scattering_ensemble.cu`,
`3951fef2cac6759c05b57167a4491cdf421f575843ca33a9e1d761c713237573`
for `scattering_ensemble_ad.cu`,
`a37459f03879199cb0365a20b0cc06fca5fc24369a7efda858e189d11822af33`
for `scattering_patch_integral.cu`,
`bf40adb74e029a520162363383ca16c39e4829ee9e0ff3252816b0bcf04bff82`
for `scattering_patch_integral_ad.cu`,
`f848b268bbca8835ac091bc49f223d0f64532925361090bb1409c93d1d50278c`
for `scattering_chain_checks.h`,
`1121a2f276d982bb2bf6efe3e20aa0d82eb7251e224b7efea8c1099c92e9afe7`
for `scattering_chain_ensemble.cu`,
`554a6ad5cdfd1aac37913e3526e8c5d252ec514db6992eafd7d43882a14956bc`
for `scattering_chain_ensemble_ad.cu`,
`e61dd957af9a2fdc9a5035040c874a12b3834a00cb8285b64e921f1d01cb72b3`
for `scattering_chain_realization.cu`, and
`63cf0704157d591307eb788b9de08c22aadd919177f8545e8cb4f8b037bb27bd`
for `scattering_chain_realization_ad.cu`. The unchanged shared table and chain-AD
helper pins remain those recorded above. Historical Phase 10B hashes remain
evidence only; governance pins the live API version 3 successors.

Multi-GPU Phase 0 device-correctness hardening re-pins the eleven affected
implementation sources. That change adds exactly one `c10::cuda::CUDAGuard` per
RF host entry, ahead of output allocation and launch, so the ambient device
matches the device whose current stream the entry already selected. It changes
no operation family, declaration, kernel body, launch count, fusion, per-TU
numeric flag, reduction order, backward atomic, AD support matrix, row schema,
or failure behavior, and it is numerically inert for every request whose
tensors already shared one device index. The live successor pins are
`061f41fe99435a60eb2afd5763f7422ccba800595e126963f2efe81d599569dd`
for `scattering.cu`,
`e96a4a0229d626a6ad55cacdbf71a16a48c438b248c18442b2d63a7a1850d60c`
for `scattering_table_eval_ad.cu`,
`e77f5a3888186ef675ba88516fa059fb2d252db6bb1420099e8b37614637d544`
for `scattering_ensemble.cu`,
`89f50f631233775d10bf33719482ec06ad16861bae7d9696d2d793fbf934910b`
for `scattering_ensemble_ad.cu`,
`61a9e2e86854880bd60ab35c77bc3d0308c07c3c61f560f4cce4f05b109a874c`
for `scattering_patch_integral.cu`,
`f5db3d5f93efe38273e28c9dad548da56cbccfc43a53f634064cc592545bfb1b`
for `scattering_patch_integral_ad.cu`,
`529e8777750c26cef2aed691a8799dda1f5035af02fdaa0a71725cf8584044ac`
for `scattering_chain_ad_common.cuh`,
`28e520b86ed622ab65509e2d8fa46a1f5f04c7cdfe64f79943fcd805adddb545`
for `scattering_chain_ensemble.cu`,
`49afe510215b5251ce4d220712f96f0b876a529401306683bb7439ade031c01f`
for `scattering_chain_ensemble_ad.cu`,
`8b41199b7e3f8c796bf933de5d8aa43432df2fcce2cfbf19764e5292f763733d`
for `scattering_chain_realization.cu`, and
`55db93ec294f91b3355876eedf6089170f49fad43f1608197e848bd53ce17eb5`
for `scattering_chain_realization_ad.cu`. The `scattering.h`,
`scattering_table.cuh`, and `scattering_chain_checks.h` pins are unchanged.

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
