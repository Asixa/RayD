# AGENTS.md

This repository contains **RayD**, a CUDA/OptiX package for differentiable ray geometry, edge queries, and multipath simulation primitives. Since 0.6.0 it is a dual-backend monorepo exposing two independent, backend-native APIs: `rayd.drjit` and `rayd.torch`.

## Environment

Default conda environment:

```bash
conda activate witwin3
```

## Build

The repository root is a meta-distribution and builds no native code. Build a backend explicitly:

```powershell
.\scripts\build_local.cmd -Backend drjit   # or: torch, all
```

## Architecture

- `backends/drjit/include/rayd/`, `backends/drjit/src/`: Dr.Jit backend C++/CUDA geometry, edge, and multipath kernels
- `backends/torch/include/rayd/torch/`, `backends/torch/src/`: Torch backend, dispatcher, and autograd bindings
- `shared/include/rayd/shared/`, `shared/src/`: backend-neutral contracts, math, edge BVH core, and accumulation kernels
- `shared/contracts/`: machine-readable public API and operation manifests
- `Scene`: mesh container plus OptiX acceleration structure
- `Mesh`: raw triangle mesh input, transforms, edge topology, secondary edge query data
- `Camera`: primary-ray sampling plus primary-edge preprocessing/sampling. **Torch backend only** (`backends/torch/python/rayd/torch/camera.py`); the Dr.Jit backend has no `Camera`
- `scene.intersect(ray)`: differentiable intersection query
- `scene.nearest_edge(point)` / `scene.nearest_edge(ray)`: scene-level nearest-edge query over a GPU BVH
- `scene.trace_reflections(...)`: specular reflection-path tracing; `symbolic=True` selects the bounce-level C++ path
- `scene.visible(...)`: batched segment visibility queries
- `scene.trace_refl_epc(...)` / `scene.trace_refl_epc_field(...)`: equivalent-path correction primitives for reflection paths
- `backends/drjit/src/multipath/`: multipath result types, OptiX launch wrappers, and CUDA/OptiX kernels
- `backends/drjit/python/rayd/drjit/`, `backends/torch/python/rayd/torch/`: the two backend Python packages; `rayd` itself is a PEP 420 namespace with no default backend

Public names follow `backends/drjit/API_NAMING_STANDARD.md`; `backends/drjit/API_RENAME.md` records the 2026-05-21 rename.

## Generic RF Scattering Ownership

ADR-0026 accepts six complete, solver-neutral scattering runtime families (17
typed operations) as RayD numerical ownership candidates. This does not add a
high-level BSDF/material framework: RayD consumes caller-owned resident table,
CDF/density, phase-screen, geometry, and material tensors, while Channel owns
their builders, lifecycle, seeds, topology, estimator, RNG/MIS, accumulation,
metadata, and public results.

- Declarations have one owner in
  `backends/torch/include/rayd/torch/rf/scattering.h`; shared table device math
  has one owner in `shared/include/rayd/shared/rf/scattering_table.cuh`.
- Move every family complete. Do not split primal from backward/JVP, copy a
  Channel implementation/header, include Channel private headers, or add a
  second Python extension/dispatcher.
- Preserve resident current-stream execution, fusion and launch count,
  reduction order, backward atomics, tape lifetime, row/output schemas,
  numerical order, and failure behavior. No CPU, Torch-expression,
  finite-difference, legacy-dispatch, zero-result, or detached-gradient
  fallback is allowed.
- Preserve family-specific AD: chain-ensemble geometry VJP requests fail
  loudly while its geometry JVP remains supported; chain-realization keeps its
  existing geometry VJP/JVP. Never silently broaden, detach, or zero an
  unsupported derivative.
- Preserve per-TU flags: table primal/sample/PDF uses default CUDA flags; the
  audited table-AD, ensemble, patch, and chain lockstep TUs use `--fmad=false`.
  Neither flag policy may spread across operation families.
- A RayD implementation is dormant until Channel pins the reviewed commit,
  validates it, switches every caller in a complete family, and deletes its
  local implementation atomically. Rollback changes the pin to a prior complete
  owner; it never adds runtime owner selection or fallback.

See `docs/adr/0026-generic-scattering-runtime-ownership.md` for the exact symbol
matrix, resource boundary, activation gates, and stop conditions.

## Stable Typed Integration Naming

The `rayd-torch` distribution owns the passive, relocatable same-graph source
bundle under `rayd/torch/_source`. Its metadata and complete per-file manifest
are packaging contracts: downstreams may locate them only through the active
Python distribution metadata and must pin and recompute their identity before
`add_subdirectory`. Do not add environment-prefix scans, execute package code
for discovery, export the native core as an unvalidated cross-build static
library, or omit source files from the manifest. Explicit source checkouts
remain higher-priority developer inputs and retain Git commit/remote/dirty
validation.

The same-graph Torch C++ boundary has one durable name:
`rayd/torch/integration.h`, with exact identity `rayd.torch.integration` and
numeric `kIntegrationApiVersion = 6`. Do not add an `integration_v2` forwarding
header, target alias, alternate identity, dispatcher, or compatibility shim.
Historical Phase 10B identity/hash evidence may retain its former label but is
not a live include path. See
`docs/adr/0028-stable-typed-integration-naming.md`.

Capacity-shaped row operations require a CUDA boolean validity tensor in the
top-level primal request. Kernels must test it before reading any row payload or
ID. Invalid primal/JVP rows and supported row gradients are bitwise zero, while
invalid rows contribute no shared-gradient atomics. Backward and JVP requests
inherit validity only through their nested primal request; an optional mask or
implicit all-valid path is forbidden.

The order-1 diffraction path exporter likewise requires
`DiffractionPathConfig.active` as a contiguous CUDA boolean tensor with exact
shape `[state_limit]`; `state_limit == 0` requires a defined empty tensor. Every
host/device export path must gate on it before reading state payload, and the
public Torch call has no omitted, `None`, scalar-broadcast, or strided-mask path.
Diffraction accumulation and coherent-accumulation configs retain their existing
contracts. See `docs/adr/0031-required-diffraction-path-validity.md`.

The typed order-1 diffraction exporter supports `Compact` and `SourceLane`
storage under ADR-0032. `Compact` remains the default for existing consumers.
`SourceLane` writes a successful logical lane only to
`((tx * rx_count + rx) * state_limit) + state`, leaves rejected lanes inert,
and retains the device count only as actual-count metadata. Both layouts share
the same traversal and UTD body; do not add a second exporter, host compaction,
or floating-point reduction to RayD. See
`docs/adr/0032-source-lane-diffraction-path-layout.md`.

The dormant ADR-0033 segment-penetration family uses the stable typed boundary
only. Every structurally active non-empty forward entry submits one OptiX
launch, with the ordered `D + 1` capacity probe inside each raygen lane.
`input_active_any=false` submits zero OptiX work and must validate the device
mask on the caller's stream. Results and tapes are fixed `[N,D]`, share the
caller's CUDA int32 `[1]` failure transaction and single assigned bit, and are
made completely inert after any failure; only the overflow diagnostic may be
retained. Backward/JVP consume frozen primitive, barycentric, restart, tie, and
denominator decisions and never retrace. Keep both policies, their exact
comparison/normal/epsilon expressions, and family-local non-FTZ/precise-divide/
precise-square-root compilation separate. Do not add a Python binding,
dispatcher, host count read, partial result, or fallback. See
`docs/adr/0033-batched-segment-penetration.md`.

## OptiX Pipeline Guardrail

- If a native multipath call fails with `OptiX error in optixPipelineCreate(multipath)`, treat it first as a multipath OptiX pipeline configuration issue, not as an input/API issue.
- The verified 2026-05-26 fix keeps scene/edge OptiX production flags separate from multipath flags: multipath uses production module optimization plus `RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=11`.
- Trace-call count and instruction count are useful diagnostics, but they are not proof of root cause; do not split reflection tracing or add fallback launches unless tests prove the pipeline shape itself is the failing variable.
- Always verify in a fresh subprocess with the actually loaded conda `.pyd`, and run `backends.drjit.tests.drjit.test_optix_pipeline_cold_create` for public API cold-create coverage.
- See `docs/optix_pipeline_create_failures.md` for the root-cause writeup and regression checklist.

## Committed PTX Source Identity

The Dr.Jit backend commits its eight generated OptiX `*_ptx.h` headers, and
PTX regeneration is opt-in and OFF by default, so editing a `.cu` file or any
header it reaches silently leaves the committed PTX describing older device
code. `backends/drjit/ptx_sources.json` records each PTX module's transitive
in-repository include closure and content digests, and
`tests/test_ptx_source_digest.py` recomputes the record on every run. Check
that record before editing any file it lists under `modules.*.sources`. A
drifted digest is repaired only by actually regenerating the affected PTX,
copying the regenerated header over the committed one, and re-running
`python backends/drjit/scripts/audit_ptx_sources.py --write`; use `--check` to
diagnose, and `--mark-verified <module>` only after byte-comparing the
regenerated header against the committed one. The record states source
identity, never correctness, and `--write` without a real regeneration only
falsifies it.

## CUDA Compile-Flag Policy

`shared/contracts/compile_policy.json` declares the per-translation-unit CUDA
numeric flag assignment for both backends over the four closed profiles
`nvcc_default`, `fast_math`, `no_fmad`, and `precise_no_ftz`;
`tests/test_compile_flag_policy_contract.py` re-derives the assignment from
the CMake sources and fails when declaration and build disagree in either
direction. The assignment is frozen by ADR-0035: changing any numeric flag,
moving a unit between profiles, adding a profile, aligning a frozen
divergence, or introducing a global or target-wide CUDA numeric flag is an
ADR-level decision that needs its own accepted record with numerical and
generated-code evidence. Editing the contract to match a flag change is not a
fix; it is the drift the test exists to catch.

## Edge BVH Status

Current edge-query acceleration design:

- triangle intersections still use OptiX through `scene.intersect(ray)`
- nearest-edge queries default to the OptiX custom-AABB edge backend; a custom Dr.Jit/CUDA BVH over scene-global `SecondaryEdgeInfo` is available as an alternate backend (`edge_bvh_backend="drjit"`)
- the Dr.Jit backend build path is:
  - GPU LBVH construction with Morton codes
  - host-side treelet optimization for large-scene query quality
  - GPU traversal for point and ray closest-edge queries
- `nearest_edge(ray)` uses segment semantics on `[0, tmax]` when `tmax` is finite
- the broad phase is detached; after the winning edge is found, the implementation re-gathers AD geometry and recomputes the exact result so gradients remain available while the winning edge stays fixed
- dynamic vertex and transform updates use BVH refit after `sync()`

Current status notes:

- the treelet path is the best current tradeoff for actual nearest-edge query throughput
- GPU treelet optimization is enabled for the verified `65,536..500,000`-primitive range; larger builds retain the valid pure LBVH topology until a dedicated large-scene primitive-coverage gate raises that bound
- the combined public backend is named `optix_drjit`; `hybrid` is a deprecated compatibility alias and is unrelated to the removed HLBVH experiment
- the former `LBVH + top-level SAH` HLBVH experiment was removed after it made large-scene queries much slower; its historical measurements are retained below
- the dead GPU-prepared flat-treelet prototype was removed; the supported treelet path keeps its host-prepared schedule and launches GPU treelet optimization kernels
- the shared edge core owns the backend-neutral OptiX AABB kernel, LBVH/treelet build stages, dirty-ancestor/dirty-level/refit launchers, compact-BVH CUDA traversal, and exact-distance launchers; every API takes raw pointers, caller-owned buffers, and an explicit stream
- Dr.Jit still owns CUB/allocation, LBVH/treelet host orchestration, host compaction, and its JIT traversal; Torch persistent compact-BVH ownership and public top-k/visibility integration landed in F1 (`backends/torch/src/torch_ext/scene/scene_cache.cpp`, `tests/test_f1_torch_global_geometry_contract.py`)

Supported custom-BVH configuration after BVH-3 convergence:

- the product build pipeline is `LBVH + gpu_treelet + overlap + atomic + flat host-prepared levels + host_upload_raw + scalar_arrays` within the verified treelet range, with an automatic pure-LBVH correctness guard above `500,000` primitives
- explicit `post_build_strategy=none` is retained only as a benchmark/reference pure-LBVH baseline, not as a public product strategy; the automatic large-scene guard is internal and backend-consistent
- serial build remains a deterministic debug mode only; it has no public performance commitment
- refit keeps `Auto`, `Full`, and `DirtyAncestors`; `Auto` is the product default while the explicit modes remain calibration and debug controls
- `PerLevelUploads`, `LevelByLevel`, `HostUploadExact`, `GpuEmit`, and `Packed` were removed from the supported configuration surface after their measured benefits failed the BVH-3 Pareto thresholds
- the quantified one-factor comparisons, including the legacy sparse-mask caveat for exact compaction, are preserved in `shared/benchmarks/baselines/bvh3_configuration_convergence_20260711.json`

Performance snapshot used for the current decision, measured on the verified Windows machine in this repository (`RTX 5080`, `Ryzen 7 9800X3D`) with a `192x192` grid mesh, `110,976` edges, and `65,536` batched queries:

| Path | `build()` | point query | finite ray query | infinite ray query | `sync()` |
| --- | ---: | ---: | ---: | ---: | ---: |
| default treelet path | `138.43 ms` | `9.99 ms avg` | `14.34 ms avg` | `15.43 ms avg` | `3.69 ms` |
| removed HLBVH top-level SAH experiment (historical) | `13.45 ms` | `102.91 ms avg` | `136.59 ms avg` | `143.16 ms avg` | `3.09 ms` |

RayD is not a full renderer and intentionally does not include:

- a high-level BSDF or material-model framework
- emitters
- integrators
- scene loaders
- bitmap/image I/O
- a material-light-integrator framework

This exclusion does not prohibit low-level, solver-neutral RF scattering
table/sample/integral primitives governed by ADR-0026.

The dormant typed axial-edge visibility candidate governed by ADR-0029 owns
only its exact four-sample OptiX numerical primitive. Keep its Params/PTX/
pipeline separate from legacy segment visibility, inherit the legacy OptiX
compile policy for traversal, and lock only point construction with inline PTX
round-to-nearest add/subtract/multiply instructions without FTZ or FMA. Do not
bind it to Python or the legacy dispatcher. Activation requires an atomic
downstream pin, switch, parity proof, and deletion of the former numerical
owner.

## Tests

```bash
python -m unittest backends.drjit.tests.drjit.test_geometry -v
```
