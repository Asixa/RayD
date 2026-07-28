# RayD Torch Native Gap Analysis

> **Historical record, frozen 2026-06-12.** This document predates the 0.6.0
> dual-backend monorepo and the F1-F5 phases. It uses the pre-rename names
> "RayDN" / `raydn` for what is now `rayd.torch`, and "RayD" for what is now
> `rayd.drjit`. For current capability state use `backend_capabilities()` or
> `shared/contracts/public_api.json`; for what landed since, see `CHANGELOG.md`
> and `docs/archived/rayd_improvement_execution_20260711.md`. Surfel is not discussed
> below but is formally Dr.Jit-only by decision F5
> (`docs/adr/0001-surfel-remains-drjit-only.md`).

Status (2026-06-12): RayDN contains RayD-style OptiX PTX pipelines for scene
intersection, edge query, reflection, and diffraction paths, plus Torch
VJP/JVP kernels for the supported continuous outputs under the fixed-winner
contract. The same-script RayD/RayDN benchmark measures RayDN faster for
every covered operation on the grid-64/query-4096 static and dynamic shapes,
and the multipath standard benchmark measures RayDN faster for reflection
trace (65K/1M rays, 2/4 bounces) and diffraction export (65K/1M states) with
matching checksums. Far-from-surface point nearest-edge no longer traverses
the scene-diagonal OptiX tier; it uses the tightest tier plus an exact tiled
fallback scan (native grid-192/65,536-query nearest-edge: 230.8 ms -> 27.8 ms).
Release-size and Nsight-counter-backed runs remain the broader performance
gate; Nsight Compute counters still require an elevated session
(ERR_NVGPUCTRPERM).

Known remaining RayD-favored case: static public VJP through Torch eager
`.backward()`. After the Autograd-key registration and GIL-free dispatch of
the intersect AD ops (2026-06-12), RayDN measures 0.165-0.218 ms vs RayD
0.082-0.125 ms on the 16K/65K-ray stress shapes (gap narrowed from
~1.7-2.9x to ~1.4-2.1x). The residual cost is the PyTorch eager engine
floor itself: a trivial one-node `mul` backward with one AccumulateGrad
measures ~0.07-0.13 ms on this machine, which bounds any eager `.backward()`
implementation. The torch.compile path captures the static t-only VJP with
zero graph breaks for integration into larger compiled models, but does not
beat tuned eager on the isolated microbenchmark.

## Current Multipath Implementation

The current code should be treated as a source-port with active release-scale
performance validation remaining:

- `src/torch_ext/common/optix_pipeline.cpp` owns the shared OptiX launch
  pipeline/cache.
- `src/torch_ext/reflection/ops.cpp` owns reflection Torch/PyBind bindings.
- `src/torch_ext/diffraction/ops.cpp` owns diffraction Torch/PyBind bindings.
- `src/torch_ext/reflection/pipeline.cpp` and
  `src/torch_ext/diffraction/pipeline.cpp` own the reflection and diffraction
  PTX entry configurations separately.
- `src/torch_ext/reflection/{visibility_optix.cu,trace_optix.cu,epc_optix.cu,accum_optix.cu,dedup.cu,epc_field.cu}`
  contain the reflection-side native kernels/glue.
- `src/torch_ext/diffraction/{paths_optix.cu,accum_optix.cu,accum_ad.cu}`
  contain the diffraction path search, accumulation, and direct AD kernels.
- `CMakeLists.txt` builds scene, edge, reflection, and diffraction PTX targets.

## RayD Multipath Kernel Coverage

The corresponding Dr.Jit-backend source files now live under
`backends/drjit/src/multipath/` (they were at `src/multipath/` when this record
was written). Reflection and diffraction forward kernels have Torch source
ports, including:

- `backends/drjit/src/multipath/segment_visibility.cu`
- `backends/drjit/src/multipath/reflection_trace.cu`
- `backends/drjit/src/multipath/reflection_accumulation.cu`
- `backends/drjit/src/multipath/reflection_dedup.cu`
- `backends/drjit/src/multipath/reflection_epc.cu`
- `backends/drjit/src/multipath/reflection_epc_field.cu`
- `backends/drjit/src/multipath/diffraction_paths.cu`
- `backends/drjit/src/multipath/diffraction_accumulation.cu`
- `backends/drjit/src/multipath/diffraction_accumulation_ad.cu`

These RayD files should remain the parity source of truth when extending the
current ports.

## Performance Gate Status

Current status:

- Same-script, same-data, same-batch RayD vs RayDN performance comparison
  is implemented in `backends/torch/tests/benchmark_rayd_backends.py` (named
  `tests/benchmark_rayd_vs_raydn.py` when this record was written).
- Current corrected same-script results cover both static-vs-static and
  dynamic-vs-dynamic runs. RayDN is faster for scene build, `intersect`,
  point `nearest_edge`, reflection trace, and direct diffraction accumulation
  on the recorded grid-64/query-4096 benchmark.
- The earlier point `nearest_edge` regression was fixed by removing the
  measured-slower AoS query path, keeping RayD's SoA OptiX query layout, using a
  persistent OptiX params buffer, and adding a Torch no-AD forward path that
  skips autograd tape allocation/writes for non-AD callers.
- Remaining performance work should focus on release-size runs, Nsight
  confirmation of the hot kernels, and any larger workloads that stress
  reflection/diffraction accumulation atomics.

## Required Acceptance Gate

Before this work can be considered complete, RayDN needs:

1. Same-script RayD/RayDN performance runs for the release benchmark shapes.
2. Nsight-backed confirmation or accepted thresholds for the release workloads.
3. Full native and opt-in RayD parity test runs after any performance changes.
