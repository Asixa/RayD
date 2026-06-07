# RayDTorch Native Gap Analysis

Status: RayDTorch is still not a complete Torch-native port of every RayD
multipath/diffraction derivative and performance path, but the native extension
now contains RayD-style OptiX PTX pipelines for scene intersection, edge query,
reflection, and diffraction forward paths. Full completion still requires the
remaining VJP/JVP parity work and same-script performance acceptance.

## Current Multipath Implementation

The current code should be treated as an in-progress source-port, not completed
RayD multipath parity:

- `src/torch_ext/common/optix_pipeline.cpp` owns the shared OptiX launch
  pipeline/cache.
- `src/torch_ext/common/ops_multipath.cpp` currently owns the cross-domain
  Torch/PyBind multipath bindings while the reflection/diffraction host binding
  code is being split further.
- `src/torch_ext/reflection/pipeline.cpp` and
  `src/torch_ext/diffraction/pipeline.cpp` own the reflection and diffraction
  PTX entry configurations separately.
- `src/torch_ext/reflection/{visibility_optix.cu,trace_optix.cu,epc_optix.cu,accum_optix.cu,dedup.cu,epc_field.cu}`
  contain the reflection-side native kernels/glue.
- `src/torch_ext/diffraction/{paths_optix.cu,accum_optix.cu,accum_ad.cu}`
  contain the diffraction path search, accumulation, and direct AD kernels.
- `CMakeLists.txt` builds scene, edge, reflection, and diffraction PTX targets.

## Missing RayD Multipath Kernel Coverage

The corresponding RayD source files exist in `E:\Code\RayDi`. Reflection and
diffraction forward kernels now have RayDTorch source ports. The remaining gap
is completion-quality derivative/performance parity, not the absence of
diffraction PTX targets:

- `E:\Code\RayDi\src\multipath\diffraction_paths.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation_ad.cu`

These RayD files should remain the parity source of truth when extending the
current ports beyond the covered forward/direct-AD cases.

## Not Yet Completion-Quality

These areas are still open and must not be counted as complete RayD parity:

- `intersect` VJP/JVP parity against RayD.
- ray `nearest_edge` forward support and VJP/JVP parity against RayD.
- reflection VJP/JVP parity against RayD.
- EPC forward/VJP/JVP parity against RayD.
- diffraction forward/VJP/JVP parity against RayD.
- Same-script, same-data, same-batch RayD vs RayDTorch performance comparisons.
- Same-script RayD/RayDTorch nearest-edge performance acceptance. The previous
  RayDTorch-only nearest-edge regression from an overly large AABB/search radius
  has been reduced, but a completion-quality comparison still needs the same
  script, scene, query tensors, batch size, and machine for both backends.

## Required Acceptance Gate

Before this work can be considered complete, RayDTorch needs:

1. Failing parity tests that exercise the missing RayD behaviors on identical
   scene/query tensors.
2. Torch-native rewrites of the RayD multipath pipeline and CUDA kernels.
3. Explicit VJP and JVP implementations for the continuous values under the
   fixed-winner contract.
4. A nearest-edge broad-phase fix with a benchmark proving the regression is
   addressed.
5. Same-script RayD/RayDTorch benchmarks for intersect, nearest-edge,
   reflection/EPC, and diffraction.
