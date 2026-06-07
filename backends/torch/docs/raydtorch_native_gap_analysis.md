# RayDTorch Native Gap Analysis

Status: RayDTorch now contains RayD-style OptiX PTX pipelines for scene
intersection, edge query, reflection, and diffraction paths. The current code
also includes Torch VJP/JVP kernels for the supported continuous outputs under
the fixed-winner contract. Full completion still requires performance
acceptance: the same-script RayD/RayDTorch benchmark exists, but current results
do not yet show performance parity across the covered kernels.

## Current Multipath Implementation

The current code should be treated as a source-port with active performance
work remaining:

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

The corresponding RayD source files exist in `E:\Code\RayDi`. Reflection and
diffraction forward kernels now have RayDTorch source ports, including:

- `E:\Code\RayDi\src\multipath\segment_visibility.cu`
- `E:\Code\RayDi\src\multipath\reflection_trace.cu`
- `E:\Code\RayDi\src\multipath\reflection_accumulation.cu`
- `E:\Code\RayDi\src\multipath\reflection_dedup.cu`
- `E:\Code\RayDi\src\multipath\reflection_epc.cu`
- `E:\Code\RayDi\src\multipath\reflection_epc_field.cu`
- `E:\Code\RayDi\src\multipath\diffraction_paths.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation_ad.cu`

These RayD files should remain the parity source of truth when extending the
current ports.

## Not Yet Completion-Quality Performance

These areas are still open and must not be counted as performance complete:

- Same-script, same-data, same-batch RayD vs RayDTorch performance comparison
  is implemented in `tests/benchmark_rayd_vs_raydtorch.py`.
- Current same-script results show RayDTorch is faster for `intersect` and close
  for `nearest_edge`, but slower for scene build, reflection trace, and
  diffraction direct accumulation on the recorded quick benchmark shape.
- Completion-quality performance work should focus on reflection trace and
  diffraction direct throughput first, then scene build.

## Required Acceptance Gate

Before this work can be considered complete, RayDTorch needs:

1. Same-script RayD/RayDTorch performance runs for the release benchmark shapes.
2. Performance fixes or accepted thresholds for scene build, reflection trace,
   and diffraction direct accumulation.
3. Full native and opt-in RayD parity test runs after any performance changes.
