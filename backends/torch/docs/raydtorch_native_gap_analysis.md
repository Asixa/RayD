# RayDTorch Native Gap Analysis

Status: RayDTorch is not a complete Torch-native port of RayD multipath or
diffraction. The current native extension contains usable geometry and edge
query scaffolding plus a source-port of the RayD reflection-side multipath
kernels, but full RayD parity still requires scene split/IAS parity, reflection
AD parity, diffraction kernels, and same-script validation.

## Current Multipath Implementation

The current code should be treated as an in-progress source-port, not completed
RayD multipath parity:

- `src/torch_ext/multipath_pipeline.cpp` ports the RayD-style shared OptiX
  launch pipeline/cache for reflection-side PTX modules.
- `src/torch_ext/kernels/segment_visibility_optix.cu`,
  `reflection_trace_optix.cu`, `reflection_epc_optix.cu`,
  `reflection_accumulation_optix.cu`, `reflection_dedup.cu`, and
  `reflection_epc_field.cu` are source ports from RayD with Torch/ATen host
  glue.
- `src/torch_ext/kernels/multipath_backward.cu` still mainly routes reflection
  and EPC gradients through simplified fixed-winner intersection AD.
- `src/torch_ext/kernels/diffraction_accumulation_ad.cu` implements a
  simplified direct accumulation based on `(edge_pos + edge_dir) dot src`.
- `CMakeLists.txt` builds reflection-side multipath PTX targets, but the
  diffraction path and accumulation PTX targets are still absent.

## Missing RayD Multipath Kernel Coverage

The corresponding RayD source files exist in `E:\Code\RayDi`. Reflection-side
files now have RayDTorch source ports; diffraction files are still missing:

- `E:\Code\RayDi\src\multipath\diffraction_paths.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation_ad.cu`

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
