# ADR-0044: Torch OptiX PTX at the Volta floor

- Status: Accepted by the user on 2026-11-01; pending review by Xingyu Chen
- Date: 2026-11-01
- Decision ID: `torch-optix-ptx-at-the-volta-floor`
- Scope: the virtual architecture of the 11 Torch ray-tracing PTX modules, and
  the end of ADR-0035 divergence `D2`

## Context

`docs/dev/gpu_support.md` establishes `sm_70` as RayD's support floor. Prebuilt
`rayd-torch` wheels carry an `sm_70` CUDA image, but ADR-0035 froze their OptiX
PTX at `compute_75` (divergence `D2`) while Dr.Jit used `compute_70`. OptiX
rejects PTX newer than the device, so on Volta the prebuilt Torch wheel could
run its CUDA paths and could not create a single OptiX module. No decision
record explained the `compute_75` choice. `docs/torch/cuda_performance_analysis.md`
only says an explicit architecture was needed for warp intrinsics such as
`__match_any_sync`, which `compute_70` already provides.

## Decision

All 11 Torch ray-tracing PTX modules compile with
`--gpu-architecture=compute_70`, in every build. Together with the eight
committed Dr.Jit modules, all 19 ray-tracing modules now share one architecture,
which is the support floor:

- `D2` is removed from `contracts/compile_policy.json`, as the ADR-0035 stop
  condition requires, and its id is retired.
- Numeric flags and profiles are unchanged. Only the architecture moves.
- `RAYD_CUDA_ARCHS` no longer changes the OptiX PTX architecture of either
  backend, so the helper that lowered it for source builds is removed.
- `tests/test_compile_flag_policy_contract.py` requires `compute_70` for every
  ray-tracing module, so a higher architecture cannot return unnoticed.

The runtime device check keeps its OptiX rule. It still rejects a build record
whose OptiX target is newer than the device.

Pre-Volta GPUs remain unsupported by decision of the same date:
`docs/dev/gpu_support.md` records why.

## Evidence

Measured on 2026-11-01 on Windows with an RTX 4090 Laptop GPU (`sm_89`), driver
572.16, CUDA 12.8.93, MSVC 19.42, and OptiX 8.1 headers. Two `rayd-torch` wheels
used the exact Windows release lists from `.github/workflows/pypi.yml`
(`CMAKE_CUDA_ARCHITECTURES`, `TORCH_CUDA_ARCH_LIST`, and
`RAYD_TORCH_CUDA_GENCODE_FLAGS`). They came from one build tree, with only the
architecture changed, so all CUDA objects are shared.

**Generated code.** All 11 PTX modules are `.target sm_70` in the new build and
`.target sm_75` in the old one, and that line is the only difference in each
module. Both are PTX ISA 8.7. The wheel links nine of the 11 modules; the dormant
axial-edge and segment-penetration modules are compiled but not linked into it.
The wheel's CUDA images are `sm_70` through `sm_120` plus `compute_120` PTX, as
before.

**Numerics.** The scene is a closed box of 47,628 bumpy triangles and 200,000 rays,
with 1,507,254 valid bounces of 1,600,000. Each wheel ran five times in alternating
fresh processes, with the OptiX disk cache off. Every result below is bitwise
identical across all five `compute_70` and `compute_75` pairs:

- `trace_reflections` at 8 bounces: `valid`, `t`, `image_sources`, `prim_ids`
- `intersect`: all 11 output fields
- `visible` and `nearest_edge(ray)`
- `trace_refl_epc_field`, whose 256 test rows were all invalid, so this shows only
  that the module loads and returns the same inert rows

The `intersect` vertex gradient is not bitwise reproducible even between two runs
of one wheel, which gave five distinct results each. The largest difference is
2.4e-7 against a maximum magnitude of 3.1, and it is the same within one wheel and
across wheels, so the architecture is not its cause.

**Timing.** Medians of the five runs, in milliseconds, with the observed range. The
first call includes OptiX module and pipeline creation. All ranges overlap, so no
difference is measurable.

| Measurement | `compute_70` | `compute_75` |
| --- | --- | --- |
| `trace_reflections` first call | 337.1 (327.1-342.1) | 341.6 (332.3-344.5) |
| `trace_reflections` warm, median of 50 | 0.717 (0.710-0.946) | 0.736 (0.715-0.918) |
| `intersect` warm, median of 20 | 0.173 (0.163-0.241) | 0.234 (0.165-0.243) |
| `nearest_edge` warm, median of 20 | 35.78 (35.51-35.89) | 35.81 (35.34-37.03) |

**Regression.** `tests/native/test_multipath.py` gives the same result on both
wheels: 32 run, 27 pass, and the same five tests fail with
`optixPipelineCreate(multipath)` code 7251 in Torch diffraction-path and
reflection-accumulation calls on driver 572.16. This
matches the five errors the crash-hardening work reports for this driver.

**Not verified.** No run used Volta hardware, so Volta execution rests on the
compatibility rule and on PTX that differs only in its target line. The modules
behind those five errors, along with top-k edge and diffraction accumulation, were
not compared at run time here. Only Windows was built.

## Consequences

- The prebuilt `rayd-torch` wheel targets Volta for its OptiX modules, and on
  newer GPUs behaves as before within the evidence above. It has not been run on
  Volta hardware.
- A kernel that needs a post-Volta PTX instruction can no longer use it in a
  ray-tracing module without a new decision record, because it would drop Volta
  from that module.
