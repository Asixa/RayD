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

The local build and comparison evidence is added with the verification commit.

## Consequences

- The prebuilt `rayd-torch` wheel can create its OptiX modules on Volta. This
  has not been run on Volta hardware.
- A kernel that needs a post-Volta PTX instruction can no longer use it in a
  ray-tracing module without a new decision record, because it would drop Volta
  from that module.
