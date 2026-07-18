# RayD improvement plan execution record (2026-07-11)

This record maps `docs/rayd_improvement_plan.md` to the implementation commits
and grouped acceptance evidence. Historical phase JSON files remain immutable;
this document records the final grouped native acceptance.

## Phase commits

| Scope | Commits |
| --- | --- |
| F0 | `f8bda6a`, `6a93a07`, `557dcc3`, `638540e` |
| BVH-0..4 | `a1cae72`, `1e20972`, `bc1f1c2`, `e52c1db`, `9eabbbc` |
| Share-1..5 | `8040734`, `8d3ba3f`, `507feca`, `5efcfda`, `a8ace23` |
| F1 | `4e36a07`, `2b0ca9e`, `cb07d59` |
| F2 | `08898a5`, `2e6da77`, `c742e94` |
| F3 | `b271b23` |
| F4 | `47f715b` |
| F5 | `95139ea` |
| Share-6 | `b43aa2b`, `714da19` |
| Performance gate | `06f08bc`, `630dcd0`, `69e6634`, `3a3c060`, `2c7afaa` |
| Contract maintenance | `a7bf714` |
| Local CUDA build workflow | `86f7b70`, `df2a59c`, `6b4cf90` |

## Build configuration

- Environment: `witwin3`, CPython 3.10.20.
- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0.
- CUDA toolkit/runtime: 12.9; driver 596.49.
- Compiler: MSVC 19.44.35207.1, Release.
- Local Torch build: `CMAKE_CUDA_ARCHITECTURES=120` and
  `TORCH_CUDA_ARCH_LIST=12.0`.
- Local Dr.Jit build: `RAYD_CUDA_GENCODE_ARCHES=120`, without a local PTX
  architecture.
- Release CI retains its explicit multi-architecture SASS and forward PTX
  matrix. Local source builds compile one native SASS architecture.

Both editable native builds completed from fresh source changes. The Torch
build produced the metadata `_C` shim, `_legacy_ops.dll`, and the untagged
Stable ABI `_stable_ops.dll`. The Dr.Jit build produced and loaded the fresh
CPython extension from the same environment.

## Grouped acceptance

| Gate | Result |
| --- | --- |
| Root source/contracts | 165 passed, 3 expected skips |
| Torch complete suite with Dr.Jit parity enabled | 143 passed, 1 documented invalid-fixture skip |
| Dr.Jit complete suite, including 29 cold-create subprocess cases | 196 passed |
| Targeted Torch ray-edge VJP/JVP and top-k/compile | 7 passed |
| Packaging metadata, wheel layout, clean install order, uninstall and legacy upgrade | 16 passed, 1 CI-only release-matrix skip |
| Stable ABI source and built-wheel dependency audit | passed |
| Share-2 host C++ math smoke | passed |

The remaining Torch skip is the existing `trace_refl_epc_field` cross-backend
fixture. Its public Torch call derives the incident ray from source to receiver;
the old single-plane fixture places those endpoints on opposite sides and is
not a physically valid reflected path. The production EPC forward/backward/JVP
tests pass; the skipped test is not counted as evidence for cross-backend field
parity.

## Performance acceptance

The first post-change smoke run was rejected because the GPU was simultaneously
occupied by another application. A subsequent GPU-idle smoke candidate,
`artifacts/benchmarks/edge_bvh_matrix/final_smoke_df2a59c_r3.json`, passed all
11 cases against `bvh0_smoke_f8bda6a_v2.json`.

The authoritative GPU-idle full candidate is
`artifacts/benchmarks/edge_bvh_matrix/final_full_2c7afaa.json`. It passed all 22
cases and all 110 metric comparisons against the frozen
`bvh0_full_f8bda6a.json` baseline, with no correctness or AD failures. The gate
used `--allow-legacy-launch-baseline` only because the frozen pre-change
baseline predates launch-audit fields; the candidate retains complete per-stage
launch audits.

Representative largest-scene results (2,000,000 edges) were:

| Metric | Baseline | Final | Change |
| --- | ---: | ---: | ---: |
| hot point query | 132.804 ms | 96.814 ms | -27.10% |
| build | 1363.856 ms | 725.047 ms | -46.84% |
| cold create | 9266.200 ms | 5204.856 ms | -43.83% |
| peak device memory | 1,342,177,280 B | 1,386,217,472 B | +3.28% |

The memory increase remains below the frozen +5% limit. To preserve correctness
on very large scenes, product builds apply GPU treelet optimization through
500,000 primitives and retain a valid LBVH above that size. Static scenes also
release dynamic-refit-only buffers after compaction; the final 2,000,000-edge
candidate therefore remains within the memory gate while keeping complete
query launch auditing.
