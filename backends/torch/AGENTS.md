# RayDTorch Agent Rules

## Native CUDA/OptiX Incremental Build

- Do not use `python -m pip install --no-build-isolation -e .` for every debug iteration. That command creates or refreshes a full editable build/install flow and is too slow for CUDA/OptiX work.
- The persistent native build directory is `artifacts/skbuild`. It is ignored by git and should be reused across edits.
- If `artifacts/skbuild` does not exist, initialize it once:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install --no-build-isolation -e . -Cbuild-dir=artifacts/skbuild
```

- After native `.cpp`, `.cu`, `.h`, CMake, or PTX embedding changes, use the incremental helper:

```powershell
powershell -ExecutionPolicy Bypass -File E:\Code\RayDTorch\scripts\dev_build_native.ps1
```

- The helper runs `cmake --build artifacts/skbuild --config Release --target _raydtorch` and copies the resulting `_raydtorch*.pyd` to the conda site-packages path that the editable import hook actually loads.
- Run focused tests with the environment Python directly, for example:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover tests.raydtorch_native -v
```

- Use full `pip install -e .` again only when intentionally regenerating the editable install metadata, changing packaging behavior, or recreating the persistent build directory from scratch.

## Native Numeric And Performance Acceptance

Use the current worktree and command output as authoritative. Do not mark the
multipath/diffraction migration complete just because the lightweight native
tests pass; performance parity is still an active gate.

Run the CUDA tests with the `witwin2` environment Python:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_edge_queries -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath tests.raydtorch_native.test_scene_cache -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover tests.raydtorch_native -v
```

Latest recorded native test results:

- `tests.raydtorch_native.test_edge_queries -v`: 7 tests passed.
- `tests.raydtorch_native.test_multipath -v`: 20 tests passed.
- `tests.raydtorch_native.test_multipath tests.raydtorch_native.test_scene_cache -v`: 24 tests passed.
- `unittest discover tests.raydtorch_native -v`: 59 tests passed, 12 skipped.

Run external RayD parity explicitly; the normal discover run skips these tests:

```powershell
$env:RAYDTORCH_RUN_DR_JIT_PARITY='1'
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_drjit_parity -v
```

Latest recorded opt-in RayD parity result:

- 12 tests passed.
- Covered forward parity: scene intersection, multi-mesh global ids, point
  nearest edge, visibility, reflection tracing, diffraction paths, direct
  diffraction accumulation, Keller accumulation, suffix accumulation,
  order-2/order-3 chain accumulation, and coherent direct accumulation.
- The run may print `jitc_llvm_init(): LLVM API initialization failed ..`;
  this warning appeared in the passing run and did not invalidate the parity
  assertions.

Run same-script RayD vs RayDTorch performance comparison:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30
```

Latest recorded same-script performance result:

```json
{
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 62.563,
    "diffraction_direct_ms": 0.409,
    "intersect_ms": 0.212,
    "nearest_edge_ms": 1.460,
    "reflection_trace_ms": 0.274
  },
  "raydtorch": {
    "build_ms": 101.735,
    "diffraction_direct_ms": 0.458,
    "intersect_ms": 0.089,
    "nearest_edge_ms": 1.379,
    "reflection_trace_ms": 0.471
  },
  "repeat": 30,
  "warmup": 5
}
```

Latest isolated RayDTorch reflection trace microbenchmark on the same grid:

```json
{
  "trace_reflections_forward_noad_ms": 0.291,
  "trace_reflections_forward_ad_outputs_ms": 0.461,
  "scene_trace_reflections_python_ms": 0.319
}
```

Current acceptance interpretation:

- Numeric parity is currently demonstrated for the covered forward cases and
  fixed-winner Torch VJP/JVP tests.
- RayDTorch `intersect` is faster in the recorded same-script run.
- RayDTorch `nearest_edge` is close to RayD and no longer shows the previous
  all-edge-scan-scale regression.
- RayDTorch scene build, reflection trace, and diffraction direct accumulation
  are still slower in the same-script benchmark.
- A no-AD reflection trace fast path exists and helps isolated RayDTorch timing,
  but same-script RayD performance parity is not closed.
- Therefore the full goal is not complete until performance thresholds are
  fixed or explicitly accepted after the same-script RayD/RayDTorch benchmark.
