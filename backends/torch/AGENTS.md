# RayD Torch Agent Rules

## Native CUDA/OptiX Incremental Build

- Do not use `python -m pip install --no-build-isolation -e .` for every debug iteration. That command creates or refreshes a full editable build/install flow and is too slow for CUDA/OptiX work.
- The persistent native build directory is `build/local-120` (the `dev_build_native.ps1` default; `pyproject.toml` uses `build/{wheel_tag}`). It is ignored by git and should be reused across edits.
- If the build directory does not exist, initialize it once:

```powershell
python -m pip install --no-build-isolation -e . -Cbuild-dir=build/local-120
```

- After native `.cpp`, `.cu`, `.h`, CMake, or PTX embedding changes, use the incremental helper:

```powershell
powershell -ExecutionPolicy Bypass -File E:\Code\RayDi\backends\torch\scripts\dev_build_native.ps1
```

- The helper runs `cmake --build build/local-120 --config Release --target rayd_torch_stable_ops rayd_torch_legacy_ops _C` and copies the resulting `_stable_ops*.dll`, `_legacy_ops*.dll`, and `_C*.pyd` to the conda site-packages path that the editable import hook actually loads. See `abi_audit.json` for the authoritative `_C` / `_stable_ops` / `_legacy_ops` boundary.
- Run focused tests with the environment Python directly, for example:

```powershell
python -m unittest backends.torch.tests.torch_backend.test_multipath -v
python -m unittest discover backends.torch.tests.torch_backend -v
```

- Use full `pip install -e .` again only when intentionally regenerating the editable install metadata, changing packaging behavior, or recreating the persistent build directory from scratch.

## Native Numeric And Performance Acceptance

Use the current worktree and command output as authoritative. Do not use a full
editable reinstall for normal CUDA/OptiX iteration; use the incremental helper
above and then run focused numeric/performance tests.

Run the CUDA tests from the repository root:

```powershell
python -m unittest backends.torch.tests.torch_backend.test_edge_queries -v
python -m unittest backends.torch.tests.torch_backend.test_multipath -v
python -m unittest backends.torch.tests.torch_backend.test_multipath backends.torch.tests.torch_backend.test_scene_cache -v
python -m unittest discover backends.torch.tests.torch_backend -v
```

Latest recorded native test results, after the nearest-edge no-AD fast path and
RayD edge topology/cache updates:

- `tests.torch_backend.test_edge_queries -v`: 9 tests passed.
- `unittest discover tests.torch_backend -v`: 61 tests passed, 12 skipped.

Run external RayD parity explicitly; the normal discover run skips these tests:

```powershell
$env:RAYD_TORCH_RUN_DR_JIT_PARITY='1'
python -m unittest backends.torch.tests.torch_backend.test_drjit_parity -v
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

Run same-script RayD vs RayD Torch performance comparison:

```powershell
python -m backends.torch.tests.benchmark_rayd_backends --grid 64 --queries 4096 --warmup 5 --repeat 30
python -m backends.torch.tests.benchmark_rayd_backends --grid 64 --queries 4096 --warmup 5 --repeat 30 --dynamic
```

Latest recorded same-script static-vs-static performance result (2026-06-12,
warm OptiX disk cache, conda-forge torch 2.10; `build_ms` is dominated by
per-process OptiX init and swings tens of ms run to run):

```json
{
  "dynamic": false,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 65.1478,
    "diffraction_direct_ms": 0.43555,
    "diffraction_paths_ms": 0.31534,
    "intersect_flags_none_ms": 0.16069,
    "intersect_ms": 0.15019,
    "nearest_edge_ms": 1.34690,
    "reflection_trace_ms": 0.27304
  },
  "raydn": {
    "build_ms": 84.9524,
    "diffraction_direct_ms": 0.23441,
    "diffraction_paths_ms": 0.12266,
    "intersect_flags_none_ms": 0.04146,
    "intersect_ms": 0.06684,
    "nearest_edge_ms": 1.08190,
    "reflection_trace_ms": 0.08338
  },
  "repeat": 60,
  "warmup": 8
}
```

Latest recorded native benchmark highlights (`--grid 192 --queries 65536`,
random far-from-surface points): `nearest_edge_ms` 27.85 (was 230.8 before the
tiled fallback), `diffraction_direct_ms` 0.214, `dynamic_sync_ms` 1.53.

Current acceptance interpretation for the covered benchmark shape:

- Numeric parity is currently demonstrated for the covered forward cases and
  fixed-winner Torch VJP/JVP tests (109 native tests, 12 opt-in parity tests).
- RayD Torch is faster than RayD in the latest same-script static and dynamic runs
  for `intersect` (both modes), point `nearest_edge`, reflection trace,
  diffraction paths, and direct diffraction accumulation. Build wall time is
  per-process-init-dominated for both libraries at this scale; steady-state
  in-process RayD Torch scene build measures 2.4-5.2 ms (grid 64-192).
- Far-from-surface point nearest-edge queries resolve through the tightest
  OptiX tier plus an exact tiled fallback scan instead of the scene-diagonal
  tier (native grid-192 query shape: 230.8 -> 27.8 ms).
- The native build must be configured with explicit CUDA architectures:
  PyTorch's cmake config clobbers `CMAKE_CUDA_ARCHITECTURES` during
  `find_package(Torch)`; `CMakeLists.txt` restores it and fails configure if
  it resolves empty.
- Keep running release-size and Nsight-backed benchmarks before claiming broad
  performance superiority across all multipath/diffraction workloads.
