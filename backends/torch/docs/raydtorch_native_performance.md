# RayDTorch Native Performance

Measured on Windows with NVIDIA GeForce RTX 5080 and Torch CUDA 12.8.

Command:

```powershell
conda run -n witwin2 python -m tests.benchmark_raydtorch_native --grid 192 --queries 65536
```

Current RayDTorch-native result:

```json
{
  "build_ms": 1550.21,
  "dynamic_sync_ms": 2.29,
  "grid": 192,
  "intersect_ms": 0.241,
  "nearest_edge_ms": 0.423,
  "queries": 65536
}
```

The benchmark covers native scene build, dynamic vertex sync, OptiX triangle intersection, and nearest-edge query throughput. Query timings are per benchmark iteration over 65,536 inputs.

## RayD Comparison Status

Same-script benchmark command:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30 --dynamic
```

The command above is a fast RayD/RayDTorch multipath regression shape. It is
not the only acceptance shape: it casts only 4,096 rays. For RayD latest-style
intersection pressure and Mitsuba comparison, use:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydtorch_rayd_mitsuba_stress `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --scenario rayd-latest:64:128 `
  --scenario release:192:256 `
  --repeats 5 --warmup 2 --mitsuba-preliminary
```

This stress script matches RayD's `mesh_resolution/ray_grid_side` convention.
`rayd-latest:64:128` casts 16,384 rays, while `release:192:256` casts 65,536
rays. It reports RayDTorch, RayD, and Mitsuba static/dynamic intersection
performance for full materialized fields and reduced t-only paths; Mitsuba
`ray_intersect_preliminary` is reported as an extra Mitsuba-only lower-level
baseline when `--mitsuba-preliminary` is set.

For scaling sweeps instead of a few fixed sizes, use:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydtorch_rayd_mitsuba_sweep `
  --preset standard `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary
```

The sweep emits `sweep.json`, `sweep.csv`, and PNG grouped-bar plots under
`artifacts/benchmarks/scaling/<preset>/`. Presets:

- `smoke`: quick script/plot validation.
- `standard`: up to 768 mesh resolution, about 1.18M triangles, and 1M requested rays.
- `large`: up to 1024 mesh resolution, about 2.10M triangles, and 10M requested rays.
- `extreme`: includes 100,663,296 requested rays.

Large ray counts are represented by a fixed ray batch plus a batch count. By
default the script measures per-batch throughput and projects total time for the
requested ray count. Add `--execute-total-rays` when the goal is to actually run
all batches for the 10M/100M ray entries.

Current scaling interpretation:

- In `artifacts/benchmarks/scaling/codex_current_large_forward_r30/sweep.json`,
  RayDTorch is faster than RayD on the large static full intersection point.
  At 2.10M triangles and a 1,048,576-ray batch projected to 100.66M requested
  rays, static full is RayDTorch 0.3760 ms/batch vs RayD 0.6258 ms/batch.
- In the same large static t-only/reduced case, RayDTorch is also faster:
  RayDTorch 0.1478 ms/batch vs RayD 0.2136 ms/batch.
- Dynamic RayD/RayDTorch scene update timings in the same run also favor
  RayDTorch for this shape: full 1.4799 ms/batch vs RayD 22.8009 ms/batch,
  reduced 1.2745 ms/batch vs RayD 22.7541 ms/batch.
- Dynamic Mitsuba numbers are recorded, but they are not the primary comparison:
  Mitsuba dynamic scene updates perform additional work not directly comparable
  to RayD/RayDTorch.

RayD latest-style multipath path export is covered separately:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydtorch_rayd_mitsuba_multipath `
  --preset smoke --rayd-source local --rayd-root E:\Code\RayDi
```

This benchmark adds RayDTorch to RayD's path-level Mitsuba comparison for:

- `reflection_trace`: parallel reflectors, minimal path export.
- `diffraction_export`: synthetic single-edge diffraction path export.

It writes all outputs under one folder, for example
`artifacts/benchmarks/multipath/smoke_all/`, with `multipath.json`,
`multipath.csv`, and `time_ms_multipath.png`. The plot is a grouped bar chart of
absolute average time in ms only; no SVG or throughput plot is emitted.

Latest high-repeat multipath result, 65,536 rays/states, 30 repeats, 8 warmup:

| Workload | RayDTorch ms | RayD ms | Status |
|---|---:|---:|---|
| reflection trace | 0.0507 | 0.1296 | RayDTorch faster |
| diffraction export | 0.2188 | 0.2967 | RayDTorch faster |

AD backward is measured with `--include-backward`; plots are absolute projected
time grouped by backend, not throughput or speedup plots:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -B -m tests.benchmark_raydtorch_rayd_mitsuba_sweep `
  --preset smoke --mesh-resolution 64 --mesh-resolution 128 --mesh-resolution 256 `
  --total-rays 16384 --total-rays 65536 --ray-batch-side 256 `
  --repeats 5 --warmup 3 --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary --include-backward `
  --output-dir artifacts\benchmarks\scaling\ad_uv_tape
```

For RayDTorch, `t_sum_*` benchmark modes use `Scene.intersect_t_sum_vjp`, a
native scalar-loss VJP for `sum(intersection.t)`. This computes the same loss
and vertex gradients as `scene.intersect(...).t.sum().backward()` in parity
tests, while avoiding PyTorch eager construction of unused public intersection
outputs and generic autograd-engine overhead. This is the closest RayDTorch
counterpart to RayD's warm-started compiled VJP for the same scalar loss.

High-repeat static AD/VJP points, 65.5K rays, 30 repeats, 8 warmup:

| Mode | RayDTorch ms | RayD ms | Status |
|---|---:|---:|---|
| 73.7K tri forward full | 0.0601 | 0.1325 | RayDTorch faster |
| 73.7K tri forward reduced | 0.0487 | 0.1317 | RayDTorch faster |
| 73.7K tri backward `t_sum_full` | 0.0820 | 0.2092 | RayDTorch faster |
| 73.7K tri backward `t_sum_reduced` | 0.0764 | 0.1900 | RayDTorch faster |
| 131K tri forward full | 0.0614 | 0.1272 | RayDTorch faster |
| 131K tri forward reduced | 0.0461 | 0.1173 | RayDTorch faster |
| 131K tri backward `t_sum_full` | 0.0918 | 0.2220 | RayDTorch faster |
| 131K tri backward `t_sum_reduced` | 0.0774 | 0.1914 | RayDTorch faster |

The former static AD backward gap is closed for the covered benchmark shapes.
The fix combines an expanded-gradient t-only backward path, a direct native
`sum(t)` VJP with constant scalar `grad_t=1`, and avoiding eager public-output
construction when the scalar loss only depends on `t`.

Current same-script static-vs-static result:

```json
{
  "dynamic": false,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 2342.0363000041107,
    "diffraction_direct_ms": 0.4519966666218049,
    "intersect_ms": 0.14462333335056124,
    "nearest_edge_ms": 1.4299183333302306,
    "reflection_trace_ms": 0.34219333332051366
  },
  "raydtorch": {
    "build_ms": 1547.1698999972432,
    "diffraction_direct_ms": 0.43191333328043885,
    "intersect_ms": 0.10184333332290407,
    "nearest_edge_ms": 1.4051099999051075,
    "reflection_trace_ms": 0.3025700001065464
  },
  "repeat": 60,
  "warmup": 8
}
```

Current same-script dynamic-vs-dynamic result:

```json
{
  "dynamic": true,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 2337.4531000008574,
    "diffraction_direct_ms": 0.9759666667378042,
    "intersect_ms": 0.12905000015355958,
    "nearest_edge_ms": 1.5716533331821363,
    "reflection_trace_ms": 0.32191333327015553
  },
  "raydtorch": {
    "build_ms": 1547.6975999990827,
    "diffraction_direct_ms": 0.42821333336178213,
    "intersect_ms": 0.11110666673630476,
    "nearest_edge_ms": 1.4978466667040873,
    "reflection_trace_ms": 0.3103666667205592
  },
  "repeat": 30,
  "warmup": 5
}
```

An isolated RayDTorch-only microbenchmark of the no-AD reflection trace path on
the same grid measured:

```json
{
  "trace_reflections_forward_noad_ms": 0.291,
  "trace_reflections_forward_ad_outputs_ms": 0.461,
  "scene_trace_reflections_python_ms": 0.319
}
```

Current interpretation for this benchmark shape:

- RayDTorch `intersect` is faster in the latest static and dynamic same-script
  runs.
- RayDTorch large static full and reduced intersection are faster in the latest
  RayD/RayDTorch sweep run.
- RayDTorch static `sum(t)` backward/VJP is faster than RayD in the latest
  65.5K-ray sweep for both full and reduced modes.
- RayDTorch multipath reflection trace and diffraction export are faster in the
  latest RayD path-export benchmark.
- RayDTorch `diffraction_direct` is faster in the latest static and dynamic
  same-script runs.
- RayDTorch point `nearest_edge` is faster in the latest static and dynamic
  same-script runs after the no-AD Torch path removed unnecessary tape
  allocation/writes for non-AD callers.
- RayDTorch scene build is faster in the latest static and dynamic same-script
  runs.
- RayDTorch reflection trace is faster in the latest static and dynamic
  same-script runs.
- Keep release-size and Nsight-backed runs as the broader performance gate;
  this document records parity for the covered benchmark, not universal
  superiority for every multipath/diffraction workload.

## Parity Coverage Status

The current opt-in external RayD parity test covers same-scene forward cases for:

- `intersect`
- multi-mesh global ids
- point `nearest_edge`
- visibility
- reflection tracing
- diffraction paths
- direct, Keller, suffix, order-2, and order-3 diffraction accumulation
- coherent direct diffraction accumulation

Torch-native AD tests cover fixed-winner VJP/JVP for:

- `intersect` VJP/JVP
- `intersect(..., flags=RayFlags.None)` VJP/JVP through hidden tape
- point/ray `nearest_edge` VJP/JVP
- reflection trace VJP/JVP
- reflection EPC forward/VJP/JVP
- diffraction accumulation forward/VJP/JVP

Visibility returns a discrete bool and has no continuous gradient contract.

## Multipath Implementation Status

RayDTorch now contains source ports for the RayD reflection and diffraction
`src/multipath` execution paths, including segment visibility, reflection trace,
reflection EPC, EPC field, reflection dedup, reflection accumulation,
diffraction path search, diffraction accumulation, chain accumulation, suffix
reflection, and coherent direct accumulation. Performance remains the active
completion risk.

See `docs/raydtorch_native_gap_analysis.md` for the tracked gap list.
