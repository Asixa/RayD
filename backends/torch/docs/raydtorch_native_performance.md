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
