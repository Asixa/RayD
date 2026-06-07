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
```

Current same-script result:

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

An isolated RayDTorch-only microbenchmark of the no-AD reflection trace path on
the same grid measured:

```json
{
  "trace_reflections_forward_noad_ms": 0.291,
  "trace_reflections_forward_ad_outputs_ms": 0.461,
  "scene_trace_reflections_python_ms": 0.319
}
```

Current interpretation:

- RayDTorch `intersect` is faster in this run.
- RayDTorch `nearest_edge` is close to RayD and no longer shows the previous
  all-edge-scan-scale regression.
- RayDTorch scene build, reflection trace, and diffraction direct accumulation
  are still slower in the same-script benchmark.
- RayDTorch has a no-AD reflection trace fast path that avoids exporting the
  full AD tape when inputs have neither reverse-mode gradients nor forward-mode
  tangents; isolated timing shows the fast path helps, but same-script RayD
  parity performance is not yet closed.

The implementation is now benchmarkable against RayD in one script, but should
not yet be treated as performance-equivalent or performance-superior across the
requested multipath/diffraction surface.

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
