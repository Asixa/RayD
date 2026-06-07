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
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 3 --repeat 10
```

Current same-script result:

```json
{
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 63.655,
    "diffraction_direct_ms": 0.493,
    "intersect_ms": 0.249,
    "nearest_edge_ms": 1.363,
    "reflection_trace_ms": 0.291
  },
  "raydtorch": {
    "build_ms": 99.065,
    "diffraction_direct_ms": 0.756,
    "intersect_ms": 0.082,
    "nearest_edge_ms": 1.302,
    "reflection_trace_ms": 0.737
  }
}
```

Current interpretation:

- RayDTorch `intersect` is faster in this run.
- RayDTorch `nearest_edge` is close to RayD and no longer shows the previous
  all-edge-scan-scale regression.
- RayDTorch scene build, reflection trace, and diffraction direct accumulation
  are slower in this run.

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
