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

Known same-scale RayD edge benchmark snapshot:

- RayD default treelet build: 138.4 ms
- RayD default treelet sync: 3.69 ms
- RayD default treelet nearest-edge query: 9.99 ms
- RayD HLBVH nearest-edge experiment: 102.9 ms

Current interpretation:

- RayDTorch scene build is currently slower for this benchmark shape.
- RayDTorch dynamic sync now rebuilds the edge acceleration data after dynamic vertex updates, so it should not be compared to the earlier stale-edge-accel timing.
- RayDTorch `nearest_edge` no longer shows the previous all-edge-scan-scale regression in this benchmark.
- There is no same-script RayD `intersect` baseline checked into this repository, so `intersect_ms` cannot yet be used to claim a RayD speedup.

The current implementation should be treated as runnable but not yet proven performance-equivalent or performance-superior to RayD. A same-script RayD/RayDTorch benchmark is still required for completion-quality performance acceptance.

## Parity Coverage Status

The current opt-in external RayD parity test covers simple same-scene forward cases for:

- `intersect`
- point `nearest_edge`
- visibility
- reflection tracing

It still does not prove parity for:

- `intersect` VJP/JVP
- ray `nearest_edge`
- point/ray `nearest_edge` VJP/JVP
- visibility gradients for future continuous visibility outputs
- reflection trace VJP/JVP
- reflection EPC forward/VJP/JVP
- diffraction accumulation forward/VJP/JVP

A completion-quality parity/performance gate needs to run RayD and RayDTorch on the same scene, same query tensors, same batch sizes, and same machine for those APIs.

## Multipath Implementation Status

RayDTorch now contains source ports for the RayD reflection-side `src/multipath`
execution path: segment visibility, reflection trace, reflection EPC, EPC
field, reflection dedup, and reflection accumulation. This still should not be
read as completed RayD parity until the same-script RayD/RayDTorch numerical and
performance gates cover those kernels. Diffraction path search and diffraction
accumulation remain incomplete.

See `docs/raydtorch_native_gap_analysis.md` for the tracked gap list.
