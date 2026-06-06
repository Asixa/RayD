# RayDTorch Native Performance

Measured on Windows with NVIDIA GeForce RTX 5080 and Torch CUDA 12.8.

Command:

```powershell
conda run -n witwin2 python -m tests.benchmark_raydtorch_native --grid 192 --queries 65536
```

Results:

```json
{
  "build_ms": 1518.6566999182105,
  "dynamic_sync_ms": 0.18119998276233673,
  "grid": 192,
  "intersect_ms": 0.07871999405324459,
  "nearest_edge_ms": 172.7742799790576,
  "queries": 65536
}
```

The benchmark covers native scene build, dynamic vertex sync, OptiX triangle intersection, and nearest-edge query throughput. Query timings are per benchmark iteration over 65,536 inputs.

No checked-in Dr.Jit performance snapshot exists in this repository. The Task 17 parity baseline records functional outputs only, so these RayDTorch-native numbers are the current performance baseline for this migration branch.
