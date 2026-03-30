# Edge BVH Acceleration Structure - Bottleneck Analysis

> Analysis date: 2026-03-30
> Hardware reference: RTX 5080, Ryzen 7 9800X3D, 192x192 grid mesh, 110,976 edges, 65,536 queries
> Latest implementation status and Stage 3 plan: [docs/edge_bvh_stage3_preparation.md](E:\Code\RayDi\docs\edge_bvh_stage3_preparation.md)

## 1. Build Phase

### 1.1 GPU LBVH Build Pipeline

The build is a serial 7-stage GPU pipeline, with `cudaDeviceSynchronize()` after every kernel:

| Stage | Kernel | Complexity | Bottleneck Type |
|-------|--------|-----------|----------------|
| Primitive AABB | `compute_primitive_bounds_kernel` | O(N) | Bandwidth-bound, lightweight |
| Scene bounds | CUB `DeviceReduce` | O(N) | Library call, minimal overhead |
| Morton codes | `compute_morton_codes_kernel` | O(N) | Compute-bound (bit interleave) |
| Radix sort | CUB `DeviceRadixSort::SortPairs` | O(N) | **Primary cost**, 64-bit key |
| Radix tree | `build_radix_tree_kernel` | O(N) | Branch-heavy, LCP binary search |
| Leaf finalize | `finalize_leaves_and_bounds_kernel` | O(N) | **Atomic contention**, bottom-up merge |
| Treelet opt | CPU recursive | O(N * 2^k) | **CPU serial**, GPU -> CPU transfer |

### 1.2 Bottleneck 1: Excessive Synchronization

Every kernel is followed by `cudaDeviceSynchronize()` (`edge_bvh.cu:841-843`). Stages 1, 2, and 3 could partially overlap (scene bounds are only needed for Morton code normalization), but the current implementation is fully serial. For small meshes (< 10K edges), kernel launch overhead may account for 30-50% of total time.

### 1.3 Bottleneck 2: 64-bit Radix Sort

Morton codes use 64-bit (`edge_bvh.cu:140-177`), with 21 bits per axis. For most scenes, 21-bit precision far exceeds actual requirements. CUB RadixSort performs twice as many passes for 64-bit keys vs 32-bit. Reducing to 30-bit (10 bits per axis) + 30-bit primitive index with a 32-bit sort could speed up the sort stage by ~40-50%.

### 1.4 Bottleneck 3: Bottom-up Atomic Merge

`finalize_leaves_and_bounds_kernel` (`edge_bvh.cu:778-834`) uses `atomicAdd(merge_counters)` for bottom-up AABB merging. Near the root, only a handful of threads are active (fan-in = 1 at upper tree levels), resulting in extremely low GPU utilization. For a ~110K edge tree, the last 10 levels have ~100 active threads while thousands of GPU cores sit idle.

### 1.5 Bottleneck 4: CPU Treelet Optimization

This is the **single largest bottleneck** in the build phase. The current flow:

1. GPU builds LBVH -> `cudaMemcpy` entire tree back to host (`scene_edge.cpp:1180-1185`)
2. CPU recursively computes subtree leaf counts (`edge_bvh.cu:869-896`)
3. CPU serially executes treelet DP optimization (7-leaf frontier, 2^7 = 128 subset enumeration)
4. Optimized tree is uploaded back to GPU

For 110K edges, there are ~110K internal nodes, with potentially thousands satisfying `subtree_leaves >= 32`, each requiring O(2^7) DP. This CPU phase takes a significant portion of the 138ms `build()` time. The GPU treelet optimization exists (`edge_bvh.cu:517-607`) but is disabled (`EdgeBVHEnableGpuTreeletOptimization = false`).

### 1.6 Bottleneck 5: CPU Serial Compaction

`emit_compacted_bvh_preorder` (`scene_edge.cpp:1280-1290`) performs a preorder traversal on CPU to reorder nodes for better traversal memory locality (parent and child adjacent in memory). This is an O(N) CPU serial recursive process.

### 1.7 Build Phase Optimization Opportunities

| Optimization | Expected Speedup | Difficulty |
|-------------|-----------------|-----------|
| **CUDA stream overlap for kernel launches** | Build < 10K edges: ~2x | Low |
| **32-bit Morton code** (10 bits per axis) | Sort stage ~1.5-2x | Low |
| **Re-enable/optimize GPU treelet optimization** | Eliminate CPU transfer + serial DP | Medium |
| **Replace atomic bottom-up with level-by-level parallel** | Upper tree construction ~2-3x | Medium |
| **GPU compaction** (preorder relayout on GPU) | Eliminate final D2H/H2D round-trip | Medium |
| **PLOC (Parallel Locally-Ordered Clustering) replace LBVH** | Significantly better BVH quality, faster queries | High |

---

## 2. Query (Traversal) Phase

### 2.1 Bottleneck 1: Stack Management Overhead

`stack_push` and `stack_pop` (`scene_edge.cpp:1003-1038`) iterate through **32 slots** on every operation:

```cpp
for (size_t index = 0; index < EdgeBVHTraversalStackSize; ++index) {
    const MaskDetached write = active && (stack_size == static_cast<int>(index));
    stack[index] = select(write, value, stack[index]);
}
```

Each push/pop requires 32 `select` operations because Dr.Jit's vectorized model requires each lane to potentially have a different stack depth. For a BVH of depth ~17 (110K edges), each query does ~30-50 push/pop operations, resulting in **~2000 select operations solely for stack management**.

### 2.2 Bottleneck 2: Random Gather Access Pattern

Each node access during traversal requires gathers:
- 2x `gather<Vector3fDetached>` (bbox_min, bbox_max) = 6 float gathers
- 1x `gather<IntDetached>` (left_child)
- Child access repeats 2x gathers

Different lanes access completely different nodes, producing **fully random gather patterns**. While preorder compaction improves parent-child locality, cross-lane gathers remain random. L2 cache miss rate is the core bandwidth bottleneck of the query phase.

### 2.3 Bottleneck 3: Conservative AABB Distance Lower Bounds

**Ray-to-AABB lower bound** (`utils.h:273-300`) uses axis-separated approximation:

```cpp
Float_ dx = axis_distance(origin.x(), direction.x(), bbox_min.x(), bbox_max.x());
return fmadd(dx, dx, fmadd(dy, dy, dz * dz));
```

This lower bound is **very conservative** when the ray direction makes a large angle with the AABB diagonal. Conservative bounds mean more nodes fail to be pruned, leading to deeper traversal and more gather operations.

**Segment-to-AABB** (`utils.h:260-270`) is similarly conservative - it uses the bounding box of the segment endpoints and the AABB separating axis distance, ignoring the precise geometry of the segment path.

### 2.4 Bottleneck 4: Leaf Linear Scan

Each leaf node contains up to 4 primitives (`EdgeBVHLeafSize = 4`), processed sequentially:

```cpp
for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
    // gather primitive, compute distance...
}
```

4 unrolled iterations, each with 2x `gather<Vector3fDetached>` (p0, e1) + full `closest_point_on_segment` / `closest_segment_segment` computation. The segment-to-segment narrow-phase is particularly expensive (5 candidate point updates, `utils.h:92-177`).

### 2.5 Bottleneck 5: Lane Divergence

Dr.Jit's `while_loop` continues until **all lanes complete**. If one query's target point happens to be in a region with extremely high edge density, it may need 100+ iterations while other lanes finished long ago. These "straggler" lanes force all other lanes to execute no-ops (`active = false` but still participating in select/gather execution).

### 2.6 AD Re-computation

In AD mode (`scene.cpp:877-964`):
1. Detached traversal finds nearest edge ID
2. Re-gather edge geometry (AD types)
3. Recompute `closest_point_on_segment<false>`

The re-gather adds 2x `gather<Vector3f>` (AD version), but since it's only for 1 primitive, the overhead is ~10-15%. **This is not a major bottleneck** - the design is sound.

### 2.7 Query Phase Optimization Opportunities

| Optimization | Expected Speedup | Difficulty |
|-------------|-----------------|-----------|
| **Stack optimization**: bit-stack or restart trail replacing 32-slot linear scan | Query ~1.3-1.5x | Medium |
| **Tighter ray-AABB lower bound**: slab-based distance considering ray direction | ~20-30% fewer node visits | Medium |
| **BVH quality improvement** (PLOC/SAH): reduce traversal depth and leaf visits | Query ~2-3x (largest single optimization) | High |
| **Leaf size tuning**: leaf=1 for point queries, leaf=4 for ray queries | Query ~1.1-1.2x | Low |
| **Wavefront compaction**: periodically compress active lanes | Reduce tail-lane waste | High |
| **Node layout packing**: pack bbox_min/max/left/right into same cache line | Fewer gather operations | Medium |

---

## 3. Refit (sync) Phase

The refit path (`scene_edge.cpp:1338-1508`) is already reasonably efficient:

- Dirty range -> compress -> only scatter changed primitives
- Leaf bound recomputation only involves dirty leaves x 4 slots
- Bottom-up uses pre-computed `refit_levels_` for per-level parallel scatter

Remaining issues:
1. **Full leaf slot rescan**: Even if only 1 primitive changed, all 4 primitives in the leaf are gathered to recompute the AABB.
2. **Level-by-level sync**: One `drjit::eval` + `sync_thread` per level; for a depth-17 tree, that is 17 synchronization points.

---

## 4. Priority-Ranked Optimization Roadmap

Sorted by **benefit/cost ratio**:

1. **32-bit Morton code** - Minimal effort, sort speedup ~1.5x
2. **CUDA stream overlap** - Eliminate idle waits between kernel launches
3. **Tighter ray-AABB bound** - Medium effort, ~20-30% fewer node visits
4. **Stack operation improvement** - Eliminate 32-slot linear scan with bit encoding or short-stack
5. **GPU treelet / PLOC** - Largest quality improvement, but highest development cost
6. **Node data packing** - Pack 6 float bbox + 2 int child into 32 bytes, reduce gather count

### Performance Profile Summary

In the current **138ms build + ~10-15ms query** profile:
- **Build** time is dominated by sort + CPU treelet + D2H/H2D transfers
- **Query** time is dominated by stack management and random gather cache misses
- **BVH quality** (treelet-optimized LBVH is still inferior to SAH/PLOC) is the fundamental upper bound on query performance
