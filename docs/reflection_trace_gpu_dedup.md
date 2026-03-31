# GPU-Side Reflection Path Deduplication

Extends `trace_reflections()` with an optional on-GPU deduplication pass that
collapses N raw ray chains into M unique reflection paths without any CPU-GPU
synchronization.

## Motivation

The current `trace_reflections()` returns one chain per input ray.
Typical usage fires 1K-1M rays but only discovers 10-100 geometrically
unique reflection paths.  The caller must deduplicate on the host:

```
trace_reflections()  →  [N, B]  →  torch.unique  →  [M, B]
     ~3ms GPU               ~5ms CPU-GPU sync
```

Moving dedup into the GPU kernel eliminates the sync and produces final
unique paths directly:

```
trace_reflections(deduplicate=True)  →  [M, B]
              ~4ms GPU, zero sync
```

---

## API Changes

### C++

```cpp
// New option struct
struct ReflectionTraceOptions {
    bool deduplicate = false;

    // Canonical primitive remapping table (device pointer).
    // If non-null, prim_ids are remapped through this LUT before
    // chain hashing:  canonical_prim = table[prim_id].
    // Size must be >= n_triangles.  Entries < 0 are treated as identity.
    const int *canonical_prim_table = nullptr;
    int canonical_prim_table_size = 0;

    // Image-source clustering tolerance (per-component L-inf).
    // Chains with identical canonical prim sequence whose image sources
    // differ by less than this in all xyz components are merged.
    float image_source_tolerance = 1e-5f;
};

// Extended overloads on Scene:
template <bool Detached>
ReflectionChainT<Detached> trace_reflections(
    const RayT<Detached> &ray,
    int max_bounces,
    MaskT<Detached> active = true) const;                    // existing

template <bool Detached>
ReflectionChainT<Detached> trace_reflections(
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active = true) const;                    // new
```

### ReflectionChainData additions

```cpp
template <typename Float_>
struct ReflectionChainData {
    // ... existing fields ...

    // Dedup-specific (only populated when deduplicate=true):
    Int_ discovery_count = full<Int_>(0, 1);  // [M] rays per unique path

    // Geometry for replay (per unique path, per bounce):
    Vec3f plane_points  = zeros<Vec3f>(1);    // [M*B] = hit_point
    Vec3f plane_normals = zeros<Vec3f>(1);    // [M*B] = geo_normal
};
```

### Python

```python
# DrJit native
chain = scene.trace_reflections(ray, max_bounces=3, deduplicate=True)
chain = scene.trace_reflections(ray, max_bounces=3,
    deduplicate=True,
    canonical_prim_table=table,   # IntDetached [T]
    image_source_tolerance=1e-5)

# PyTorch
chain = scene.trace_reflections(ray, max_bounces=3, deduplicate=True)
chain.ray_count        # M (unique paths, not input ray count)
chain.prim_ids         # [M, B]
chain.image_sources    # [M, B, 3]
chain.discovery_count  # [M]
```

---

## Algorithm

### Step 1: Trace (existing raygen kernel, unchanged)

Produces raw output arrays `[N * B]` as today.

### Step 2: Build Chain Keys

New CUDA kernel: `reflection_dedup_build_keys`

For each ray `i`:
1. If `bounce_count[i] == 0` → key = `UINT64_MAX` (dead lane, sorted last)
2. Optionally remap prim_ids through `canonical_prim_table`
3. Hash the prim_id chain into a 64-bit key:

```cuda
__global__ void reflection_dedup_build_keys(
    int n_rays,
    int max_bounces,
    const int *bounce_count,     // [N]
    const int *prim_ids,         // [N*B]
    const int *shape_ids,        // [N*B]
    const int *face_offsets,     // [n_meshes]
    int n_meshes,
    const int *canonical_table,  // [T] or nullptr
    int canonical_table_size,
    uint64_t *out_keys,          // [N]
    int *out_ray_indices         // [N] (identity permutation, for sort)
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;

    int bc = bounce_count[i];
    if (bc <= 0) {
        out_keys[i] = UINT64_MAX;
        out_ray_indices[i] = i;
        return;
    }

    // Build global prim chain + optional canonical remap
    uint64_t hash = 14695981039346656037ULL; // FNV-1a offset basis
    int base = i * max_bounces;
    for (int b = 0; b < bc; ++b) {
        int sid = shape_ids[base + b];
        int lprim = prim_ids[base + b];
        int foff = (sid >= 0 && sid < n_meshes) ? face_offsets[sid] : 0;
        int gprim = foff + lprim;

        // Canonical remap
        if (canonical_table != nullptr &&
            gprim >= 0 && gprim < canonical_table_size) {
            int mapped = canonical_table[gprim];
            if (mapped >= 0) gprim = mapped;
        }

        // FNV-1a hash step
        hash ^= static_cast<uint64_t>(static_cast<uint32_t>(gprim));
        hash *= 1099511628211ULL;
    }
    // Encode bounce count into key to separate different-depth paths
    hash ^= static_cast<uint64_t>(bc) << 56;

    out_keys[i] = hash;
    out_ray_indices[i] = i;
}
```

### Step 3: Sort by Key

CUB radix sort — pairs `(key, ray_index)`:

```cpp
cub::DeviceRadixSort::SortPairs(
    temp, temp_size,
    keys_in, keys_out,
    ray_indices_in, ray_indices_out,
    n_rays, 0, 64, stream);
```

After sort, rays with identical chain hashes are contiguous.

### Step 4: Find Unique Runs + Image Source Clustering

New CUDA kernel: `reflection_dedup_find_unique`

Two-pass approach:

**Pass A** — Mark run boundaries (where key changes):
```cuda
__global__ void reflection_dedup_mark_boundaries(
    int n_rays,
    const uint64_t *sorted_keys,
    int *out_boundary_flags    // [N] — 1 at first element of each run
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;
    if (sorted_keys[i] == UINT64_MAX) {
        out_boundary_flags[i] = 0;
        return;
    }
    out_boundary_flags[i] = (i == 0 || sorted_keys[i] != sorted_keys[i-1]) ? 1 : 0;
}
```

**CUB prefix sum** to assign group IDs:
```cpp
cub::DeviceScan::ExclusiveSum(temp, temp_size,
    boundary_flags, group_ids, n_active, stream);
```

**Pass B** — Image source clustering within each hash group.

Within a hash group, rays might have slightly different image sources
(same face sequence, different numerical image source due to floating
point).  We sub-cluster using quantized image source coordinates:

```cuda
__global__ void reflection_dedup_sub_cluster(
    int n_rays,
    int max_bounces,
    const uint64_t *sorted_keys,
    const int *sorted_ray_indices,
    const int *hash_group_ids,       // [N] from prefix sum
    const int *bounce_count,         // [N] original
    const float *img_x,              // [N*B] original
    const float *img_y,
    const float *img_z,
    float tolerance,
    uint64_t *out_cluster_keys,      // [N] — (hash_group_id, quantized_img_src)
    int *out_cluster_ray_indices     // [N]
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;
    if (sorted_keys[i] == UINT64_MAX) {
        out_cluster_keys[i] = UINT64_MAX;
        out_cluster_ray_indices[i] = sorted_ray_indices[i];
        return;
    }

    int ray_idx = sorted_ray_indices[i];
    int bc = bounce_count[ray_idx];
    int last_slot = ray_idx * max_bounces + (bc > 0 ? bc - 1 : 0);

    // Quantize final image source to grid
    float inv_tol = 1.0f / fmaxf(tolerance, 1e-12f);
    int qx = __float2int_rn(img_x[last_slot] * inv_tol);
    int qy = __float2int_rn(img_y[last_slot] * inv_tol);
    int qz = __float2int_rn(img_z[last_slot] * inv_tol);

    // Pack: upper 32 bits = hash_group_id, lower 32 = spatial hash
    uint32_t spatial = static_cast<uint32_t>(qx * 73856093u ^ qy * 19349663u ^ qz * 83492791u);
    uint64_t key = (static_cast<uint64_t>(hash_group_ids[i]) << 32) | spatial;

    out_cluster_keys[i] = key;
    out_cluster_ray_indices[i] = ray_idx;
}
```

**Second CUB radix sort** on cluster_keys to group identical clusters.

**Mark final unique boundaries + prefix sum** → final unique path IDs.

### Step 5: Compact Unique Paths

New CUDA kernel: `reflection_dedup_compact`

For each unique path, pick the first ray in the cluster as representative
and copy its data to the compacted output:

```cuda
__global__ void reflection_dedup_compact(
    int n_rays,
    int max_bounces,
    const uint64_t *final_sorted_keys,
    const int *final_sorted_ray_indices,
    const int *unique_path_ids,         // [N] from prefix sum
    // Raw trace output (indexed by original ray index)
    const int *raw_bounce_count,
    const int *raw_shape_ids,
    const int *raw_prim_ids,
    const float *raw_t,
    const float *raw_bary_u,
    const float *raw_bary_v,
    const float *raw_hit_x, const float *raw_hit_y, const float *raw_hit_z,
    const float *raw_norm_x, const float *raw_norm_y, const float *raw_norm_z,
    const float *raw_img_x, const float *raw_img_y, const float *raw_img_z,
    // Compacted output (indexed by unique_path_id)
    int *out_n_unique,                  // [1] atomic counter
    int *out_bounce_count,              // [M]
    int *out_shape_ids,                 // [M*B]
    int *out_prim_ids,                  // [M*B]
    float *out_t,                       // [M*B]
    float *out_bary_u,                  // [M*B]
    float *out_bary_v,                  // [M*B]
    float *out_hit_x, float *out_hit_y, float *out_hit_z,   // [M*B]
    float *out_norm_x, float *out_norm_y, float *out_norm_z, // [M*B]
    float *out_img_x, float *out_img_y, float *out_img_z,   // [M*B]
    int *out_discovery_count            // [M]
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;
    if (final_sorted_keys[i] == UINT64_MAX) return;

    // Am I the first element in my unique group?
    bool is_representative = (i == 0 || unique_path_ids[i] != unique_path_ids[i-1]);
    if (!is_representative) {
        // Count contribution only
        // (discovery_count incremented atomically by representative scan below)
        return;
    }

    int uid = unique_path_ids[i];
    int ray_idx = final_sorted_ray_indices[i];
    int bc = raw_bounce_count[ray_idx];

    out_bounce_count[uid] = bc;

    // Count rays in this cluster
    int count = 1;
    for (int j = i + 1; j < n_rays; ++j) {
        if (unique_path_ids[j] != uid) break;
        if (final_sorted_keys[j] == UINT64_MAX) break;
        count++;
    }
    out_discovery_count[uid] = count;

    // Copy representative data
    int src_base = ray_idx * max_bounces;
    int dst_base = uid * max_bounces;
    for (int b = 0; b < bc; ++b) {
        int s = src_base + b;
        int d = dst_base + b;
        out_shape_ids[d] = raw_shape_ids[s];
        out_prim_ids[d]  = raw_prim_ids[s];
        out_t[d]         = raw_t[s];
        out_bary_u[d]    = raw_bary_u[s];
        out_bary_v[d]    = raw_bary_v[s];
        out_hit_x[d] = raw_hit_x[s]; out_hit_y[d] = raw_hit_y[s]; out_hit_z[d] = raw_hit_z[s];
        out_norm_x[d] = raw_norm_x[s]; out_norm_y[d] = raw_norm_y[s]; out_norm_z[d] = raw_norm_z[s];
        out_img_x[d] = raw_img_x[s]; out_img_y[d] = raw_img_y[s]; out_img_z[d] = raw_img_z[s];
    }
}
```

### Step 6: Read Back Unique Count

A single `cudaMemcpyAsync` of 4 bytes (the prefix-sum total) to get `M`.
This is the only device→host transfer in the entire pipeline.

---

## Complete Kernel Launch Sequence

```
optixLaunch           (raygen: N threads, multi-bounce trace)
  │
  ▼ raw output: [N * B]
build_keys            (1 kernel: N threads)
  │
  ▼ keys: [N], ray_indices: [N]
CUB RadixSort::SortPairs
  │
  ▼ sorted_keys, sorted_ray_indices
mark_boundaries       (1 kernel: N threads)
  │
  ▼ boundary_flags: [N]
CUB Scan::ExclusiveSum
  │
  ▼ hash_group_ids: [N]
sub_cluster           (1 kernel: N threads)
  │
  ▼ cluster_keys: [N], cluster_ray_indices: [N]
CUB RadixSort::SortPairs
  │
  ▼ final_sorted_keys, final_sorted_ray_indices
mark_boundaries       (1 kernel: N threads, reuse)
  │
  ▼ unique_boundary_flags: [N]
CUB Scan::ExclusiveSum
  │
  ▼ unique_path_ids: [N], last element = M-1
compact               (1 kernel: N threads)
  │
  ▼ compacted output: [M * B]
cudaMemcpyAsync       (4 bytes: read M)
```

Total: **1 OptiX launch + 4 CUDA kernels + 2 CUB sorts + 2 CUB scans + 1 tiny D2H**.

All on the same CUDA stream, fully pipelined. No CPU-GPU synchronization
until the final 4-byte read of `M`.

---

## New Files

```
src/scene/
    reflection_dedup.cu     — CUDA kernels (build_keys, mark_boundaries,
                               sub_cluster, compact)
    reflection_dedup.h      — Host-callable function declarations

include/rayd/
    reflection.h            — Add discovery_count, plane_points, plane_normals
                               to ReflectionChainData (modify)
```

### reflection_dedup.h

```cpp
#pragma once
#include <cstddef>

namespace rayd {

/// Run GPU deduplication on raw reflection trace output.
/// Returns the number of unique paths found (M).
///
/// All pointer arguments are device pointers.
/// Output arrays must be pre-allocated to at least n_rays * max_bounces
/// (compaction will only write to [0..M*max_bounces)).
int reflection_dedup_gpu(
    int n_rays,
    int max_bounces,
    // Raw trace output (input, [N] and [N*B])
    const int *bounce_count,
    const int *shape_ids,
    const int *prim_ids,
    const float *t,
    const float *bary_u,
    const float *bary_v,
    const float *hit_x, const float *hit_y, const float *hit_z,
    const float *norm_x, const float *norm_y, const float *norm_z,
    const float *img_x, const float *img_y, const float *img_z,
    // Dedup config
    const int *face_offsets,         // [n_meshes]
    int n_meshes,
    const int *canonical_prim_table, // [T] or nullptr
    int canonical_table_size,
    float image_source_tolerance,
    // Compacted output (pre-allocated to [N] and [N*B], only [M]/[M*B] written)
    int *out_bounce_count,           // [M]
    int *out_shape_ids,              // [M*B]
    int *out_prim_ids,               // [M*B]
    float *out_t,                    // [M*B]
    float *out_bary_u,               // [M*B]
    float *out_bary_v,               // [M*B]
    float *out_hit_x, float *out_hit_y, float *out_hit_z,
    float *out_norm_x, float *out_norm_y, float *out_norm_z,
    float *out_img_x, float *out_img_y, float *out_img_z,
    int *out_discovery_count         // [M]
);

}  // namespace rayd
```

---

## Host Integration (scene.cpp)

Inside `Scene::trace_reflections()`, after `reflection_pipeline_->launch(params)`:

```cpp
if (options.deduplicate) {
    // Allocate compacted output buffers (same size as raw — only [M] used)
    ReflectionTraceRaw compacted;
    compacted.max_bounces = max_bounces;
    // ... allocate same as raw ...
    IntDetached discovery_count = empty<IntDetached>(ray_count);

    int n_unique = reflection_dedup_gpu(
        ray_count, max_bounces,
        // raw input
        raw.bounce_count.data(), raw.shape_ids.data(), raw.prim_ids.data(),
        raw.t.data(), raw.bary_u.data(), raw.bary_v.data(),
        raw.hit_x.data(), raw.hit_y.data(), raw.hit_z.data(),
        raw.norm_x.data(), raw.norm_y.data(), raw.norm_z.data(),
        raw.img_x.data(), raw.img_y.data(), raw.img_z.data(),
        // config
        face_offsets_.data(), mesh_count_,
        options.canonical_prim_table, options.canonical_prim_table_size,
        options.image_source_tolerance,
        // compacted output
        compacted.bounce_count.data(), compacted.shape_ids.data(),
        compacted.prim_ids.data(),
        compacted.t.data(), compacted.bary_u.data(), compacted.bary_v.data(),
        compacted.hit_x.data(), compacted.hit_y.data(), compacted.hit_z.data(),
        compacted.norm_x.data(), compacted.norm_y.data(), compacted.norm_z.data(),
        compacted.img_x.data(), compacted.img_y.data(), compacted.img_z.data(),
        discovery_count.data()
    );

    // Resize DrJit arrays to actual unique count
    const int M = n_unique;
    const int MB = M * max_bounces;
    result.ray_count = M;  // now M unique paths, not N input rays
    result.bounce_count  = head<IntDetached>(compacted.bounce_count, M);
    result.shape_ids     = head<IntDetached>(compacted.shape_ids, MB);
    result.prim_ids      = head<IntDetached>(compacted.prim_ids, MB);
    result.t             = head<FloatDetached>(compacted.t, MB);
    result.hit_points    = head<Vector3fDetached>(
        Vector3fDetached(compacted.hit_x, compacted.hit_y, compacted.hit_z), MB);
    result.geo_normals   = head<Vector3fDetached>(
        Vector3fDetached(compacted.norm_x, compacted.norm_y, compacted.norm_z), MB);
    result.image_sources = head<Vector3fDetached>(
        Vector3fDetached(compacted.img_x, compacted.img_y, compacted.img_z), MB);
    result.discovery_count = head<IntDetached>(discovery_count, M);
}
```

---

## CMake Changes

Add `reflection_dedup.cu` to the existing CUDA compilation alongside `edge_bvh.cu`:

```cmake
# Compile reflection_dedup.cu to object file (same pattern as edge_bvh.cu)
set(RAYD_REFLECTION_DEDUP_CU
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/reflection_dedup.cu")
set(RAYD_REFLECTION_DEDUP_OBJ
    "${CMAKE_CURRENT_BINARY_DIR}/reflection_dedup${CMAKE_CXX_OUTPUT_EXTENSION}")

# Windows
add_custom_command(OUTPUT "${RAYD_REFLECTION_DEDUP_OBJ}"
    COMMAND "${CUDA_NVCC_EXECUTABLE}" --extended-lambda -std=c++17 -c
        "${RAYD_REFLECTION_DEDUP_CU}"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        -Xcompiler "/MD /O2 /EHsc /wd4819"
        -o "${RAYD_REFLECTION_DEDUP_OBJ}"
    DEPENDS "${RAYD_REFLECTION_DEDUP_CU}" ...)

target_sources(rayd_core PRIVATE "${RAYD_REFLECTION_DEDUP_OBJ}")
```

---

## Python Binding Changes

### nanobind (rayd.cpp)

```cpp
nb::class_<ReflectionTraceOptions>(m, "ReflectionTraceOptions")
    .def(nb::init<>())
    .def_rw("deduplicate", &ReflectionTraceOptions::deduplicate)
    .def_rw("image_source_tolerance", &ReflectionTraceOptions::image_source_tolerance)
    .def_rw("canonical_prim_table", ...)
    .def_rw("canonical_prim_table_size", ...);

// Existing overloads unchanged, add new:
.def("trace_reflections",
     [](const Scene &s, const RayDetached &ray, int max_bounces,
        const ReflectionTraceOptions &opts, MaskDetached active) {
         return s.trace_reflections<true>(ray, max_bounces, opts, active);
     },
     nb::arg("ray").noconvert(), "max_bounces"_a,
     "options"_a, "active"_a = true)
```

### PyTorch (scene.py)

```python
def trace_reflections(self, ray, max_bounces, *,
                      deduplicate=False,
                      canonical_prim_table=None,
                      image_source_tolerance=1e-5,
                      active=True):
    ...
```

---

## witwin.channel Integration

Replace `_trace_reflection_paths` + `_collect_unique_reflection_paths`:

```python
def _trace_reflection_paths_rayd(*, tx_pos, scene, n_rays, max_reflections, ...):
    """Single GPU call replaces Python bounce loop + Python dedup."""

    ray_dir = _select_reflection_ray_directions(...)
    ray = rd.RayDetached(broadcast_point(tx_pos, n_rays), ray_dir)

    # One call: trace + dedup, fully on GPU
    chain = scene._rayd_scene.trace_reflections(
        ray, max_reflections,
        rd.ReflectionTraceOptions(
            deduplicate=True,
            canonical_prim_table=scene.tri_data_gpu["surface_canonical_prim"],
            image_source_tolerance=1e-5,
        ),
    )

    # chain.ray_count is now M (unique paths), not N (input rays)
    # chain.prim_ids is [M*B], chain.image_sources is [M*B]
    # chain.discovery_count is [M]
    # → directly convert to source_paths_per_bounce format
    return _chain_to_source_paths(chain, max_reflections)
```

---

## Performance

| Step | Kernel launches | Sync points |
|------|----------------|-------------|
| OptiX trace | 1 | 0 |
| build_keys | 1 | 0 |
| CUB sort #1 | ~3 (internal) | 0 |
| mark_boundaries | 1 | 0 |
| CUB prefix sum #1 | ~2 | 0 |
| sub_cluster | 1 | 0 |
| CUB sort #2 | ~3 | 0 |
| mark_boundaries | 1 | 0 |
| CUB prefix sum #2 | ~2 | 0 |
| compact | 1 | 0 |
| read M (4 bytes) | 0 | **1** |
| **Total** | **~16** | **1** |

vs. current Python pipeline: **~60 kernel launches + ~9 CPU-GPU syncs**.

Expected timing (100K rays, 3 bounces, 4-box scene):

| | Current | OptiX raygen + GPU dedup |
|---|---|---|
| Trace | 90 ms | ~3 ms |
| Dedup | 5 ms (Python) | ~1 ms (GPU) |
| Total | **95 ms** | **~4 ms** |
| Speedup | | **~24x** |

### End-to-end 5TX x 5RX multi-frame

```
Current naive:                375 ms/frame    2.7 FPS
OptiX raygen + GPU dedup:     ~35 ms/frame   ~29 FPS
+ diffraction native kernel:  ~20 ms/frame   ~50 FPS
```
