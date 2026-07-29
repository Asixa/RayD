// Copyright Xingyu Chen.
// Defines internal reflection launch contracts and backend adapters.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

namespace rayd::shared::multipath {

struct ReflectionDedupBuildKeysParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::int32_t* bounce_count;
    const std::int32_t* shape_ids;
    const std::int32_t* prim_ids;
    const std::int32_t* face_offsets;
    std::int32_t mesh_count;
    const std::int32_t* canonical_table;
    std::int32_t canonical_table_size;
    std::uint64_t* out_keys;
    std::int32_t* out_ray_indices;
    cudaStream_t stream;
};

struct ReflectionDedupSubClusterParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::uint64_t* sorted_keys;
    const std::int32_t* sorted_ray_indices;
    const std::int32_t* hash_group_ids;
    const std::int32_t* bounce_count;
    const float* image_x;
    const float* image_y;
    const float* image_z;
    float tolerance;
    std::uint64_t* out_cluster_keys;
    std::int32_t* out_cluster_ray_indices;
    cudaStream_t stream;
};

struct ReflectionDedupCompactParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::uint64_t* sorted_keys;
    const std::int32_t* sorted_ray_indices;
    const std::int32_t* unique_path_ids;
    const std::int32_t* raw_bounce_count;
    const std::int32_t* raw_shape_ids;
    const std::int32_t* raw_prim_ids;
    const float* raw_t;
    const float* raw_bary_u;
    const float* raw_bary_v;
    const float* raw_hit_x;
    const float* raw_hit_y;
    const float* raw_hit_z;
    const float* raw_norm_x;
    const float* raw_norm_y;
    const float* raw_norm_z;
    const float* raw_image_x;
    const float* raw_image_y;
    const float* raw_image_z;
    std::int32_t* out_unique_count;
    std::int32_t* out_bounce_count;
    std::int32_t* out_shape_ids;
    std::int32_t* out_prim_ids;
    float* out_t;
    float* out_bary_u;
    float* out_bary_v;
    float* out_hit_x;
    float* out_hit_y;
    float* out_hit_z;
    float* out_norm_x;
    float* out_norm_y;
    float* out_norm_z;
    float* out_image_x;
    float* out_image_y;
    float* out_image_z;
    std::int32_t* out_discovery_count;
    std::int32_t* out_representative_ray_index;
    cudaStream_t stream;
};

/// Enqueue reflection-path hash construction; no validation, allocation, or synchronization.
void launch_reflection_dedup_build_keys(const ReflectionDedupBuildKeysParams& params);

/// Enqueue marking of valid sorted-key run boundaries around an inclusive scan.
void launch_reflection_dedup_mark_boundaries(std::int32_t ray_count, const std::uint64_t* sorted_keys,
                                             std::int32_t* out_boundary_flags, cudaStream_t stream);

/// Enqueue conversion of inclusive boundary counts to zero-based path IDs.
void launch_reflection_dedup_zero_base_ids(std::int32_t ray_count, const std::uint64_t* sorted_keys,
                                           std::int32_t* inout_ids, cudaStream_t stream);

/// Enqueue image-source spatial sub-cluster key construction.
void launch_reflection_dedup_sub_cluster(const ReflectionDedupSubClusterParams& params);

/// Enqueue representative-path compaction into caller-owned output arrays.
void launch_reflection_dedup_compact(const ReflectionDedupCompactParams& params);

/// Ordered failure sites of the fused dedup sequence. Reported alongside the
/// failing CUDA result so each caller can attach its own per-step error
/// message; this layer never formats or raises errors itself.
enum class ReflectionDedupSequenceStep : std::int32_t {
    kNone = 0,
    kBuildKeys,
    kFirstSort,
    kFirstBoundaries,
    kFirstScan,
    kFirstZeroBase,
    kSubCluster,
    kSecondSort,
    kSecondBoundaries,
    kSecondScan,
    kSecondZeroBase,
    kCompact,
};

/// First failing CUDA result and the step it happened at;
/// {cudaSuccess, kNone} on success.
struct ReflectionDedupSequenceStatus {
    cudaError_t error;
    ReflectionDedupSequenceStep step;
};

/// Sort/scan passes the sequence delegates back to its caller. The caller owns
/// the sort/scan provider (and therefore keeps those template kernels
/// instantiated in its own translation unit, so this layer moves no device
/// code); the sequence owns only the pass order.
enum class ReflectionDedupDevicePass : std::int32_t {
    kFirstSort = 0,
    kFirstScan,
    kSecondSort,
    kSecondScan,
};

/// Full input/scratch/output pointer set for the ten-step dedup sequence.
/// Every buffer is caller-owned and sized for ray_count elements (scratch) or
/// the caller's output capacity; the temp pointers hold at least the byte
/// counts the caller's own sort/scan sizing queries reported for ray_count.
struct ReflectionDedupSequenceParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::int32_t* bounce_count;
    const std::int32_t* shape_ids;
    const std::int32_t* prim_ids;
    const std::int32_t* face_offsets;
    std::int32_t mesh_count;
    const std::int32_t* canonical_table;
    std::int32_t canonical_table_size;
    float image_source_tolerance;
    const float* raw_t;
    const float* raw_bary_u;
    const float* raw_bary_v;
    const float* raw_hit_x;
    const float* raw_hit_y;
    const float* raw_hit_z;
    const float* raw_norm_x;
    const float* raw_norm_y;
    const float* raw_norm_z;
    const float* raw_image_x;
    const float* raw_image_y;
    const float* raw_image_z;
    std::uint64_t* keys_in;
    std::uint64_t* keys_out;
    std::int32_t* ray_indices_in;
    std::int32_t* ray_indices_out;
    std::int32_t* boundary_flags;
    std::int32_t* hash_group_ids;
    std::uint64_t* cluster_keys_in;
    std::uint64_t* cluster_keys_out;
    std::int32_t* cluster_ray_indices_in;
    std::int32_t* cluster_ray_indices_out;
    std::int32_t* unique_path_ids;
    void* sort_temp;
    std::size_t sort_temp_bytes;
    void* scan_temp;
    std::size_t scan_temp_bytes;
    void* cluster_sort_temp;
    std::size_t cluster_sort_temp_bytes;
    std::int32_t* out_unique_count;
    std::int32_t* out_bounce_count;
    std::int32_t* out_shape_ids;
    std::int32_t* out_prim_ids;
    float* out_t;
    float* out_bary_u;
    float* out_bary_v;
    float* out_hit_x;
    float* out_hit_y;
    float* out_hit_z;
    float* out_norm_x;
    float* out_norm_y;
    float* out_norm_z;
    float* out_image_x;
    float* out_image_y;
    float* out_image_z;
    std::int32_t* out_discovery_count;
    std::int32_t* out_representative_ray_index;
    /// Enqueues one delegated sort/scan pass on stream and returns its result.
    cudaError_t (*run_pass)(const ReflectionDedupSequenceParams& params, ReflectionDedupDevicePass pass);
    cudaStream_t stream;
};

/// Enqueue the ten-step dedup sequence (build_keys, radix sort, mark, scan,
/// zero_base, sub_cluster, radix sort, mark, scan, compact) on params.stream,
/// delegating the sort/scan passes to params.run_pass. Enqueue-only: no
/// validation, allocation, host copy, or synchronization; the caller owns
/// every buffer plus the final count readback.
ReflectionDedupSequenceStatus launch_reflection_dedup_sequence(const ReflectionDedupSequenceParams& params);

static_assert(std::is_standard_layout_v<ReflectionDedupBuildKeysParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupBuildKeysParams>);
static_assert(std::is_standard_layout_v<ReflectionDedupSubClusterParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupSubClusterParams>);
static_assert(std::is_standard_layout_v<ReflectionDedupCompactParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupCompactParams>);
static_assert(std::is_standard_layout_v<ReflectionDedupSequenceStatus>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupSequenceStatus>);
static_assert(std::is_standard_layout_v<ReflectionDedupSequenceParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupSequenceParams>);

} // namespace rayd::shared::multipath

#include <cstdint>
#include <type_traits>

#include <vector_types.h>

namespace rayd::shared::optix {

/// Backend-neutral reflection trace launch layout. Buffers and pipeline ownership stay in adapters.
struct ReflectionTraceParams {
    std::uint64_t primary_handle = 0;
    std::uint64_t secondary_handle = 0;
    int split_mode = 0;

    const float* tri_p0_x = nullptr;
    const float* tri_p0_y = nullptr;
    const float* tri_p0_z = nullptr;
    const float* tri_e1_x = nullptr;
    const float* tri_e1_y = nullptr;
    const float* tri_e1_z = nullptr;
    const float* tri_e2_x = nullptr;
    const float* tri_e2_y = nullptr;
    const float* tri_e2_z = nullptr;
    const float* tri_fn_x = nullptr;
    const float* tri_fn_y = nullptr;
    const float* tri_fn_z = nullptr;
    const float4* tri_p0_packed = nullptr;
    const float4* tri_e1_packed = nullptr;
    const float4* tri_e2_packed = nullptr;
    const float4* tri_fn_packed = nullptr;

    const int* face_offsets = nullptr;
    int n_meshes = 0;
    int n_triangles = 0;

    const float* ray_ox = nullptr;
    const float* ray_oy = nullptr;
    const float* ray_oz = nullptr;
    const float* ray_dx = nullptr;
    const float* ray_dy = nullptr;
    const float* ray_dz = nullptr;
    const float* ray_o_aos = nullptr;
    const float* ray_d_aos = nullptr;
    const float* ray_tmax = nullptr;
    const std::uint8_t* active_mask = nullptr;
    int n_rays = 0;
    int max_bounces = 0;
    int export_mode = 0;
    int return_trailing = 0;
    int output_layout = 0;

    std::uint8_t* out_valid = nullptr;
    int* out_bounce_count = nullptr;
    int* out_shape_ids = nullptr;
    int* out_prim_ids = nullptr;
    int* out_global_prim_ids = nullptr;
    float* out_t = nullptr;
    float* out_bary_u = nullptr;
    float* out_bary_v = nullptr;
    float* out_bary = nullptr;
    float* out_hit_x = nullptr;
    float* out_hit_y = nullptr;
    float* out_hit_z = nullptr;
    float* out_hit = nullptr;
    float* out_norm_x = nullptr;
    float* out_norm_y = nullptr;
    float* out_norm_z = nullptr;
    float* out_norm = nullptr;
    float* out_img_x = nullptr;
    float* out_img_y = nullptr;
    float* out_img_z = nullptr;
    float* out_img = nullptr;
    float* out_trailing_t = nullptr;
    int* out_trailing_prim = nullptr;
    float* out_trailing_dir_x = nullptr;
    float* out_trailing_dir_y = nullptr;
    float* out_trailing_dir_z = nullptr;
    float* out_trailing_origin_x = nullptr;
    float* out_trailing_origin_y = nullptr;
    float* out_trailing_origin_z = nullptr;
};

static_assert(std::is_standard_layout_v<ReflectionTraceParams>);
static_assert(std::is_trivially_copyable_v<ReflectionTraceParams>);

} // namespace rayd::shared::optix

#include <cstdint>
#include <type_traits>

namespace rayd::shared::optix {

constexpr int ReflEpcMaxBounces = 8;
constexpr int ReflEpcVisibilityIgnorePrimitive = 0;
constexpr int ReflEpcVisibilityIgnoreSurfaceGroup = 1;

/// Exact shared EPC params/result/tape launch layout used by both native backends.
struct ReflEpcParams {
    std::uint64_t primary_handle;
    std::uint64_t secondary_handle;
    int split_mode;
    const float* tri_p0_x;
    const float* tri_p0_y;
    const float* tri_p0_z;
    const float* tri_e1_x;
    const float* tri_e1_y;
    const float* tri_e1_z;
    const float* tri_e2_x;
    const float* tri_e2_y;
    const float* tri_e2_z;
    const float* tri_fn_x;
    const float* tri_fn_y;
    const float* tri_fn_z;
    const int* face_offsets;
    int n_meshes;
    int n_triangles;
    const int* expected_prim_ids;
    int expected_prim_count;
    const int* surface_group_id;
    int surface_group_id_count;
    const int* surface_group_size;
    int surface_group_count;
    const int* surface_group_members;
    int surface_max_group_size;
    int visibility_ignore_mode;
    const int* final_ignore_group_ids;
    int final_ignore_group_count;
    const float* ray_ox;
    const float* ray_oy;
    const float* ray_oz;
    const float* ray_dx;
    const float* ray_dy;
    const float* ray_dz;
    const float* ray_tmax;
    const float* direct_plane_point_x;
    const float* direct_plane_point_y;
    const float* direct_plane_point_z;
    const float* direct_plane_normal_x;
    const float* direct_plane_normal_y;
    const float* direct_plane_normal_z;
    const float* rx_x;
    const float* rx_y;
    const float* rx_z;
    int rx_count;
    const std::uint8_t* active_mask;
    int n_rays;
    int max_bounces;
    float plane_tolerance;
    std::uint8_t* out_valid;
    int* out_bounce_count;
    float* out_path_length;
    float* out_point_x;
    float* out_point_y;
    float* out_point_z;
    int* out_trace_prim_ids;
    int* out_resolved_prim_ids;
    int* out_surface_group_ids;
    float* out_plane_normal_x;
    float* out_plane_normal_y;
    float* out_plane_normal_z;
    int* out_first_blocked_segment;
    int* out_first_blocked_prim;
    int* out_first_blocked_group;
};

static_assert(std::is_standard_layout_v<ReflEpcParams>);
static_assert(std::is_trivially_copyable_v<ReflEpcParams>);

} // namespace rayd::shared::optix

namespace rayd::torch_backend {

/// \brief Deduplicate reflection paths that share the same reflector sequence and image source.
///
/// Reads the ray-major (n_rays * max_bounces) trace arrays, collapses duplicate paths,
/// and writes the kept paths plus a per-kept discovery_count and representative ray index
/// into the caller-owned out_* buffers. All pointers are flat device arrays.
///
/// \p device_index is the CUDA device that owns every pointer below; the dedup
/// launches, temporary allocations, and stream all follow it.
///
/// \return Number of unique paths kept.
int reflection_dedup_gpu(int device_index, int n_rays, int max_bounces, const int* bounce_count, const int* shape_ids,
                         const int* prim_ids, const float* t, const float* bary_u, const float* bary_v,
                         const float* hit_x, const float* hit_y, const float* hit_z, const float* norm_x,
                         const float* norm_y, const float* norm_z, const float* img_x, const float* img_y,
                         const float* img_z, const int* face_offsets, int n_meshes, const int* canonical_prim_table,
                         int canonical_table_size, float image_source_tolerance, int* out_bounce_count,
                         int* out_shape_ids, int* out_prim_ids, float* out_t, float* out_bary_u, float* out_bary_v,
                         float* out_hit_x, float* out_hit_y, float* out_hit_z, float* out_norm_x, float* out_norm_y,
                         float* out_norm_z, float* out_img_x, float* out_img_y, float* out_img_z,
                         int* out_discovery_count, int* out_representative_ray_index);

} // namespace rayd::torch_backend

namespace rayd {

/// \brief Deduplicate reflection paths that share the same reflector sequence and image source.
///
/// Reads the ray-major (n_rays * max_bounces) trace arrays, collapses duplicate paths,
/// and writes the kept paths plus a per-kept discovery_count and representative ray index
/// into the caller-owned out_* buffers. All pointers are flat device arrays.
///
/// \return Number of unique paths kept.
int reflection_dedup_gpu(int n_rays, int max_bounces, const int* bounce_count, const int* shape_ids,
                         const int* prim_ids, const float* t, const float* bary_u, const float* bary_v,
                         const float* hit_x, const float* hit_y, const float* hit_z, const float* norm_x,
                         const float* norm_y, const float* norm_z, const float* img_x, const float* img_y,
                         const float* img_z, const int* face_offsets, int n_meshes, const int* canonical_prim_table,
                         int canonical_table_size, float image_source_tolerance, int* out_bounce_count,
                         int* out_shape_ids, int* out_prim_ids, float* out_t, float* out_bary_u, float* out_bary_v,
                         float* out_hit_x, float* out_hit_y, float* out_hit_z, float* out_norm_x, float* out_norm_y,
                         float* out_norm_z, float* out_img_x, float* out_img_y, float* out_img_z,
                         int* out_discovery_count, int* out_representative_ray_index);

} // namespace rayd

namespace rayd::torch_backend {

using ReflectionTraceParams = shared::optix::ReflectionTraceParams;

} // namespace rayd::torch_backend

namespace rayd {

using ReflectionTraceParams = shared::optix::ReflectionTraceParams;

} // namespace rayd

namespace rayd::torch_backend {

using ReflEpcParams = shared::optix::ReflEpcParams;
inline constexpr int ReflEpcMaxBounces = shared::optix::ReflEpcMaxBounces;
inline constexpr int ReflEpcVisibilityIgnorePrimitive = shared::optix::ReflEpcVisibilityIgnorePrimitive;
inline constexpr int ReflEpcVisibilityIgnoreSurfaceGroup = shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;

} // namespace rayd::torch_backend

namespace rayd {

using ReflEpcParams = shared::optix::ReflEpcParams;
inline constexpr int ReflEpcMaxBounces = shared::optix::ReflEpcMaxBounces;
inline constexpr int ReflEpcVisibilityIgnorePrimitive = shared::optix::ReflEpcVisibilityIgnorePrimitive;
inline constexpr int ReflEpcVisibilityIgnoreSurfaceGroup = shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;

} // namespace rayd
