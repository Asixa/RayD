// Copyright Xingyu Chen.
// Defines shared reflection support for dedup.

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
