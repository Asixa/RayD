#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

namespace rayd::shared::multipath {

struct ReflectionDedupBuildKeysParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::int32_t *bounce_count;
    const std::int32_t *shape_ids;
    const std::int32_t *prim_ids;
    const std::int32_t *face_offsets;
    std::int32_t mesh_count;
    const std::int32_t *canonical_table;
    std::int32_t canonical_table_size;
    std::uint64_t *out_keys;
    std::int32_t *out_ray_indices;
    cudaStream_t stream;
};

struct ReflectionDedupSubClusterParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::uint64_t *sorted_keys;
    const std::int32_t *sorted_ray_indices;
    const std::int32_t *hash_group_ids;
    const std::int32_t *bounce_count;
    const float *image_x;
    const float *image_y;
    const float *image_z;
    float tolerance;
    std::uint64_t *out_cluster_keys;
    std::int32_t *out_cluster_ray_indices;
    cudaStream_t stream;
};

struct ReflectionDedupCompactParams {
    std::int32_t ray_count;
    std::int32_t max_bounces;
    const std::uint64_t *sorted_keys;
    const std::int32_t *sorted_ray_indices;
    const std::int32_t *unique_path_ids;
    const std::int32_t *raw_bounce_count;
    const std::int32_t *raw_shape_ids;
    const std::int32_t *raw_prim_ids;
    const float *raw_t;
    const float *raw_bary_u;
    const float *raw_bary_v;
    const float *raw_hit_x;
    const float *raw_hit_y;
    const float *raw_hit_z;
    const float *raw_norm_x;
    const float *raw_norm_y;
    const float *raw_norm_z;
    const float *raw_image_x;
    const float *raw_image_y;
    const float *raw_image_z;
    std::int32_t *out_unique_count;
    std::int32_t *out_bounce_count;
    std::int32_t *out_shape_ids;
    std::int32_t *out_prim_ids;
    float *out_t;
    float *out_bary_u;
    float *out_bary_v;
    float *out_hit_x;
    float *out_hit_y;
    float *out_hit_z;
    float *out_norm_x;
    float *out_norm_y;
    float *out_norm_z;
    float *out_image_x;
    float *out_image_y;
    float *out_image_z;
    std::int32_t *out_discovery_count;
    std::int32_t *out_representative_ray_index;
    cudaStream_t stream;
};

/// Enqueue reflection-path hash construction; no validation, allocation, or synchronization.
void launch_reflection_dedup_build_keys(const ReflectionDedupBuildKeysParams &params);

/// Enqueue marking of valid sorted-key run boundaries around an inclusive scan.
void launch_reflection_dedup_mark_boundaries(
    std::int32_t ray_count,
    const std::uint64_t *sorted_keys,
    std::int32_t *out_boundary_flags,
    cudaStream_t stream);

/// Enqueue conversion of inclusive boundary counts to zero-based path IDs.
void launch_reflection_dedup_zero_base_ids(
    std::int32_t ray_count,
    const std::uint64_t *sorted_keys,
    std::int32_t *inout_ids,
    cudaStream_t stream);

/// Enqueue image-source spatial sub-cluster key construction.
void launch_reflection_dedup_sub_cluster(const ReflectionDedupSubClusterParams &params);

/// Enqueue representative-path compaction into caller-owned output arrays.
void launch_reflection_dedup_compact(const ReflectionDedupCompactParams &params);

static_assert(std::is_standard_layout_v<ReflectionDedupBuildKeysParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupBuildKeysParams>);
static_assert(std::is_standard_layout_v<ReflectionDedupSubClusterParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupSubClusterParams>);
static_assert(std::is_standard_layout_v<ReflectionDedupCompactParams>);
static_assert(std::is_trivially_copyable_v<ReflectionDedupCompactParams>);

} // namespace rayd::shared::multipath
