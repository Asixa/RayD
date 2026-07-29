// Copyright Xingyu Chen.
// Implements reflection support for dedup shared.

#include <rayd/reflection/dedup.h>

#include <cuda_runtime.h>

namespace rayd::shared::multipath {
namespace {

constexpr int kBlockSize = 256;

__global__ void reflection_dedup_build_keys_kernel(ReflectionDedupBuildKeysParams params) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= params.ray_count)
        return;

    const int bounce_count = params.bounce_count[i];
    params.out_ray_indices[i] = i;
    if (bounce_count <= 0) {
        params.out_keys[i] = UINT64_MAX;
        return;
    }

    std::uint64_t hash = 14695981039346656037ull;
    const int base = i * params.max_bounces;
    for (int bounce = 0; bounce < bounce_count; ++bounce) {
        const int shape_id = params.shape_ids[base + bounce];
        const int local_prim = params.prim_ids[base + bounce];
        const int face_offset = (shape_id >= 0 && shape_id < params.mesh_count) ? params.face_offsets[shape_id] : 0;
        int global_prim = face_offset + local_prim;
        if (params.canonical_table != nullptr && global_prim >= 0 && global_prim < params.canonical_table_size) {
            const int mapped = params.canonical_table[global_prim];
            if (mapped >= 0)
                global_prim = mapped;
        }
        hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(global_prim));
        hash *= 1099511628211ull;
    }
    hash ^= static_cast<std::uint64_t>(bounce_count) << 56;
    params.out_keys[i] = hash;
}

__global__ void reflection_dedup_mark_boundaries_kernel(int ray_count, const std::uint64_t* sorted_keys,
                                                        std::int32_t* out_boundary_flags) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= ray_count)
        return;
    if (sorted_keys[i] == UINT64_MAX) {
        out_boundary_flags[i] = 0;
        return;
    }
    out_boundary_flags[i] = (i == 0 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0;
}

__global__ void reflection_dedup_zero_base_ids_kernel(int ray_count, const std::uint64_t* sorted_keys,
                                                      std::int32_t* inout_ids) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < ray_count && sorted_keys[i] != UINT64_MAX)
        inout_ids[i] -= 1;
}

__global__ void reflection_dedup_sub_cluster_kernel(ReflectionDedupSubClusterParams params) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= params.ray_count)
        return;

    const int ray_index = params.sorted_ray_indices[i];
    params.out_cluster_ray_indices[i] = ray_index;
    if (params.sorted_keys[i] == UINT64_MAX) {
        params.out_cluster_keys[i] = UINT64_MAX;
        return;
    }

    const int bounce_count = params.bounce_count[ray_index];
    const int last_slot = ray_index * params.max_bounces + (bounce_count > 0 ? bounce_count - 1 : 0);
    const float inv_tolerance = 1.0f / fmaxf(params.tolerance, 1e-12f);
    const int qx = __float2int_rn(params.image_x[last_slot] * inv_tolerance);
    const int qy = __float2int_rn(params.image_y[last_slot] * inv_tolerance);
    const int qz = __float2int_rn(params.image_z[last_slot] * inv_tolerance);
    const std::uint32_t spatial = static_cast<std::uint32_t>(qx * 73856093u ^ qy * 19349663u ^ qz * 83492791u);
    params.out_cluster_keys[i] =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(params.hash_group_ids[i])) << 32) |
        static_cast<std::uint64_t>(spatial);
}

__global__ void reflection_dedup_compact_kernel(ReflectionDedupCompactParams params) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= params.ray_count || params.sorted_keys[i] == UINT64_MAX)
        return;

    const int unique_id = params.unique_path_ids[i];
    atomicAdd(params.out_discovery_count + unique_id, 1);
    if (i != 0 && params.unique_path_ids[i - 1] == unique_id)
        return;

    atomicMax(params.out_unique_count, unique_id + 1);
    const int ray_index = params.sorted_ray_indices[i];
    params.out_representative_ray_index[unique_id] = ray_index;
    const int bounce_count = params.raw_bounce_count[ray_index];
    params.out_bounce_count[unique_id] = bounce_count;
    const int source_base = ray_index * params.max_bounces;
    const int destination_base = unique_id * params.max_bounces;
    for (int bounce = 0; bounce < bounce_count; ++bounce) {
        const int source = source_base + bounce;
        const int destination = destination_base + bounce;
        params.out_shape_ids[destination] = params.raw_shape_ids[source];
        params.out_prim_ids[destination] = params.raw_prim_ids[source];
        params.out_t[destination] = params.raw_t[source];
        params.out_bary_u[destination] = params.raw_bary_u[source];
        params.out_bary_v[destination] = params.raw_bary_v[source];
        params.out_hit_x[destination] = params.raw_hit_x[source];
        params.out_hit_y[destination] = params.raw_hit_y[source];
        params.out_hit_z[destination] = params.raw_hit_z[source];
        params.out_norm_x[destination] = params.raw_norm_x[source];
        params.out_norm_y[destination] = params.raw_norm_y[source];
        params.out_norm_z[destination] = params.raw_norm_z[source];
        params.out_image_x[destination] = params.raw_image_x[source];
        params.out_image_y[destination] = params.raw_image_y[source];
        params.out_image_z[destination] = params.raw_image_z[source];
    }
}

int block_count(int count) {
    return (count + kBlockSize - 1) / kBlockSize;
}

} // namespace

void launch_reflection_dedup_build_keys(const ReflectionDedupBuildKeysParams& params) {
    if (params.ray_count == 0)
        return;
    reflection_dedup_build_keys_kernel<<<block_count(params.ray_count), kBlockSize, 0, params.stream>>>(params);
}

void launch_reflection_dedup_mark_boundaries(std::int32_t ray_count, const std::uint64_t* sorted_keys,
                                             std::int32_t* out_boundary_flags, cudaStream_t stream) {
    if (ray_count == 0)
        return;
    reflection_dedup_mark_boundaries_kernel<<<block_count(ray_count), kBlockSize, 0, stream>>>(ray_count, sorted_keys,
                                                                                               out_boundary_flags);
}

void launch_reflection_dedup_zero_base_ids(std::int32_t ray_count, const std::uint64_t* sorted_keys,
                                           std::int32_t* inout_ids, cudaStream_t stream) {
    if (ray_count == 0)
        return;
    reflection_dedup_zero_base_ids_kernel<<<block_count(ray_count), kBlockSize, 0, stream>>>(ray_count, sorted_keys,
                                                                                             inout_ids);
}

void launch_reflection_dedup_sub_cluster(const ReflectionDedupSubClusterParams& params) {
    if (params.ray_count == 0)
        return;
    reflection_dedup_sub_cluster_kernel<<<block_count(params.ray_count), kBlockSize, 0, params.stream>>>(params);
}

void launch_reflection_dedup_compact(const ReflectionDedupCompactParams& params) {
    if (params.ray_count == 0)
        return;
    reflection_dedup_compact_kernel<<<block_count(params.ray_count), kBlockSize, 0, params.stream>>>(params);
}

ReflectionDedupSequenceStatus launch_reflection_dedup_sequence(const ReflectionDedupSequenceParams& params) {
    launch_reflection_dedup_build_keys({params.ray_count, params.max_bounces, params.bounce_count, params.shape_ids,
                                        params.prim_ids, params.face_offsets, params.mesh_count, params.canonical_table,
                                        params.canonical_table_size, params.keys_in, params.ray_indices_in,
                                        params.stream});
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kBuildKeys};

    error = params.run_pass(params, ReflectionDedupDevicePass::kFirstSort);
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kFirstSort};

    launch_reflection_dedup_mark_boundaries(params.ray_count, params.keys_out, params.boundary_flags, params.stream);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kFirstBoundaries};

    error = params.run_pass(params, ReflectionDedupDevicePass::kFirstScan);
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kFirstScan};

    launch_reflection_dedup_zero_base_ids(params.ray_count, params.keys_out, params.hash_group_ids, params.stream);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kFirstZeroBase};

    launch_reflection_dedup_sub_cluster({params.ray_count, params.max_bounces, params.keys_out, params.ray_indices_out,
                                         params.hash_group_ids, params.bounce_count, params.raw_image_x,
                                         params.raw_image_y, params.raw_image_z, params.image_source_tolerance,
                                         params.cluster_keys_in, params.cluster_ray_indices_in, params.stream});
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kSubCluster};

    error = params.run_pass(params, ReflectionDedupDevicePass::kSecondSort);
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kSecondSort};

    launch_reflection_dedup_mark_boundaries(params.ray_count, params.cluster_keys_out, params.boundary_flags,
                                            params.stream);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kSecondBoundaries};

    error = params.run_pass(params, ReflectionDedupDevicePass::kSecondScan);
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kSecondScan};

    launch_reflection_dedup_zero_base_ids(params.ray_count, params.cluster_keys_out, params.unique_path_ids,
                                          params.stream);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kSecondZeroBase};

    launch_reflection_dedup_compact({params.ray_count,
                                     params.max_bounces,
                                     params.cluster_keys_out,
                                     params.cluster_ray_indices_out,
                                     params.unique_path_ids,
                                     params.bounce_count,
                                     params.shape_ids,
                                     params.prim_ids,
                                     params.raw_t,
                                     params.raw_bary_u,
                                     params.raw_bary_v,
                                     params.raw_hit_x,
                                     params.raw_hit_y,
                                     params.raw_hit_z,
                                     params.raw_norm_x,
                                     params.raw_norm_y,
                                     params.raw_norm_z,
                                     params.raw_image_x,
                                     params.raw_image_y,
                                     params.raw_image_z,
                                     params.out_unique_count,
                                     params.out_bounce_count,
                                     params.out_shape_ids,
                                     params.out_prim_ids,
                                     params.out_t,
                                     params.out_bary_u,
                                     params.out_bary_v,
                                     params.out_hit_x,
                                     params.out_hit_y,
                                     params.out_hit_z,
                                     params.out_norm_x,
                                     params.out_norm_y,
                                     params.out_norm_z,
                                     params.out_image_x,
                                     params.out_image_y,
                                     params.out_image_z,
                                     params.out_discovery_count,
                                     params.out_representative_ray_index,
                                     params.stream});
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return {error, ReflectionDedupSequenceStep::kCompact};

    return {cudaSuccess, ReflectionDedupSequenceStep::kNone};
}

} // namespace rayd::shared::multipath
