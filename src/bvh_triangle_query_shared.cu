// Copyright Xingyu Chen.
// Implements the shared BVH triangle-query kernels.

#include <src/bvh_triangle_query.h>

#include <src/bvh_query_device.cuh>
#include <src/bvh_topology.h>

#include <cmath>

// The per-ray traversal cores now live in
// src/bvh_query_device.cuh so the standalone BVH kernels below
// and the on-device CudaBvhTraverser share one implementation. These kernels are
// thin __global__ wrappers: index guard, output init, ray load, then a single
// call into the shared core, followed by the P3 overflow-repair bookkeeping.
// Behavior and the one-launch-per-query contract are unchanged from P3.

namespace rayd::shared::bvh {
namespace {

constexpr int kBlockSize = 128;

__global__ void closest_hit_kernel(TriangleClosestHitParams params) {
    const std::size_t ray = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= params.rays.count) {
        return;
    }

    params.output.t[ray] = INFINITY;
    params.output.bary_u[ray] = 0.0f;
    params.output.bary_v[ray] = 0.0f;
    params.output.shape_id[ray] = -1;
    params.output.local_prim_id[ray] = -1;
    if (params.scratch.overflow != nullptr) {
        params.scratch.overflow[ray] = 0;
    }

    if (params.rays.active != nullptr && params.rays.active[ray] == 0) {
        return;
    }
    if (params.topology.node_count == 0) {
        return;
    }

    float ox, oy, oz, dx, dy, dz, t_max;
    load_ray(params.rays, ray, ox, oy, oz, dx, dy, dz, t_max);
    const float inv_dx = safe_rcp(dx);
    const float inv_dy = safe_rcp(dy);
    const float inv_dz = safe_rcp(dz);

    float best_t = t_max;
    int best_prim = -1;
    float best_u = 0.0f;
    float best_v = 0.0f;
    bool overflowed = false;
    traverse_closest<true>(params.triangles, params.node_bounds, params.topology, params.scratch, ray, ox, oy, oz, dx,
                           dy, dz, inv_dx, inv_dy, inv_dz, params.t_min, best_t, best_prim, best_u, best_v, overflowed);

    if (overflowed) {
        if (params.scratch.overflow != nullptr) {
            params.scratch.overflow[ray] = 1;
        }
        return;
    }
    if (best_prim >= 0) {
        params.output.t[ray] = best_t;
        params.output.bary_u[ray] = best_u;
        params.output.bary_v[ray] = best_v;
        params.output.shape_id[ray] = params.prim_map.shape_id[best_prim];
        params.output.local_prim_id[ray] = params.prim_map.local_prim_id[best_prim];
    }
}

__global__ void occluded_kernel(TriangleOccludedParams params) {
    const std::size_t ray = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= params.rays.count) {
        return;
    }

    params.out_hit[ray] = 0;
    if (params.scratch.overflow != nullptr) {
        params.scratch.overflow[ray] = 0;
    }
    if (params.rays.active != nullptr && params.rays.active[ray] == 0) {
        return;
    }
    if (params.topology.node_count == 0) {
        return;
    }

    float ox, oy, oz, dx, dy, dz, t_max;
    load_ray(params.rays, ray, ox, oy, oz, dx, dy, dz, t_max);
    const float inv_dx = safe_rcp(dx);
    const float inv_dy = safe_rcp(dy);
    const float inv_dz = safe_rcp(dz);

    bool overflowed = false;
    const bool hit = traverse_any_hit(params.triangles, params.node_bounds, params.topology, params.scratch, ray, ox,
                                      oy, oz, dx, dy, dz, inv_dx, inv_dy, inv_dz, params.t_min, t_max, overflowed);
    if (overflowed) {
        if (params.scratch.overflow != nullptr) {
            params.scratch.overflow[ray] = 1;
        }
        return;
    }
    params.out_hit[ray] = hit ? 1 : 0;
}

__global__ void first_blocker_kernel(TriangleFirstBlockerParams params) {
    const std::size_t ray = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= params.rays.count) {
        return;
    }

    params.out_global_prim_id[ray] = -1;
    if (params.scratch.overflow != nullptr) {
        params.scratch.overflow[ray] = 0;
    }
    if (params.rays.active != nullptr && params.rays.active[ray] == 0) {
        return;
    }
    if (params.topology.node_count == 0) {
        return;
    }

    float ox, oy, oz, dx, dy, dz, t_max;
    load_ray(params.rays, ray, ox, oy, oz, dx, dy, dz, t_max);
    const float inv_dx = safe_rcp(dx);
    const float inv_dy = safe_rcp(dy);
    const float inv_dz = safe_rcp(dz);

    float best_t = t_max;
    int best_prim = -1;
    bool overflowed = false;
    traverse_first_blocker(params.triangles, params.node_bounds, params.topology, params.scratch,
                           params.ignore_prim_ids, params.ignore_stride, ray, ox, oy, oz, dx, dy, dz, inv_dx, inv_dy,
                           inv_dz, params.t_min, best_t, best_prim, overflowed);

    if (overflowed) {
        if (params.scratch.overflow != nullptr) {
            params.scratch.overflow[ray] = 1;
        }
        return;
    }
    params.out_global_prim_id[ray] = best_prim;
}

__global__ void closest_hit_repair_kernel(TriangleClosestHitParams params) {
    const std::size_t ray = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= params.rays.count) {
        return;
    }
    if (params.scratch.overflow == nullptr || params.scratch.overflow[ray] == 0) {
        return;
    }

    float ox, oy, oz, dx, dy, dz, t_max;
    load_ray(params.rays, ray, ox, oy, oz, dx, dy, dz, t_max);
    float best_t = t_max;
    int best_prim = -1;
    float best_u = 0.0f;
    float best_v = 0.0f;
    brute_force_closest(params.triangles, ox, oy, oz, dx, dy, dz, params.t_min, best_t, best_prim, best_u, best_v);
    if (best_prim >= 0) {
        params.output.t[ray] = best_t;
        params.output.bary_u[ray] = best_u;
        params.output.bary_v[ray] = best_v;
        params.output.shape_id[ray] = params.prim_map.shape_id[best_prim];
        params.output.local_prim_id[ray] = params.prim_map.local_prim_id[best_prim];
    }
}

__global__ void occluded_repair_kernel(TriangleOccludedParams params) {
    const std::size_t ray = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= params.rays.count) {
        return;
    }
    if (params.scratch.overflow == nullptr || params.scratch.overflow[ray] == 0) {
        return;
    }

    float ox, oy, oz, dx, dy, dz, t_max;
    load_ray(params.rays, ray, ox, oy, oz, dx, dy, dz, t_max);
    const bool hit = brute_force_occluded(params.triangles, ox, oy, oz, dx, dy, dz, params.t_min, t_max);
    params.out_hit[ray] = hit ? 1 : 0;
}

int block_count(std::size_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

} // namespace

void launch_triangle_closest_hit_async(const TriangleClosestHitParams& params) {
    if (params.rays.count == 0) {
        return;
    }
    closest_hit_kernel<<<block_count(params.rays.count), kBlockSize, 0, params.stream>>>(params);
}

void launch_triangle_occluded_async(const TriangleOccludedParams& params) {
    if (params.rays.count == 0) {
        return;
    }
    occluded_kernel<<<block_count(params.rays.count), kBlockSize, 0, params.stream>>>(params);
}

void launch_triangle_first_blocker_async(const TriangleFirstBlockerParams& params) {
    if (params.rays.count == 0) {
        return;
    }
    first_blocker_kernel<<<block_count(params.rays.count), kBlockSize, 0, params.stream>>>(params);
}

void launch_triangle_closest_hit_repair_async(const TriangleClosestHitParams& params) {
    if (params.rays.count == 0) {
        return;
    }
    closest_hit_repair_kernel<<<block_count(params.rays.count), kBlockSize, 0, params.stream>>>(params);
}

void launch_triangle_occluded_repair_async(const TriangleOccludedParams& params) {
    if (params.rays.count == 0) {
        return;
    }
    occluded_repair_kernel<<<block_count(params.rays.count), kBlockSize, 0, params.stream>>>(params);
}

} // namespace rayd::shared::bvh
