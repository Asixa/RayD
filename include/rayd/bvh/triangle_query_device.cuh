// Copyright Xingyu Chen.
// Defines shared bvh support for triangle query device.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <rayd/bvh/topology.h>
#include <rayd/bvh/traversal_common.cuh>
#include <rayd/bvh/triangle_intersect.h>
#include <rayd/bvh/triangle_query.h>

// Device traversal cores for the scene triangle BVH. Extracted verbatim from
// src/bvh/triangle_query_shared.cu so both the standalone traversal kernels (thin
// __global__ wrappers that keep their P3 behavior and one-launch-per-query
// contract) and the on-device CudaBvhTraverser (shared/bvh/cuda_bvh_traverser.h)
// share exactly one implementation. Device-only: every routine is __device__ and
// depends on the CUDA runtime, so this header is compiled only under nvcc.

namespace rayd::shared::bvh {

/// Vertices A/B/C of one triangle in edge-vector form.
struct TriangleVertices {
    float ax, ay, az;
    float bx, by, bz;
    float cx, cy, cz;
};

__device__ __forceinline__ TriangleVertices load_triangle(const TriangleSoAView& tri, int prim) {
    const float p0x = tri.p0_x[prim];
    const float p0y = tri.p0_y[prim];
    const float p0z = tri.p0_z[prim];
    const float e1x = tri.e1_x[prim];
    const float e1y = tri.e1_y[prim];
    const float e1z = tri.e1_z[prim];
    const float e2x = tri.e2_x[prim];
    const float e2y = tri.e2_y[prim];
    const float e2z = tri.e2_z[prim];
    return {p0x, p0y, p0z, p0x + e1x, p0y + e1y, p0z + e1z, p0x + e2x, p0y + e2y, p0z + e2z};
}

/// Reciprocal of a ray-direction component that never yields 0*inf NaNs: a zero
/// (or sub-tiny) component is clamped to a large finite magnitude, so the slab
/// test degenerates conservatively rather than dropping the node.
__device__ __forceinline__ float safe_rcp(float value) {
    const float magnitude = fabsf(value);
    if (magnitude < 1.0e-20f) {
        return value < 0.0f ? -1.0e20f : 1.0e20f;
    }
    return 1.0f / value;
}

/// Slab test of the ray against node `node`'s bounds over [t_min, t_max_cap].
/// Returns whether the ray overlaps the box and writes the entry distance used
/// for near/far ordering.
__device__ __forceinline__ bool intersect_node_bounds(const AabbSoAView& bounds, int node, float ox, float oy, float oz,
                                                      float inv_dx, float inv_dy, float inv_dz, float t_min,
                                                      float t_max_cap, float& t_entry) {
    const float t0x = (bounds.min_x[node] - ox) * inv_dx;
    const float t1x = (bounds.max_x[node] - ox) * inv_dx;
    const float t0y = (bounds.min_y[node] - oy) * inv_dy;
    const float t1y = (bounds.max_y[node] - oy) * inv_dy;
    const float t0z = (bounds.min_z[node] - oz) * inv_dz;
    const float t1z = (bounds.max_z[node] - oz) * inv_dz;

    const float t_near = fmaxf(fmaxf(fminf(t0x, t1x), fminf(t0y, t1y)), fmaxf(fminf(t0z, t1z), t_min));
    const float t_far = fminf(fminf(fmaxf(t0x, t1x), fmaxf(t0y, t1y)), fminf(fmaxf(t0z, t1z), t_max_cap));
    t_entry = t_near;
    return t_near <= t_far;
}

__device__ __forceinline__ bool is_leaf_node(const CompactBvhTopologyView& topology, int node) {
    return topology.left_child[node] < 0;
}

/// Descend from `entry_node` collecting leaf nodes, pushing far children onto
/// the caller's depth-major stack. Returns the closest-hit winner via the
/// (best_t, best_prim) accumulators. Sets *overflowed when the stack is
/// exhausted so the host repair can rerun the lane.
template <bool OrderChildren>
__device__ __forceinline__ void traverse_closest(const TriangleSoAView& tri, const AabbSoAView& bounds,
                                                 const CompactBvhTopologyView& topology,
                                                 const TriangleTraversalScratchView& scratch, std::size_t ray, float ox,
                                                 float oy, float oz, float dx, float dy, float dz, float inv_dx,
                                                 float inv_dy, float inv_dz, float t_min, float& best_t, int& best_prim,
                                                 float& best_u, float& best_v, bool& overflowed) {
    int sp = 0;
    int node = 0;
    for (;;) {
        while (node >= 0 && !is_leaf_node(topology, node)) {
            const int left = topology.left_child[node];
            const int right = topology.right_child[node];
            float t_left = 0.0f;
            float t_right = 0.0f;
            const bool hit_left =
                intersect_node_bounds(bounds, left, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, best_t, t_left);
            const bool hit_right =
                intersect_node_bounds(bounds, right, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, best_t, t_right);
            if (hit_left && hit_right) {
                int near_child = left;
                int far_child = right;
                if (OrderChildren && !near_child_is_left(t_left, t_right, left, right)) {
                    near_child = right;
                    far_child = left;
                }
                if (!stack_push(scratch, ray, static_cast<std::size_t>(sp), far_child)) {
                    overflowed = true;
                    return;
                }
                ++sp;
                node = near_child;
            } else if (hit_left) {
                node = left;
            } else if (hit_right) {
                node = right;
            } else {
                node = -1;
            }
        }

        if (node >= 0) {
            const int leaf_begin = -topology.left_child[node] - 1;
            const int leaf_count = topology.right_child[node];
            for (int slot = 0; slot < leaf_count; ++slot) {
                const int prim = topology.leaf_primitives[leaf_begin + slot];
                const TriangleVertices vertices = load_triangle(tri, prim);
                const WatertightTriangleHit hit =
                    intersect_triangle_watertight(ox, oy, oz, dx, dy, dz, vertices.ax, vertices.ay, vertices.az,
                                                  vertices.bx, vertices.by, vertices.bz, vertices.cx, vertices.cy,
                                                  vertices.cz, t_min, best_t);
                if (hit.hit && (hit.t < best_t || (hit.t == best_t && prim < best_prim))) {
                    best_t = hit.t;
                    best_prim = prim;
                    best_u = hit.u;
                    best_v = hit.v;
                }
            }
        }

        if (sp == 0) {
            return;
        }
        --sp;
        node = stack_load(scratch, ray, static_cast<std::size_t>(sp));
    }
}

/// Occlusion traversal: returns true on the first surface within [t_min, t_max].
__device__ __forceinline__ bool traverse_any_hit(const TriangleSoAView& tri, const AabbSoAView& bounds,
                                                 const CompactBvhTopologyView& topology,
                                                 const TriangleTraversalScratchView& scratch, std::size_t ray, float ox,
                                                 float oy, float oz, float dx, float dy, float dz, float inv_dx,
                                                 float inv_dy, float inv_dz, float t_min, float t_max,
                                                 bool& overflowed) {
    int sp = 0;
    int node = 0;
    for (;;) {
        while (node >= 0 && !is_leaf_node(topology, node)) {
            const int left = topology.left_child[node];
            const int right = topology.right_child[node];
            float t_left = 0.0f;
            float t_right = 0.0f;
            const bool hit_left =
                intersect_node_bounds(bounds, left, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, t_max, t_left);
            const bool hit_right =
                intersect_node_bounds(bounds, right, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, t_max, t_right);
            if (hit_left && hit_right) {
                if (!stack_push(scratch, ray, static_cast<std::size_t>(sp), right)) {
                    overflowed = true;
                    return false;
                }
                ++sp;
                node = left;
            } else if (hit_left) {
                node = left;
            } else if (hit_right) {
                node = right;
            } else {
                node = -1;
            }
        }

        if (node >= 0) {
            const int leaf_begin = -topology.left_child[node] - 1;
            const int leaf_count = topology.right_child[node];
            for (int slot = 0; slot < leaf_count; ++slot) {
                const int prim = topology.leaf_primitives[leaf_begin + slot];
                const TriangleVertices vertices = load_triangle(tri, prim);
                const WatertightTriangleHit hit =
                    intersect_triangle_watertight(ox, oy, oz, dx, dy, dz, vertices.ax, vertices.ay, vertices.az,
                                                  vertices.bx, vertices.by, vertices.bz, vertices.cx, vertices.cy,
                                                  vertices.cz, t_min, t_max);
                if (hit.hit) {
                    return true;
                }
            }
        }

        if (sp == 0) {
            return false;
        }
        --sp;
        node = stack_load(scratch, ray, static_cast<std::size_t>(sp));
    }
}

__device__ __forceinline__ bool prim_is_ignored(const std::int32_t* ignore_prim_ids, std::int32_t ignore_stride,
                                                std::size_t ray, int prim) {
    if (ignore_prim_ids == nullptr || ignore_stride <= 0) {
        return false;
    }
    const std::int32_t* row = ignore_prim_ids + ray * static_cast<std::size_t>(ignore_stride);
    for (std::int32_t i = 0; i < ignore_stride; ++i) {
        if (row[i] == prim) {
            return true;
        }
    }
    return false;
}

/// Closest-blocker traversal honoring a per-ray ignore list, returning the
/// winning global primitive id via `best_prim` (or leaving it untouched on a
/// miss). Mirrors the first-blocker kernel loop exactly.
__device__ __forceinline__ void traverse_first_blocker(const TriangleSoAView& tri, const AabbSoAView& bounds,
                                                       const CompactBvhTopologyView& topology,
                                                       const TriangleTraversalScratchView& scratch,
                                                       const std::int32_t* ignore_prim_ids, std::int32_t ignore_stride,
                                                       std::size_t ray, float ox, float oy, float oz, float dx,
                                                       float dy, float dz, float inv_dx, float inv_dy, float inv_dz,
                                                       float t_min, float& best_t, int& best_prim, bool& overflowed) {
    int sp = 0;
    int node = 0;
    for (;;) {
        while (node >= 0 && !is_leaf_node(topology, node)) {
            const int left = topology.left_child[node];
            const int right = topology.right_child[node];
            float t_left = 0.0f;
            float t_right = 0.0f;
            const bool hit_left =
                intersect_node_bounds(bounds, left, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, best_t, t_left);
            const bool hit_right =
                intersect_node_bounds(bounds, right, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, best_t, t_right);
            if (hit_left && hit_right) {
                int near_child = left;
                int far_child = right;
                if (!near_child_is_left(t_left, t_right, left, right)) {
                    near_child = right;
                    far_child = left;
                }
                if (!stack_push(scratch, ray, static_cast<std::size_t>(sp), far_child)) {
                    overflowed = true;
                    break;
                }
                ++sp;
                node = near_child;
            } else if (hit_left) {
                node = left;
            } else if (hit_right) {
                node = right;
            } else {
                node = -1;
            }
        }
        if (overflowed) {
            break;
        }

        if (node >= 0) {
            const int leaf_begin = -topology.left_child[node] - 1;
            const int leaf_count = topology.right_child[node];
            for (int slot = 0; slot < leaf_count; ++slot) {
                const int prim = topology.leaf_primitives[leaf_begin + slot];
                if (prim_is_ignored(ignore_prim_ids, ignore_stride, ray, prim)) {
                    continue;
                }
                const TriangleVertices vertices = load_triangle(tri, prim);
                const WatertightTriangleHit hit =
                    intersect_triangle_watertight(ox, oy, oz, dx, dy, dz, vertices.ax, vertices.ay, vertices.az,
                                                  vertices.bx, vertices.by, vertices.bz, vertices.cx, vertices.cy,
                                                  vertices.cz, t_min, best_t);
                if (hit.hit && (hit.t < best_t || (hit.t == best_t && prim < best_prim))) {
                    best_t = hit.t;
                    best_prim = prim;
                }
            }
        }

        if (sp == 0) {
            break;
        }
        --sp;
        node = stack_load(scratch, ray, static_cast<std::size_t>(sp));
    }
}

/// Brute-force closest hit over every primitive (traversal-stack overflow repair).
__device__ __forceinline__ void brute_force_closest(const TriangleSoAView& tri, float ox, float oy, float oz, float dx,
                                                    float dy, float dz, float t_min, float& best_t, int& best_prim,
                                                    float& best_u, float& best_v) {
    const int prim_count = static_cast<int>(tri.count);
    for (int prim = 0; prim < prim_count; ++prim) {
        const TriangleVertices vertices = load_triangle(tri, prim);
        const WatertightTriangleHit hit =
            intersect_triangle_watertight(ox, oy, oz, dx, dy, dz, vertices.ax, vertices.ay, vertices.az, vertices.bx,
                                          vertices.by, vertices.bz, vertices.cx, vertices.cy, vertices.cz, t_min,
                                          best_t);
        if (hit.hit && (hit.t < best_t || (hit.t == best_t && prim < best_prim))) {
            best_t = hit.t;
            best_prim = prim;
            best_u = hit.u;
            best_v = hit.v;
        }
    }
}

/// Brute-force occlusion over every primitive (traversal-stack overflow repair).
__device__ __forceinline__ bool brute_force_occluded(const TriangleSoAView& tri, float ox, float oy, float oz, float dx,
                                                     float dy, float dz, float t_min, float t_max) {
    const int prim_count = static_cast<int>(tri.count);
    for (int prim = 0; prim < prim_count; ++prim) {
        const TriangleVertices vertices = load_triangle(tri, prim);
        const WatertightTriangleHit hit =
            intersect_triangle_watertight(ox, oy, oz, dx, dy, dz, vertices.ax, vertices.ay, vertices.az, vertices.bx,
                                          vertices.by, vertices.bz, vertices.cx, vertices.cy, vertices.cz, t_min,
                                          t_max);
        if (hit.hit) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ void load_ray(const TriangleRaySoAView& rays, std::size_t ray, float& ox, float& oy,
                                         float& oz, float& dx, float& dy, float& dz, float& t_max) {
    ox = rays.origin_x[ray];
    oy = rays.origin_y[ray];
    oz = rays.origin_z[ray];
    dx = rays.dir_x[ray];
    dy = rays.dir_y[ray];
    dz = rays.dir_z[ray];
    t_max = rays.t_max[ray];
}

} // namespace rayd::shared::bvh
