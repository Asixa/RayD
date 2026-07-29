// Copyright Xingyu Chen.
// Defines shared BVH device traversal and the CUDA traverser.

#pragma once

#include <cstddef>

namespace rayd::shared::bvh {

// Primitive-agnostic depth-major traversal-stack helpers. The scratch view is
// supplied by the caller so any BVH consumer (edge queries today, other
// primitives later) can share the identical coalesced push/pop indexing. The
// Scratch type must expose node_indices, stack_depth, query_stride, and
// capacity. Behaviour is bitwise identical to an inlined per-query stack.

/// Store `node` at the given stack depth for `query`. Returns false without
/// writing when the depth exceeds capacity or the view is unallocated.
template <typename Scratch>
__device__ __forceinline__ bool stack_push(const Scratch& scratch, std::size_t query, std::size_t depth, int node) {
    if (scratch.node_indices == nullptr || depth >= scratch.stack_depth) {
        return false;
    }
    const std::size_t slot = depth * scratch.query_stride + query;
    if (query >= scratch.query_stride || slot >= scratch.capacity) {
        return false;
    }
    scratch.node_indices[slot] = node;
    return true;
}

/// Load the node deferred at `depth_index` for `query`.
template <typename Scratch>
__device__ __forceinline__ int stack_load(const Scratch& scratch, std::size_t query, std::size_t depth_index) {
    return scratch.node_indices[depth_index * scratch.query_stride + query];
}

/// Deterministic near/far ordering for a two-child descent: the smaller bound
/// wins, ties break on the lower child index so traversal order is stable.
__device__ __forceinline__ bool near_child_is_left(float left_bound, float right_bound, int left, int right) {
    return left_bound < right_bound || (left_bound == right_bound && left < right);
}

} // namespace rayd::shared::bvh

#include <cmath>

// Watertight ray/triangle intersection (Woop, Benthin, Wald 2013,
// "Watertight Ray/Triangle Intersection"). Host/device dual through the same
// __CUDACC__ inline pattern as rayd/math.h so the routine is both a
// device leaf test and a host-unit-testable pure function. It is
// primitive-only: no Dr.Jit, Torch, OptiX, or CUDA runtime dependency.
//
// The returned (u, v) match the Moller-Trumbore convention used by
// include/rayd/jit/core.h ray_intersect_triangle: a triangle given
// as (p0, e1, e2) with vertices A = p0, B = p0 + e1, C = p0 + e2 has hit point
// P = p0 + u*e1 + v*e2 = (1 - u - v)*A + u*B + v*C. So u is the barycentric
// weight of B and v the weight of C, and t is measured along the (unnormalized)
// ray direction, exactly like the Moller-Trumbore reference.

#if defined(__CUDACC__)
#define RAYD_SHARED_BVH_TRI_INLINE __host__ __device__ __forceinline__
#else
#define RAYD_SHARED_BVH_TRI_INLINE inline
#endif

namespace rayd::shared::bvh {

/// Result of a watertight ray/triangle test. On a miss `hit` is false and the
/// remaining fields are unspecified.
struct WatertightTriangleHit {
    bool hit;
    float t;
    float u; ///< Barycentric weight of vertex B (p0 + e1).
    float v; ///< Barycentric weight of vertex C (p0 + e2).
};

/// Correctly-rounded `a*b - c*d` (Kahan). This is FMA-contraction-proof: when
/// `a*b == c*d` mathematically it returns exactly 0.0f regardless of the
/// compiler's `--fmad` setting, which is what keeps the watertight edge tests
/// from mistaking an on-edge hit (a genuinely zero edge function) for a
/// mixed-sign miss.
RAYD_SHARED_BVH_TRI_INLINE float diff_of_products(float a, float b, float c, float d) {
    const float cd = c * d;
    const float error = fmaf(-c, d, cd); // exact rounding error of c*d
    const float diff = fmaf(a, b, -cd);  // a*b - cd, with a*b exact
    return diff + error;
}

/// Watertight ray/triangle intersection with no backface culling. Accepts hits
/// on triangle boundaries (an edge function of exactly zero) so a ray grazing a
/// shared edge is reported by both adjacent triangles; the caller's closest-hit
/// (t, primitive-id) reduction then selects a single deterministic winner, which
/// is what keeps a shared-edge crossing exactly-once in closest-hit semantics.
///
/// \param ox,oy,oz  Ray origin.
/// \param dx,dy,dz  Ray direction (need not be normalized; t is in its units).
/// \param ax,ay,az  Vertex A = p0.
/// \param bx,by,bz  Vertex B = p0 + e1.
/// \param cx,cy,cz  Vertex C = p0 + e2.
/// \param t_min     Inclusive lower bound on the accepted hit distance.
/// \param t_max     Inclusive upper bound on the accepted hit distance.
RAYD_SHARED_BVH_TRI_INLINE WatertightTriangleHit intersect_triangle_watertight(float ox, float oy, float oz, float dx,
                                                                               float dy, float dz, float ax, float ay,
                                                                               float az, float bx, float by, float bz,
                                                                               float cx, float cy, float cz,
                                                                               float t_min, float t_max) {
    WatertightTriangleHit result{false, 0.0f, 0.0f, 0.0f};

    // Vertices relative to the ray origin.
    float a[3] = {ax - ox, ay - oy, az - oz};
    float b[3] = {bx - ox, by - oy, bz - oz};
    float c[3] = {cx - ox, cy - oy, cz - oz};
    const float d[3] = {dx, dy, dz};

    // Pick kz = axis of largest |direction| and a cyclic permutation kx, ky.
    int kz = 0;
    float max_component = fabsf(d[0]);
    if (fabsf(d[1]) > max_component) {
        max_component = fabsf(d[1]);
        kz = 1;
    }
    if (fabsf(d[2]) > max_component) {
        kz = 2;
    }
    int kx = kz + 1;
    if (kx == 3) {
        kx = 0;
    }
    int ky = kx + 1;
    if (ky == 3) {
        ky = 0;
    }
    // Preserve winding when the ray points down the chosen axis.
    if (d[kz] < 0.0f) {
        const int swap = kx;
        kx = ky;
        ky = swap;
    }

    const float dz_axis = d[kz];
    // A degenerate direction cannot form a shear frame.
    if (dz_axis == 0.0f) {
        return result;
    }
    const float sx = d[kx] / dz_axis;
    const float sy = d[ky] / dz_axis;
    const float sz = 1.0f / dz_axis;

    // Shear and scale the vertices into ray space.
    const float ax2 = a[kx] - sx * a[kz];
    const float ay2 = a[ky] - sy * a[kz];
    const float bx2 = b[kx] - sx * b[kz];
    const float by2 = b[ky] - sy * b[kz];
    const float cx2 = c[kx] - sx * c[kz];
    const float cy2 = c[ky] - sy * c[kz];

    // Scaled barycentric coordinates: u_A = U, u_B = V, u_C = W. The
    // FMA-contraction-proof difference of products makes a mathematically zero
    // edge function come out exactly 0.0f (rather than a tiny residual with an
    // arbitrary sign that would spuriously reject an on-edge crossing).
    float u = diff_of_products(cx2, by2, cy2, bx2);
    float v = diff_of_products(ax2, cy2, ay2, cx2);
    float w = diff_of_products(bx2, ay2, by2, ax2);

    // Exact-zero edge functions fall back to double precision so shared-edge
    // signs stay consistent between adjacent triangles (no gaps).
    if (u == 0.0f || v == 0.0f || w == 0.0f) {
        const double cxby = static_cast<double>(cx2) * static_cast<double>(by2);
        const double cybx = static_cast<double>(cy2) * static_cast<double>(bx2);
        u = static_cast<float>(cxby - cybx);
        const double axcy = static_cast<double>(ax2) * static_cast<double>(cy2);
        const double aycx = static_cast<double>(ay2) * static_cast<double>(cx2);
        v = static_cast<float>(axcy - aycx);
        const double bxay = static_cast<double>(bx2) * static_cast<double>(ay2);
        const double byax = static_cast<double>(by2) * static_cast<double>(ax2);
        w = static_cast<float>(bxay - byax);
    }

    // No backface culling: reject only when the edge functions disagree in sign.
    if ((u < 0.0f || v < 0.0f || w < 0.0f) && (u > 0.0f || v > 0.0f || w > 0.0f)) {
        return result;
    }

    const float det = u + v + w;
    if (det == 0.0f) {
        return result;
    }

    // Scaled hit distance along the sheared axis.
    const float az2 = sz * a[kz];
    const float bz2 = sz * b[kz];
    const float cz2 = sz * c[kz];
    const float scaled_t = u * az2 + v * bz2 + w * cz2;

    const float rcp_det = 1.0f / det;
    const float t = scaled_t * rcp_det;
    if (t < t_min || t > t_max) {
        return result;
    }

    result.hit = true;
    result.t = t;
    result.u = v * rcp_det; // weight of vertex B == Moller-Trumbore u
    result.v = w * rcp_det; // weight of vertex C == Moller-Trumbore v
    return result;
}

} // namespace rayd::shared::bvh

#undef RAYD_SHARED_BVH_TRI_INLINE

#include <cstdint>

#include <src/bvh_topology.h>
#include <src/bvh_triangle_query.h>

// Device traversal cores for the scene triangle BVH. Extracted verbatim from
// src/bvh_triangle_query_shared.cu so both the standalone traversal kernels (thin
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

#include <rayd/math.h>
#include <src/runtime/rt_device.cuh>

// Pure-CUDA BVH traverser: a per-lane rt::is_traverser oracle over the scene
// triangle BVH, implemented entirely on the shared traversal cores in
// triangle_query_device.cuh. It is the Dr.Jit-backend eager-native counterpart
// of the OptiX shim and the acceleration-structure axis a future
// CudaFusedExecutor (P4d) folds the migrated pipeline algorithms onto. In P4
// Stage A it is compiled and concept-checked but not yet wired to a pipeline.
//
// Device-only (depends on the CUDA runtime through triangle_query_device.cuh).

namespace rayd::shared::bvh {

/// Read-only view of one built scene triangle BVH: node-bounds SoA, compacted
/// preorder topology, world-space triangle SoA, and the per-primitive scene id
/// map. Matches the buffers the standalone traversal kernels consume.
struct CudaBvhView {
    TriangleSoAView triangles;
    AabbSoAView node_bounds;
    CompactBvhTopologyView topology;
    TrianglePrimIdMapView prim_map;
};

/// Per-lane traverser over a CudaBvhView. `scratch` is the caller-owned
/// depth-major traversal stack (indexed by `lane`); `lane` is this thread's ray
/// index into that stack. Its four const methods satisfy rt::is_traverser and
/// decode winners into the backend-neutral rt::TriangleHit (prim = mesh-local
/// primitive id, instance = shape id) exactly as the OptiX shim does.
struct CudaBvhTraverser {
    CudaBvhView view;
    TriangleTraversalScratchView scratch;
    std::size_t lane;

    __device__ __forceinline__ rt::TriangleHit trace_closest(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                             float tmax) const {
        const float inv_dx = safe_rcp(direction.x);
        const float inv_dy = safe_rcp(direction.y);
        const float inv_dz = safe_rcp(direction.z);
        float best_t = tmax;
        int best_prim = -1;
        float best_u = 0.0f;
        float best_v = 0.0f;
        bool overflowed = false;
        traverse_closest<true>(view.triangles, view.node_bounds, view.topology, scratch, lane, origin.x, origin.y,
                               origin.z, direction.x, direction.y, direction.z, inv_dx, inv_dy, inv_dz, tmin, best_t,
                               best_prim, best_u, best_v, overflowed);
        return decode(best_prim, best_t, best_u, best_v, tmax);
    }

    __device__ __forceinline__ bool trace_occluded(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                   float tmax) const {
        const float inv_dx = safe_rcp(direction.x);
        const float inv_dy = safe_rcp(direction.y);
        const float inv_dz = safe_rcp(direction.z);
        bool overflowed = false;
        return traverse_any_hit(view.triangles, view.node_bounds, view.topology, scratch, lane, origin.x, origin.y,
                                origin.z, direction.x, direction.y, direction.z, inv_dx, inv_dy, inv_dz, tmin, tmax,
                                overflowed);
    }

    __device__ __forceinline__ bool trace_occluded_ignore(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                          float tmax, const std::int32_t* ignore,
                                                          int ignore_count) const {
        int best_prim = -1;
        first_blocker(origin, direction, tmin, tmax, ignore, ignore_count, best_prim);
        return best_prim >= 0;
    }

    __device__ __forceinline__ rt::TriangleHit trace_first_blocker(math::Vec3f origin, math::Vec3f direction,
                                                                   float tmin, float tmax, const std::int32_t* ignore,
                                                                   int ignore_count) const {
        int best_prim = -1;
        const float best_t = first_blocker(origin, direction, tmin, tmax, ignore, ignore_count, best_prim);
        return decode(best_prim, best_t, 0.0f, 0.0f, tmax);
    }

  private:
    /// Closest non-ignored blocker. The generic Traverser contract supplies
    /// `ignore` already advanced to this lane's row; `lane` therefore selects
    /// only the depth-major traversal-scratch column. The local scratch view is
    /// rebased to that column while retaining the original depth stride, so
    /// query zero addresses `base + depth * original_query_stride` and its
    /// reduced capacity still bounds every reachable slot.
    __device__ __forceinline__ float first_blocker(math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
                                                   const std::int32_t* ignore, int ignore_count, int& best_prim) const {
        const float inv_dx = safe_rcp(direction.x);
        const float inv_dy = safe_rcp(direction.y);
        const float inv_dz = safe_rcp(direction.z);
        float best_t = tmax;
        bool overflowed = false;
        // The visibility algorithms pass `ignore` already advanced to this
        // lane's row.  Rebase only the traversal scratch and query it as lane
        // zero so traverse_first_blocker does not advance the ignore row a
        // second time for lanes > 0.
        TriangleTraversalScratchView lane_scratch = scratch;
        lane_scratch.node_indices = scratch.node_indices != nullptr ? scratch.node_indices + lane : nullptr;
        lane_scratch.overflow = scratch.overflow != nullptr ? scratch.overflow + lane : nullptr;
        lane_scratch.capacity = scratch.capacity > lane ? scratch.capacity - lane : 0;
        lane_scratch.overflow_capacity = scratch.overflow_capacity > lane ? scratch.overflow_capacity - lane : 0;
        traverse_first_blocker(view.triangles, view.node_bounds, view.topology, lane_scratch, ignore,
                               static_cast<std::int32_t>(ignore_count), 0, origin.x, origin.y, origin.z, direction.x,
                               direction.y, direction.z, inv_dx, inv_dy, inv_dz, tmin, best_t, best_prim, overflowed);
        return best_t;
    }

    __device__ __forceinline__ rt::TriangleHit decode(int best_prim, float best_t, float best_u, float best_v,
                                                      float miss_t) const {
        rt::TriangleHit hit;
        if (best_prim >= 0) {
            hit.t = best_t;
            hit.bary_u = best_u;
            hit.bary_v = best_v;
            hit.prim = view.prim_map.local_prim_id[best_prim];
            hit.instance = view.prim_map.shape_id[best_prim];
            hit.hit = 1u;
        } else {
            hit.t = miss_t;
            hit.bary_u = 0.0f;
            hit.bary_v = 0.0f;
            hit.prim = -1;
            hit.instance = -1;
            hit.hit = 0u;
        }
        return hit;
    }
};

static_assert(rt::is_traverser_v<CudaBvhTraverser>, "CudaBvhTraverser must satisfy the rt::Traverser concept.");

} // namespace rayd::shared::bvh
