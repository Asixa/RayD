#pragma once

#include <cstddef>
#include <cstdint>

#include <rayd/detail/bvh/topology.h>
#include <rayd/detail/bvh/triangle_query.h>
#include <rayd/detail/bvh/triangle_query_device.cuh>
#include <rayd/detail/vec3.h>
#include <rayd/detail/rt/traverser.h>

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

    __device__ __forceinline__ rt::TriangleHit trace_closest(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        const float inv_dx = safe_rcp(direction.x);
        const float inv_dy = safe_rcp(direction.y);
        const float inv_dz = safe_rcp(direction.z);
        float best_t = tmax;
        int best_prim = -1;
        float best_u = 0.0f;
        float best_v = 0.0f;
        bool overflowed = false;
        traverse_closest<true>(view.triangles, view.node_bounds, view.topology,
                               scratch, lane, origin.x, origin.y, origin.z,
                               direction.x, direction.y, direction.z,
                               inv_dx, inv_dy, inv_dz, tmin,
                               best_t, best_prim, best_u, best_v, overflowed);
        return decode(best_prim, best_t, best_u, best_v, tmax);
    }

    __device__ __forceinline__ bool trace_occluded(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        const float inv_dx = safe_rcp(direction.x);
        const float inv_dy = safe_rcp(direction.y);
        const float inv_dz = safe_rcp(direction.z);
        bool overflowed = false;
        return traverse_any_hit(view.triangles, view.node_bounds, view.topology,
                                scratch, lane, origin.x, origin.y, origin.z,
                                direction.x, direction.y, direction.z,
                                inv_dx, inv_dy, inv_dz, tmin, tmax, overflowed);
    }

    __device__ __forceinline__ bool trace_occluded_ignore(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count) const {
        int best_prim = -1;
        first_blocker(origin, direction, tmin, tmax, ignore, ignore_count, best_prim);
        return best_prim >= 0;
    }

    __device__ __forceinline__ rt::TriangleHit trace_first_blocker(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count) const {
        int best_prim = -1;
        const float best_t = first_blocker(origin, direction, tmin, tmax,
                                           ignore, ignore_count, best_prim);
        return decode(best_prim, best_t, 0.0f, 0.0f, tmax);
    }

private:
    /// Closest non-ignored blocker. The generic Traverser contract supplies
    /// `ignore` already advanced to this lane's row; `lane` therefore selects
    /// only the depth-major traversal-scratch column. The local scratch view is
    /// rebased to that column while retaining the original depth stride, so
    /// query zero addresses `base + depth * original_query_stride` and its
    /// reduced capacity still bounds every reachable slot.
    __device__ __forceinline__ float first_blocker(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count, int &best_prim) const {
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
        lane_scratch.node_indices = scratch.node_indices != nullptr
            ? scratch.node_indices + lane
            : nullptr;
        lane_scratch.overflow = scratch.overflow != nullptr
            ? scratch.overflow + lane
            : nullptr;
        lane_scratch.capacity = scratch.capacity > lane
            ? scratch.capacity - lane
            : 0;
        lane_scratch.overflow_capacity = scratch.overflow_capacity > lane
            ? scratch.overflow_capacity - lane
            : 0;
        traverse_first_blocker(view.triangles, view.node_bounds, view.topology,
                               lane_scratch, ignore, static_cast<std::int32_t>(ignore_count),
                               0, origin.x, origin.y, origin.z,
                               direction.x, direction.y, direction.z,
                               inv_dx, inv_dy, inv_dz, tmin,
                               best_t, best_prim, overflowed);
        return best_t;
    }

    __device__ __forceinline__ rt::TriangleHit decode(
        int best_prim, float best_t, float best_u, float best_v, float miss_t) const {
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

static_assert(rt::is_traverser_v<CudaBvhTraverser>,
              "CudaBvhTraverser must satisfy the rt::Traverser concept.");

} // namespace rayd::shared::bvh
