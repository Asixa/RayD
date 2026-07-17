#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/shared/bvh/topology.h>

// Scene-level triangle BVH traversal kernels. Three separate closest-hit,
// occluded, and first-blocker launchers (not one branchy kernel) over a
// compacted preorder BVH, in the same allocation-free, stream-parametered,
// POD-params house style as shared/edge/bvh_query.h. Every launcher allocates
// nothing, performs no synchronization, and is asynchronous on params.stream.

namespace rayd::shared::bvh {

/// World-space triangle geometry in edge-vector form, indexed by global
/// primitive id. Vertices are A = p0, B = p0 + e1, C = p0 + e2.
struct TriangleSoAView {
    const float *p0_x;
    const float *p0_y;
    const float *p0_z;
    const float *e1_x;
    const float *e1_y;
    const float *e1_z;
    const float *e2_x;
    const float *e2_y;
    const float *e2_z;
    std::size_t count;
};

/// SoA ray batch. `t_max` is already remapped by the caller (finite or the
/// legacy 1e8 half-ray cap). `active` is one int per ray (0 skips the lane);
/// a null pointer means every lane is active.
struct TriangleRaySoAView {
    const float *origin_x;
    const float *origin_y;
    const float *origin_z;
    const float *dir_x;
    const float *dir_y;
    const float *dir_z;
    const float *t_max;
    const std::int32_t *active;
    std::size_t count;
};

/// Per-primitive scene id maps, indexed by global primitive id.
struct TrianglePrimIdMapView {
    const std::int32_t *shape_id;
    const std::int32_t *local_prim_id;
    std::size_t count;
};

/// Caller-owned, preallocated closest-hit output (OptixIntersection-shaped).
/// A miss writes t = +inf, bary = 0, and shape_id = local_prim_id = -1.
struct TriangleClosestHitOutputView {
    float *t;
    float *bary_u;
    float *bary_v;
    std::int32_t *shape_id;
    std::int32_t *local_prim_id;
    std::size_t count;
};

/// Caller-owned depth-major traversal stack plus one overflow flag per ray. The
/// caller must supply at least query_stride * stack_depth node slots and
/// query_count overflow ints. Depth-major storage keeps equal-depth warp
/// accesses coalesced. The kernel writes zero to overflow before traversal and
/// one if the ray exhausts stack_depth, so the host can repair those lanes.
struct TriangleTraversalScratchView {
    std::int32_t *node_indices;
    std::int32_t *overflow;
    std::size_t query_stride;
    std::size_t stack_depth;
    std::size_t capacity;
    std::size_t overflow_capacity;
};

struct TriangleClosestHitParams {
    TriangleSoAView triangles;
    AabbSoAView node_bounds;
    CompactBvhTopologyView topology;
    TriangleRaySoAView rays;
    TrianglePrimIdMapView prim_map;
    TriangleClosestHitOutputView output;
    TriangleTraversalScratchView scratch;
    float t_min;
    cudaStream_t stream;
};

struct TriangleOccludedParams {
    TriangleSoAView triangles;
    AabbSoAView node_bounds;
    CompactBvhTopologyView topology;
    TriangleRaySoAView rays;
    std::int32_t *out_hit;  ///< One int per ray: 1 when any surface is within [t_min, t_max].
    TriangleTraversalScratchView scratch;
    float t_min;
    cudaStream_t stream;
};

struct TriangleFirstBlockerParams {
    TriangleSoAView triangles;
    AabbSoAView node_bounds;
    CompactBvhTopologyView topology;
    TriangleRaySoAView rays;
    std::int32_t *out_global_prim_id;  ///< Closest blocker global id, or -1 on a miss.
    /// Optional per-ray ignore list, row-major with `ignore_stride` entries per
    /// ray; entries name global primitive ids to treat as non-occluding, -1 pads
    /// unused slots. Null (or ignore_stride == 0) means no ignores.
    const std::int32_t *ignore_prim_ids;
    std::int32_t ignore_stride;
    TriangleTraversalScratchView scratch;
    float t_min;
    cudaStream_t stream;
};

/// Closest-hit BVH traversal writing the winning (t, bary, shape_id,
/// local_prim_id) per ray with a deterministic (t, global_prim_id) tie-break.
void launch_triangle_closest_hit_async(const TriangleClosestHitParams &params);

/// Any-hit occlusion traversal: out_hit[ray] = 1 on the first surface in range.
void launch_triangle_occluded_async(const TriangleOccludedParams &params);

/// Closest-blocker traversal honoring a per-ray ignore list.
void launch_triangle_first_blocker_async(const TriangleFirstBlockerParams &params);

/// Brute-force closest-hit over all primitives, restricted to rays whose
/// scratch.overflow flag is set. Used only as the traversal-stack overflow
/// repair path; a no-op when no lane overflowed.
void launch_triangle_closest_hit_repair_async(const TriangleClosestHitParams &params);

/// Brute-force occlusion repair for overflowed lanes (see the closest-hit repair).
void launch_triangle_occluded_repair_async(const TriangleOccludedParams &params);

#define RAYD_SHARED_BVH_TRI_ASSERT_POD(Type)                                  \
    static_assert(std::is_standard_layout_v<Type>);                           \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleSoAView);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleRaySoAView);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TrianglePrimIdMapView);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleClosestHitOutputView);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleTraversalScratchView);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleClosestHitParams);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleOccludedParams);
RAYD_SHARED_BVH_TRI_ASSERT_POD(TriangleFirstBlockerParams);

#undef RAYD_SHARED_BVH_TRI_ASSERT_POD

} // namespace rayd::shared::bvh
