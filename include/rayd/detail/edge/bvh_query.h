#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/detail/edge/bvh_types.h>
#include <rayd/detail/edge/edge_distance.h>

namespace rayd::shared::edge {

/// Caller-owned top-k result storage, flattened with result_stride per query.
struct EdgeQueryOutputView {
    std::int32_t *edge_ids;
    float *squared_distance;
    float *edge_parameter;
    float *query_parameter;
    std::size_t query_count;
    std::size_t result_count;
    std::size_t result_stride;
    std::size_t capacity;
};

/// Caller-owned depth-major traversal stack. The caller must provide at least
/// query_stride * stack_depth entries and one overflow byte per query.
struct BvhTraversalScratchView {
    std::int32_t *node_indices;
    /// One byte per query. The kernel writes zero before traversal and one if
    /// the query exhausts stack_depth or observes invalid topology.
    std::uint8_t *overflow;
    /// Distance between adjacent stack depths. Set to at least query_count;
    /// depth-major storage keeps equal-depth warp accesses coalesced.
    std::size_t query_stride;
    /// Maximum number of deferred nodes per query.
    std::size_t stack_depth;
    /// Total number of entries in node_indices.
    std::size_t capacity;
    /// Total number of entries in overflow.
    std::size_t overflow_capacity;
};

struct PointBvhQueryParams {
    EdgeSoAView edges;
    AabbSoAView node_bounds;
    CompactBvhTopologyView topology;
    PointSoAView points;
    EdgeQueryOutputView output;
    BvhTraversalScratchView scratch;
    const std::uint8_t *active_mask;
    const std::uint8_t *edge_mask;
    cudaStream_t stream;
};

struct RayBvhQueryParams {
    EdgeSoAView edges;
    AabbSoAView node_bounds;
    CompactBvhTopologyView topology;
    RaySoAView rays;
    EdgeQueryOutputView output;
    BvhTraversalScratchView scratch;
    const std::uint8_t *active_mask;
    const std::uint8_t *edge_mask;
    cudaStream_t stream;
};

inline constexpr std::size_t EdgeBvhTopKMax =
    static_cast<std::size_t>(kBvhTopKMax);

/// Smallest compiled query-state capacity that can hold k results. Returning
/// zero marks an unsupported k. Keeping this mapping in the shared contract
/// lets both backend adapters use the same runtime dispatch semantics.
constexpr std::size_t edge_bvh_topk_capacity(std::size_t k) noexcept {
    return k == 0 || k > EdgeBvhTopKMax
        ? 0
        : k <= 1 ? 1
        : k <= 2 ? 2
        : k <= 4 ? 4
        : k <= 8 ? 8
                 : 16;
}

static_assert(EdgeBvhTopKMax == 16);
static_assert(edge_bvh_topk_capacity(0) == 0);
static_assert(edge_bvh_topk_capacity(1) == 1);
static_assert(edge_bvh_topk_capacity(2) == 2);
static_assert(edge_bvh_topk_capacity(3) == 4);
static_assert(edge_bvh_topk_capacity(4) == 4);
static_assert(edge_bvh_topk_capacity(5) == 8);
static_assert(edge_bvh_topk_capacity(6) == 8);
static_assert(edge_bvh_topk_capacity(7) == 8);
static_assert(edge_bvh_topk_capacity(8) == 8);
static_assert(edge_bvh_topk_capacity(9) == 16);
static_assert(edge_bvh_topk_capacity(10) == 16);
static_assert(edge_bvh_topk_capacity(11) == 16);
static_assert(edge_bvh_topk_capacity(12) == 16);
static_assert(edge_bvh_topk_capacity(13) == 16);
static_assert(edge_bvh_topk_capacity(14) == 16);
static_assert(edge_bvh_topk_capacity(15) == 16);
static_assert(edge_bvh_topk_capacity(16) == 16);
static_assert(edge_bvh_topk_capacity(17) == 0);

/// Traverse a compacted edge BVH for point queries. result_count selects k and
/// must be in [1, EdgeBvhTopKMax]. Runtime k dispatches to local-state capacity
/// 1, 2, 4, 8, or 16, rounding non-power-of-two k upward. Results are ordered by
/// (squared_distance, edge_id), so equal-distance ties are deterministic.
/// output.query_count must equal points.count, output.result_stride must be at
/// least k, and output.capacity must cover the final strided result. Null masks
/// mean all-active. The launch allocates nothing, performs no synchronization,
/// and is asynchronous on `params.stream`.
void launch_point_bvh_query_async(const PointBvhQueryParams &params);

/// Traverse a compacted edge BVH for ray queries. Finite t_max uses segment
/// semantics on [0, max(t_max, 0)]; positive infinity uses half-ray semantics.
/// result_count may be in [1, EdgeBvhTopKMax]. Null masks mean all-active. The
/// output and scratch shape rules match launch_point_bvh_query_async. The launch
/// allocates nothing, performs no synchronization, and is asynchronous on
/// `params.stream`.
void launch_ray_bvh_query_async(const RayBvhQueryParams &params);

#define RAYD_SHARED_EDGE_ASSERT_POD(Type)                                     \
    static_assert(std::is_standard_layout_v<Type>);                           \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_EDGE_ASSERT_POD(EdgeQueryOutputView);
RAYD_SHARED_EDGE_ASSERT_POD(BvhTraversalScratchView);
RAYD_SHARED_EDGE_ASSERT_POD(PointBvhQueryParams);
RAYD_SHARED_EDGE_ASSERT_POD(RayBvhQueryParams);

#undef RAYD_SHARED_EDGE_ASSERT_POD

} // namespace rayd::shared::edge
