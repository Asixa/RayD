#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/shared/edge/bvh_types.h>
#include <rayd/shared/edge/edge_distance.h>

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

/// Caller-owned per-query traversal stack.
struct BvhTraversalScratchView {
    std::int32_t *node_indices;
    std::size_t capacity;
};

struct PointBvhQueryParams {
    EdgeSoAView edges;
    AabbSoAView node_bounds;
    BvhTopologyView topology;
    PointSoAView points;
    EdgeQueryOutputView output;
    BvhTraversalScratchView scratch;
    cudaStream_t stream;
};

struct RayBvhQueryParams {
    EdgeSoAView edges;
    AabbSoAView node_bounds;
    BvhTopologyView topology;
    RaySoAView rays;
    EdgeQueryOutputView output;
    BvhTraversalScratchView scratch;
    cudaStream_t stream;
};

// This phase freezes the backend-neutral query layout only. The shared CUDA
// traversal entry points are added when Torch takes ownership of persistent BVH
// buffers in Share-5/F1; declaring unimplemented launchers here would create a
// misleading link-time API.

#define RAYD_SHARED_EDGE_ASSERT_POD(Type)                                     \
    static_assert(std::is_standard_layout_v<Type>);                           \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_EDGE_ASSERT_POD(EdgeQueryOutputView);
RAYD_SHARED_EDGE_ASSERT_POD(BvhTraversalScratchView);
RAYD_SHARED_EDGE_ASSERT_POD(PointBvhQueryParams);
RAYD_SHARED_EDGE_ASSERT_POD(RayBvhQueryParams);

#undef RAYD_SHARED_EDGE_ASSERT_POD

} // namespace rayd::shared::edge
