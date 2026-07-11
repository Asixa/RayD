#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/shared/edge/bvh_types.h>

namespace rayd::shared::edge {

/// Structure-of-arrays point input for exact point-to-edge distance evaluation.
struct PointSoAView {
    const float *x;
    const float *y;
    const float *z;
    std::size_t count;
};

/// Structure-of-arrays ray input. Finite t_max gives segment semantics on [0, t_max].
struct RaySoAView {
    const float *origin_x;
    const float *origin_y;
    const float *origin_z;
    const float *direction_x;
    const float *direction_y;
    const float *direction_z;
    const float *t_max;
    std::size_t count;
};

/// Candidate edge IDs indexed as query_index * candidate_stride + candidate_index.
struct EdgeCandidateView {
    const std::int32_t *edge_ids;
    std::size_t query_count;
    std::size_t candidate_count;
    std::size_t candidate_stride;
};

/// Caller-owned exact-distance outputs using the same indexing as EdgeCandidateView.
struct EdgeDistanceOutputView {
    float *squared_distance;
    float *edge_parameter;
    float *query_parameter;
    std::size_t capacity;
};

struct PointEdgeDistanceParams {
    EdgeSoAView edges;
    PointSoAView points;
    EdgeCandidateView candidates;
    EdgeDistanceOutputView output;
    cudaStream_t stream;
};

struct RayEdgeDistanceParams {
    EdgeSoAView edges;
    RaySoAView rays;
    EdgeCandidateView candidates;
    EdgeDistanceOutputView output;
    cudaStream_t stream;
};

// Exact-distance launchers are introduced formula-by-formula in Share-2. These
// PODs define the stable raw-pointer/stream contract without advertising symbols
// that are not linked yet.

#define RAYD_SHARED_EDGE_ASSERT_POD(Type)                                     \
    static_assert(std::is_standard_layout_v<Type>);                           \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_EDGE_ASSERT_POD(PointSoAView);
RAYD_SHARED_EDGE_ASSERT_POD(RaySoAView);
RAYD_SHARED_EDGE_ASSERT_POD(EdgeCandidateView);
RAYD_SHARED_EDGE_ASSERT_POD(EdgeDistanceOutputView);
RAYD_SHARED_EDGE_ASSERT_POD(PointEdgeDistanceParams);
RAYD_SHARED_EDGE_ASSERT_POD(RayEdgeDistanceParams);

#undef RAYD_SHARED_EDGE_ASSERT_POD

} // namespace rayd::shared::edge
