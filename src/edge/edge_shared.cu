// Copyright Xingyu Chen.
// Implements edge support for edge shared.

#include <src/edge/bvh_build.h>

#include <src/bvh_build.h>

namespace rayd::shared::edge {
namespace {

constexpr int kBlockSize = 256;

__device__ inline void store_bounds(int index, const BvhBounds3& value, MutableAabbSoAView bounds) {
    bounds.min_x[index] = value.min.x;
    bounds.min_y[index] = value.min.y;
    bounds.min_z[index] = value.min.z;
    bounds.max_x[index] = value.max.x;
    bounds.max_y[index] = value.max.y;
    bounds.max_z[index] = value.max.z;
}

__global__ void compute_primitive_bounds_kernel(PrimitiveBoundsParams params) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= static_cast<int>(params.edges.count))
        return;
    const BvhFloat3 p0{params.edges.p0_x[primitive], params.edges.p0_y[primitive], params.edges.p0_z[primitive]};
    const BvhFloat3 p1{p0.x + params.edges.direction_x[primitive], p0.y + params.edges.direction_y[primitive],
                       p0.z + params.edges.direction_z[primitive]};
    const BvhBounds3 bounds{math::component_min(p0, p1), math::component_max(p0, p1)};
    store_bounds(primitive, bounds, params.primitive_bounds);
    params.packed_bounds[primitive] = bounds;
}

} // namespace

void launch_compute_primitive_bounds_async(const PrimitiveBoundsParams& params) {
    const int count = static_cast<int>(params.edges.count);
    if (count == 0)
        return;
    compute_primitive_bounds_kernel<<<(count + kBlockSize - 1) / kBlockSize, kBlockSize, 0, params.stream>>>(params);
}

// Thin edge adapters that forward to the shared primitive-agnostic launchers.
// The edge parameter names alias the shared bvh structs (see bvh_build.h), so
// these calls stay byte-for-byte identical to the shared launch sequence.
void launch_init_sequence_async(const SequenceInitParams& params) {
    bvh::launch_init_sequence_async(params);
}

void launch_compute_morton_codes_async(const MortonCodeParams& params) {
    bvh::launch_compute_morton_codes_async(params);
}

void launch_build_radix_tree_async(const RadixTreeParams& params) {
    bvh::launch_build_radix_tree_async(params);
}

void launch_finalize_leaves_and_bounds_async(const LeafBoundsFinalizeParams& params) {
    bvh::launch_finalize_leaves_and_bounds_async(params);
}

void launch_initialize_leaf_costs_async(const LeafCostParams& params) {
    bvh::launch_initialize_leaf_costs_async(params);
}

void launch_initialize_internal_costs_async(const InternalCostParams& params) {
    bvh::launch_initialize_internal_costs_async(params);
}

void launch_optimize_selected_treelets_async(const TreeletOptimizeParams& params) {
    bvh::launch_optimize_selected_treelets_async(params);
}

void launch_mark_dirty_ancestors_async(const DirtyAncestorMarkParams& params) {
    bvh::launch_mark_dirty_ancestors_async(params);
}

void launch_compact_dirty_level_async(const DirtyLevelCompactParams& params) {
    bvh::launch_compact_dirty_level_async(params);
}

void launch_refit_selected_internal_nodes_async(const InternalNodeRefitParams& params) {
    bvh::launch_refit_selected_internal_nodes_async(params);
}

} // namespace rayd::shared::edge

#include <src/edge/bvh_query.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <src/bvh_query_device.cuh>
#include <src/edge/edge_distance.h>
#include <rayd/math.h>

namespace rayd::shared::edge {
namespace query_detail {

constexpr int kBlockSize = 128;

struct QueryGeometry {
    math::Vec3f origin;
    math::Vec3f direction;
    float extent;
    bool is_ray;
    bool is_finite;
    bool valid;
};

struct CandidateDistance {
    float squared_distance;
    float edge_parameter;
    float query_parameter;
};

__device__ __forceinline__ math::Vec3f load_edge_origin(const EdgeSoAView& edges, int edge) {
    return {edges.p0_x[edge], edges.p0_y[edge], edges.p0_z[edge]};
}

__device__ __forceinline__ math::Vec3f load_edge_vector(const EdgeSoAView& edges, int edge) {
    return {edges.direction_x[edge], edges.direction_y[edge], edges.direction_z[edge]};
}

__device__ __forceinline__ math::Vec3f load_bound_min(const AabbSoAView& bounds, int node) {
    return {bounds.min_x[node], bounds.min_y[node], bounds.min_z[node]};
}

__device__ __forceinline__ math::Vec3f load_bound_max(const AabbSoAView& bounds, int node) {
    return {bounds.max_x[node], bounds.max_y[node], bounds.max_z[node]};
}

__device__ __forceinline__ float point_aabb_distance_squared(math::Vec3f point, math::Vec3f lower, math::Vec3f upper) {
    const float dx = fmaxf(lower.x - point.x, 0.0f) + fmaxf(point.x - upper.x, 0.0f);
    const float dy = fmaxf(lower.y - point.y, 0.0f) + fmaxf(point.y - upper.y, 0.0f);
    const float dz = fmaxf(lower.z - point.z, 0.0f) + fmaxf(point.z - upper.z, 0.0f);
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float line_aabb_sphere_lower_bound_squared(math::Vec3f origin, math::Vec3f direction,
                                                                      math::Vec3f lower, math::Vec3f upper) {
    const float direction_squared = math::squared_norm(direction);
    if (!(direction_squared > EdgeDistanceDeviceEpsilon)) {
        return 0.0f;
    }
    const math::Vec3f center = math::scale(math::add(lower, upper), 0.5f);
    const math::Vec3f half_extent = math::scale(math::subtract(upper, lower), 0.5f);
    const float t = math::dot(math::subtract(center, origin), direction) / direction_squared;
    const math::Vec3f closest = math::add(origin, math::scale(direction, t));
    const float center_distance = sqrtf(fmaxf(math::squared_norm(math::subtract(center, closest)), 0.0f));
    const float radius = sqrtf(fmaxf(math::squared_norm(half_extent), 0.0f));
    const float separation = fmaxf(center_distance - radius, 0.0f);
    return separation * separation;
}

__device__ __forceinline__ float segment_aabb_lower_bound_squared(math::Vec3f origin, math::Vec3f segment,
                                                                  math::Vec3f lower, math::Vec3f upper) {
    const math::Vec3f finish = math::add(origin, segment);
    const math::Vec3f path_min = {fminf(origin.x, finish.x), fminf(origin.y, finish.y), fminf(origin.z, finish.z)};
    const math::Vec3f path_max = {fmaxf(origin.x, finish.x), fmaxf(origin.y, finish.y), fmaxf(origin.z, finish.z)};
    const float box_bound = point_aabb_distance_squared({fminf(fmaxf(lower.x, path_min.x), path_max.x),
                                                         fminf(fmaxf(lower.y, path_min.y), path_max.y),
                                                         fminf(fmaxf(lower.z, path_min.z), path_max.z)},
                                                        lower, upper);
    const float line_bound = line_aabb_sphere_lower_bound_squared(origin, segment, lower, upper);
    return fmaxf(box_bound, line_bound);
}

__device__ __forceinline__ float ray_axis_separation(float origin, float direction, float lower, float upper) {
    if (direction > EdgeDistanceDeviceEpsilon) {
        return fmaxf(origin - upper, 0.0f);
    }
    if (direction < -EdgeDistanceDeviceEpsilon) {
        return fmaxf(lower - origin, 0.0f);
    }
    return fmaxf(lower - origin, 0.0f) + fmaxf(origin - upper, 0.0f);
}

__device__ __forceinline__ float ray_aabb_lower_bound_squared(math::Vec3f origin, math::Vec3f direction,
                                                              math::Vec3f lower, math::Vec3f upper) {
    const float dx = ray_axis_separation(origin.x, direction.x, lower.x, upper.x);
    const float dy = ray_axis_separation(origin.y, direction.y, lower.y, upper.y);
    const float dz = ray_axis_separation(origin.z, direction.z, lower.z, upper.z);
    const float axis_bound = dx * dx + dy * dy + dz * dz;
    const float line_bound = line_aabb_sphere_lower_bound_squared(origin, direction, lower, upper);
    return fmaxf(axis_bound, line_bound);
}

__device__ __forceinline__ float query_bound_squared(const QueryGeometry& query, math::Vec3f lower, math::Vec3f upper) {
    if (!query.is_ray) {
        return point_aabb_distance_squared(query.origin, lower, upper);
    }
    if (query.is_finite) {
        return segment_aabb_lower_bound_squared(query.origin, math::scale(query.direction, query.extent), lower, upper);
    }
    return ray_aabb_lower_bound_squared(query.origin, query.direction, lower, upper);
}

__device__ __forceinline__ CandidateDistance exact_distance(const QueryGeometry& query, math::Vec3f edge_origin,
                                                            math::Vec3f edge_vector) {
    if (!query.is_ray) {
        const PointSegmentDistance result = point_segment_distance(query.origin, edge_origin, edge_vector);
        return {result.squared_distance, result.edge_parameter, 0.0f};
    }
    if (query.is_finite) {
        const SegmentSegmentDistance result =
            segment_segment_distance(query.origin, math::scale(query.direction, query.extent), edge_origin,
                                     edge_vector);
        return {result.squared_distance, result.edge_parameter, result.query_parameter * query.extent};
    }
    const RaySegmentDistance result = ray_segment_distance(query.origin, query.direction, edge_origin, edge_vector);
    return {result.squared_distance, result.edge_parameter, result.ray_parameter};
}

__device__ __forceinline__ bool candidate_precedes(float distance, int edge, float slot_distance, int slot_edge) {
    return distance < slot_distance || (distance == slot_distance && edge < slot_edge);
}

template <int TopKCapacity>
__device__ __forceinline__ void insert_candidate(int k, int edge, CandidateDistance candidate,
                                                 int (&edge_ids)[TopKCapacity], float (&distances)[TopKCapacity],
                                                 float (&edge_parameters)[TopKCapacity],
                                                 float (&query_parameters)[TopKCapacity]) {
#pragma unroll
    for (int rank = 0; rank < TopKCapacity; ++rank) {
        if (rank >= k) {
            break;
        }
        if (!candidate_precedes(candidate.squared_distance, edge, distances[rank], edge_ids[rank])) {
            continue;
        }
        const int displaced_edge = edge_ids[rank];
        const CandidateDistance displaced = {distances[rank], edge_parameters[rank], query_parameters[rank]};
        edge_ids[rank] = edge;
        distances[rank] = candidate.squared_distance;
        edge_parameters[rank] = candidate.edge_parameter;
        query_parameters[rank] = candidate.query_parameter;
        edge = displaced_edge;
        candidate = displaced;
        if (edge < 0) {
            break;
        }
    }
}

__device__ __forceinline__ void initialize_output(const EdgeQueryOutputView& output, std::size_t query, int k) {
    for (int rank = 0; rank < k; ++rank) {
        const std::size_t slot = query * output.result_stride + rank;
        if (slot >= output.capacity) {
            continue;
        }
        output.edge_ids[slot] = -1;
        output.squared_distance[slot] = CUDART_INF_F;
        if (output.edge_parameter != nullptr) {
            output.edge_parameter[slot] = 0.0f;
        }
        if (output.query_parameter != nullptr) {
            output.query_parameter[slot] = 0.0f;
        }
    }
}

// The depth-major stack push/load helpers and the near/far tie-break are shared
// with any BVH consumer via <src/bvh_query_device.cuh>; the edge
// query calls bvh::stack_push / bvh::stack_load / bvh::near_child_is_left so the
// coalesced indexing and traversal order stay bitwise identical.

template <int TopKCapacity, bool RayQuery, typename Params> __global__ void bvh_query_kernel(Params params) {
    static_assert(TopKCapacity == 1 || TopKCapacity == 2 || TopKCapacity == 4 || TopKCapacity == 8 ||
                  TopKCapacity == 16);
    const std::size_t query = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t query_count = 0;
    if constexpr (RayQuery) {
        query_count = params.rays.count;
    } else {
        query_count = params.points.count;
    }
    if (query >= query_count || query >= params.output.query_count) {
        return;
    }

    const int k = static_cast<int>(params.output.result_count);
    if (k < 1 || k > TopKCapacity) {
        return;
    }
    initialize_output(params.output, query, k);
    if (params.scratch.overflow != nullptr && query < params.scratch.overflow_capacity) {
        params.scratch.overflow[query] = 0u;
    }
    if (params.active_mask != nullptr && params.active_mask[query] == 0u) {
        return;
    }

    QueryGeometry geometry = {};
    if constexpr (RayQuery) {
        geometry.origin = {params.rays.origin_x[query], params.rays.origin_y[query], params.rays.origin_z[query]};
        geometry.direction = {params.rays.direction_x[query], params.rays.direction_y[query],
                              params.rays.direction_z[query]};
        const float t_max = params.rays.t_max != nullptr ? params.rays.t_max[query] : CUDART_INF_F;
        geometry.is_ray = true;
        geometry.is_finite = isfinite(t_max);
        geometry.extent = geometry.is_finite ? fmaxf(t_max, 0.0f) : CUDART_INF_F;
        geometry.valid = !isnan(t_max) && (geometry.is_finite || t_max > 0.0f);
    } else {
        geometry.origin = {params.points.x[query], params.points.y[query], params.points.z[query]};
        geometry.direction = {0.0f, 0.0f, 0.0f};
        geometry.extent = 0.0f;
        geometry.is_ray = false;
        geometry.is_finite = true;
        geometry.valid = true;
    }
    if (!geometry.valid || params.topology.node_count == 0 || params.topology.primitive_count == 0 ||
        params.node_bounds.count < params.topology.node_count || params.topology.left_child == nullptr ||
        params.topology.right_child == nullptr || params.topology.leaf_primitives == nullptr) {
        return;
    }

    int edge_ids[TopKCapacity];
    float distances[TopKCapacity];
    float edge_parameters[TopKCapacity];
    float query_parameters[TopKCapacity];
#pragma unroll
    for (int rank = 0; rank < TopKCapacity; ++rank) {
        edge_ids[rank] = -1;
        distances[rank] = CUDART_INF_F;
        edge_parameters[rank] = 0.0f;
        query_parameters[rank] = 0.0f;
    }

    int current = 0;
    std::size_t stack_size = 0;
    bool overflow = false;
    while (current >= 0) {
        if (static_cast<std::size_t>(current) >= params.topology.node_count) {
            overflow = true;
            break;
        }
        if (params.topology.node_active_count != nullptr && params.topology.node_active_count[current] == 0) {
            current = stack_size > 0 ? bvh::stack_load(params.scratch, query, --stack_size) : -1;
            continue;
        }

        const math::Vec3f lower = load_bound_min(params.node_bounds, current);
        const math::Vec3f upper = load_bound_max(params.node_bounds, current);
        if (query_bound_squared(geometry, lower, upper) > distances[k - 1]) {
            current = stack_size > 0 ? bvh::stack_load(params.scratch, query, --stack_size) : -1;
            continue;
        }

        const int encoded_left = params.topology.left_child[current];
        if (encoded_left < 0) {
            const std::size_t begin = static_cast<std::size_t>(-encoded_left - 1);
            const int leaf_count = params.topology.right_child[current];
            if (leaf_count < 0 || leaf_count > kBvhLeafSize || begin > params.topology.leaf_primitive_count ||
                static_cast<std::size_t>(leaf_count) > params.topology.leaf_primitive_count - begin) {
                overflow = true;
                break;
            }
            for (int item = 0; item < leaf_count; ++item) {
                const int edge = params.topology.leaf_primitives[begin + item];
                if (edge < 0 || static_cast<std::size_t>(edge) >= params.edges.count ||
                    (params.edge_mask != nullptr && params.edge_mask[edge] == 0u)) {
                    continue;
                }
                insert_candidate<TopKCapacity>(k, edge,
                                               exact_distance(geometry, load_edge_origin(params.edges, edge),
                                                              load_edge_vector(params.edges, edge)),
                                               edge_ids, distances, edge_parameters, query_parameters);
            }
            current = stack_size > 0 ? bvh::stack_load(params.scratch, query, --stack_size) : -1;
            continue;
        }

        const int left = encoded_left;
        const int right = params.topology.right_child[current];
        if (left < 0 || right < 0 || static_cast<std::size_t>(left) >= params.topology.node_count ||
            static_cast<std::size_t>(right) >= params.topology.node_count) {
            overflow = true;
            break;
        }
        const float left_bound = query_bound_squared(geometry, load_bound_min(params.node_bounds, left),
                                                     load_bound_max(params.node_bounds, left));
        const float right_bound = query_bound_squared(geometry, load_bound_min(params.node_bounds, right),
                                                      load_bound_max(params.node_bounds, right));
        const bool left_active =
            params.topology.node_active_count == nullptr || params.topology.node_active_count[left] > 0;
        const bool right_active =
            params.topology.node_active_count == nullptr || params.topology.node_active_count[right] > 0;
        const bool visit_left = left_active && left_bound <= distances[k - 1];
        const bool visit_right = right_active && right_bound <= distances[k - 1];
        if (visit_left && visit_right) {
            const bool left_first = bvh::near_child_is_left(left_bound, right_bound, left, right);
            const int near_child = left_first ? left : right;
            const int far_child = left_first ? right : left;
            if (!bvh::stack_push(params.scratch, query, stack_size, far_child)) {
                overflow = true;
                break;
            }
            ++stack_size;
            current = near_child;
        } else if (visit_left) {
            current = left;
        } else if (visit_right) {
            current = right;
        } else {
            current = stack_size > 0 ? bvh::stack_load(params.scratch, query, --stack_size) : -1;
        }
    }

    if (overflow) {
        if (params.scratch.overflow != nullptr && query < params.scratch.overflow_capacity) {
            params.scratch.overflow[query] = 1u;
        }
        return;
    }
    for (int rank = 0; rank < k; ++rank) {
        const std::size_t slot = query * params.output.result_stride + rank;
        if (slot >= params.output.capacity || edge_ids[rank] < 0) {
            continue;
        }
        params.output.edge_ids[slot] = edge_ids[rank];
        params.output.squared_distance[slot] = distances[rank];
        if (params.output.edge_parameter != nullptr) {
            params.output.edge_parameter[slot] = edge_parameters[rank];
        }
        if (params.output.query_parameter != nullptr) {
            params.output.query_parameter[slot] = query_parameters[rank];
        }
    }
}

template <int TopKCapacity, bool RayQuery, typename Params>
void launch_bvh_query_capacity(const Params& params, unsigned int blocks) {
    bvh_query_kernel<TopKCapacity, RayQuery, Params><<<blocks, kBlockSize, 0, params.stream>>>(params);
}

template <bool RayQuery, typename Params> void dispatch_bvh_query_capacity(const Params& params, unsigned int blocks) {
    switch (edge_bvh_topk_capacity(params.output.result_count)) {
    case 1:
        launch_bvh_query_capacity<1, RayQuery>(params, blocks);
        break;
    case 2:
        launch_bvh_query_capacity<2, RayQuery>(params, blocks);
        break;
    case 4:
        launch_bvh_query_capacity<4, RayQuery>(params, blocks);
        break;
    case 8:
        launch_bvh_query_capacity<8, RayQuery>(params, blocks);
        break;
    case 16:
        launch_bvh_query_capacity<16, RayQuery>(params, blocks);
        break;
    default:
        break;
    }
}

} // namespace query_detail

void launch_point_bvh_query_async(const PointBvhQueryParams& params) {
    const std::size_t count = params.points.count;
    if (count == 0 || params.output.query_count == 0 || params.output.result_count == 0 ||
        params.output.result_count > EdgeBvhTopKMax) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(query_detail::kBlockSize) - 1) / query_detail::kBlockSize);
    query_detail::dispatch_bvh_query_capacity<false>(params, blocks);
}

void launch_ray_bvh_query_async(const RayBvhQueryParams& params) {
    const std::size_t count = params.rays.count;
    if (count == 0 || params.output.query_count == 0 || params.output.result_count == 0 ||
        params.output.result_count > EdgeBvhTopKMax) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(query_detail::kBlockSize) - 1) / query_detail::kBlockSize);
    query_detail::dispatch_bvh_query_capacity<true>(params, blocks);
}

} // namespace rayd::shared::edge

namespace rayd::shared::edge {
namespace aabb_detail {

__global__ void compute_edge_aabbs_kernel(int edge_count, const float* edge_p0_x, const float* edge_p0_y,
                                          const float* edge_p0_z, const float* edge_e1_x, const float* edge_e1_y,
                                          const float* edge_e1_z, float inflation, float* out_aabbs) {
    const int edge = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (edge >= edge_count)
        return;

    const float p0_x = edge_p0_x[edge];
    const float p0_y = edge_p0_y[edge];
    const float p0_z = edge_p0_z[edge];
    const float p1_x = p0_x + edge_e1_x[edge];
    const float p1_y = p0_y + edge_e1_y[edge];
    const float p1_z = p0_z + edge_e1_z[edge];
    const float radius = fmaxf(inflation, 0.0f);
    const int base = edge * 6;
    out_aabbs[base + 0] = fminf(p0_x, p1_x) - radius;
    out_aabbs[base + 1] = fminf(p0_y, p1_y) - radius;
    out_aabbs[base + 2] = fminf(p0_z, p1_z) - radius;
    out_aabbs[base + 3] = fmaxf(p0_x, p1_x) + radius;
    out_aabbs[base + 4] = fmaxf(p0_y, p1_y) + radius;
    out_aabbs[base + 5] = fmaxf(p0_z, p1_z) + radius;
}

} // namespace aabb_detail

void launch_edge_aabb(int edge_count, const float* edge_p0_x, const float* edge_p0_y, const float* edge_p0_z,
                      const float* edge_e1_x, const float* edge_e1_y, const float* edge_e1_z, float inflation,
                      float* out_aabbs, cudaStream_t stream) {
    if (edge_count == 0)
        return;

    constexpr int block_size = 256;
    const int block_count = (edge_count + block_size - 1) / block_size;
    aabb_detail::compute_edge_aabbs_kernel<<<block_count, block_size, 0, stream>>>(edge_count, edge_p0_x, edge_p0_y,
                                                                                   edge_p0_z, edge_e1_x, edge_e1_y,
                                                                                   edge_e1_z, inflation, out_aabbs);
}

} // namespace rayd::shared::edge

namespace rayd::shared::edge {
namespace distance_detail {

constexpr int kBlockSize = 256;

__device__ __forceinline__ math::Vec3f load_point(const PointSoAView& points, std::size_t index) {
    return {points.x[index], points.y[index], points.z[index]};
}

__device__ __forceinline__ math::Vec3f load_ray_origin(const RaySoAView& rays, std::size_t index) {
    return {rays.origin_x[index], rays.origin_y[index], rays.origin_z[index]};
}

__device__ __forceinline__ math::Vec3f load_ray_direction(const RaySoAView& rays, std::size_t index) {
    return {rays.direction_x[index], rays.direction_y[index], rays.direction_z[index]};
}

__device__ __forceinline__ math::Vec3f load_edge_origin(const EdgeSoAView& edges, int edge) {
    return {edges.p0_x[edge], edges.p0_y[edge], edges.p0_z[edge]};
}

__device__ __forceinline__ math::Vec3f load_edge_vector(const EdgeSoAView& edges, int edge) {
    return {edges.direction_x[edge], edges.direction_y[edge], edges.direction_z[edge]};
}

__device__ __forceinline__ void write_invalid(const EdgeDistanceOutputView& output, std::size_t slot) {
    if (slot >= output.capacity) {
        return;
    }
    output.squared_distance[slot] = CUDART_INF_F;
    output.edge_parameter[slot] = 0.0f;
    output.query_parameter[slot] = 0.0f;
}

__global__ void point_edge_distances_kernel(PointEdgeDistanceParams params) {
    const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t item_count = params.candidates.query_count * params.candidates.candidate_count;
    if (item >= item_count) {
        return;
    }

    const std::size_t query = item / params.candidates.candidate_count;
    const std::size_t rank = item - query * params.candidates.candidate_count;
    const std::size_t slot = query * params.candidates.candidate_stride + rank;
    write_invalid(params.output, slot);
    if (slot >= params.output.capacity || query >= params.points.count ||
        (params.active_mask != nullptr && params.active_mask[query] == 0u)) {
        return;
    }

    const int edge = params.candidates.edge_ids[slot];
    if (edge < 0 || static_cast<std::size_t>(edge) >= params.edges.count ||
        (params.edge_mask != nullptr && params.edge_mask[edge] == 0u)) {
        return;
    }

    const PointSegmentDistance result =
        point_segment_distance(load_point(params.points, query), load_edge_origin(params.edges, edge),
                               load_edge_vector(params.edges, edge));
    params.output.squared_distance[slot] = result.squared_distance;
    params.output.edge_parameter[slot] = result.edge_parameter;
}

__global__ void ray_edge_distances_kernel(RayEdgeDistanceParams params) {
    const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t item_count = params.candidates.query_count * params.candidates.candidate_count;
    if (item >= item_count) {
        return;
    }

    const std::size_t query = item / params.candidates.candidate_count;
    const std::size_t rank = item - query * params.candidates.candidate_count;
    const std::size_t slot = query * params.candidates.candidate_stride + rank;
    write_invalid(params.output, slot);
    if (slot >= params.output.capacity || query >= params.rays.count ||
        (params.active_mask != nullptr && params.active_mask[query] == 0u)) {
        return;
    }

    const int edge = params.candidates.edge_ids[slot];
    if (edge < 0 || static_cast<std::size_t>(edge) >= params.edges.count ||
        (params.edge_mask != nullptr && params.edge_mask[edge] == 0u)) {
        return;
    }

    const math::Vec3f origin = load_ray_origin(params.rays, query);
    const math::Vec3f direction = load_ray_direction(params.rays, query);
    const math::Vec3f edge_origin = load_edge_origin(params.edges, edge);
    const math::Vec3f edge_vector = load_edge_vector(params.edges, edge);
    const float t_max = params.rays.t_max != nullptr ? params.rays.t_max[query] : CUDART_INF_F;
    if (isnan(t_max)) {
        return;
    }

    if (isfinite(t_max)) {
        const float extent = fmaxf(t_max, 0.0f);
        const SegmentSegmentDistance result =
            segment_segment_distance(origin, math::scale(direction, extent), edge_origin, edge_vector);
        params.output.squared_distance[slot] = result.squared_distance;
        params.output.edge_parameter[slot] = result.edge_parameter;
        params.output.query_parameter[slot] = result.query_parameter * extent;
    } else if (t_max > 0.0f) {
        const RaySegmentDistance result = ray_segment_distance(origin, direction, edge_origin, edge_vector);
        params.output.squared_distance[slot] = result.squared_distance;
        params.output.edge_parameter[slot] = result.edge_parameter;
        params.output.query_parameter[slot] = result.ray_parameter;
    }
}

} // namespace distance_detail

void launch_point_edge_distances_async(const PointEdgeDistanceParams& params) {
    const std::size_t count = params.candidates.query_count * params.candidates.candidate_count;
    if (count == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(distance_detail::kBlockSize) - 1) / distance_detail::kBlockSize);
    distance_detail::point_edge_distances_kernel<<<blocks, distance_detail::kBlockSize, 0, params.stream>>>(params);
}

void launch_ray_edge_distances_async(const RayEdgeDistanceParams& params) {
    const std::size_t count = params.candidates.query_count * params.candidates.candidate_count;
    if (count == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(distance_detail::kBlockSize) - 1) / distance_detail::kBlockSize);
    distance_detail::ray_edge_distances_kernel<<<blocks, distance_detail::kBlockSize, 0, params.stream>>>(params);
}

} // namespace rayd::shared::edge
