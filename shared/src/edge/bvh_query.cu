#include <rayd/shared/edge/bvh_query.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <rayd/shared/edge/edge_distance_math.h>
#include <rayd/shared/math/vec3.h>

namespace rayd::shared::edge {
namespace {

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

__device__ __forceinline__ math::Vec3f load_edge_origin(const EdgeSoAView &edges,
                                                         int edge) {
    return {edges.p0_x[edge], edges.p0_y[edge], edges.p0_z[edge]};
}

__device__ __forceinline__ math::Vec3f load_edge_vector(const EdgeSoAView &edges,
                                                         int edge) {
    return {edges.direction_x[edge], edges.direction_y[edge], edges.direction_z[edge]};
}

__device__ __forceinline__ math::Vec3f load_bound_min(const AabbSoAView &bounds,
                                                       int node) {
    return {bounds.min_x[node], bounds.min_y[node], bounds.min_z[node]};
}

__device__ __forceinline__ math::Vec3f load_bound_max(const AabbSoAView &bounds,
                                                       int node) {
    return {bounds.max_x[node], bounds.max_y[node], bounds.max_z[node]};
}

__device__ __forceinline__ float point_aabb_distance_squared(math::Vec3f point,
                                                              math::Vec3f lower,
                                                              math::Vec3f upper) {
    const float dx = fmaxf(lower.x - point.x, 0.0f) +
                     fmaxf(point.x - upper.x, 0.0f);
    const float dy = fmaxf(lower.y - point.y, 0.0f) +
                     fmaxf(point.y - upper.y, 0.0f);
    const float dz = fmaxf(lower.z - point.z, 0.0f) +
                     fmaxf(point.z - upper.z, 0.0f);
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float line_aabb_sphere_lower_bound_squared(
    math::Vec3f origin,
    math::Vec3f direction,
    math::Vec3f lower,
    math::Vec3f upper) {
    const float direction_squared = math::squared_norm(direction);
    if (!(direction_squared > EdgeDistanceDeviceEpsilon)) {
        return 0.0f;
    }
    const math::Vec3f center = math::scale(math::add(lower, upper), 0.5f);
    const math::Vec3f half_extent =
        math::scale(math::subtract(upper, lower), 0.5f);
    const float t = math::dot(math::subtract(center, origin), direction) /
                    direction_squared;
    const math::Vec3f closest = math::add(origin, math::scale(direction, t));
    const float center_distance =
        sqrtf(fmaxf(math::squared_norm(math::subtract(center, closest)), 0.0f));
    const float radius = sqrtf(fmaxf(math::squared_norm(half_extent), 0.0f));
    const float separation = fmaxf(center_distance - radius, 0.0f);
    return separation * separation;
}

__device__ __forceinline__ float segment_aabb_lower_bound_squared(
    math::Vec3f origin,
    math::Vec3f segment,
    math::Vec3f lower,
    math::Vec3f upper) {
    const math::Vec3f finish = math::add(origin, segment);
    const math::Vec3f path_min = {
        fminf(origin.x, finish.x), fminf(origin.y, finish.y), fminf(origin.z, finish.z)};
    const math::Vec3f path_max = {
        fmaxf(origin.x, finish.x), fmaxf(origin.y, finish.y), fmaxf(origin.z, finish.z)};
    const float box_bound = point_aabb_distance_squared(
        {fminf(fmaxf(lower.x, path_min.x), path_max.x),
         fminf(fmaxf(lower.y, path_min.y), path_max.y),
         fminf(fmaxf(lower.z, path_min.z), path_max.z)},
        lower,
        upper);
    const float line_bound =
        line_aabb_sphere_lower_bound_squared(origin, segment, lower, upper);
    return fmaxf(box_bound, line_bound);
}

__device__ __forceinline__ float ray_axis_separation(float origin,
                                                      float direction,
                                                      float lower,
                                                      float upper) {
    if (direction > EdgeDistanceDeviceEpsilon) {
        return fmaxf(origin - upper, 0.0f);
    }
    if (direction < -EdgeDistanceDeviceEpsilon) {
        return fmaxf(lower - origin, 0.0f);
    }
    return fmaxf(lower - origin, 0.0f) + fmaxf(origin - upper, 0.0f);
}

__device__ __forceinline__ float ray_aabb_lower_bound_squared(
    math::Vec3f origin,
    math::Vec3f direction,
    math::Vec3f lower,
    math::Vec3f upper) {
    const float dx = ray_axis_separation(origin.x, direction.x, lower.x, upper.x);
    const float dy = ray_axis_separation(origin.y, direction.y, lower.y, upper.y);
    const float dz = ray_axis_separation(origin.z, direction.z, lower.z, upper.z);
    const float axis_bound = dx * dx + dy * dy + dz * dz;
    const float line_bound =
        line_aabb_sphere_lower_bound_squared(origin, direction, lower, upper);
    return fmaxf(axis_bound, line_bound);
}

__device__ __forceinline__ float query_bound_squared(const QueryGeometry &query,
                                                      math::Vec3f lower,
                                                      math::Vec3f upper) {
    if (!query.is_ray) {
        return point_aabb_distance_squared(query.origin, lower, upper);
    }
    if (query.is_finite) {
        return segment_aabb_lower_bound_squared(
            query.origin, math::scale(query.direction, query.extent), lower, upper);
    }
    return ray_aabb_lower_bound_squared(query.origin, query.direction, lower, upper);
}

__device__ __forceinline__ CandidateDistance exact_distance(
    const QueryGeometry &query,
    math::Vec3f edge_origin,
    math::Vec3f edge_vector) {
    if (!query.is_ray) {
        const PointSegmentDistance result =
            point_segment_distance(query.origin, edge_origin, edge_vector);
        return {result.squared_distance, result.edge_parameter, 0.0f};
    }
    if (query.is_finite) {
        const SegmentSegmentDistance result = segment_segment_distance(
            query.origin,
            math::scale(query.direction, query.extent),
            edge_origin,
            edge_vector);
        return {result.squared_distance,
                result.edge_parameter,
                result.query_parameter * query.extent};
    }
    const RaySegmentDistance result =
        ray_segment_distance(query.origin, query.direction, edge_origin, edge_vector);
    return {result.squared_distance, result.edge_parameter, result.ray_parameter};
}

__device__ __forceinline__ bool candidate_precedes(float distance,
                                                    int edge,
                                                    float slot_distance,
                                                    int slot_edge) {
    return distance < slot_distance ||
           (distance == slot_distance && edge < slot_edge);
}

template <int TopKCapacity>
__device__ __forceinline__ void insert_candidate(
    int k,
    int edge,
    CandidateDistance candidate,
    int (&edge_ids)[TopKCapacity],
    float (&distances)[TopKCapacity],
    float (&edge_parameters)[TopKCapacity],
    float (&query_parameters)[TopKCapacity]) {
#pragma unroll
    for (int rank = 0; rank < TopKCapacity; ++rank) {
        if (rank >= k) {
            break;
        }
        if (!candidate_precedes(candidate.squared_distance,
                                edge,
                                distances[rank],
                                edge_ids[rank])) {
            continue;
        }
        const int displaced_edge = edge_ids[rank];
        const CandidateDistance displaced = {
            distances[rank], edge_parameters[rank], query_parameters[rank]};
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

__device__ __forceinline__ void initialize_output(const EdgeQueryOutputView &output,
                                                   std::size_t query,
                                                   int k) {
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

__device__ __forceinline__ bool stack_push(const BvhTraversalScratchView &scratch,
                                            std::size_t query,
                                            std::size_t depth,
                                            int node) {
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

__device__ __forceinline__ int stack_load(const BvhTraversalScratchView &scratch,
                                           std::size_t query,
                                           std::size_t depth_index) {
    return scratch.node_indices[depth_index * scratch.query_stride + query];
}

template <int TopKCapacity, bool RayQuery, typename Params>
__global__ void bvh_query_kernel(Params params) {
    static_assert(TopKCapacity == 1 || TopKCapacity == 2 ||
                  TopKCapacity == 4 || TopKCapacity == 8 ||
                  TopKCapacity == 16);
    const std::size_t query =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
    if (params.scratch.overflow != nullptr &&
        query < params.scratch.overflow_capacity) {
        params.scratch.overflow[query] = 0u;
    }
    if (params.active_mask != nullptr && params.active_mask[query] == 0u) {
        return;
    }

    QueryGeometry geometry = {};
    if constexpr (RayQuery) {
        geometry.origin = {params.rays.origin_x[query],
                           params.rays.origin_y[query],
                           params.rays.origin_z[query]};
        geometry.direction = {params.rays.direction_x[query],
                              params.rays.direction_y[query],
                              params.rays.direction_z[query]};
        const float t_max = params.rays.t_max != nullptr
            ? params.rays.t_max[query]
            : CUDART_INF_F;
        geometry.is_ray = true;
        geometry.is_finite = isfinite(t_max);
        geometry.extent = geometry.is_finite ? fmaxf(t_max, 0.0f) : CUDART_INF_F;
        geometry.valid = !isnan(t_max) && (geometry.is_finite || t_max > 0.0f);
    } else {
        geometry.origin = {params.points.x[query],
                           params.points.y[query],
                           params.points.z[query]};
        geometry.direction = {0.0f, 0.0f, 0.0f};
        geometry.extent = 0.0f;
        geometry.is_ray = false;
        geometry.is_finite = true;
        geometry.valid = true;
    }
    if (!geometry.valid || params.topology.node_count == 0 ||
        params.topology.primitive_count == 0 ||
        params.node_bounds.count < params.topology.node_count ||
        params.topology.left_child == nullptr ||
        params.topology.right_child == nullptr ||
        params.topology.leaf_primitives == nullptr) {
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
        if (params.topology.node_active_count != nullptr &&
            params.topology.node_active_count[current] == 0) {
            current = stack_size > 0
                ? stack_load(params.scratch, query, --stack_size)
                : -1;
            continue;
        }

        const math::Vec3f lower = load_bound_min(params.node_bounds, current);
        const math::Vec3f upper = load_bound_max(params.node_bounds, current);
        if (query_bound_squared(geometry, lower, upper) > distances[k - 1]) {
            current = stack_size > 0
                ? stack_load(params.scratch, query, --stack_size)
                : -1;
            continue;
        }

        const int encoded_left = params.topology.left_child[current];
        if (encoded_left < 0) {
            const std::size_t begin = static_cast<std::size_t>(-encoded_left - 1);
            const int leaf_count = params.topology.right_child[current];
            if (leaf_count < 0 || leaf_count > kBvhLeafSize ||
                begin > params.topology.leaf_primitive_count ||
                static_cast<std::size_t>(leaf_count) >
                    params.topology.leaf_primitive_count - begin) {
                overflow = true;
                break;
            }
            for (int item = 0; item < leaf_count; ++item) {
                const int edge = params.topology.leaf_primitives[begin + item];
                if (edge < 0 ||
                    static_cast<std::size_t>(edge) >= params.edges.count ||
                    (params.edge_mask != nullptr && params.edge_mask[edge] == 0u)) {
                    continue;
                }
                insert_candidate<TopKCapacity>(
                    k,
                    edge,
                    exact_distance(geometry,
                                   load_edge_origin(params.edges, edge),
                                   load_edge_vector(params.edges, edge)),
                    edge_ids,
                    distances,
                    edge_parameters,
                    query_parameters);
            }
            current = stack_size > 0
                ? stack_load(params.scratch, query, --stack_size)
                : -1;
            continue;
        }

        const int left = encoded_left;
        const int right = params.topology.right_child[current];
        if (left < 0 || right < 0 ||
            static_cast<std::size_t>(left) >= params.topology.node_count ||
            static_cast<std::size_t>(right) >= params.topology.node_count) {
            overflow = true;
            break;
        }
        const float left_bound = query_bound_squared(
            geometry,
            load_bound_min(params.node_bounds, left),
            load_bound_max(params.node_bounds, left));
        const float right_bound = query_bound_squared(
            geometry,
            load_bound_min(params.node_bounds, right),
            load_bound_max(params.node_bounds, right));
        const bool left_active = params.topology.node_active_count == nullptr ||
                                 params.topology.node_active_count[left] > 0;
        const bool right_active = params.topology.node_active_count == nullptr ||
                                  params.topology.node_active_count[right] > 0;
        const bool visit_left = left_active && left_bound <= distances[k - 1];
        const bool visit_right = right_active && right_bound <= distances[k - 1];
        if (visit_left && visit_right) {
            const bool left_first = left_bound < right_bound ||
                                    (left_bound == right_bound && left < right);
            const int near_child = left_first ? left : right;
            const int far_child = left_first ? right : left;
            if (!stack_push(params.scratch, query, stack_size, far_child)) {
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
            current = stack_size > 0
                ? stack_load(params.scratch, query, --stack_size)
                : -1;
        }
    }

    if (overflow) {
        if (params.scratch.overflow != nullptr &&
            query < params.scratch.overflow_capacity) {
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
void launch_bvh_query_capacity(const Params &params,
                               unsigned int blocks) {
    bvh_query_kernel<TopKCapacity, RayQuery, Params>
        <<<blocks, kBlockSize, 0, params.stream>>>(params);
}

template <bool RayQuery, typename Params>
void dispatch_bvh_query_capacity(const Params &params,
                                 unsigned int blocks) {
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

} // namespace

void launch_point_bvh_query_async(const PointBvhQueryParams &params) {
    const std::size_t count = params.points.count;
    if (count == 0 || params.output.query_count == 0 ||
        params.output.result_count == 0 ||
        params.output.result_count > EdgeBvhTopKMax) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(kBlockSize) - 1) / kBlockSize);
    dispatch_bvh_query_capacity<false>(params, blocks);
}

void launch_ray_bvh_query_async(const RayBvhQueryParams &params) {
    const std::size_t count = params.rays.count;
    if (count == 0 || params.output.query_count == 0 ||
        params.output.result_count == 0 ||
        params.output.result_count > EdgeBvhTopKMax) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(kBlockSize) - 1) / kBlockSize);
    dispatch_bvh_query_capacity<true>(params, blocks);
}

} // namespace rayd::shared::edge
