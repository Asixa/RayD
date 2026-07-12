#include <rayd/shared/edge/edge_distance.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <rayd/shared/edge/edge_distance_math.h>
#include <rayd/shared/math/vec3.h>

namespace rayd::shared::edge {
namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ math::Vec3f load_point(const PointSoAView &points,
                                                   std::size_t index) {
    return {points.x[index], points.y[index], points.z[index]};
}

__device__ __forceinline__ math::Vec3f load_ray_origin(const RaySoAView &rays,
                                                        std::size_t index) {
    return {rays.origin_x[index], rays.origin_y[index], rays.origin_z[index]};
}

__device__ __forceinline__ math::Vec3f load_ray_direction(const RaySoAView &rays,
                                                           std::size_t index) {
    return {rays.direction_x[index], rays.direction_y[index], rays.direction_z[index]};
}

__device__ __forceinline__ math::Vec3f load_edge_origin(const EdgeSoAView &edges,
                                                         int edge) {
    return {edges.p0_x[edge], edges.p0_y[edge], edges.p0_z[edge]};
}

__device__ __forceinline__ math::Vec3f load_edge_vector(const EdgeSoAView &edges,
                                                         int edge) {
    return {edges.direction_x[edge], edges.direction_y[edge], edges.direction_z[edge]};
}

__device__ __forceinline__ void write_invalid(const EdgeDistanceOutputView &output,
                                               std::size_t slot) {
    if (slot >= output.capacity) {
        return;
    }
    output.squared_distance[slot] = CUDART_INF_F;
    output.edge_parameter[slot] = 0.0f;
    output.query_parameter[slot] = 0.0f;
}

__global__ void point_edge_distances_kernel(PointEdgeDistanceParams params) {
    const std::size_t item =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t item_count =
        params.candidates.query_count * params.candidates.candidate_count;
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

    const PointSegmentDistance result = point_segment_distance(
        load_point(params.points, query),
        load_edge_origin(params.edges, edge),
        load_edge_vector(params.edges, edge));
    params.output.squared_distance[slot] = result.squared_distance;
    params.output.edge_parameter[slot] = result.edge_parameter;
}

__global__ void ray_edge_distances_kernel(RayEdgeDistanceParams params) {
    const std::size_t item =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t item_count =
        params.candidates.query_count * params.candidates.candidate_count;
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
    const float t_max = params.rays.t_max != nullptr
        ? params.rays.t_max[query]
        : CUDART_INF_F;
    if (isnan(t_max)) {
        return;
    }

    if (isfinite(t_max)) {
        const float extent = fmaxf(t_max, 0.0f);
        const SegmentSegmentDistance result = segment_segment_distance(
            origin, math::scale(direction, extent), edge_origin, edge_vector);
        params.output.squared_distance[slot] = result.squared_distance;
        params.output.edge_parameter[slot] = result.edge_parameter;
        params.output.query_parameter[slot] = result.query_parameter * extent;
    } else if (t_max > 0.0f) {
        const RaySegmentDistance result =
            ray_segment_distance(origin, direction, edge_origin, edge_vector);
        params.output.squared_distance[slot] = result.squared_distance;
        params.output.edge_parameter[slot] = result.edge_parameter;
        params.output.query_parameter[slot] = result.ray_parameter;
    }
}

} // namespace

void launch_point_edge_distances_async(const PointEdgeDistanceParams &params) {
    const std::size_t count =
        params.candidates.query_count * params.candidates.candidate_count;
    if (count == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(kBlockSize) - 1) / kBlockSize);
    point_edge_distances_kernel<<<blocks, kBlockSize, 0, params.stream>>>(params);
}

void launch_ray_edge_distances_async(const RayEdgeDistanceParams &params) {
    const std::size_t count =
        params.candidates.query_count * params.candidates.candidate_count;
    if (count == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (count + static_cast<std::size_t>(kBlockSize) - 1) / kBlockSize);
    ray_edge_distances_kernel<<<blocks, kBlockSize, 0, params.stream>>>(params);
}

} // namespace rayd::shared::edge
