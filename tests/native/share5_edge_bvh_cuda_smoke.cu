// Copyright Xingyu Chen.
// Exercises share5 edge bvh cuda smoke in a native smoke test.

#include <src/edge/bvh_query.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>

using namespace rayd::shared::edge;

template <typename T> T* managed(std::size_t count) {
    T* pointer = nullptr;
    return cudaMallocManaged(&pointer, sizeof(T) * count) == cudaSuccess ? pointer : nullptr;
}

int main() {
    float* edge = managed<float>(18);
    float* bounds = managed<float>(18);
    std::int32_t* topology = managed<std::int32_t>(6);
    std::int32_t* leaf_primitives = managed<std::int32_t>(3);
    float* query_data = managed<float>(10);
    std::int32_t* out_ids = managed<std::int32_t>(2);
    float* out_values = managed<float>(6);
    std::int32_t* stack = managed<std::int32_t>(1);
    std::uint8_t* overflow = managed<std::uint8_t>(1);
    std::uint8_t* active = managed<std::uint8_t>(1);
    std::uint8_t* mask = managed<std::uint8_t>(3);
    if (!edge || !bounds || !topology || !leaf_primitives || !query_data || !out_ids || !out_values || !stack ||
        !overflow || !active || !mask) {
        return 2;
    }

    float* p0_x = edge;
    float* p0_y = edge + 3;
    float* p0_z = edge + 6;
    float* d_x = edge + 9;
    float* d_y = edge + 12;
    float* d_z = edge + 15;
    p0_x[0] = -1.0f;
    p0_y[0] = 1.0f;
    p0_z[0] = 0.0f;
    d_x[0] = 2.0f;
    d_y[0] = 0.0f;
    d_z[0] = 0.0f;
    p0_x[1] = -1.0f;
    p0_y[1] = -1.0f;
    p0_z[1] = 0.0f;
    d_x[1] = 2.0f;
    d_y[1] = 0.0f;
    d_z[1] = 0.0f;
    p0_x[2] = 2.0f;
    p0_y[2] = -1.0f;
    p0_z[2] = 0.0f;
    d_x[2] = 0.0f;
    d_y[2] = 2.0f;
    d_z[2] = 0.0f;

    float* min_x = bounds;
    float* min_y = bounds + 3;
    float* min_z = bounds + 6;
    float* max_x = bounds + 9;
    float* max_y = bounds + 12;
    float* max_z = bounds + 15;
    min_x[0] = -1.0f;
    min_y[0] = -1.0f;
    min_z[0] = 0.0f;
    max_x[0] = 2.0f;
    max_y[0] = 1.0f;
    max_z[0] = 0.0f;
    min_x[1] = -1.0f;
    min_y[1] = -1.0f;
    min_z[1] = 0.0f;
    max_x[1] = 2.0f;
    max_y[1] = 1.0f;
    max_z[1] = 0.0f;
    min_x[2] = -1.0f;
    min_y[2] = 1.0f;
    min_z[2] = 0.0f;
    max_x[2] = 1.0f;
    max_y[2] = 1.0f;
    max_z[2] = 0.0f;

    std::int32_t* left = topology;
    std::int32_t* right = topology + 3;
    left[0] = 1;
    right[0] = 2;
    left[1] = -1;
    right[1] = 2;
    left[2] = -3;
    right[2] = 1;
    leaf_primitives[0] = 1;
    leaf_primitives[1] = 2;
    leaf_primitives[2] = 0;
    active[0] = 1u;
    mask[0] = mask[1] = mask[2] = 1u;
    query_data[0] = query_data[1] = query_data[2] = 0.0f;

    const EdgeSoAView edges = {p0_x, p0_y, p0_z, d_x, d_y, d_z, 3};
    const AabbSoAView node_bounds = {min_x, min_y, min_z, max_x, max_y, max_z, 3};
    const CompactBvhTopologyView compact = {left, right, leaf_primitives, nullptr, 3, 3, 3};
    EdgeQueryOutputView output = {out_ids, out_values, out_values + 2, out_values + 4, 1, 2, 2, 2};
    const BvhTraversalScratchView scratch = {stack, overflow, 1, 1, 1, 1};
    PointBvhQueryParams point_params = {edges,  node_bounds, compact, {query_data, query_data + 1, query_data + 2, 1},
                                        output, scratch,     active,  mask,
                                        nullptr};

    launch_point_bvh_query_async(point_params);
    if (cudaDeviceSynchronize() != cudaSuccess || out_ids[0] != 0 || out_ids[1] != 1 ||
        std::fabs(out_values[0] - 1.0f) > 1.0e-5f) {
        return 3;
    }

    mask[0] = 0u;
    point_params.output.result_count = 1;
    launch_point_bvh_query_async(point_params);
    if (cudaDeviceSynchronize() != cudaSuccess || out_ids[0] != 1) {
        return 4;
    }
    mask[0] = 1u;

    point_params.scratch.stack_depth = 0;
    launch_point_bvh_query_async(point_params);
    if (cudaDeviceSynchronize() != cudaSuccess || overflow[0] != 1u || out_ids[0] != -1) {
        return 5;
    }

    float* ray = query_data + 3;
    ray[0] = ray[1] = ray[2] = 0.0f;
    ray[3] = 1.0f;
    ray[4] = 0.0f;
    ray[5] = 0.0f;
    ray[6] = 0.5f;
    output.result_count = 1;
    RayBvhQueryParams ray_params = {edges,   node_bounds,
                                    compact, {ray, ray + 1, ray + 2, ray + 3, ray + 4, ray + 5, ray + 6, 1},
                                    output,  scratch,
                                    active,  mask,
                                    nullptr};
    launch_ray_bvh_query_async(ray_params);
    if (cudaDeviceSynchronize() != cudaSuccess || out_ids[0] != 0 || std::fabs(out_values[0] - 1.0f) > 1.0e-5f) {
        return 6;
    }

    ray[6] = std::numeric_limits<float>::infinity();
    launch_ray_bvh_query_async(ray_params);
    if (cudaDeviceSynchronize() != cudaSuccess || out_ids[0] != 2 || std::fabs(out_values[0]) > 1.0e-5f ||
        std::fabs(out_values[4] - 2.0f) > 1.0e-5f) {
        return 7;
    }

    std::puts("Share-5 edge BVH CUDA smoke passed.");
    return 0;
}
