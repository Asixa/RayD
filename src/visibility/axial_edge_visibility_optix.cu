// Copyright Xingyu Chen.
// Implements visibility support for axial edge visibility optix.

#include <rayd/math.h>
#include <rayd/visibility/segment_algo.h>
#include <rayd/visibility/segment_optix_device.cuh>
#include <rayd/rt/numeric_policy.h>
#include <src/visibility/axial_edge_visibility_params.h>

#include <cmath>
#include <cstdint>

namespace rayd::torch_backend {

extern "C" {
__constant__ AxialEdgeVisibilityParams params;
}

namespace {

using ExactPolicy = shared::optix::SegmentVisibilityDevicePolicy<true, false>;
using ExactTraverser =
    shared::optix::segment_visibility::SegmentVisibilityOptixTraverser<true>;
using ExactConfig = shared::rt::TraceConfig<ExactPolicy, ExactTraverser>;
using shared::math::Vec3f;

__device__ __forceinline__ bool finite(float value) {
    return isfinite(value) != 0;
}

__device__ __forceinline__ bool finite(Vec3f value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

__device__ __forceinline__ Vec3f load_aos3(const float *values, unsigned int row) {
    const unsigned int offset = row * 3u;
    return {values[offset], values[offset + 1u], values[offset + 2u]};
}

__device__ __forceinline__ float point_sub_rn(float lhs, float rhs) {
    float result;
    asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float point_mul_rn(float lhs, float rhs) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float point_add_rn(float lhs, float rhs) {
    float result;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ Vec3f exact_sample_point(
    Vec3f edge_position,
    Vec3f edge_direction,
    float edge_t_min,
    float edge_t_max,
    float fraction) {
    const float span = point_sub_rn(edge_t_max, edge_t_min);
    const float t = point_add_rn(edge_t_min, point_mul_rn(fraction, span));
    return {
        point_add_rn(edge_position.x, point_mul_rn(t, edge_direction.x)),
        point_add_rn(edge_position.y, point_mul_rn(t, edge_direction.y)),
        point_add_rn(edge_position.z, point_mul_rn(t, edge_direction.z)),
    };
}

} // namespace

extern "C" __global__ void __closesthit__axial_edge_visibility_exact() {
    optixSetPayload_0(0u);
    optixSetPayload_1(0xFFFFFFFFu);
}

extern "C" __global__ void __miss__axial_edge_visibility_exact() {}

extern "C" __global__ void __raygen__axial_edge_visibility_exact() {
    const unsigned int state = optixGetLaunchIndex().x;
    if (state >= static_cast<unsigned int>(params.state_count))
        return;

    if (params.active != nullptr && params.active[state] == 0u) {
        params.out_any_visible[state] = 0u;
        return;
    }

    const Vec3f tx = {params.tx[0], params.tx[1], params.tx[2]};
    const Vec3f edge_position = load_aos3(params.edge_position, state);
    const Vec3f edge_direction = load_aos3(params.edge_direction, state);
    const float edge_t_min = params.edge_t_min[state];
    const float edge_t_max = params.edge_t_max[state];
    if (!finite(tx) || !finite(edge_position) || !finite(edge_direction) ||
        !finite(edge_t_min) || !finite(edge_t_max)) {
        params.out_any_visible[state] = 0u;
        return;
    }

    const ExactTraverser traverser{
        static_cast<::OptixTraversableHandle>(params.trace.handle), nullptr};
    std::uint32_t any_visible = 0u;
#pragma unroll
    for (int sample_index = 0;
         sample_index < AxialEdgeVisibilitySampleCount;
         ++sample_index) {
        const Vec3f sample = exact_sample_point(
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            params.sample_fractions[sample_index]);
        if (finite(sample)) {
            any_visible |= shared::multipath::segment_visibility_algo_detail::
                trace_segment<ExactConfig>(
                    params.trace,
                    tx,
                    sample,
                    true,
                    0u,
                    traverser,
                    nullptr);
        }
    }
    params.out_any_visible[state] = any_visible != 0u ? 1u : 0u;
}

} // namespace rayd::torch_backend
