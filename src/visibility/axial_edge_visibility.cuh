// Copyright Xingyu Chen.
// Defines the exact four-sample axial-edge visibility lane shared by the OptiX and CUDA trace backends.

#pragma once

#include <rayd/math.h>
#include <src/visibility/axial_edge_visibility_params.h>
#include <src/visibility/segment_visibility.cuh>

#include <cmath>
#include <cstdint>

namespace rayd::torch_backend::axial_edge_visibility {

using shared::math::Vec3f;

__device__ __forceinline__ bool finite(float value) {
    return isfinite(value) != 0;
}

__device__ __forceinline__ bool finite(Vec3f value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

__device__ __forceinline__ Vec3f load_aos3(const float* values, unsigned int row) {
    const unsigned int offset = row * 3u;
    return {values[offset], values[offset + 1u], values[offset + 2u]};
}

// Sample points are built only from round-to-nearest add, subtract, and multiply instructions, without FTZ and without
// FMA contraction, so every trace backend and every compile flag set reaches the same float32 point.
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

__device__ __forceinline__ Vec3f exact_sample_point(Vec3f edge_position, Vec3f edge_direction, float edge_t_min,
                                                    float edge_t_max, float fraction) {
    const float span = point_sub_rn(edge_t_max, edge_t_min);
    const float t = point_add_rn(edge_t_min, point_mul_rn(fraction, span));
    return {
        point_add_rn(edge_position.x, point_mul_rn(t, edge_direction.x)),
        point_add_rn(edge_position.y, point_mul_rn(t, edge_direction.y)),
        point_add_rn(edge_position.z, point_mul_rn(t, edge_direction.z)),
    };
}

// One state: visible when any of the four exact samples on the edge span reaches the transmitter unoccluded. Inactive
// states and states with a non-finite input are not visible. `params.trace` carries the occlusion handle only, so a
// blocker is never re-tested against its own surface plane.
template <typename Config>
__device__ __forceinline__ void exact_state(const AxialEdgeVisibilityParams& params, unsigned int state,
                                            const typename Config::Traverser& traverser) {
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
    if (!finite(tx) || !finite(edge_position) || !finite(edge_direction) || !finite(edge_t_min) ||
        !finite(edge_t_max)) {
        params.out_any_visible[state] = 0u;
        return;
    }

    std::uint32_t any_visible = 0u;
#pragma unroll
    for (int sample_index = 0; sample_index < AxialEdgeVisibilitySampleCount; ++sample_index) {
        const Vec3f sample = exact_sample_point(edge_position, edge_direction, edge_t_min, edge_t_max,
                                                params.sample_fractions[sample_index]);
        if (finite(sample)) {
            any_visible |=
                shared::multipath::segment_visibility_algo_detail::trace_segment<Config>(params.trace, tx, sample, true,
                                                                                         0u, traverser, nullptr);
        }
    }
    params.out_any_visible[state] = any_visible != 0u ? 1u : 0u;
}

} // namespace rayd::torch_backend::axial_edge_visibility
