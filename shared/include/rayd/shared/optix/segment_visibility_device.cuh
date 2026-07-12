#pragma once

#ifdef __CUDACC__

#include <optix.h>
#include <optix_device.h>

#include <rayd/shared/optix/device_hit.h>
#include <rayd/shared/optix/segment_visibility_params.h>

namespace rayd::shared::optix {

template <bool DisableAnyHitWithoutIgnore, bool WriteOutputT>
struct SegmentVisibilityDevicePolicy {
    static constexpr bool disable_anyhit_without_ignore =
        DisableAnyHitWithoutIgnore;
    static constexpr bool write_output_t = WriteOutputT;
};

namespace segment_visibility {

constexpr float TraceTMin = 1e-5f;
constexpr float RayBias = 1e-5f;
constexpr float MinSegmentLength = 2e-5f;

static __forceinline__ __device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 multiply(float scalar, float3 value) {
    return make_float3(scalar * value.x, scalar * value.y, scalar * value.z);
}

static __forceinline__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 load_aos_vec3(const float *value,
                                                       unsigned int index) {
    const unsigned int base = index * 3u;
    return make_float3(value[base], value[base + 1u], value[base + 2u]);
}

static __forceinline__ __device__ float3 load_soa_vec3(const float *x,
                                                       const float *y,
                                                       const float *z,
                                                       unsigned int index) {
    return make_float3(x[index], y[index], z[index]);
}

static __forceinline__ __device__ bool is_active(
    const SegmentVisibilityParams &params,
    unsigned int ray) {
    return params.active_mask == nullptr || params.active_mask[ray] != 0u;
}

static __forceinline__ __device__ float3 load_start(
    const SegmentVisibilityParams &params,
    unsigned int ray) {
    return params.start_aos != nullptr
        ? load_aos_vec3(params.start_aos, ray)
        : load_soa_vec3(params.start_x, params.start_y, params.start_z, ray);
}

static __forceinline__ __device__ float3 load_end_a(
    const SegmentVisibilityParams &params,
    unsigned int ray) {
    return params.end_aos != nullptr
        ? load_aos_vec3(params.end_aos, ray)
        : load_soa_vec3(params.end_x, params.end_y, params.end_z, ray);
}

static __forceinline__ __device__ float3 load_end_b(
    const SegmentVisibilityParams &params,
    unsigned int ray) {
    return params.end_b_aos != nullptr
        ? load_aos_vec3(params.end_b_aos, ray)
        : load_soa_vec3(params.end_b_x, params.end_b_y, params.end_b_z, ray);
}

static __forceinline__ __device__ float3 load_chain_point(
    const SegmentVisibilityParams &params,
    unsigned int chain,
    int point_index) {
    const int slot = static_cast<int>(chain) * params.max_points + point_index;
    return make_float3(params.chain_point_x[slot],
                       params.chain_point_y[slot],
                       params.chain_point_z[slot]);
}

template <typename Policy>
static __forceinline__ __device__ std::uint32_t trace_segment(
    const SegmentVisibilityParams &params,
    float3 start,
    float3 end,
    bool active,
    unsigned int ignore_base,
    std::uint32_t *blocker_prim) {
    if (!active || params.handle == 0ull) {
        if (blocker_prim != nullptr)
            *blocker_prim = 0xFFFFFFFFu;
        return 0u;
    }

    float3 direction = subtract(end, start);
    const float length = sqrtf(dot(direction, direction));
    if (length <= MinSegmentLength) {
        if (blocker_prim != nullptr)
            *blocker_prim = 0xFFFFFFFFu;
        return 1u;
    }

    direction = multiply(1.0f / length, direction);
    const float3 origin = add(start, multiply(RayBias, direction));
    const float tmax = fmaxf(length - 2.0f * RayBias, 0.0f);

    std::uint32_t visible = 1u;
    std::uint32_t blocker = 0xFFFFFFFFu;
    unsigned int ray_flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    if constexpr (Policy::disable_anyhit_without_ignore) {
        if (params.ignore_prim_ids == nullptr || params.ignore_k <= 0)
            ray_flags |= OPTIX_RAY_FLAG_DISABLE_ANYHIT;
    }

    optixTrace(static_cast<OptixTraversableHandle>(params.handle),
               origin,
               direction,
               TraceTMin,
               tmax,
               0.0f,
               255u,
               ray_flags,
               0,
               1,
               0,
               visible,
               blocker,
               ignore_base);
    if (blocker_prim != nullptr)
        *blocker_prim = blocker;
    return visible;
}

static __forceinline__ __device__ void anyhit(
    const SegmentVisibilityParams &params) {
    if (params.ignore_prim_ids == nullptr || params.ignore_k <= 0)
        return;

    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int global_prim = global_primitive_id(
        shape_id,
        static_cast<int>(optixGetPrimitiveIndex()),
        params.face_offsets,
        params.n_meshes);
    const unsigned int ignore_base = optixGetPayload_2();

    for (int slot = 0; slot < params.ignore_k; ++slot) {
        if (params.ignore_prim_ids[ignore_base + slot] == global_prim) {
            optixIgnoreIntersection();
            return;
        }
    }
}

static __forceinline__ __device__ void closesthit(
    const SegmentVisibilityParams &params) {
    optixSetPayload_0(0u);
    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int global_prim = global_primitive_id(
        shape_id,
        static_cast<int>(optixGetPrimitiveIndex()),
        params.face_offsets,
        params.n_meshes);
    optixSetPayload_1(static_cast<unsigned int>(global_prim));
}

static __forceinline__ __device__ void miss() {
    // Payload 0 is initialized to 1 by raygen and remains clear on miss.
}

template <typename Policy>
static __forceinline__ __device__ void raygen_segment(
    const SegmentVisibilityParams &params) {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays))
        return;

    std::uint32_t blocker = 0xFFFFFFFFu;
    const bool collect_blocker = params.out_first_blocked_prim != nullptr;
    const std::uint32_t visible = trace_segment<Policy>(
        params,
        load_start(params, ray),
        load_end_a(params, ray),
        is_active(params, ray),
        ray * params.ignore_k,
        collect_blocker ? &blocker : nullptr);
    params.out_visible[ray] = visible != 0u ? 1u : 0u;
    if (collect_blocker) {
        params.out_first_blocked_prim[ray] =
            visible == 0u && blocker != 0xFFFFFFFFu
                ? static_cast<int>(blocker)
                : -1;
    }
    if constexpr (Policy::write_output_t) {
        if (params.out_t != nullptr)
            params.out_t[ray] = __uint_as_float(0x7f800000u);
    }
}

template <typename Policy>
static __forceinline__ __device__ void raygen_segment_pair(
    const SegmentVisibilityParams &params) {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays))
        return;

    const bool active = is_active(params, ray);
    const float3 start = load_start(params, ray);
    const unsigned int ignore_base = ray * params.ignore_k;
    params.out_visible[ray] = trace_segment<Policy>(
        params, start, load_end_a(params, ray), active, ignore_base, nullptr) != 0u;
    params.out_visible_b[ray] = trace_segment<Policy>(
        params, start, load_end_b(params, ray), active, ignore_base, nullptr) != 0u;
}

template <typename Policy>
static __forceinline__ __device__ void raygen_axial_edge(
    const SegmentVisibilityParams &params) {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays))
        return;

    const bool active = is_active(params, ray);
    const float3 source = load_start(params, ray);
    const float3 edge_pos = load_end_a(params, ray);
    const float3 edge_dir = load_soa_vec3(
        params.edge_dir_x, params.edge_dir_y, params.edge_dir_z, ray);
    const float line_min = params.edge_t_min[ray];
    const float span = fmaxf(params.edge_t_max[ray] - line_min, 0.0f);
    std::uint32_t any_visible = 0u;

    #pragma unroll
    for (int i = 0; i < SegmentVisibilityMaxSamples; ++i) {
        if (i < params.sample_count) {
            const float t = line_min + params.sample_fractions[i] * span;
            const float3 sample = add(edge_pos, multiply(t, edge_dir));
            any_visible |= trace_segment<Policy>(
                params, source, sample, active, 0u, nullptr);
        }
    }
    params.out_visible[ray] = any_visible != 0u ? 1u : 0u;
}

template <typename Policy>
static __forceinline__ __device__ void raygen_segment_chain(
    const SegmentVisibilityParams &params) {
    const unsigned int chain = optixGetLaunchIndex().x;
    if (chain >= static_cast<unsigned int>(params.n_rays))
        return;

    if (!is_active(params, chain)) {
        params.out_visible[chain] = 0u;
        params.out_first_blocked_segment[chain] = -1;
        params.out_first_blocked_prim[chain] = -1;
        return;
    }

    int segment_count = params.chain_length != nullptr
        ? params.chain_length[chain]
        : params.max_segments;
    segment_count = segment_count < 0 ? 0 : segment_count;
    segment_count = segment_count > params.max_segments
        ? params.max_segments
        : segment_count;

    std::uint32_t all_visible = 1u;
    int first_blocked_segment = -1;
    int first_blocked_prim = -1;
    for (int segment = 0; segment < segment_count; ++segment) {
        const float3 start = load_chain_point(params, chain, segment);
        const float3 end = load_chain_point(params, chain, segment + 1);
        const unsigned int ignore_base = params.ignore_k > 0
            ? (chain * static_cast<unsigned int>(params.max_segments) +
               static_cast<unsigned int>(segment)) *
                  static_cast<unsigned int>(params.ignore_k)
            : 0u;

        std::uint32_t blocker_prim = 0xFFFFFFFFu;
        if (trace_segment<Policy>(
                params, start, end, true, ignore_base, &blocker_prim) == 0u) {
            all_visible = 0u;
            first_blocked_segment = segment;
            first_blocked_prim = static_cast<int>(blocker_prim);
            break;
        }
    }

    params.out_visible[chain] = all_visible != 0u ? 1u : 0u;
    params.out_first_blocked_segment[chain] = first_blocked_segment;
    params.out_first_blocked_prim[chain] = first_blocked_prim;
}

} // namespace segment_visibility
} // namespace rayd::shared::optix

#endif
