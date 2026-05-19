#include <optix.h>
#include <optix_device.h>

#include "segment_visibility_params.h"

namespace rayd {

extern "C" {
__constant__ SegmentVisibilityParams params;
}

namespace {

constexpr float kTraceTMin = 1e-5f;
constexpr float kRayBias = 1e-5f;
constexpr float kMinSegmentLength = 2e-5f;

static __forceinline__ __device__ float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float s, float3 v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ bool is_active(unsigned int ray) {
    return params.active_mask == nullptr || params.active_mask[ray] != 0u;
}

static __forceinline__ __device__ float3 load_start(unsigned int ray) {
    return make_vec3(params.start_x[ray], params.start_y[ray], params.start_z[ray]);
}

static __forceinline__ __device__ float3 load_end_a(unsigned int ray) {
    return make_vec3(params.end_x[ray], params.end_y[ray], params.end_z[ray]);
}

static __forceinline__ __device__ float3 load_end_b(unsigned int ray) {
    return make_vec3(params.end_b_x[ray], params.end_b_y[ray], params.end_b_z[ray]);
}

static __forceinline__ __device__ uint32_t trace_segment(float3 start,
                                                         float3 end,
                                                         bool active) {
    if (!active || params.handle == 0ull) {
        return 0u;
    }

    float3 direction = end - start;
    const float length_sq = dot3(direction, direction);
    const float length = sqrtf(length_sq);
    if (length <= kMinSegmentLength) {
        return 1u;
    }

    direction = (1.0f / length) * direction;
    const float3 origin = start + kRayBias * direction;
    const float tmax = fmaxf(length - 2.0f * kRayBias, 0.0f);

    uint32_t visible = 1u;
    optixTrace(static_cast<OptixTraversableHandle>(params.handle),
               origin,
               direction,
               kTraceTMin,
               tmax,
               0.0f,
               255u,
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0,
               1,
               0,
               visible);
    return visible;
}

} // namespace

extern "C" __global__ void __anyhit__segment_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int face_offset =
        (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
    const int global_prim = face_offset + static_cast<int>(optixGetPrimitiveIndex());

    for (int slot = 0; slot < params.ignore_k; ++slot) {
        const int ignored = params.ignore_prim_ids[ray * params.ignore_k + slot];
        if (ignored == global_prim) {
            optixIgnoreIntersection();
            return;
        }
    }
}

extern "C" __global__ void __closesthit__segment_visibility() {
    optixSetPayload_0(0u);
}

extern "C" __global__ void __miss__segment_visibility() {
    // Payload 0 is initialized to 1 by raygen and remains clear on miss.
}

extern "C" __global__ void __raygen__segment_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    params.out_visible[ray] =
        trace_segment(load_start(ray), load_end_a(ray), is_active(ray)) != 0u ? 1u : 0u;
}

extern "C" __global__ void __raygen__segment_pair_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const bool active = is_active(ray);
    const float3 start = load_start(ray);
    params.out_visible[ray] = trace_segment(start, load_end_a(ray), active) != 0u ? 1u : 0u;
    params.out_visible_b[ray] = trace_segment(start, load_end_b(ray), active) != 0u ? 1u : 0u;
}

extern "C" __global__ void __raygen__axial_edge_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const bool active = is_active(ray);
    const float3 source = load_start(ray);
    const float3 edge_pos = load_end_a(ray);
    const float3 edge_dir = make_vec3(params.edge_dir_x[ray],
                                      params.edge_dir_y[ray],
                                      params.edge_dir_z[ray]);
    const float line_min = params.edge_line_min[ray];
    const float line_max = params.edge_line_max[ray];
    const float span = fmaxf(line_max - line_min, 0.0f);
    uint32_t any_visible = 0u;

    #pragma unroll
    for (int i = 0; i < SegmentVisibilityMaxSamples; ++i) {
        if (i < params.sample_count) {
            const float t = line_min + params.sample_fractions[i] * span;
            const float3 sample = edge_pos + t * edge_dir;
            any_visible |= trace_segment(source, sample, active);
        }
    }

    params.out_visible[ray] = any_visible != 0u ? 1u : 0u;
}

} // namespace rayd
