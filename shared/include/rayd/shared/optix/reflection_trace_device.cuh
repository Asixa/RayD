#pragma once

#include <optix.h>
#include <optix_device.h>

#include <rayd/shared/optix/device_hit.h>
#include <rayd/shared/optix/reflection_trace_params.h>
#include <rayd/shared/reflection/reflection_geometry.h>
#include <rayd/shared/rt/numeric_policy.h>

namespace rayd::shared::optix {

/// Compile-time adapter for backend-specific reflection trace storage conventions.
template <bool AllowAoSInputs,
          bool AllowPackedTriangles,
          bool HonorOutputLayout,
          bool ClearEmptySlots,
          bool NullableRayTMax,
          bool AllowExtendedOutputs>
struct ReflectionTracePolicy {
    static constexpr bool allow_aos_inputs = AllowAoSInputs;
    static constexpr bool allow_packed_triangles = AllowPackedTriangles;
    static constexpr bool honor_output_layout = HonorOutputLayout;
    static constexpr bool clear_empty_slots = ClearEmptySlots;
    static constexpr bool nullable_ray_tmax = NullableRayTMax;
    static constexpr bool allow_extended_outputs = AllowExtendedOutputs;
};

using DrJitReflectionTracePolicy =
    ReflectionTracePolicy<false, false, false, false, false, false>;
using TorchReflectionTracePolicy =
    ReflectionTracePolicy<true, true, true, true, true, true>;

namespace reflection_trace_detail {

constexpr float kTraceTMin = 1e-5f;
constexpr float kTraceTMax = 1e8f;
constexpr float kRayBias = 1e-5f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kTraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
// This family clears missed slots to kTraceTMax rather than +inf.
static_assert(kTraceTMax == ::rayd::shared::rt::kReflectionTraceMissDistance);

using HitPayload = TriangleHitPayload;

struct TriangleData {
    float3 p0;
    float3 e1;
    float3 e2;
    float3 fn;
};

static __forceinline__ __device__ float3 vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 vec3(const float *values) {
    return vec3(values[0], values[1], values[2]);
}

static __forceinline__ __device__ float3 vec3(float4 value) {
    return vec3(value.x, value.y, value.z);
}

static __forceinline__ __device__ float3 add(float3 a, float3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 scale(float value, float3 v) {
    return vec3(value * v.x, value * v.y, value * v.z);
}

static __forceinline__ __device__ float3 madd(float3 base, float value, float3 v) {
    return add(base, scale(value, v));
}

static __forceinline__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 normalize(float3 value) {
    return scale(rsqrtf(fmaxf(dot(value, value), 1e-12f)), value);
}

static __forceinline__ __device__ math::Vec3f to_shared(float3 value) {
    return math::make_vec3(value.x, value.y, value.z);
}

static __forceinline__ __device__ float3 from_shared(math::Vec3f value) {
    return vec3(value.x, value.y, value.z);
}

static __forceinline__ __device__ void trace_handle(
    ::OptixTraversableHandle handle,
    float3 origin,
    float3 direction,
    float tmax,
    HitPayload &payload) {
    clear_triangle_hit(payload, kTraceTMax);
    if (handle == 0ull)
        return;

    optixTrace(
        handle,
        origin,
        direction,
        kTraceTMin,
        tmax,
        0.0f,
        255u,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        payload.hit,
        payload.t,
        payload.bary_u,
        payload.bary_v,
        payload.prim,
        payload.instance);
}

template <typename Policy>
static __forceinline__ __device__ int output_slot(
    const ReflectionTraceParams &params,
    unsigned int ray_index,
    int bounce) {
    if constexpr (Policy::honor_output_layout) {
        if (params.output_layout != 0)
            return bounce * params.n_rays + static_cast<int>(ray_index);
    }
    return static_cast<int>(ray_index) * params.max_bounces + bounce;
}

template <typename Policy>
static __forceinline__ __device__ void clear_output_slot(
    const ReflectionTraceParams &params,
    unsigned int ray_index,
    int bounce) {
    if constexpr (!Policy::clear_empty_slots) {
        return;
    } else {
        const int slot = output_slot<Policy>(params, ray_index, bounce);
        if (params.out_t != nullptr)
            params.out_t[slot] = kTraceTMax;
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = -1;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = -1;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = -1;
        if (params.out_valid != nullptr)
            params.out_valid[ray_index * params.max_bounces + bounce] = 0u;
        if (params.out_bary != nullptr) {
            params.out_bary[slot * 3 + 0] = 0.0f;
            params.out_bary[slot * 3 + 1] = 0.0f;
            params.out_bary[slot * 3 + 2] = 0.0f;
        }
        if (params.out_hit != nullptr) {
            params.out_hit[slot * 3 + 0] = 0.0f;
            params.out_hit[slot * 3 + 1] = 0.0f;
            params.out_hit[slot * 3 + 2] = 0.0f;
        }
        if (params.out_norm != nullptr) {
            params.out_norm[slot * 3 + 0] = 0.0f;
            params.out_norm[slot * 3 + 1] = 0.0f;
            params.out_norm[slot * 3 + 2] = 0.0f;
        }
        if (params.out_img != nullptr) {
            params.out_img[slot * 3 + 0] = 0.0f;
            params.out_img[slot * 3 + 1] = 0.0f;
            params.out_img[slot * 3 + 2] = 0.0f;
        }
    }
}

template <typename Policy>
static __forceinline__ __device__ TriangleData load_triangle(
    const ReflectionTraceParams &params,
    int prim) {
    if constexpr (Policy::allow_packed_triangles) {
        if (params.tri_p0_packed != nullptr &&
            params.tri_e1_packed != nullptr &&
            params.tri_e2_packed != nullptr &&
            params.tri_fn_packed != nullptr) {
            return {
                vec3(params.tri_p0_packed[prim]),
                vec3(params.tri_e1_packed[prim]),
                vec3(params.tri_e2_packed[prim]),
                vec3(params.tri_fn_packed[prim])};
        }
    }

    return {
        vec3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]),
        vec3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]),
        vec3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]),
        vec3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim])};
}

template <typename Policy>
static __forceinline__ __device__ void load_ray(
    const ReflectionTraceParams &params,
    unsigned int ray_index,
    float3 &origin,
    float3 &direction) {
    if constexpr (Policy::allow_aos_inputs) {
        if (params.ray_o_aos != nullptr) {
            origin = vec3(params.ray_o_aos + ray_index * 3);
            direction = vec3(params.ray_d_aos + ray_index * 3);
            return;
        }
    }

    origin = vec3(params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    direction = vec3(params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]);
}

template <typename Policy>
static __forceinline__ __device__ float first_trace_tmax(
    const ReflectionTraceParams &params,
    unsigned int ray_index) {
    if constexpr (Policy::nullable_ray_tmax) {
        return params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kTraceTMax;
    }
    return params.ray_tmax[ray_index];
}

} // namespace reflection_trace_detail

static __forceinline__ __device__ void reflection_trace_closest_hit() {
    TriangleHitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    const float2 bary = optixGetTriangleBarycentrics();
    payload.bary_u = __float_as_uint(bary.x);
    payload.bary_v = __float_as_uint(bary.y);
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_triangle_hit_payload(payload);
}

static __forceinline__ __device__ void reflection_trace_miss() {
    optixSetPayload_0(0u);
}

template <typename Policy>
static __forceinline__ __device__ void reflection_trace_raygen(
    const ReflectionTraceParams &params) {
    using namespace reflection_trace_detail;

    const unsigned int ray_index = optixGetLaunchIndex().x;
    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        if constexpr (Policy::clear_empty_slots) {
            for (int bounce = 0; bounce < params.max_bounces; ++bounce)
                clear_output_slot<Policy>(params, ray_index, bounce);
        }
        if (params.out_bounce_count != nullptr)
            params.out_bounce_count[ray_index] = 0;
        return;
    }

    const int bounce_limit = params.max_bounces;
    float3 origin;
    float3 direction;
    load_ray<Policy>(params, ray_index, origin, direction);
    float3 image_source = origin;
    int bounce_count = 0;

    for (int bounce = 0; bounce < bounce_limit; ++bounce) {
        const float tmax_input = bounce == 0
            ? first_trace_tmax<Policy>(params, ray_index)
            : kTraceTMax;
        const float trace_tmax = isfinite(tmax_input) ? tmax_input : kTraceTMax;

        HitPayload primary;
        trace_handle(params.primary_handle, origin, direction, trace_tmax, primary);
        HitPayload hit = primary;
        if (params.split_mode != 0) {
            HitPayload secondary;
            trace_handle(params.secondary_handle, origin, direction, trace_tmax, secondary);
            hit = choose_nearest_hit(primary, secondary);
        }
        if (hit.hit == 0u)
            break;

        const int shape_id = static_cast<int>(hit.instance);
        const int local_prim = static_cast<int>(hit.prim);
        const int face_offset =
            shape_id >= 0 && shape_id < params.n_meshes ? params.face_offsets[shape_id] : 0;
        const int global_prim = face_offset + local_prim;
        const float t = __uint_as_float(hit.t);
        const float bary_u = __uint_as_float(hit.bary_u);
        const float bary_v = __uint_as_float(hit.bary_v);

        float3 hit_point = madd(origin, t, direction);
        float3 geo_normal = vec3(0.0f, 0.0f, 1.0f);
        if (global_prim >= 0 && global_prim < params.n_triangles) {
            const TriangleData tri = load_triangle<Policy>(params, global_prim);
            hit_point = madd(madd(tri.p0, bary_u, tri.e1), bary_v, tri.e2);
            geo_normal = normalize(tri.fn);
        }
        geo_normal = from_shared(reflection::orient_normal_against(
            to_shared(direction), to_shared(geo_normal)));

        const bool write_image_source =
            (Policy::allow_extended_outputs && params.out_img != nullptr) ||
            (params.out_img_x != nullptr && params.out_img_y != nullptr &&
             params.out_img_z != nullptr);
        if (write_image_source) {
            image_source = from_shared(reflection::reflect_point_across_plane(
                to_shared(image_source), to_shared(hit_point), to_shared(geo_normal)));
        }

        const int slot = output_slot<Policy>(params, ray_index, bounce);
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_valid != nullptr)
                params.out_valid[ray_index * params.max_bounces + bounce] = 1u;
        }
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = shape_id;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = local_prim;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = global_prim;
        if (params.out_t != nullptr)
            params.out_t[slot] = t;
        if (params.out_bary_u != nullptr)
            params.out_bary_u[slot] = bary_u;
        if (params.out_bary_v != nullptr)
            params.out_bary_v[slot] = bary_v;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_bary != nullptr) {
                params.out_bary[slot * 3 + 0] = 1.0f - bary_u - bary_v;
                params.out_bary[slot * 3 + 1] = bary_u;
                params.out_bary[slot * 3 + 2] = bary_v;
            }
        }
        if (params.out_hit_x != nullptr)
            params.out_hit_x[slot] = hit_point.x;
        if (params.out_hit_y != nullptr)
            params.out_hit_y[slot] = hit_point.y;
        if (params.out_hit_z != nullptr)
            params.out_hit_z[slot] = hit_point.z;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_hit != nullptr) {
                params.out_hit[slot * 3 + 0] = hit_point.x;
                params.out_hit[slot * 3 + 1] = hit_point.y;
                params.out_hit[slot * 3 + 2] = hit_point.z;
            }
        }
        if (params.out_norm_x != nullptr)
            params.out_norm_x[slot] = geo_normal.x;
        if (params.out_norm_y != nullptr)
            params.out_norm_y[slot] = geo_normal.y;
        if (params.out_norm_z != nullptr)
            params.out_norm_z[slot] = geo_normal.z;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_norm != nullptr) {
                params.out_norm[slot * 3 + 0] = geo_normal.x;
                params.out_norm[slot * 3 + 1] = geo_normal.y;
                params.out_norm[slot * 3 + 2] = geo_normal.z;
            }
        }
        if (write_image_source) {
            if constexpr (Policy::allow_extended_outputs) {
                if (params.out_img != nullptr) {
                    params.out_img[slot * 3 + 0] = image_source.x;
                    params.out_img[slot * 3 + 1] = image_source.y;
                    params.out_img[slot * 3 + 2] = image_source.z;
                } else {
                    params.out_img_x[slot] = image_source.x;
                    params.out_img_y[slot] = image_source.y;
                    params.out_img_z[slot] = image_source.z;
                }
            } else {
                params.out_img_x[slot] = image_source.x;
                params.out_img_y[slot] = image_source.y;
                params.out_img_z[slot] = image_source.z;
            }
        }

        direction = from_shared(reflection::reflect_direction(
            to_shared(direction), to_shared(geo_normal)));
        origin = madd(hit_point, kRayBias, direction);
        bounce_count = bounce + 1;
    }

    if constexpr (Policy::clear_empty_slots) {
        for (int bounce = bounce_count; bounce < bounce_limit; ++bounce)
            clear_output_slot<Policy>(params, ray_index, bounce);
    }

    if (bounce_count > 0 && params.return_trailing != 0) {
        if (params.out_trailing_dir_x != nullptr)
            params.out_trailing_dir_x[ray_index] = direction.x;
        if (params.out_trailing_dir_y != nullptr)
            params.out_trailing_dir_y[ray_index] = direction.y;
        if (params.out_trailing_dir_z != nullptr)
            params.out_trailing_dir_z[ray_index] = direction.z;
        if (params.out_trailing_origin_x != nullptr)
            params.out_trailing_origin_x[ray_index] = origin.x;
        if (params.out_trailing_origin_y != nullptr)
            params.out_trailing_origin_y[ray_index] = origin.y;
        if (params.out_trailing_origin_z != nullptr)
            params.out_trailing_origin_z[ray_index] = origin.z;

        HitPayload primary;
        trace_handle(params.primary_handle, origin, direction, kTraceTMax, primary);
        HitPayload trailing = primary;
        if (params.split_mode != 0) {
            HitPayload secondary;
            trace_handle(params.secondary_handle, origin, direction, kTraceTMax, secondary);
            trailing = choose_nearest_hit(primary, secondary);
        }
        if (trailing.hit != 0u) {
            const int shape_id = static_cast<int>(trailing.instance);
            const int local_prim = static_cast<int>(trailing.prim);
            const int face_offset = shape_id >= 0 && shape_id < params.n_meshes
                ? params.face_offsets[shape_id]
                : 0;
            if (params.out_trailing_t != nullptr)
                params.out_trailing_t[ray_index] = __uint_as_float(trailing.t);
            if (params.out_trailing_prim != nullptr)
                params.out_trailing_prim[ray_index] = face_offset + local_prim;
        }
    }

    if (params.out_bounce_count != nullptr)
        params.out_bounce_count[ray_index] = bounce_count;
}

} // namespace rayd::shared::optix
