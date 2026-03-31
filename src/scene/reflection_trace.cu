#include <optix.h>
#include <optix_device.h>

#include "reflection_trace_params.h"

namespace rayd {

namespace {

constexpr float kTraceTMin = 1e-5f;
constexpr float kTraceTMax = 1e8f;
constexpr float kRayBias = 1e-5f;

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int bary_u = 0u;
    unsigned int bary_v = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

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

static __forceinline__ __device__ float3 normalize3(float3 v) {
    const float inv_len = rsqrtf(fmaxf(dot3(v, v), 1e-12f));
    return inv_len * v;
}

static __forceinline__ __device__ void clear_payload(HitPayload &payload) {
    payload.hit = 0u;
    payload.t = __float_as_uint(kTraceTMax);
    payload.bary_u = 0u;
    payload.bary_v = 0u;
    payload.prim = 0u;
    payload.instance = 0u;
}

static __forceinline__ __device__ void set_payload(const HitPayload &payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.bary_u);
    optixSetPayload_3(payload.bary_v);
    optixSetPayload_4(payload.prim);
    optixSetPayload_5(payload.instance);
}

static __forceinline__ __device__ void trace_handle(OptixTraversableHandle handle,
                                                    float3 origin,
                                                    float3 direction,
                                                    float tmax,
                                                    HitPayload &payload) {
    clear_payload(payload);
    if (handle == 0ull) {
        return;
    }

    optixTrace(handle,
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

static __forceinline__ __device__ HitPayload choose_hit(const HitPayload &a,
                                                        const HitPayload &b) {
    if (a.hit == 0u) {
        return b;
    }
    if (b.hit == 0u) {
        return a;
    }
    return __uint_as_float(b.t) < __uint_as_float(a.t) ? b : a;
}

} // namespace

extern "C" {
__constant__ ReflectionTraceParams params;
}

extern "C" __global__ void __closesthit__reflection() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    const float2 bary = optixGetTriangleBarycentrics();
    payload.bary_u = __float_as_uint(bary.x);
    payload.bary_v = __float_as_uint(bary.y);
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

extern "C" __global__ void __miss__reflection() {
    optixSetPayload_0(0u);
}

extern "C" __global__ void __raygen__reflection_trace() {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    if (ray_index >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        params.out_bounce_count[ray_index] = 0;
        return;
    }

    const int B = params.max_bounces;
    const int base = static_cast<int>(ray_index) * B;

    float3 origin = make_vec3(params.ray_ox[ray_index],
                              params.ray_oy[ray_index],
                              params.ray_oz[ray_index]);
    float3 direction = make_vec3(params.ray_dx[ray_index],
                                 params.ray_dy[ray_index],
                                 params.ray_dz[ray_index]);
    float3 image_source = origin;
    int bounce_count = 0;

    for (int bounce = 0; bounce < B; ++bounce) {
        const float tmax_input = bounce == 0 ? params.ray_tmax[ray_index] : kTraceTMax;
        const float trace_tmax = isfinite(tmax_input) ? tmax_input : kTraceTMax;

        HitPayload hit_primary;
        trace_handle(params.primary_handle, origin, direction, trace_tmax, hit_primary);

        HitPayload hit = hit_primary;
        if (params.split_mode != 0) {
            HitPayload hit_secondary;
            trace_handle(params.secondary_handle, origin, direction, trace_tmax, hit_secondary);
            hit = choose_hit(hit_primary, hit_secondary);
        }

        if (hit.hit == 0u) {
            break;
        }

        const int shape_id = static_cast<int>(hit.instance);
        const int local_prim = static_cast<int>(hit.prim);
        const int face_offset =
            (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
        const int global_prim = face_offset + local_prim;

        const float t = __uint_as_float(hit.t);
        const float bary_u = __uint_as_float(hit.bary_u);
        const float bary_v = __uint_as_float(hit.bary_v);

        float3 hit_point = origin + t * direction;
        float3 geo_normal = make_vec3(0.0f, 0.0f, 1.0f);

        if (global_prim >= 0 && global_prim < params.n_triangles) {
            const float p0x = params.tri_p0_x[global_prim];
            const float p0y = params.tri_p0_y[global_prim];
            const float p0z = params.tri_p0_z[global_prim];
            const float e1x = params.tri_e1_x[global_prim];
            const float e1y = params.tri_e1_y[global_prim];
            const float e1z = params.tri_e1_z[global_prim];
            const float e2x = params.tri_e2_x[global_prim];
            const float e2y = params.tri_e2_y[global_prim];
            const float e2z = params.tri_e2_z[global_prim];

            hit_point = make_vec3(p0x + bary_u * e1x + bary_v * e2x,
                                  p0y + bary_u * e1y + bary_v * e2y,
                                  p0z + bary_u * e1z + bary_v * e2z);
            geo_normal = normalize3(make_vec3(params.tri_fn_x[global_prim],
                                              params.tri_fn_y[global_prim],
                                              params.tri_fn_z[global_prim]));
        }

        if (dot3(direction, geo_normal) > 0.0f) {
            geo_normal = -1.0f * geo_normal;
        }

        const float image_distance = dot3(image_source - hit_point, geo_normal);
        image_source = image_source - 2.0f * image_distance * geo_normal;

        const int slot = base + bounce;
        params.out_shape_ids[slot] = shape_id;
        params.out_prim_ids[slot] = local_prim;
        params.out_t[slot] = t;
        params.out_bary_u[slot] = bary_u;
        params.out_bary_v[slot] = bary_v;
        params.out_hit_x[slot] = hit_point.x;
        params.out_hit_y[slot] = hit_point.y;
        params.out_hit_z[slot] = hit_point.z;
        params.out_norm_x[slot] = geo_normal.x;
        params.out_norm_y[slot] = geo_normal.y;
        params.out_norm_z[slot] = geo_normal.z;
        params.out_img_x[slot] = image_source.x;
        params.out_img_y[slot] = image_source.y;
        params.out_img_z[slot] = image_source.z;

        const float dot_dn = dot3(direction, geo_normal);
        direction = direction - 2.0f * dot_dn * geo_normal;
        origin = hit_point + kRayBias * direction;
        bounce_count = bounce + 1;
    }

    params.out_bounce_count[ray_index] = bounce_count;
}

} // namespace rayd
