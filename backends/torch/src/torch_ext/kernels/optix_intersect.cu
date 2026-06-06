#include <raydtorch/optix_intersect_params.h>

#include <optix_device.h>

extern "C" {
__constant__ raydtorch::OptixIntersectParams params;
}

extern "C" __global__ void __raygen__intersect() {
    const unsigned int ray_idx = optixGetLaunchIndex().x;
    if (ray_idx >= static_cast<unsigned int>(params.ray_count))
        return;

    float t = __uint_as_float(0x7f800000u);
    int prim_id = -1;
    float u = 0.f;
    float v = 0.f;

    if (params.active[ray_idx]) {
        const float3 origin = make_float3(
            params.ray_o[ray_idx * 3 + 0],
            params.ray_o[ray_idx * 3 + 1],
            params.ray_o[ray_idx * 3 + 2]);
        const float3 direction = make_float3(
            params.ray_d[ray_idx * 3 + 0],
            params.ray_d[ray_idx * 3 + 1],
            params.ray_d[ray_idx * 3 + 2]);
        unsigned int p0 = __float_as_uint(t);
        unsigned int p1 = static_cast<unsigned int>(prim_id);
        unsigned int p2 = __float_as_uint(u);
        unsigned int p3 = __float_as_uint(v);
        optixTrace(
            params.traversable,
            origin,
            direction,
            1e-6f,
            params.ray_tmax[ray_idx],
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            p0,
            p1,
            p2,
            p3);
        t = __uint_as_float(p0);
        prim_id = static_cast<int>(p1);
        u = __uint_as_float(p2);
        v = __uint_as_float(p3);
    }

    params.out_t[ray_idx] = t;
    params.out_prim_id[ray_idx] = prim_id;
    params.out_bary_uv[ray_idx * 2 + 0] = u;
    params.out_bary_uv[ray_idx * 2 + 1] = v;
}

extern "C" __global__ void __miss__intersect() {
}

extern "C" __global__ void __closesthit__intersect() {
    const float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(static_cast<unsigned int>(optixGetPrimitiveIndex()));
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
}
