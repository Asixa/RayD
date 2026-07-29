#include <src/scene/optix_intersect_params.h>
#include <rayd/detail/contracts.h>
#include <rayd/detail/scene/optix_device.cuh>

#include <optix_device.h>

extern "C" {
__constant__ rayd::torch_backend::OptixIntersectParams params;
}

extern "C" __global__ void __raygen__intersect() {
    const unsigned int ray_idx = optixGetLaunchIndex().x;
    if (ray_idx >= static_cast<unsigned int>(params.ray_count))
        return;

    float t = __uint_as_float(0x7f800000u);
    int shape_id = rayd::shared::InvalidSignedId;
    int local_prim_id = rayd::shared::InvalidSignedId;
    int global_prim_id = rayd::shared::InvalidSignedId;
    float u = 0.f;
    float v = 0.f;

    if (params.active == nullptr || params.active[ray_idx]) {
        const float3 origin = make_float3(
            params.ray_o[ray_idx * 3 + 0],
            params.ray_o[ray_idx * 3 + 1],
            params.ray_o[ray_idx * 3 + 2]);
        const float3 direction = make_float3(
            params.ray_d[ray_idx * 3 + 0],
            params.ray_d[ray_idx * 3 + 1],
            params.ray_d[ray_idx * 3 + 2]);
        rayd::shared::optix::SceneIntersectionPayload payload {
            __float_as_uint(t),
            static_cast<unsigned int>(shape_id),
            __float_as_uint(u),
            __float_as_uint(v),
            static_cast<unsigned int>(local_prim_id),
        };
        const float trace_tmax =
            params.ray_tmax != nullptr ? params.ray_tmax[ray_idx] : __uint_as_float(0x7f7fffffu);
        optixTrace(
            params.traversable,
            origin,
            direction,
            rayd::shared::SmallEpsilon,
            trace_tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            payload.ray_t,
            payload.shape_id,
            payload.barycentric_u,
            payload.barycentric_v,
            payload.local_primitive_id);
        t = __uint_as_float(payload.ray_t);
        shape_id = static_cast<int>(payload.shape_id);
        u = __uint_as_float(payload.barycentric_u);
        v = __uint_as_float(payload.barycentric_v);
        local_prim_id = static_cast<int>(payload.local_primitive_id);
        if (shape_id >= 0 && shape_id < params.mesh_count && local_prim_id >= 0) {
            global_prim_id = params.face_offsets[shape_id] + local_prim_id;
        }
    }

    if (params.out_t != nullptr)
        params.out_t[ray_idx] = t;
    if (params.out_shape_id != nullptr)
        params.out_shape_id[ray_idx] = shape_id;
    if (params.out_local_prim_id != nullptr)
        params.out_local_prim_id[ray_idx] = local_prim_id;
    if (params.out_global_prim_id != nullptr)
        params.out_global_prim_id[ray_idx] = global_prim_id;
    if (params.out_bary_uv != nullptr) {
        params.out_bary_uv[ray_idx * 2 + 0] = u;
        params.out_bary_uv[ray_idx * 2 + 1] = v;
    }
}

extern "C" __global__ void __miss__intersect() {
}

extern "C" __global__ void __closesthit__intersect() {
    const float2 bary = optixGetTriangleBarycentrics();
    rayd::shared::optix::set_scene_intersection_payload(
        optixGetRayTmax(),
        optixGetInstanceId(),
        bary.x,
        bary.y,
        static_cast<unsigned int>(optixGetPrimitiveIndex()));
}
