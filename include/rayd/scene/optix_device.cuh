// Copyright Xingyu Chen.
// Defines shared scene support for optix device.

#pragma once

#include <optix_device.h>

#include <rayd/scene/optix_contracts.h>

namespace rayd::shared::optix {

static __forceinline__ __device__ void set_scene_intersection_payload(float ray_t, unsigned int shape_id,
                                                                      float barycentric_u, float barycentric_v,
                                                                      unsigned int local_primitive_id) {
    optixSetPayload_0(__float_as_uint(ray_t));
    optixSetPayload_1(shape_id);
    optixSetPayload_2(__float_as_uint(barycentric_u));
    optixSetPayload_3(__float_as_uint(barycentric_v));
    optixSetPayload_4(local_primitive_id);
}

} // namespace rayd::shared::optix