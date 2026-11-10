// Copyright Xingyu Chen.
// Implements penetration support for penetration optix.

#include <optix.h>
#include <optix_device.h>

#include <cuda_runtime.h>

#include <src/penetration/segment_penetration_march.cuh>

namespace rayd::torch_backend {

extern "C" {
__constant__ SegmentPenetrationParams params;
}

extern "C" __global__ void __closesthit__segment_penetration() {
    const float2 barycentric = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(static_cast<unsigned int>(optixGetInstanceId()));
    optixSetPayload_2(__float_as_uint(barycentric.x));
    optixSetPayload_3(__float_as_uint(barycentric.y));
    optixSetPayload_4(static_cast<unsigned int>(optixGetPrimitiveIndex()));
}

extern "C" __global__ void __miss__segment_penetration() {}

extern "C" __global__ void __raygen__segment_penetration() {
    segment_penetration::march(params, optixGetLaunchIndex().x,
                               [](float3 origin, float3 direction, float tmin, float tmax) {
                                   unsigned int payload_t = 0x7f800000u;
                                   unsigned int payload_shape = 0xFFFFFFFFu;
                                   unsigned int payload_u = 0u;
                                   unsigned int payload_v = 0u;
                                   unsigned int payload_local_primitive = 0xFFFFFFFFu;
                                   optixTrace(params.traversable, origin, direction, tmin, tmax, 0.0f,
                                              OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
                                              payload_t, payload_shape, payload_u, payload_v, payload_local_primitive);
                                   return segment_penetration::ProbeHit{__uint_as_float(payload_t),
                                                                        static_cast<int>(payload_shape),
                                                                        __uint_as_float(payload_u),
                                                                        __uint_as_float(payload_v),
                                                                        static_cast<int>(payload_local_primitive)};
                               });
}

} // namespace rayd::torch_backend
