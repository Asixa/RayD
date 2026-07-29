// Copyright Xingyu Chen.
// Defines shared reflection support for accumulation optix device.

#pragma once

#include <optix.h>
#include <optix_device.h>

#include <rayd/reflection/accumulation_algo.h>
#include <rayd/reflection/optix_hit.h>
#include <rayd/reflection/optix_traverser.h>

// OptiX entry layer for reflection accumulation. The algorithm body now lives in
// the host-compilable rayd/reflection/accumulation_algo.h; this
// header keeps only the OptiX-specific pieces: the closesthit / miss programs,
// the six-register payload codec (shared/reflection/optix_hit.h), and the raygen
// entry that instantiates the shared OptixTraverser (the sole home of the one
// closest-hit optixTrace) and dispatches to the algorithm. Reflection
// accumulation shares the reflection-trace six-register TriangleHitPayload, so it
// reuses the shared OptixTraverser directly. The backend .cu adapters keep their
// closest_hit / miss / raygen<Params, Policy> entry names and signatures.

namespace rayd::shared::multipath::reflection_accumulation {

static __forceinline__ __device__ void closest_hit() {
    ::rayd::shared::optix::TriangleHitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    const float2 barycentrics = optixGetTriangleBarycentrics();
    payload.bary_u = __float_as_uint(barycentrics.x);
    payload.bary_v = __float_as_uint(barycentrics.y);
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    ::rayd::shared::optix::set_triangle_hit_payload(payload);
}

static __forceinline__ __device__ void miss() {
    optixSetPayload_0(0u);
}

/// Single-handle OptiX traverser for the reflection accumulation closest cast:
/// no anyhit, SBT offset 0 / stride 1, miss program 0, and the family's TraceTMax
/// miss sentinel (a null handle yields the same cleared hit).
static __forceinline__ __device__ ::rayd::shared::optix::OptixTraverser make_traverser(
    ::OptixTraversableHandle handle) {
    return ::rayd::shared::optix::OptixTraverser{
        handle,
        static_cast<unsigned int>(OPTIX_RAY_FLAG_DISABLE_ANYHIT),
        0u,
        1u,
        0u,
        reflection_accumulation_algo_detail::TraceTMax};
}

template <typename Params, typename Policy>
static __forceinline__ __device__ void raygen(const Params &params) {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    const ::rayd::shared::optix::OptixTraverser primary = make_traverser(params.primary_handle);
    const ::rayd::shared::optix::OptixTraverser secondary = make_traverser(params.secondary_handle);
    ::rayd::shared::multipath::reflection_accumulation_algo<
        Params, Policy, ::rayd::shared::optix::OptixTraverser>(
        params, ray_index, primary, secondary);
}

} // namespace rayd::shared::multipath::reflection_accumulation
