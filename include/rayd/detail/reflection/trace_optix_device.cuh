#pragma once

#include <optix.h>
#include <optix_device.h>

#include <rayd/detail/reflection/trace_algo.h>
#include <rayd/detail/reflection/optix_hit.h>
#include <rayd/detail/reflection/optix_traverser.h>
#include <rayd/detail/reflection/trace_params.h>
#include <rayd/detail/rt/traverser.h>

// OptiX entry layer for reflection trace. The algorithm body now lives in the
// host-compilable rayd/detail/reflection/trace_algo.h; this header
// keeps only the OptiX-specific pieces: the raygen/closesthit/miss program
// entries, the six-register payload codec (shared/reflection/optix_hit.h), the
// OptixTraverser instantiation, and the ReflectionTracePolicy layout policies
// that become the Layout axis of TraceConfig.

namespace rayd::shared::optix {

/// Compile-time adapter for backend-specific reflection trace storage conventions.
/// This is the Layout axis of rt::TraceConfig (see reflection_trace_algo.h).
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

/// Build the single-handle OptiX traverser used by the reflection trace: no
/// anyhit, SBT offset 0 / stride 1, miss program 0, and the family's
/// kTraceTMax miss sentinel (a null handle yields the same cleared hit).
static __forceinline__ __device__ OptixTraverser make_reflection_traverser(
    ::OptixTraversableHandle handle) {
    return OptixTraverser{
        handle,
        static_cast<unsigned int>(OPTIX_RAY_FLAG_DISABLE_ANYHIT),
        0u,
        1u,
        0u,
        ::rayd::shared::multipath::reflection_trace_algo_detail::kTraceTMax};
}

template <typename Policy>
static __forceinline__ __device__ void reflection_trace_raygen(
    const ReflectionTraceParams &params) {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    const OptixTraverser primary = make_reflection_traverser(params.primary_handle);
    const OptixTraverser secondary = make_reflection_traverser(params.secondary_handle);
    using Config = ::rayd::shared::rt::TraceConfig<Policy, OptixTraverser>;
    ::rayd::shared::multipath::reflection_trace_algo<Config>(params, ray_index, primary, secondary);
}

} // namespace rayd::shared::optix
