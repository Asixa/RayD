// Copyright Xingyu Chen.
// Defines shared reflection support for epc optix device.

#pragma once

#include <cstdint>

#include <optix.h>
#include <optix_device.h>

#include <rayd/reflection/epc_algo.h>
#include <rayd/reflection/optix_hit.h>
#include <rayd/reflection/epc_params.h>
#include <rayd/rt/traverser.h>

// OptiX entry layer for reflection EPC. The algorithm body now lives in the
// host-compilable rayd/reflection/epc_algo.h; this header keeps
// only the OptiX-specific pieces: the mode-switched anyhit / closesthit / miss
// programs (one SBT serves both the reflector trace and the visibility check,
// selected by payload 5), the ReflEpcOptixTraverser (the sole home of the two
// optixTrace sites), and the raygen entry that instantiates the traversers and
// dispatches to the algorithm.

namespace rayd::shared::optix {

extern "C" {
extern __constant__ ReflEpcParams params;
}

namespace reflection_epc_device {

namespace algo_detail = ::rayd::shared::multipath::reflection_epc_algo_detail;

constexpr unsigned int kInvalidPrim = rayd::shared::InvalidUnsignedId;
constexpr unsigned int kTraceModeReflection = 0u;
constexpr unsigned int kTraceModeVisibility = 1u;

/// Single-handle OptiX traverser for the EPC pipeline. The same SBT serves both
/// ray families, switched by payload 5 (the trace mode): trace_closest runs the
/// reflector scene trace (reflection mode, closest hit, no anyhit) and reports the
/// mesh-local prim / shape; trace_first_blocker runs the segment visibility check
/// (visibility mode, terminate-on-first-hit, the anyhit ignore filter) and reports
/// the blocker's already-global prim with shape -1. `DisableAnyHitWithoutIgnore` is
/// the compile-time layout choice of whether a visibility ray with an empty ignore
/// list skips the anyhit.
template <bool DisableAnyHitWithoutIgnore> struct ReflEpcOptixTraverser {
    ::OptixTraversableHandle handle;

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_closest(math::Vec3f origin, math::Vec3f direction,
                                                                             float tmin, float tmax) const {
        TriangleHitPayload payload;
        // instance defaults to 0 == kTraceModeReflection, the mode the SBT reads.
        clear_triangle_hit(payload, algo_detail::kTraceTMax);
        if (handle != 0ull) {
            optixTrace(handle, make_float3(origin.x, origin.y, origin.z),
                       make_float3(direction.x, direction.y, direction.z), tmin, tmax, 0.0f, 255u,
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, payload.hit, payload.t, payload.bary_u, payload.bary_v,
                       payload.prim, payload.instance);
        }
        ::rayd::shared::rt::TriangleHit hit;
        hit.t = __uint_as_float(payload.t);
        hit.bary_u = __uint_as_float(payload.bary_u);
        hit.bary_v = __uint_as_float(payload.bary_v);
        hit.prim = static_cast<std::int32_t>(payload.prim);
        hit.instance = static_cast<std::int32_t>(payload.instance);
        hit.hit = payload.hit;
        return hit;
    }

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_first_blocker(math::Vec3f origin,
                                                                                   math::Vec3f direction, float tmin,
                                                                                   float tmax,
                                                                                   const std::int32_t* ignore,
                                                                                   int ignore_count) const {
        ::rayd::shared::rt::TriangleHit hit;
        hit.t = tmax;
        hit.bary_u = 0.0f;
        hit.bary_v = 0.0f;
        hit.prim = -1;
        hit.instance = -1;
        hit.hit = 0u;
        if (handle == 0ull)
            return hit;

        std::uint32_t visible = 1u;
        std::uint32_t blocker = 0xFFFFFFFFu;
        unsigned int ignore0 = ignore_count > 0 && ignore[0] >= 0 ? static_cast<unsigned int>(ignore[0]) : kInvalidPrim;
        unsigned int ignore1 = ignore_count > 1 && ignore[1] >= 0 ? static_cast<unsigned int>(ignore[1]) : kInvalidPrim;
        unsigned int ignore2 = ignore_count > 2 && ignore[2] >= 0 ? static_cast<unsigned int>(ignore[2]) : kInvalidPrim;
        unsigned int mode = kTraceModeVisibility;

        unsigned int ray_flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        if constexpr (DisableAnyHitWithoutIgnore) {
            const bool has_ignore = ignore0 != kInvalidPrim || ignore1 != kInvalidPrim || ignore2 != kInvalidPrim;
            if (!has_ignore) {
                ray_flags |= OPTIX_RAY_FLAG_DISABLE_ANYHIT;
            }
        }

        optixTrace(handle, make_float3(origin.x, origin.y, origin.z),
                   make_float3(direction.x, direction.y, direction.z), tmin, tmax, 0.0f, 255u, ray_flags, 0, 1, 0,
                   visible, blocker, ignore0, ignore1, ignore2, mode);
        hit.hit = visible == 0u ? 1u : 0u;
        hit.prim = static_cast<std::int32_t>(blocker);
        hit.instance = -1;
        return hit;
    }

    __device__ __forceinline__ bool trace_occluded_ignore(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                          float tmax, const std::int32_t* ignore,
                                                          int ignore_count) const {
        return trace_first_blocker(origin, direction, tmin, tmax, ignore, ignore_count).hit != 0u;
    }

    __device__ __forceinline__ bool trace_occluded(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                   float tmax) const {
        return trace_occluded_ignore(origin, direction, tmin, tmax, nullptr, 0);
    }
};

static_assert(::rayd::shared::rt::is_traverser_v<ReflEpcOptixTraverser<false>>,
              "ReflEpcOptixTraverser must satisfy the rt::Traverser concept.");
static_assert(::rayd::shared::rt::is_traverser_v<ReflEpcOptixTraverser<true>>,
              "ReflEpcOptixTraverser must satisfy the rt::Traverser concept.");

// OptiX programs for the shared EPC pipeline; one raygen launch per ray. The same
// programs serve reflection tracing and visibility checks, switched by payload 5.

/// Anyhit (visibility mode only): skip occluders on the ignore list (primitive or surface group).
static __forceinline__ __device__ void anyhit() {
    if (optixGetPayload_5() != kTraceModeVisibility) {
        return;
    }

    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int global_prim = algo_detail::global_primitive_id(shape_id, static_cast<int>(optixGetPrimitiveIndex()),
                                                             params.face_offsets, params.n_meshes);
    const unsigned int ignore0 = optixGetPayload_2();
    const unsigned int ignore1 = optixGetPayload_3();
    const unsigned int ignore2 = optixGetPayload_4();
    const int candidate = params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup
                              ? algo_detail::surface_group_for_prim(params, global_prim)
                              : global_prim;
    if ((ignore0 != kInvalidPrim && candidate == static_cast<int>(ignore0)) ||
        (ignore1 != kInvalidPrim && candidate == static_cast<int>(ignore1)) ||
        (ignore2 != kInvalidPrim && candidate == static_cast<int>(ignore2))) {
        optixIgnoreIntersection();
    }
}

/// Closest-hit: in visibility mode record the blocker; otherwise pack the reflection hit into payload.
static __forceinline__ __device__ void closesthit() {
    if (optixGetPayload_5() == kTraceModeVisibility) {
        const int shape_id = static_cast<int>(optixGetInstanceId());
        const int global_prim = algo_detail::global_primitive_id(shape_id, static_cast<int>(optixGetPrimitiveIndex()),
                                                                 params.face_offsets, params.n_meshes);
        optixSetPayload_0(0u);
        optixSetPayload_1(static_cast<unsigned int>(global_prim));
        return;
    }

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

/// Miss: in reflection mode mark "no hit"; in visibility mode a miss means unoccluded.
static __forceinline__ __device__ void miss() {
    if (optixGetPayload_5() != kTraceModeVisibility) {
        optixSetPayload_0(0u);
    }
}

/// Raygen: instantiate the primary / secondary EPC traversers and dispatch to the
/// shared algorithm. The same programs (above) back both trace families.
template <typename Policy, bool DirectOnly, bool PrimaryVisibilityOnly>
static __forceinline__ __device__ void run_reflection_epc_raygen() {
    using Traverser = ReflEpcOptixTraverser<Policy::DisableAnyHitWithoutIgnore>;
    using Config = ::rayd::shared::rt::TraceConfig<Policy, Traverser>;
    const Traverser primary{static_cast<::OptixTraversableHandle>(params.primary_handle)};
    const Traverser secondary{static_cast<::OptixTraversableHandle>(params.secondary_handle)};
    ::rayd::shared::multipath::run_reflection_epc_algo<Config, DirectOnly, PrimaryVisibilityOnly>(
        params, optixGetLaunchIndex().x, primary, secondary);
}

} // namespace reflection_epc_device
} // namespace rayd::shared::optix
