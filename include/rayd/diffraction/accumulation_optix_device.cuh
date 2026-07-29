// Copyright Xingyu Chen.
// Defines shared diffraction support for accumulation optix device.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include <rayd/math.h>
#include <rayd/diffraction/accumulation_algo.h>
#include <rayd/rt/traverser.h>

// OptiX entry layer for diffraction accumulation. The algorithm body now lives in
// the host-compilable, traverser-templated
// rayd/diffraction/accumulation_algo.h; this header keeps only
// the OptiX-specific pieces: the four-register closesthit / miss programs, the
// DiffractionAccumulationOptixTraverser (the two optixTrace sites: a closest cast
// and a terminate-on-first-hit occlusion cast), and the DiffractionAccumulation
// Device<Policy> entry struct whose raygen methods read the launch index,
// instantiate the algorithm with the OptiX traversers, and dispatch. The backend
// .cu adapters keep their `Device::closesthit()` / `Device::miss()` /
// `Device::run_*_raygen<...>()` entry names and signatures unchanged.

namespace rayd::shared::multipath::diffraction_accumulation {

/// Single-handle OptiX traverser for the diffraction-accumulation casts. Wraps
/// the four-register closest-hit optixTrace (payload 0 = hit, 1 = t, 2 = prim,
/// 3 = instance) and the one-register terminate-on-first-hit occlusion optixTrace,
/// both with the pipeline's degenerate-tmax guard. trace_closest decodes into
/// rt::TriangleHit (barycentrics unused, cleared to 0); trace_occluded returns
/// whether anything blocks the segment. A null handle or a collapsed segment
/// yields the cleared miss hit / unblocked result.
struct DiffractionAccumulationOptixTraverser {
    ::OptixTraversableHandle handle;

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_closest(math::Vec3f origin, math::Vec3f direction,
                                                                             float tmin, float tmax) const {
        unsigned int hit = 0u;
        unsigned int t = __float_as_uint(1e8f);
        unsigned int prim = 0u;
        unsigned int instance = 0u;
        if (handle != 0ull && tmax > tmin) {
            optixTrace(handle, make_float3(origin.x, origin.y, origin.z),
                       make_float3(direction.x, direction.y, direction.z), tmin, tmax, 0.0f, 255u,
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, hit, t, prim, instance);
        }
        ::rayd::shared::rt::TriangleHit result;
        result.t = __uint_as_float(t);
        result.bary_u = 0.0f;
        result.bary_v = 0.0f;
        result.prim = static_cast<std::int32_t>(prim);
        result.instance = static_cast<std::int32_t>(instance);
        result.hit = hit;
        return result;
    }

    __device__ __forceinline__ bool trace_occluded(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                   float tmax) const {
        if (handle == 0ull || tmax <= tmin) {
            return false;
        }
        unsigned int blocked = 1u;
        optixTrace(handle, make_float3(origin.x, origin.y, origin.z),
                   make_float3(direction.x, direction.y, direction.z), tmin, tmax, 0.0f, 255u,
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   0, 1, 0, blocked);
        return blocked != 0u;
    }

    __device__ __forceinline__ bool trace_occluded_ignore(math::Vec3f origin, math::Vec3f direction, float tmin,
                                                          float tmax, const std::int32_t* /*ignore*/,
                                                          int /*ignore_count*/) const {
        return trace_occluded(origin, direction, tmin, tmax);
    }

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_first_blocker(math::Vec3f origin,
                                                                                   math::Vec3f direction, float tmin,
                                                                                   float tmax,
                                                                                   const std::int32_t* /*ignore*/,
                                                                                   int /*ignore_count*/) const {
        return trace_closest(origin, direction, tmin, tmax);
    }
};

static_assert(::rayd::shared::rt::is_traverser_v<DiffractionAccumulationOptixTraverser>,
              "DiffractionAccumulationOptixTraverser must satisfy the rt::Traverser concept.");

/// OptiX entry struct. Keeps the pre-migration name / static-method surface so the
/// backend .cu adapters are unchanged; each raygen reads the launch index and
/// dispatches to the shared algorithm instantiated with the OptiX traverser.
template <typename Policy> struct DiffractionAccumulationDevice {
    using Algo = ::rayd::shared::multipath::DiffractionAccumulationAlgo<Policy, DiffractionAccumulationOptixTraverser>;

    static __forceinline__ __device__ Algo make_algo() {
        return Algo{DiffractionAccumulationOptixTraverser{
                        static_cast<::OptixTraversableHandle>(Policy::params().primary_handle)},
                    DiffractionAccumulationOptixTraverser{
                        static_cast<::OptixTraversableHandle>(Policy::params().secondary_handle)}};
    }

    static __forceinline__ __device__ void closesthit() {
        optixSetPayload_0(1u);
        optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
        optixSetPayload_2(optixGetPrimitiveIndex());
        optixSetPayload_3(optixGetInstanceId());
    }

    static __forceinline__ __device__ void miss() { optixSetPayload_0(0u); }

    template <bool PrimaryOnly, bool IncludeCoherent, bool IncludeDirect, bool IncludeKeller, bool IncludeSuffix>
    static __forceinline__ __device__ void run_diffraction_order1_accumulation_raygen() {
        make_algo()
            .template run_diffraction_order1_accumulation_algo<PrimaryOnly, IncludeCoherent, IncludeDirect,
                                                               IncludeKeller, IncludeSuffix>(optixGetLaunchIndex().x);
    }

    template <bool PrimaryOnly>
    static __forceinline__ __device__ void run_diffraction_order1_source_visibility_raygen() {
        make_algo().template run_diffraction_order1_source_visibility_algo<PrimaryOnly>(optixGetLaunchIndex().x);
    }

    template <bool PrimaryOnly>
    static __forceinline__ __device__ void run_diffraction_order1_no_suffix_target_accumulation_raygen() {
        make_algo().template run_diffraction_order1_no_suffix_target_accumulation_algo<PrimaryOnly>(
            optixGetLaunchIndex().x);
    }

    template <bool PrimaryOnly>
    static __forceinline__ __device__ void run_diffraction_order1_suffix_first_visibility_raygen() {
        make_algo().template run_diffraction_order1_suffix_first_visibility_algo<PrimaryOnly>(optixGetLaunchIndex().x);
    }

    template <bool PrimaryOnly>
    static __forceinline__ __device__ void run_diffraction_order1_suffix_target_accumulation_raygen() {
        make_algo().template run_diffraction_order1_suffix_target_accumulation_algo<PrimaryOnly>(
            optixGetLaunchIndex().x);
    }

    template <bool PrimaryOnly>
    static __forceinline__ __device__ void run_diffraction_order1_coherent_accumulation_raygen() {
        make_algo().template run_diffraction_order1_coherent_accumulation_algo<PrimaryOnly>(optixGetLaunchIndex().x);
    }

    template <bool PrimaryOnly> static __forceinline__ __device__ void run_diffraction_chain_accumulation_raygen() {
        make_algo().template run_diffraction_chain_accumulation_algo<PrimaryOnly>(optixGetLaunchIndex().x);
    }
};

} // namespace rayd::shared::multipath::diffraction_accumulation
