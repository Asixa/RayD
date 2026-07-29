#pragma once

#include <cstdint>

#include <optix.h>
#include <optix_device.h>

#include <rayd/detail/vec3.h>
#include <rayd/detail/diffraction/paths_algo.h>
#include <rayd/detail/rt/traverser.h>

// OptiX entry layer for first-order diffraction path export. The algorithm
// bodies now live in the host-compilable
// rayd/detail/diffraction/paths_algo.h; this header keeps only the
// OptiX-specific pieces: the four-register closesthit / miss programs, the
// DiffractionPathsOptixTraverser (the sole home of this pipeline's closest-hit
// optixTrace, used for the visibility casts), and the raygen entries that
// instantiate the traversers and dispatch to the algorithm. The traverser owns
// ONE handle; the primary/secondary "choose nearest" logic is pipeline semantics
// and stays in the algorithm, which owns two of these traversers.

namespace rayd::shared::optix {

namespace diffraction_paths {

namespace algo_detail = ::rayd::shared::multipath::diffraction_paths_algo_detail;

/// Single-handle OptiX traverser for the diffraction path-export visibility
/// casts. Wraps the four-register closest-hit optixTrace (payload 0 = hit, 1 = t,
/// 2 = prim, 3 = instance) with the pipeline's degenerate-tmax guard, and decodes
/// it into rt::TriangleHit (the barycentrics are unused by the export, so they
/// clear to 0). A null handle or a collapsed segment yields the cleared miss hit.
struct DiffractionPathsOptixTraverser {
    ::OptixTraversableHandle handle;

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_closest(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        unsigned int hit = 0u;
        unsigned int t = __float_as_uint(1e8f);
        unsigned int prim = 0u;
        unsigned int instance = 0u;
        if (handle != 0ull && tmax > tmin) {
            optixTrace(handle,
                       make_float3(origin.x, origin.y, origin.z),
                       make_float3(direction.x, direction.y, direction.z),
                       tmin,
                       tmax,
                       0.0f,
                       255u,
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                       0,
                       1,
                       0,
                       hit,
                       t,
                       prim,
                       instance);
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

    __device__ __forceinline__ bool trace_occluded(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        return trace_closest(origin, direction, tmin, tmax).hit != 0u;
    }

    __device__ __forceinline__ bool trace_occluded_ignore(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t * /*ignore*/, int /*ignore_count*/) const {
        return trace_occluded(origin, direction, tmin, tmax);
    }

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_first_blocker(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t * /*ignore*/, int /*ignore_count*/) const {
        return trace_closest(origin, direction, tmin, tmax);
    }
};

static_assert(::rayd::shared::rt::is_traverser_v<DiffractionPathsOptixTraverser>,
              "DiffractionPathsOptixTraverser must satisfy the rt::Traverser concept.");

/// Closest-hit: record the four-register hit payload (hit / t / prim / instance).
static __forceinline__ __device__ void closesthit() {
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetPrimitiveIndex());
    optixSetPayload_3(optixGetInstanceId());
}

/// Miss: payload 0 stays clear (the traverser initializes it to 1 = blocked).
static __forceinline__ __device__ void miss() {
    optixSetPayload_0(0u);
}

/// Combined order-1 export raygen. `SplitScene` selects the primary-only vs
/// primary+secondary visibility casts.
template <typename Params, bool SplitScene>
static __forceinline__ __device__ void raygen_order1(const Params &params) {
    const unsigned int lane = optixGetLaunchIndex().x;
    const DiffractionPathsOptixTraverser primary{params.primary_handle};
    const DiffractionPathsOptixTraverser secondary{params.secondary_handle};
    ::rayd::shared::multipath::trace_paths_order1_algo<
        Params, DiffractionPathsOptixTraverser, SplitScene>(params, lane, primary, secondary);
}

/// Two-phase source-visibility prepass raygen (primary handle only).
template <typename Params>
static __forceinline__ __device__ void raygen_source_visibility(const Params &params) {
    const unsigned int lane = optixGetLaunchIndex().x;
    const DiffractionPathsOptixTraverser primary{params.primary_handle};
    const DiffractionPathsOptixTraverser secondary{params.secondary_handle};
    ::rayd::shared::multipath::trace_paths_source_visibility_algo<
        Params, DiffractionPathsOptixTraverser>(params, lane, primary, secondary);
}

/// Two-phase target-export raygen (primary handle only).
template <typename Params>
static __forceinline__ __device__ void raygen_target_export(const Params &params) {
    const unsigned int lane = optixGetLaunchIndex().x;
    const DiffractionPathsOptixTraverser primary{params.primary_handle};
    const DiffractionPathsOptixTraverser secondary{params.secondary_handle};
    ::rayd::shared::multipath::trace_paths_target_export_algo<
        Params, DiffractionPathsOptixTraverser>(params, lane, primary, secondary);
}

} // namespace diffraction_paths
} // namespace rayd::shared::optix
