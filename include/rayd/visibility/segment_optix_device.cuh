// Copyright Xingyu Chen.
// Defines shared visibility support for segment optix device.

#pragma once

#ifdef __CUDACC__

#include <cstdint>

#include <optix.h>
#include <optix_device.h>

#include <rayd/math.h>
#include <rayd/rt/optix_primitive_id.h>
#include <rayd/visibility/segment_algo.h>
#include <rayd/visibility/segment_params.h>
#include <rayd/rt/traverser.h>

// OptiX entry layer for segment visibility. The algorithm bodies now live in the
// host-compilable rayd/visibility/segment_algo.h; this header
// keeps only the OptiX-specific pieces: the anyhit ignore-filter / closesthit /
// miss programs, the SegmentVisibilityOptixTraverser (the sole home of the
// occlusion optixTrace), the SegmentVisibilityDevicePolicy layout policy that
// becomes the Layout axis of TraceConfig, and the raygen entries that instantiate
// the traverser and dispatch to the algorithm.

namespace rayd::shared::optix {

template <bool DisableAnyHitWithoutIgnore, bool WriteOutputT>
struct SegmentVisibilityDevicePolicy {
    static constexpr bool disable_anyhit_without_ignore =
        DisableAnyHitWithoutIgnore;
    static constexpr bool write_output_t = WriteOutputT;
};

namespace segment_visibility {

/// Single-handle OptiX occlusion traverser for segment visibility. Wraps the one
/// occlusion optixTrace (TERMINATE_ON_FIRST_HIT, payload 0 = visible, payload 1 =
/// blocker, payload 2 = ignore-row base for the anyhit filter) and decodes it into
/// rt::TriangleHit. `DisableAnyHitWithoutIgnore` is the compile-time layout choice
/// of whether a ray with no ignore list skips the anyhit. The blocker is reported
/// as the already-global prim (payload 1, set by closesthit) with instance = -1, so
/// the algorithm's global_primitive_id passes it through unchanged; a mesh-local
/// traverser (CudaBvhTraverser) instead sets prim = local / instance = shape and the
/// same algorithm helper resolves the global id. `ignore_prim_ids` is the params
/// base pointer used to turn the algorithm's ignore sub-pointer back into the row
/// index the anyhit expects.
template <bool DisableAnyHitWithoutIgnore>
struct SegmentVisibilityOptixTraverser {
    ::OptixTraversableHandle handle;
    const int *ignore_prim_ids;

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_first_blocker(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count) const {
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
        unsigned int ignore_base =
            (ignore != nullptr && ignore_prim_ids != nullptr)
                ? static_cast<unsigned int>(ignore - ignore_prim_ids)
                : 0u;
        unsigned int ray_flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        if constexpr (DisableAnyHitWithoutIgnore) {
            if (ignore == nullptr || ignore_count <= 0)
                ray_flags |= OPTIX_RAY_FLAG_DISABLE_ANYHIT;
        }
        optixTrace(handle,
                   make_float3(origin.x, origin.y, origin.z),
                   make_float3(direction.x, direction.y, direction.z),
                   tmin,
                   tmax,
                   0.0f,
                   255u,
                   ray_flags,
                   0,
                   1,
                   0,
                   visible,
                   blocker,
                   ignore_base);
        hit.hit = visible == 0u ? 1u : 0u;
        hit.prim = static_cast<std::int32_t>(blocker);
        hit.instance = -1;
        return hit;
    }

    __device__ __forceinline__ bool trace_occluded_ignore(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count) const {
        return trace_first_blocker(origin, direction, tmin, tmax, ignore, ignore_count).hit != 0u;
    }

    __device__ __forceinline__ bool trace_occluded(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        return trace_occluded_ignore(origin, direction, tmin, tmax, nullptr, 0);
    }

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_closest(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        return trace_first_blocker(origin, direction, tmin, tmax, nullptr, 0);
    }
};

static_assert(::rayd::shared::rt::is_traverser_v<SegmentVisibilityOptixTraverser<false>>,
              "SegmentVisibilityOptixTraverser must satisfy the rt::Traverser concept.");
static_assert(::rayd::shared::rt::is_traverser_v<SegmentVisibilityOptixTraverser<true>>,
              "SegmentVisibilityOptixTraverser must satisfy the rt::Traverser concept.");

/// Anyhit (ignore filter): skip occluders whose global prim id is on this ray's
/// ignore row. Unchanged from the pre-migration program.
static __forceinline__ __device__ void anyhit(
    const SegmentVisibilityParams &params) {
    if (params.ignore_prim_ids == nullptr || params.ignore_k <= 0)
        return;

    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int global_prim = global_primitive_id(
        shape_id,
        static_cast<int>(optixGetPrimitiveIndex()),
        params.face_offsets,
        params.n_meshes);
    const unsigned int ignore_base = optixGetPayload_2();

    for (int slot = 0; slot < params.ignore_k; ++slot) {
        if (params.ignore_prim_ids[ignore_base + slot] == global_prim) {
            optixIgnoreIntersection();
            return;
        }
    }
}

static __forceinline__ __device__ void closesthit(
    const SegmentVisibilityParams &params) {
    optixSetPayload_0(0u);
    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int global_prim = global_primitive_id(
        shape_id,
        static_cast<int>(optixGetPrimitiveIndex()),
        params.face_offsets,
        params.n_meshes);
    optixSetPayload_1(static_cast<unsigned int>(global_prim));
}

static __forceinline__ __device__ void miss() {
    // Payload 0 is initialized to 1 by the traverser and remains clear on miss.
}

template <typename Policy>
static __forceinline__ __device__
SegmentVisibilityOptixTraverser<Policy::disable_anyhit_without_ignore>
make_segment_traverser(const SegmentVisibilityParams &params) {
    return SegmentVisibilityOptixTraverser<Policy::disable_anyhit_without_ignore>{
        static_cast<::OptixTraversableHandle>(params.handle), params.ignore_prim_ids};
}

template <typename Policy>
using SegmentVisibilityConfig = ::rayd::shared::rt::TraceConfig<
    Policy, SegmentVisibilityOptixTraverser<Policy::disable_anyhit_without_ignore>>;

template <typename Policy>
static __forceinline__ __device__ void raygen_segment(
    const SegmentVisibilityParams &params) {
    ::rayd::shared::multipath::segment_visibility_algo<SegmentVisibilityConfig<Policy>>(
        params, optixGetLaunchIndex().x, make_segment_traverser<Policy>(params));
}

template <typename Policy>
static __forceinline__ __device__ void raygen_segment_pair(
    const SegmentVisibilityParams &params) {
    ::rayd::shared::multipath::segment_pair_visibility_algo<SegmentVisibilityConfig<Policy>>(
        params, optixGetLaunchIndex().x, make_segment_traverser<Policy>(params));
}

template <typename Policy>
static __forceinline__ __device__ void raygen_axial_edge(
    const SegmentVisibilityParams &params) {
    ::rayd::shared::multipath::axial_edge_visibility_algo<SegmentVisibilityConfig<Policy>>(
        params, optixGetLaunchIndex().x, make_segment_traverser<Policy>(params));
}

template <typename Policy>
static __forceinline__ __device__ void raygen_segment_chain(
    const SegmentVisibilityParams &params) {
    ::rayd::shared::multipath::segment_chain_visibility_algo<SegmentVisibilityConfig<Policy>>(
        params, optixGetLaunchIndex().x, make_segment_traverser<Policy>(params));
}

} // namespace segment_visibility
} // namespace rayd::shared::optix

#endif
