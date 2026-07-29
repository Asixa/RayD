// Copyright Xingyu Chen.
// Defines shared reflection support for optix traverser.

#pragma once

#include <cstdint>

#include <optix.h>
#include <optix_device.h>

#include <rayd/math.h>
#include <rayd/reflection/optix_hit.h>
#include <rayd/rt/traverser.h>

// OptiX traverser shim: the only place the migrated multipath algorithms touch
// optixTrace and the six-register TriangleHitPayload codec. It wraps ONE
// OptixTraversableHandle and decodes each cast into the backend-neutral
// rt::TriangleHit, so the algorithm bodies stay host-compilable and never see an
// optix* token or a payload register. The dual-handle "choose nearest" logic is
// pipeline semantics and stays in the algorithm, which simply owns two of these
// shims (primary + secondary handle).
//
// Device-only (includes optix_device.h). Lives under shared/reflection/ because the
// grep-gate's explicit OptiX-shim exception.

namespace rayd::shared::optix {

struct OptixTraverser {
    ::OptixTraversableHandle handle;
    unsigned int ray_flags;
    unsigned int sbt_offset;
    unsigned int sbt_stride;
    unsigned int miss_index;
    float miss_t;  ///< Distance the cleared payload reports on a miss / null handle.

    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit trace_closest(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        TriangleHitPayload payload;
        clear_triangle_hit(payload, miss_t);
        if (handle != 0ull) {
            optixTrace(
                handle,
                make_float3(origin.x, origin.y, origin.z),
                make_float3(direction.x, direction.y, direction.z),
                tmin,
                tmax,
                0.0f,
                255u,
                ray_flags,
                sbt_offset,
                sbt_stride,
                miss_index,
                payload.hit,
                payload.t,
                payload.bary_u,
                payload.bary_v,
                payload.prim,
                payload.instance);
        }
        return decode(payload);
    }

    __device__ __forceinline__ bool trace_occluded(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        return trace_closest(origin, direction, tmin, tmax).hit != 0u;
    }

    // The reflection SBT carries no anyhit ignore filter, so the ignore-aware
    // OptiX occlusion / first-blocker paths are wired with the P4 Stage B
    // segment-visibility migration. These satisfy rt::is_traverser and fall back
    // to the plain closest cast (ignore list unused) until then.
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

private:
    __device__ __forceinline__ ::rayd::shared::rt::TriangleHit decode(
        const TriangleHitPayload &payload) const {
        ::rayd::shared::rt::TriangleHit hit;
        hit.t = __uint_as_float(payload.t);
        hit.bary_u = __uint_as_float(payload.bary_u);
        hit.bary_v = __uint_as_float(payload.bary_v);
        hit.prim = static_cast<std::int32_t>(payload.prim);
        hit.instance = static_cast<std::int32_t>(payload.instance);
        hit.hit = payload.hit;
        return hit;
    }
};

static_assert(::rayd::shared::rt::is_traverser_v<OptixTraverser>,
              "OptixTraverser must satisfy the rt::Traverser concept.");

} // namespace rayd::shared::optix
