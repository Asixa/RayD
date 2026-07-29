// Copyright Xingyu Chen.
// Defines shared visibility support for segment algo.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <rayd/math.h>
#include <rayd/visibility/segment_params.h>
#include <rayd/rt/numeric_policy.h>
#include <rayd/rt/qualifiers.h>
#include <rayd/rt/traverser.h>

// Host-compilable segment-visibility algorithm. This is the de-CUDA-ised body of
// the former segment_visibility raygen family: math is math::Vec3f throughout
// (mirroring the exact arithmetic op order of the old local CUDA vector helpers so
// device codegen stays bit-identical), every occlusion cast goes through an
// rt::is_traverser Traverser (so no OptiX ray-cast intrinsic, payload register, or
// launch-index query appears here), and the lane index is a plain parameter.
// segment_visibility_device.cuh instantiates it with
// TraceConfig<SegmentVisibilityDevicePolicy, SegmentVisibilityOptixTraverser>; the
// CUDA fused executor (P4d) will reuse it with CudaBvhTraverser.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace segment_visibility_algo_detail {

using math::Vec3f;
using ::rayd::shared::optix::SegmentVisibilityParams;

constexpr float kTraceTMin = 1e-5f;
constexpr float kRayBias = 1e-5f;
constexpr float kMinSegmentLength = 2e-5f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
static_assert(kMinSegmentLength == ::rayd::shared::rt::kMinSegmentLength);

// Bit-cast of a uint sentinel to float. On device this is __uint_as_float; on the
// host it is a byte copy. 0x7f800000 is +inf, the segment-visibility out_t sentinel.
RAYD_HOST_DEVICE float uint_as_float(unsigned int bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// Host-compilable mirror of device_hit.h's global_primitive_id (that header is
// device-only). With shape_id outside [0, mesh_count) the face offset is 0 and the
// call passes the primitive through unchanged; the OptiX occlusion traverser relies
// on this by returning the already-global blocker prim with instance = -1.
RAYD_HOST_DEVICE int global_primitive_id(int shape_id, int local_primitive, const int* face_offsets, int mesh_count) {
    const int face_offset = (shape_id >= 0 && shape_id < mesh_count) ? face_offsets[shape_id] : 0;
    return face_offset + local_primitive;
}

RAYD_HOST_DEVICE Vec3f load_aos_vec3(const float* value, unsigned int index) {
    const unsigned int base = index * 3u;
    return math::make_vec3(value[base], value[base + 1u], value[base + 2u]);
}

RAYD_HOST_DEVICE Vec3f load_soa_vec3(const float* x, const float* y, const float* z, unsigned int index) {
    return math::make_vec3(x[index], y[index], z[index]);
}

RAYD_HOST_DEVICE bool is_active(const SegmentVisibilityParams& params, unsigned int ray) {
    return params.active_mask == nullptr || params.active_mask[ray] != 0u;
}

RAYD_HOST_DEVICE Vec3f load_start(const SegmentVisibilityParams& params, unsigned int ray) {
    return params.start_aos != nullptr ? load_aos_vec3(params.start_aos, ray)
                                       : load_soa_vec3(params.start_x, params.start_y, params.start_z, ray);
}

RAYD_HOST_DEVICE Vec3f load_end_a(const SegmentVisibilityParams& params, unsigned int ray) {
    return params.end_aos != nullptr ? load_aos_vec3(params.end_aos, ray)
                                     : load_soa_vec3(params.end_x, params.end_y, params.end_z, ray);
}

RAYD_HOST_DEVICE Vec3f load_end_b(const SegmentVisibilityParams& params, unsigned int ray) {
    return params.end_b_aos != nullptr ? load_aos_vec3(params.end_b_aos, ray)
                                       : load_soa_vec3(params.end_b_x, params.end_b_y, params.end_b_z, ray);
}

RAYD_HOST_DEVICE Vec3f load_chain_point(const SegmentVisibilityParams& params, unsigned int chain, int point_index) {
    const int slot = static_cast<int>(chain) * params.max_points + point_index;
    return math::make_vec3(params.chain_point_x[slot], params.chain_point_y[slot], params.chain_point_z[slot]);
}

/// One segment occlusion cast. Mirrors the old trace_segment: the active / null-
/// handle guard and the degenerate MinSegmentLength guard are algorithm semantics;
/// the acceleration-structure cast goes through the Traverser. `ignore_base` is the
/// row offset into params.ignore_prim_ids the OptiX anyhit indexes. When
/// `blocker_prim` is set the routine collects the winning blocker's global prim id
/// (via global_primitive_id, so it is correct whether the traverser reports the
/// blocker as an already-global prim with shape -1 or a mesh-local prim + shape).
template <typename Config>
RAYD_DEVICE std::uint32_t trace_segment(const SegmentVisibilityParams& params, Vec3f start, Vec3f end, bool active,
                                        unsigned int ignore_base, const typename Config::Traverser& traverser,
                                        std::uint32_t* blocker_prim) {
    if (!active || params.handle == 0ull) {
        if (blocker_prim != nullptr)
            *blocker_prim = 0xFFFFFFFFu;
        return 0u;
    }

    Vec3f direction = math::subtract(end, start);
    const float length = sqrtf(math::dot(direction, direction));
    if (length <= kMinSegmentLength) {
        if (blocker_prim != nullptr)
            *blocker_prim = 0xFFFFFFFFu;
        return 1u;
    }

    direction = math::scale(direction, 1.0f / length);
    const Vec3f origin = math::add(start, math::scale(direction, kRayBias));
    const float tmax = fmaxf(length - 2.0f * kRayBias, 0.0f);

    const int ignore_count = params.ignore_prim_ids != nullptr ? params.ignore_k : 0;
    const std::int32_t* ignore_ptr = params.ignore_prim_ids != nullptr ? params.ignore_prim_ids + ignore_base : nullptr;

    if (blocker_prim == nullptr) {
        return traverser.trace_occluded_ignore(origin, direction, kTraceTMin, tmax, ignore_ptr, ignore_count) ? 0u : 1u;
    }

    const ::rayd::shared::rt::TriangleHit hit =
        traverser.trace_first_blocker(origin, direction, kTraceTMin, tmax, ignore_ptr, ignore_count);
    *blocker_prim = hit.hit != 0u
                        ? static_cast<std::uint32_t>(
                              global_primitive_id(hit.instance, hit.prim, params.face_offsets, params.n_meshes))
                        : 0xFFFFFFFFu;
    return hit.hit != 0u ? 0u : 1u;
}

} // namespace segment_visibility_algo_detail

/// Single-segment visibility for one lane (former raygen_segment).
template <typename Config>
RAYD_DEVICE void segment_visibility_algo(const ::rayd::shared::optix::SegmentVisibilityParams& params,
                                         std::uint32_t ray, const typename Config::Traverser& traverser) {
    using namespace segment_visibility_algo_detail;
    using Policy = typename Config::Layout;

    if (ray >= static_cast<unsigned int>(params.n_rays))
        return;

    std::uint32_t blocker = 0xFFFFFFFFu;
    const bool collect_blocker = params.out_first_blocked_prim != nullptr;
    const std::uint32_t visible =
        trace_segment<Config>(params, load_start(params, ray), load_end_a(params, ray), is_active(params, ray),
                              ray * static_cast<unsigned int>(params.ignore_k), traverser,
                              collect_blocker ? &blocker : nullptr);
    params.out_visible[ray] = visible != 0u ? 1u : 0u;
    if (collect_blocker) {
        params.out_first_blocked_prim[ray] = visible == 0u && blocker != 0xFFFFFFFFu ? static_cast<int>(blocker) : -1;
    }
    if constexpr (Policy::write_output_t) {
        if (params.out_t != nullptr)
            params.out_t[ray] = uint_as_float(0x7f800000u);
    }
}

/// Segment-pair visibility for one lane (former raygen_segment_pair).
template <typename Config>
RAYD_DEVICE void segment_pair_visibility_algo(const ::rayd::shared::optix::SegmentVisibilityParams& params,
                                              std::uint32_t ray, const typename Config::Traverser& traverser) {
    using namespace segment_visibility_algo_detail;

    if (ray >= static_cast<unsigned int>(params.n_rays))
        return;

    const bool active = is_active(params, ray);
    const Vec3f start = load_start(params, ray);
    const unsigned int ignore_base = ray * static_cast<unsigned int>(params.ignore_k);
    params.out_visible[ray] =
        trace_segment<Config>(params, start, load_end_a(params, ray), active, ignore_base, traverser, nullptr) != 0u;
    params.out_visible_b[ray] =
        trace_segment<Config>(params, start, load_end_b(params, ray), active, ignore_base, traverser, nullptr) != 0u;
}

/// Axial-edge visibility for one lane (former raygen_axial_edge): visible if any
/// sample along the edge span is reachable from the source.
template <typename Config>
RAYD_DEVICE void axial_edge_visibility_algo(const ::rayd::shared::optix::SegmentVisibilityParams& params,
                                            std::uint32_t ray, const typename Config::Traverser& traverser) {
    using namespace segment_visibility_algo_detail;

    if (ray >= static_cast<unsigned int>(params.n_rays))
        return;

    const bool active = is_active(params, ray);
    const Vec3f source = load_start(params, ray);
    const Vec3f edge_pos = load_end_a(params, ray);
    const Vec3f edge_dir = load_soa_vec3(params.edge_dir_x, params.edge_dir_y, params.edge_dir_z, ray);
    const float line_min = params.edge_t_min[ray];
    const float span = fmaxf(params.edge_t_max[ray] - line_min, 0.0f);
    std::uint32_t any_visible = 0u;

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (int i = 0; i < ::rayd::shared::optix::SegmentVisibilityMaxSamples; ++i) {
        if (i < params.sample_count) {
            const float t = line_min + params.sample_fractions[i] * span;
            const Vec3f sample = math::add(edge_pos, math::scale(edge_dir, t));
            any_visible |= trace_segment<Config>(params, source, sample, active, 0u, traverser, nullptr);
        }
    }
    params.out_visible[ray] = any_visible != 0u ? 1u : 0u;
}

/// Segment-chain visibility for one lane (former raygen_segment_chain): the chain
/// is visible only if every segment is unoccluded; on the first blocked segment the
/// segment index and the blocker's global prim id are recorded.
template <typename Config>
RAYD_DEVICE void segment_chain_visibility_algo(const ::rayd::shared::optix::SegmentVisibilityParams& params,
                                               std::uint32_t chain, const typename Config::Traverser& traverser) {
    using namespace segment_visibility_algo_detail;

    if (chain >= static_cast<unsigned int>(params.n_rays))
        return;

    if (!is_active(params, chain)) {
        params.out_visible[chain] = 0u;
        params.out_first_blocked_segment[chain] = -1;
        params.out_first_blocked_prim[chain] = -1;
        return;
    }

    int segment_count = params.chain_length != nullptr ? params.chain_length[chain] : params.max_segments;
    segment_count = segment_count < 0 ? 0 : segment_count;
    segment_count = segment_count > params.max_segments ? params.max_segments : segment_count;

    std::uint32_t all_visible = 1u;
    int first_blocked_segment = -1;
    int first_blocked_prim = -1;
    for (int segment = 0; segment < segment_count; ++segment) {
        const Vec3f start = load_chain_point(params, chain, segment);
        const Vec3f end = load_chain_point(params, chain, segment + 1);
        const unsigned int ignore_base =
            params.ignore_k > 0
                ? (chain * static_cast<unsigned int>(params.max_segments) + static_cast<unsigned int>(segment)) *
                      static_cast<unsigned int>(params.ignore_k)
                : 0u;

        std::uint32_t blocker_prim = 0xFFFFFFFFu;
        if (trace_segment<Config>(params, start, end, true, ignore_base, traverser, &blocker_prim) == 0u) {
            all_visible = 0u;
            first_blocked_segment = segment;
            first_blocked_prim = static_cast<int>(blocker_prim);
            break;
        }
    }

    params.out_visible[chain] = all_visible != 0u ? 1u : 0u;
    params.out_first_blocked_segment[chain] = first_blocked_segment;
    params.out_first_blocked_prim[chain] = first_blocked_prim;
}

} // namespace rayd::shared::multipath
