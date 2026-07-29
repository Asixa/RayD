#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <rayd/detail/contracts.h>
#include <rayd/detail/vec3.h>
#include <rayd/detail/reflection/epc_params.h>
#include <rayd/detail/reflection/epc_chain.h>
#include <rayd/detail/reflection/reflection_geometry.h>
#include <rayd/detail/rt/numeric_policy.h>
#include <rayd/detail/rt/qualifiers.h>
#include <rayd/detail/rt/traverser.h>

// Host-compilable reflection-EPC discovery algorithm. This is the de-CUDA-ised
// body of the former run_reflection_epc_raygen: math is math::Vec3f throughout
// (mirroring the exact arithmetic op order of the old local CUDA vector helpers so
// device codegen stays bit-identical), the two ray-cast families (reflector scene
// trace + segment visibility) go through an rt::is_traverser Traverser, and the
// lane index is a plain parameter. reflection_epc_device.cuh instantiates it with
// TraceConfig<ReflEpc layout policy, ReflEpcOptixTraverser>; the same Traverser
// serves both trace families (trace_closest = reflector scene, trace_first_blocker
// = visibility), switched by the OptiX payload mode inside the shim.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace reflection_epc_algo_detail {

using math::Vec3f;
using ::rayd::shared::optix::ReflEpcMaxBounces;
using ::rayd::shared::optix::ReflEpcParams;
using ::rayd::shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;
namespace reflection = ::rayd::shared::reflection;

constexpr float kTraceTMin = rayd::shared::GeneralEpsilon;
constexpr float kTraceTMax = 1e8f;
constexpr float kRayBias = rayd::shared::GeneralEpsilon;
constexpr float kMinSegmentLength = 2e-5f;
constexpr float kEpcTolerance = 1e-4f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kTraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
static_assert(kMinSegmentLength == ::rayd::shared::rt::kMinSegmentLength);
static_assert(kEpcTolerance == ::rayd::shared::rt::kEpcBarycentricSlack);

// Bit-cast of a uint sentinel to float. On device this is __uint_as_float; on the
// host a byte copy. 0x7f800000 is +inf, the EPC out_path_length invalid sentinel.
RAYD_HOST_DEVICE float uint_as_float(unsigned int bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

RAYD_HOST_DEVICE bool is_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// Host-compilable mirror of device_hit.h's global_primitive_id (that header is
// device-only). With shape_id outside [0, mesh_count) the face offset is 0 and the
// primitive passes through unchanged; the OptiX visibility traverser relies on this
// by reporting the already-global blocker prim with shape -1.
RAYD_HOST_DEVICE int global_primitive_id(
    int shape_id, int local_primitive, const int *face_offsets, int mesh_count) {
    const int face_offset =
        (shape_id >= 0 && shape_id < mesh_count) ? face_offsets[shape_id] : 0;
    return face_offset + local_primitive;
}

// length3(v) = sqrtf(fmaxf(dot(v, v), 0)) - bit-identical to the discovery kernel.
RAYD_HOST_DEVICE float length3(Vec3f value) {
    return sqrtf(fmaxf(math::dot(value, value), 0.0f));
}

// normalize3(v) = rsqrtf(fmaxf(dot(v, v), 1e-12)) * v, which is exactly
// reflection::epc_normalize (verified bit-identical on device).
RAYD_HOST_DEVICE Vec3f normalize3(Vec3f value) {
    return reflection::epc_normalize(value);
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_nearest_hit(
    ::rayd::shared::rt::TriangleHit a,
    ::rayd::shared::rt::TriangleHit b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return b.t < a.t ? b : a;
}

RAYD_HOST_DEVICE Vec3f load_triangle_p0(const ReflEpcParams &params, int prim) {
    return math::make_vec3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]);
}

RAYD_HOST_DEVICE Vec3f load_triangle_e1(const ReflEpcParams &params, int prim) {
    return math::make_vec3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]);
}

RAYD_HOST_DEVICE Vec3f load_triangle_e2(const ReflEpcParams &params, int prim) {
    return math::make_vec3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]);
}

RAYD_HOST_DEVICE Vec3f load_triangle_normal(const ReflEpcParams &params, int prim) {
    return normalize3(math::make_vec3(
        params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim]));
}

RAYD_HOST_DEVICE bool has_surface_groups(const ReflEpcParams &params) {
    return params.surface_group_id != nullptr &&
           params.surface_group_size != nullptr &&
           params.surface_group_members != nullptr &&
           params.surface_group_count > 0 &&
           params.surface_max_group_size > 0;
}

RAYD_HOST_DEVICE int surface_group_for_prim(const ReflEpcParams &params, int prim) {
    if (!has_surface_groups(params) || prim < 0 || prim >= params.surface_group_id_count) {
        return -1;
    }
    const int group = params.surface_group_id[prim];
    return group >= 0 && group < params.surface_group_count ? group : -1;
}

RAYD_HOST_DEVICE int expected_prim_for_bounce(const ReflEpcParams &params, int slot) {
    if (params.expected_prim_ids == nullptr ||
        slot < 0 ||
        slot >= params.expected_prim_count) {
        return -1;
    }
    return params.expected_prim_ids[slot];
}

RAYD_HOST_DEVICE bool direct_plane_mode(const ReflEpcParams &params) {
    return params.direct_plane_point_x != nullptr &&
           params.direct_plane_point_y != nullptr &&
           params.direct_plane_point_z != nullptr &&
           params.direct_plane_normal_x != nullptr &&
           params.direct_plane_normal_y != nullptr &&
           params.direct_plane_normal_z != nullptr;
}

RAYD_HOST_DEVICE int final_ignore_group_for_ray(const ReflEpcParams &params, int ray_index) {
    if (params.final_ignore_group_ids == nullptr ||
        params.final_ignore_group_count <= 0) {
        return -1;
    }
    const int index = params.final_ignore_group_count == 1 ? 0 : ray_index;
    if (index < 0 || index >= params.final_ignore_group_count) {
        return -1;
    }
    return params.final_ignore_group_ids[index];
}

RAYD_HOST_DEVICE bool point_inside_triangle(
    const ReflEpcParams &params, int prim, Vec3f point) {
    if (prim < 0 || prim >= params.n_triangles) {
        return false;
    }
    const Vec3f p0 = load_triangle_p0(params, prim);
    const Vec3f e1 = load_triangle_e1(params, prim);
    const Vec3f e2 = load_triangle_e2(params, prim);
    const Vec3f vp = math::subtract(point, p0);
    const float d00 = math::dot(e1, e1);
    const float d01 = math::dot(e1, e2);
    const float d11 = math::dot(e2, e2);
    const float d20 = math::dot(vp, e1);
    const float d21 = math::dot(vp, e2);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) <= 1e-12f) {
        return false;
    }
    const float plane_deviation = math::dot(vp, math::cross(e1, e2));
    const float scale_sq = fmaxf(fmaxf(d00, d11), 1.0f);
    const float plane_tolerance = fmaxf(params.plane_tolerance, 0.0f);
    if (plane_deviation * plane_deviation >
        plane_tolerance * plane_tolerance * scale_sq * denom) {
        return false;
    }
    const float inv_denom = 1.0f / denom;
    const float u = (d11 * d20 - d01 * d21) * inv_denom;
    const float v = (d00 * d21 - d01 * d20) * inv_denom;
    return u >= -kEpcTolerance &&
           v >= -kEpcTolerance &&
           u + v <= 1.0f + kEpcTolerance;
}

RAYD_HOST_DEVICE bool point_inside_surface_group(
    const ReflEpcParams &params, int group, Vec3f point, int &resolved_prim) {
    resolved_prim = -1;
    if (!has_surface_groups(params) ||
        group < 0 ||
        group >= params.surface_group_count ||
        params.surface_max_group_size <= 0) {
        return false;
    }

    int member_count = params.surface_group_size[group];
    if (member_count < 0) {
        member_count = 0;
    }
    if (member_count > params.surface_max_group_size) {
        member_count = params.surface_max_group_size;
    }
    const int base = group * params.surface_max_group_size;
    for (int i = 0; i < member_count; ++i) {
        const int prim = params.surface_group_members[base + i];
        if (prim < 0) {
            continue;
        }
        if (point_inside_triangle(params, prim, point)) {
            resolved_prim = prim;
            return true;
        }
    }
    return false;
}

// Segment-plane intersection with the discovery kernel's guards. Retained from the
// device header (parallel tolerance 1e-7, segment tolerance kEpcTolerance) so
// reflection_geometry.h's intersect_segment_plane stays host-exercised; the
// discovery kernel itself does its plane solves through epc_backtrace_and_length.
RAYD_HOST_DEVICE bool intersect_line_plane(
    Vec3f line_start, Vec3f line_end, Vec3f plane_point, Vec3f plane_normal, Vec3f &point) {
    Vec3f shared_point = {};
    if (!reflection::intersect_segment_plane(
            line_start, line_end, plane_point, plane_normal, 1e-7f, kEpcTolerance,
            shared_point)) {
        return false;
    }
    point = shared_point;
    return is_finite(point.x) && is_finite(point.y) && is_finite(point.z);
}

RAYD_HOST_DEVICE void store_invalid(
    const ReflEpcParams &params,
    unsigned int ray_index,
    int bounce_count,
    int first_blocked_segment,
    int first_blocked_prim,
    int first_blocked_group) {
    params.out_valid[ray_index] = 0u;
    params.out_bounce_count[ray_index] = bounce_count;
    params.out_path_length[ray_index] = uint_as_float(0x7f800000u);
    params.out_first_blocked_segment[ray_index] = first_blocked_segment;
    params.out_first_blocked_prim[ray_index] = first_blocked_prim;
    params.out_first_blocked_group[ray_index] = first_blocked_group;
}

/// Reflector scene trace (former trace_scene): closest hit against the primary and,
/// when split_mode is on, the secondary acceleration structure. Uses the Traverser's
/// trace_closest, which the OptiX shim maps to the reflection-mode ray cast.
template <typename Config>
RAYD_DEVICE ::rayd::shared::rt::TriangleHit trace_scene(
    const ReflEpcParams &params,
    const typename Config::Traverser &primary,
    const typename Config::Traverser &secondary,
    Vec3f origin,
    Vec3f direction,
    float tmax) {
    const ::rayd::shared::rt::TriangleHit hit_primary =
        primary.trace_closest(origin, direction, kTraceTMin, tmax);
    if (params.split_mode == 0) {
        return hit_primary;
    }
    const ::rayd::shared::rt::TriangleHit hit_secondary =
        secondary.trace_closest(origin, direction, kTraceTMin, tmax);
    return choose_nearest_hit(hit_primary, hit_secondary);
}

struct VisibilityResult {
    std::uint32_t visible;
    int blocker;
};

/// Segment visibility (former trace_visibility / trace_visibility_primary): the
/// active/degenerate guard is algorithm semantics; the occlusion cast goes through
/// the Traverser's trace_first_blocker (visibility-mode ray cast with the ignore
/// filter). With PrimaryOnly the secondary structure is never consulted; otherwise
/// an unoccluded primary and split_mode fall through to the secondary. The blocker
/// is resolved to its global prim id via global_primitive_id (a pass-through when
/// the traverser already reports a global prim with shape -1).
template <typename Config, bool PrimaryOnly>
RAYD_DEVICE VisibilityResult trace_visibility_segment(
    const ReflEpcParams &params,
    const typename Config::Traverser &primary,
    const typename Config::Traverser &secondary,
    Vec3f start,
    Vec3f end,
    int ignore0,
    int ignore1,
    int ignore2) {
    Vec3f direction = math::subtract(end, start);
    const float length = length3(direction);
    if (length <= kMinSegmentLength) {
        return {1u, -1};
    }
    direction = math::scale(direction, 1.0f / length);
    const Vec3f origin = math::add(start, math::scale(direction, kRayBias));
    const float tmax = fmaxf(length - 2.0f * kRayBias, 0.0f);
    const std::int32_t ignore[3] = {ignore0, ignore1, ignore2};

    ::rayd::shared::rt::TriangleHit result =
        primary.trace_first_blocker(origin, direction, kTraceTMin, tmax, ignore, 3);
    if constexpr (!PrimaryOnly) {
        if (result.hit == 0u && params.split_mode != 0) {
            result = secondary.trace_first_blocker(origin, direction, kTraceTMin, tmax, ignore, 3);
        }
    }

    VisibilityResult out;
    out.visible = result.hit == 0u ? 1u : 0u;
    out.blocker = result.hit != 0u
        ? global_primitive_id(result.instance, result.prim, params.face_offsets, params.n_meshes)
        : -1;
    return out;
}

} // namespace reflection_epc_algo_detail

/// Reflection-EPC discovery for one lane (former run_reflection_epc_raygen). Traces
/// the expected reflector sequence (or applies a supplied plane sequence in direct
/// mode), runs the shared fixed-winner back-trace, freezes containment, and checks
/// segment visibility to the receiver, writing per-slot geometry and per-ray
/// validity. `primary` / `secondary` are Config::Traverser oracles over the two
/// acceleration structures.
template <typename Config, bool DirectOnly, bool PrimaryVisibilityOnly>
RAYD_DEVICE void run_reflection_epc_algo(
    const ::rayd::shared::optix::ReflEpcParams &params,
    std::uint32_t ray_index,
    const typename Config::Traverser &primary,
    const typename Config::Traverser &secondary) {
    using namespace reflection_epc_algo_detail;
    namespace reflection = ::rayd::shared::reflection;

    if (ray_index >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const int B = params.max_bounces;
    const int base = static_cast<int>(ray_index) * B;
    for (int bounce = 0; bounce < B; ++bounce) {
        const int slot = base + bounce;
        params.out_point_x[slot] = 0.0f;
        params.out_point_y[slot] = 0.0f;
        params.out_point_z[slot] = 0.0f;
        params.out_trace_prim_ids[slot] = -1;
        params.out_resolved_prim_ids[slot] = -1;
        params.out_surface_group_ids[slot] = -1;
        params.out_plane_normal_x[slot] = 0.0f;
        params.out_plane_normal_y[slot] = 0.0f;
        params.out_plane_normal_z[slot] = 0.0f;
    }

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        store_invalid(params, ray_index, 0, -1, -1, -1);
        return;
    }

    Vec3f origin = math::make_vec3(
        params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    const int rx_id = params.rx_count == 1 ? 0 : static_cast<int>(ray_index);
    const Vec3f receiver =
        math::make_vec3(params.rx_x[rx_id], params.rx_y[rx_id], params.rx_z[rx_id]);

    Vec3f plane_points[ReflEpcMaxBounces];
    Vec3f plane_normals[ReflEpcMaxBounces];
    int trace_prim_ids[ReflEpcMaxBounces];
    int resolved_prim_ids[ReflEpcMaxBounces];
    int surface_group_ids[ReflEpcMaxBounces];
    Vec3f image_sources[ReflEpcMaxBounces + 1];
    Vec3f reflection_points[ReflEpcMaxBounces];
    image_sources[0] = origin;

    int bounce_count = 0;
    Vec3f image_source = origin;

    if (direct_plane_mode(params)) {
        for (int bounce = 0; bounce < B; ++bounce) {
            const int slot = base + bounce;
            const int expected_prim = expected_prim_for_bounce(params, slot);
            const int expected_group = surface_group_for_prim(params, expected_prim);
            if (expected_prim < 0 ||
                expected_prim >= params.n_triangles ||
                !is_finite(params.direct_plane_point_x[slot]) ||
                !is_finite(params.direct_plane_point_y[slot]) ||
                !is_finite(params.direct_plane_point_z[slot]) ||
                !is_finite(params.direct_plane_normal_x[slot]) ||
                !is_finite(params.direct_plane_normal_y[slot]) ||
                !is_finite(params.direct_plane_normal_z[slot])) {
                store_invalid(params, ray_index, bounce_count, -1, -1, -1);
                return;
            }

            const Vec3f plane_point = math::make_vec3(
                params.direct_plane_point_x[slot],
                params.direct_plane_point_y[slot],
                params.direct_plane_point_z[slot]);
            const Vec3f plane_normal = normalize3(math::make_vec3(
                params.direct_plane_normal_x[slot],
                params.direct_plane_normal_y[slot],
                params.direct_plane_normal_z[slot]));
            if (length3(plane_normal) <= 0.0f) {
                store_invalid(params, ray_index, bounce_count, -1, -1, -1);
                return;
            }

            image_source = reflection::reflect_point_across_plane(
                image_source, plane_point, plane_normal);
            image_sources[bounce + 1] = image_source;
            plane_points[bounce] = plane_point;
            plane_normals[bounce] = plane_normal;
            trace_prim_ids[bounce] = expected_prim;
            resolved_prim_ids[bounce] = -1;
            surface_group_ids[bounce] = expected_group;
            ++bounce_count;
        }
    } else {
        if constexpr (DirectOnly) {
            store_invalid(params, ray_index, bounce_count, -1, -1, -1);
            return;
        } else {
            Vec3f trace_origin = origin;
            Vec3f trace_direction = normalize3(math::make_vec3(
                params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]));

            for (int bounce = 0; bounce < B; ++bounce) {
                const float tmax_input =
                    bounce == 0 && params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kTraceTMax;
                const float trace_tmax = is_finite(tmax_input) ? tmax_input : kTraceTMax;
                const ::rayd::shared::rt::TriangleHit hit =
                    trace_scene<Config>(params, primary, secondary, trace_origin, trace_direction, trace_tmax);
                if (hit.hit == 0u) {
                    break;
                }

                const int shape_id = hit.instance;
                const int local_prim = hit.prim;
                const int global_prim = global_primitive_id(
                    shape_id, local_prim, params.face_offsets, params.n_meshes);
                const int actual_group = surface_group_for_prim(params, global_prim);
                const int slot = base + bounce;
                const int expected_prim = expected_prim_for_bounce(params, slot);
                const int expected_group = surface_group_for_prim(params, expected_prim);
                const bool expected_matches =
                    expected_prim < 0 ||
                    (has_surface_groups(params)
                         ? (actual_group >= 0 && actual_group == expected_group)
                         : (global_prim == expected_prim));
                if (!expected_matches) {
                    store_invalid(params, ray_index, bounce_count, -1, -1, -1);
                    return;
                }
                const float bary_u = hit.bary_u;
                const float bary_v = hit.bary_v;
                const float t = hit.t;

                Vec3f hit_point = math::add(trace_origin, math::scale(trace_direction, t));
                Vec3f geo_normal = math::make_vec3(0.0f, 0.0f, 1.0f);
                if (global_prim >= 0 && global_prim < params.n_triangles) {
                    const Vec3f p0 = load_triangle_p0(params, global_prim);
                    const Vec3f e1 = load_triangle_e1(params, global_prim);
                    const Vec3f e2 = load_triangle_e2(params, global_prim);
                    hit_point = math::add(
                        math::add(p0, math::scale(e1, bary_u)), math::scale(e2, bary_v));
                    geo_normal = load_triangle_normal(params, global_prim);
                }
                if (math::dot(trace_direction, geo_normal) > 0.0f) {
                    geo_normal = math::scale(geo_normal, -1.0f);
                }

                const float image_distance =
                    math::dot(math::subtract(image_source, hit_point), geo_normal);
                image_source = math::subtract(
                    image_source, math::scale(geo_normal, 2.0f * image_distance));
                image_sources[bounce + 1] = image_source;
                plane_points[bounce] = hit_point;
                plane_normals[bounce] = geo_normal;
                trace_prim_ids[bounce] = global_prim;
                resolved_prim_ids[bounce] = -1;
                surface_group_ids[bounce] =
                    expected_group >= 0 ? expected_group : actual_group;
                ++bounce_count;

                const float ray_dot_normal = math::dot(trace_direction, geo_normal);
                trace_direction = normalize3(math::subtract(
                    trace_direction, math::scale(geo_normal, 2.0f * ray_dot_normal)));
                trace_origin = math::add(hit_point, math::scale(trace_direction, kRayBias));
            }
        }
    }

    if (bounce_count != B) {
        store_invalid(params, ray_index, bounce_count, -1, -1, -1);
        return;
    }

    // Shared fixed-winner back-trace and path length (reflection/epc_chain.h): the
    // planes and image sources were built per discovery mode above; the geometry
    // from here on is mode-independent and everything is already math::Vec3f, so it
    // feeds epc_backtrace_and_length with no conversion.
    float path_length = 0.0f;
    if (!reflection::epc_backtrace_and_length<ReflEpcMaxBounces>(
            plane_points, plane_normals, image_sources, B,
            origin, receiver, reflection_points,
            nullptr, nullptr, path_length)) {
        store_invalid(params, ray_index, bounce_count, -1, -1, -1);
        return;
    }

    // Freeze which primitive each interaction lands in (the discrete winner). The
    // back-trace above is pure geometry, containment is the discovery decision, so
    // they are separated. resolved_prim_ids feeds the visibility ignore lists below
    // and must be fully populated before them.
    for (int bounce = B - 1; bounce >= 0; --bounce) {
        const Vec3f point = reflection_points[bounce];
        int resolved_prim = -1;
        bool inside;
        if (has_surface_groups(params) && surface_group_ids[bounce] >= 0) {
            inside = point_inside_surface_group(
                params, surface_group_ids[bounce], point, resolved_prim);
        } else {
            const int expected_prim = expected_prim_for_bounce(params, base + bounce);
            const int containment_prim =
                expected_prim >= 0 ? expected_prim : trace_prim_ids[bounce];
            inside = point_inside_triangle(params, containment_prim, point);
            resolved_prim = inside ? containment_prim : -1;
        }
        resolved_prim_ids[bounce] = resolved_prim;
        if (!inside) {
            store_invalid(params, ray_index, bounce_count, -1, -1, -1);
            return;
        }
    }

    bool valid = true;
    int first_blocked_segment = -1;
    int first_blocked_prim = -1;
    int first_blocked_group = -1;
    const int final_ignore_group =
        final_ignore_group_for_ray(params, static_cast<int>(ray_index));
    for (int segment = 0; segment <= B; ++segment) {
        const Vec3f start = segment == 0 ? origin : reflection_points[segment - 1];
        const Vec3f end = segment == B ? receiver : reflection_points[segment];

        const bool ignore_surface_group =
            params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup &&
            has_surface_groups(params);
        const int ignore0 = segment > 0
                                ? (ignore_surface_group
                                       ? surface_group_ids[segment - 1]
                                       : resolved_prim_ids[segment - 1])
                                : -1;
        const int ignore1 = segment < B
                                ? (ignore_surface_group
                                       ? surface_group_ids[segment]
                                       : resolved_prim_ids[segment])
                                : -1;
        const int ignore2 =
            ignore_surface_group && segment == B ? final_ignore_group : -1;
        VisibilityResult visibility;
        if constexpr (PrimaryVisibilityOnly) {
            visibility = trace_visibility_segment<Config, true>(
                params, primary, secondary, start, end, ignore0, ignore1, ignore2);
        } else {
            visibility = trace_visibility_segment<Config, false>(
                params, primary, secondary, start, end, ignore0, ignore1, ignore2);
        }
        if (visibility.visible == 0u) {
            first_blocked_segment = segment;
            first_blocked_prim = visibility.blocker;
            first_blocked_group = surface_group_for_prim(params, first_blocked_prim);
            valid = false;
            break;
        }
    }

    if (!valid) {
        store_invalid(params,
                      ray_index,
                      bounce_count,
                      first_blocked_segment,
                      first_blocked_prim,
                      first_blocked_group);
        return;
    }

    params.out_valid[ray_index] = 1u;
    params.out_bounce_count[ray_index] = bounce_count;
    params.out_path_length[ray_index] = path_length;
    params.out_first_blocked_segment[ray_index] = -1;
    params.out_first_blocked_prim[ray_index] = -1;
    params.out_first_blocked_group[ray_index] = -1;
    for (int bounce = 0; bounce < B; ++bounce) {
        const int slot = base + bounce;
        params.out_point_x[slot] = reflection_points[bounce].x;
        params.out_point_y[slot] = reflection_points[bounce].y;
        params.out_point_z[slot] = reflection_points[bounce].z;
        params.out_trace_prim_ids[slot] = trace_prim_ids[bounce];
        params.out_resolved_prim_ids[slot] = resolved_prim_ids[bounce];
        params.out_surface_group_ids[slot] = surface_group_ids[bounce];
        params.out_plane_normal_x[slot] = plane_normals[bounce].x;
        params.out_plane_normal_y[slot] = plane_normals[bounce].y;
        params.out_plane_normal_z[slot] = plane_normals[bounce].z;
    }
}

} // namespace rayd::shared::multipath
