#pragma once

#include <cmath>
#include <cstdint>

#include <rayd/detail/contracts.h>
#include <rayd/detail/field_math.h>
#include <rayd/detail/vec3.h>
#include <rayd/detail/rt/numeric_policy.h>
#include <rayd/detail/rt/qualifiers.h>
#include <rayd/detail/rt/traverser.h>

// Host-compilable reflection-accumulation algorithm. This is the de-CUDA-ised
// body of the former reflection_accumulation::raygen: math is math::Vec3f
// throughout (mirroring the exact arithmetic op order of the old local CUDA
// vector helpers so device codegen stays bit-identical), the closest-hit ray
// cast goes through an rt::is_traverser Traverser (so no OptiX ray-cast
// intrinsic, payload register, or launch-index query appears here), and the lane
// index is a plain parameter. The local 6-field HitPayload dissolves into
// rt::TriangleHit. accumulation_optix_device.cuh instantiates it with the
// shared OptixTraverser; the CUDA fused executor (P4d) will reuse it with
// CudaBvhTraverser. The wedge-event slot reservation and the grid commit remain
// the caller's Policy responsibility (device atomics); the host atomic_add
// fallback below only exists so this header parses under a pure host compiler.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace reflection_accumulation_algo_detail {

using math::Vec3f;
namespace field = ::rayd::shared::field;
using field::Complex;
using field::Complex3;

inline constexpr float TraceTMin = 1.0e-5f;
inline constexpr float TraceTMax = 1.0e8f;
inline constexpr float RayBias = 1.0e-5f;
inline constexpr float Epsilon = shared::SmallEpsilon;
inline constexpr float SpeedOfLight = shared::SpeedOfLight;
inline constexpr float Pi = 3.14159265358979323846f;

static_assert(TraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(TraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(RayBias == ::rayd::shared::rt::kMultipathRayBias);
// This family clears missed hits to TraceTMax rather than +inf.
static_assert(TraceTMax == ::rayd::shared::rt::kReflectionTraceMissDistance);

RAYD_HOST_DEVICE float reciprocal_sqrt(float value) {
#if defined(__CUDA_ARCH__)
    return rsqrtf(value);
#else
    return 1.0f / std::sqrt(value);
#endif
}

RAYD_HOST_DEVICE bool is_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// Integer min/max, the host-safe form of the device min()/max() builtins.
RAYD_HOST_DEVICE int imax(int a, int b) { return a > b ? a : b; }
RAYD_HOST_DEVICE int imin(int a, int b) { return a < b ? a : b; }

// atomicAdd on device; a non-atomic byte-equivalent on the host so the wedge
// slot reservation compiles off-device (the host path is never executed).
RAYD_HOST_DEVICE int atomic_add(int *address, int value) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    const int old = *address;
    *address += value;
    return old;
#endif
}

RAYD_HOST_DEVICE float dot3(Vec3f a, Vec3f b) { return math::dot(a, b); }

RAYD_HOST_DEVICE float norm3(Vec3f value) {
    return sqrtf(fmaxf(math::dot(value, value), 0.0f));
}

RAYD_HOST_DEVICE Vec3f normalize3(Vec3f value) {
    return math::scale(value, reciprocal_sqrt(fmaxf(math::dot(value, value), 1.0e-12f)));
}

RAYD_HOST_DEVICE Vec3f fallback_axis(Vec3f direction) {
    return fabsf(direction.z) < 0.9f
        ? math::make_vec3(0.0f, 0.0f, 1.0f)
        : math::make_vec3(0.0f, 1.0f, 0.0f);
}

RAYD_HOST_DEVICE Vec3f stable_perpendicular(Vec3f direction, Vec3f preferred) {
    const Vec3f normalized_direction = normalize3(direction);
    Vec3f projected = math::subtract(
        preferred, math::scale(normalized_direction, dot3(preferred, normalized_direction)));
    if (dot3(projected, projected) > 1.0e-12f)
        return normalize3(projected);
    const Vec3f axis = fallback_axis(normalized_direction);
    projected = math::subtract(axis, math::scale(normalized_direction, dot3(axis, normalized_direction)));
    return normalize3(projected);
}

RAYD_HOST_DEVICE float max_abs_component(Vec3f value) {
    return fmaxf(fabsf(value.x), fmaxf(fabsf(value.y), fabsf(value.z)));
}

RAYD_HOST_DEVICE Vec3f offset_surface_point(Vec3f point, Vec3f direction, Vec3f normal) {
    const float offset = RayBias * (1.0f + max_abs_component(point));
    const float signed_offset = dot3(direction, normal) >= 0.0f ? offset : -offset;
    return math::add(point, math::scale(normal, signed_offset));
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_hit(
    ::rayd::shared::rt::TriangleHit primary,
    ::rayd::shared::rt::TriangleHit secondary) {
    if (primary.hit == 0u)
        return secondary;
    if (secondary.hit == 0u)
        return primary;
    return secondary.t < primary.t ? secondary : primary;
}

template <typename Traverser>
RAYD_DEVICE ::rayd::shared::rt::TriangleHit trace_scene(
    int split_mode,
    const Traverser &primary,
    const Traverser &secondary,
    Vec3f origin,
    Vec3f direction,
    float tmax) {
    const ::rayd::shared::rt::TriangleHit primary_hit =
        primary.trace_closest(origin, direction, TraceTMin, tmax);
    if (split_mode == 0)
        return primary_hit;
    const ::rayd::shared::rt::TriangleHit secondary_hit =
        secondary.trace_closest(origin, direction, TraceTMin, tmax);
    return choose_hit(primary_hit, secondary_hit);
}

RAYD_HOST_DEVICE float component(Vec3f value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

RAYD_HOST_DEVICE void plane_coords(Vec3f value, int axis, float &coord0, float &coord1) {
    if (axis == 0) {
        coord0 = value.y;
        coord1 = value.z;
    } else if (axis == 1) {
        coord0 = value.x;
        coord1 = value.z;
    } else {
        coord0 = value.x;
        coord1 = value.y;
    }
}

RAYD_HOST_DEVICE Vec3f axis_plane_point(int axis, float position, float coord0, float coord1) {
    if (axis == 0)
        return math::make_vec3(position, coord0, coord1);
    if (axis == 1)
        return math::make_vec3(coord0, position, coord1);
    return math::make_vec3(coord0, coord1, position);
}

RAYD_HOST_DEVICE unsigned int hash_u32(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

RAYD_HOST_DEVICE float uniform01(unsigned int ray_index, unsigned int depth, unsigned int seed) {
    const unsigned int hash = hash_u32(ray_index ^ (depth * 0x9e3779b9u) ^ seed);
    return static_cast<float>(hash & 0x00ffffffu) * (1.0f / 16777216.0f);
}

template <typename Params>
RAYD_HOST_DEVICE bool material_reflection_coefficients(
    const Params &params,
    int global_primitive,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    r_te = field::c_make(0.0f, 0.0f);
    r_tm = field::c_make(0.0f, 0.0f);
    if (global_primitive < 0 || global_primitive >= params.material_count ||
        params.material_valid == nullptr || params.material_valid[global_primitive] == 0u)
        return false;
    const float omega = fmaxf(
        2.0f * Pi * SpeedOfLight / fmaxf(params.wavelength, Epsilon), Epsilon);
    return field::fresnel_reflection_coefficients(
        params.material_eta_r[global_primitive],
        params.material_sigma[global_primitive],
        params.material_mu_r[global_primitive],
        params.material_gain[global_primitive],
        omega,
        cos_theta,
        r_te,
        r_tm,
        Epsilon);
}

template <typename Params>
RAYD_HOST_DEVICE Complex3 reflect_field_vector(
    const Params &params,
    Complex3 field_value,
    Vec3f incident_direction,
    Vec3f normal,
    int global_primitive,
    Vec3f &reflected_direction) {
    const Vec3f incident_hat = normalize3(incident_direction);
    const Vec3f normal_hat = normalize3(normal);
    const float direction_dot_normal = dot3(incident_hat, normal_hat);
    reflected_direction = normalize3(
        math::subtract(incident_hat, math::scale(normal_hat, 2.0f * direction_dot_normal)));

    Vec3f s_hat = math::cross(normal_hat, incident_hat);
    s_hat = dot3(s_hat, s_hat) <= 1.0e-12f
        ? stable_perpendicular(incident_hat, normal_hat)
        : normalize3(s_hat);
    Vec3f p_in_hat = math::cross(s_hat, incident_hat);
    p_in_hat = dot3(p_in_hat, p_in_hat) <= 1.0e-12f
        ? stable_perpendicular(incident_hat, normal_hat)
        : normalize3(p_in_hat);
    Vec3f p_out_hat = math::cross(s_hat, reflected_direction);
    p_out_hat = dot3(p_out_hat, p_out_hat) <= 1.0e-12f
        ? stable_perpendicular(reflected_direction, normal_hat)
        : normalize3(p_out_hat);

    Complex r_te;
    Complex r_tm;
    if (!material_reflection_coefficients(
            params, global_primitive, fabsf(direction_dot_normal), r_te, r_tm))
        return field::c3_zero();
    const Complex e_s = field::c3_dot_real(field_value, s_hat);
    const Complex e_p = field::c3_dot_real(field_value, p_in_hat);
    return field::c3_add(field::c3_scale_complex(s_hat, field::c_mul(r_te, e_s)),
                         field::c3_scale_complex(p_out_hat, field::c_mul(r_tm, e_p)));
}

template <typename Params>
RAYD_HOST_DEVICE void store_wedge_event(
    const Params &params,
    unsigned int ray_index,
    int depth,
    int global_primitive,
    Vec3f hit_point,
    Vec3f normal,
    Vec3f incident_direction,
    Vec3f source_point,
    float source_power,
    Vec3f initial_direction) {
    if (params.collect_wedges == 0 || params.out_wedge_count == nullptr)
        return;
    if (depth > 0 && params.collect_wedge_prefixes == 0)
        return;
    float stored_source_power = source_power;
    const int sample_stride = imax(params.wedge_sample_stride, 1);
    if (params.collect_wedge_prefixes != 0 && sample_stride > 1) {
        const unsigned int max_prefix_depth =
            static_cast<unsigned int>(imax(params.max_bounces, 1));
        const unsigned int ordinal =
            ray_index * max_prefix_depth + static_cast<unsigned int>(depth);
        const unsigned int phase =
            static_cast<unsigned int>(params.seed) % static_cast<unsigned int>(sample_stride);
        if ((ordinal + phase) % static_cast<unsigned int>(sample_stride) != 0u)
            return;
        stored_source_power *= static_cast<float>(sample_stride);
    }

    const int slot = atomic_add(params.out_wedge_count, 1);
    if (slot < 0 || slot >= params.wedge_capacity)
        return;
    params.out_wedge_ray_index[slot] = static_cast<int>(ray_index);
    params.out_wedge_hit_x[slot] = hit_point.x;
    params.out_wedge_hit_y[slot] = hit_point.y;
    params.out_wedge_hit_z[slot] = hit_point.z;
    params.out_wedge_normal_x[slot] = normal.x;
    params.out_wedge_normal_y[slot] = normal.y;
    params.out_wedge_normal_z[slot] = normal.z;
    params.out_wedge_prim_id[slot] = global_primitive;
    params.out_wedge_dir_x[slot] = incident_direction.x;
    params.out_wedge_dir_y[slot] = incident_direction.y;
    params.out_wedge_dir_z[slot] = incident_direction.z;
    params.out_wedge_source_x[slot] = source_point.x;
    params.out_wedge_source_y[slot] = source_point.y;
    params.out_wedge_source_z[slot] = source_point.z;
    params.out_wedge_source_power[slot] = stored_source_power;
    params.out_wedge_initial_dir_x[slot] = initial_direction.x;
    params.out_wedge_initial_dir_y[slot] = initial_direction.y;
    params.out_wedge_initial_dir_z[slot] = initial_direction.z;
    params.out_wedge_bounce_depth[slot] = depth;
}

template <typename Params, typename Policy>
RAYD_HOST_DEVICE bool accumulate_plane(
    const Params &params,
    unsigned int ray_index,
    int depth,
    Vec3f origin,
    Vec3f direction,
    float blocker_t,
    Vec3f image_source,
    Complex3 field_value) {
    if (!Policy::include_depth(params, depth) || field::c3_power(field_value) <= 0.0f)
        return false;
    const int axis = params.grid_axis;
    const float axis_direction = component(direction, axis);
    if (fabsf(axis_direction) <= Epsilon)
        return false;
    const float safe_axis_direction =
        axis_direction + (axis_direction >= 0.0f ? Epsilon : -Epsilon);
    const float t_plane =
        (params.grid_position - component(origin, axis)) / safe_axis_direction;
    if (!(t_plane > RayBias && t_plane < blocker_t))
        return false;

    const Vec3f target = math::add(origin, math::scale(direction, t_plane));
    float coord0 = 0.0f;
    float coord1 = 0.0f;
    plane_coords(target, axis, coord0, coord1);
    if (coord0 < params.grid_coord0_min || coord0 >= params.grid_coord0_max ||
        coord1 < params.grid_coord1_min || coord1 >= params.grid_coord1_max)
        return false;
    const float span0 = params.grid_coord0_max - params.grid_coord0_min;
    const float span1 = params.grid_coord1_max - params.grid_coord1_min;
    if (span0 <= 0.0f || span1 <= 0.0f ||
        params.grid_resolution0 <= 0 || params.grid_resolution1 <= 0)
        return false;

    const float u = (coord0 - params.grid_coord0_min) / span0;
    const float v = (coord1 - params.grid_coord1_min) / span1;
    const int ix = imin(imax(static_cast<int>(u * params.grid_resolution0), 0),
                        params.grid_resolution0 - 1);
    const int iy = imin(imax(static_cast<int>(v * params.grid_resolution1), 0),
                        params.grid_resolution1 - 1);
    const int cell = iy * params.grid_resolution0 + ix;

    const Vec3f target_plane = axis_plane_point(axis, params.grid_position, coord0, coord1);
    const float unfolded_distance = norm3(math::subtract(target_plane, image_source));
    const float fspl = field::free_space_amplitude(
        params.wavelength, unfolded_distance, Epsilon);
    const float cos_theta = fmaxf(fabsf(axis_direction), Epsilon);
    const float geometry_power_scale =
        params.solid_angle_per_ray / fmaxf(params.cell_area, Epsilon) *
        unfolded_distance * unfolded_distance / cos_theta;
    const float amplitude_scale = fspl * sqrtf(fmaxf(geometry_power_scale, 0.0f));
    const float wave_number = fabsf(params.k) > Epsilon
        ? params.k
        : (2.0f * Pi / fmaxf(params.wavelength, Epsilon));
    const Complex phase = field::propagation_phase(wave_number, unfolded_distance);
    const Complex coefficient = field::c_scale(phase, amplitude_scale);
    const Complex3 contribution_field = field::c3_mul_complex(field_value, coefficient);
    if (!field::finite_complex3(contribution_field))
        return false;
    const float contribution_power = field::c3_power(contribution_field);
    if (!(contribution_power > 0.0f) || !is_finite(contribution_power))
        return false;
    Policy::commit(
        params, ray_index, depth, cell, contribution_field, contribution_power);
    return true;
}

} // namespace reflection_accumulation_algo_detail

/// Reflection-field accumulation for one lane (former reflection_accumulation::
/// raygen). `primary` / `secondary` are the Traverser oracles over the two
/// acceleration structures (secondary consulted only when params.split_mode !=
/// 0), and `ray_index` is this lane's ray id. `Policy` supplies the compile-time
/// include-depth predicate and the grid commit (device atomics).
template <typename Params, typename Policy, typename Traverser>
RAYD_DEVICE void reflection_accumulation_algo(
    const Params &params,
    std::uint32_t ray_index,
    const Traverser &primary,
    const Traverser &secondary) {
    using namespace reflection_accumulation_algo_detail;
    using math::Vec3f;
    using field::Complex3;
    using ::rayd::shared::rt::TriangleHit;

    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;
    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u)
        return;

    Vec3f origin = math::make_vec3(
        params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    Vec3f direction = normalize3(math::make_vec3(
        params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]));
    const Vec3f initial_direction = direction;
    Vec3f image_source = math::make_vec3(
        params.tx_x[ray_index], params.tx_y[ray_index], params.tx_z[ray_index]);
    const Vec3f tx_polarization = math::make_vec3(
        params.tx_pol_x[ray_index], params.tx_pol_y[ray_index], params.tx_pol_z[ray_index]);
    Vec3f transverse_polarization =
        math::subtract(tx_polarization, math::scale(direction, dot3(tx_polarization, direction)));
    transverse_polarization = dot3(transverse_polarization, transverse_polarization) <= 1.0e-12f
        ? stable_perpendicular(direction, tx_polarization)
        : normalize3(transverse_polarization);
    Complex3 field_value = field::c3_from_real(transverse_polarization);
    float path_length = 0.0f;

    for (int depth = 0; depth <= params.max_bounces; ++depth) {
        const float tmax_input = depth == 0 && params.ray_tmax != nullptr
            ? params.ray_tmax[ray_index]
            : TraceTMax;
        const float trace_tmax = is_finite(tmax_input) ? tmax_input : TraceTMax;
        const TriangleHit hit =
            trace_scene(params.split_mode, primary, secondary, origin, direction, trace_tmax);
        const float blocker_t = hit.hit != 0u ? hit.t : TraceTMax;

        accumulate_plane<Params, Policy>(
            params, ray_index, depth, origin, direction, blocker_t, image_source, field_value);
        if (hit.hit == 0u || depth >= params.max_bounces)
            break;

        const int shape_id = static_cast<int>(hit.instance);
        const int local_primitive = static_cast<int>(hit.prim);
        const int face_offset = shape_id >= 0 && shape_id < params.n_meshes
            ? params.face_offsets[shape_id]
            : 0;
        const int global_primitive = face_offset + local_primitive;
        const float bary_u = hit.bary_u;
        const float bary_v = hit.bary_v;

        Vec3f hit_point = math::add(origin, math::scale(direction, blocker_t));
        Vec3f geometric_normal = math::make_vec3(0.0f, 0.0f, 1.0f);
        if (global_primitive >= 0 && global_primitive < params.n_triangles) {
            hit_point = math::make_vec3(
                params.tri_p0_x[global_primitive] + bary_u * params.tri_e1_x[global_primitive] +
                    bary_v * params.tri_e2_x[global_primitive],
                params.tri_p0_y[global_primitive] + bary_u * params.tri_e1_y[global_primitive] +
                    bary_v * params.tri_e2_y[global_primitive],
                params.tri_p0_z[global_primitive] + bary_u * params.tri_e1_z[global_primitive] +
                    bary_v * params.tri_e2_z[global_primitive]);
            geometric_normal = normalize3(math::make_vec3(
                params.tri_fn_x[global_primitive],
                params.tri_fn_y[global_primitive],
                params.tri_fn_z[global_primitive]));
        }
        if (dot3(direction, geometric_normal) > 0.0f)
            geometric_normal = math::scale(geometric_normal, -1.0f);

        Vec3f reflected_direction;
        const float source_power = field::c3_power(field_value) * params.solid_angle_per_ray;
        const Complex3 reflected_field = reflect_field_vector(
            params,
            field_value,
            direction,
            geometric_normal,
            global_primitive,
            reflected_direction);
        if (field::c3_power(reflected_field) <= 0.0f)
            break;

        store_wedge_event(
            params,
            ray_index,
            depth,
            global_primitive,
            hit_point,
            geometric_normal,
            direction,
            image_source,
            source_power,
            initial_direction);

        const float image_distance = dot3(math::subtract(image_source, hit_point), geometric_normal);
        image_source = math::subtract(image_source, math::scale(geometric_normal, 2.0f * image_distance));
        path_length += blocker_t;
        field_value = reflected_field;
        direction = reflected_direction;
        origin = offset_surface_point(hit_point, direction, geometric_normal);

        const int next_depth = depth + 1;
        if (params.rr_depth > 0 && params.rr_prob < 1.0f && next_depth >= params.rr_depth) {
            const float field_power = field::c3_power(field_value);
            const float continue_probability =
                fminf(fmaxf(field_power, 1.0e-8f), fmaxf(params.rr_prob, 1.0e-8f));
            if (uniform01(ray_index,
                          static_cast<unsigned int>(next_depth),
                          static_cast<unsigned int>(params.seed)) >= continue_probability)
                break;
            const float roulette_scale = reciprocal_sqrt(fmaxf(continue_probability, 1.0e-8f));
            field_value.x = field::c_scale(field_value.x, roulette_scale);
            field_value.y = field::c_scale(field_value.y, roulette_scale);
            field_value.z = field::c_scale(field_value.z, roulette_scale);
        }

        if (params.stop_threshold > 0.0f) {
            const float fspl = field::free_space_amplitude(
                params.wavelength, path_length, Epsilon);
            if (field::c3_power(field_value) * fspl * fspl <= params.stop_threshold)
                break;
        }
    }
}

} // namespace rayd::shared::multipath
