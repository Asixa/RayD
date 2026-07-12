#pragma once

#include <optix.h>
#include <optix_device.h>

#include <rayd/shared/contracts.h>
#include <rayd/shared/field_math.h>

namespace rayd::shared::multipath::reflection_accumulation {

inline constexpr float TraceTMin = 1.0e-5f;
inline constexpr float TraceTMax = 1.0e8f;
inline constexpr float RayBias = 1.0e-5f;
inline constexpr float Epsilon = shared::SmallEpsilon;
inline constexpr float SpeedOfLight = shared::SpeedOfLight;
inline constexpr float Pi = 3.14159265358979323846f;

using field::Complex;
using field::Complex3;
using field::c_make;
using field::c_mul;
using field::c_scale;
using field::c3_add;
using field::c3_dot_real;
using field::c3_from_real;
using field::c3_mul_complex;
using field::c3_power;
using field::c3_scale_complex;
using field::c3_zero;

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int bary_u = 0u;
    unsigned int bary_v = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

static __forceinline__ __device__ float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float s, float3 value) {
    return make_float3(s * value.x, s * value.y, s * value.z);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
    return make_vec3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float norm3(float3 value) {
    return sqrtf(fmaxf(dot3(value, value), 0.0f));
}

static __forceinline__ __device__ float3 normalize3(float3 value) {
    return rsqrtf(fmaxf(dot3(value, value), 1.0e-12f)) * value;
}

static __forceinline__ __device__ float3 fallback_axis(float3 direction) {
    return fabsf(direction.z) < 0.9f
        ? make_vec3(0.0f, 0.0f, 1.0f)
        : make_vec3(0.0f, 1.0f, 0.0f);
}

static __forceinline__ __device__ float3 stable_perpendicular(
    float3 direction,
    float3 preferred) {
    const float3 normalized_direction = normalize3(direction);
    float3 projected = preferred - dot3(preferred, normalized_direction) * normalized_direction;
    if (dot3(projected, projected) > 1.0e-12f)
        return normalize3(projected);
    const float3 axis = fallback_axis(normalized_direction);
    projected = axis - dot3(axis, normalized_direction) * normalized_direction;
    return normalize3(projected);
}

static __forceinline__ __device__ float max_abs_component(float3 value) {
    return fmaxf(fabsf(value.x), fmaxf(fabsf(value.y), fabsf(value.z)));
}

static __forceinline__ __device__ float3 offset_surface_point(
    float3 point,
    float3 direction,
    float3 normal) {
    const float offset = RayBias * (1.0f + max_abs_component(point));
    const float signed_offset = dot3(direction, normal) >= 0.0f ? offset : -offset;
    return point + signed_offset * normal;
}

static __forceinline__ __device__ void clear_payload(HitPayload &payload) {
    payload.hit = 0u;
    payload.t = __float_as_uint(TraceTMax);
    payload.bary_u = 0u;
    payload.bary_v = 0u;
    payload.prim = 0u;
    payload.instance = 0u;
}

static __forceinline__ __device__ void set_payload(const HitPayload &payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.bary_u);
    optixSetPayload_3(payload.bary_v);
    optixSetPayload_4(payload.prim);
    optixSetPayload_5(payload.instance);
}

static __forceinline__ __device__ void trace_handle(
    OptixTraversableHandle handle,
    float3 origin,
    float3 direction,
    float tmax,
    HitPayload &payload) {
    clear_payload(payload);
    if (handle == 0ull)
        return;
    optixTrace(handle,
               origin,
               direction,
               TraceTMin,
               tmax,
               0.0f,
               255u,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0,
               1,
               0,
               payload.hit,
               payload.t,
               payload.bary_u,
               payload.bary_v,
               payload.prim,
               payload.instance);
}

static __forceinline__ __device__ HitPayload choose_hit(
    const HitPayload &primary,
    const HitPayload &secondary) {
    if (primary.hit == 0u)
        return secondary;
    if (secondary.hit == 0u)
        return primary;
    return __uint_as_float(secondary.t) < __uint_as_float(primary.t)
        ? secondary
        : primary;
}

template <typename Params>
static __forceinline__ __device__ HitPayload trace_scene(
    const Params &params,
    float3 origin,
    float3 direction,
    float tmax) {
    HitPayload primary;
    trace_handle(params.primary_handle, origin, direction, tmax, primary);
    if (params.split_mode == 0)
        return primary;
    HitPayload secondary;
    trace_handle(params.secondary_handle, origin, direction, tmax, secondary);
    return choose_hit(primary, secondary);
}

static __forceinline__ __device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

static __forceinline__ __device__ void plane_coords(
    float3 value,
    int axis,
    float &coord0,
    float &coord1) {
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

static __forceinline__ __device__ float3 axis_plane_point(
    int axis,
    float position,
    float coord0,
    float coord1) {
    if (axis == 0)
        return make_vec3(position, coord0, coord1);
    if (axis == 1)
        return make_vec3(coord0, position, coord1);
    return make_vec3(coord0, coord1, position);
}

static __forceinline__ __device__ unsigned int hash_u32(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

static __forceinline__ __device__ float uniform01(
    unsigned int ray_index,
    unsigned int depth,
    unsigned int seed) {
    const unsigned int hash = hash_u32(ray_index ^ (depth * 0x9e3779b9u) ^ seed);
    return static_cast<float>(hash & 0x00ffffffu) * (1.0f / 16777216.0f);
}

template <typename Params>
static __forceinline__ __device__ bool material_reflection_coefficients(
    const Params &params,
    int global_primitive,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    r_te = c_make(0.0f, 0.0f);
    r_tm = c_make(0.0f, 0.0f);
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
static __forceinline__ __device__ Complex3 reflect_field_vector(
    const Params &params,
    Complex3 field_value,
    float3 incident_direction,
    float3 normal,
    int global_primitive,
    float3 &reflected_direction) {
    const float3 incident_hat = normalize3(incident_direction);
    const float3 normal_hat = normalize3(normal);
    const float direction_dot_normal = dot3(incident_hat, normal_hat);
    reflected_direction = normalize3(
        incident_hat - 2.0f * direction_dot_normal * normal_hat);

    float3 s_hat = cross3(normal_hat, incident_hat);
    s_hat = dot3(s_hat, s_hat) <= 1.0e-12f
        ? stable_perpendicular(incident_hat, normal_hat)
        : normalize3(s_hat);
    float3 p_in_hat = cross3(s_hat, incident_hat);
    p_in_hat = dot3(p_in_hat, p_in_hat) <= 1.0e-12f
        ? stable_perpendicular(incident_hat, normal_hat)
        : normalize3(p_in_hat);
    float3 p_out_hat = cross3(s_hat, reflected_direction);
    p_out_hat = dot3(p_out_hat, p_out_hat) <= 1.0e-12f
        ? stable_perpendicular(reflected_direction, normal_hat)
        : normalize3(p_out_hat);

    Complex r_te;
    Complex r_tm;
    if (!material_reflection_coefficients(
            params, global_primitive, fabsf(direction_dot_normal), r_te, r_tm))
        return c3_zero();
    const Complex e_s = c3_dot_real(field_value, s_hat);
    const Complex e_p = c3_dot_real(field_value, p_in_hat);
    return c3_add(c3_scale_complex(s_hat, c_mul(r_te, e_s)),
                  c3_scale_complex(p_out_hat, c_mul(r_tm, e_p)));
}

template <typename Params>
static __forceinline__ __device__ void store_wedge_event(
    const Params &params,
    unsigned int ray_index,
    int depth,
    int global_primitive,
    float3 hit_point,
    float3 normal,
    float3 incident_direction,
    float3 source_point,
    float source_power,
    float3 initial_direction) {
    if (params.collect_wedges == 0 || params.out_wedge_count == nullptr)
        return;
    if (depth > 0 && params.collect_wedge_prefixes == 0)
        return;
    float stored_source_power = source_power;
    const int sample_stride = max(params.wedge_sample_stride, 1);
    if (params.collect_wedge_prefixes != 0 && sample_stride > 1) {
        const unsigned int max_prefix_depth =
            static_cast<unsigned int>(max(params.max_bounces, 1));
        const unsigned int ordinal =
            ray_index * max_prefix_depth + static_cast<unsigned int>(depth);
        const unsigned int phase =
            static_cast<unsigned int>(params.seed) % static_cast<unsigned int>(sample_stride);
        if ((ordinal + phase) % static_cast<unsigned int>(sample_stride) != 0u)
            return;
        stored_source_power *= static_cast<float>(sample_stride);
    }

    const int slot = atomicAdd(params.out_wedge_count, 1);
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
static __forceinline__ __device__ bool accumulate_plane(
    const Params &params,
    unsigned int ray_index,
    int depth,
    float3 origin,
    float3 direction,
    float blocker_t,
    float3 image_source,
    Complex3 field_value) {
    if (!Policy::include_depth(params, depth) || c3_power(field_value) <= 0.0f)
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

    const float3 target = origin + t_plane * direction;
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
    const int ix = min(max(static_cast<int>(u * params.grid_resolution0), 0),
                       params.grid_resolution0 - 1);
    const int iy = min(max(static_cast<int>(v * params.grid_resolution1), 0),
                       params.grid_resolution1 - 1);
    const int cell = iy * params.grid_resolution0 + ix;

    const float3 target_plane = axis_plane_point(axis, params.grid_position, coord0, coord1);
    const float unfolded_distance = norm3(target_plane - image_source);
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
    const Complex coefficient = c_scale(phase, amplitude_scale);
    const Complex3 contribution_field = c3_mul_complex(field_value, coefficient);
    if (!field::finite_complex3(contribution_field))
        return false;
    const float contribution_power = c3_power(contribution_field);
    if (!(contribution_power > 0.0f) || !isfinite(contribution_power))
        return false;
    Policy::commit(
        params, ray_index, depth, cell, contribution_field, contribution_power);
    return true;
}

static __forceinline__ __device__ void closest_hit() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    const float2 barycentrics = optixGetTriangleBarycentrics();
    payload.bary_u = __float_as_uint(barycentrics.x);
    payload.bary_v = __float_as_uint(barycentrics.y);
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

static __forceinline__ __device__ void miss() {
    optixSetPayload_0(0u);
}

template <typename Params, typename Policy>
static __forceinline__ __device__ void raygen(const Params &params) {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;
    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u)
        return;

    float3 origin = make_vec3(
        params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    float3 direction = normalize3(make_vec3(
        params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]));
    const float3 initial_direction = direction;
    float3 image_source = make_vec3(
        params.tx_x[ray_index], params.tx_y[ray_index], params.tx_z[ray_index]);
    const float3 tx_polarization = make_vec3(
        params.tx_pol_x[ray_index], params.tx_pol_y[ray_index], params.tx_pol_z[ray_index]);
    float3 transverse_polarization =
        tx_polarization - dot3(tx_polarization, direction) * direction;
    transverse_polarization = dot3(transverse_polarization, transverse_polarization) <= 1.0e-12f
        ? stable_perpendicular(direction, tx_polarization)
        : normalize3(transverse_polarization);
    Complex3 field_value = c3_from_real(transverse_polarization);
    float path_length = 0.0f;

    for (int depth = 0; depth <= params.max_bounces; ++depth) {
        const float tmax_input = depth == 0 && params.ray_tmax != nullptr
            ? params.ray_tmax[ray_index]
            : TraceTMax;
        const float trace_tmax = isfinite(tmax_input) ? tmax_input : TraceTMax;
        const HitPayload hit = trace_scene(params, origin, direction, trace_tmax);
        const float blocker_t = hit.hit != 0u ? __uint_as_float(hit.t) : TraceTMax;

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
        const float bary_u = __uint_as_float(hit.bary_u);
        const float bary_v = __uint_as_float(hit.bary_v);

        float3 hit_point = origin + blocker_t * direction;
        float3 geometric_normal = make_vec3(0.0f, 0.0f, 1.0f);
        if (global_primitive >= 0 && global_primitive < params.n_triangles) {
            hit_point = make_vec3(
                params.tri_p0_x[global_primitive] + bary_u * params.tri_e1_x[global_primitive] +
                    bary_v * params.tri_e2_x[global_primitive],
                params.tri_p0_y[global_primitive] + bary_u * params.tri_e1_y[global_primitive] +
                    bary_v * params.tri_e2_y[global_primitive],
                params.tri_p0_z[global_primitive] + bary_u * params.tri_e1_z[global_primitive] +
                    bary_v * params.tri_e2_z[global_primitive]);
            geometric_normal = normalize3(make_vec3(
                params.tri_fn_x[global_primitive],
                params.tri_fn_y[global_primitive],
                params.tri_fn_z[global_primitive]));
        }
        if (dot3(direction, geometric_normal) > 0.0f)
            geometric_normal = -1.0f * geometric_normal;

        float3 reflected_direction;
        const float source_power = c3_power(field_value) * params.solid_angle_per_ray;
        const Complex3 reflected_field = reflect_field_vector(
            params,
            field_value,
            direction,
            geometric_normal,
            global_primitive,
            reflected_direction);
        if (c3_power(reflected_field) <= 0.0f)
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

        const float image_distance = dot3(image_source - hit_point, geometric_normal);
        image_source = image_source - 2.0f * image_distance * geometric_normal;
        path_length += blocker_t;
        field_value = reflected_field;
        direction = reflected_direction;
        origin = offset_surface_point(hit_point, direction, geometric_normal);

        const int next_depth = depth + 1;
        if (params.rr_depth > 0 && params.rr_prob < 1.0f && next_depth >= params.rr_depth) {
            const float field_power = c3_power(field_value);
            const float continue_probability =
                fminf(fmaxf(field_power, 1.0e-8f), fmaxf(params.rr_prob, 1.0e-8f));
            if (uniform01(ray_index,
                          static_cast<unsigned int>(next_depth),
                          static_cast<unsigned int>(params.seed)) >= continue_probability)
                break;
            const float roulette_scale = rsqrtf(fmaxf(continue_probability, 1.0e-8f));
            field_value.x = c_scale(field_value.x, roulette_scale);
            field_value.y = c_scale(field_value.y, roulette_scale);
            field_value.z = c_scale(field_value.z, roulette_scale);
        }

        if (params.stop_threshold > 0.0f) {
            const float fspl = field::free_space_amplitude(
                params.wavelength, path_length, Epsilon);
            if (c3_power(field_value) * fspl * fspl <= params.stop_threshold)
                break;
        }
    }
}

} // namespace rayd::shared::multipath::reflection_accumulation
