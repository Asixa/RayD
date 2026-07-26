#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include "scene_internal.h"
#include <rayd/multipath/diffraction_accumulation_ad.h>
#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>
#include <rayd/trace/cuda_multipath_gpu.h>

#include "scene_multipath_internal.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct AccumRaw {
    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    int wedge_capacity = 0;
    Float reflection_power;
    Float field_x_re;
    Float field_x_im;
    Float field_y_re;
    Float field_y_im;
    Float field_z_re;
    Float field_z_im;
    Int reflection_count;
    Int wedge_count;
    Int wedge_ray_index;
    Float wedge_hit_x;
    Float wedge_hit_y;
    Float wedge_hit_z;
    Float wedge_normal_x;
    Float wedge_normal_y;
    Float wedge_normal_z;
    Int wedge_prim_id;
    Float wedge_dir_x;
    Float wedge_dir_y;
    Float wedge_dir_z;
    Float wedge_source_x;
    Float wedge_source_y;
    Float wedge_source_z;
    Float wedge_source_power;
    Float wedge_initial_dir_x;
    Float wedge_initial_dir_y;
    Float wedge_initial_dir_z;
    Int wedge_bounce_depth;
};

AccumRaw allocate_reflection_accumulation_raw(int ray_count,
                                                               int max_bounces,
                                                               int grid_cell_count,
                                                               int wedge_capacity) {
    AccumRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.grid_cell_count = grid_cell_count;
    raw.wedge_capacity = wedge_capacity;
    raw.reflection_power = empty<Float>(grid_cell_count);
    raw.field_x_re = empty<Float>(grid_cell_count);
    raw.field_x_im = empty<Float>(grid_cell_count);
    raw.field_y_re = empty<Float>(grid_cell_count);
    raw.field_y_im = empty<Float>(grid_cell_count);
    raw.field_z_re = empty<Float>(grid_cell_count);
    raw.field_z_im = empty<Float>(grid_cell_count);
    raw.reflection_count = empty<Int>(1);
    raw.wedge_count = empty<Int>(1);
    const int event_count = std::max(1, wedge_capacity);
    raw.wedge_ray_index = empty<Int>(event_count);
    raw.wedge_hit_x = empty<Float>(event_count);
    raw.wedge_hit_y = empty<Float>(event_count);
    raw.wedge_hit_z = empty<Float>(event_count);
    raw.wedge_normal_x = empty<Float>(event_count);
    raw.wedge_normal_y = empty<Float>(event_count);
    raw.wedge_normal_z = empty<Float>(event_count);
    raw.wedge_prim_id = empty<Int>(event_count);
    raw.wedge_dir_x = empty<Float>(event_count);
    raw.wedge_dir_y = empty<Float>(event_count);
    raw.wedge_dir_z = empty<Float>(event_count);
    raw.wedge_source_x = empty<Float>(event_count);
    raw.wedge_source_y = empty<Float>(event_count);
    raw.wedge_source_z = empty<Float>(event_count);
    raw.wedge_source_power = empty<Float>(event_count);
    raw.wedge_initial_dir_x = empty<Float>(event_count);
    raw.wedge_initial_dir_y = empty<Float>(event_count);
    raw.wedge_initial_dir_z = empty<Float>(event_count);
    raw.wedge_bounce_depth = empty<Int>(event_count);
    return raw;
}

void initialize_reflection_accumulation_raw(AccumRaw &raw) {
    const int zero_i = 0;
    const int minus_one_i = -1;
    const float zero_f = 0.f;
    const int event_count = std::max(1, raw.wedge_capacity);

    jit_memset_async(JitBackend::CUDA,
                     raw.reflection_power.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.reflection_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.wedge_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_ray_index.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_prim_id.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_power.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_initial_dir_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_initial_dir_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_initial_dir_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_bounce_depth.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
}

} // namespace

template <bool Detached>
AccumResultT<Detached> Scene::accumulate_reflections(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &tx_position,
    const AccumGrid &grid,
    const MaterialT<Detached> &material,
    int max_bounces,
    const AccumOptions &options,
    MaskT<Detached> active,
    const Vector3fT<Detached> &tx_polarization) const {
    require(is_ready(), "Scene::accumulate_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::accumulate_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::accumulate_reflections(): max_bounces must be positive.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accumulate_reflections(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accumulate_reflections(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accumulate_reflections(): grid bounds must be ordered.");
    require(options.wavelength > 0.f,
            "Scene::accumulate_reflections(): wavelength must be positive.");
    require(options.cell_area > 0.f,
            "Scene::accumulate_reflections(): cell_area must be positive.");
    require(options.solid_angle_per_ray >= 0.f,
            "Scene::accumulate_reflections(): solid_angle_per_ray must be non-negative.");
    require(options.wedge_capacity >= 0,
            "Scene::accumulate_reflections(): wedge_capacity must be non-negative.");
    require(options.wedge_sample_stride >= 1,
            "Scene::accumulate_reflections(): wedge_sample_stride must be >= 1.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    AccumResultT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    result.grid_cell_count = grid_cell_count;

    if constexpr (!Detached) {
        ReflectionChainAD chain;
        if (max_bounces == 1) {
            const ReflectionTraceAD trace =
                this->template trace_bounces<false>(ray, 1, active);
            chain = initialize_reflection_chain_result<false>(ray_count, 1);
            chain.bounce_count = trace.bounce_count;
            chain.discovery_count = trace.discovery_count;
            chain.representative_ray_index = trace.representative_ray_index;

            if (!trace.bounces.empty()) {
                const ReflectionBounceAD &bounce = trace.bounces.front();
                chain.t = bounce.t;
                chain.hit_points = bounce.hit_points;
                chain.geo_normals = bounce.geo_normals;
                chain.image_sources = bounce.image_sources;
                chain.plane_points = bounce.plane_points;
                chain.plane_normals = bounce.plane_normals;
                chain.shape_ids = bounce.shape_ids;
                chain.prim_ids = bounce.prim_ids;
                chain.local_prim_ids = bounce.local_prim_ids;
                chain.global_prim_ids = bounce.global_prim_ids;

                const MaskAD trailing_active = trace.bounce_count > 0;
                const Vector3fAD reflected_direction =
                    ray.d - 2.f * dot(ray.d, bounce.geo_normals) * bounce.geo_normals;
                const Vector3fAD trailing_origin =
                    bounce.hit_points + Epsilon * reflected_direction;
                const RayAD trailing_ray(
                    trailing_origin,
                    reflected_direction,
                    full<FloatAD>(Infinity, ray_count));
                const IntersectionAD trailing =
                    this->template intersect<false>(
                        trailing_ray, trailing_active, RayFlags::Geometric);
                const MaskAD trailing_hit =
                    trailing_active && trailing.is_valid();
                chain.trailing_t =
                    select(trailing_hit,
                           trailing.t,
                           full<FloatAD>(Infinity, ray_count));
                chain.trailing_prim =
                    select(trailing_hit,
                           trailing.global_prim_id,
                           full<IntAD>(-1, ray_count));
                chain.trailing_dir =
                    select(trailing_active,
                           reflected_direction,
                           zeros<Vector3fAD>(ray_count));
                chain.trailing_origin =
                    select(trailing_active,
                           trailing_origin,
                           zeros<Vector3fAD>(ray_count));
            }
        } else {
            chain = this->template trace_reflections<false>(ray, max_bounces, active);
        }

        result.reflection_power = zeros<FloatAD>(grid_cell_count);
        result.reflection_field_x =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.reflection_field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.reflection_field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.reflection_count = full<IntAD>(0, 1);
        result.wedge_events.capacity = options.wedge_capacity;
        result.wedge_events.count = full<IntAD>(0, 1);
        const int event_count = std::max(1, options.wedge_capacity);
        result.wedge_events.ray_index = full<IntAD>(-1, event_count);
        result.wedge_events.hit_points = zeros<Vector3fAD>(event_count);
        result.wedge_events.normals = zeros<Vector3fAD>(event_count);
        result.wedge_events.prim_id = full<IntAD>(-1, event_count);
        result.wedge_events.directions = zeros<Vector3fAD>(event_count);
        result.wedge_events.source_points = zeros<Vector3fAD>(event_count);
        result.wedge_events.src_power = zeros<FloatAD>(event_count);
        result.wedge_events.initial_directions = zeros<Vector3fAD>(event_count);
        result.wedge_events.bounce_depth = full<IntAD>(-1, event_count);

        if (ray_count <= 0 || grid_cell_count <= 0) {
            return result;
        }
        const int material_count = static_cast<int>(slices(material.gain));
        require(material_count > 0,
                "Scene::accumulate_reflections(): material payload must not be empty.");
        require(static_cast<int>(slices(material.eta_r)) == material_count &&
                    static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accumulate_reflections(): material payload fields must have matching widths.");

        auto component = [](const Vector3fAD &value, int axis) -> FloatAD {
            if (axis == 0) {
                return value.x();
            }
            if (axis == 1) {
                return value.y();
            }
            return value.z();
        };
        auto plane_point = [](int axis,
                              const FloatAD &position,
                              const FloatAD &coord0,
                              const FloatAD &coord1) -> Vector3fAD {
            if (axis == 0) {
                return Vector3fAD(position, coord0, coord1);
            }
            if (axis == 1) {
                return Vector3fAD(coord0, position, coord1);
            }
            return Vector3fAD(coord0, coord1, position);
        };
        auto coords_from_point = [](const Vector3fAD &point,
                                    int axis,
                                    FloatAD &coord0,
                                    FloatAD &coord1) {
            if (axis == 0) {
                coord0 = point.y();
                coord1 = point.z();
            } else if (axis == 1) {
                coord0 = point.x();
                coord1 = point.z();
            } else {
                coord0 = point.x();
                coord1 = point.y();
            }
        };
        auto broadcast_vec = [](const Vector3fAD &value, int width) -> Vector3fAD {
            const int value_width = static_cast<int>(slices(value));
            if (value_width == width) {
                return value;
            }
            const UIntAD zero_index = zeros<UIntAD>(width);
            const MaskAD active = full<MaskAD>(true, width);
            return gather<Vector3fAD>(value, zero_index, active);
        };
        struct ComplexADValue {
            FloatAD re;
            FloatAD im;
        };
        struct ComplexVectorAD {
            Vector3fAD re;
            Vector3fAD im;
        };
        auto complex_add = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re + b.re, a.im + b.im};
        };
        auto complex_sub = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re - b.re, a.im - b.im};
        };
        auto complex_mul = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re * b.re - a.im * b.im,
                    a.re * b.im + a.im * b.re};
        };
        auto complex_scale = [](const ComplexADValue &a,
                                const FloatAD &scale) -> ComplexADValue {
            return {a.re * scale, a.im * scale};
        };
        auto complex_div = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            const FloatAD denom =
                maximum(b.re * b.re + b.im * b.im, FloatAD(Epsilon));
            return {(a.re * b.re + a.im * b.im) / denom,
                    (a.im * b.re - a.re * b.im) / denom};
        };
        auto complex_sqrt = [](const ComplexADValue &a) -> ComplexADValue {
            const FloatAD mag =
                sqrt(maximum(a.re * a.re + a.im * a.im, FloatAD(0.f)));
            const MaskAD positive_real_axis =
                (abs(a.im) <= FloatAD(Epsilon)) && (a.re > FloatAD(Epsilon));
            const FloatAD real_part =
                sqrt(maximum(FloatAD(0.5f) * (mag + a.re), FloatAD(0.f)));
            const FloatAD imag_abs =
                sqrt(maximum(FloatAD(0.5f) * (mag - a.re), FloatAD(1e-20f)));
            const FloatAD imag_sign =
                select(a.im < FloatAD(0.f), FloatAD(-1.f), FloatAD(1.f));
            return {
                select(positive_real_axis, sqrt(a.re), real_part),
                select(positive_real_axis, FloatAD(0.f), imag_sign * imag_abs),
            };
        };
        auto normalize_safe = [](const Vector3fAD &value,
                                 const Vector3fAD &fallback) -> Vector3fAD {
            const FloatAD value_norm = norm(value);
            return select(value_norm > FloatAD(Epsilon),
                          value / maximum(value_norm, FloatAD(Epsilon)),
                          fallback);
        };
        auto stable_perpendicular = [&](const Vector3fAD &direction,
                                        const Vector3fAD &preferred) -> Vector3fAD {
            const Vector3fAD dir =
                normalize_safe(direction, Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            const Vector3fAD projected = preferred - dot(preferred, dir) * dir;
            const Vector3fAD axis =
                select(abs(dir.x()) < FloatAD(0.9f),
                       Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)),
                       Vector3fAD(FloatAD(0.f), FloatAD(1.f), FloatAD(0.f)));
            const Vector3fAD fallback = axis - dot(axis, dir) * dir;
            return select(squared_norm(projected) > FloatAD(1e-12f),
                          normalize_safe(projected, axis),
                          normalize_safe(fallback,
                                         Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f))));
        };
        auto complex_dot_real = [](const ComplexVectorAD &field,
                                   const Vector3fAD &basis) -> ComplexADValue {
            return {dot(field.re, basis), dot(field.im, basis)};
        };
        auto complex_vector_power = [](const ComplexVectorAD &field) -> FloatAD {
            return squared_norm(field.re) + squared_norm(field.im);
        };
        auto material_reflection_coefficients =
            [&](const IntAD &prim,
                const FloatAD &cos_theta,
                const MaskAD &slot_active) -> std::pair<ComplexADValue, ComplexADValue> {
            const MaskAD prim_in_range =
                slot_active && (prim >= IntAD(0)) && (prim < IntAD(material_count));
            const IntAD safe_prim = select(prim_in_range, prim, IntAD(0));
            const MaskAD prim_valid =
                prim_in_range && gather<MaskAD>(material.valid, safe_prim, prim_in_range);
            const FloatAD eta_r =
                maximum(gather<FloatAD>(material.eta_r, safe_prim, prim_valid),
                        FloatAD(Epsilon));
            const FloatAD sigma =
                maximum(gather<FloatAD>(material.sigma, safe_prim, prim_valid),
                        FloatAD(0.f));
            const FloatAD gain = gather<FloatAD>(material.gain, safe_prim, prim_valid);
            const FloatAD mu_r =
                maximum(gather<FloatAD>(material.mu_r, safe_prim, prim_valid),
                        FloatAD(Epsilon));
            const FloatAD omega = maximum(
                FloatAD(2.f * Pi) * FloatAD(299792458.f) /
                    maximum(FloatAD(options.wavelength), FloatAD(Epsilon)),
                FloatAD(Epsilon));
            const ComplexADValue eta = {
                eta_r,
                -sigma / (omega * FloatAD(8.854187817e-12f))
            };
            const ComplexADValue mu = {mu_r, FloatAD(0.f)};
            const FloatAD cos_clamped =
                minimum(maximum(abs(cos_theta), FloatAD(Epsilon)), FloatAD(1.f));
            const FloatAD sin2 =
                maximum(FloatAD(0.f), FloatAD(1.f) - cos_clamped * cos_clamped);
            const ComplexADValue a =
                complex_sqrt(complex_sub(complex_mul(mu, eta),
                                         ComplexADValue{sin2, FloatAD(0.f)}));
            const ComplexADValue mu_cos = {mu_r * cos_clamped, FloatAD(0.f)};
            const ComplexADValue eta_cos = {eta.re * cos_clamped,
                                            eta.im * cos_clamped};
            const ComplexADValue zero = {FloatAD(0.f), FloatAD(0.f)};
            const ComplexADValue r_te_raw =
                complex_scale(
                    complex_div(complex_sub(mu_cos, a),
                                complex_add(mu_cos, a)),
                    gain);
            const ComplexADValue r_tm_raw =
                complex_scale(
                    complex_div(complex_sub(eta_cos, a),
                                complex_add(eta_cos, a)),
                    gain);
            const ComplexADValue r_te = {
                select(prim_valid, r_te_raw.re, zero.re),
                select(prim_valid, r_te_raw.im, zero.im),
            };
            const ComplexADValue r_tm = {
                select(prim_valid, r_tm_raw.re, zero.re),
                select(prim_valid, r_tm_raw.im, zero.im),
            };
            return {r_te, r_tm};
        };
        auto reflect_field_vector =
            [&](const ComplexVectorAD &field,
                const Vector3fAD &incident_dir,
                const Vector3fAD &slot_normal,
                const IntAD &prim,
                const MaskAD &slot_active) -> std::pair<ComplexVectorAD, Vector3fAD> {
            const Vector3fAD incident_hat =
                normalize_safe(incident_dir,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            Vector3fAD normal_hat =
                normalize_safe(slot_normal,
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            normal_hat = select(dot(incident_hat, normal_hat) > FloatAD(0.f),
                                -normal_hat,
                                normal_hat);
            const FloatAD dot_dn = dot(incident_hat, normal_hat);
            const Vector3fAD reflected_dir =
                normalize_safe(incident_hat - FloatAD(2.f) * dot_dn * normal_hat,
                               -incident_hat);
            Vector3fAD s_hat = cross(normal_hat, incident_hat);
            s_hat = select(squared_norm(s_hat) > FloatAD(1e-12f),
                           normalize_safe(s_hat, stable_perpendicular(incident_hat, normal_hat)),
                           stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_in_hat = cross(s_hat, incident_hat);
            p_in_hat =
                select(squared_norm(p_in_hat) > FloatAD(1e-12f),
                       normalize_safe(p_in_hat, stable_perpendicular(incident_hat, normal_hat)),
                       stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_out_hat = cross(s_hat, reflected_dir);
            p_out_hat =
                select(squared_norm(p_out_hat) > FloatAD(1e-12f),
                       normalize_safe(p_out_hat, stable_perpendicular(reflected_dir, normal_hat)),
                       stable_perpendicular(reflected_dir, normal_hat));
            const auto [r_te, r_tm] =
                material_reflection_coefficients(prim, abs(dot(incident_hat, normal_hat)), slot_active);
            const ComplexADValue e_s = complex_dot_real(field, s_hat);
            const ComplexADValue e_p = complex_dot_real(field, p_in_hat);
            const ComplexADValue out_s = complex_mul(r_te, e_s);
            const ComplexADValue out_p = complex_mul(r_tm, e_p);
            return {{
                s_hat * out_s.re + p_out_hat * out_p.re,
                s_hat * out_s.im + p_out_hat * out_p.im,
            }, reflected_dir};
        };

        const UIntAD ray_index = arange<UIntAD>(ray_count);
        const IntAD ray_slot = IntAD(ray_index);
        const IntAD base_slot = ray_slot * IntAD(max_bounces);
        const MaskAD active_ad = active;
        Vector3fAD origin = ray.o;
        Vector3fAD direction = normalize(ray.d);
        Vector3fAD image_source = broadcast_vec(tx_position, ray_count);
        Vector3fAD tx_pol = broadcast_vec(tx_polarization, ray_count);
        Vector3fAD transverse_pol = tx_pol - dot(tx_pol, direction) * direction;
        transverse_pol = select(
            squared_norm(transverse_pol) > FloatAD(1e-12f),
            normalize_safe(transverse_pol,
                           stable_perpendicular(direction, tx_pol)),
            stable_perpendicular(direction, tx_pol));
        ComplexVectorAD field = {transverse_pol, zeros<Vector3fAD>(ray_count)};
        FloatAD path_length = zeros<FloatAD>(ray_count);
        MaskAD current_active = active_ad;

        const FloatAD span0 = FloatAD(grid.coord0_max - grid.coord0_min);
        const FloatAD span1 = FloatAD(grid.coord1_max - grid.coord1_min);
        const FloatAD wavelength = FloatAD(options.wavelength);
        const FloatAD wave_gain = wavelength / FloatAD(4.f * Pi);
        const FloatAD solid_angle = FloatAD(options.solid_angle_per_ray);
        const FloatAD cell_area = FloatAD(options.cell_area);
        const FloatAD wave_k =
            select(abs(FloatAD(options.k)) > FloatAD(Epsilon),
                   FloatAD(options.k),
                   FloatAD(2.f * Pi) / maximum(wavelength, FloatAD(Epsilon)));

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const IntAD slot = base_slot + IntAD(bounce);
            const MaskAD bounce_active =
                current_active && (chain.bounce_count > IntAD(bounce));
            if (drjit::none(detach<false>(bounce_active))) {
                break;
            }

            const Vector3fAD hit_point =
                gather<Vector3fAD>(chain.hit_points, slot, bounce_active);
            Vector3fAD normal =
                gather<Vector3fAD>(chain.geo_normals, slot, bounce_active);
            normal = normalize(normal);
            normal = select(dot(direction, normal) > FloatAD(0.f), -normal, normal);
            const IntAD prim =
                gather<IntAD>(chain.global_prim_ids, slot, bounce_active);
            const MaskAD material_active = bounce_active;

            const Vector3fAD event_source = image_source;
            const Vector3fAD event_direction = direction;
            const FloatAD event_source_power = complex_vector_power(field) * solid_angle;
            const FloatAD image_distance = dot(image_source - hit_point, normal);
            image_source = select(
                material_active,
                image_source - FloatAD(2.f) * image_distance * normal,
                image_source);
            const auto reflected = reflect_field_vector(field, direction, normal, prim, material_active);
            direction = select(material_active, reflected.second, direction);
            origin = select(
                material_active,
                hit_point + FloatAD(Epsilon) * direction,
                origin);
            field = {
                select(material_active, reflected.first.re, zeros<Vector3fAD>(ray_count)),
                select(material_active, reflected.first.im, zeros<Vector3fAD>(ray_count)),
            };
            path_length = select(
                material_active,
                path_length + gather<FloatAD>(chain.t, slot, bounce_active),
                path_length);

            if (options.collect_wedges && options.wedge_capacity > 0) {
                const IntAD event_slot =
                    (ray_slot * IntAD(max_bounces) + IntAD(bounce)) /
                    IntAD(options.wedge_sample_stride);
                const MaskAD wedge_active =
                    material_active && (event_slot >= IntAD(0)) &&
                    (event_slot < IntAD(options.wedge_capacity));
                scatter(
                    result.wedge_events.ray_index,
                    ray_slot,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.hit_points,
                    hit_point,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.normals,
                    normal,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.prim_id,
                    prim,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.directions,
                    event_direction,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.source_points,
                    event_source,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.src_power,
                    event_source_power,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.initial_directions,
                    normalize(ray.d),
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.bounce_depth,
                    IntAD(bounce),
                    event_slot,
                    wedge_active);
                scatter_reduce(
                    ReduceOp::Add,
                    result.wedge_events.count,
                    IntAD(1),
                    zeros<IntAD>(ray_count),
                    wedge_active);
            }

            FloatAD blocker_t = chain.trailing_t;
            if (bounce + 1 < max_bounces) {
                const IntAD next_slot = slot + IntAD(1);
                const MaskAD next_valid = chain.bounce_count > IntAD(bounce + 1);
                blocker_t = select(
                    next_valid,
                    gather<FloatAD>(chain.t, next_slot, next_valid),
                    blocker_t);
            }

            const FloatAD axis_dir = component(direction, grid.axis);
            const FloatAD safe_axis_dir =
                axis_dir + select(axis_dir >= FloatAD(0.f), FloatAD(Epsilon), FloatAD(-Epsilon));
            const FloatAD t_plane =
                (FloatAD(grid.position) - component(origin, grid.axis)) / safe_axis_dir;
            const Vector3fAD target = origin + direction * t_plane;
            FloatAD coord0 = zeros<FloatAD>(ray_count);
            FloatAD coord1 = zeros<FloatAD>(ray_count);
            coords_from_point(target, grid.axis, coord0, coord1);

            const MaskAD plane_active =
                material_active &&
                (complex_vector_power(field) > FloatAD(0.f)) &&
                (t_plane > FloatAD(RayEpsilon)) &&
                (t_plane < blocker_t) &&
                (coord0 >= FloatAD(grid.coord0_min)) &&
                (coord0 < FloatAD(grid.coord0_max)) &&
                (coord1 >= FloatAD(grid.coord1_min)) &&
                (coord1 < FloatAD(grid.coord1_max));
            const FloatAD u = (coord0 - FloatAD(grid.coord0_min)) / span0;
            const FloatAD v = (coord1 - FloatAD(grid.coord1_min)) / span1;
            IntAD ix = IntAD(u * FloatAD(grid.resolution0));
            IntAD iy = IntAD(v * FloatAD(grid.resolution1));
            ix = minimum(maximum(ix, IntAD(0)), IntAD(grid.resolution0 - 1));
            iy = minimum(maximum(iy, IntAD(0)), IntAD(grid.resolution1 - 1));
            const IntAD cell = iy * IntAD(grid.resolution0) + ix;

            const Vector3fAD target_plane =
                plane_point(grid.axis, FloatAD(grid.position), coord0, coord1);
            const FloatAD unfolded_distance =
                maximum(norm(target_plane - image_source), FloatAD(Epsilon));
            const FloatAD cos_theta = maximum(abs(axis_dir), FloatAD(Epsilon));
            const FloatAD geometry_power_scale =
                solid_angle / cell_area *
                unfolded_distance * unfolded_distance / cos_theta;
            const FloatAD amplitude_scale =
                wave_gain / unfolded_distance *
                sqrt(maximum(geometry_power_scale, FloatAD(0.f)));
            const ComplexADValue phase = {
                cos(wave_k * unfolded_distance),
                -sin(wave_k * unfolded_distance)
            };
            const ComplexADValue coeff = complex_scale(phase, amplitude_scale);
            const ComplexVectorAD contribution_field = {
                field.re * coeff.re - field.im * coeff.im,
                field.re * coeff.im + field.im * coeff.re,
            };
            const FloatAD contribution_power =
                complex_vector_power(contribution_field);
            const MaskAD contribution_active =
                plane_active && drjit::isfinite(contribution_power) && (contribution_power > FloatAD(0.f));
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_x.x(),
                contribution_field.re.x(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_x.y(),
                contribution_field.im.x(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_y.x(),
                contribution_field.re.y(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_y.y(),
                contribution_field.im.y(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_z.x(),
                contribution_field.re.z(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_z.y(),
                contribution_field.im.z(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_power,
                contribution_power,
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_count,
                IntAD(1),
                zeros<IntAD>(ray_count),
                contribution_active);

            const int next_depth = bounce + 1;
            MaskAD continue_active = material_active;
            if (options.rr_depth > 0 && options.rr_prob < 1.f && next_depth >= options.rr_depth) {
                const FloatAD rr_field_power = complex_vector_power(field);
                const FloatAD continue_prob =
                    minimum(maximum(rr_field_power, FloatAD(1e-8f)),
                            maximum(FloatAD(options.rr_prob), FloatAD(1e-8f)));
                const FloatAD rr_scale =
                    FloatAD(1.f) / sqrt(maximum(continue_prob, FloatAD(1e-8f)));
                field = {
                    select(continue_active, field.re * rr_scale, field.re),
                    select(continue_active, field.im * rr_scale, field.im),
                };
            }
            if (options.stop_threshold > 0.f) {
                const FloatAD fspl =
                    wavelength / (FloatAD(4.f * Pi) * maximum(path_length, FloatAD(Epsilon)));
                continue_active =
                    continue_active &&
                    (complex_vector_power(field) * fspl * fspl > FloatAD(options.stop_threshold));
            }
            current_active = continue_active;
        }
        return result;
    } else {
        ScopedNativeLaunchStage native_launch_stage(
            NativeLaunchStage::AccumulateReflections);
        auto initialize_result_storage = [&]() {
            result.reflection_power = zeros<Float>(grid_cell_count);
            result.reflection_field_x =
                drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                      zeros<Float>(grid_cell_count));
            result.reflection_field_y =
                drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                      zeros<Float>(grid_cell_count));
            result.reflection_field_z =
                drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                      zeros<Float>(grid_cell_count));
            result.reflection_count = full<Int>(0, 1);
            result.wedge_events.capacity = options.wedge_capacity;
            result.wedge_events.count = full<Int>(0, 1);
            const int event_count = std::max(1, options.wedge_capacity);
            result.wedge_events.ray_index = full<Int>(-1, event_count);
            result.wedge_events.hit_points = zeros<Vector3f>(event_count);
            result.wedge_events.normals = zeros<Vector3f>(event_count);
            result.wedge_events.prim_id = full<Int>(-1, event_count);
            result.wedge_events.directions = zeros<Vector3f>(event_count);
            result.wedge_events.source_points = zeros<Vector3f>(event_count);
            result.wedge_events.src_power = zeros<Float>(event_count);
            result.wedge_events.initial_directions = zeros<Vector3f>(event_count);
            result.wedge_events.bounce_depth = full<Int>(-1, event_count);
        };
        if (ray_count == 0) {
            initialize_result_storage();
            return result;
        }

        require(static_cast<int>(slices(ray.d)) == ray_count &&
                    static_cast<int>(slices(ray.tmax)) == ray_count,
                "Scene::accumulate_reflections(): ray fields must have matching widths.");
        const int tx_count = static_cast<int>(slices(tx_position));
        require(tx_count == 1 || tx_count == ray_count,
                "Scene::accumulate_reflections(): tx_position width must be 1 or match ray count.");
        const int tx_pol_count = static_cast<int>(slices(tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::accumulate_reflections(): tx_polarization width must be 1 or match ray count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accumulate_reflections(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accumulate_reflections(): material payload fields must have matching widths.");

        const int triangle_count = static_cast<int>(slices(triangle_info_detached_.p0));
        require(material_count >= triangle_count,
                "Scene::accumulate_reflections(): material payload must provide one entry per global primitive.");

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            const int hitgroup_record_count = scenes.hitgroup_record_count;

            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accumulate_reflections(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::accumulate_reflections(): invalid hitgroup record count.");

            ensure_pipeline(reflection_accumulation_pipeline_, primary_scene->context(),
                            hitgroup_record_count, reflection_accumulation_pipeline_config());
        }

        initialize_result_storage();

        Vector3f tx_detached = tx_position;
        if (tx_count == 1 && ray_count > 1) {
            const Int zero_index = full<Int>(0, ray_count);
            tx_detached = Vector3f(
                gather<Float>(tx_position.x(), zero_index),
                gather<Float>(tx_position.y(), zero_index),
                gather<Float>(tx_position.z(), zero_index));
        }
        Vector3f tx_pol_detached = tx_polarization;
        if (tx_pol_count == 1 && ray_count > 1) {
            const Int zero_index = full<Int>(0, ray_count);
            tx_pol_detached = Vector3f(
                gather<Float>(tx_polarization.x(), zero_index),
                gather<Float>(tx_polarization.y(), zero_index),
                gather<Float>(tx_polarization.z(), zero_index));
        }

        Mask active_detached = sanitize_reflection_active<true>(ray, active);
        active_detached &= drjit::isfinite(tx_detached.x()) &&
                           drjit::isfinite(tx_detached.y()) &&
                           drjit::isfinite(tx_detached.z()) &&
                           drjit::isfinite(tx_pol_detached.x()) &&
                           drjit::isfinite(tx_pol_detached.y()) &&
                           drjit::isfinite(tx_pol_detached.z());
        if (drjit::none(active_detached)) {
            return result;
        }

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    tx_detached,
                    tx_pol_detached,
                    active_detached,
                    triangle_info_detached_.p0,
                    triangle_info_detached_.e1,
                    triangle_info_detached_.e2,
                    triangle_info_detached_.face_normal,
                    face_offsets_,
                    material.eta_r,
                    material.sigma,
                    material.gain,
                    material.mu_r,
                    material.valid);

        AccumRaw raw = allocate_reflection_accumulation_raw(
            ray_count, max_bounces, grid_cell_count, options.wedge_capacity);
        initialize_reflection_accumulation_raw(raw);

        AccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.tri_p0_x = triangle_info_detached_.p0.x().data();
        params.tri_p0_y = triangle_info_detached_.p0.y().data();
        params.tri_p0_z = triangle_info_detached_.p0.z().data();
        params.tri_e1_x = triangle_info_detached_.e1.x().data();
        params.tri_e1_y = triangle_info_detached_.e1.y().data();
        params.tri_e1_z = triangle_info_detached_.e1.z().data();
        params.tri_e2_x = triangle_info_detached_.e2.x().data();
        params.tri_e2_y = triangle_info_detached_.e2.y().data();
        params.tri_e2_z = triangle_info_detached_.e2.z().data();
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.tx_x = tx_detached.x().data();
        params.tx_y = tx_detached.y().data();
        params.tx_z = tx_detached.z().data();
        params.tx_pol_x = tx_pol_detached.x().data();
        params.tx_pol_y = tx_pol_detached.y().data();
        params.tx_pol_z = tx_pol_detached.z().data();
        params.max_bounces = max_bounces;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.solid_angle_per_ray = options.solid_angle_per_ray;
        params.cell_area = options.cell_area;
        params.seed = options.seed;
        params.rr_depth = options.rr_depth;
        params.rr_prob = options.rr_prob;
        params.stop_threshold = options.stop_threshold;
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_gain = material.gain.data();
        params.material_mu_r = material.mu_r.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.collect_wedges = options.collect_wedges ? 1 : 0;
        params.collect_wedge_prefixes = options.collect_wedge_prefixes ? 1 : 0;
        params.wedge_capacity = options.wedge_capacity;
        params.wedge_sample_stride = options.wedge_sample_stride;
        params.out_reflection_power = raw.reflection_power.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_reflection_count = raw.reflection_count.data();
        params.out_wedge_count = raw.wedge_count.data();
        params.out_wedge_ray_index = raw.wedge_ray_index.data();
        params.out_wedge_hit_x = raw.wedge_hit_x.data();
        params.out_wedge_hit_y = raw.wedge_hit_y.data();
        params.out_wedge_hit_z = raw.wedge_hit_z.data();
        params.out_wedge_normal_x = raw.wedge_normal_x.data();
        params.out_wedge_normal_y = raw.wedge_normal_y.data();
        params.out_wedge_normal_z = raw.wedge_normal_z.data();
        params.out_wedge_prim_id = raw.wedge_prim_id.data();
        params.out_wedge_dir_x = raw.wedge_dir_x.data();
        params.out_wedge_dir_y = raw.wedge_dir_y.data();
        params.out_wedge_dir_z = raw.wedge_dir_z.data();
        params.out_wedge_source_x = raw.wedge_source_x.data();
        params.out_wedge_source_y = raw.wedge_source_y.data();
        params.out_wedge_source_z = raw.wedge_source_z.data();
        params.out_wedge_source_power = raw.wedge_source_power.data();
        params.out_wedge_initial_dir_x = raw.wedge_initial_dir_x.data();
        params.out_wedge_initial_dir_y = raw.wedge_initial_dir_y.data();
        params.out_wedge_initial_dir_z = raw.wedge_initial_dir_z.data();
        params.out_wedge_bounce_depth = raw.wedge_bounce_depth.data();

        if (cuda_trace) {
            cuda_backend().run_reflection_accumulation(params, ray_count);
        } else {
            reflection_accumulation_pipeline_->launch(0, params);
        }

        result.reflection_power = raw.reflection_power;
        result.reflection_field_x =
            drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.reflection_field_y =
            drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.reflection_field_z =
            drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.reflection_count = raw.reflection_count;
        result.wedge_events.capacity = options.wedge_capacity;
        result.wedge_events.count = raw.wedge_count;
        result.wedge_events.ray_index = raw.wedge_ray_index;
        result.wedge_events.hit_points =
            Vector3f(raw.wedge_hit_x, raw.wedge_hit_y, raw.wedge_hit_z);
        result.wedge_events.normals =
            Vector3f(raw.wedge_normal_x, raw.wedge_normal_y, raw.wedge_normal_z);
        result.wedge_events.prim_id = raw.wedge_prim_id;
        result.wedge_events.directions =
            Vector3f(raw.wedge_dir_x, raw.wedge_dir_y, raw.wedge_dir_z);
        result.wedge_events.source_points =
            Vector3f(raw.wedge_source_x, raw.wedge_source_y, raw.wedge_source_z);
        result.wedge_events.src_power = raw.wedge_source_power;
        result.wedge_events.initial_directions = Vector3f(
            raw.wedge_initial_dir_x,
            raw.wedge_initial_dir_y,
            raw.wedge_initial_dir_z);
        result.wedge_events.bounce_depth = raw.wedge_bounce_depth;
        return result;
    }
}

template AccumResult Scene::accumulate_reflections<true>(
    const Ray &ray,
    const Vector3f &tx_position,
    const AccumGrid &grid,
    const Material &material,
    int max_bounces,
    const AccumOptions &options,
    Mask active,
    const Vector3f &tx_polarization) const;
template AccumResultAD Scene::accumulate_reflections<false>(
    const RayAD &ray,
    const Vector3fAD &tx_position,
    const AccumGrid &grid,
    const MaterialAD &material,
    int max_bounces,
    const AccumOptions &options,
    MaskAD active,
    const Vector3fAD &tx_polarization) const;

} // namespace rayd
