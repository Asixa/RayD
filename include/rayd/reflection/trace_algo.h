// Copyright Xingyu Chen.
// Defines shared reflection support for trace algo.

#pragma once

#include <cmath>
#include <cstdint>

#include <vector_types.h> // float4 for the optional packed-triangle inputs.

#include <rayd/math.h>
#include <rayd/reflection/trace_params.h>
#include <rayd/reflection/reflection_geometry.h>
#include <rayd/rt/numeric_policy.h>
#include <rayd/rt/qualifiers.h>
#include <rayd/rt/traverser.h>

// Host-compilable reflection-trace algorithm. This is the de-CUDA-ised body of
// the former reflection_trace_raygen: math is math::Vec3f throughout (mirroring
// the exact arithmetic op order of the old local CUDA vector helpers so device
// codegen stays bit-identical), every ray cast goes through an rt::is_traverser
// Traverser (so no OptiX ray-cast intrinsic, payload register, or launch-index
// query appears here), and the lane index is a plain parameter.
// reflection_trace_device.cuh instantiates it with
// TraceConfig<ReflectionTracePolicy, OptixTraverser>; the CUDA fused executor
// (P4d) will reuse it with CudaBvhTraverser.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace reflection_trace_algo_detail {

using math::Vec3f;

constexpr float kTraceTMin = 1e-5f;
constexpr float kTraceTMax = 1e8f;
constexpr float kRayBias = 1e-5f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kTraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
// This family clears missed slots to kTraceTMax rather than +inf.
static_assert(kTraceTMax == ::rayd::shared::rt::kReflectionTraceMissDistance);

using ::rayd::shared::optix::ReflectionTraceParams;

struct TriangleData {
    Vec3f p0;
    Vec3f e1;
    Vec3f e2;
    Vec3f fn;
};

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

RAYD_HOST_DEVICE Vec3f load_vec3(const float* values) {
    return math::make_vec3(values[0], values[1], values[2]);
}

RAYD_HOST_DEVICE Vec3f load_vec3(float4 value) {
    return math::make_vec3(value.x, value.y, value.z);
}

RAYD_HOST_DEVICE Vec3f normalize(Vec3f value) {
    return math::scale(value, reciprocal_sqrt(fmaxf(math::dot(value, value), 1e-12f)));
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_nearest_hit(::rayd::shared::rt::TriangleHit a,
                                                                    ::rayd::shared::rt::TriangleHit b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return b.t < a.t ? b : a;
}

template <typename Policy>
RAYD_HOST_DEVICE int output_slot(const ReflectionTraceParams& params, unsigned int ray_index, int bounce) {
    if constexpr (Policy::honor_output_layout) {
        if (params.output_layout != 0)
            return bounce * params.n_rays + static_cast<int>(ray_index);
    }
    return static_cast<int>(ray_index) * params.max_bounces + bounce;
}

template <typename Policy>
RAYD_HOST_DEVICE void clear_output_slot(const ReflectionTraceParams& params, unsigned int ray_index, int bounce) {
    if constexpr (!Policy::clear_empty_slots) {
        return;
    } else {
        const int slot = output_slot<Policy>(params, ray_index, bounce);
        if (params.out_t != nullptr)
            params.out_t[slot] = kTraceTMax;
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = -1;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = -1;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = -1;
        if (params.out_valid != nullptr)
            params.out_valid[ray_index * params.max_bounces + bounce] = 0u;
        if (params.out_bary != nullptr) {
            params.out_bary[slot * 3 + 0] = 0.0f;
            params.out_bary[slot * 3 + 1] = 0.0f;
            params.out_bary[slot * 3 + 2] = 0.0f;
        }
        if (params.out_hit != nullptr) {
            params.out_hit[slot * 3 + 0] = 0.0f;
            params.out_hit[slot * 3 + 1] = 0.0f;
            params.out_hit[slot * 3 + 2] = 0.0f;
        }
        if (params.out_norm != nullptr) {
            params.out_norm[slot * 3 + 0] = 0.0f;
            params.out_norm[slot * 3 + 1] = 0.0f;
            params.out_norm[slot * 3 + 2] = 0.0f;
        }
        if (params.out_img != nullptr) {
            params.out_img[slot * 3 + 0] = 0.0f;
            params.out_img[slot * 3 + 1] = 0.0f;
            params.out_img[slot * 3 + 2] = 0.0f;
        }
    }
}

template <typename Policy> RAYD_HOST_DEVICE TriangleData load_triangle(const ReflectionTraceParams& params, int prim) {
    if constexpr (Policy::allow_packed_triangles) {
        if (params.tri_p0_packed != nullptr && params.tri_e1_packed != nullptr && params.tri_e2_packed != nullptr &&
            params.tri_fn_packed != nullptr) {
            return {load_vec3(params.tri_p0_packed[prim]), load_vec3(params.tri_e1_packed[prim]),
                    load_vec3(params.tri_e2_packed[prim]), load_vec3(params.tri_fn_packed[prim])};
        }
    }

    return {math::make_vec3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]),
            math::make_vec3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]),
            math::make_vec3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]),
            math::make_vec3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim])};
}

template <typename Policy>
RAYD_HOST_DEVICE void load_ray(const ReflectionTraceParams& params, unsigned int ray_index, Vec3f& origin,
                               Vec3f& direction) {
    if constexpr (Policy::allow_aos_inputs) {
        if (params.ray_o_aos != nullptr) {
            origin = load_vec3(params.ray_o_aos + ray_index * 3);
            direction = load_vec3(params.ray_d_aos + ray_index * 3);
            return;
        }
    }

    origin = math::make_vec3(params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    direction = math::make_vec3(params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]);
}

template <typename Policy>
RAYD_HOST_DEVICE float first_trace_tmax(const ReflectionTraceParams& params, unsigned int ray_index) {
    if constexpr (Policy::nullable_ray_tmax) {
        return params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kTraceTMax;
    }
    return params.ray_tmax[ray_index];
}

} // namespace reflection_trace_algo_detail

/// Reflection-path trace for one lane. `primary` / `secondary` are Config::
/// Traverser oracles over the two acceleration structures (secondary consulted
/// only when params.split_mode != 0), and `ray_index` is this lane's ray id.
template <typename Config>
RAYD_DEVICE void reflection_trace_algo(const ::rayd::shared::optix::ReflectionTraceParams& params,
                                       std::uint32_t ray_index, const typename Config::Traverser& primary,
                                       const typename Config::Traverser& secondary) {
    using namespace reflection_trace_algo_detail;
    using Policy = typename Config::Layout;
    using ::rayd::shared::rt::TriangleHit;
    namespace reflection = ::rayd::shared::reflection;

    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        if constexpr (Policy::clear_empty_slots) {
            for (int bounce = 0; bounce < params.max_bounces; ++bounce)
                clear_output_slot<Policy>(params, ray_index, bounce);
        }
        if (params.out_bounce_count != nullptr)
            params.out_bounce_count[ray_index] = 0;
        return;
    }

    const int bounce_limit = params.max_bounces;
    Vec3f origin;
    Vec3f direction;
    load_ray<Policy>(params, ray_index, origin, direction);
    Vec3f image_source = origin;
    int bounce_count = 0;

    for (int bounce = 0; bounce < bounce_limit; ++bounce) {
        const float tmax_input = bounce == 0 ? first_trace_tmax<Policy>(params, ray_index) : kTraceTMax;
        const float trace_tmax = is_finite(tmax_input) ? tmax_input : kTraceTMax;

        const TriangleHit primary_hit = primary.trace_closest(origin, direction, kTraceTMin, trace_tmax);
        TriangleHit hit = primary_hit;
        if (params.split_mode != 0) {
            const TriangleHit secondary_hit = secondary.trace_closest(origin, direction, kTraceTMin, trace_tmax);
            hit = choose_nearest_hit(primary_hit, secondary_hit);
        }
        if (hit.hit == 0u)
            break;

        const int shape_id = hit.instance;
        const int local_prim = hit.prim;
        const int face_offset = shape_id >= 0 && shape_id < params.n_meshes ? params.face_offsets[shape_id] : 0;
        const int global_prim = face_offset + local_prim;
        const float t = hit.t;
        const float bary_u = hit.bary_u;
        const float bary_v = hit.bary_v;

        Vec3f hit_point = math::add(origin, math::scale(direction, t));
        Vec3f geo_normal = math::make_vec3(0.0f, 0.0f, 1.0f);
        if (global_prim >= 0 && global_prim < params.n_triangles) {
            const TriangleData tri = load_triangle<Policy>(params, global_prim);
            hit_point = math::add(math::add(tri.p0, math::scale(tri.e1, bary_u)), math::scale(tri.e2, bary_v));
            geo_normal = normalize(tri.fn);
        }
        geo_normal = reflection::orient_normal_against(direction, geo_normal);

        const bool write_image_source =
            (Policy::allow_extended_outputs && params.out_img != nullptr) ||
            (params.out_img_x != nullptr && params.out_img_y != nullptr && params.out_img_z != nullptr);
        if (write_image_source) {
            image_source = reflection::reflect_point_across_plane(image_source, hit_point, geo_normal);
        }

        const int slot = output_slot<Policy>(params, ray_index, bounce);
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_valid != nullptr)
                params.out_valid[ray_index * params.max_bounces + bounce] = 1u;
        }
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = shape_id;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = local_prim;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = global_prim;
        if (params.out_t != nullptr)
            params.out_t[slot] = t;
        if (params.out_bary_u != nullptr)
            params.out_bary_u[slot] = bary_u;
        if (params.out_bary_v != nullptr)
            params.out_bary_v[slot] = bary_v;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_bary != nullptr) {
                params.out_bary[slot * 3 + 0] = 1.0f - bary_u - bary_v;
                params.out_bary[slot * 3 + 1] = bary_u;
                params.out_bary[slot * 3 + 2] = bary_v;
            }
        }
        if (params.out_hit_x != nullptr)
            params.out_hit_x[slot] = hit_point.x;
        if (params.out_hit_y != nullptr)
            params.out_hit_y[slot] = hit_point.y;
        if (params.out_hit_z != nullptr)
            params.out_hit_z[slot] = hit_point.z;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_hit != nullptr) {
                params.out_hit[slot * 3 + 0] = hit_point.x;
                params.out_hit[slot * 3 + 1] = hit_point.y;
                params.out_hit[slot * 3 + 2] = hit_point.z;
            }
        }
        if (params.out_norm_x != nullptr)
            params.out_norm_x[slot] = geo_normal.x;
        if (params.out_norm_y != nullptr)
            params.out_norm_y[slot] = geo_normal.y;
        if (params.out_norm_z != nullptr)
            params.out_norm_z[slot] = geo_normal.z;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_norm != nullptr) {
                params.out_norm[slot * 3 + 0] = geo_normal.x;
                params.out_norm[slot * 3 + 1] = geo_normal.y;
                params.out_norm[slot * 3 + 2] = geo_normal.z;
            }
        }
        if (write_image_source) {
            if constexpr (Policy::allow_extended_outputs) {
                if (params.out_img != nullptr) {
                    params.out_img[slot * 3 + 0] = image_source.x;
                    params.out_img[slot * 3 + 1] = image_source.y;
                    params.out_img[slot * 3 + 2] = image_source.z;
                } else {
                    params.out_img_x[slot] = image_source.x;
                    params.out_img_y[slot] = image_source.y;
                    params.out_img_z[slot] = image_source.z;
                }
            } else {
                params.out_img_x[slot] = image_source.x;
                params.out_img_y[slot] = image_source.y;
                params.out_img_z[slot] = image_source.z;
            }
        }

        direction = reflection::reflect_direction(direction, geo_normal);
        origin = math::add(hit_point, math::scale(direction, kRayBias));
        bounce_count = bounce + 1;
    }

    if constexpr (Policy::clear_empty_slots) {
        for (int bounce = bounce_count; bounce < bounce_limit; ++bounce)
            clear_output_slot<Policy>(params, ray_index, bounce);
    }

    if (bounce_count > 0 && params.return_trailing != 0) {
        if (params.out_trailing_dir_x != nullptr)
            params.out_trailing_dir_x[ray_index] = direction.x;
        if (params.out_trailing_dir_y != nullptr)
            params.out_trailing_dir_y[ray_index] = direction.y;
        if (params.out_trailing_dir_z != nullptr)
            params.out_trailing_dir_z[ray_index] = direction.z;
        if (params.out_trailing_origin_x != nullptr)
            params.out_trailing_origin_x[ray_index] = origin.x;
        if (params.out_trailing_origin_y != nullptr)
            params.out_trailing_origin_y[ray_index] = origin.y;
        if (params.out_trailing_origin_z != nullptr)
            params.out_trailing_origin_z[ray_index] = origin.z;

        const TriangleHit primary_hit = primary.trace_closest(origin, direction, kTraceTMin, kTraceTMax);
        TriangleHit trailing = primary_hit;
        if (params.split_mode != 0) {
            const TriangleHit secondary_hit = secondary.trace_closest(origin, direction, kTraceTMin, kTraceTMax);
            trailing = choose_nearest_hit(primary_hit, secondary_hit);
        }
        if (trailing.hit != 0u) {
            const int shape_id = trailing.instance;
            const int local_prim = trailing.prim;
            const int face_offset = shape_id >= 0 && shape_id < params.n_meshes ? params.face_offsets[shape_id] : 0;
            if (params.out_trailing_t != nullptr)
                params.out_trailing_t[ray_index] = trailing.t;
            if (params.out_trailing_prim != nullptr)
                params.out_trailing_prim[ray_index] = face_offset + local_prim;
        }
    }

    if (params.out_bounce_count != nullptr)
        params.out_bounce_count[ray_index] = bounce_count;
}

} // namespace rayd::shared::multipath
