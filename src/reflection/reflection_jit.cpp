#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray/drjit.h>
#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>
#include <src/reflection/dedup_jit.h>
#include <src/reflection/epc_field_jit.h>
#include <src/runtime/optix_pipelines_jit.h>
#include <rayd/diagnostics/drjit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>

#include "multipath_internal_jit.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct ReflectionTraceRaw {
    int max_bounces = 0;
    int ray_count = 0;
    Int bounce_count;
    Int discovery_count;
    Int representative_ray_index;
    Int shape_ids;
    Int prim_ids;
    Int global_prim_ids;
    Float t;
    Float bary_u;
    Float bary_v;
    Float hit_x;
    Float hit_y;
    Float hit_z;
    Float norm_x;
    Float norm_y;
    Float norm_z;
    Float img_x;
    Float img_y;
    Float img_z;
    Float trailing_t;
    Int trailing_prim;
    Float trailing_dir_x;
    Float trailing_dir_y;
    Float trailing_dir_z;
    Float trailing_origin_x;
    Float trailing_origin_y;
    Float trailing_origin_z;
};

/// Convert per-mesh (shape_id, local primitive id) pairs into scene-global primitive ids;
/// invalid or out-of-range inputs map to -1.
Int globalize_primitive_ids(const Int &local_prim_ids,
                                    const Int &shape_ids,
                                    const Int &face_offsets) {
    const int ray_count = static_cast<int>(slices(local_prim_ids));
    if (ray_count == 0) {
        return Int();
    }

    const int mesh_count = std::max(0, static_cast<int>(slices(face_offsets)) - 1);
    const Mask valid =
        (local_prim_ids >= 0) && (shape_ids >= 0) && (shape_ids < mesh_count);
    const Int safe_shape_ids = select(valid, shape_ids, zeros<Int>(ray_count));
    const Int mesh_face_offsets =
        gather<Int>(face_offsets, safe_shape_ids, valid);
    return select(valid,
                  local_prim_ids + mesh_face_offsets,
                  full<Int>(-1, ray_count));
}

template <bool Detached>
ReflectionBounceT<Detached> initialize_reflection_bounce_result(int ray_count) {
    ReflectionBounceT<Detached> result;
    result.t = full<FloatT<Detached>>(Infinity, ray_count);
    result.hit_points = zeros<Vector3fT<Detached>>(ray_count);
    result.geo_normals = zeros<Vector3fT<Detached>>(ray_count);
    result.image_sources = zeros<Vector3fT<Detached>>(ray_count);
    result.plane_points = zeros<Vector3fT<Detached>>(ray_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(ray_count);
    result.shape_ids = full<IntT<Detached>>(-1, ray_count);
    result.prim_ids = full<IntT<Detached>>(-1, ray_count);
    result.local_prim_ids = full<IntT<Detached>>(-1, ray_count);
    result.global_prim_ids = full<IntT<Detached>>(-1, ray_count);
    return result;
}

template <bool Detached>
ReflectionTraceT<Detached> initialize_reflection_trace_result(
    int ray_count,
    int max_bounces) {
    ReflectionTraceT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.discovery_count = full<IntT<Detached>>(0, ray_count);
    result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    result.dedup_keep_mask = full<MaskT<Detached>>(false, ray_count);
    result.bounces.reserve(static_cast<size_t>(max_bounces));
    return result;
}

ReflectionTraceRaw allocate_reflection_trace_raw(
    int ray_count,
    int max_bounces,
    int export_mode = RAYD_REFLECTION_EXPORT_FULL,
    bool return_trailing = true,
    bool include_shape_ids = true) {
    const int slot_count = ray_count * max_bounces;
    const bool full_export = export_mode == RAYD_REFLECTION_EXPORT_FULL;
    const bool minimal_export = export_mode == RAYD_REFLECTION_EXPORT_MINIMAL;
    const bool slot_ids_export = full_export || minimal_export;

    ReflectionTraceRaw raw;
    raw.max_bounces = max_bounces;
    raw.ray_count = ray_count;
    raw.bounce_count = empty<Int>(ray_count);
    if (full_export) {
        raw.discovery_count = empty<Int>(ray_count);
        raw.representative_ray_index = empty<Int>(ray_count);
    }
    if (slot_ids_export && include_shape_ids) {
        raw.shape_ids = empty<Int>(slot_count);
    }
    if (slot_ids_export) {
        raw.prim_ids = empty<Int>(slot_count);
        raw.t = empty<Float>(slot_count);
    }
    if (minimal_export) {
        raw.global_prim_ids = empty<Int>(slot_count);
    }
    if (full_export) {
        raw.bary_u = empty<Float>(slot_count);
        raw.bary_v = empty<Float>(slot_count);
        raw.hit_x = empty<Float>(slot_count);
        raw.hit_y = empty<Float>(slot_count);
        raw.hit_z = empty<Float>(slot_count);
        raw.norm_x = empty<Float>(slot_count);
        raw.norm_y = empty<Float>(slot_count);
        raw.norm_z = empty<Float>(slot_count);
        raw.img_x = empty<Float>(slot_count);
        raw.img_y = empty<Float>(slot_count);
        raw.img_z = empty<Float>(slot_count);
    }
    if (return_trailing) {
        raw.trailing_t = empty<Float>(ray_count);
        raw.trailing_prim = empty<Int>(ray_count);
        raw.trailing_dir_x = empty<Float>(ray_count);
        raw.trailing_dir_y = empty<Float>(ray_count);
        raw.trailing_dir_z = empty<Float>(ray_count);
        raw.trailing_origin_x = empty<Float>(ray_count);
        raw.trailing_origin_y = empty<Float>(ray_count);
        raw.trailing_origin_z = empty<Float>(ray_count);
    }
    return raw;
}

void initialize_reflection_trace_raw(ReflectionTraceRaw &raw,
                                     bool initialize_bounce_count = true) {
    const int ray_count = raw.ray_count;
    const int slot_count = raw.ray_count * raw.max_bounces;
    const int zero_i = 0;
    const int minus_one_i = -1;
    const float zero_f = 0.f;
    const float inf_f = Infinity;

    if (initialize_bounce_count) {
        jit_memset_async(JitBackend::CUDA, raw.bounce_count.data(), ray_count, sizeof(int), &zero_i);
    }
    if (slices(raw.discovery_count) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.discovery_count.data(), ray_count, sizeof(int), &zero_i);
    }
    if (slices(raw.representative_ray_index) > 0) {
        jit_memset_async(JitBackend::CUDA,
                         raw.representative_ray_index.data(),
                         ray_count,
                         sizeof(int),
                         &minus_one_i);
    }
    if (slices(raw.shape_ids) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.shape_ids.data(), slot_count, sizeof(int), &minus_one_i);
    }
    if (slices(raw.prim_ids) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.prim_ids.data(), slot_count, sizeof(int), &minus_one_i);
    }
    if (slices(raw.global_prim_ids) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.global_prim_ids.data(), slot_count, sizeof(int), &minus_one_i);
    }
    if (slices(raw.t) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.t.data(), slot_count, sizeof(float), &inf_f);
    }
    if (slices(raw.bary_u) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.bary_u.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.bary_v) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.bary_v.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.hit_x) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.hit_x.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.hit_y) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.hit_y.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.hit_z) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.hit_z.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.norm_x) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.norm_x.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.norm_y) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.norm_y.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.norm_z) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.norm_z.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.img_x) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.img_x.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.img_y) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.img_y.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.img_z) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.img_z.data(), slot_count, sizeof(float), &zero_f);
    }
    if (slices(raw.trailing_t) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_t.data(), ray_count, sizeof(float), &inf_f);
    }
    if (slices(raw.trailing_prim) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_prim.data(), ray_count, sizeof(int), &minus_one_i);
    }
    if (slices(raw.trailing_dir_x) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_dir_x.data(), ray_count, sizeof(float), &zero_f);
    }
    if (slices(raw.trailing_dir_y) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_dir_y.data(), ray_count, sizeof(float), &zero_f);
    }
    if (slices(raw.trailing_dir_z) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_dir_z.data(), ray_count, sizeof(float), &zero_f);
    }
    if (slices(raw.trailing_origin_x) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_origin_x.data(), ray_count, sizeof(float), &zero_f);
    }
    if (slices(raw.trailing_origin_y) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_origin_y.data(), ray_count, sizeof(float), &zero_f);
    }
    if (slices(raw.trailing_origin_z) > 0) {
        jit_memset_async(JitBackend::CUDA, raw.trailing_origin_z.data(), ray_count, sizeof(float), &zero_f);
    }
}

template <typename ArrayD>
ArrayD prefix_array(const ArrayD &value, int count) {
    return gather<ArrayD>(value, arange<Int>(count));
}

template <typename ArrayD>
ArrayD concat_array_sequence(const std::vector<ArrayD> &parts) {
    require(!parts.empty(),
            "concat_array_sequence(): at least one array is required.");
    ArrayD result = parts.front();
    for (size_t i = 1; i < parts.size(); ++i) {
        result = concat(result, parts[i]);
    }
    return result;
}

Int reflection_trace_ray_major_indices(int ray_count, int max_bounces) {
    const Int slot = arange<Int>(ray_count * max_bounces);
    const Int ray_index = slot / Int(max_bounces);
    const Int bounce_index = slot - ray_index * Int(max_bounces);
    return bounce_index * Int(ray_count) + ray_index;
}

template <bool Detached>
ReflectionTraceT<Detached> trace_bounces_impl(
    const Scene &scene,
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active) {
    require(!options.deduplicate,
            "Scene::trace_reflections(): deduplicate=true is not implemented with symbolic=true yet.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionTraceT<Detached> result =
        initialize_reflection_trace_result<Detached>(ray_count, max_bounces);
    result.deduplicate_requested = options.deduplicate;
    if (ray_count == 0) {
        return result;
    }

    const Mask sanitized_active_detached =
        sanitize_reflection_active<Detached>(ray, active);

    RayT<Detached> current_ray = ray;
    MaskT<Detached> current_active;
    if constexpr (Detached) {
        current_active = sanitized_active_detached;
        result.representative_ray_index = arange<Int>(ray_count);
    } else {
        current_active = MaskAD(sanitized_active_detached);
        result.representative_ray_index = IntAD(arange<Int>(ray_count));
    }
    Vector3fT<Detached> current_image_source = ray.o;

    const FloatT<Detached> miss_t = full<FloatT<Detached>>(Infinity, ray_count);
    const IntT<Detached> miss_id = full<IntT<Detached>>(-1, ray_count);
    const Vector3fT<Detached> zero_v = zeros<Vector3fT<Detached>>(ray_count);
    const IntT<Detached> one_i = full<IntT<Detached>>(1, ray_count);
    const IntT<Detached> zero_i = full<IntT<Detached>>(0, ray_count);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        const IntersectionT<Detached> its =
            scene.template intersect<Detached>(current_ray, current_active, RayFlags::Geometric);
        const MaskT<Detached> bounce_hit = current_active && its.is_valid();

        Vector3fT<Detached> geo_normal = its.geo_n;
        geo_normal = select(dot(current_ray.d, geo_normal) > 0.f, -geo_normal, geo_normal);
        const FloatT<Detached> plane_distance =
            dot(current_image_source - its.p, geo_normal);
        const Vector3fT<Detached> reflected_image_source =
            current_image_source - 2.f * plane_distance * geo_normal;

        ReflectionBounceT<Detached> bounce_result =
            initialize_reflection_bounce_result<Detached>(ray_count);
        bounce_result.t = select(bounce_hit, its.t, miss_t);
        bounce_result.hit_points = select(bounce_hit, its.p, zero_v);
        bounce_result.geo_normals = select(bounce_hit, geo_normal, zero_v);
        bounce_result.image_sources = select(bounce_hit, reflected_image_source, zero_v);
        bounce_result.plane_points = bounce_result.hit_points;
        bounce_result.plane_normals = bounce_result.geo_normals;
        bounce_result.shape_ids = select(bounce_hit, its.shape_id, miss_id);
        bounce_result.prim_ids = select(bounce_hit, its.prim_id, miss_id);
        bounce_result.local_prim_ids = select(bounce_hit, its.local_prim_id, miss_id);
        bounce_result.global_prim_ids = select(bounce_hit, its.global_prim_id, miss_id);
        result.bounces.push_back(std::move(bounce_result));

        result.bounce_count += select(bounce_hit, one_i, zero_i);

        const FloatT<Detached> ray_dot_normal = dot(current_ray.d, geo_normal);
        const Vector3fT<Detached> reflected_direction =
            current_ray.d - 2.f * ray_dot_normal * geo_normal;
        current_ray.o = select(bounce_hit,
                               its.p + Epsilon * reflected_direction,
                               current_ray.o);
        current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
        current_ray.tmax = select(bounce_hit,
                                  full<FloatT<Detached>>(Infinity, ray_count),
                                  current_ray.tmax);
        current_image_source =
            select(bounce_hit, reflected_image_source, current_image_source);
        current_active = bounce_hit;
    }

    result.dedup_keep_mask = result.bounce_count > 0;
    result.discovery_count = select(result.dedup_keep_mask, one_i, zero_i);
    result.representative_ray_index =
        select(result.dedup_keep_mask,
               result.representative_ray_index,
               full<IntT<Detached>>(-1, ray_count));
    return result;
}

} // namespace

template <bool Detached>
ReflectionChainT<Detached> Scene::trace_reflections(const RayT<Detached> &ray,
                                                    int max_bounces,
                                                    MaskT<Detached> active) const {
    return this->template trace_reflections<Detached>(
        ray, max_bounces, ReflectionTraceOptions(), active);
}

template <bool Detached>
ReflectionTraceT<Detached> Scene::trace_bounces(
    const RayT<Detached> &ray,
    int max_bounces,
    MaskT<Detached> active) const {
    return this->template trace_bounces<Detached>(
        ray, max_bounces, ReflectionTraceOptions(), active);
}

template <bool Detached>
ReflectionTraceT<Detached> Scene::trace_bounces(
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflections(): max_bounces must be positive.");
    return trace_bounces_impl<Detached>(
        *this, ray, max_bounces, options, active);
}

template <bool Detached>
ReflectionChainT<Detached> Scene::trace_reflections(const RayT<Detached> &ray,
                                                    int max_bounces,
                                                    const ReflectionTraceOptions &options,
                                                    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0, "Scene::trace_reflections(): max_bounces must be positive.");
    require(options.export_mode == RAYD_REFLECTION_EXPORT_FULL ||
                options.export_mode == RAYD_REFLECTION_EXPORT_MINIMAL ||
                options.export_mode == RAYD_REFLECTION_EXPORT_COUNT_ONLY,
            "Scene::trace_reflections(): invalid reflection export mode.");
    require(!options.deduplicate ||
                options.export_mode == RAYD_REFLECTION_EXPORT_FULL,
            "Scene::trace_reflections(): deduplicate requires full export mode.");
    const int export_mode =
        Detached ? options.export_mode : RAYD_REFLECTION_EXPORT_FULL;

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool include_shape_ids =
        export_mode == RAYD_REFLECTION_EXPORT_FULL || !Detached;
    const bool return_trailing =
        options.return_trailing &&
        export_mode != RAYD_REFLECTION_EXPORT_COUNT_ONLY;
    ReflectionChainT<Detached> result =
        initialize_reflection_chain_result<Detached>(
            ray_count,
            max_bounces,
            export_mode,
            return_trailing,
            include_shape_ids);
    if (ray_count == 0) {
        return result;
    }

    const bool symbolic_reflection_trace = recording_reflections();
    require(!symbolic_reflection_trace || !options.deduplicate,
            "Scene::trace_reflections(): symbolic recording does not support deduplicate=true yet.");
    if (symbolic_reflection_trace) {
        require(max_bounces == 1,
                "Scene::trace_reflections(): symbolic recording currently supports max_bounces=1 only.");
        const ReflectionTraceT<Detached> trace =
            this->template trace_bounces<Detached>(ray, 1, options, active);
        result.bounce_count = trace.bounce_count;
        result.discovery_count = trace.discovery_count;
        result.representative_ray_index = trace.representative_ray_index;
        if (!trace.bounces.empty()) {
            const ReflectionBounceT<Detached> &bounce = trace.bounces.front();
            result.t = bounce.t;
            result.hit_points = bounce.hit_points;
            result.geo_normals = bounce.geo_normals;
            result.image_sources = bounce.image_sources;
            result.plane_points = bounce.plane_points;
            result.plane_normals = bounce.plane_normals;
            result.shape_ids = bounce.shape_ids;
            result.prim_ids = bounce.prim_ids;
            result.local_prim_ids = bounce.local_prim_ids;
            result.global_prim_ids = bounce.global_prim_ids;

            if (return_trailing) {
                const MaskT<Detached> trailing_active = trace.bounce_count > 0;
                const Vector3fT<Detached> reflected_direction =
                    ray.d - 2.f * dot(ray.d, bounce.geo_normals) * bounce.geo_normals;
                const Vector3fT<Detached> trailing_origin =
                    bounce.hit_points + Epsilon * reflected_direction;
                RayT<Detached> trailing_ray(
                    trailing_origin,
                    reflected_direction,
                    full<FloatT<Detached>>(Infinity, ray_count));
                const IntersectionT<Detached> trailing =
                    this->template intersect<Detached>(
                        trailing_ray, trailing_active, RayFlags::Geometric);
                const MaskT<Detached> trailing_hit =
                    trailing_active && trailing.is_valid();
                result.trailing_t =
                    select(trailing_hit,
                           trailing.t,
                           full<FloatT<Detached>>(Infinity, ray_count));
                result.trailing_prim =
                    select(trailing_hit,
                           trailing.global_prim_id,
                           full<IntT<Detached>>(-1, ray_count));
                result.trailing_dir =
                    select(trailing_active,
                           reflected_direction,
                           zeros<Vector3fT<Detached>>(ray_count));
                result.trailing_origin =
                    select(trailing_active,
                           trailing_origin,
                           zeros<Vector3fT<Detached>>(ray_count));
            }
        }
        return result;
    }

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
                "Scene::trace_reflections(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_reflections(): invalid hitgroup record count.");

        ensure_pipeline(reflection_pipeline_,
                        primary_scene->context(),
                        hitgroup_record_count,
                        reflection_trace_pipeline_config());
    }

    const Mask active_detached = sanitize_reflection_active<Detached>(ray, active);
    if (drjit::none(active_detached)) {
        return result;
    }
    if constexpr (!Detached) {
        drjit::eval(triangle_info_.p0,
                    triangle_info_.e1,
                    triangle_info_.e2,
                    triangle_info_.face_normal);
    }

    Ray broadphase_ray;
    if constexpr (!Detached) {
        broadphase_ray = Ray(detach<false>(ray.o),
                                     detach<false>(ray.d),
                                     detach<false>(ray.tmax));
    } else {
        broadphase_ray = ray;
    }

    drjit::eval(broadphase_ray.o,
                broadphase_ray.d,
                broadphase_ray.tmax,
                active_detached,
                triangle_info_detached_.p0,
                triangle_info_detached_.e1,
                triangle_info_detached_.e2,
                triangle_info_detached_.face_normal,
                face_offsets_);
    if (options.deduplicate && slices(options.canonical_prim_table) > 0) {
        drjit::eval(options.canonical_prim_table);
    }

    ReflectionTraceRaw raw =
        allocate_reflection_trace_raw(ray_count,
                                      max_bounces,
                                      export_mode,
                                      return_trailing,
                                      include_shape_ids);
    initialize_reflection_trace_raw(raw, false);

    ReflectionTraceParams params = {};
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
    params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
    params.ray_ox = broadphase_ray.o.x().data();
    params.ray_oy = broadphase_ray.o.y().data();
    params.ray_oz = broadphase_ray.o.z().data();
    params.ray_dx = broadphase_ray.d.x().data();
    params.ray_dy = broadphase_ray.d.y().data();
    params.ray_dz = broadphase_ray.d.z().data();
    params.ray_tmax = broadphase_ray.tmax.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = ray_count;
    params.max_bounces = max_bounces;
    params.export_mode = export_mode;
    params.return_trailing = return_trailing ? 1 : 0;
    params.out_bounce_count = raw.bounce_count.data();
    params.out_shape_ids = slices(raw.shape_ids) > 0 ? raw.shape_ids.data() : nullptr;
    params.out_prim_ids = slices(raw.prim_ids) > 0 ? raw.prim_ids.data() : nullptr;
    params.out_global_prim_ids =
        slices(raw.global_prim_ids) > 0 ? raw.global_prim_ids.data() : nullptr;
    params.out_t = slices(raw.t) > 0 ? raw.t.data() : nullptr;
    params.out_bary_u = slices(raw.bary_u) > 0 ? raw.bary_u.data() : nullptr;
    params.out_bary_v = slices(raw.bary_v) > 0 ? raw.bary_v.data() : nullptr;
    params.out_hit_x = slices(raw.hit_x) > 0 ? raw.hit_x.data() : nullptr;
    params.out_hit_y = slices(raw.hit_y) > 0 ? raw.hit_y.data() : nullptr;
    params.out_hit_z = slices(raw.hit_z) > 0 ? raw.hit_z.data() : nullptr;
    params.out_norm_x = slices(raw.norm_x) > 0 ? raw.norm_x.data() : nullptr;
    params.out_norm_y = slices(raw.norm_y) > 0 ? raw.norm_y.data() : nullptr;
    params.out_norm_z = slices(raw.norm_z) > 0 ? raw.norm_z.data() : nullptr;
    params.out_img_x = slices(raw.img_x) > 0 ? raw.img_x.data() : nullptr;
    params.out_img_y = slices(raw.img_y) > 0 ? raw.img_y.data() : nullptr;
    params.out_img_z = slices(raw.img_z) > 0 ? raw.img_z.data() : nullptr;
    params.out_trailing_t =
        slices(raw.trailing_t) > 0 ? raw.trailing_t.data() : nullptr;
    params.out_trailing_prim =
        slices(raw.trailing_prim) > 0 ? raw.trailing_prim.data() : nullptr;
    params.out_trailing_dir_x =
        slices(raw.trailing_dir_x) > 0 ? raw.trailing_dir_x.data() : nullptr;
    params.out_trailing_dir_y =
        slices(raw.trailing_dir_y) > 0 ? raw.trailing_dir_y.data() : nullptr;
    params.out_trailing_dir_z =
        slices(raw.trailing_dir_z) > 0 ? raw.trailing_dir_z.data() : nullptr;
    params.out_trailing_origin_x =
        slices(raw.trailing_origin_x) > 0 ? raw.trailing_origin_x.data() : nullptr;
    params.out_trailing_origin_y =
        slices(raw.trailing_origin_y) > 0 ? raw.trailing_origin_y.data() : nullptr;
    params.out_trailing_origin_z =
        slices(raw.trailing_origin_z) > 0 ? raw.trailing_origin_z.data() : nullptr;

    if (cuda_trace) {
        cuda_backend().run_reflection_trace(params, ray_count);
    } else {
        reflection_pipeline_->launch(0, params);
    }

    int trace_ray_count = ray_count;
    Int trace_bounce_count = raw.bounce_count;
    Int trace_discovery_count;
    Int trace_representative_ray_index;
    if (export_mode == RAYD_REFLECTION_EXPORT_FULL) {
        trace_discovery_count =
            select(raw.bounce_count > 0,
                   full<Int>(1, ray_count),
                   full<Int>(0, ray_count));
        trace_representative_ray_index = arange<Int>(ray_count);
    }
    Int trace_shape_ids = raw.shape_ids;
    Int trace_prim_ids = raw.prim_ids;
    Int trace_global_prim_ids = raw.global_prim_ids;
    Float trace_t = raw.t;
    Float trace_hit_x = raw.hit_x;
    Float trace_hit_y = raw.hit_y;
    Float trace_hit_z = raw.hit_z;
    Float trace_norm_x = raw.norm_x;
    Float trace_norm_y = raw.norm_y;
    Float trace_norm_z = raw.norm_z;
    Float trace_img_x = raw.img_x;
    Float trace_img_y = raw.img_y;
    Float trace_img_z = raw.img_z;
    Float trace_trailing_t = raw.trailing_t;
    Int trace_trailing_prim = raw.trailing_prim;
    Float trace_trailing_dir_x = raw.trailing_dir_x;
    Float trace_trailing_dir_y = raw.trailing_dir_y;
    Float trace_trailing_dir_z = raw.trailing_dir_z;
    Float trace_trailing_origin_x = raw.trailing_origin_x;
    Float trace_trailing_origin_y = raw.trailing_origin_y;
    Float trace_trailing_origin_z = raw.trailing_origin_z;

    if (options.deduplicate) {
        ReflectionTraceRaw compacted = allocate_reflection_trace_raw(ray_count, max_bounces);
        initialize_reflection_trace_raw(compacted, false);

        const Int canonical_table = options.canonical_prim_table;
        const int canonical_table_size = static_cast<int>(slices(canonical_table));
        const int n_unique = reflection_dedup_gpu(
            ray_count,
            max_bounces,
            raw.bounce_count.data(),
            raw.shape_ids.data(),
            raw.prim_ids.data(),
            raw.t.data(),
            raw.bary_u.data(),
            raw.bary_v.data(),
            raw.hit_x.data(),
            raw.hit_y.data(),
            raw.hit_z.data(),
            raw.norm_x.data(),
            raw.norm_y.data(),
            raw.norm_z.data(),
            raw.img_x.data(),
            raw.img_y.data(),
            raw.img_z.data(),
            face_offsets_.data(),
            mesh_count_,
            canonical_table_size > 0 ? canonical_table.data() : nullptr,
            canonical_table_size,
            options.image_source_tolerance,
            compacted.bounce_count.data(),
            compacted.shape_ids.data(),
            compacted.prim_ids.data(),
            compacted.t.data(),
            compacted.bary_u.data(),
            compacted.bary_v.data(),
            compacted.hit_x.data(),
            compacted.hit_y.data(),
            compacted.hit_z.data(),
            compacted.norm_x.data(),
            compacted.norm_y.data(),
            compacted.norm_z.data(),
            compacted.img_x.data(),
            compacted.img_y.data(),
            compacted.img_z.data(),
            compacted.discovery_count.data(),
            compacted.representative_ray_index.data());

        trace_ray_count = n_unique;
        const int unique_slot_count = trace_ray_count * max_bounces;
        trace_bounce_count = prefix_array(compacted.bounce_count, trace_ray_count);
        trace_discovery_count = prefix_array(compacted.discovery_count, trace_ray_count);
        trace_representative_ray_index =
            prefix_array(compacted.representative_ray_index, trace_ray_count);
        trace_shape_ids = prefix_array(compacted.shape_ids, unique_slot_count);
        trace_prim_ids = prefix_array(compacted.prim_ids, unique_slot_count);
        trace_t = prefix_array(compacted.t, unique_slot_count);
        trace_hit_x = prefix_array(compacted.hit_x, unique_slot_count);
        trace_hit_y = prefix_array(compacted.hit_y, unique_slot_count);
        trace_hit_z = prefix_array(compacted.hit_z, unique_slot_count);
        trace_norm_x = prefix_array(compacted.norm_x, unique_slot_count);
        trace_norm_y = prefix_array(compacted.norm_y, unique_slot_count);
        trace_norm_z = prefix_array(compacted.norm_z, unique_slot_count);
        trace_img_x = prefix_array(compacted.img_x, unique_slot_count);
        trace_img_y = prefix_array(compacted.img_y, unique_slot_count);
        trace_img_z = prefix_array(compacted.img_z, unique_slot_count);
        const Mask unique_mask = full<Mask>(true, trace_ray_count);
        trace_trailing_t =
            gather<Float>(raw.trailing_t, trace_representative_ray_index, unique_mask);
        trace_trailing_prim =
            gather<Int>(raw.trailing_prim, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_x =
            gather<Float>(raw.trailing_dir_x, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_y =
            gather<Float>(raw.trailing_dir_y, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_z =
            gather<Float>(raw.trailing_dir_z, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_x =
            gather<Float>(raw.trailing_origin_x, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_y =
            gather<Float>(raw.trailing_origin_y, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_z =
            gather<Float>(raw.trailing_origin_z, trace_representative_ray_index, unique_mask);
        result.ray_count = trace_ray_count;
    }

    if (slices(trace_global_prim_ids) == 0 &&
        slices(trace_prim_ids) > 0 &&
        slices(trace_shape_ids) > 0) {
        trace_global_prim_ids =
            globalize_primitive_ids(trace_prim_ids, trace_shape_ids, face_offsets_);
    }

    if constexpr (Detached) {
        result.bounce_count = trace_bounce_count;
        if (slices(trace_discovery_count) > 0) {
            result.discovery_count = trace_discovery_count;
        }
        if (slices(trace_representative_ray_index) > 0) {
            result.representative_ray_index = trace_representative_ray_index;
        }
        if (slices(trace_t) > 0) {
            result.t = trace_t;
        }
        if (slices(trace_hit_x) > 0) {
            const Vector3f hit_points(trace_hit_x, trace_hit_y, trace_hit_z);
            result.hit_points = hit_points;
            result.plane_points = hit_points;
        }
        if (slices(trace_norm_x) > 0) {
            const Vector3f plane_normals(trace_norm_x, trace_norm_y, trace_norm_z);
            result.geo_normals = plane_normals;
            result.plane_normals = plane_normals;
        }
        if (slices(trace_img_x) > 0) {
            result.image_sources = Vector3f(trace_img_x, trace_img_y, trace_img_z);
        }
        if (slices(trace_shape_ids) > 0) {
            result.shape_ids = trace_shape_ids;
        }
        if (slices(trace_prim_ids) > 0) {
            result.prim_ids = trace_prim_ids;
            result.local_prim_ids = trace_prim_ids;
        }
        if (slices(trace_global_prim_ids) > 0) {
            result.global_prim_ids = trace_global_prim_ids;
        }
        if (return_trailing) {
            result.trailing_t = trace_trailing_t;
            result.trailing_prim = trace_trailing_prim;
            result.trailing_dir = Vector3f(trace_trailing_dir_x,
                                                   trace_trailing_dir_y,
                                                   trace_trailing_dir_z);
            result.trailing_origin = Vector3f(trace_trailing_origin_x,
                                                      trace_trailing_origin_y,
                                                      trace_trailing_origin_z);
        }
        return result;
    } else {
        result = initialize_reflection_chain_result<false>(trace_ray_count, max_bounces);
        result.bounce_count = IntAD(trace_bounce_count);
        result.discovery_count = IntAD(trace_discovery_count);
        result.representative_ray_index = IntAD(trace_representative_ray_index);
        result.shape_ids = IntAD(trace_shape_ids);
        result.prim_ids = IntAD(trace_prim_ids);
        result.local_prim_ids = IntAD(trace_prim_ids);
        result.global_prim_ids = IntAD(trace_global_prim_ids);
        result.trailing_t = FloatAD(trace_trailing_t);
        result.trailing_prim = IntAD(trace_trailing_prim);
        result.trailing_dir = Vector3fAD(FloatAD(trace_trailing_dir_x),
                                       FloatAD(trace_trailing_dir_y),
                                       FloatAD(trace_trailing_dir_z));
        result.trailing_origin = Vector3fAD(FloatAD(trace_trailing_origin_x),
                                          FloatAD(trace_trailing_origin_y),
                                          FloatAD(trace_trailing_origin_z));

        if (trace_ray_count == 0) {
            return result;
        }

        const MaskAD representative_mask = full<MaskAD>(true, trace_ray_count);
        const Mask representative_mask_detached =
            full<Mask>(true, trace_ray_count);
        const IntAD representative_ray_index = IntAD(trace_representative_ray_index);
        RayAD current_ray(
            gather<Vector3fAD>(ray.o, representative_ray_index, representative_mask),
            gather<Vector3fAD>(ray.d, representative_ray_index, representative_mask),
            gather<FloatAD>(ray.tmax, representative_ray_index, representative_mask));
        Mask current_active_detached =
            gather<Mask>(active_detached,
                                 trace_representative_ray_index,
                                 representative_mask_detached);
        Vector3fAD current_image_source = current_ray.o;
        const Int bounce_slots =
            arange<Int>(trace_ray_count) * Int(max_bounces);

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot_detached = bounce_slots + bounce;
            const IntAD slot = IntAD(slot_detached);
            const Int shape_id_detached =
                gather<Int>(trace_shape_ids, slot_detached, current_active_detached);
            const Int prim_id_detached =
                gather<Int>(trace_prim_ids, slot_detached, current_active_detached);
            const Mask broadphase_hit =
                current_active_detached && (shape_id_detached >= 0) && (prim_id_detached >= 0);
            if (drjit::none(broadphase_hit)) {
                break;
            }

            const Int mesh_face_offset =
                gather<Int>(face_offsets_, shape_id_detached, broadphase_hit);
            const Int global_prim_detached = mesh_face_offset + prim_id_detached;
            const IntAD global_prim = IntAD(global_prim_detached);
            const MaskAD hit_mask = MaskAD(broadphase_hit);

            const Vector3fAD triangle_p0 = gather<Vector3fAD>(triangle_info_.p0, global_prim, hit_mask);
            const Vector3fAD triangle_e1 = gather<Vector3fAD>(triangle_info_.e1, global_prim, hit_mask);
            const Vector3fAD triangle_e2 = gather<Vector3fAD>(triangle_info_.e2, global_prim, hit_mask);

            Vector2fAD triangle_barycentric;
            FloatAD hit_distance;
            std::tie(triangle_barycentric, hit_distance) =
                ray_intersect_triangle<false>(triangle_p0, triangle_e1, triangle_e2, current_ray);

            MaskAD bounce_hit =
                hit_mask && drjit::isfinite(hit_distance) && (hit_distance < current_ray.tmax);
            const FloatAD safe_t =
                select(bounce_hit, hit_distance, full<FloatAD>(Infinity, trace_ray_count));
            Vector3fAD geo_normal = gather<Vector3fAD>(triangle_info_.face_normal, global_prim, hit_mask);
            geo_normal = normalize(select(hit_mask, geo_normal, Vector3fAD(0.f, 0.f, 1.f)));
            geo_normal = select(dot(current_ray.d, geo_normal) > 0.f, -geo_normal, geo_normal);
            const Vector3fAD hit_point =
                current_ray(select(bounce_hit, safe_t, zeros<FloatAD>(trace_ray_count)));
            const FloatAD plane_distance = dot(current_image_source - hit_point, geo_normal);
            const Vector3fAD reflected_image_source =
                current_image_source - 2.f * plane_distance * geo_normal;

            scatter(result.t, safe_t, slot, bounce_hit);
            scatter(result.hit_points, hit_point, slot, bounce_hit);
            scatter(result.geo_normals, geo_normal, slot, bounce_hit);
            scatter(result.image_sources, reflected_image_source, slot, bounce_hit);
            scatter(result.plane_points, hit_point, slot, bounce_hit);
            scatter(result.plane_normals, geo_normal, slot, bounce_hit);

            const FloatAD ray_dot_normal = dot(current_ray.d, geo_normal);
            const Vector3fAD reflected_direction =
                current_ray.d - 2.f * ray_dot_normal * geo_normal;
            current_ray.o = select(bounce_hit,
                                   hit_point + Epsilon * reflected_direction,
                                   current_ray.o);
            current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
            current_ray.tmax = select(bounce_hit,
                                      full<FloatAD>(Infinity, trace_ray_count),
                                      current_ray.tmax);
            current_image_source =
                select(bounce_hit, reflected_image_source, current_image_source);
            current_active_detached = detach<false>(bounce_hit);
        }

        if (return_trailing) {
            const MaskAD trailing_active = result.bounce_count > 0;
            const IntersectionAD trailing =
                this->template intersect<false>(
                    current_ray, trailing_active, RayFlags::Geometric);
            const MaskAD trailing_hit = trailing_active && trailing.is_valid();
            result.trailing_t =
                select(trailing_hit,
                       trailing.t,
                       full<FloatAD>(Infinity, trace_ray_count));
            result.trailing_prim =
                select(trailing_hit,
                       trailing.global_prim_id,
                       full<IntAD>(-1, trace_ray_count));
            result.trailing_dir =
                select(trailing_active,
                       current_ray.d,
                       zeros<Vector3fAD>(trace_ray_count));
            result.trailing_origin =
                select(trailing_active,
                       current_ray.o,
                       zeros<Vector3fAD>(trace_ray_count));
        }

        return result;
    }
}

template ReflectionChain Scene::trace_reflections<true>(const Ray &ray,
                                                                int max_bounces,
                                                                const ReflectionTraceOptions &options,
                                                                Mask active) const;
template ReflectionChainAD Scene::trace_reflections<false>(const RayAD &ray,
                                                         int max_bounces,
                                                         const ReflectionTraceOptions &options,
                                                         MaskAD active) const;
template ReflectionChain Scene::trace_reflections<true>(const Ray &ray,
                                                                int max_bounces,
                                                                Mask active) const;
template ReflectionChainAD Scene::trace_reflections<false>(const RayAD &ray,
                                                         int max_bounces,
                                                         MaskAD active) const;

template ReflectionTrace Scene::trace_bounces<true>(
    const Ray &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    Mask active) const;
template ReflectionTraceAD Scene::trace_bounces<false>(
    const RayAD &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskAD active) const;
template ReflectionTrace Scene::trace_bounces<true>(
    const Ray &ray,
    int max_bounces,
    Mask active) const;
template ReflectionTraceAD Scene::trace_bounces<false>(
    const RayAD &ray,
    int max_bounces,
    MaskAD active) const;

} // namespace rayd

// Consolidated reflection accumulation host facade.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray/drjit.h>
#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>
#include <src/reflection/dedup_jit.h>
#include <src/reflection/epc_field_jit.h>
#include <src/runtime/optix_pipelines_jit.h>
#include <rayd/diagnostics/drjit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>

#include "multipath_internal_jit.h"

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

// Consolidated reflection EPC host facade.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray/drjit.h>
#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>
#include <src/reflection/dedup_jit.h>
#include <src/reflection/epc_field_jit.h>
#include <src/runtime/optix_pipelines_jit.h>
#include <rayd/diagnostics/drjit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>

#include "multipath_internal_jit.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct ReflEpcRaw {
    int ray_count = 0;
    int max_bounces = 0;
    Mask valid;
    Int bounce_count;
    Float path_length;
    Float point_x;
    Float point_y;
    Float point_z;
    Int trace_prim_ids;
    Int resolved_prim_ids;
    Int surface_group_ids;
    Float plane_normal_x;
    Float plane_normal_y;
    Float plane_normal_z;
    Int first_blocked_segment;
    Int first_blocked_prim;
    Int first_blocked_group;
};

template <bool Detached>
ReflEpcT<Detached> init_refl_epc(int ray_count,
                                                                int max_bounces) {
    ReflEpcT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    result.valid = full<MaskT<Detached>>(false, ray_count);
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.path_length = full<FloatT<Detached>>(Infinity, ray_count);
    result.reflection_points = zeros<Vector3fT<Detached>>(slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trace_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.resolved_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.surface_group_ids = full<IntT<Detached>>(-1, slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_group = full<IntT<Detached>>(-1, ray_count);
    return result;
}

ReflEpcRaw alloc_refl_epc_raw(int ray_count, int max_bounces) {
    const int slot_count = ray_count * max_bounces;
    ReflEpcRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.valid = empty<Mask>(ray_count);
    raw.bounce_count = empty<Int>(ray_count);
    raw.path_length = empty<Float>(ray_count);
    raw.point_x = empty<Float>(slot_count);
    raw.point_y = empty<Float>(slot_count);
    raw.point_z = empty<Float>(slot_count);
    raw.trace_prim_ids = empty<Int>(slot_count);
    raw.resolved_prim_ids = empty<Int>(slot_count);
    raw.surface_group_ids = empty<Int>(slot_count);
    raw.plane_normal_x = empty<Float>(slot_count);
    raw.plane_normal_y = empty<Float>(slot_count);
    raw.plane_normal_z = empty<Float>(slot_count);
    raw.first_blocked_segment = empty<Int>(ray_count);
    raw.first_blocked_prim = empty<Int>(ray_count);
    raw.first_blocked_group = empty<Int>(ray_count);
    return raw;
}

template <bool Detached>
ReflEpcFieldT<Detached> init_refl_epc_field(
    int ray_count,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options) {
    ReflEpcFieldT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    const bool return_geom = options.return_geom;
    const bool return_endpoints = options.return_endpoints;
    const bool return_hit_points =
        return_geom && options.return_hit_points;
    const bool return_normals = return_geom && options.return_normals;
    const bool return_resolved_prim_ids =
        return_geom && options.return_resolved_prim_ids;
    const bool return_surface_group_ids =
        return_geom && options.return_surface_group_ids;

    result.valid = empty<MaskT<Detached>>(ray_count);
    result.bounce_count = empty<IntT<Detached>>(ray_count);
    result.path_length = empty<FloatT<Detached>>(ray_count);
    result.field_x_re = empty<FloatT<Detached>>(ray_count);
    result.field_x_im = empty<FloatT<Detached>>(ray_count);
    result.field_y_re = empty<FloatT<Detached>>(ray_count);
    result.field_y_im = empty<FloatT<Detached>>(ray_count);
    result.field_z_re = empty<FloatT<Detached>>(ray_count);
    result.field_z_im = empty<FloatT<Detached>>(ray_count);

    if (return_endpoints) {
        result.tx_pos =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.first_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.last_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
    } else {
        result.tx_pos = zeros<Vector3fT<Detached>>(0);
        result.first_hit = zeros<Vector3fT<Detached>>(0);
        result.last_hit = zeros<Vector3fT<Detached>>(0);
    }

    if (return_hit_points) {
        result.hit_points =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.hit_points = zeros<Vector3fT<Detached>>(0);
    }
    if (return_normals) {
        result.normals =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.normals = zeros<Vector3fT<Detached>>(0);
    }
    if (return_resolved_prim_ids) {
        result.resolved_prim_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.resolved_prim_ids = full<IntT<Detached>>(-1, 0);
    }
    if (return_surface_group_ids) {
        result.surface_group_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.surface_group_ids = full<IntT<Detached>>(-1, 0);
    }

    return result;
}

ReflEpcOptions epc_options_from_field_options(
    const ReflEpcFieldOptions &options) {
    ReflEpcOptions epc_options;
    epc_options.expected_prim_ids = options.expected_prim_ids;
    epc_options.surface_group_id = options.surface_group_id;
    epc_options.surface_group_size = options.surface_group_size;
    epc_options.surface_group_members = options.surface_group_members;
    epc_options.surface_max_group_size = options.surface_max_group_size;
    epc_options.visibility_ignore_mode = options.visibility_ignore_mode;
    epc_options.final_ignore_group_ids = options.final_ignore_group_ids;
    return epc_options;
}

} // namespace

template <bool Detached>
ReflEpcT<Detached> Scene::trace_refl_epc(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc(): max_bounces exceeds the native EPC limit.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflEpcT<Detached> result =
        init_refl_epc<Detached>(ray_count, max_bounces);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_refl_epc(): native EPC is a non-AD native fast path. "
                "Pass a non-AD Ray and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc(): receiver width must be 1 or match ray count.");
        const ReflEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_refl_epc_vis_ignore(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflEpcVisibilityIgnoreMode::SurfaceGroup;
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(expected_prim_count == 0 || expected_prim_count == slot_count,
                "Scene::trace_refl_epc(): expected_prim_ids width must be n_rays * max_bounces.");
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_refl_epc(): final_ignore_group_ids width must be 1 or match ray count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_refl_epc(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_refl_epc(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_refl_epc(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_refl_epc(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_refl_epc(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_refl_epc(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        const Mask active_detached =
            sanitize_reflection_active<Detached>(ray, active);
        if (drjit::none(active_detached)) {
            return result;
        }

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
                    "Scene::trace_refl_epc(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::trace_refl_epc(): invalid hitgroup record count.");

            ensure_pipeline(reflection_epc_pipeline_, primary_scene->context(),
                            hitgroup_record_count, reflection_epc_pipeline_config());
        }

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    receiver,
                    active_detached);
        if (expected_prim_count > 0) {
            drjit::eval(options.expected_prim_ids);
        }
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ensure_reflection_epc_geometry_ready();

        ReflEpcRaw raw = alloc_refl_epc_raw(ray_count, max_bounces);

        ReflEpcParams params = {};
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
        params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        params.expected_prim_ids =
            expected_prim_count > 0 ? options.expected_prim_ids.data() : nullptr;
        params.expected_prim_count = expected_prim_count;
        params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        params.surface_group_id_count = surface_group_id_count;
        params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        params.surface_group_count = surface_group_count;
        params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        params.visibility_ignore_mode =
            surface_group_ignore ? ReflEpcVisibilityIgnoreSurfaceGroup
                                 : ReflEpcVisibilityIgnorePrimitive;
        params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        params.final_ignore_group_count = final_ignore_group_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.plane_tolerance = options.plane_tolerance;
        params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        params.out_bounce_count = raw.bounce_count.data();
        params.out_path_length = raw.path_length.data();
        params.out_point_x = raw.point_x.data();
        params.out_point_y = raw.point_y.data();
        params.out_point_z = raw.point_z.data();
        params.out_trace_prim_ids = raw.trace_prim_ids.data();
        params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        params.out_surface_group_ids = raw.surface_group_ids.data();
        params.out_plane_normal_x = raw.plane_normal_x.data();
        params.out_plane_normal_y = raw.plane_normal_y.data();
        params.out_plane_normal_z = raw.plane_normal_z.data();
        params.out_first_blocked_segment = raw.first_blocked_segment.data();
        params.out_first_blocked_prim = raw.first_blocked_prim.data();
        params.out_first_blocked_group = raw.first_blocked_group.data();

        if (cuda_trace) {
            cuda_backend().run_reflection_epc(params, /*direct_only=*/false,
                                              /*primary_visibility_only=*/false, ray_count);
        } else {
            reflection_epc_pipeline_->launch(0, params);
        }

        result.valid = raw.valid;
        result.bounce_count = raw.bounce_count;
        result.path_length = raw.path_length;
        result.reflection_points =
            Vector3f(raw.point_x, raw.point_y, raw.point_z);
        result.prim_ids = raw.trace_prim_ids;
        result.trace_prim_ids = raw.trace_prim_ids;
        result.resolved_prim_ids = raw.resolved_prim_ids;
        result.surface_group_ids = raw.surface_group_ids;
        result.plane_normals =
            Vector3f(raw.plane_normal_x,
                             raw.plane_normal_y,
                             raw.plane_normal_z);
        result.first_blocked_segment = raw.first_blocked_segment;
        result.first_blocked_prim = raw.first_blocked_prim;
        result.first_blocked_group = raw.first_blocked_group;
        return result;
    }
}

template <bool Detached>
ReflEpcFieldT<Detached> Scene::trace_refl_epc_field(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_refl_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_refl_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflEpcFieldT<Detached> result =
        init_refl_epc_field<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_refl_epc_field(): native EPC field is a non-AD native fast path. "
                "Pass a non-AD Ray and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match ray count.");
        const int slot_count = ray_count * max_bounces;
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match ray count.");

        drjit::eval(options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);

        const ReflEpcOptions epc_options =
            epc_options_from_field_options(options);
        const ReflEpc epc =
            trace_refl_epc<true>(
                ray,
                receiver,
                max_bounces,
                epc_options,
                active);

        ReflEpcFieldParams params = {};
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.epc_valid = reinterpret_cast<const uint8_t *>(epc.valid.data());
        params.epc_bounce_count = epc.bounce_count.data();
        params.epc_path_length = epc.path_length.data();
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.hit_x = epc.reflection_points.x().data();
        params.hit_y = epc.reflection_points.y().data();
        params.hit_z = epc.reflection_points.z().data();
        params.epc_normal_x = epc.plane_normals.x().data();
        params.epc_normal_y = epc.plane_normals.y().data();
        params.epc_normal_z = epc.plane_normals.z().data();
        const bool return_resolved_prim_ids =
            options.return_geom && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geom && options.return_surface_group_ids;
        params.resolved_prim_ids =
            return_resolved_prim_ids ? epc.resolved_prim_ids.data() : nullptr;
        params.surface_group_ids =
            return_surface_group_ids ? epc.surface_group_ids.data() : nullptr;
        params.slot_normal_x = options.slot_plane_normal.x().data();
        params.slot_normal_y = options.slot_plane_normal.y().data();
        params.slot_normal_z = options.slot_plane_normal.z().data();
        params.slot_eta_r = options.slot_eta_r.data();
        params.slot_mu_r = options.slot_mu_r.data();
        params.slot_sigma = options.slot_sigma.data();
        params.slot_gain = options.slot_gain.data();
        params.tx_pol_x = options.tx_polarization.x().data();
        params.tx_pol_y = options.tx_polarization.y().data();
        params.tx_pol_z = options.tx_polarization.z().data();
        params.tx_pol_count = tx_pol_count;
        params.omega = options.omega;
        params.wavelength = options.wavelength;
        params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        params.out_bounce_count = result.bounce_count.data();
        params.out_path_length = result.path_length.data();
        params.out_field_x_re = result.field_x_re.data();
        params.out_field_x_im = result.field_x_im.data();
        params.out_field_y_re = result.field_y_re.data();
        params.out_field_y_im = result.field_y_im.data();
        params.out_field_z_re = result.field_z_re.data();
        params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            params.out_tx_x = result.tx_pos.x().data();
            params.out_tx_y = result.tx_pos.y().data();
            params.out_tx_z = result.tx_pos.z().data();
            params.out_first_hit_x = result.first_hit.x().data();
            params.out_first_hit_y = result.first_hit.y().data();
            params.out_first_hit_z = result.first_hit.z().data();
            params.out_last_hit_x = result.last_hit.x().data();
            params.out_last_hit_y = result.last_hit.y().data();
            params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geom && options.return_hit_points) {
            params.out_hit_x = result.hit_points.x().data();
            params.out_hit_y = result.hit_points.y().data();
            params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geom && options.return_normals) {
            params.out_normal_x = result.normals.x().data();
            params.out_normal_y = result.normals.y().data();
            params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(params);
        return result;
    }
}

template <bool Detached>
ReflEpcFieldT<Detached> Scene::trace_refl_epc_field(
    const Vector3fT<Detached> &tx_position,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_refl_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_refl_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(tx_position));
    ReflEpcFieldT<Detached> result =
        init_refl_epc_field<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_refl_epc_field(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match transmitter count.");

        Mask active_detached = sanitize_segment_active<false>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            result.valid = full<MaskAD>(false, ray_count);
            result.bounce_count = full<IntAD>(0, ray_count);
            result.path_length = full<FloatAD>(Infinity, ray_count);
            result.field_x_re = zeros<FloatAD>(ray_count);
            result.field_x_im = zeros<FloatAD>(ray_count);
            result.field_y_re = zeros<FloatAD>(ray_count);
            result.field_y_im = zeros<FloatAD>(ray_count);
            result.field_z_re = zeros<FloatAD>(ray_count);
            result.field_z_im = zeros<FloatAD>(ray_count);
            return result;
        }

        const Int slot_base = arange<Int>(ray_count) * Int(max_bounces);
        const MaskAD active_ad = MaskAD(active_detached);

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
        auto slot_reflection_coefficients =
            [&](const IntAD &slot,
                const FloatAD &cos_theta,
                const MaskAD &slot_active) -> std::pair<ComplexADValue, ComplexADValue> {
            const FloatAD eta_r =
                maximum(gather<FloatAD>(options.slot_eta_r, slot, slot_active),
                        FloatAD(Epsilon));
            const FloatAD sigma =
                maximum(gather<FloatAD>(options.slot_sigma, slot, slot_active),
                        FloatAD(0.f));
            const FloatAD gain = gather<FloatAD>(options.slot_gain, slot, slot_active);
            const FloatAD mu_r =
                maximum(gather<FloatAD>(options.slot_mu_r, slot, slot_active),
                        FloatAD(Epsilon));
            const FloatAD omega = maximum(FloatAD(options.omega), FloatAD(Epsilon));
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
            const ComplexADValue r_te =
                complex_scale(
                    complex_div(complex_sub(mu_cos, a),
                                complex_add(mu_cos, a)),
                    gain);
            const ComplexADValue r_tm =
                complex_scale(
                    complex_div(complex_sub(eta_cos, a),
                                complex_add(eta_cos, a)),
                    gain);
            return {r_te, r_tm};
        };
        auto reflect_field_vector =
            [&](const ComplexVectorAD &field,
                const Vector3fAD &incident_dir,
                const Vector3fAD &slot_normal,
                const IntAD &slot,
                const MaskAD &slot_active) -> ComplexVectorAD {
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
                slot_reflection_coefficients(slot, abs(dot(incident_hat, normal_hat)), slot_active);
            const ComplexADValue e_s = complex_dot_real(field, s_hat);
            const ComplexADValue e_p = complex_dot_real(field, p_in_hat);
            const ComplexADValue out_s = complex_mul(r_te, e_s);
            const ComplexADValue out_p = complex_mul(r_tm, e_p);
            return {
                s_hat * out_s.re + p_out_hat * out_p.re,
                s_hat * out_s.im + p_out_hat * out_p.im,
            };
        };

        std::vector<Vector3fAD> images;
        images.reserve(static_cast<size_t>(max_bounces) + 1);
        images.push_back(tx_position);
        MaskAD valid = active_ad;

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            Vector3fAD plane_point =
                gather<Vector3fAD>(options.slot_plane_point, slot_ad, active_ad);
            Vector3fAD plane_normal =
                normalize_safe(gather<Vector3fAD>(options.slot_plane_normal, slot_ad, active_ad),
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            const Int expected_prim =
                gather<Int>(options.expected_prim_ids, slot, active_detached);
            valid = valid && MaskAD(expected_prim >= Int(0)) &&
                    (squared_norm(plane_normal) > FloatAD(0.f));
            const FloatAD plane_distance =
                dot(images.back() - plane_point, plane_normal);
            images.push_back(
                select(valid,
                       images.back() - FloatAD(2.f) * plane_distance * plane_normal,
                       images.back()));
        }

        Vector3fAD rx = receiver;
        if (receiver_count == 1 && ray_count > 1) {
            rx = gather<Vector3fAD>(receiver, zeros<IntAD>(ray_count), full<MaskAD>(true, ray_count));
        }
        Vector3fAD target = rx;
        std::vector<Vector3fAD> hits(static_cast<size_t>(max_bounces));
        std::vector<Vector3fAD> normals(static_cast<size_t>(max_bounces));
        for (int bounce = max_bounces - 1; bounce >= 0; --bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            const Vector3fAD plane_point =
                gather<Vector3fAD>(options.slot_plane_point, slot_ad, active_ad);
            const Vector3fAD plane_normal =
                normalize_safe(gather<Vector3fAD>(options.slot_plane_normal, slot_ad, active_ad),
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            const Vector3fAD line = target - images[static_cast<size_t>(bounce + 1)];
            const FloatAD denom = dot(line, plane_normal);
            const FloatAD t =
                dot(plane_point - images[static_cast<size_t>(bounce + 1)], plane_normal) /
                denom;
            const MaskAD hit_valid =
                valid &&
                drjit::isfinite(t) &&
                (abs(denom) > FloatAD(Epsilon)) &&
                (t > FloatAD(0.f)) &&
                (t < FloatAD(1.f));
            const Vector3fAD hit =
                images[static_cast<size_t>(bounce + 1)] + t * line;
            hits[static_cast<size_t>(bounce)] = select(hit_valid, hit, zeros<Vector3fAD>(ray_count));
            normals[static_cast<size_t>(bounce)] =
                select(hit_valid, plane_normal, zeros<Vector3fAD>(ray_count));
            if (options.return_geom && options.return_hit_points) {
                scatter(result.hit_points, hits[static_cast<size_t>(bounce)], slot_ad, hit_valid);
            }
            if (options.return_geom && options.return_normals) {
                scatter(result.normals, normals[static_cast<size_t>(bounce)], slot_ad, hit_valid);
            }
            target = select(hit_valid, hit, target);
            valid = hit_valid;
        }

        FloatAD path_length = zeros<FloatAD>(ray_count);
        Vector3fAD previous = tx_position;
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Vector3fAD hit = hits[static_cast<size_t>(bounce)];
            path_length += norm(hit - previous);
            previous = hit;
        }
        path_length += norm(rx - previous);
        valid = valid && (path_length > FloatAD(Epsilon)) && drjit::isfinite(path_length);

        const Int pol_idx =
            tx_pol_count == 1 ? zeros<Int>(ray_count) : arange<Int>(ray_count);
        const Vector3fAD tx_pol =
            gather<Vector3fAD>(options.tx_polarization, IntAD(pol_idx), active_ad);
        const Vector3fAD first_dir =
            normalize_safe(hits.front() - tx_position,
                           Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
        const Vector3fAD transverse_pol =
            stable_perpendicular(first_dir, tx_pol);
        ComplexVectorAD field = {
            transverse_pol,
            zeros<Vector3fAD>(ray_count),
        };
        Vector3fAD field_previous = tx_position;
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            const Vector3fAD hit = hits[static_cast<size_t>(bounce)];
            const Vector3fAD incident_dir =
                normalize_safe(hit - field_previous,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            field = reflect_field_vector(
                field,
                incident_dir,
                normals[static_cast<size_t>(bounce)],
                slot_ad,
                active_ad);
            field_previous = hit;
        }
        const FloatAD wave_k =
            FloatAD(2.f * Pi) / maximum(FloatAD(options.wavelength), FloatAD(Epsilon));
        const FloatAD phase = -wave_k * path_length;
        const FloatAD amplitude =
            FloatAD(options.wavelength) /
            (FloatAD(4.f * Pi) * maximum(path_length, FloatAD(Epsilon)));
        const FloatAD phase_cos = cos(phase);
        const FloatAD phase_sin = sin(phase);
        const Vector3fAD out_re =
            amplitude * (field.re * phase_cos - field.im * phase_sin);
        const Vector3fAD out_im =
            amplitude * (field.re * phase_sin + field.im * phase_cos);
        valid = valid &&
                drjit::isfinite(out_re.x()) &&
                drjit::isfinite(out_re.y()) &&
                drjit::isfinite(out_re.z()) &&
                drjit::isfinite(out_im.x()) &&
                drjit::isfinite(out_im.y()) &&
                drjit::isfinite(out_im.z());

        result.valid = valid;
        result.bounce_count =
            select(valid, full<IntAD>(max_bounces, ray_count), full<IntAD>(0, ray_count));
        result.path_length =
            select(valid, path_length, full<FloatAD>(Infinity, ray_count));
        result.field_x_re = select(valid, out_re.x(), FloatAD(0.f));
        result.field_x_im = select(valid, out_im.x(), FloatAD(0.f));
        result.field_y_re = select(valid, out_re.y(), FloatAD(0.f));
        result.field_y_im = select(valid, out_im.y(), FloatAD(0.f));
        result.field_z_re = select(valid, out_re.z(), FloatAD(0.f));
        result.field_z_im = select(valid, out_im.z(), FloatAD(0.f));

        if (options.return_endpoints) {
            result.tx_pos = tx_position;
            result.first_hit = max_bounces > 0 ? hits.front() : zeros<Vector3fAD>(ray_count);
            result.last_hit = max_bounces > 0 ? hits.back() : zeros<Vector3fAD>(ray_count);
        }
        if (options.return_geom && options.return_resolved_prim_ids) {
            result.resolved_prim_ids = IntAD(options.expected_prim_ids);
        }
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_refl_epc_field(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match transmitter count.");

        const ReflEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_refl_epc_vis_ignore(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflEpcVisibilityIgnoreMode::SurfaceGroup;
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_refl_epc_field(): final_ignore_group_ids width must be 1 or match transmitter count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_refl_epc_field(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_refl_epc_field(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_refl_epc_field(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_refl_epc_field(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_refl_epc_field(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_refl_epc_field(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        Mask active_detached = sanitize_segment_active<Detached>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        std::shared_ptr<OptixLaunchPipeline> *epc_pipeline = nullptr;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            const int hitgroup_record_count = scenes.hitgroup_record_count;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::trace_refl_epc_field(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::trace_refl_epc_field(): invalid hitgroup record count.");

            epc_pipeline = split_mode == 0 ? &reflection_epc_direct_primary_pipeline_
                                           : &reflection_epc_direct_pipeline_;
            const OptixPipelineConfig epc_pipeline_config =
                split_mode == 0 ? reflection_epc_direct_primary_pipeline_config()
                                : reflection_epc_direct_pipeline_config();

            ensure_pipeline(*epc_pipeline, primary_scene->context(),
                            hitgroup_record_count, epc_pipeline_config);
        }

        drjit::eval(tx_position,
                    receiver,
                    active_detached,
                    options.expected_prim_ids,
                    options.slot_plane_point,
                    options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ensure_reflection_epc_geometry_ready();

        ReflEpcRaw raw = alloc_refl_epc_raw(ray_count, max_bounces);
        ReflEpcParams epc_params = {};
        epc_params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        epc_params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        epc_params.split_mode = split_mode;
        epc_params.tri_p0_x = triangle_info_detached_.p0.x().data();
        epc_params.tri_p0_y = triangle_info_detached_.p0.y().data();
        epc_params.tri_p0_z = triangle_info_detached_.p0.z().data();
        epc_params.tri_e1_x = triangle_info_detached_.e1.x().data();
        epc_params.tri_e1_y = triangle_info_detached_.e1.y().data();
        epc_params.tri_e1_z = triangle_info_detached_.e1.z().data();
        epc_params.tri_e2_x = triangle_info_detached_.e2.x().data();
        epc_params.tri_e2_y = triangle_info_detached_.e2.y().data();
        epc_params.tri_e2_z = triangle_info_detached_.e2.z().data();
        epc_params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        epc_params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        epc_params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        epc_params.face_offsets = face_offsets_.data();
        epc_params.n_meshes = mesh_count_;
        epc_params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        epc_params.expected_prim_ids = options.expected_prim_ids.data();
        epc_params.expected_prim_count = expected_prim_count;
        epc_params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        epc_params.surface_group_id_count = surface_group_id_count;
        epc_params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        epc_params.surface_group_count = surface_group_count;
        epc_params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        epc_params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        epc_params.visibility_ignore_mode =
            surface_group_ignore ? ReflEpcVisibilityIgnoreSurfaceGroup
                                 : ReflEpcVisibilityIgnorePrimitive;
        epc_params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        epc_params.final_ignore_group_count = final_ignore_group_count;
        epc_params.ray_ox = tx_position.x().data();
        epc_params.ray_oy = tx_position.y().data();
        epc_params.ray_oz = tx_position.z().data();
        epc_params.ray_dx = nullptr;
        epc_params.ray_dy = nullptr;
        epc_params.ray_dz = nullptr;
        epc_params.ray_tmax = nullptr;
        epc_params.direct_plane_point_x = options.slot_plane_point.x().data();
        epc_params.direct_plane_point_y = options.slot_plane_point.y().data();
        epc_params.direct_plane_point_z = options.slot_plane_point.z().data();
        epc_params.direct_plane_normal_x = options.slot_plane_normal.x().data();
        epc_params.direct_plane_normal_y = options.slot_plane_normal.y().data();
        epc_params.direct_plane_normal_z = options.slot_plane_normal.z().data();
        epc_params.rx_x = receiver.x().data();
        epc_params.rx_y = receiver.y().data();
        epc_params.rx_z = receiver.z().data();
        epc_params.rx_count = receiver_count;
        epc_params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        epc_params.n_rays = ray_count;
        epc_params.max_bounces = max_bounces;
        epc_params.plane_tolerance = options.plane_tolerance;
        epc_params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        epc_params.out_bounce_count = raw.bounce_count.data();
        epc_params.out_path_length = raw.path_length.data();
        epc_params.out_point_x = raw.point_x.data();
        epc_params.out_point_y = raw.point_y.data();
        epc_params.out_point_z = raw.point_z.data();
        epc_params.out_trace_prim_ids = raw.trace_prim_ids.data();
        epc_params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        epc_params.out_surface_group_ids = raw.surface_group_ids.data();
        epc_params.out_plane_normal_x = raw.plane_normal_x.data();
        epc_params.out_plane_normal_y = raw.plane_normal_y.data();
        epc_params.out_plane_normal_z = raw.plane_normal_z.data();
        epc_params.out_first_blocked_segment = raw.first_blocked_segment.data();
        epc_params.out_first_blocked_prim = raw.first_blocked_prim.data();
        epc_params.out_first_blocked_group = raw.first_blocked_group.data();
        if (cuda_trace) {
            // Single-scene CUDA: split_mode is 0, so the direct-primary variant.
            cuda_backend().run_reflection_epc(epc_params, /*direct_only=*/true,
                                              /*primary_visibility_only=*/true, ray_count);
        } else {
            (*epc_pipeline)->launch(0, epc_params);
        }

        ReflEpcFieldParams field_params = {};
        field_params.n_rays = ray_count;
        field_params.max_bounces = max_bounces;
        field_params.epc_valid = reinterpret_cast<const uint8_t *>(raw.valid.data());
        field_params.epc_bounce_count = raw.bounce_count.data();
        field_params.epc_path_length = raw.path_length.data();
        field_params.ray_ox = tx_position.x().data();
        field_params.ray_oy = tx_position.y().data();
        field_params.ray_oz = tx_position.z().data();
        field_params.rx_x = receiver.x().data();
        field_params.rx_y = receiver.y().data();
        field_params.rx_z = receiver.z().data();
        field_params.rx_count = receiver_count;
        field_params.hit_x = raw.point_x.data();
        field_params.hit_y = raw.point_y.data();
        field_params.hit_z = raw.point_z.data();
        field_params.epc_normal_x = raw.plane_normal_x.data();
        field_params.epc_normal_y = raw.plane_normal_y.data();
        field_params.epc_normal_z = raw.plane_normal_z.data();
        const bool return_resolved_prim_ids =
            options.return_geom && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geom && options.return_surface_group_ids;
        field_params.resolved_prim_ids =
            return_resolved_prim_ids ? raw.resolved_prim_ids.data() : nullptr;
        field_params.surface_group_ids =
            return_surface_group_ids ? raw.surface_group_ids.data() : nullptr;
        field_params.slot_normal_x = options.slot_plane_normal.x().data();
        field_params.slot_normal_y = options.slot_plane_normal.y().data();
        field_params.slot_normal_z = options.slot_plane_normal.z().data();
        field_params.slot_eta_r = options.slot_eta_r.data();
        field_params.slot_mu_r = options.slot_mu_r.data();
        field_params.slot_sigma = options.slot_sigma.data();
        field_params.slot_gain = options.slot_gain.data();
        field_params.tx_pol_x = options.tx_polarization.x().data();
        field_params.tx_pol_y = options.tx_polarization.y().data();
        field_params.tx_pol_z = options.tx_polarization.z().data();
        field_params.tx_pol_count = tx_pol_count;
        field_params.omega = options.omega;
        field_params.wavelength = options.wavelength;
        field_params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        field_params.out_bounce_count = result.bounce_count.data();
        field_params.out_path_length = result.path_length.data();
        field_params.out_field_x_re = result.field_x_re.data();
        field_params.out_field_x_im = result.field_x_im.data();
        field_params.out_field_y_re = result.field_y_re.data();
        field_params.out_field_y_im = result.field_y_im.data();
        field_params.out_field_z_re = result.field_z_re.data();
        field_params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            field_params.out_tx_x = result.tx_pos.x().data();
            field_params.out_tx_y = result.tx_pos.y().data();
            field_params.out_tx_z = result.tx_pos.z().data();
            field_params.out_first_hit_x = result.first_hit.x().data();
            field_params.out_first_hit_y = result.first_hit.y().data();
            field_params.out_first_hit_z = result.first_hit.z().data();
            field_params.out_last_hit_x = result.last_hit.x().data();
            field_params.out_last_hit_y = result.last_hit.y().data();
            field_params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geom && options.return_hit_points) {
            field_params.out_hit_x = result.hit_points.x().data();
            field_params.out_hit_y = result.hit_points.y().data();
            field_params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geom && options.return_normals) {
            field_params.out_normal_x = result.normals.x().data();
            field_params.out_normal_y = result.normals.y().data();
            field_params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            field_params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            field_params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(field_params);
        return result;
    }
}

template ReflEpc Scene::trace_refl_epc<true>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    Mask active) const;
template ReflEpcAD Scene::trace_refl_epc<false>(
    const RayAD &ray,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    MaskAD active) const;
template ReflEpcField Scene::trace_refl_epc_field<true>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcFieldOptions &options,
    Mask active) const;
template ReflEpcFieldAD Scene::trace_refl_epc_field<false>(
    const RayAD &ray,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsAD &options,
    MaskAD active) const;
template ReflEpcField Scene::trace_refl_epc_field<true>(
    const Vector3f &tx_position,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcFieldOptions &options,
    Mask active) const;
template ReflEpcFieldAD Scene::trace_refl_epc_field<false>(
    const Vector3fAD &tx_position,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsAD &options,
    MaskAD active) const;

} // namespace rayd
