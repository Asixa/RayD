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
