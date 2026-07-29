#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/jit/ray.h>
#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>
#include <src/reflection/dedup_jit.h>
#include <src/reflection/epc_field_jit.h>
#include <src/runtime/optix_pipelines_jit.h>
#include <rayd/jit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>

namespace rayd {

namespace multipath_detail {

/// Backend for segment-visibility traces (env RAYD_TRACE_VISIBILITY_BACKEND): Dr.Jit HitObject vs. native optixLaunch.
enum class TraceVisibilityBackend {
    Auto,
    Jit,
    Native
};

/// How EPC visibility decides which primitives to ignore: exact primitive ids vs. surface groups.
enum class ReflEpcVisibilityIgnoreMode {
    Primitive,
    SurfaceGroup
};

TraceVisibilityBackend active_trace_visibility_backend();

ReflEpcVisibilityIgnoreMode parse_refl_epc_vis_ignore(
    const std::string &value);

bool use_jit_trace_visibility_path(int ignore_k);

bool recording_reflections();

bool uses_symbolic_optix_query_path();

void ensure_pipeline(std::shared_ptr<OptixLaunchPipeline> &pipeline,
                     OptixDeviceContext context,
                     int hitgroup_record_count,
                     const OptixPipelineConfig &config);

template <bool Detached>
ReflectionChainT<Detached> initialize_reflection_chain_result(
    int ray_count,
    int max_bounces,
    int export_mode = RAYD_REFLECTION_EXPORT_FULL,
    bool return_trailing = true,
    bool include_shape_ids = true) {
    ReflectionChainT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;

    const int slot_count = ray_count * max_bounces;
    const bool full_export = export_mode == RAYD_REFLECTION_EXPORT_FULL;
    const bool minimal_export = export_mode == RAYD_REFLECTION_EXPORT_MINIMAL;
    const bool slot_ids_export = full_export || minimal_export;
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    if (full_export) {
        result.discovery_count = full<IntT<Detached>>(0, ray_count);
        result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    }
    if (slot_ids_export) {
        result.t = full<FloatT<Detached>>(Infinity, slot_count);
        result.prim_ids = full<IntT<Detached>>(-1, slot_count);
        result.local_prim_ids = full<IntT<Detached>>(-1, slot_count);
        result.global_prim_ids = full<IntT<Detached>>(-1, slot_count);
    }
    if (slot_ids_export && include_shape_ids) {
        result.shape_ids = full<IntT<Detached>>(-1, slot_count);
    }
    if (full_export) {
        result.hit_points = zeros<Vector3fT<Detached>>(slot_count);
        result.geo_normals = zeros<Vector3fT<Detached>>(slot_count);
        result.image_sources = zeros<Vector3fT<Detached>>(slot_count);
        result.plane_points = zeros<Vector3fT<Detached>>(slot_count);
        result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    }
    if (return_trailing) {
        result.trailing_t = full<FloatT<Detached>>(Infinity, ray_count);
        result.trailing_prim = full<IntT<Detached>>(-1, ray_count);
        result.trailing_dir = zeros<Vector3fT<Detached>>(ray_count);
        result.trailing_origin = zeros<Vector3fT<Detached>>(ray_count);
    }
    return result;
}

template <bool Detached>
Mask sanitize_reflection_active(const RayT<Detached> &ray,
                                        MaskT<Detached> active) {
    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3f(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(detach<false>(ray.tmax)) ||
                           (detach<false>(ray.tmax) > 0.f);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) &&
                           drjit::isfinite(ray.o.y()) &&
                           drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) &&
                           drjit::isfinite(ray.d.y()) &&
                           drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(ray.tmax) || (ray.tmax > 0.f);
    }
    return active_detached;
}

template <bool Detached>
Mask sanitize_segment_active(const Vector3fT<Detached> &start,
                                     const Vector3fT<Detached> &end,
                                     MaskT<Detached> active) {
    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(start.x())) &&
                           drjit::isfinite(detach<false>(start.y())) &&
                           drjit::isfinite(detach<false>(start.z()));
        active_detached &= drjit::isfinite(detach<false>(end.x())) &&
                           drjit::isfinite(detach<false>(end.y())) &&
                           drjit::isfinite(detach<false>(end.z()));
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(start.x()) &&
                           drjit::isfinite(start.y()) &&
                           drjit::isfinite(start.z());
        active_detached &= drjit::isfinite(end.x()) &&
                           drjit::isfinite(end.y()) &&
                           drjit::isfinite(end.z());
    }
    return active_detached;
}

inline void eval_segment_visibility_common(const Vector3f &start,
                                    const Int &face_offsets,
                                    const Int &ignore_prim_ids,
                                    int ignore_k,
                                    const Mask &active_detached) {
    if (ignore_k > 0) {
        drjit::eval(start, face_offsets, ignore_prim_ids, active_detached);
    } else {
        drjit::eval(start, face_offsets, active_detached);
    }
}

inline SegmentVisibilityParams make_segment_visibility_params(
    const OptixScene &optix_scene,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached,
    int ray_count) {
    SegmentVisibilityParams params = {};
    params.handle = optix_scene.ias_handle();
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.start_x = start.x().data();
    params.start_y = start.y().data();
    params.start_z = start.z().data();
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_ids.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = ray_count;
    return params;
}

struct SegmentVisibilityLaunchResult {
    Mask visible;
    Int blocker_prim;
};

inline SegmentVisibilityLaunchResult launch_segment_visibility_detached(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Vector3f &end,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached,
    bool collect_blocker_prim = false) {
    const int ray_count = static_cast<int>(slices(start));
    if (ray_count == 0) {
        return {Mask(), Int()};
    }

    Mask visible = empty<Mask>(ray_count);
    Int blocker_prim = collect_blocker_prim
        ? full<Int>(-1, ray_count)
        : Int();
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       start,
                                       ignore_prim_ids,
                                       ignore_k,
                                       active_detached,
                                       ray_count);
    params.end_x = end.x().data();
    params.end_y = end.y().data();
    params.end_z = end.z().data();
    params.out_visible = reinterpret_cast<uint8_t *>(visible.data());
    params.out_first_blocked_prim =
        collect_blocker_prim ? blocker_prim.data() : nullptr;
    pipeline.launch(0, params);
    return {visible, blocker_prim};
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Vector3f &end,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const SegmentVisibilityLaunchResult launched =
        launch_segment_visibility_detached(optix_scene,
                                           pipeline,
                                           face_offsets,
                                           mesh_count,
                                           start,
                                           end,
                                           ignore_prim_ids,
                                           ignore_k,
                                           active_detached);
    if constexpr (!Detached) {
        result.visible = MaskAD(launched.visible);
    } else {
        result.visible = launched.visible;
    }
    return result;
}

} // namespace multipath_detail

} // namespace rayd
