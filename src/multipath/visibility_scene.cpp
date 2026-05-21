#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/edge/scene_edge.h>

#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>

#include "../scene/scene_internal.h"

namespace rayd {

namespace {

enum class TraceVisibilityBackend {
    Auto,
    Jit,
    Native
};

TraceVisibilityBackend active_trace_visibility_backend() {
    static const TraceVisibilityBackend value = []() {
        const char *raw = std::getenv("RAYD_TRACE_VISIBILITY_BACKEND");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return TraceVisibilityBackend::Auto;
        }
        if (normalized == "jit" || normalized == "drjit" ||
            normalized == "hitobject" || normalized == "hit_object") {
            return TraceVisibilityBackend::Jit;
        }
        if (normalized == "native" || normalized == "optixlaunch" ||
            normalized == "optix_launch") {
            return TraceVisibilityBackend::Native;
        }
        throw std::runtime_error(
            "Invalid RAYD_TRACE_VISIBILITY_BACKEND. Expected one of: auto, jit, native.");
    }();
    return value;
}

bool use_jit_trace_visibility_path(int ignore_k) {
    const TraceVisibilityBackend backend = active_trace_visibility_backend();
    if (backend == TraceVisibilityBackend::Native) {
        return false;
    }
    if (backend == TraceVisibilityBackend::Jit) {
        require(ignore_k == 0,
                "RAYD_TRACE_VISIBILITY_BACKEND=jit does not support ignore lists yet.");
        return true;
    }
    return ignore_k == 0;
}

void eval_segment_visibility_common(const Vector3fDetached &start,
                                    const IntDetached &face_offsets,
                                    const IntDetached &ignore_prim_ids,
                                    int ignore_k,
                                    const MaskDetached &active_detached) {
    if (ignore_k > 0) {
        drjit::eval(start, face_offsets, ignore_prim_ids, active_detached);
    } else {
        drjit::eval(start, face_offsets, active_detached);
    }
}

SegmentVisibilityParams make_segment_visibility_params(
    const OptixScene &optix_scene,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached,
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

MaskDetached launch_segment_visibility_detached(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    if (ray_count == 0) {
        return MaskDetached();
    }

    MaskDetached visible = empty<MaskDetached>(ray_count);
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
    pipeline.launch(static_cast<int>(SegmentVisibilityLaunchKind::Segment), params);
    return visible;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit =
        optix_scene.segment_hit<true>(start, end, active_detached);
    if constexpr (!Detached) {
        result.visible = Mask(hit.visible);
    } else {
        result.visible = hit.visible;
    }
    return result;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const MaskDetached visible_detached =
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
        result.visible = Mask(visible_detached);
    } else {
        result.visible = visible_detached;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3fDetached &start,
    const Vector3fDetached &end_a,
    const Vector3fDetached &end_b,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit_a =
        optix_scene.segment_hit<true>(start, end_a, active_detached);
    const OptixSegmentHit hit_b =
        optix_scene.segment_hit<true>(start, end_b, active_detached);
    if constexpr (!Detached) {
        result.visible_a = Mask(hit_a.visible);
        result.visible_b = Mask(hit_b.visible);
    } else {
        result.visible_a = hit_a.visible;
        result.visible_b = hit_b.visible;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const Vector3fDetached &end_a,
    const Vector3fDetached &end_b,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    MaskDetached visible_a = empty<MaskDetached>(ray_count);
    MaskDetached visible_b = empty<MaskDetached>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end_a, end_b);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       start,
                                       ignore_prim_ids,
                                       ignore_k,
                                       active_detached,
                                       ray_count);
    params.end_x = end_a.x().data();
    params.end_y = end_a.y().data();
    params.end_z = end_a.z().data();
    params.end_b_x = end_b.x().data();
    params.end_b_y = end_b.y().data();
    params.end_b_z = end_b.z().data();
    params.out_visible = reinterpret_cast<uint8_t *>(visible_a.data());
    params.out_visible_b = reinterpret_cast<uint8_t *>(visible_b.data());
    pipeline.launch(static_cast<int>(SegmentVisibilityLaunchKind::SegmentPair), params);

    if constexpr (!Detached) {
        result.visible_a = Mask(visible_a);
        result.visible_b = Mask(visible_b);
    } else {
        result.visible_a = visible_a;
        result.visible_b = visible_b;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_jit(
    const OptixScene &optix_scene,
    const Vector3fDetached &source_pos,
    const Vector3fDetached &edge_pos,
    const Vector3fDetached &edge_dir,
    const FloatDetached &edge_line_min,
    const FloatDetached &edge_line_max,
    const std::vector<float> &sample_fractions,
    const MaskDetached &active_detached) {
    const int state_count = static_cast<int>(slices(source_pos));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    MaskDetached any_visible = full<MaskDetached>(false, state_count);
    const FloatDetached span =
        maximum(edge_line_max - edge_line_min, FloatDetached(0.f));
    for (float fraction : sample_fractions) {
        const FloatDetached sample_t = edge_line_min + fraction * span;
        const Vector3fDetached sample_pos = edge_pos + sample_t * edge_dir;
        const OptixSegmentHit hit =
            optix_scene.segment_hit<true>(source_pos, sample_pos, active_detached);
        any_visible = any_visible || hit.visible;
    }

    if constexpr (!Detached) {
        result.any_visible = Mask(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &source_pos,
    const Vector3fDetached &edge_pos,
    const Vector3fDetached &edge_dir,
    const FloatDetached &edge_line_min,
    const FloatDetached &edge_line_max,
    const std::vector<float> &sample_fractions,
    const MaskDetached &active_detached) {
    const int state_count = static_cast<int>(slices(source_pos));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    MaskDetached any_visible = empty<MaskDetached>(state_count);
    drjit::eval(source_pos,
                edge_pos,
                edge_dir,
                edge_line_min,
                edge_line_max,
                face_offsets,
                active_detached);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       source_pos,
                                       IntDetached(),
                                       0,
                                       active_detached,
                                       state_count);
    params.end_x = edge_pos.x().data();
    params.end_y = edge_pos.y().data();
    params.end_z = edge_pos.z().data();
    params.edge_dir_x = edge_dir.x().data();
    params.edge_dir_y = edge_dir.y().data();
    params.edge_dir_z = edge_dir.z().data();
    params.edge_line_min = edge_line_min.data();
    params.edge_line_max = edge_line_max.data();
    params.sample_count = static_cast<int>(sample_fractions.size());
    for (size_t i = 0; i < sample_fractions.size(); ++i) {
        params.sample_fractions[i] = sample_fractions[i];
    }
    params.out_visible = reinterpret_cast<uint8_t *>(any_visible.data());
    pipeline.launch(static_cast<int>(SegmentVisibilityLaunchKind::AxialEdge), params);

    if constexpr (!Detached) {
        result.any_visible = Mask(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3fDetached &points,
    const IntDetached &chain_length,
    int chain_count,
    int max_points,
    int max_segments,
    const MaskDetached &active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    const IntDetached chain_index = arange<IntDetached>(chain_count);
    const IntDetached chain_base = chain_index * max_points;
    MaskDetached all_visible = active_detached;
    IntDetached first_blocked_segment = full<IntDetached>(-1, chain_count);
    IntDetached first_blocked_prim = full<IntDetached>(-1, chain_count);

    for (int segment = 0; segment < max_segments; ++segment) {
        const MaskDetached segment_active =
            active_detached && all_visible && (chain_length > segment);
        const IntDetached start_index = chain_base + segment;
        const Vector3fDetached start_point =
            gather<Vector3fDetached>(points, start_index, segment_active);
        const Vector3fDetached end_point =
            gather<Vector3fDetached>(points, start_index + 1, segment_active);
        const OptixSegmentHit hit =
            optix_scene.segment_hit<true>(start_point, end_point, segment_active);
        const MaskDetached blocked = segment_active && !hit.visible;
        all_visible &= !blocked;
        first_blocked_segment =
            select(blocked, IntDetached(segment), first_blocked_segment);
        first_blocked_prim =
            select(blocked, hit.global_prim_id, first_blocked_prim);
    }

    if constexpr (!Detached) {
        result.all_visible = Mask(all_visible);
        result.first_blocked_segment = Int(first_blocked_segment);
        result.first_blocked_prim = Int(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    int ignore_k,
    int chain_count,
    int max_points,
    int max_segments,
    const MaskDetached &active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    MaskDetached all_visible = empty<MaskDetached>(chain_count);
    IntDetached first_blocked_segment = empty<IntDetached>(chain_count);
    IntDetached first_blocked_prim = empty<IntDetached>(chain_count);
    if (ignore_k > 0) {
        drjit::eval(points,
                    chain_length,
                    ignore_prim_per_segment,
                    face_offsets,
                    active_detached);
    } else {
        drjit::eval(points, chain_length, face_offsets, active_detached);
    }

    SegmentVisibilityParams params = {};
    params.handle = optix_scene.ias_handle();
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.chain_point_x = points.x().data();
    params.chain_point_y = points.y().data();
    params.chain_point_z = points.z().data();
    params.chain_length = chain_length.data();
    params.max_points = max_points;
    params.max_segments = max_segments;
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_per_segment.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = chain_count;
    params.out_visible = reinterpret_cast<uint8_t *>(all_visible.data());
    params.out_first_blocked_segment = first_blocked_segment.data();
    params.out_first_blocked_prim = first_blocked_prim.data();
    pipeline.launch(static_cast<int>(SegmentVisibilityLaunchKind::SegmentChain), params);

    if constexpr (!Detached) {
        result.all_visible = Mask(all_visible);
        result.first_blocked_segment = Int(first_blocked_segment);
        result.first_blocked_prim = Int(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

} // namespace

template <bool Detached>
MaskT<Detached> Scene::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::shadow_test(): scene is not built.");
    require(!pending_updates_, "Scene::shadow_test(): scene has pending updates. Call Scene::sync() first.");

    const bool symbolic_optix_query = optix_split_active_ && uses_symbolic_optix_query_path();
    if (!optix_split_active_ || symbolic_optix_query) {
        return optix_scene_->template shadow_test<Detached>(ray, active);
    }

    const MaskT<Detached> static_hit =
        optix_static_scene_->template shadow_test<Detached>(ray, active);
    const MaskT<Detached> dynamic_hit =
        optix_dynamic_scene_->template shadow_test<Detached>(ray, active);
    return static_hit || dynamic_hit;
}

template <bool Detached>
SegmentVisibilityT<Detached> Scene::trace_segment_visibility(
    const Vector3fT<Detached> &start,
    const Vector3fT<Detached> &end,
    const IntDetached &ignore_prim_ids,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::trace_segment_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_segment_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end)) == ray_count,
            "Scene::trace_segment_visibility(): start and end must have the same width.");

    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;
    result.visible = full<MaskT<Detached>>(false, ray_count);
    if (ray_count == 0) {
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::trace_segment_visibility(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const MaskDetached active_detached = sanitize_segment_active<Detached>(start, end, active);
    Vector3fDetached start_detached;
    Vector3fDetached end_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_detached = detach<false>(end);
    } else {
        start_detached = start;
        end_detached = end;
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_visibility_jit_no_ignore<Detached>(
            *optix_scene_, start_detached, end_detached, active_detached);
    }

    ensure_pipeline(segment_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, segment_visibility_pipeline_config());
    return trace_segment_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        start_detached,
        end_detached,
        ignore_prim_ids,
        ignore_k,
        active_detached);
}

template <bool Detached>
SegmentPairVisibilityT<Detached> Scene::trace_segment_pair_visibility(
    const Vector3fT<Detached> &start,
    const Vector3fT<Detached> &end_a,
    const Vector3fT<Detached> &end_b,
    const IntDetached &ignore_prim_ids,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::trace_segment_pair_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_segment_pair_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end_a)) == ray_count &&
                static_cast<int>(slices(end_b)) == ray_count,
            "Scene::trace_segment_pair_visibility(): start, end_a, and end_b must have the same width.");

    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;
    result.visible_a = full<MaskT<Detached>>(false, ray_count);
    result.visible_b = full<MaskT<Detached>>(false, ray_count);
    if (ray_count == 0) {
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::trace_segment_pair_visibility(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const MaskDetached active_detached =
        sanitize_segment_active<Detached>(start, end_a, active) &&
        sanitize_segment_active<Detached>(start, end_b, active);
    Vector3fDetached start_detached;
    Vector3fDetached end_a_detached;
    Vector3fDetached end_b_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_a_detached = detach<false>(end_a);
        end_b_detached = detach<false>(end_b);
    } else {
        start_detached = start;
        end_a_detached = end_a;
        end_b_detached = end_b;
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_pair_visibility_jit_no_ignore<Detached>(
            *optix_scene_,
            start_detached,
            end_a_detached,
            end_b_detached,
            active_detached);
    }

    ensure_pipeline(segment_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, segment_visibility_pipeline_config());
    return trace_segment_pair_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        start_detached,
        end_a_detached,
        end_b_detached,
        ignore_prim_ids,
        ignore_k,
        active_detached);
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> Scene::trace_axial_edge_visibility(
    const Vector3fT<Detached> &source_pos,
    const Vector3fT<Detached> &edge_pos,
    const Vector3fT<Detached> &edge_dir,
    const FloatT<Detached> &edge_line_min,
    const FloatT<Detached> &edge_line_max,
    const std::vector<float> &sample_fractions,
    MaskT<Detached> active) const {
    require(!sample_fractions.empty(),
            "Scene::trace_axial_edge_visibility(): sample_fractions must not be empty.");
    require(sample_fractions.size() <= SegmentVisibilityMaxSamples,
            "Scene::trace_axial_edge_visibility(): at most 16 sample fractions are supported.");
    require(is_ready(), "Scene::trace_axial_edge_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_axial_edge_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int state_count = static_cast<int>(slices(source_pos));
    require(static_cast<int>(slices(edge_pos)) == state_count &&
                static_cast<int>(slices(edge_dir)) == state_count &&
                static_cast<int>(slices(edge_line_min)) == state_count &&
                static_cast<int>(slices(edge_line_max)) == state_count,
            "Scene::trace_axial_edge_visibility(): all inputs must have the same width.");

    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;
    result.any_visible = full<MaskT<Detached>>(false, state_count);
    if (state_count == 0) {
        return result;
    }

    MaskDetached active_detached;
    Vector3fDetached source_detached;
    Vector3fDetached edge_pos_detached;
    Vector3fDetached edge_dir_detached;
    FloatDetached edge_line_min_detached;
    FloatDetached edge_line_max_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        source_detached = detach<false>(source_pos);
        edge_pos_detached = detach<false>(edge_pos);
        edge_dir_detached = detach<false>(edge_dir);
        edge_line_min_detached = detach<false>(edge_line_min);
        edge_line_max_detached = detach<false>(edge_line_max);
    } else {
        active_detached = active;
        source_detached = source_pos;
        edge_pos_detached = edge_pos;
        edge_dir_detached = edge_dir;
        edge_line_min_detached = edge_line_min;
        edge_line_max_detached = edge_line_max;
    }

    active_detached &= drjit::isfinite(source_detached.x()) &&
                       drjit::isfinite(source_detached.y()) &&
                       drjit::isfinite(source_detached.z()) &&
                       drjit::isfinite(edge_pos_detached.x()) &&
                       drjit::isfinite(edge_pos_detached.y()) &&
                       drjit::isfinite(edge_pos_detached.z()) &&
                       drjit::isfinite(edge_dir_detached.x()) &&
                       drjit::isfinite(edge_dir_detached.y()) &&
                       drjit::isfinite(edge_dir_detached.z()) &&
                       drjit::isfinite(edge_line_min_detached) &&
                       drjit::isfinite(edge_line_max_detached);

    if (active_trace_visibility_backend() != TraceVisibilityBackend::Native) {
        return trace_axial_edge_visibility_jit<Detached>(
            *optix_scene_,
            source_detached,
            edge_pos_detached,
            edge_dir_detached,
            edge_line_min_detached,
            edge_line_max_detached,
            sample_fractions,
            active_detached);
    }

    ensure_pipeline(segment_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, segment_visibility_pipeline_config());
    return trace_axial_edge_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        source_detached,
        edge_pos_detached,
        edge_dir_detached,
        edge_line_min_detached,
        edge_line_max_detached,
        sample_fractions,
        active_detached);
}

template <bool Detached>
SegmentChainVisibilityT<Detached> Scene::trace_segment_chain_visibility(
    const Vector3fT<Detached> &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::trace_segment_chain_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_segment_chain_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int chain_count = static_cast<int>(slices(chain_length));
    const int point_count = static_cast<int>(slices(points));

    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = 0;
    result.all_visible = full<MaskT<Detached>>(false, chain_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, chain_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, chain_count);
    if (chain_count == 0) {
        return result;
    }

    require(point_count % chain_count == 0,
            "Scene::trace_segment_chain_visibility(): points width must be a multiple of chain count.");
    const int max_points = point_count / chain_count;
    require(max_points >= 2,
            "Scene::trace_segment_chain_visibility(): each chain must contain at least two points.");
    const int max_segments = max_points - 1;
    result.max_segments = max_segments;

    const int ignore_count = static_cast<int>(slices(ignore_prim_per_segment));
    int ignore_k = 0;
    if (ignore_count > 0) {
        const int ignore_slots = chain_count * max_segments;
        require(ignore_count % ignore_slots == 0,
                "Scene::trace_segment_chain_visibility(): ignore_prim_per_segment width must be a multiple of chain_count * max_segments.");
        ignore_k = ignore_count / ignore_slots;
    }

    MaskDetached active_detached;
    Vector3fDetached points_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        points_detached = detach<false>(points);
    } else {
        active_detached = active;
        points_detached = points;
    }
    active_detached &= chain_length >= 0;

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_chain_visibility_jit_no_ignore<Detached>(
            *optix_scene_,
            points_detached,
            chain_length,
            chain_count,
            max_points,
            max_segments,
            active_detached);
    }

    ensure_pipeline(segment_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, segment_visibility_pipeline_config());
    return trace_segment_chain_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        points_detached,
        chain_length,
        ignore_prim_per_segment,
        ignore_k,
        chain_count,
        max_points,
        max_segments,
        active_detached);
}

template MaskDetached Scene::shadow_test<true>(const RayDetached &ray, MaskDetached active) const;

template Mask Scene::shadow_test<false>(const Ray &ray, Mask active) const;

template SegmentVisibilityDetached Scene::trace_segment_visibility<true>(
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const IntDetached &ignore_prim_ids,
    MaskDetached active) const;

template SegmentVisibility Scene::trace_segment_visibility<false>(
    const Vector3f &start,
    const Vector3f &end,
    const IntDetached &ignore_prim_ids,
    Mask active) const;

template SegmentPairVisibilityDetached Scene::trace_segment_pair_visibility<true>(
    const Vector3fDetached &start,
    const Vector3fDetached &end_a,
    const Vector3fDetached &end_b,
    const IntDetached &ignore_prim_ids,
    MaskDetached active) const;

template SegmentPairVisibility Scene::trace_segment_pair_visibility<false>(
    const Vector3f &start,
    const Vector3f &end_a,
    const Vector3f &end_b,
    const IntDetached &ignore_prim_ids,
    Mask active) const;

template AxialEdgeVisibilityDetached Scene::trace_axial_edge_visibility<true>(
    const Vector3fDetached &source_pos,
    const Vector3fDetached &edge_pos,
    const Vector3fDetached &edge_dir,
    const FloatDetached &edge_line_min,
    const FloatDetached &edge_line_max,
    const std::vector<float> &sample_fractions,
    MaskDetached active) const;

template AxialEdgeVisibility Scene::trace_axial_edge_visibility<false>(
    const Vector3f &source_pos,
    const Vector3f &edge_pos,
    const Vector3f &edge_dir,
    const Float &edge_line_min,
    const Float &edge_line_max,
    const std::vector<float> &sample_fractions,
    Mask active) const;

template SegmentChainVisibilityDetached Scene::trace_segment_chain_visibility<true>(
    const Vector3fDetached &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    MaskDetached active) const;

template SegmentChainVisibility Scene::trace_segment_chain_visibility<false>(
    const Vector3f &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    Mask active) const;

} // namespace rayd
