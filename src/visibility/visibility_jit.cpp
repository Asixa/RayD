// Copyright Xingyu Chen.
// Implements visibility support for visibility Dr.Jit.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/jit/core.h>
#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>
#include <src/reflection/reflection_internal.h>
#include <src/reflection/epc_field_jit.h>
#include <src/runtime/optix_pipelines_jit.h>
#include <rayd/jit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>

#include "multipath_internal_jit.h"

namespace rayd {

using namespace multipath_detail;

namespace {

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_jit_no_ignore(const OptixScene& optix_scene,
                                                                    const Vector3f& start, const Vector3f& end,
                                                                    const Mask& active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit = optix_scene.segment_hit<true>(start, end, active_detached);
    if constexpr (!Detached) {
        result.visible = MaskAD(hit.visible);
    } else {
        result.visible = hit.visible;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_jit_no_ignore(const OptixScene& optix_scene,
                                                                             const Vector3f& start,
                                                                             const Vector3f& end_a,
                                                                             const Vector3f& end_b,
                                                                             const Mask& active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit_a = optix_scene.segment_hit<true>(start, end_a, active_detached);
    const OptixSegmentHit hit_b = optix_scene.segment_hit<true>(start, end_b, active_detached);
    if constexpr (!Detached) {
        result.visible_a = MaskAD(hit_a.visible);
        result.visible_b = MaskAD(hit_b.visible);
    } else {
        result.visible_a = hit_a.visible;
        result.visible_b = hit_b.visible;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_native(const OptixScene& optix_scene,
                                                                      const OptixLaunchPipeline& pipeline,
                                                                      const Int& face_offsets, int mesh_count,
                                                                      const Vector3f& start, const Vector3f& end_a,
                                                                      const Vector3f& end_b, const Int& ignore_prim_ids,
                                                                      int ignore_k, const Mask& active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    Mask visible_a = empty<Mask>(ray_count);
    Mask visible_b = empty<Mask>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end_a, end_b);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene, face_offsets, mesh_count, start, ignore_prim_ids, ignore_k,
                                       active_detached, ray_count);
    params.end_x = end_a.x().data();
    params.end_y = end_a.y().data();
    params.end_z = end_a.z().data();
    params.end_b_x = end_b.x().data();
    params.end_b_y = end_b.y().data();
    params.end_b_z = end_b.z().data();
    params.out_visible = reinterpret_cast<uint8_t*>(visible_a.data());
    params.out_visible_b = reinterpret_cast<uint8_t*>(visible_b.data());
    pipeline.launch(0, params);

    if constexpr (!Detached) {
        result.visible_a = MaskAD(visible_a);
        result.visible_b = MaskAD(visible_b);
    } else {
        result.visible_a = visible_a;
        result.visible_b = visible_b;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_jit(const OptixScene& optix_scene, const Vector3f& src,
                                                               const Vector3f& edge_pos, const Vector3f& edge_dir,
                                                               const Float& edge_t_min, const Float& edge_t_max,
                                                               const std::vector<float>& sample_fractions,
                                                               const Mask& active_detached) {
    const int state_count = static_cast<int>(slices(src));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    Mask any_visible = full<Mask>(false, state_count);
    const Float span = maximum(edge_t_max - edge_t_min, Float(0.f));
    for (float fraction : sample_fractions) {
        const Float sample_t = edge_t_min + fraction * span;
        const Vector3f sample_pos = edge_pos + sample_t * edge_dir;
        const OptixSegmentHit hit = optix_scene.segment_hit<true>(src, sample_pos, active_detached);
        any_visible = any_visible || hit.visible;
    }

    if constexpr (!Detached) {
        result.any_visible = MaskAD(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_native(
    const OptixScene& optix_scene, const OptixLaunchPipeline& pipeline, const Int& face_offsets, int mesh_count,
    const Vector3f& src, const Vector3f& edge_pos, const Vector3f& edge_dir, const Float& edge_t_min,
    const Float& edge_t_max, const std::vector<float>& sample_fractions, const Mask& active_detached) {
    const int state_count = static_cast<int>(slices(src));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    Mask any_visible = empty<Mask>(state_count);
    drjit::eval(src, edge_pos, edge_dir, edge_t_min, edge_t_max, face_offsets, active_detached);

    SegmentVisibilityParams params = make_segment_visibility_params(optix_scene, face_offsets, mesh_count, src, Int(),
                                                                    0, active_detached, state_count);
    params.end_x = edge_pos.x().data();
    params.end_y = edge_pos.y().data();
    params.end_z = edge_pos.z().data();
    params.edge_dir_x = edge_dir.x().data();
    params.edge_dir_y = edge_dir.y().data();
    params.edge_dir_z = edge_dir.z().data();
    params.edge_t_min = edge_t_min.data();
    params.edge_t_max = edge_t_max.data();
    params.sample_count = static_cast<int>(sample_fractions.size());
    for (size_t i = 0; i < sample_fractions.size(); ++i) {
        params.sample_fractions[i] = sample_fractions[i];
    }
    params.out_visible = reinterpret_cast<uint8_t*>(any_visible.data());
    pipeline.launch(0, params);

    if constexpr (!Detached) {
        result.any_visible = MaskAD(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_jit_no_ignore(const OptixScene& optix_scene,
                                                                               const Vector3f& points,
                                                                               const Int& chain_length, int chain_count,
                                                                               int max_points, int max_segments,
                                                                               const Mask& active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    const Int chain_index = arange<Int>(chain_count);
    const Int chain_base = chain_index * max_points;
    Mask all_visible = active_detached;
    Int first_blocked_segment = full<Int>(-1, chain_count);
    Int first_blocked_prim = full<Int>(-1, chain_count);

    for (int segment = 0; segment < max_segments; ++segment) {
        const Mask segment_active = active_detached && all_visible && (chain_length > segment);
        const Int start_index = chain_base + segment;
        const Vector3f start_point = gather<Vector3f>(points, start_index, segment_active);
        const Vector3f end_point = gather<Vector3f>(points, start_index + 1, segment_active);
        const OptixSegmentHit hit = optix_scene.segment_hit<true>(start_point, end_point, segment_active);
        const Mask blocked = segment_active && !hit.visible;
        all_visible &= !blocked;
        first_blocked_segment = select(blocked, Int(segment), first_blocked_segment);
        first_blocked_prim = select(blocked, hit.global_prim_id, first_blocked_prim);
    }

    if constexpr (!Detached) {
        result.all_visible = MaskAD(all_visible);
        result.first_blocked_segment = IntAD(first_blocked_segment);
        result.first_blocked_prim = IntAD(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_native(
    const OptixScene& optix_scene, const OptixLaunchPipeline& pipeline, const Int& face_offsets, int mesh_count,
    const Vector3f& points, const Int& chain_length, const Int& ignore_prim_per_segment, int ignore_k, int chain_count,
    int max_points, int max_segments, const Mask& active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    Mask all_visible = empty<Mask>(chain_count);
    Int first_blocked_segment = empty<Int>(chain_count);
    Int first_blocked_prim = empty<Int>(chain_count);
    eval_segment_visibility_common(points, face_offsets, ignore_prim_per_segment, ignore_k, active_detached);
    drjit::eval(chain_length);

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
    params.active_mask = reinterpret_cast<const uint8_t*>(active_detached.data());
    params.n_rays = chain_count;
    params.out_visible = reinterpret_cast<uint8_t*>(all_visible.data());
    params.out_first_blocked_segment = first_blocked_segment.data();
    params.out_first_blocked_prim = first_blocked_prim.data();
    pipeline.launch(0, params);

    if constexpr (!Detached) {
        result.all_visible = MaskAD(all_visible);
        result.first_blocked_segment = IntAD(first_blocked_segment);
        result.first_blocked_prim = IntAD(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

// -- CUDA fused segment-visibility marshaling (P4 Stage D) --------------------
// These mirror the OptiX native functions above field-for-field but launch the
// pure-CUDA kernel over the scene triangle BVH instead of an OptiX pipeline. The
// handle is left 0 here; CudaTraceBackend::run_segment_visibility sets the
// non-zero scene sentinel. They are the CUDA arm the Scene::visible* dispatch
// takes when trace_backend='cuda'; the jit-symbolic variants stay OptiX-only.

SegmentVisibilityParams make_segment_visibility_params_cuda(const Int& face_offsets, int mesh_count,
                                                            const Vector3f& start, const Int& ignore_prim_ids,
                                                            int ignore_k, const Mask& active_detached, int ray_count) {
    SegmentVisibilityParams params = {};
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.start_x = start.x().data();
    params.start_y = start.y().data();
    params.start_z = start.z().data();
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_ids.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t*>(active_detached.data());
    params.n_rays = ray_count;
    return params;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_cuda(const CudaTraceBackend& cuda, const Int& face_offsets,
                                                           int mesh_count, const Vector3f& start, const Vector3f& end,
                                                           const Int& ignore_prim_ids, int ignore_k,
                                                           const Mask& active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    Mask visible = empty<Mask>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end);

    SegmentVisibilityParams params =
        make_segment_visibility_params_cuda(face_offsets, mesh_count, start, ignore_prim_ids, ignore_k, active_detached,
                                            ray_count);
    params.end_x = end.x().data();
    params.end_y = end.y().data();
    params.end_z = end.z().data();
    params.out_visible = reinterpret_cast<uint8_t*>(visible.data());
    cuda.run_segment_visibility(params, CudaSegmentVisibilityVariant::Single, ray_count);

    if constexpr (!Detached) {
        result.visible = MaskAD(visible);
    } else {
        result.visible = visible;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_cuda(const CudaTraceBackend& cuda,
                                                                    const Int& face_offsets, int mesh_count,
                                                                    const Vector3f& start, const Vector3f& end_a,
                                                                    const Vector3f& end_b, const Int& ignore_prim_ids,
                                                                    int ignore_k, const Mask& active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    Mask visible_a = empty<Mask>(ray_count);
    Mask visible_b = empty<Mask>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end_a, end_b);

    SegmentVisibilityParams params =
        make_segment_visibility_params_cuda(face_offsets, mesh_count, start, ignore_prim_ids, ignore_k, active_detached,
                                            ray_count);
    params.end_x = end_a.x().data();
    params.end_y = end_a.y().data();
    params.end_z = end_a.z().data();
    params.end_b_x = end_b.x().data();
    params.end_b_y = end_b.y().data();
    params.end_b_z = end_b.z().data();
    params.out_visible = reinterpret_cast<uint8_t*>(visible_a.data());
    params.out_visible_b = reinterpret_cast<uint8_t*>(visible_b.data());
    cuda.run_segment_visibility(params, CudaSegmentVisibilityVariant::Pair, ray_count);

    if constexpr (!Detached) {
        result.visible_a = MaskAD(visible_a);
        result.visible_b = MaskAD(visible_b);
    } else {
        result.visible_a = visible_a;
        result.visible_b = visible_b;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_cuda(const CudaTraceBackend& cuda, const Int& face_offsets,
                                                                int mesh_count, const Vector3f& src,
                                                                const Vector3f& edge_pos, const Vector3f& edge_dir,
                                                                const Float& edge_t_min, const Float& edge_t_max,
                                                                const std::vector<float>& sample_fractions,
                                                                const Mask& active_detached) {
    const int state_count = static_cast<int>(slices(src));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    Mask any_visible = empty<Mask>(state_count);
    drjit::eval(src, edge_pos, edge_dir, edge_t_min, edge_t_max, face_offsets, active_detached);

    SegmentVisibilityParams params =
        make_segment_visibility_params_cuda(face_offsets, mesh_count, src, Int(), 0, active_detached, state_count);
    params.end_x = edge_pos.x().data();
    params.end_y = edge_pos.y().data();
    params.end_z = edge_pos.z().data();
    params.edge_dir_x = edge_dir.x().data();
    params.edge_dir_y = edge_dir.y().data();
    params.edge_dir_z = edge_dir.z().data();
    params.edge_t_min = edge_t_min.data();
    params.edge_t_max = edge_t_max.data();
    params.sample_count = static_cast<int>(sample_fractions.size());
    for (size_t i = 0; i < sample_fractions.size(); ++i) {
        params.sample_fractions[i] = sample_fractions[i];
    }
    params.out_visible = reinterpret_cast<uint8_t*>(any_visible.data());
    cuda.run_segment_visibility(params, CudaSegmentVisibilityVariant::AxialEdge, state_count);

    if constexpr (!Detached) {
        result.any_visible = MaskAD(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_cuda(const CudaTraceBackend& cuda,
                                                                      const Int& face_offsets, int mesh_count,
                                                                      const Vector3f& points, const Int& chain_length,
                                                                      const Int& ignore_prim_per_segment, int ignore_k,
                                                                      int chain_count, int max_points, int max_segments,
                                                                      const Mask& active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    Mask all_visible = empty<Mask>(chain_count);
    Int first_blocked_segment = empty<Int>(chain_count);
    Int first_blocked_prim = empty<Int>(chain_count);
    eval_segment_visibility_common(points, face_offsets, ignore_prim_per_segment, ignore_k, active_detached);
    drjit::eval(chain_length);

    SegmentVisibilityParams params = {};
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
    params.active_mask = reinterpret_cast<const uint8_t*>(active_detached.data());
    params.n_rays = chain_count;
    params.out_visible = reinterpret_cast<uint8_t*>(all_visible.data());
    params.out_first_blocked_segment = first_blocked_segment.data();
    params.out_first_blocked_prim = first_blocked_prim.data();
    cuda.run_segment_visibility(params, CudaSegmentVisibilityVariant::Chain, chain_count);

    if constexpr (!Detached) {
        result.all_visible = MaskAD(all_visible);
        result.first_blocked_segment = IntAD(first_blocked_segment);
        result.first_blocked_prim = IntAD(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

} // namespace

template <bool Detached> MaskT<Detached> Scene::shadow_test(const RayT<Detached>& ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::shadow_test(): scene is not built.");
    require(!pending_updates_, "Scene::shadow_test(): scene has pending updates. Call Scene::sync() first.");

    if (triangle_kind_ == TraceBackendKind::Cuda) {
        require(!jit_flag(JitFlag::Recording),
                "trace_backend='cuda' cannot serve shadow_test() inside a Dr.Jit symbolic "
                "recording region; use trace_backend='optix' or evaluate outside the "
                "recorded loop.");
        return cuda_backend().template shadow_test<Detached>(ray, active);
    }

    const bool symbolic_optix_query = optix_split_active() && uses_symbolic_optix_query_path();
    if (!optix_split_active() || symbolic_optix_query) {
        return optix_scene().template shadow_test<Detached>(ray, active);
    }

    const MaskT<Detached> static_hit = optix_static_scene().template shadow_test<Detached>(ray, active);
    const MaskT<Detached> dynamic_hit = optix_dynamic_scene().template shadow_test<Detached>(ray, active);
    return static_hit || dynamic_hit;
}

template <bool Detached>
SegmentVisibilityT<Detached> Scene::visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                            const Int& ignore_prim_ids, MaskT<Detached> active) const {
    require(is_ready(), "Scene::visible(): scene is not built.");
    require(!pending_updates_, "Scene::visible(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end)) == ray_count, "Scene::visible(): start and end must have the same width.");

    if (ray_count == 0) {
        SegmentVisibilityT<Detached> result;
        result.ray_count = ray_count;
        result.visible = full<MaskT<Detached>>(false, ray_count);
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::visible(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const Mask active_detached = sanitize_segment_active<Detached>(start, end, active);
    Vector3f start_detached;
    Vector3f end_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_detached = detach<false>(end);
    } else {
        start_detached = start;
        end_detached = end;
    }

    if (triangle_kind_ == TraceBackendKind::Cuda) {
        return trace_segment_visibility_cuda<Detached>(cuda_backend(), face_offsets_, mesh_count_, start_detached,
                                                       end_detached, ignore_prim_ids, ignore_k, active_detached);
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_visibility_jit_no_ignore<Detached>(optix_scene(), start_detached, end_detached,
                                                                active_detached);
    }

    ensure_pipeline(segment_visibility_pipeline_, optix_scene().context(), mesh_count_,
                    segment_visibility_pipeline_config());

    return trace_segment_visibility_native<Detached>(optix_scene(), *segment_visibility_pipeline_, face_offsets_,
                                                     mesh_count_, start_detached, end_detached, ignore_prim_ids,
                                                     ignore_k, active_detached);
}

template <bool Detached>
SegmentPairVisibilityT<Detached> Scene::visible_pair(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end_a,
                                                     const Vector3fT<Detached>& end_b, const Int& ignore_prim_ids,
                                                     MaskT<Detached> active) const {
    require(is_ready(), "Scene::visible_pair(): scene is not built.");
    require(!pending_updates_, "Scene::visible_pair(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end_a)) == ray_count && static_cast<int>(slices(end_b)) == ray_count,
            "Scene::visible_pair(): start, end_a, and end_b must have the same width.");

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
                "Scene::visible_pair(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const Mask active_detached = sanitize_segment_active<Detached>(start, end_a, active) &&
                                 sanitize_segment_active<Detached>(start, end_b, active);
    Vector3f start_detached;
    Vector3f end_a_detached;
    Vector3f end_b_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_a_detached = detach<false>(end_a);
        end_b_detached = detach<false>(end_b);
    } else {
        start_detached = start;
        end_a_detached = end_a;
        end_b_detached = end_b;
    }

    if (triangle_kind_ == TraceBackendKind::Cuda) {
        return trace_segment_pair_visibility_cuda<Detached>(cuda_backend(), face_offsets_, mesh_count_, start_detached,
                                                            end_a_detached, end_b_detached, ignore_prim_ids, ignore_k,
                                                            active_detached);
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_pair_visibility_jit_no_ignore<Detached>(optix_scene(), start_detached, end_a_detached,
                                                                     end_b_detached, active_detached);
    }

    ensure_pipeline(segment_pair_visibility_pipeline_, optix_scene().context(), mesh_count_,
                    segment_pair_visibility_pipeline_config());

    return trace_segment_pair_visibility_native<Detached>(optix_scene(), *segment_pair_visibility_pipeline_,
                                                          face_offsets_, mesh_count_, start_detached, end_a_detached,
                                                          end_b_detached, ignore_prim_ids, ignore_k, active_detached);
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> Scene::visible_edge(const Vector3fT<Detached>& src, const Vector3fT<Detached>& edge_pos,
                                                   const Vector3fT<Detached>& edge_dir,
                                                   const FloatT<Detached>& edge_t_min,
                                                   const FloatT<Detached>& edge_t_max,
                                                   const std::vector<float>& sample_fractions,
                                                   MaskT<Detached> active) const {
    require(!sample_fractions.empty(), "Scene::visible_edge(): sample_fractions must not be empty.");
    require(sample_fractions.size() <= SegmentVisibilityMaxSamples,
            "Scene::visible_edge(): at most 16 sample fractions are supported.");
    require(is_ready(), "Scene::visible_edge(): scene is not built.");
    require(!pending_updates_, "Scene::visible_edge(): scene has pending updates. Call Scene::sync() first.");

    const int state_count = static_cast<int>(slices(src));
    require(static_cast<int>(slices(edge_pos)) == state_count && static_cast<int>(slices(edge_dir)) == state_count &&
                static_cast<int>(slices(edge_t_min)) == state_count &&
                static_cast<int>(slices(edge_t_max)) == state_count,
            "Scene::visible_edge(): all inputs must have the same width.");

    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;
    result.any_visible = full<MaskT<Detached>>(false, state_count);
    if (state_count == 0) {
        return result;
    }

    Mask active_detached;
    Vector3f source_detached;
    Vector3f edge_pos_detached;
    Vector3f edge_dir_detached;
    Float edge_t_min_detached;
    Float edge_t_max_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        source_detached = detach<false>(src);
        edge_pos_detached = detach<false>(edge_pos);
        edge_dir_detached = detach<false>(edge_dir);
        edge_t_min_detached = detach<false>(edge_t_min);
        edge_t_max_detached = detach<false>(edge_t_max);
    } else {
        active_detached = active;
        source_detached = src;
        edge_pos_detached = edge_pos;
        edge_dir_detached = edge_dir;
        edge_t_min_detached = edge_t_min;
        edge_t_max_detached = edge_t_max;
    }

    active_detached &= drjit::isfinite(source_detached.x()) && drjit::isfinite(source_detached.y()) &&
                       drjit::isfinite(source_detached.z()) && drjit::isfinite(edge_pos_detached.x()) &&
                       drjit::isfinite(edge_pos_detached.y()) && drjit::isfinite(edge_pos_detached.z()) &&
                       drjit::isfinite(edge_dir_detached.x()) && drjit::isfinite(edge_dir_detached.y()) &&
                       drjit::isfinite(edge_dir_detached.z()) && drjit::isfinite(edge_t_min_detached) &&
                       drjit::isfinite(edge_t_max_detached);

    if (triangle_kind_ == TraceBackendKind::Cuda) {
        return trace_axial_edge_visibility_cuda<Detached>(cuda_backend(), face_offsets_, mesh_count_, source_detached,
                                                          edge_pos_detached, edge_dir_detached, edge_t_min_detached,
                                                          edge_t_max_detached, sample_fractions, active_detached);
    }

    if (active_trace_visibility_backend() != TraceVisibilityBackend::Native) {
        return trace_axial_edge_visibility_jit<Detached>(optix_scene(), source_detached, edge_pos_detached,
                                                         edge_dir_detached, edge_t_min_detached, edge_t_max_detached,
                                                         sample_fractions, active_detached);
    }

    ensure_pipeline(axial_edge_visibility_pipeline_, optix_scene().context(), mesh_count_,
                    axial_edge_visibility_pipeline_config());
    return trace_axial_edge_visibility_native<Detached>(optix_scene(), *axial_edge_visibility_pipeline_, face_offsets_,
                                                        mesh_count_, source_detached, edge_pos_detached,
                                                        edge_dir_detached, edge_t_min_detached, edge_t_max_detached,
                                                        sample_fractions, active_detached);
}

template <bool Detached>
SegmentChainVisibilityT<Detached> Scene::visible_chain(const Vector3fT<Detached>& points, const Int& chain_length,
                                                       const Int& ignore_prim_per_segment,
                                                       MaskT<Detached> active) const {
    require(is_ready(), "Scene::visible_chain(): scene is not built.");
    require(!pending_updates_, "Scene::visible_chain(): scene has pending updates. Call Scene::sync() first.");

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

    require(point_count % chain_count == 0, "Scene::visible_chain(): points width must be a multiple of chain count.");
    const int max_points = point_count / chain_count;
    require(max_points >= 2, "Scene::visible_chain(): each chain must contain at least two points.");
    const int max_segments = max_points - 1;
    result.max_segments = max_segments;

    const int ignore_count = static_cast<int>(slices(ignore_prim_per_segment));
    int ignore_k = 0;
    if (ignore_count > 0) {
        const int ignore_slots = chain_count * max_segments;
        require(
            ignore_count % ignore_slots == 0,
            "Scene::visible_chain(): ignore_prim_per_segment width must be a multiple of chain_count * max_segments.");
        ignore_k = ignore_count / ignore_slots;
    }

    Mask active_detached;
    Vector3f points_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        points_detached = detach<false>(points);
    } else {
        active_detached = active;
        points_detached = points;
    }

    active_detached &= chain_length >= 0;

    if (triangle_kind_ == TraceBackendKind::Cuda) {
        return trace_segment_chain_visibility_cuda<Detached>(cuda_backend(), face_offsets_, mesh_count_,
                                                             points_detached, chain_length, ignore_prim_per_segment,
                                                             ignore_k, chain_count, max_points, max_segments,
                                                             active_detached);
    }

    const bool use_jit_visibility = use_jit_trace_visibility_path(ignore_k);

    if (use_jit_visibility) {
        return trace_segment_chain_visibility_jit_no_ignore<Detached>(optix_scene(), points_detached, chain_length,
                                                                      chain_count, max_points, max_segments,
                                                                      active_detached);
    }

    ensure_pipeline(segment_chain_visibility_pipeline_, optix_scene().context(), mesh_count_,
                    segment_chain_visibility_pipeline_config());

    return trace_segment_chain_visibility_native<Detached>(optix_scene(), *segment_chain_visibility_pipeline_,
                                                           face_offsets_, mesh_count_, points_detached, chain_length,
                                                           ignore_prim_per_segment, ignore_k, chain_count, max_points,
                                                           max_segments, active_detached);
}

template Mask Scene::shadow_test<true>(const Ray& ray, Mask active) const;
template MaskAD Scene::shadow_test<false>(const RayAD& ray, MaskAD active) const;
template SegmentVisibility Scene::visible<true>(const Vector3f& start, const Vector3f& end, const Int& ignore_prim_ids,
                                                Mask active) const;
template SegmentVisibilityAD Scene::visible<false>(const Vector3fAD& start, const Vector3fAD& end,
                                                   const Int& ignore_prim_ids, MaskAD active) const;
template SegmentPairVisibility Scene::visible_pair<true>(const Vector3f& start, const Vector3f& end_a,
                                                         const Vector3f& end_b, const Int& ignore_prim_ids,
                                                         Mask active) const;
template SegmentPairVisibilityAD Scene::visible_pair<false>(const Vector3fAD& start, const Vector3fAD& end_a,
                                                            const Vector3fAD& end_b, const Int& ignore_prim_ids,
                                                            MaskAD active) const;
template AxialEdgeVisibility Scene::visible_edge<true>(const Vector3f& src, const Vector3f& edge_pos,
                                                       const Vector3f& edge_dir, const Float& edge_t_min,
                                                       const Float& edge_t_max,
                                                       const std::vector<float>& sample_fractions, Mask active) const;
template AxialEdgeVisibilityAD Scene::visible_edge<false>(const Vector3fAD& src, const Vector3fAD& edge_pos,
                                                          const Vector3fAD& edge_dir, const FloatAD& edge_t_min,
                                                          const FloatAD& edge_t_max,
                                                          const std::vector<float>& sample_fractions,
                                                          MaskAD active) const;
template SegmentChainVisibility Scene::visible_chain<true>(const Vector3f& points, const Int& chain_length,
                                                           const Int& ignore_prim_per_segment, Mask active) const;
template SegmentChainVisibilityAD Scene::visible_chain<false>(const Vector3fAD& points, const Int& chain_length,
                                                              const Int& ignore_prim_per_segment, MaskAD active) const;

} // namespace rayd
