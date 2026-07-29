// Copyright Xingyu Chen.
// Implements diffraction support for diffraction Dr.Jit.

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

#include "multipath_internal_jit.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct DfrPathsRaw {
    int capacity = 0;
    Int count;
    Mask valid;
    Int tx_id;
    Int rx_id;
    Int order;
    Int edge0;
    Int edge1;
    Int edge2;
    Float delay;
    Float field_x_re;
    Float field_x_im;
    Float field_y_re;
    Float field_y_im;
    Float field_z_re;
    Float field_z_im;
    Vector3f p0;
    Vector3f p1;
    Vector3f p2;
};

DfrPathsRaw alloc_dfr_paths_raw(int capacity) {
    DfrPathsRaw raw;
    raw.capacity = capacity;
    raw.count = empty<Int>(1);
    raw.valid = empty<Mask>(capacity);
    raw.tx_id = empty<Int>(capacity);
    raw.rx_id = empty<Int>(capacity);
    raw.order = empty<Int>(capacity);
    raw.edge0 = empty<Int>(capacity);
    raw.edge1 = empty<Int>(capacity);
    raw.edge2 = empty<Int>(capacity);
    raw.delay = empty<Float>(capacity);
    raw.field_x_re = empty<Float>(capacity);
    raw.field_x_im = empty<Float>(capacity);
    raw.field_y_re = empty<Float>(capacity);
    raw.field_y_im = empty<Float>(capacity);
    raw.field_z_re = empty<Float>(capacity);
    raw.field_z_im = empty<Float>(capacity);
    raw.p0 =
        Vector3f(empty<Float>(capacity), empty<Float>(capacity), empty<Float>(capacity));
    raw.p1 =
        Vector3f(empty<Float>(capacity), empty<Float>(capacity), empty<Float>(capacity));
    raw.p2 =
        Vector3f(empty<Float>(capacity), empty<Float>(capacity), empty<Float>(capacity));
    return raw;
}

void init_dfr_paths_raw(DfrPathsRaw &raw) {
    const int zero_i = 0;
    const int minus_one_i = -1;
    const uint8_t zero_b = 0u;
    const float zero_f = 0.f;
    jit_memset_async(JitBackend::CUDA, raw.count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.valid.data(), raw.capacity, sizeof(uint8_t), &zero_b);
    jit_memset_async(JitBackend::CUDA, raw.tx_id.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.rx_id.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.order.data(), raw.capacity, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.edge0.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.edge1.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.edge2.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.delay.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_x_re.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_x_im.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_y_re.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_y_im.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_z_re.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_z_im.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p0.x().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p0.y().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p0.z().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p1.x().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p1.y().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p1.z().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p2.x().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p2.y().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p2.z().data(), raw.capacity, sizeof(float), &zero_f);
}

} // namespace

template <bool Detached>
DfrPathsT<Detached> Scene::trace_dfr_paths(
    const Vector3fT<Detached> &tx_positions,
    const Vector3fT<Detached> &rx_positions,
    const DfrStatesT<Detached> &states,
    const DfrMaterialT<Detached> &material,
    const DfrPathOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::trace_dfr_paths(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_dfr_paths(): scene has pending updates. Call Scene::sync() first.");
    require(options.wavelength > 0.f,
            "Scene::trace_dfr_paths(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::trace_dfr_paths(): only max_order == 1 is supported.");
    require(options.max_paths > 0,
            "Scene::trace_dfr_paths(): max_paths must be positive.");
    require((options.strategy_mask & RAYD_DFR_DIRECT) != 0,
            "Scene::trace_dfr_paths(): first-order path export requires direct diffraction.");

    DfrPathsT<Detached> result;
    if constexpr (!Detached) {
        const int tx_count = static_cast<int>(slices(tx_positions));
        const int rx_width = static_cast<int>(slices(rx_positions));
        const int rx_count = options.max_rx > 0
                                 ? std::min(rx_width, options.max_rx)
                                 : rx_width;
        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (tx_count == 0 || rx_count == 0 || state_count == 0) {
            result.capacity = 0;
            result.count = full<IntAD>(0, 1);
            result.valid = full<MaskAD>(false, 0);
            result.tx_id = full<IntAD>(-1, 0);
            result.rx_id = full<IntAD>(-1, 0);
            result.order = full<IntAD>(0, 0);
            result.edge0 = full<IntAD>(-1, 0);
            result.edge1 = full<IntAD>(-1, 0);
            result.edge2 = full<IntAD>(-1, 0);
            result.delay = zeros<FloatAD>(0);
            result.field_x = drjit::Complex<FloatAD>(zeros<FloatAD>(0), zeros<FloatAD>(0));
            result.field_y = drjit::Complex<FloatAD>(zeros<FloatAD>(0), zeros<FloatAD>(0));
            result.field_z = drjit::Complex<FloatAD>(zeros<FloatAD>(0), zeros<FloatAD>(0));
            result.p0 = zeros<Vector3fAD>(0);
            result.p1 = zeros<Vector3fAD>(0);
            result.p2 = zeros<Vector3fAD>(0);
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::trace_dfr_paths(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count,
                "Scene::trace_dfr_paths(): state fields must cover state count.");
        require(rx_count <= rx_width,
                "Scene::trace_dfr_paths(): invalid receiver count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::trace_dfr_paths(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::trace_dfr_paths(): material payload fields must have matching widths.");

        MaskAD active_ad = active;
        int active_width = static_cast<int>(slices(active_ad));
        if (active_width == 1 && state_count > 1) {
            active_ad = gather<MaskAD>(active_ad, zeros<IntAD>(state_count));
            active_width = state_count;
        } else {
            require(active_width == state_count,
                    "Scene::trace_dfr_paths(): active width must be 1 or match state count.");
        }

        const int state_limit = std::min(state_count, options.max_paths);
        const int64_t capacity64 =
            static_cast<int64_t>(tx_count) *
            static_cast<int64_t>(rx_count) *
            static_cast<int64_t>(state_limit);
        require(capacity64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "Scene::trace_dfr_paths(): requested path capacity exceeds int range.");
        const int capacity = static_cast<int>(capacity64);
        result.capacity = capacity;
        result.count = full<IntAD>(0, 1);
        result.valid = full<MaskAD>(false, capacity);
        result.tx_id = full<IntAD>(-1, capacity);
        result.rx_id = full<IntAD>(-1, capacity);
        result.order = full<IntAD>(0, capacity);
        result.edge0 = full<IntAD>(-1, capacity);
        result.edge1 = full<IntAD>(-1, capacity);
        result.edge2 = full<IntAD>(-1, capacity);
        result.delay = zeros<FloatAD>(capacity);
        result.field_x = drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.field_y = drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.field_z = drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.p0 = zeros<Vector3fAD>(capacity);
        result.p1 = zeros<Vector3fAD>(capacity);
        result.p2 = zeros<Vector3fAD>(capacity);
        if (capacity == 0) {
            return result;
        }

        auto material_gain_for_faces = [&](const IntAD &prim0,
                                           const IntAD &prim1,
                                           const MaskAD &path_active) -> FloatAD {
            const MaskAD prim0_in_range =
                path_active && (prim0 >= IntAD(0)) && (prim0 < IntAD(material_count));
            const IntAD safe0 = select(prim0_in_range, prim0, IntAD(0));
            const MaskAD prim0_valid =
                prim0_in_range && gather<MaskAD>(material.valid, safe0, prim0_in_range);
            const MaskAD prim1_in_range =
                path_active && (prim1 >= IntAD(0)) && (prim1 < IntAD(material_count));
            const IntAD safe1 = select(prim1_in_range, prim1, IntAD(0));
            const MaskAD prim1_valid =
                prim1_in_range && gather<MaskAD>(material.valid, safe1, prim1_in_range);
            const IntAD chosen = select(prim0_valid, safe0, safe1);
            const MaskAD chosen_valid = prim0_valid || prim1_valid;
            const FloatAD gain =
                gather<FloatAD>(material.gain, chosen, chosen_valid);
            return select(chosen_valid, maximum(gain, FloatAD(0.f)), FloatAD(1.f));
        };

        const UIntAD lane_u = arange<UIntAD>(capacity);
        const IntAD lane = IntAD(lane_u);
        const IntAD state_idx = IntAD(lane_u % UIntAD(state_limit));
        const UIntAD pair_idx = lane_u / UIntAD(state_limit);
        const IntAD rx_idx = IntAD(pair_idx % UIntAD(rx_count));
        const IntAD tx_idx = IntAD(pair_idx / UIntAD(rx_count));
        const MaskAD lane_active = full<MaskAD>(true, capacity);
        const MaskAD state_active =
            gather<MaskAD>(active_ad, state_idx, lane_active);

        const Vector3fAD edge_pos =
            gather<Vector3fAD>(states.edge_pos, state_idx, state_active);
        const Vector3fAD edge_dir =
            normalize(gather<Vector3fAD>(states.edge_dir, state_idx, state_active));
        const FloatAD edge_t_min =
            gather<FloatAD>(states.edge_t_min, state_idx, state_active);
        const FloatAD edge_t_max =
            gather<FloatAD>(states.edge_t_max, state_idx, state_active);
        const FloatAD edge_t = FloatAD(0.5f) * (edge_t_min + edge_t_max);
        const Vector3fAD edge_point = edge_pos + edge_t * edge_dir;
        const Vector3fAD source =
            gather<Vector3fAD>(states.src, state_idx, state_active);
        const Vector3fAD receiver =
            gather<Vector3fAD>(rx_positions, rx_idx, lane_active);
        const FloatAD src_power =
            gather<FloatAD>(states.src_power, state_idx, state_active);
        const IntAD prim0 = gather<IntAD>(states.prim0, state_idx, state_active);
        const IntAD prim1 = gather<IntAD>(states.prim1, state_idx, state_active);
        const FloatAD exterior_angle =
            gather<FloatAD>(states.exterior_angle, state_idx, state_active);
        const IntAD edge_index =
            gather<IntAD>(states.edge_index, state_idx, state_active);

        const MaskAD finite_active =
            state_active &&
            drjit::isfinite(source.x()) &&
            drjit::isfinite(source.y()) &&
            drjit::isfinite(source.z()) &&
            drjit::isfinite(edge_point.x()) &&
            drjit::isfinite(edge_point.y()) &&
            drjit::isfinite(edge_point.z()) &&
            drjit::isfinite(receiver.x()) &&
            drjit::isfinite(receiver.y()) &&
            drjit::isfinite(receiver.z()) &&
            drjit::isfinite(src_power);
        const SegmentPairVisibilityAD visibility =
            this->template visible_pair<false>(
                edge_point,
                source,
                receiver,
                Int(),
                finite_active);
        const MaskAD visible = visibility.visible_a && visibility.visible_b;
        const FloatAD source_distance =
            maximum(norm(edge_point - source), FloatAD(Epsilon));
        const FloatAD receiver_distance =
            maximum(norm(receiver - edge_point), FloatAD(Epsilon));
        const FloatAD edge_length =
            maximum(edge_t_max - edge_t_min, FloatAD(0.f));
        const FloatAD wedge_scale =
            minimum(
                maximum(exterior_angle, FloatAD(0.25f * Pi)) / FloatAD(2.f * Pi),
                FloatAD(2.f));
        const FloatAD material_gain =
            material_gain_for_faces(prim0, prim1, finite_active);
        const FloatAD wave_gain =
            FloatAD(options.wavelength) / FloatAD(4.f * Pi);
        const FloatAD contribution =
            src_power *
            material_gain *
            edge_length *
            wedge_scale *
            wave_gain *
            wave_gain /
            (source_distance * source_distance * receiver_distance * receiver_distance);
        const MaskAD path_active =
            visible && (contribution > FloatAD(0.f)) && drjit::isfinite(contribution);
        const FloatAD path_length = source_distance + receiver_distance;
        const FloatAD phase = -FloatAD(options.k) * path_length;
        const FloatAD amplitude = sqrt(maximum(contribution, FloatAD(0.f)));

        result.valid = path_active;
        result.tx_id = select(path_active, tx_idx, IntAD(-1));
        result.rx_id = select(path_active, rx_idx, IntAD(-1));
        result.order = select(path_active, IntAD(1), IntAD(0));
        result.edge0 = select(path_active, edge_index, IntAD(-1));
        result.edge1 = full<IntAD>(-1, capacity);
        result.edge2 = full<IntAD>(-1, capacity);
        result.delay =
            select(path_active, path_length / FloatAD(299792458.f), FloatAD(0.f));
        result.field_x =
            drjit::Complex<FloatAD>(
                select(path_active, amplitude * cos(phase), FloatAD(0.f)),
                select(path_active, amplitude * sin(phase), FloatAD(0.f)));
        result.field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.p0 = select(path_active, edge_point, zeros<Vector3fAD>(capacity));
        result.p1 = zeros<Vector3fAD>(capacity);
        result.p2 = zeros<Vector3fAD>(capacity);
        scatter_reduce(
            ReduceOp::Add,
            result.count,
            IntAD(1),
            zeros<IntAD>(capacity),
            path_active);
        return result;
    } else {
        const int tx_count = static_cast<int>(slices(tx_positions));
        const int rx_width = static_cast<int>(slices(rx_positions));
        const int rx_count = options.max_rx > 0
                                 ? std::min(rx_width, options.max_rx)
                                 : rx_width;
        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (tx_count == 0 || rx_count == 0 || state_count == 0) {
            result.capacity = 0;
            result.count = full<Int>(0, 1);
            result.valid = full<Mask>(false, 0);
            result.tx_id = full<Int>(-1, 0);
            result.rx_id = full<Int>(-1, 0);
            result.order = full<Int>(0, 0);
            result.edge0 = full<Int>(-1, 0);
            result.edge1 = full<Int>(-1, 0);
            result.edge2 = full<Int>(-1, 0);
            result.delay = zeros<Float>(0);
            result.field_x = drjit::Complex<Float>(zeros<Float>(0), zeros<Float>(0));
            result.field_y = drjit::Complex<Float>(zeros<Float>(0), zeros<Float>(0));
            result.field_z = drjit::Complex<Float>(zeros<Float>(0), zeros<Float>(0));
            result.p0 = zeros<Vector3f>(0);
            result.p1 = zeros<Vector3f>(0);
            result.p2 = zeros<Vector3f>(0);
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::trace_dfr_paths(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count,
                "Scene::trace_dfr_paths(): state fields must cover state count.");
        require(rx_count <= rx_width,
                "Scene::trace_dfr_paths(): invalid receiver count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::trace_dfr_paths(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::trace_dfr_paths(): material payload fields must have matching widths.");

        Mask active_detached = active;
        int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
            active_width = state_count;
        } else {
            require(active_width == state_count,
                    "Scene::trace_dfr_paths(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.src.x()) &&
                           drjit::isfinite(states.src.y()) &&
                           drjit::isfinite(states.src.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z()) &&
                           drjit::isfinite(states.src_power);

        const int state_limit = std::min(state_count, options.max_paths);
        const int64_t capacity64 =
            static_cast<int64_t>(tx_count) *
            static_cast<int64_t>(rx_count) *
            static_cast<int64_t>(state_limit);
        require(capacity64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "Scene::trace_dfr_paths(): requested path capacity exceeds int range.");
        const int capacity = static_cast<int>(capacity64);

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        int hitgroup_record_count = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            hitgroup_record_count = scenes.hitgroup_record_count;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::trace_dfr_paths(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::trace_dfr_paths(): invalid hitgroup record count.");

            if (split_mode != 0) {
                ensure_pipeline(diffraction_paths_pipeline_,
                                primary_scene->context(),
                                hitgroup_record_count,
                                diffraction_paths_pipeline_config());
            }
        }

        drjit::eval(tx_positions,
                    rx_positions,
                    states.edge_index,
                    states.edge_pos,
                    states.edge_dir,
                    states.edge_t_min,
                    states.edge_t_max,
                    states.n0,
                    states.n1,
                    states.prim0,
                    states.prim1,
                    states.exterior_angle,
                    states.src,
                    states.src_power,
                    active_detached,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);

        DfrPathsRaw raw = alloc_dfr_paths_raw(capacity);
        init_dfr_paths_raw(raw);

        DfrPathParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = capacity;
        params.capacity = capacity;
        params.tx_pos_x = tx_positions.x().data();
        params.tx_pos_y = tx_positions.y().data();
        params.tx_pos_z = tx_positions.z().data();
        params.tx_count = tx_count;
        params.rx_pos_x = rx_positions.x().data();
        params.rx_pos_y = rx_positions.y().data();
        params.rx_pos_z = rx_positions.z().data();
        params.rx_count = rx_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.active_width = active_width;
        params.state_count = state_count;
        params.state_limit = state_limit;
        params.state_edge_index = states.edge_index.data();
        params.state_edge_pos_x = states.edge_pos.x().data();
        params.state_edge_pos_y = states.edge_pos.y().data();
        params.state_edge_pos_z = states.edge_pos.z().data();
        params.state_edge_dir_x = states.edge_dir.x().data();
        params.state_edge_dir_y = states.edge_dir.y().data();
        params.state_edge_dir_z = states.edge_dir.z().data();
        params.state_edge_t_min = states.edge_t_min.data();
        params.state_edge_t_max = states.edge_t_max.data();
        params.state_n0_x = states.n0.x().data();
        params.state_n0_y = states.n0.y().data();
        params.state_n0_z = states.n0.z().data();
        params.state_n1_x = states.n1.x().data();
        params.state_n1_y = states.n1.y().data();
        params.state_n1_z = states.n1.z().data();
        params.state_prim0 = states.prim0.data();
        params.state_prim1 = states.prim1.data();
        params.state_exterior_angle = states.exterior_angle.data();
        params.state_src_x = states.src.x().data();
        params.state_src_y = states.src.y().data();
        params.state_src_z = states.src.z().data();
        params.state_src_power = states.src_power.data();
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.omega = 2.0f * Pi * 299792458.0f / options.wavelength;
        params.seed = options.seed;
        params.max_order = options.max_order;
        params.strategy_mask = options.strategy_mask;
        params.sample_count = options.sample_count;
        params.return_geom = options.return_geom;
        params.receiver_model = options.receiver_model;
        params.out_count = raw.count.data();
        params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        params.out_tx_id = raw.tx_id.data();
        params.out_rx_id = raw.rx_id.data();
        params.out_order = raw.order.data();
        params.out_edge0 = raw.edge0.data();
        params.out_edge1 = raw.edge1.data();
        params.out_edge2 = raw.edge2.data();
        params.out_delay = raw.delay.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_p0_x = raw.p0.x().data();
        params.out_p0_y = raw.p0.y().data();
        params.out_p0_z = raw.p0.z().data();
        params.out_p1_x = raw.p1.x().data();
        params.out_p1_y = raw.p1.y().data();
        params.out_p1_z = raw.p1.z().data();
        params.out_p2_x = raw.p2.x().data();
        params.out_p2_y = raw.p2.y().data();
        params.out_p2_z = raw.p2.z().data();

        if (split_mode == 0) {
            Mask temp_visibility = full<Mask>(false, capacity);
            drjit::eval(temp_visibility);
            params.temp_visibility =
                reinterpret_cast<uint8_t *>(temp_visibility.data());

            if (cuda_trace) {
                cuda_backend().run_dfr_paths(params, capacity);
            } else {
                ensure_pipeline(diffraction_paths_source_visibility_primary_pipeline_,
                                primary_scene->context(),
                                hitgroup_record_count,
                                diffraction_paths_source_visibility_primary_pipeline_config());
                diffraction_paths_source_visibility_primary_pipeline_->launch(0, params);

                ensure_pipeline(diffraction_paths_target_export_primary_pipeline_,
                                primary_scene->context(),
                                hitgroup_record_count,
                                diffraction_paths_target_export_primary_pipeline_config());
                diffraction_paths_target_export_primary_pipeline_->launch(0, params);
                drjit::sync_thread();
            }
        } else {
            params.temp_visibility = nullptr;
            diffraction_paths_pipeline_->launch(0, params);
        }

        result.capacity = capacity;
        result.count = raw.count;
        result.valid = raw.valid;
        result.tx_id = raw.tx_id;
        result.rx_id = raw.rx_id;
        result.order = raw.order;
        result.edge0 = raw.edge0;
        result.edge1 = raw.edge1;
        result.edge2 = raw.edge2;
        result.delay = raw.delay;
        result.field_x = drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.field_y = drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.field_z = drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.p0 = raw.p0;
        result.p1 = raw.p1;
        result.p2 = raw.p2;
        return result;
    }
}

template DfrPaths Scene::trace_dfr_paths<true>(
    const Vector3f &tx_positions,
    const Vector3f &rx_positions,
    const DfrStates &states,
    const DfrMaterial &material,
    const DfrPathOptions &options,
    Mask active) const;
template DfrPathsAD Scene::trace_dfr_paths<false>(
    const Vector3fAD &tx_positions,
    const Vector3fAD &rx_positions,
    const DfrStatesAD &states,
    const DfrMaterialAD &material,
    const DfrPathOptions &options,
    MaskAD active) const;

} // namespace rayd

// Consolidated diffraction accumulation host facade.
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

#include "multipath_internal_jit.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct DfrAccumRaw {
    int grid_cell_count = 0;
    Float power;
    Float field_x_re;
    Float field_x_im;
    Float field_y_re;
    Float field_y_im;
    Float field_z_re;
    Float field_z_im;
    Int direct_count;
    Int keller_count;
    Int suffix_count;
    Int vis_rejects;
    Int edge_vis_rejects;
    Int utd_rejects;
    Int edge_uses;
};

DfrAccumRaw alloc_dfr_accum_raw(int grid_cell_count) {
    DfrAccumRaw raw;
    raw.grid_cell_count = grid_cell_count;
    raw.power = empty<Float>(grid_cell_count);
    raw.field_x_re = empty<Float>(grid_cell_count);
    raw.field_x_im = empty<Float>(grid_cell_count);
    raw.field_y_re = empty<Float>(grid_cell_count);
    raw.field_y_im = empty<Float>(grid_cell_count);
    raw.field_z_re = empty<Float>(grid_cell_count);
    raw.field_z_im = empty<Float>(grid_cell_count);
    raw.direct_count = empty<Int>(1);
    raw.keller_count = empty<Int>(1);
    raw.suffix_count = empty<Int>(1);
    raw.vis_rejects = empty<Int>(1);
    raw.edge_vis_rejects = empty<Int>(1);
    raw.utd_rejects = empty<Int>(1);
    raw.edge_uses = empty<Int>(1);
    return raw;
}

void init_dfr_accum_raw(DfrAccumRaw &raw) {
    const int zero_i = 0;
    const float zero_f = 0.f;
    jit_memset_async(JitBackend::CUDA,
                     raw.power.data(),
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
    jit_memset_async(JitBackend::CUDA, raw.direct_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.keller_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.suffix_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.vis_rejects.data(),
                     1,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.edge_vis_rejects.data(),
                     1,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.utd_rejects.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.edge_uses.data(), 1, sizeof(int), &zero_i);
}

int dfr_accum_direct_sample_count(const DfrOptions &options) {
    return (options.strategy_mask & RAYD_DFR_DIRECT) != 0
               ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
               : 0;
}

int dfr_accum_keller_sample_count(const DfrOptions &options) {
    return (options.strategy_mask & RAYD_DFR_KELLER) != 0
               ? options.keller_samples
               : 0;
}

int dfr_accum_suffix_sample_count(const DfrOptions &options) {
    return (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0
               ? options.suffix_samples
               : 0;
}

int dfr_accum_launch_count(const DfrOptions &options) {
    return dfr_accum_direct_sample_count(options) +
           dfr_accum_keller_sample_count(options) +
           dfr_accum_suffix_sample_count(options);
}

} // namespace

void Scene::ensure_dfr_order1_accumulation_pipeline() const {
    const OptixSceneSelection scenes = select_optix_scenes();
    const OptixScene *primary_scene = scenes.primary;
    require(primary_scene != nullptr && primary_scene->is_ready(),
            "Scene::accum_dfr_direct(): OptiX scene is not ready.");
    require(scenes.hitgroup_record_count > 0,
            "Scene::accum_dfr_direct(): invalid hitgroup record count.");
    auto &pipeline = scenes.split_mode == 0
        ? diffraction_order1_accumulation_primary_pipeline_
        : diffraction_order1_accumulation_pipeline_;
    const OptixPipelineConfig config = scenes.split_mode == 0
        ? diffraction_order1_accumulation_primary_pipeline_config()
        : diffraction_order1_accumulation_pipeline_config();
    ensure_pipeline(pipeline,
                    primary_scene->context(),
                    scenes.hitgroup_record_count,
                    config);
}

void Scene::ensure_dfr_chain_accumulation_pipeline() const {
    const OptixSceneSelection scenes = select_optix_scenes();
    const OptixScene *primary_scene = scenes.primary;
    require(primary_scene != nullptr && primary_scene->is_ready(),
            "Scene::accum_dfr(): OptiX scene is not ready.");
    require(scenes.hitgroup_record_count > 0,
            "Scene::accum_dfr(): invalid hitgroup record count.");
    auto &pipeline = scenes.split_mode == 0
        ? diffraction_chain_accumulation_primary_pipeline_
        : diffraction_chain_accumulation_pipeline_;
    const OptixPipelineConfig config = scenes.split_mode == 0
        ? diffraction_chain_accumulation_primary_pipeline_config()
        : diffraction_chain_accumulation_pipeline_config();
    ensure_pipeline(pipeline,
                    primary_scene->context(),
                    scenes.hitgroup_record_count,
                    config);
}

template <bool Detached>
DfrAccumT<Detached> Scene::accum_dfr_direct(
    const DfrStatesT<Detached> &states,
    const DfrGrid &grid,
    const DfrMaterialT<Detached> &material,
    const DfrOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr_direct(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr_direct(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr_direct(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr_direct(): grid bounds must be ordered.");
    require(grid.cell_area > 0.f,
            "Scene::accum_dfr_direct(): grid.cell_area must be positive.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr_direct(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::accum_dfr_direct(): only max_order == 1 is supported.");

    DfrAccumT<Detached> result;
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    result.grid_cell_count = grid_cell_count;
    if constexpr (!Detached) {
        require_dfr_direct_custom_ad_supported(options);
        return dfr_direct_accum_custom_op(
            this,
            states,
            grid,
            material,
            options,
            triangle_info_.p0,
            triangle_info_.face_normal,
            global_geometry_.vertices,
            global_geometry_.faces,
            active);

        result.power = zeros<FloatAD>(grid_cell_count);
        result.field_x =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.direct_count = full<IntAD>(0, 1);
        result.keller_count = full<IntAD>(0, 1);
        result.suffix_count = full<IntAD>(0, 1);
        result.vis_rejects = full<IntAD>(0, 1);
        result.edge_vis_rejects = full<IntAD>(0, 1);
        result.utd_rejects = full<IntAD>(0, 1);
        result.edge_uses = full<IntAD>(0, 1);

        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (state_count == 0) {
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::accum_dfr_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count &&
                    static_cast<int>(slices(states.wi)) >= state_count &&
                    static_cast<int>(slices(states.d0)) >= state_count &&
                    static_cast<int>(slices(states.prefix_depth)) >= state_count,
                "Scene::accum_dfr_direct(): state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr_direct(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr_direct(): material payload fields must have matching widths.");
        {
            const OptixSceneSelection scenes = select_optix_scenes();
            const OptixScene *primary_scene = scenes.primary;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_direct(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr_direct(): invalid hitgroup record count.");
            ensure_pipeline(diffraction_order1_accumulation_pipeline_,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            diffraction_order1_accumulation_pipeline_config());
        }

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        MaskAD active_ad = active;
        const int active_width = static_cast<int>(slices(active_ad));
        if (active_width == 1 && state_count > 1) {
            active_ad = gather<MaskAD>(active_ad, zeros<IntAD>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_direct(): active width must be 1 or match state count.");
        }

        auto grid_cell_center = [](const DfrGrid &grid_desc,
                                   const IntAD &cell) -> Vector3fAD {
            const IntAD ix = cell % IntAD(grid_desc.resolution0);
            const IntAD iy = cell / IntAD(grid_desc.resolution0);
            const FloatAD u =
                (FloatAD(ix) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution0), FloatAD(1.f));
            const FloatAD v =
                (FloatAD(iy) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution1), FloatAD(1.f));
            const FloatAD c0 =
                FloatAD(grid_desc.coord0_min) +
                u * FloatAD(grid_desc.coord0_max - grid_desc.coord0_min);
            const FloatAD c1 =
                FloatAD(grid_desc.coord1_min) +
                v * FloatAD(grid_desc.coord1_max - grid_desc.coord1_min);
            if (grid_desc.axis == 0) {
                return Vector3fAD(FloatAD(grid_desc.position), c0, c1);
            }
            if (grid_desc.axis == 1) {
                return Vector3fAD(c0, FloatAD(grid_desc.position), c1);
            }
            return Vector3fAD(c0, c1, FloatAD(grid_desc.position));
        };
        auto hash_u32 = [](UIntAD value) -> UIntAD {
            value ^= value >> 16u;
            value *= UIntAD(0x7feb352du);
            value ^= value >> 15u;
            value *= UIntAD(0x846ca68bu);
            value ^= value >> 16u;
            return value;
        };
        auto uniform01 = [&](const UIntAD &sample_lane, unsigned int stream) -> FloatAD {
            const UIntAD h =
                hash_u32(sample_lane ^ (UIntAD(stream) * UIntAD(0x9e3779b9u)) ^
                         UIntAD(static_cast<unsigned int>(options.seed)));
            return FloatAD(h & UIntAD(0x00ffffffu)) * FloatAD(1.f / 16777216.f);
        };
        auto material_gain_for_faces = [&](const IntAD &prim0,
                                           const IntAD &prim1,
                                           const MaskAD &sample_active) -> FloatAD {
            const MaskAD prim0_in_range =
                sample_active && (prim0 >= IntAD(0)) && (prim0 < IntAD(material_count));
            const IntAD safe0 = select(prim0_in_range, prim0, IntAD(0));
            const MaskAD prim0_valid =
                prim0_in_range && gather<MaskAD>(material.valid, safe0, prim0_in_range);
            const MaskAD prim1_in_range =
                sample_active && (prim1 >= IntAD(0)) && (prim1 < IntAD(material_count));
            const IntAD safe1 = select(prim1_in_range, prim1, IntAD(0));
            const MaskAD prim1_valid =
                prim1_in_range && gather<MaskAD>(material.valid, safe1, prim1_in_range);
            const IntAD chosen = select(prim0_valid, safe0, safe1);
            const MaskAD chosen_valid = prim0_valid || prim1_valid;
            const FloatAD gain =
                gather<FloatAD>(material.gain, chosen, chosen_valid);
            return select(chosen_valid, maximum(gain, FloatAD(0.f)), FloatAD(1.f));
        };

        const UIntAD lane = arange<UIntAD>(launch_count);
        const IntAD lane_i = IntAD(lane);
        const MaskAD is_direct = lane_i < IntAD(direct_samples);
        const MaskAD is_keller =
            !is_direct && (lane_i < IntAD(direct_samples + keller_samples));
        const MaskAD is_suffix =
            !is_direct && !is_keller && (lane_i < IntAD(launch_count));
        const IntAD state_idx = IntAD(lane % UIntAD(state_count));
        const IntAD cell =
            IntAD((lane / UIntAD(state_count)) % UIntAD(grid_cell_count));
        const MaskAD lane_active = full<MaskAD>(true, launch_count);
        const MaskAD state_active =
            gather<MaskAD>(active_ad, state_idx, lane_active);

        const Vector3fAD edge_pos =
            gather<Vector3fAD>(states.edge_pos, state_idx, state_active);
        const Vector3fAD edge_dir =
            normalize(gather<Vector3fAD>(states.edge_dir, state_idx, state_active));
        const FloatAD edge_t_min =
            gather<FloatAD>(states.edge_t_min, state_idx, state_active);
        const FloatAD edge_t_max =
            gather<FloatAD>(states.edge_t_max, state_idx, state_active);
        const FloatAD edge_t =
            edge_t_min + uniform01(lane, 0u) * (edge_t_max - edge_t_min);
        const Vector3fAD edge_point = edge_pos + edge_t * edge_dir;
        const Vector3fAD source =
            gather<Vector3fAD>(states.src, state_idx, state_active);
        const FloatAD src_power =
            gather<FloatAD>(states.src_power, state_idx, state_active);
        const IntAD prim0 = gather<IntAD>(states.prim0, state_idx, state_active);
        const IntAD prim1 = gather<IntAD>(states.prim1, state_idx, state_active);
        const FloatAD exterior_angle =
            gather<FloatAD>(states.exterior_angle, state_idx, state_active);
        const Vector3fAD target = grid_cell_center(grid, cell);

        const MaskAD finite_active =
            state_active &&
            drjit::isfinite(source.x()) &&
            drjit::isfinite(source.y()) &&
            drjit::isfinite(source.z()) &&
            drjit::isfinite(edge_point.x()) &&
            drjit::isfinite(edge_point.y()) &&
            drjit::isfinite(edge_point.z()) &&
            drjit::isfinite(src_power);
        const SegmentPairVisibilityAD visibility =
            this->template visible_pair<false>(
                edge_point,
                source,
                target,
                Int(),
                finite_active);
        const MaskAD visible = visibility.visible_a && visibility.visible_b;

        const FloatAD source_distance =
            maximum(norm(edge_point - source), FloatAD(Epsilon));
        const FloatAD target_distance =
            maximum(norm(target - edge_point), FloatAD(Epsilon));
        const FloatAD edge_length =
            maximum(edge_t_max - edge_t_min, FloatAD(0.f));
        const FloatAD wedge_scale =
            minimum(
                maximum(exterior_angle, FloatAD(0.25f * Pi)) / FloatAD(2.f * Pi),
                FloatAD(2.f));
        const FloatAD material_gain =
            material_gain_for_faces(prim0, prim1, finite_active);
        const IntAD strategy_samples = select(
            is_direct,
            IntAD(std::max(direct_samples, 1)),
            select(is_keller,
                   IntAD(std::max(keller_samples, 1)),
                   IntAD(std::max(suffix_samples, 1))));
        const FloatAD contribution =
            src_power *
            material_gain *
            edge_length *
            FloatAD(grid.cell_area) *
            wedge_scale /
            FloatAD(strategy_samples) /
            (source_distance * source_distance * target_distance * target_distance);
        const MaskAD contribution_active =
            visible && (contribution > FloatAD(0.f)) && drjit::isfinite(contribution);
        scatter_reduce(
            ReduceOp::Add,
            result.power,
            contribution,
            cell,
            contribution_active);
        const FloatAD amplitude =
            sqrt(maximum(contribution, FloatAD(0.f)));
        scatter_reduce(
            ReduceOp::Add,
            result.field_x.x(),
            amplitude,
            cell,
            contribution_active);
        scatter_reduce(
            ReduceOp::Add,
            result.direct_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_direct);
        scatter_reduce(
            ReduceOp::Add,
            result.keller_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_keller);
        scatter_reduce(
            ReduceOp::Add,
            result.suffix_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_suffix);
        scatter_reduce(
            ReduceOp::Add,
            result.edge_uses,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && options.collect_edge_use);
        scatter_reduce(
            ReduceOp::Add,
            result.vis_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            finite_active && !visible && options.collect_debug_counts);
        return result;
    } else {
        result.power = zeros<Float>(grid_cell_count);
        result.field_x =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_y =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_z =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.direct_count = full<Int>(0, 1);
        result.keller_count = full<Int>(0, 1);
        result.suffix_count = full<Int>(0, 1);
        result.vis_rejects = full<Int>(0, 1);
        result.edge_vis_rejects = full<Int>(0, 1);
        result.utd_rejects = full<Int>(0, 1);
        result.edge_uses = full<Int>(0, 1);

        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (state_count == 0) {
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::accum_dfr_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count &&
                    static_cast<int>(slices(states.wi)) >= state_count &&
                    static_cast<int>(slices(states.d0)) >= state_count &&
                    static_cast<int>(slices(states.prefix_depth)) >= state_count,
                "Scene::accum_dfr_direct(): state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr_direct(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr_direct(): material payload fields must have matching widths.");

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_direct(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.src.x()) &&
                           drjit::isfinite(states.src.y()) &&
                           drjit::isfinite(states.src.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z()) &&
                           drjit::isfinite(states.src_power);

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        int hitgroup_record_count = 0;
        const int triangle_count =
            static_cast<int>(slices(triangle_info_detached_.p0));
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            hitgroup_record_count = scenes.hitgroup_record_count;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_direct(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::accum_dfr_direct(): invalid hitgroup record count.");
        }
        if (suffix_samples > 0) {
            require(triangle_count > 0,
                    "Scene::accum_dfr_direct(): suffix reflection requires scene triangles.");
            require(material_count >= triangle_count,
                    "Scene::accum_dfr_direct(): suffix reflection requires per-triangle materials.");
        }

        const bool has_suffix_strategy = suffix_samples > 0;
        const bool has_non_suffix_strategy =
            direct_samples > 0 || keller_samples > 0;
        // The CUDA backend is single-scene (split_mode == 0), so it always takes
        // the staged path below (source-visibility prepass then target phases).
        const bool staged_primary = split_mode == 0;
        std::shared_ptr<OptixLaunchPipeline> *dfr_pipeline = nullptr;
        OptixPipelineConfig dfr_pipeline_config;
        if (has_suffix_strategy && has_non_suffix_strategy) {
            dfr_pipeline = split_mode == 0
                ? &diffraction_order1_accumulation_primary_pipeline_
                : &diffraction_order1_accumulation_pipeline_;
            dfr_pipeline_config = split_mode == 0
                ? diffraction_order1_accumulation_primary_pipeline_config()
                : diffraction_order1_accumulation_pipeline_config();
        } else if (has_suffix_strategy) {
            dfr_pipeline = split_mode == 0
                ? &diffraction_order1_accumulation_suffix_primary_pipeline_
                : &diffraction_order1_accumulation_suffix_pipeline_;
            dfr_pipeline_config = split_mode == 0
                ? diffraction_order1_accumulation_suffix_primary_pipeline_config()
                : diffraction_order1_accumulation_suffix_pipeline_config();
        } else {
            dfr_pipeline = split_mode == 0
                ? &diffraction_order1_accumulation_no_suffix_primary_pipeline_
                : &diffraction_order1_accumulation_no_suffix_pipeline_;
            dfr_pipeline_config = split_mode == 0
                ? diffraction_order1_accumulation_no_suffix_primary_pipeline_config()
                : diffraction_order1_accumulation_no_suffix_pipeline_config();
        }

        drjit::eval(states.edge_index,
                    states.edge_pos,
                    states.edge_dir,
                    states.edge_t_min,
                    states.edge_t_max,
                    states.n0,
                    states.n1,
                    states.prim0,
                    states.prim1,
                    states.exterior_angle,
                    states.src,
                    states.src_power,
                    states.wi,
                    states.d0,
                    states.prefix_depth,
                    active_detached,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);
        if (suffix_samples > 0) {
            drjit::eval(triangle_info_detached_.p0,
                        triangle_info_detached_.e1,
                        triangle_info_detached_.e2,
                        triangle_info_detached_.face_normal,
                        face_offsets_);
        }
        if (!staged_primary) {
            ensure_pipeline(*dfr_pipeline,
                            primary_scene->context(),
                            hitgroup_record_count,
                            dfr_pipeline_config);
        }

        DfrAccumRaw raw = alloc_dfr_accum_raw(grid_cell_count);
        init_dfr_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = state_count;
        params.state_edge_index = states.edge_index.data();
        params.state_edge_pos_x = states.edge_pos.x().data();
        params.state_edge_pos_y = states.edge_pos.y().data();
        params.state_edge_pos_z = states.edge_pos.z().data();
        params.state_edge_dir_x = states.edge_dir.x().data();
        params.state_edge_dir_y = states.edge_dir.y().data();
        params.state_edge_dir_z = states.edge_dir.z().data();
        params.state_edge_t_min = states.edge_t_min.data();
        params.state_edge_t_max = states.edge_t_max.data();
        params.state_n0_x = states.n0.x().data();
        params.state_n0_y = states.n0.y().data();
        params.state_n0_z = states.n0.z().data();
        params.state_n1_x = states.n1.x().data();
        params.state_n1_y = states.n1.y().data();
        params.state_n1_z = states.n1.z().data();
        params.state_prim0 = states.prim0.data();
        params.state_prim1 = states.prim1.data();
        params.state_exterior_angle = states.exterior_angle.data();
        params.state_src_x = states.src.x().data();
        params.state_src_y = states.src.y().data();
        params.state_src_z = states.src.z().data();
        params.state_src_power = states.src_power.data();
        params.state_wi_x = states.wi.x().data();
        params.state_wi_y = states.wi.y().data();
        params.state_wi_z = states.wi.z().data();
        params.state_d0_x = states.d0.x().data();
        params.state_d0_y = states.d0.y().data();
        params.state_d0_z = states.d0.z().data();
        params.state_prefix_depth = states.prefix_depth.data();
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.tri_p0_x = suffix_samples > 0 ? triangle_info_detached_.p0.x().data() : nullptr;
        params.tri_p0_y = suffix_samples > 0 ? triangle_info_detached_.p0.y().data() : nullptr;
        params.tri_p0_z = suffix_samples > 0 ? triangle_info_detached_.p0.z().data() : nullptr;
        params.tri_e1_x = suffix_samples > 0 ? triangle_info_detached_.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_samples > 0 ? triangle_info_detached_.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_samples > 0 ? triangle_info_detached_.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_samples > 0 ? triangle_info_detached_.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_samples > 0 ? triangle_info_detached_.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_samples > 0 ? triangle_info_detached_.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_samples > 0 ? triangle_info_detached_.face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_samples > 0 ? triangle_info_detached_.face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_samples > 0 ? triangle_info_detached_.face_normal.z().data() : nullptr;
        params.face_offsets = suffix_samples > 0 ? face_offsets_.data() : nullptr;
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.suffix_candidate_prim_id = nullptr;
        params.suffix_candidate_count = 0;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.seed = options.seed;
        params.samples = options.samples;
        params.max_order = options.max_order;
        params.direct_samples = direct_samples;
        params.keller_samples = keller_samples;
        params.suffix_samples = suffix_samples;
        params.strategy_mask = options.strategy_mask;
        params.sample_sequence = options.sample_sequence;
        params.receiver_model = options.receiver_model;
        params.collect_edge_use = options.collect_edge_use ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.out_power = raw.power.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_direct_count = raw.direct_count.data();
        params.out_keller_count = raw.keller_count.data();
        params.out_suffix_count = raw.suffix_count.data();
        params.out_vis_rejects = raw.vis_rejects.data();
        params.out_edge_vis_rejects =
            raw.edge_vis_rejects.data();
        params.out_utd_rejects = raw.utd_rejects.data();
        params.out_edge_uses = raw.edge_uses.data();
        if (active_dfr_direct_tape_capture != nullptr &&
            active_dfr_direct_tape_capture->launch_count == launch_count) {
            params.tape_active = reinterpret_cast<uint8_t *>(
                active_dfr_direct_tape_capture->active.data());
            params.tape_state_idx =
                active_dfr_direct_tape_capture->state_idx.data();
            params.tape_cell =
                active_dfr_direct_tape_capture->cell.data();
            params.tape_material_idx =
                active_dfr_direct_tape_capture->material_idx.data();
            params.tape_edge_u =
                active_dfr_direct_tape_capture->edge_u.data();
        }

        if (staged_primary) {
            Mask temp_visibility = full<Mask>(false, launch_count);
            drjit::eval(temp_visibility);
            params.temp_visibility =
                reinterpret_cast<uint8_t *>(temp_visibility.data());
            if (cuda_trace) {
                cuda_backend().run_dfr_accum_direct(params, has_non_suffix_strategy,
                                                    has_suffix_strategy, launch_count);
            } else {
                ensure_pipeline(diffraction_order1_source_visibility_primary_pipeline_,
                                primary_scene->context(),
                                hitgroup_record_count,
                                diffraction_order1_source_visibility_primary_pipeline_config());
                diffraction_order1_source_visibility_primary_pipeline_->launch(0, params);
                if (has_non_suffix_strategy) {
                    ensure_pipeline(diffraction_order1_no_suffix_target_primary_pipeline_,
                                    primary_scene->context(),
                                    hitgroup_record_count,
                                    diffraction_order1_no_suffix_target_primary_pipeline_config());
                    diffraction_order1_no_suffix_target_primary_pipeline_->launch(0, params);
                }
                if (has_suffix_strategy) {
                    ensure_pipeline(diffraction_order1_suffix_first_visibility_primary_pipeline_,
                                    primary_scene->context(),
                                    hitgroup_record_count,
                                    diffraction_order1_suffix_first_visibility_primary_pipeline_config());
                    ensure_pipeline(diffraction_order1_suffix_target_primary_pipeline_,
                                    primary_scene->context(),
                                    hitgroup_record_count,
                                    diffraction_order1_suffix_target_primary_pipeline_config());
                    diffraction_order1_suffix_first_visibility_primary_pipeline_->launch(0, params);
                    diffraction_order1_suffix_target_primary_pipeline_->launch(0, params);
                }
                drjit::sync_thread();
            }
        } else {
            // Split-scene (split_mode != 0) is OptiX-only; the CUDA backend is
            // single-scene and always takes the staged path above.
            (*dfr_pipeline)->launch(0, params);
        }

        result.power = raw.power;
        result.field_x =
            drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.field_y =
            drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.field_z =
            drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.direct_count = raw.direct_count;
        result.keller_count = raw.keller_count;
        result.suffix_count = raw.suffix_count;
        result.vis_rejects = raw.vis_rejects;
        result.edge_vis_rejects =
            raw.edge_vis_rejects;
        result.utd_rejects = raw.utd_rejects;
        result.edge_uses = raw.edge_uses;
        return result;
    }
}

template <bool Detached>
DfrAccumT<Detached> Scene::accum_dfr(
    const DfrStatesT<Detached> &initial_states,
    const DfrStatesT<Detached> &recursive_states,
    const DfrGrid &grid,
    const DfrMaterialT<Detached> &material,
    const DfrOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr(): grid bounds must be ordered.");
    require(grid.cell_area > 0.f,
            "Scene::accum_dfr(): grid.cell_area must be positive.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr(): wavelength must be positive.");
    require(options.max_order == 2 || options.max_order == 3,
            "Scene::accum_dfr(): only max_order 2 or 3 is supported.");

    DfrAccumT<Detached> result;
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    result.grid_cell_count = grid_cell_count;
    if constexpr (!Detached) {
        require_dfr_chain_custom_ad_supported(options);
        return dfr_chain_accum_custom_op(
            this,
            initial_states,
            recursive_states,
            grid,
            material,
            options,
            triangle_info_.p0,
            triangle_info_.face_normal,
            global_geometry_.vertices,
            global_geometry_.faces,
            active);

        result.power = zeros<FloatAD>(grid_cell_count);
        result.field_x =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.direct_count = full<IntAD>(0, 1);
        result.keller_count = full<IntAD>(0, 1);
        result.suffix_count = full<IntAD>(0, 1);
        result.vis_rejects = full<IntAD>(0, 1);
        result.edge_vis_rejects = full<IntAD>(0, 1);
        result.utd_rejects = full<IntAD>(0, 1);
        result.edge_uses = full<IntAD>(0, 1);

        const int initial_width = static_cast<int>(slices(initial_states.edge_index));
        const int initial_count =
            initial_states.count > 0 ? initial_states.count : initial_width;
        const int recursive_width = static_cast<int>(slices(recursive_states.edge_index));
        const int recursive_count =
            recursive_states.count > 0 ? recursive_states.count : recursive_width;
        if (initial_count == 0 || recursive_count == 0) {
            return result;
        }
        require(initial_count > 0 && initial_count <= initial_width,
                "Scene::accum_dfr(): invalid initial state count.");
        require(recursive_count > 0 && recursive_count <= recursive_width,
                "Scene::accum_dfr(): invalid recursive state count.");
        require(static_cast<int>(slices(initial_states.edge_pos)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_dir)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_min)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_max)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim0)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim1)) >= initial_count &&
                    static_cast<int>(slices(initial_states.exterior_angle)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src_power)) >= initial_count,
                "Scene::accum_dfr(): initial state fields must cover state count.");
        require(static_cast<int>(slices(recursive_states.edge_index)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_pos)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_dir)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_min)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_max)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim0)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim1)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.exterior_angle)) >= recursive_count,
                "Scene::accum_dfr(): recursive state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr(): material payload fields must have matching widths.");
        {
            const OptixSceneSelection scenes = select_optix_scenes();
            const OptixScene *primary_scene = scenes.primary;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr(): invalid hitgroup record count.");
            ensure_pipeline(diffraction_chain_accumulation_pipeline_,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            diffraction_chain_accumulation_pipeline_config());
        }

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        MaskAD active_ad = active;
        const int active_width = static_cast<int>(slices(active_ad));
        if (active_width == 1 && initial_count > 1) {
            active_ad = gather<MaskAD>(active_ad, zeros<IntAD>(initial_count));
        } else {
            require(active_width == initial_count,
                    "Scene::accum_dfr(): active width must be 1 or match initial state count.");
        }

        auto grid_cell_center = [](const DfrGrid &grid_desc,
                                   const IntAD &cell) -> Vector3fAD {
            const IntAD ix = cell % IntAD(grid_desc.resolution0);
            const IntAD iy = cell / IntAD(grid_desc.resolution0);
            const FloatAD u =
                (FloatAD(ix) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution0), FloatAD(1.f));
            const FloatAD v =
                (FloatAD(iy) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution1), FloatAD(1.f));
            const FloatAD c0 =
                FloatAD(grid_desc.coord0_min) +
                u * FloatAD(grid_desc.coord0_max - grid_desc.coord0_min);
            const FloatAD c1 =
                FloatAD(grid_desc.coord1_min) +
                v * FloatAD(grid_desc.coord1_max - grid_desc.coord1_min);
            if (grid_desc.axis == 0) {
                return Vector3fAD(FloatAD(grid_desc.position), c0, c1);
            }
            if (grid_desc.axis == 1) {
                return Vector3fAD(c0, FloatAD(grid_desc.position), c1);
            }
            return Vector3fAD(c0, c1, FloatAD(grid_desc.position));
        };
        auto hash_u32 = [](UIntAD value) -> UIntAD {
            value ^= value >> 16u;
            value *= UIntAD(0x7feb352du);
            value ^= value >> 15u;
            value *= UIntAD(0x846ca68bu);
            value ^= value >> 16u;
            return value;
        };
        auto uniform01 = [&](const UIntAD &sample_lane, unsigned int stream) -> FloatAD {
            const UIntAD h =
                hash_u32(sample_lane ^ (UIntAD(stream) * UIntAD(0x9e3779b9u)) ^
                         UIntAD(static_cast<unsigned int>(options.seed)));
            return FloatAD(h & UIntAD(0x00ffffffu)) * FloatAD(1.f / 16777216.f);
        };

        auto material_gain_for_faces = [&](const IntAD &prim0,
                                           const IntAD &prim1,
                                           const MaskAD &sample_active) -> FloatAD {
            const MaskAD prim0_in_range =
                sample_active && (prim0 >= IntAD(0)) && (prim0 < IntAD(material_count));
            const IntAD safe0 = select(prim0_in_range, prim0, IntAD(0));
            const MaskAD prim0_valid =
                prim0_in_range && gather<MaskAD>(material.valid, safe0, prim0_in_range);
            const MaskAD prim1_in_range =
                sample_active && (prim1 >= IntAD(0)) && (prim1 < IntAD(material_count));
            const IntAD safe1 = select(prim1_in_range, prim1, IntAD(0));
            const MaskAD prim1_valid =
                prim1_in_range && gather<MaskAD>(material.valid, safe1, prim1_in_range);
            const IntAD chosen = select(prim0_valid, safe0, safe1);
            const MaskAD chosen_valid = prim0_valid || prim1_valid;
            const FloatAD gain =
                gather<FloatAD>(material.gain, chosen, chosen_valid);
            return select(chosen_valid, maximum(gain, FloatAD(0.f)), FloatAD(1.f));
        };

        auto chain_event_weight = [&](const FloatAD &src_power,
                                      const IntAD &face0_prim,
                                      const IntAD &face1_prim,
                                      const FloatAD &edge_t_min,
                                      const FloatAD &edge_t_max,
                                      const FloatAD &exterior_angle,
                                      const Vector3fAD &source,
                                      const Vector3fAD &edge_point,
                                      const Vector3fAD &target,
                                      const MaskAD &sample_active) -> FloatAD {
            const FloatAD source_distance =
                maximum(norm(edge_point - source), FloatAD(Epsilon));
            const FloatAD target_distance =
                maximum(norm(target - edge_point), FloatAD(Epsilon));
            const FloatAD edge_length =
                maximum(edge_t_max - edge_t_min, FloatAD(0.f));
            const FloatAD wedge_scale =
                minimum(
                    maximum(exterior_angle, FloatAD(0.25f * Pi)) / FloatAD(2.f * Pi),
                    FloatAD(2.f));
            const FloatAD material_gain =
                material_gain_for_faces(face0_prim, face1_prim, sample_active);
            return src_power *
                   material_gain *
                   edge_length *
                   wedge_scale /
                   (source_distance * source_distance * target_distance * target_distance);
        };

        const UIntAD lane = arange<UIntAD>(launch_count);
        const IntAD lane_i = IntAD(lane);
        const MaskAD is_direct = lane_i < IntAD(direct_samples);
        const MaskAD is_keller =
            !is_direct && (lane_i < IntAD(direct_samples + keller_samples));
        const MaskAD is_suffix =
            !is_direct && !is_keller && (lane_i < IntAD(launch_count));
        const IntAD first_idx = IntAD(lane % UIntAD(initial_count));
        const UIntAD second_hash = hash_u32(
            lane ^ (UIntAD(static_cast<unsigned int>(options.seed)) * UIntAD(0x9e3779b9u)) ^
            UIntAD(0x51ed270bu));
        const IntAD second_idx = IntAD(second_hash % UIntAD(recursive_count));
        const UIntAD third_hash = hash_u32(
            lane ^ (UIntAD(static_cast<unsigned int>(options.seed)) * UIntAD(0x85ebca6bu)) ^
            UIntAD(0xc2b2ae35u));
        const IntAD third_idx =
            IntAD(third_hash % UIntAD(recursive_count));
        const IntAD cell =
            IntAD((lane / UIntAD(initial_count)) % UIntAD(grid_cell_count));
        const MaskAD lane_active = full<MaskAD>(true, launch_count);
        const MaskAD first_active =
            gather<MaskAD>(active_ad, first_idx, lane_active);
        const IntAD first_edge_index =
            gather<IntAD>(initial_states.edge_index, first_idx, first_active);
        const IntAD second_edge_index =
            gather<IntAD>(recursive_states.edge_index, second_idx, first_active);
        const IntAD third_edge_index =
            gather<IntAD>(recursive_states.edge_index, third_idx, first_active);
        const MaskAD distinct_edges =
            (first_edge_index != second_edge_index) &&
            ((IntAD(options.max_order) == IntAD(2)) ||
             ((first_edge_index != third_edge_index) &&
              (second_edge_index != third_edge_index)));

        const Vector3fAD first_edge_pos =
            gather<Vector3fAD>(initial_states.edge_pos, first_idx, first_active);
        const Vector3fAD first_edge_dir =
            normalize(gather<Vector3fAD>(initial_states.edge_dir, first_idx, first_active));
        const FloatAD first_t_min =
            gather<FloatAD>(initial_states.edge_t_min, first_idx, first_active);
        const FloatAD first_t_max =
            gather<FloatAD>(initial_states.edge_t_max, first_idx, first_active);
        const FloatAD first_t =
            first_t_min + uniform01(lane, 0u) * (first_t_max - first_t_min);
        const Vector3fAD first_point = first_edge_pos + first_t * first_edge_dir;

        const Vector3fAD second_edge_pos =
            gather<Vector3fAD>(recursive_states.edge_pos, second_idx, first_active);
        const Vector3fAD second_edge_dir =
            normalize(gather<Vector3fAD>(recursive_states.edge_dir, second_idx, first_active));
        const FloatAD second_t_min =
            gather<FloatAD>(recursive_states.edge_t_min, second_idx, first_active);
        const FloatAD second_t_max =
            gather<FloatAD>(recursive_states.edge_t_max, second_idx, first_active);
        const FloatAD second_t =
            second_t_min + uniform01(lane, 2u) * (second_t_max - second_t_min);
        const Vector3fAD second_point = second_edge_pos + second_t * second_edge_dir;

        const Vector3fAD third_edge_pos =
            gather<Vector3fAD>(recursive_states.edge_pos, third_idx, first_active);
        const Vector3fAD third_edge_dir =
            normalize(gather<Vector3fAD>(recursive_states.edge_dir, third_idx, first_active));
        const FloatAD third_t_min =
            gather<FloatAD>(recursive_states.edge_t_min, third_idx, first_active);
        const FloatAD third_t_max =
            gather<FloatAD>(recursive_states.edge_t_max, third_idx, first_active);
        const FloatAD third_t =
            third_t_min + uniform01(lane, 4u) * (third_t_max - third_t_min);
        const Vector3fAD third_point = third_edge_pos + third_t * third_edge_dir;

        const Vector3fAD source =
            gather<Vector3fAD>(initial_states.src, first_idx, first_active);
        const FloatAD src_power =
            gather<FloatAD>(initial_states.src_power, first_idx, first_active);
        const Vector3fAD target = grid_cell_center(grid, cell);
        const Vector3fAD terminal_point =
            select(IntAD(options.max_order) == IntAD(3), third_point, second_point);

        const MaskAD finite_active =
            first_active &&
            distinct_edges &&
            drjit::isfinite(source.x()) &&
            drjit::isfinite(source.y()) &&
            drjit::isfinite(source.z()) &&
            drjit::isfinite(first_point.x()) &&
            drjit::isfinite(first_point.y()) &&
            drjit::isfinite(first_point.z()) &&
            drjit::isfinite(second_point.x()) &&
            drjit::isfinite(second_point.y()) &&
            drjit::isfinite(second_point.z()) &&
            drjit::isfinite(src_power);

        const SegmentPairVisibilityAD first_visibility =
            this->template visible_pair<false>(
                first_point,
                source,
                second_point,
                Int(),
                finite_active);
        const SegmentPairVisibilityAD terminal_visibility =
            this->template visible_pair<false>(
                terminal_point,
                select(IntAD(options.max_order) == IntAD(3), second_point, target),
                target,
                Int(),
                finite_active);
        const MaskAD source_visible = first_visibility.visible_a;
        const MaskAD first_edge_visible = first_visibility.visible_b;
        const MaskAD second_edge_visible =
            select(IntAD(options.max_order) == IntAD(3),
                   terminal_visibility.visible_a,
                   full<MaskAD>(true, launch_count));
        const MaskAD target_visible =
            select(IntAD(options.max_order) == IntAD(3),
                   terminal_visibility.visible_b,
                   terminal_visibility.visible_a);
        const MaskAD visible =
            source_visible && first_edge_visible && second_edge_visible && target_visible;

        const IntAD first_prim0 =
            gather<IntAD>(initial_states.prim0, first_idx, finite_active);
        const IntAD first_prim1 =
            gather<IntAD>(initial_states.prim1, first_idx, finite_active);
        const FloatAD first_exterior =
            gather<FloatAD>(initial_states.exterior_angle, first_idx, finite_active);
        const FloatAD first_weight = chain_event_weight(
            src_power,
            first_prim0,
            first_prim1,
            first_t_min,
            first_t_max,
            first_exterior,
            source,
            first_point,
            second_point,
            finite_active);

        const IntAD second_prim0 =
            gather<IntAD>(recursive_states.prim0, second_idx, finite_active);
        const IntAD second_prim1 =
            gather<IntAD>(recursive_states.prim1, second_idx, finite_active);
        const FloatAD second_exterior =
            gather<FloatAD>(recursive_states.exterior_angle, second_idx, finite_active);
        const Vector3fAD second_target =
            select(IntAD(options.max_order) == IntAD(3), third_point, target);
        const FloatAD second_weight = chain_event_weight(
            FloatAD(1.f),
            second_prim0,
            second_prim1,
            second_t_min,
            second_t_max,
            second_exterior,
            first_point,
            second_point,
            second_target,
            finite_active);

        const IntAD third_prim0 =
            gather<IntAD>(recursive_states.prim0, third_idx, finite_active);
        const IntAD third_prim1 =
            gather<IntAD>(recursive_states.prim1, third_idx, finite_active);
        const FloatAD third_exterior =
            gather<FloatAD>(recursive_states.exterior_angle, third_idx, finite_active);
        const FloatAD third_weight = chain_event_weight(
            FloatAD(1.f),
            third_prim0,
            third_prim1,
            third_t_min,
            third_t_max,
            third_exterior,
            second_point,
            third_point,
            target,
            finite_active);
        FloatAD chain_weight = first_weight * second_weight;
        chain_weight = select(IntAD(options.max_order) == IntAD(3),
                              chain_weight * third_weight,
                              chain_weight);

        const FloatAD wave_gain_per_event =
            (FloatAD(options.wavelength) / FloatAD(4.f * Pi)) *
            (FloatAD(options.wavelength) / FloatAD(4.f * Pi));
        const FloatAD wave_gain =
            select(IntAD(options.max_order) == IntAD(3),
                   wave_gain_per_event * wave_gain_per_event,
                   wave_gain_per_event);
        const IntAD strategy_samples = select(
            is_direct,
            IntAD(std::max(direct_samples, 1)),
            select(is_keller,
                   IntAD(std::max(keller_samples, 1)),
                   IntAD(std::max(suffix_samples, 1))));
        const FloatAD contribution =
            chain_weight *
            wave_gain *
            FloatAD(grid.cell_area) /
            FloatAD(strategy_samples);
        const MaskAD contribution_active =
            visible && (contribution > FloatAD(0.f)) && drjit::isfinite(contribution);
        scatter_reduce(
            ReduceOp::Add,
            result.power,
            contribution,
            cell,
            contribution_active);
        const FloatAD amplitude =
            sqrt(maximum(contribution, FloatAD(0.f)));
        scatter_reduce(
            ReduceOp::Add,
            result.field_x.x(),
            amplitude,
            cell,
            contribution_active);
        scatter_reduce(
            ReduceOp::Add,
            result.direct_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_direct);
        scatter_reduce(
            ReduceOp::Add,
            result.keller_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_keller);
        scatter_reduce(
            ReduceOp::Add,
            result.suffix_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_suffix);
        scatter_reduce(
            ReduceOp::Add,
            result.edge_uses,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && options.collect_edge_use);
        scatter_reduce(
            ReduceOp::Add,
            result.vis_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            finite_active && !target_visible && options.collect_debug_counts);
        scatter_reduce(
            ReduceOp::Add,
            result.edge_vis_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            finite_active && target_visible &&
                (!source_visible || !first_edge_visible || !second_edge_visible) &&
                options.collect_debug_counts);
        scatter_reduce(
            ReduceOp::Add,
            result.utd_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            first_active && !distinct_edges && options.collect_debug_counts);
        return result;
    } else {
        result.power = zeros<Float>(grid_cell_count);
        result.field_x =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_y =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_z =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.direct_count = full<Int>(0, 1);
        result.keller_count = full<Int>(0, 1);
        result.suffix_count = full<Int>(0, 1);
        result.vis_rejects = full<Int>(0, 1);
        result.edge_vis_rejects = full<Int>(0, 1);
        result.utd_rejects = full<Int>(0, 1);
        result.edge_uses = full<Int>(0, 1);

        const int initial_width = static_cast<int>(slices(initial_states.edge_index));
        const int initial_count =
            initial_states.count > 0 ? initial_states.count : initial_width;
        const int recursive_width = static_cast<int>(slices(recursive_states.edge_index));
        const int recursive_count =
            recursive_states.count > 0 ? recursive_states.count : recursive_width;
        if (initial_count == 0 || recursive_count == 0) {
            return result;
        }
        require(initial_count > 0 && initial_count <= initial_width,
                "Scene::accum_dfr(): invalid initial state count.");
        require(recursive_count > 0 && recursive_count <= recursive_width,
                "Scene::accum_dfr(): invalid recursive state count.");
        require(static_cast<int>(slices(initial_states.edge_pos)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_dir)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_min)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_max)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim0)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim1)) >= initial_count &&
                    static_cast<int>(slices(initial_states.exterior_angle)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src_power)) >= initial_count,
                "Scene::accum_dfr(): initial state fields must cover state count.");
        require(static_cast<int>(slices(recursive_states.edge_pos)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_dir)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_min)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_max)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim0)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim1)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.exterior_angle)) >= recursive_count,
                "Scene::accum_dfr(): recursive state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr(): material payload fields must have matching widths.");

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && initial_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(initial_count));
        } else {
            require(active_width == initial_count,
                    "Scene::accum_dfr(): active width must be 1 or match initial state count.");
        }
        active_detached &= drjit::isfinite(initial_states.src.x()) &&
                           drjit::isfinite(initial_states.src.y()) &&
                           drjit::isfinite(initial_states.src.z()) &&
                           drjit::isfinite(initial_states.edge_pos.x()) &&
                           drjit::isfinite(initial_states.edge_pos.y()) &&
                           drjit::isfinite(initial_states.edge_pos.z()) &&
                           drjit::isfinite(initial_states.src_power);
        Mask recursive_active = drjit::isfinite(recursive_states.edge_pos.x()) &&
                                drjit::isfinite(recursive_states.edge_pos.y()) &&
                                drjit::isfinite(recursive_states.edge_pos.z()) &&
                                drjit::isfinite(recursive_states.edge_dir.x()) &&
                                drjit::isfinite(recursive_states.edge_dir.y()) &&
                                drjit::isfinite(recursive_states.edge_dir.z());

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        int hitgroup_record_count = 0;
        const int triangle_count =
            static_cast<int>(slices(triangle_info_detached_.p0));
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            hitgroup_record_count = scenes.hitgroup_record_count;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::accum_dfr(): invalid hitgroup record count.");
        }
        if (suffix_samples > 0) {
            require(triangle_count > 0,
                    "Scene::accum_dfr(): suffix reflection requires scene triangles.");
            require(material_count >= triangle_count,
                    "Scene::accum_dfr(): suffix reflection requires per-triangle materials.");
        }

        if (!cuda_trace) {
            auto &dfr_pipeline = split_mode == 0
                ? diffraction_chain_accumulation_primary_pipeline_
                : diffraction_chain_accumulation_pipeline_;
            const OptixPipelineConfig dfr_pipeline_config = split_mode == 0
                ? diffraction_chain_accumulation_primary_pipeline_config()
                : diffraction_chain_accumulation_pipeline_config();

            ensure_pipeline(dfr_pipeline,
                            primary_scene->context(),
                            hitgroup_record_count,
                            dfr_pipeline_config);
        }

        drjit::eval(initial_states.edge_index,
                    initial_states.edge_pos,
                    initial_states.edge_dir,
                    initial_states.edge_t_min,
                    initial_states.edge_t_max,
                    initial_states.prim0,
                    initial_states.prim1,
                    initial_states.exterior_angle,
                    initial_states.src,
                    initial_states.src_power,
                    recursive_states.edge_index,
                    recursive_states.edge_pos,
                    recursive_states.edge_dir,
                    recursive_states.edge_t_min,
                    recursive_states.edge_t_max,
                    recursive_states.prim0,
                    recursive_states.prim1,
                    recursive_states.exterior_angle,
                    active_detached,
                    recursive_active,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);
        if (suffix_samples > 0) {
            drjit::eval(triangle_info_detached_.p0,
                        triangle_info_detached_.e1,
                        triangle_info_detached_.e2,
                        triangle_info_detached_.face_normal,
                        face_offsets_);
        }

        DfrAccumRaw raw = alloc_dfr_accum_raw(grid_cell_count);
        init_dfr_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = initial_count;
        params.state_edge_index = initial_states.edge_index.data();
        params.state_edge_pos_x = initial_states.edge_pos.x().data();
        params.state_edge_pos_y = initial_states.edge_pos.y().data();
        params.state_edge_pos_z = initial_states.edge_pos.z().data();
        params.state_edge_dir_x = initial_states.edge_dir.x().data();
        params.state_edge_dir_y = initial_states.edge_dir.y().data();
        params.state_edge_dir_z = initial_states.edge_dir.z().data();
        params.state_edge_t_min = initial_states.edge_t_min.data();
        params.state_edge_t_max = initial_states.edge_t_max.data();
        params.state_prim0 = initial_states.prim0.data();
        params.state_prim1 = initial_states.prim1.data();
        params.state_exterior_angle = initial_states.exterior_angle.data();
        params.state_src_x = initial_states.src.x().data();
        params.state_src_y = initial_states.src.y().data();
        params.state_src_z = initial_states.src.z().data();
        params.state_src_power = initial_states.src_power.data();

        params.recursive_state_count = recursive_count;
        params.recursive_active_mask =
            reinterpret_cast<const uint8_t *>(recursive_active.data());
        params.recursive_state_edge_index = recursive_states.edge_index.data();
        params.recursive_state_edge_pos_x = recursive_states.edge_pos.x().data();
        params.recursive_state_edge_pos_y = recursive_states.edge_pos.y().data();
        params.recursive_state_edge_pos_z = recursive_states.edge_pos.z().data();
        params.recursive_state_edge_dir_x = recursive_states.edge_dir.x().data();
        params.recursive_state_edge_dir_y = recursive_states.edge_dir.y().data();
        params.recursive_state_edge_dir_z = recursive_states.edge_dir.z().data();
        params.recursive_state_edge_t_min = recursive_states.edge_t_min.data();
        params.recursive_state_edge_t_max = recursive_states.edge_t_max.data();
        params.recursive_state_prim0 = recursive_states.prim0.data();
        params.recursive_state_prim1 = recursive_states.prim1.data();
        params.recursive_state_exterior_angle = recursive_states.exterior_angle.data();

        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.tri_p0_x = suffix_samples > 0 ? triangle_info_detached_.p0.x().data() : nullptr;
        params.tri_p0_y = suffix_samples > 0 ? triangle_info_detached_.p0.y().data() : nullptr;
        params.tri_p0_z = suffix_samples > 0 ? triangle_info_detached_.p0.z().data() : nullptr;
        params.tri_e1_x = suffix_samples > 0 ? triangle_info_detached_.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_samples > 0 ? triangle_info_detached_.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_samples > 0 ? triangle_info_detached_.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_samples > 0 ? triangle_info_detached_.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_samples > 0 ? triangle_info_detached_.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_samples > 0 ? triangle_info_detached_.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_samples > 0 ? triangle_info_detached_.face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_samples > 0 ? triangle_info_detached_.face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_samples > 0 ? triangle_info_detached_.face_normal.z().data() : nullptr;
        params.face_offsets = suffix_samples > 0 ? face_offsets_.data() : nullptr;
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.suffix_candidate_prim_id = nullptr;
        params.suffix_candidate_count = 0;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.seed = options.seed;
        params.samples = options.samples;
        params.max_order = options.max_order;
        params.direct_samples = direct_samples;
        params.keller_samples = keller_samples;
        params.suffix_samples = suffix_samples;
        params.strategy_mask = options.strategy_mask;
        params.sample_sequence = options.sample_sequence;
        params.receiver_model = options.receiver_model;
        params.collect_edge_use = options.collect_edge_use ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.out_power = raw.power.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_direct_count = raw.direct_count.data();
        params.out_keller_count = raw.keller_count.data();
        params.out_suffix_count = raw.suffix_count.data();
        params.out_vis_rejects = raw.vis_rejects.data();
        params.out_edge_vis_rejects =
            raw.edge_vis_rejects.data();
        params.out_utd_rejects = raw.utd_rejects.data();
        params.out_edge_uses = raw.edge_uses.data();
        if (active_dfr_direct_tape_capture != nullptr &&
            active_dfr_direct_tape_capture->launch_count == launch_count) {
            params.tape_active = reinterpret_cast<uint8_t *>(
                active_dfr_direct_tape_capture->active.data());
            params.tape_state_idx =
                active_dfr_direct_tape_capture->state_idx.data();
            params.tape_cell =
                active_dfr_direct_tape_capture->cell.data();
            params.tape_material_idx =
                active_dfr_direct_tape_capture->material_idx.data();
            params.tape_edge_u =
                active_dfr_direct_tape_capture->edge_u.data();
        }

        if (cuda_trace) {
            cuda_backend().run_dfr_accum_chain(params, launch_count);
        } else {
            (split_mode == 0 ? diffraction_chain_accumulation_primary_pipeline_
                             : diffraction_chain_accumulation_pipeline_)
                ->launch(0, params);
        }

        result.power = raw.power;
        result.field_x =
            drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.field_y =
            drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.field_z =
            drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.direct_count = raw.direct_count;
        result.keller_count = raw.keller_count;
        result.suffix_count = raw.suffix_count;
        result.vis_rejects = raw.vis_rejects;
        result.edge_vis_rejects =
            raw.edge_vis_rejects;
        result.utd_rejects = raw.utd_rejects;
        result.edge_uses = raw.edge_uses;
        return result;
    }
}

template DfrAccum Scene::accum_dfr_direct<true>(
    const DfrStates &states,
    const DfrGrid &grid,
    const DfrMaterial &material,
    const DfrOptions &options,
    Mask active) const;
template DfrAccumAD Scene::accum_dfr_direct<false>(
    const DfrStatesAD &states,
    const DfrGrid &grid,
    const DfrMaterialAD &material,
    const DfrOptions &options,
    MaskAD active) const;

template DfrAccum Scene::accum_dfr<true>(
    const DfrStates &initial_states,
    const DfrStates &recursive_states,
    const DfrGrid &grid,
    const DfrMaterial &material,
    const DfrOptions &options,
    Mask active) const;
template DfrAccumAD Scene::accum_dfr<false>(
    const DfrStatesAD &initial_states,
    const DfrStatesAD &recursive_states,
    const DfrGrid &grid,
    const DfrMaterialAD &material,
    const DfrOptions &options,
    MaskAD active) const;

} // namespace rayd

// Consolidated coherent diffraction host facade.
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

#include "multipath_internal_jit.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct DfrCoherentAccumRaw {
    int grid_cell_count = 0;
    Float direct_field_x_re;
    Float direct_field_x_im;
    Float direct_field_y_re;
    Float direct_field_y_im;
    Float direct_field_z_re;
    Float direct_field_z_im;
    Float multi_field_x_re;
    Float multi_field_x_im;
    Float multi_field_y_re;
    Float multi_field_y_im;
    Float multi_field_z_re;
    Float multi_field_z_im;
    Int direct_count;
    Int multi_count;
    Int visibility_reject_count;
    Int utd_reject_count;
};

DfrCoherentAccumRaw alloc_dfr_coherent_accum_raw(int grid_cell_count) {
    DfrCoherentAccumRaw raw;
    raw.grid_cell_count = grid_cell_count;
    raw.direct_field_x_re = empty<Float>(grid_cell_count);
    raw.direct_field_x_im = empty<Float>(grid_cell_count);
    raw.direct_field_y_re = empty<Float>(grid_cell_count);
    raw.direct_field_y_im = empty<Float>(grid_cell_count);
    raw.direct_field_z_re = empty<Float>(grid_cell_count);
    raw.direct_field_z_im = empty<Float>(grid_cell_count);
    raw.multi_field_x_re = empty<Float>(grid_cell_count);
    raw.multi_field_x_im = empty<Float>(grid_cell_count);
    raw.multi_field_y_re = empty<Float>(grid_cell_count);
    raw.multi_field_y_im = empty<Float>(grid_cell_count);
    raw.multi_field_z_re = empty<Float>(grid_cell_count);
    raw.multi_field_z_im = empty<Float>(grid_cell_count);
    raw.direct_count = empty<Int>(grid_cell_count);
    raw.multi_count = empty<Int>(grid_cell_count);
    raw.visibility_reject_count = empty<Int>(grid_cell_count);
    raw.utd_reject_count = empty<Int>(grid_cell_count);
    return raw;
}

void init_dfr_coherent_accum_raw(DfrCoherentAccumRaw &raw) {
    const int zero_i = 0;
    const float zero_f = 0.f;
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.visibility_reject_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.utd_reject_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
}

Vector3f normalize_with_fallback(const Vector3f &value, const Vector3f &fallback) {
    const Float value_norm = norm(value);
    const Float fallback_norm = norm(fallback);
    return select(value_norm > Float(1.0e-8f),
                  value / (value_norm + Float(1.0e-12f)),
                  fallback / (fallback_norm + Float(1.0e-12f)));
}

Vector3f stable_perpendicular_basis_jit(const Vector3f &ray_dir, const Vector3f &preferred) {
    const Vector3f proj = preferred - dot(preferred, ray_dir) * ray_dir;
    const Mask use_z = abs(ray_dir.z()) < Float(0.9f);
    const Vector3f alt_axis = select(use_z,
                                     Vector3f(Float(0.f), Float(0.f), Float(1.f)),
                                     Vector3f(Float(0.f), Float(1.f), Float(0.f)));
    const Vector3f alt_proj = alt_axis - dot(alt_axis, ray_dir) * ray_dir;
    return normalize_with_fallback(proj, alt_proj);
}

Mask wedge_exterior_mask_jit(const Vector3f &direction_from_edge,
                             const Vector3f &edge_dir,
                             const Vector3f &n0,
                             const Vector3f &nn) {
    const Vector3f direction_proj =
        direction_from_edge - dot(direction_from_edge, edge_dir) * edge_dir;
    return (norm(direction_proj) > Float(1.0e-8f)) &&
           ((dot(direction_proj, n0) >= Float(-1.0e-8f)) ||
            (dot(direction_proj, nn) >= Float(-1.0e-8f)));
}

Int interleave_two_ignore_slots(const Int &slot0, const Int &slot1, int width) {
    if (width <= 0) {
        return zeros<Int>(0);
    }
    const Int slot_major = concat(slot0, slot1);
    const UInt dst_idx = arange<UInt>(width * 2);
    const UInt ray_idx = dst_idx / UInt(2);
    const UInt slot_idx = dst_idx - ray_idx * UInt(2);
    const UInt src_idx = slot_idx * UInt(width) + ray_idx;
    return gather<Int>(slot_major, src_idx);
}

Int interleave_four_ignore_slots(const Int &slot0,
                                 const Int &slot1,
                                 const Int &slot2,
                                 const Int &slot3,
                                 int width) {
    if (width <= 0) {
        return zeros<Int>(0);
    }
    const Int slot_major = concat(concat(slot0, slot1), concat(slot2, slot3));
    const UInt dst_idx = arange<UInt>(width * 4);
    const UInt ray_idx = dst_idx / UInt(4);
    const UInt slot_idx = dst_idx - ray_idx * UInt(4);
    const UInt src_idx = slot_idx * UInt(width) + ray_idx;
    return gather<Int>(slot_major, src_idx);
}

Float gather_material_float(const Float &values,
                            const Int &face,
                            const Mask &valid,
                            float fallback) {
    const UInt safe = UInt(select(valid, face, Int(0)));
    return select(valid, gather<Float>(values, safe), Float(fallback));
}

Mask gather_material_mask(const Mask &values, const Int &face, const Mask &valid) {
    const UInt safe = UInt(select(valid, face, Int(0)));
    return valid && gather<Mask>(values, safe);
}

} // namespace

template <bool Detached>
DfrCoherentUtdStatesT<Detached> Scene::build_dfr_coherent_tx_states(
    const DfrCoherentEdgeT<Detached> &edges,
    const Vector3fT<Detached> &tx_position,
    const DfrMaterialT<Detached> &material,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::build_dfr_coherent_tx_states(): scene is not built.");
    require(!pending_updates_,
            "Scene::build_dfr_coherent_tx_states(): scene has pending updates. Call Scene::sync() first.");
    require(options.wavelength > 0.f,
            "Scene::build_dfr_coherent_tx_states(): wavelength must be positive.");
    require(options.k > 0.f,
            "Scene::build_dfr_coherent_tx_states(): k must be positive.");
    if constexpr (!Detached) {
        (void)edges;
        (void)tx_position;
        (void)material;
        (void)active;
        throw std::runtime_error(
            "Scene::build_dfr_coherent_tx_states(): AD inputs are not supported yet.");
    } else {
        const int edge_count = edges.count;
        require(edge_count >= 0,
                "Scene::build_dfr_coherent_tx_states(): invalid edge count.");
        DfrCoherentUtdStates result;
        if (edge_count == 0) {
            result.count = 0;
            return result;
        }
        require(static_cast<int>(slices(edges.edge_index)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_pos)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_dir)) >= edge_count &&
                    static_cast<int>(slices(edges.n0)) >= edge_count &&
                    static_cast<int>(slices(edges.n_face_n)) >= edge_count &&
                    static_cast<int>(slices(edges.wedge_n)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_line_min)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_line_max)) >= edge_count &&
                    static_cast<int>(slices(edges.adjacent_face0)) >= edge_count &&
                    static_cast<int>(slices(edges.adjacent_face1)) >= edge_count,
                "Scene::build_dfr_coherent_tx_states(): edge fields must cover edge count.");
        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::build_dfr_coherent_tx_states(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::build_dfr_coherent_tx_states(): material payload fields must have matching widths.");
        const int ignore_count = static_cast<int>(slices(edges.ignore_prim_ids));
        int ignore_k = edges.ignore_k;
        if (ignore_count > 0) {
            require(ignore_k > 0,
                    "Scene::build_dfr_coherent_tx_states(): ignore_k must be positive when ignore_prim_ids is provided.");
            require(ignore_count == edge_count * ignore_k,
                    "Scene::build_dfr_coherent_tx_states(): ignore_prim_ids width must equal edge count * ignore_k.");
        } else {
            ignore_k = 0;
        }

        Vector3f source_pos = tx_position;
        if (static_cast<int>(slices(source_pos)) == 1 && edge_count > 1) {
            source_pos = gather<Vector3f>(source_pos, zeros<UInt>(edge_count));
        } else {
            require(static_cast<int>(slices(source_pos)) == edge_count,
                    "Scene::build_dfr_coherent_tx_states(): tx_position width must be 1 or match edge count.");
        }
        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && edge_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<UInt>(edge_count));
        } else {
            require(active_width == edge_count,
                    "Scene::build_dfr_coherent_tx_states(): active width must be 1 or match edge count.");
        }

        const Vector3f edge_dir = normalize_with_fallback(
            edges.edge_dir, Vector3f(Float(0.f), Float(0.f), Float(1.f)));
        require(trace_backend_ != nullptr && optix_scene().is_ready(),
                "Scene::build_dfr_coherent_tx_states(): OptiX scene is not ready.");
        ensure_pipeline(segment_visibility_pipeline_,
                        optix_scene().context(),
                        mesh_count_,
                        segment_visibility_pipeline_config());
        const SegmentVisibility visibility_result =
            trace_segment_visibility_native<true>(optix_scene(),
                                                  *segment_visibility_pipeline_,
                                                  face_offsets_,
                                                  mesh_count_,
                                                  source_pos,
                                                  edges.edge_pos,
                                                  edges.ignore_prim_ids,
                                                  ignore_k,
                                                  active_detached);
        const Mask visibility = visibility_result.visible;
        const Mask source_exterior =
            wedge_exterior_mask_jit(source_pos - edges.edge_pos, edge_dir, edges.n0, edges.n_face_n);
        const Mask finite_line =
            (edges.edge_line_max - edges.edge_line_min) > Float(1.0e-8f);
        const Mask valid = visibility && source_exterior && finite_line && active_detached;
        const UInt keep = compress(valid);
        const int state_count = static_cast<int>(slices(keep));
        result.count = state_count;
        if (state_count == 0) {
            return result;
        }

        result.edge_index = gather<Int>(edges.edge_index, keep);
        result.edge_pos = gather<Vector3f>(edges.edge_pos, keep);
        result.edge_dir = gather<Vector3f>(edge_dir, keep);
        result.n0 = gather<Vector3f>(edges.n0, keep);
        result.n_face_n = gather<Vector3f>(edges.n_face_n, keep);
        result.wedge_n = gather<Float>(edges.wedge_n, keep);
        result.edge_line_min = gather<Float>(edges.edge_line_min, keep);
        result.edge_line_max = gather<Float>(edges.edge_line_max, keep);
        result.source_pos = gather<Vector3f>(source_pos, keep);
        result.adjacent_face0 = gather<Int>(edges.adjacent_face0, keep);
        result.adjacent_face1 = gather<Int>(edges.adjacent_face1, keep);

        const Vector3f source_to_edge = result.edge_pos - result.source_pos;
        const Float distance = norm(source_to_edge) + Float(1.0e-12f);
        const Float source_gain = Float(1.f) / (Float(2.f) * Float(options.k) * distance);
        const drjit::Complex<Float> phase =
            exp(drjit::Complex<Float>(zeros<Float>(state_count), -Float(options.k) * distance));
        result.incident_field = phase * source_gain;
        result.incident_normal_derivative =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        const Vector3f ray_dir = source_to_edge / distance;
        const Vector3f tx_pol(Float(options.tx_pol_x), Float(options.tx_pol_y), Float(options.tx_pol_z));
        const Vector3f pol_dir = stable_perpendicular_basis_jit(ray_dir, tx_pol);
        result.incident_vector_x = result.incident_field * pol_dir.x();
        result.incident_vector_y = result.incident_field * pol_dir.y();
        result.incident_vector_z = result.incident_field * pol_dir.z();
        result.incident_normal_derivative_vector_x =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        result.incident_normal_derivative_vector_y =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        result.incident_normal_derivative_vector_z =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));

        result.incident_basis_k = ray_dir;
        result.incident_basis_u = stable_perpendicular_basis_jit(ray_dir, result.edge_dir);
        result.incident_basis_v = normalize_with_fallback(
            cross(ray_dir, result.incident_basis_u),
            stable_perpendicular_basis_jit(ray_dir, Vector3f(Float(0.f), Float(1.f), Float(0.f))));
        result.incident_jones_u = result.incident_vector_x * result.incident_basis_u.x() +
                                  result.incident_vector_y * result.incident_basis_u.y() +
                                  result.incident_vector_z * result.incident_basis_u.z();
        result.incident_jones_v = result.incident_vector_x * result.incident_basis_v.x() +
                                  result.incident_vector_y * result.incident_basis_v.y() +
                                  result.incident_vector_z * result.incident_basis_v.z();
        result.incident_derivative_jones_u =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        result.incident_derivative_jones_v =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));

        const Mask valid0 = result.adjacent_face0 >= Int(0) &&
                            result.adjacent_face0 < Int(material_count);
        const Mask valid1 = result.adjacent_face1 >= Int(0) &&
                            result.adjacent_face1 < Int(material_count);
        const Mask mat0 = gather_material_mask(material.valid, result.adjacent_face0, valid0);
        const Mask mat1 = gather_material_mask(material.valid, result.adjacent_face1, valid1);
        result.face0_eta_r = gather_material_float(material.eta_r, result.adjacent_face0, mat0, 1.f);
        result.face0_mu_r = gather_material_float(material.mu_r, result.adjacent_face0, mat0, 1.f);
        result.face0_sigma = gather_material_float(material.sigma, result.adjacent_face0, mat0, 0.f);
        result.face0_gain = gather_material_float(material.gain, result.adjacent_face0, mat0, 1.f);
        result.face0_use_fresnel = Float(mat0);
        result.face1_eta_r = gather_material_float(material.eta_r, result.adjacent_face1, mat1, 1.f);
        result.face1_mu_r = gather_material_float(material.mu_r, result.adjacent_face1, mat1, 1.f);
        result.face1_sigma = gather_material_float(material.sigma, result.adjacent_face1, mat1, 0.f);
        result.face1_gain = gather_material_float(material.gain, result.adjacent_face1, mat1, 1.f);
        result.face1_use_fresnel = Float(mat1);

        const drjit::Complex<Float> zero_c(zeros<Float>(state_count), zeros<Float>(state_count));
        const drjit::Complex<Float> pec_c(full<Float>(-1.f, state_count), zeros<Float>(state_count));
        result.r_face0 = pec_c;
        result.r_face_n = pec_c;
        result.face0_operator_m00 = pec_c;
        result.face0_operator_m01 = zero_c;
        result.face0_operator_m10 = zero_c;
        result.face0_operator_m11 = pec_c;
        result.face1_operator_m00 = pec_c;
        result.face1_operator_m01 = zero_c;
        result.face1_operator_m10 = zero_c;
        result.face1_operator_m11 = pec_c;

        result.select_stationary_point = full<Float>(1.f, state_count);
        result.owner_code = zeros<Int>(state_count);
        result.path_length_prefix = distance;
        result.first_interaction_pos = result.edge_pos;
        result.source_type_code = zeros<Int>(state_count);
        result.prefix_reflection_depth = zeros<Int>(state_count);
        result.intermediate_reflection_depth = zeros<Int>(state_count);
        result.suffix_reflection_depth = zeros<Int>(state_count);
        result.approximation_mode_code = zeros<Int>(state_count);
        result.order = full<Int>(1, state_count);
        return result;
    }
}

template <bool Detached>
DfrCoherentCandidatePairsT<Detached> Scene::build_dfr_coherent_higher_candidates(
    const DfrCoherentUtdStatesT<Detached> &prev_states,
    const DfrCoherentEdgeT<Detached> &edges,
    const IntT<Detached> &global_to_local_edge_index,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::build_dfr_coherent_higher_candidates(): scene is not built.");
    require(!pending_updates_,
            "Scene::build_dfr_coherent_higher_candidates(): scene has pending updates. Call Scene::sync() first.");
    require(options.higher_probe_radius_scale > 0.f,
            "Scene::build_dfr_coherent_higher_candidates(): probe radius scale must be positive.");
    require(options.higher_probe_radius_min >= 0.f &&
                options.higher_probe_radius_min <= options.higher_probe_radius_max,
            "Scene::build_dfr_coherent_higher_candidates(): probe radius bounds must be ordered.");
    if constexpr (!Detached) {
        (void)prev_states;
        (void)edges;
        (void)global_to_local_edge_index;
        (void)active;
        throw std::runtime_error(
            "Scene::build_dfr_coherent_higher_candidates(): AD inputs are not supported yet.");
    } else {
        const int prev_count = prev_states.count;
        const int edge_count = edges.count;
        require(prev_count >= 0 && edge_count >= 0,
                "Scene::build_dfr_coherent_higher_candidates(): invalid state or edge count.");
        DfrCoherentCandidatePairs result;
        if (prev_count == 0 || edge_count == 0) {
            result.count = 0;
            return result;
        }
        require(static_cast<int>(slices(prev_states.edge_index)) >= prev_count &&
                    static_cast<int>(slices(prev_states.edge_pos)) >= prev_count &&
                    static_cast<int>(slices(prev_states.source_pos)) >= prev_count &&
                    static_cast<int>(slices(prev_states.incident_basis_u)) >= prev_count &&
                    static_cast<int>(slices(prev_states.incident_basis_v)) >= prev_count &&
                    static_cast<int>(slices(prev_states.incident_basis_k)) >= prev_count,
                "Scene::build_dfr_coherent_higher_candidates(): previous state fields must cover state count.");
        require(static_cast<int>(slices(global_to_local_edge_index)) > 0,
                "Scene::build_dfr_coherent_higher_candidates(): global-to-local edge index map must not be empty.");

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && prev_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<UInt>(prev_count));
        } else {
            require(active_width == prev_count,
                    "Scene::build_dfr_coherent_higher_candidates(): active width must be 1 or match previous state count.");
        }

        constexpr int probe_count = 18;
        constexpr int probe_grid_count = probe_count / 2;
        const int probe_lane_count = prev_count * probe_count;
        const UInt probe_idx = arange<UInt>(probe_lane_count);
        const UInt prev_idx_all = probe_idx / UInt(probe_count);
        const UInt probe_slot = probe_idx - prev_idx_all * UInt(probe_count);
        const UInt probe_grid_slot = probe_slot % UInt(probe_grid_count);
        const Float probe_u = Float(probe_grid_slot / UInt(3)) - Float(1.f);
        const Float probe_v = Float(probe_grid_slot % UInt(3)) - Float(1.f);
        const Float probe_sign = select(probe_slot < UInt(probe_grid_count), Float(1.f), Float(-1.f));

        const Vector3f edge_pos = gather<Vector3f>(prev_states.edge_pos, prev_idx_all);
        const Vector3f source_pos = gather<Vector3f>(prev_states.source_pos, prev_idx_all);
        const Vector3f basis_u = gather<Vector3f>(prev_states.incident_basis_u, prev_idx_all);
        const Vector3f basis_v = gather<Vector3f>(prev_states.incident_basis_v, prev_idx_all);
        const Vector3f basis_k = gather<Vector3f>(prev_states.incident_basis_k, prev_idx_all);
        const Int prev_edge_idx = gather<Int>(prev_states.edge_index, prev_idx_all);
        const Mask probe_active = gather<Mask>(active_detached, prev_idx_all);

        const Float source_distance = norm(edge_pos - source_pos);
        const Float unclamped_radius = source_distance * Float(options.higher_probe_radius_scale);
        const Float probe_radius =
            minimum(maximum(unclamped_radius, Float(options.higher_probe_radius_min)),
                    Float(options.higher_probe_radius_max));
        const Vector3f ray_origin = edge_pos +
                                    basis_u * (probe_radius * probe_u) +
                                    basis_v * (probe_radius * probe_v);
        const Vector3f ray_dir = basis_k * probe_sign;

        const NearestRayEdge nearest =
            this->template nearest_edge<true>(Ray(ray_origin, ray_dir), probe_active);
        Mask valid = nearest.global_edge_id >= Int(0);
        const Int safe_global_edge =
            select(valid, nearest.global_edge_id, Int(0));
        valid &= safe_global_edge < Int(static_cast<int>(slices(global_to_local_edge_index)));
        const Int local_edge_idx =
            gather<Int>(global_to_local_edge_index, UInt(safe_global_edge), valid);
        valid &= local_edge_idx >= Int(0);
        valid &= prev_edge_idx != local_edge_idx;
        valid &= probe_active;

        if (options.higher_filter_visibility) {
            require(static_cast<int>(slices(edges.edge_pos)) >= edge_count &&
                        static_cast<int>(slices(edges.adjacent_face0)) >= edge_count &&
                        static_cast<int>(slices(edges.adjacent_face1)) >= edge_count &&
                        static_cast<int>(slices(prev_states.adjacent_face0)) >= prev_count &&
                        static_cast<int>(slices(prev_states.adjacent_face1)) >= prev_count,
                    "Scene::build_dfr_coherent_higher_candidates(): visibility filtering requires edge positions and adjacent faces.");
            require(trace_backend_ != nullptr && optix_scene().is_ready(),
                    "Scene::build_dfr_coherent_higher_candidates(): OptiX scene is not ready.");
            const UInt safe_local_edge = UInt(select(valid, local_edge_idx, Int(0)));
            const Vector3f next_edge_pos = gather<Vector3f>(edges.edge_pos, safe_local_edge, valid);
            const Int prev_adjacent_face0 = gather<Int>(prev_states.adjacent_face0, prev_idx_all, valid);
            const Int prev_adjacent_face1 = gather<Int>(prev_states.adjacent_face1, prev_idx_all, valid);
            const Int next_adjacent_face0 = gather<Int>(edges.adjacent_face0, safe_local_edge, valid);
            const Int next_adjacent_face1 = gather<Int>(edges.adjacent_face1, safe_local_edge, valid);
            const Int ignore_prim_ids = interleave_four_ignore_slots(
                prev_adjacent_face0,
                prev_adjacent_face1,
                next_adjacent_face0,
                next_adjacent_face1,
                probe_lane_count);
            ensure_pipeline(segment_visibility_pipeline_,
                            optix_scene().context(),
                            mesh_count_,
                            segment_visibility_pipeline_config());
            const SegmentVisibility visibility_result =
                trace_segment_visibility_native<true>(optix_scene(),
                                                      *segment_visibility_pipeline_,
                                                      face_offsets_,
                                                      mesh_count_,
                                                      edge_pos,
                                                      next_edge_pos,
                                                      ignore_prim_ids,
                                                      4,
                                                      valid);
            valid &= visibility_result.visible;
            result.visibility_filtered = 1;
        }

        const UInt keep = compress(valid);
        const int candidate_count = static_cast<int>(slices(keep));
        result.count = candidate_count;
        if (candidate_count == 0) {
            return result;
        }
        result.prev_index = Int(gather<UInt>(prev_idx_all, keep));
        result.edge_index = gather<Int>(local_edge_idx, keep);
        return result;
    }
}

template <bool Detached>
DfrCoherentAccumT<Detached> Scene::accum_dfr_coherent_direct(
    const DfrCoherentUtdStatesT<Detached> &states,
    const DfrGrid &grid,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr_coherent_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr_coherent_direct(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr_coherent_direct(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr_coherent_direct(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr_coherent_direct(): grid bounds must be ordered.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr_coherent_direct(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::accum_dfr_coherent_direct(): only max_order == 1 is supported.");
    require(options.receiver_model == RAYD_DFR_MATCHED_ISO,
            "Scene::accum_dfr_coherent_direct(): only matched isotropic receivers are supported.");
    if constexpr (!Detached) {
        (void)states;
        (void)active;
        throw std::runtime_error(
            "Scene::accum_dfr_coherent_direct(): AD inputs are not supported yet.");
    } else {
        const int state_count = states.count;
        require(state_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n_face_n)) >= state_count &&
                    static_cast<int>(slices(states.source_pos)) >= state_count &&
                    static_cast<int>(slices(states.incident_basis_u)) >= state_count &&
                    static_cast<int>(slices(states.incident_basis_v)) >= state_count &&
                    static_cast<int>(slices(states.incident_basis_k)) >= state_count &&
                    static_cast<int>(slices(states.wedge_n)) >= state_count &&
                    static_cast<int>(slices(states.edge_line_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_line_max)) >= state_count &&
                    static_cast<int>(slices(states.face0_eta_r)) >= state_count &&
                    static_cast<int>(slices(states.face0_mu_r)) >= state_count &&
                    static_cast<int>(slices(states.face0_sigma)) >= state_count &&
                    static_cast<int>(slices(states.face0_gain)) >= state_count &&
                    static_cast<int>(slices(states.face0_use_fresnel)) >= state_count &&
                    static_cast<int>(slices(states.face1_eta_r)) >= state_count &&
                    static_cast<int>(slices(states.face1_mu_r)) >= state_count &&
                    static_cast<int>(slices(states.face1_sigma)) >= state_count &&
                    static_cast<int>(slices(states.face1_gain)) >= state_count &&
                    static_cast<int>(slices(states.face1_use_fresnel)) >= state_count &&
                    static_cast<int>(slices(states.select_stationary_point)) >= state_count &&
                    static_cast<int>(slices(states.incident_field)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative)) >= state_count &&
                    static_cast<int>(slices(states.r_face0)) >= state_count &&
                    static_cast<int>(slices(states.r_face_n)) >= state_count &&
                    static_cast<int>(slices(states.incident_vector_x)) >= state_count &&
                    static_cast<int>(slices(states.incident_vector_y)) >= state_count &&
                    static_cast<int>(slices(states.incident_vector_z)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative_vector_x)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative_vector_y)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative_vector_z)) >= state_count &&
                    static_cast<int>(slices(states.incident_jones_u)) >= state_count &&
                    static_cast<int>(slices(states.incident_jones_v)) >= state_count &&
                    static_cast<int>(slices(states.incident_derivative_jones_u)) >= state_count &&
                    static_cast<int>(slices(states.incident_derivative_jones_v)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m00)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m01)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m10)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m11)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m00)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m01)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m10)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m11)) >= state_count &&
                    static_cast<int>(slices(states.owner_code)) >= state_count &&
                    static_cast<int>(slices(states.adjacent_face0)) >= state_count &&
                    static_cast<int>(slices(states.adjacent_face1)) >= state_count,
                "Scene::accum_dfr_coherent_direct(): full UTD state fields must cover state count.");
        const int grid_cell_count = grid.resolution0 * grid.resolution1;
        const int launch_count = state_count * grid_cell_count;
        require(launch_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid launch count.");

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_coherent_direct(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.source_pos.x()) &&
                           drjit::isfinite(states.source_pos.y()) &&
                           drjit::isfinite(states.source_pos.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z());

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_coherent_direct(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr_coherent_direct(): invalid hitgroup record count.");

            auto &dfr_pipeline = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_
                : diffraction_coherent_accumulation_pipeline_;
            const OptixPipelineConfig dfr_pipeline_config = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_config()
                : diffraction_coherent_accumulation_pipeline_config();

            ensure_pipeline(dfr_pipeline,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            dfr_pipeline_config);
        }

        drjit::eval(states.edge_pos,
                    states.edge_dir,
                    states.n0,
                    states.n_face_n,
                    states.source_pos,
                    states.incident_basis_u,
                    states.incident_basis_v,
                    states.incident_basis_k,
                    states.wedge_n,
                    states.edge_line_min,
                    states.edge_line_max,
                    states.face0_eta_r,
                    states.face0_mu_r,
                    states.face0_sigma,
                    states.face0_gain,
                    states.face0_use_fresnel,
                    states.face1_eta_r,
                    states.face1_mu_r,
                    states.face1_sigma,
                    states.face1_gain,
                    states.face1_use_fresnel,
                    states.select_stationary_point,
                    states.incident_field,
                    states.incident_normal_derivative,
                    states.r_face0,
                    states.r_face_n,
                    states.incident_vector_x,
                    states.incident_vector_y,
                    states.incident_vector_z,
                    states.incident_normal_derivative_vector_x,
                    states.incident_normal_derivative_vector_y,
                    states.incident_normal_derivative_vector_z,
                    states.incident_jones_u,
                    states.incident_jones_v,
                    states.incident_derivative_jones_u,
                    states.incident_derivative_jones_v,
                    states.face0_operator_m00,
                    states.face0_operator_m01,
                    states.face0_operator_m10,
                    states.face0_operator_m11,
                    states.face1_operator_m00,
                    states.face1_operator_m01,
                    states.face1_operator_m10,
                    states.face1_operator_m11,
                    states.owner_code,
                    states.adjacent_face0,
                    states.adjacent_face1,
                    active_detached,
                    triangle_info_detached_.face_normal,
                    face_offsets_);

        DfrCoherentAccum result;
        result.grid_cell_count = grid_cell_count;
        DfrCoherentAccumRaw raw = alloc_dfr_coherent_accum_raw(grid_cell_count);
        init_dfr_coherent_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = state_count;
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.max_order = options.max_order;
        params.receiver_model = options.receiver_model;
        params.select_diffraction_point = options.select_diffraction_point ? 1 : 0;
        params.prefilter_visibility = options.prefilter_visibility ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.omega = options.omega;
        params.tx_pol_x = options.tx_pol_x;
        params.tx_pol_y = options.tx_pol_y;
        params.tx_pol_z = options.tx_pol_z;
        params.coherent_utd_slot_count = 84;
        params.utd_epx = states.edge_pos.x().data();
        params.utd_epy = states.edge_pos.y().data();
        params.utd_epz = states.edge_pos.z().data();
        params.utd_edx = states.edge_dir.x().data();
        params.utd_edy = states.edge_dir.y().data();
        params.utd_edz = states.edge_dir.z().data();
        params.utd_n0x = states.n0.x().data();
        params.utd_n0y = states.n0.y().data();
        params.utd_n0z = states.n0.z().data();
        params.utd_nnx = states.n_face_n.x().data();
        params.utd_nny = states.n_face_n.y().data();
        params.utd_nnz = states.n_face_n.z().data();
        params.utd_wn = states.wedge_n.data();
        params.utd_elm = states.edge_line_min.data();
        params.utd_elx = states.edge_line_max.data();
        params.utd_spx = states.source_pos.x().data();
        params.utd_spy = states.source_pos.y().data();
        params.utd_spz = states.source_pos.z().data();
        params.utd_ifr = drjit::real(states.incident_field).data();
        params.utd_ifi = drjit::imag(states.incident_field).data();
        params.utd_inr = drjit::real(states.incident_normal_derivative).data();
        params.utd_ini = drjit::imag(states.incident_normal_derivative).data();
        params.utd_r0r = drjit::real(states.r_face0).data();
        params.utd_r0i = drjit::imag(states.r_face0).data();
        params.utd_rnr = drjit::real(states.r_face_n).data();
        params.utd_rni = drjit::imag(states.r_face_n).data();
        params.utd_vxr = drjit::real(states.incident_vector_x).data();
        params.utd_vxi = drjit::imag(states.incident_vector_x).data();
        params.utd_vyr = drjit::real(states.incident_vector_y).data();
        params.utd_vyi = drjit::imag(states.incident_vector_y).data();
        params.utd_vzr = drjit::real(states.incident_vector_z).data();
        params.utd_vzi = drjit::imag(states.incident_vector_z).data();
        params.utd_dxr = drjit::real(states.incident_normal_derivative_vector_x).data();
        params.utd_dxi = drjit::imag(states.incident_normal_derivative_vector_x).data();
        params.utd_dyr = drjit::real(states.incident_normal_derivative_vector_y).data();
        params.utd_dyi = drjit::imag(states.incident_normal_derivative_vector_y).data();
        params.utd_dzr = drjit::real(states.incident_normal_derivative_vector_z).data();
        params.utd_dzi = drjit::imag(states.incident_normal_derivative_vector_z).data();
        params.utd_jur = drjit::real(states.incident_jones_u).data();
        params.utd_jui = drjit::imag(states.incident_jones_u).data();
        params.utd_jvr = drjit::real(states.incident_jones_v).data();
        params.utd_jvi = drjit::imag(states.incident_jones_v).data();
        params.utd_djur = drjit::real(states.incident_derivative_jones_u).data();
        params.utd_djui = drjit::imag(states.incident_derivative_jones_u).data();
        params.utd_djvr = drjit::real(states.incident_derivative_jones_v).data();
        params.utd_djvi = drjit::imag(states.incident_derivative_jones_v).data();
        params.utd_bux = states.incident_basis_u.x().data();
        params.utd_buy = states.incident_basis_u.y().data();
        params.utd_buz = states.incident_basis_u.z().data();
        params.utd_bvx = states.incident_basis_v.x().data();
        params.utd_bvy = states.incident_basis_v.y().data();
        params.utd_bvz = states.incident_basis_v.z().data();
        params.utd_bkx = states.incident_basis_k.x().data();
        params.utd_bky = states.incident_basis_k.y().data();
        params.utd_bkz = states.incident_basis_k.z().data();
        params.utd_f0m00r = drjit::real(states.face0_operator_m00).data();
        params.utd_f0m00i = drjit::imag(states.face0_operator_m00).data();
        params.utd_f0m01r = drjit::real(states.face0_operator_m01).data();
        params.utd_f0m01i = drjit::imag(states.face0_operator_m01).data();
        params.utd_f0m10r = drjit::real(states.face0_operator_m10).data();
        params.utd_f0m10i = drjit::imag(states.face0_operator_m10).data();
        params.utd_f0m11r = drjit::real(states.face0_operator_m11).data();
        params.utd_f0m11i = drjit::imag(states.face0_operator_m11).data();
        params.utd_f1m00r = drjit::real(states.face1_operator_m00).data();
        params.utd_f1m00i = drjit::imag(states.face1_operator_m00).data();
        params.utd_f1m01r = drjit::real(states.face1_operator_m01).data();
        params.utd_f1m01i = drjit::imag(states.face1_operator_m01).data();
        params.utd_f1m10r = drjit::real(states.face1_operator_m10).data();
        params.utd_f1m10i = drjit::imag(states.face1_operator_m10).data();
        params.utd_f1m11r = drjit::real(states.face1_operator_m11).data();
        params.utd_f1m11i = drjit::imag(states.face1_operator_m11).data();
        params.utd_f0er = states.face0_eta_r.data();
        params.utd_f0mu = states.face0_mu_r.data();
        params.utd_f0sg = states.face0_sigma.data();
        params.utd_f0g = states.face0_gain.data();
        params.utd_f0uf = states.face0_use_fresnel.data();
        params.utd_f1er = states.face1_eta_r.data();
        params.utd_f1mu = states.face1_mu_r.data();
        params.utd_f1sg = states.face1_sigma.data();
        params.utd_f1g = states.face1_gain.data();
        params.utd_f1uf = states.face1_use_fresnel.data();
        params.utd_select = states.select_stationary_point.data();
        params.coherent_owner_code = states.owner_code.data();
        params.coherent_adjacent_face0 = states.adjacent_face0.data();
        params.coherent_adjacent_face1 = states.adjacent_face1.data();
        params.out_direct_count = raw.direct_count.data();
        params.out_direct_field_x_re = raw.direct_field_x_re.data();
        params.out_direct_field_x_im = raw.direct_field_x_im.data();
        params.out_direct_field_y_re = raw.direct_field_y_re.data();
        params.out_direct_field_y_im = raw.direct_field_y_im.data();
        params.out_direct_field_z_re = raw.direct_field_z_re.data();
        params.out_direct_field_z_im = raw.direct_field_z_im.data();
        params.out_multi_field_x_re = raw.multi_field_x_re.data();
        params.out_multi_field_x_im = raw.multi_field_x_im.data();
        params.out_multi_field_y_re = raw.multi_field_y_re.data();
        params.out_multi_field_y_im = raw.multi_field_y_im.data();
        params.out_multi_field_z_re = raw.multi_field_z_re.data();
        params.out_multi_field_z_im = raw.multi_field_z_im.data();
        params.out_multi_count = raw.multi_count.data();
        params.out_visibility_reject_count = raw.visibility_reject_count.data();
        params.out_utd_reject_count = raw.utd_reject_count.data();

        if (cuda_trace) {
            cuda_backend().run_dfr_accum_coherent(params, launch_count);
        } else {
            (split_mode == 0 ? diffraction_coherent_accumulation_primary_pipeline_
                             : diffraction_coherent_accumulation_pipeline_)
                ->launch(0, params);
        }

        result.direct_field_x = drjit::Complex<Float>(raw.direct_field_x_re, raw.direct_field_x_im);
        result.direct_field_y = drjit::Complex<Float>(raw.direct_field_y_re, raw.direct_field_y_im);
        result.direct_field_z = drjit::Complex<Float>(raw.direct_field_z_re, raw.direct_field_z_im);
        result.multi_field_x = drjit::Complex<Float>(raw.multi_field_x_re, raw.multi_field_x_im);
        result.multi_field_y = drjit::Complex<Float>(raw.multi_field_y_re, raw.multi_field_y_im);
        result.multi_field_z = drjit::Complex<Float>(raw.multi_field_z_re, raw.multi_field_z_im);
        result.direct_count = raw.direct_count;
        result.multi_count = raw.multi_count;
        result.visibility_reject_count = raw.visibility_reject_count;
        result.utd_reject_count = raw.utd_reject_count;
        return result;
    }
}

template <bool Detached>
DfrCoherentAccumT<Detached> Scene::accum_dfr_coherent_direct(
    const DfrStatesT<Detached> &states,
    const DfrGrid &grid,
    const DfrMaterialT<Detached> &material,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr_coherent_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr_coherent_direct(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr_coherent_direct(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr_coherent_direct(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr_coherent_direct(): grid bounds must be ordered.");
    require(grid.cell_area > 0.f,
            "Scene::accum_dfr_coherent_direct(): grid.cell_area must be positive.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr_coherent_direct(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::accum_dfr_coherent_direct(): only max_order == 1 is supported.");
    require(options.receiver_model == RAYD_DFR_MATCHED_ISO,
            "Scene::accum_dfr_coherent_direct(): only matched isotropic receivers are supported.");
    if constexpr (!Detached) {
        (void)states;
        (void)material;
        (void)active;
        throw std::runtime_error(
            "Scene::accum_dfr_coherent_direct(): AD inputs are not supported yet.");
    } else {
        const int state_count = states.count;
        require(state_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_index)) >= state_count &&
                    static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count &&
                    static_cast<int>(slices(states.wi)) >= state_count &&
                    static_cast<int>(slices(states.d0)) >= state_count &&
                    static_cast<int>(slices(states.prefix_depth)) >= state_count,
                "Scene::accum_dfr_coherent_direct(): state fields must cover state count.");
        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr_coherent_direct(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr_coherent_direct(): material payload fields must have matching widths.");

        DfrCoherentAccum result;
        const int grid_cell_count = grid.resolution0 * grid.resolution1;
        const int launch_count = state_count * grid_cell_count;
        result.grid_cell_count = grid_cell_count;
        require(launch_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid launch count.");

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_coherent_direct(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.src.x()) &&
                           drjit::isfinite(states.src.y()) &&
                           drjit::isfinite(states.src.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z()) &&
                           drjit::isfinite(states.src_power);

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_coherent_direct(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr_coherent_direct(): invalid hitgroup record count.");

            auto &dfr_pipeline = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_
                : diffraction_coherent_accumulation_pipeline_;
            const OptixPipelineConfig dfr_pipeline_config = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_config()
                : diffraction_coherent_accumulation_pipeline_config();

            ensure_pipeline(dfr_pipeline,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            dfr_pipeline_config);
        }

        drjit::eval(states.edge_index,
                    states.edge_pos,
                    states.edge_dir,
                    states.edge_t_min,
                    states.edge_t_max,
                    states.n0,
                    states.n1,
                    states.prim0,
                    states.prim1,
                    states.exterior_angle,
                    states.src,
                    states.src_power,
                    states.wi,
                    states.d0,
                    states.prefix_depth,
                    active_detached,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);

        DfrCoherentAccumRaw raw =
            alloc_dfr_coherent_accum_raw(grid_cell_count);
        init_dfr_coherent_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = state_count;
        params.state_edge_index = states.edge_index.data();
        params.state_edge_pos_x = states.edge_pos.x().data();
        params.state_edge_pos_y = states.edge_pos.y().data();
        params.state_edge_pos_z = states.edge_pos.z().data();
        params.state_edge_dir_x = states.edge_dir.x().data();
        params.state_edge_dir_y = states.edge_dir.y().data();
        params.state_edge_dir_z = states.edge_dir.z().data();
        params.state_edge_t_min = states.edge_t_min.data();
        params.state_edge_t_max = states.edge_t_max.data();
        params.state_n0_x = states.n0.x().data();
        params.state_n0_y = states.n0.y().data();
        params.state_n0_z = states.n0.z().data();
        params.state_n1_x = states.n1.x().data();
        params.state_n1_y = states.n1.y().data();
        params.state_n1_z = states.n1.z().data();
        params.state_prim0 = states.prim0.data();
        params.state_prim1 = states.prim1.data();
        params.state_exterior_angle = states.exterior_angle.data();
        params.state_src_x = states.src.x().data();
        params.state_src_y = states.src.y().data();
        params.state_src_z = states.src.z().data();
        params.state_src_power = states.src_power.data();
        params.state_wi_x = states.wi.x().data();
        params.state_wi_y = states.wi.y().data();
        params.state_wi_z = states.wi.z().data();
        params.state_d0_x = states.d0.x().data();
        params.state_d0_y = states.d0.y().data();
        params.state_d0_z = states.d0.z().data();
        params.state_prefix_depth = states.prefix_depth.data();
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.max_order = options.max_order;
        params.receiver_model = options.receiver_model;
        params.select_diffraction_point = options.select_diffraction_point ? 1 : 0;
        params.prefilter_visibility = options.prefilter_visibility ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.out_direct_count = raw.direct_count.data();
        params.out_direct_field_x_re = raw.direct_field_x_re.data();
        params.out_direct_field_x_im = raw.direct_field_x_im.data();
        params.out_direct_field_y_re = raw.direct_field_y_re.data();
        params.out_direct_field_y_im = raw.direct_field_y_im.data();
        params.out_direct_field_z_re = raw.direct_field_z_re.data();
        params.out_direct_field_z_im = raw.direct_field_z_im.data();
        params.out_multi_field_x_re = raw.multi_field_x_re.data();
        params.out_multi_field_x_im = raw.multi_field_x_im.data();
        params.out_multi_field_y_re = raw.multi_field_y_re.data();
        params.out_multi_field_y_im = raw.multi_field_y_im.data();
        params.out_multi_field_z_re = raw.multi_field_z_re.data();
        params.out_multi_field_z_im = raw.multi_field_z_im.data();
        params.out_multi_count = raw.multi_count.data();
        params.out_visibility_reject_count =
            raw.visibility_reject_count.data();
        params.out_utd_reject_count = raw.utd_reject_count.data();

        if (cuda_trace) {
            cuda_backend().run_dfr_accum_coherent(params, launch_count);
        } else {
            (split_mode == 0 ? diffraction_coherent_accumulation_primary_pipeline_
                             : diffraction_coherent_accumulation_pipeline_)
                ->launch(0, params);
        }

        result.direct_field_x =
            drjit::Complex<Float>(raw.direct_field_x_re, raw.direct_field_x_im);
        result.direct_field_y =
            drjit::Complex<Float>(raw.direct_field_y_re, raw.direct_field_y_im);
        result.direct_field_z =
            drjit::Complex<Float>(raw.direct_field_z_re, raw.direct_field_z_im);
        result.multi_field_x =
            drjit::Complex<Float>(raw.multi_field_x_re, raw.multi_field_x_im);
        result.multi_field_y =
            drjit::Complex<Float>(raw.multi_field_y_re, raw.multi_field_y_im);
        result.multi_field_z =
            drjit::Complex<Float>(raw.multi_field_z_re, raw.multi_field_z_im);
        result.direct_count = raw.direct_count;
        result.multi_count = raw.multi_count;
        result.visibility_reject_count = raw.visibility_reject_count;
        result.utd_reject_count = raw.utd_reject_count;
        return result;
    }
}

template DfrCoherentAccum Scene::accum_dfr_coherent_direct<true>(
    const DfrCoherentUtdStates &states,
    const DfrGrid &grid,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentAccumAD Scene::accum_dfr_coherent_direct<false>(
    const DfrCoherentUtdStatesAD &states,
    const DfrGrid &grid,
    const DfrCoherentOptions &options,
    MaskAD active) const;
template DfrCoherentUtdStates Scene::build_dfr_coherent_tx_states<true>(
    const DfrCoherentEdge &edges,
    const Vector3f &tx_position,
    const DfrMaterial &material,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentUtdStatesAD Scene::build_dfr_coherent_tx_states<false>(
    const DfrCoherentEdgeAD &edges,
    const Vector3fAD &tx_position,
    const DfrMaterialAD &material,
    const DfrCoherentOptions &options,
    MaskAD active) const;
template DfrCoherentCandidatePairs Scene::build_dfr_coherent_higher_candidates<true>(
    const DfrCoherentUtdStates &prev_states,
    const DfrCoherentEdge &edges,
    const Int &global_to_local_edge_index,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentCandidatePairsAD Scene::build_dfr_coherent_higher_candidates<false>(
    const DfrCoherentUtdStatesAD &prev_states,
    const DfrCoherentEdgeAD &edges,
    const IntAD &global_to_local_edge_index,
    const DfrCoherentOptions &options,
    MaskAD active) const;
template DfrCoherentAccum Scene::accum_dfr_coherent_direct<true>(
    const DfrStates &states,
    const DfrGrid &grid,
    const DfrMaterial &material,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentAccumAD Scene::accum_dfr_coherent_direct<false>(
    const DfrStatesAD &states,
    const DfrGrid &grid,
    const DfrMaterialAD &material,
    const DfrCoherentOptions &options,
    MaskAD active) const;

} // namespace rayd
