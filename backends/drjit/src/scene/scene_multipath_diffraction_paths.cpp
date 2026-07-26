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
