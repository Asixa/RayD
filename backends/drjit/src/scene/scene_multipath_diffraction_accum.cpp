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
