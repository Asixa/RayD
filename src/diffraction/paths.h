// Copyright Xingyu Chen.
// Defines shared diffraction path algorithms.

#pragma once

#include <cmath>
#include <cstdint>

#include <rayd/math.h>
#include <src/diffraction/contracts.h>
#include <src/runtime/rt_device.cuh>
#include <rayd/utd.h>

// Host-compilable first-order diffraction compact-path-export algorithm. This is
// the de-CUDA-ised body of the former (Dr.Jit) diffraction_paths raygen family:
// math is math::Vec3f throughout (mirroring the exact arithmetic op order of the
// old local CUDA vector helpers so device codegen stays bit-identical), every
// visibility ray cast goes through an rt::is_traverser Traverser (so no OptiX
// ray-cast intrinsic, payload register, or launch-index query appears here), and
// the lane index is a plain parameter. The local 4-field HitPayload dissolves
// into rt::TriangleHit (the export never reads the barycentrics, so they stay
// unread). diffraction_paths_device.cuh instantiates it with the pipeline's own
// four-register OptixTraverser; the CUDA fused executor (P4d) will reuse it with
// CudaBvhTraverser. The strategy/receiver-model guards read the backend-neutral
// diffraction_contracts.h enums instead of the per-backend macros.
//
// UTD math (host-safe utd.h) speaks its own `utd::float3a` POD at the pair
// boundary; the algorithm carries geometry in math::Vec3f and converts at the
// call sites. The compact-path append is one device atomic (host non-atomic
// fallback below only exists so this header parses under a pure host compiler).

namespace rayd::shared::multipath {

namespace diffraction_paths_algo_detail {

inline constexpr int kOutputLayoutCompact = 0;
inline constexpr int kOutputLayoutSourceLane = 1;

using math::Vec3f;
namespace utd = ::rayd::shared::diffraction;

inline constexpr float kTraceTMin = 1e-5f;
inline constexpr float kRayBias = 1e-4f;
inline constexpr float kSmallEps = 1e-6f;
inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kSpeedOfLight = 299792458.f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);

inline constexpr int kStrategyDirect = static_cast<int>(::rayd::shared::optix::DiffractionStrategyBit::Direct);
inline constexpr int kReceiverMatchedIso =
    static_cast<int>(::rayd::shared::optix::DiffractionReceiverModel::MatchedIsotropic);

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

// atomicAdd on device; a non-atomic byte-equivalent on the host so the compact
// path append compiles off-device (the host path is never executed).
RAYD_HOST_DEVICE int atomic_add(int* address, int value) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    const int old = *address;
    *address += value;
    return old;
#endif
}

RAYD_HOST_DEVICE utd::float3a to_utd(Vec3f value) {
    return utd::make_f3(value.x, value.y, value.z);
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_hit(::rayd::shared::rt::TriangleHit a,
                                                            ::rayd::shared::rt::TriangleHit b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return a.t <= b.t ? a : b;
}

template <bool SplitScene, typename Traverser>
RAYD_DEVICE ::rayd::shared::rt::TriangleHit trace_scene(const Traverser& primary, const Traverser& secondary,
                                                        Vec3f origin, Vec3f direction, float tmax) {
    const ::rayd::shared::rt::TriangleHit p = primary.trace_closest(origin, direction, kTraceTMin, tmax);
    if (!SplitScene)
        return p;
    const ::rayd::shared::rt::TriangleHit s = secondary.trace_closest(origin, direction, kTraceTMin, tmax);
    return choose_hit(p, s);
}

template <bool SplitScene, typename Traverser>
RAYD_DEVICE bool visible_segment(const Traverser& primary, const Traverser& secondary, Vec3f start, Vec3f end) {
    const Vec3f delta = math::subtract(end, start);
    const float dist = math::length_f32(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const Vec3f dir = math::scale(delta, 1.f / dist);
    const ::rayd::shared::rt::TriangleHit hit =
        trace_scene<SplitScene>(primary, secondary, math::add(start, math::scale(dir, kRayBias)), dir,
                                fmaxf(dist - 2.f * kRayBias, 0.f));
    return hit.hit == 0u;
}

RAYD_HOST_DEVICE Vec3f state_vec(const float* x, const float* y, const float* z, int idx) {
    return math::make_vec3(x[idx], y[idx], z[idx]);
}

template <typename Params> RAYD_HOST_DEVICE bool state_active(const Params& params, int state_idx) {
    return params.active_mask[state_idx] != 0u;
}

template <typename Params>
RAYD_HOST_DEVICE utd::FaceMaterialParams face_material_params(const Params& params, int prim) {
    utd::FaceMaterialParams m;
    m.etaR = 1.f;
    m.muR = 1.f;
    m.sigma = 0.f;
    m.gain = 1.f;
    m.useFresnel = 1.f;
    m.present = 0.f;
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        return m;
    }
    m.present = 1.f;
    m.etaR = params.material_eta_r[prim];
    m.sigma = params.material_sigma[prim];
    m.muR = params.material_mu_r[prim];
    m.gain = fmaxf(params.material_gain[prim], 0.f);
    return m;
}

template <typename Params>
RAYD_HOST_DEVICE utd::PairInputs direct_pair_inputs(const Params& params, int state_idx, Vec3f source, Vec3f edge_pos,
                                                    Vec3f edge_dir, float t_min, float t_max) {
    utd::PairInputs p = {};
    p.edgePos = to_utd(edge_pos);
    p.edgeDir = to_utd(edge_dir);
    p.n0 = to_utd(state_vec(params.state_n0_x, params.state_n0_y, params.state_n0_z, state_idx));
    p.nn = to_utd(state_vec(params.state_n1_x, params.state_n1_y, params.state_n1_z, state_idx));
    p.wedgeN = params.state_exterior_angle[state_idx] / utd::UTD_PI;
    p.edgeLineMin = t_min;
    p.edgeLineMax = t_max;
    p.sourcePos = to_utd(source);
    p.selectStationaryPoint = 1.f;
    p.face0Material = face_material_params(params, params.state_prim0[state_idx]);
    p.face1Material = face_material_params(params, params.state_prim1[state_idx]);
    return p;
}

template <typename Params> RAYD_HOST_DEVICE utd::MaterialParams paths_material_params(const Params& params) {
    utd::MaterialParams mat;
    mat.useFresnel = 1;
    mat.etaR = 1.f;
    mat.muR = 1.f;
    mat.sigma = 0.f;
    mat.gain = 1.f;
    mat.omega = params.omega;
    mat.txPolX = 1.f;
    mat.txPolY = 0.f;
    mat.txPolZ = 0.f;
    return mat;
}

RAYD_HOST_DEVICE void write_point(float* x, float* y, float* z, int idx, Vec3f value) {
    if (x == nullptr || y == nullptr || z == nullptr) {
        return;
    }
    x[idx] = value.x;
    y[idx] = value.y;
    z[idx] = value.z;
}

template <typename Params>
RAYD_HOST_DEVICE bool paths_order1_lane(const Params& params, unsigned int lane, int& state_idx, int& rx_idx,
                                        int& tx_idx) {
    if (lane >= static_cast<unsigned int>(params.n_rays) || params.capacity <= 0 || params.tx_count <= 0 ||
        params.rx_count <= 0 || params.state_count <= 0 || params.state_limit <= 0 || params.max_order != 1 ||
        (params.strategy_mask & kStrategyDirect) == 0 || params.receiver_model != kReceiverMatchedIso) {
        return false;
    }

    const int state_limit = params.state_limit;
    const int rx_count = params.rx_count;
    state_idx = static_cast<int>(lane % static_cast<unsigned int>(state_limit));
    const int pair_idx = static_cast<int>(lane / static_cast<unsigned int>(state_limit));
    rx_idx = pair_idx % rx_count;
    tx_idx = pair_idx / rx_count;
    return tx_idx < params.tx_count && state_idx < params.state_count && state_active(params, state_idx);
}

template <typename Params> RAYD_HOST_DEVICE Vec3f paths_edge_point(const Params& params, int state_idx, int rx_idx) {
    const Vec3f edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const Vec3f edge_dir = math::normalize_f32(
        state_vec(params.state_edge_dir_x, params.state_edge_dir_y, params.state_edge_dir_z, state_idx));
    const float t_min = params.state_edge_t_min[state_idx];
    const float t_max = params.state_edge_t_max[state_idx];
    const float edge_length = t_max - t_min;
    const Vec3f edge_origin = math::add(edge_pos, math::scale(edge_dir, t_min));
    const Vec3f source = state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const Vec3f receiver = math::make_vec3(params.rx_pos_x[rx_idx], params.rx_pos_y[rx_idx], params.rx_pos_z[rx_idx]);
    const float parameter =
        utd::first_order_diffraction_parameter(to_utd(source), to_utd(receiver), to_utd(edge_origin), to_utd(edge_dir));
    if (!is_finite(parameter) || !(edge_length > kSmallEps)) {
        return math::make_vec3(NAN, NAN, NAN);
    }
    return math::add(edge_origin, math::scale(edge_dir, fminf(fmaxf(parameter, 0.f), edge_length)));
}

RAYD_HOST_DEVICE bool finite_paths_points(Vec3f source, Vec3f edge_point, Vec3f receiver) {
    return is_finite(source.x) && is_finite(source.y) && is_finite(source.z) && is_finite(edge_point.x) &&
           is_finite(edge_point.y) && is_finite(edge_point.z) && is_finite(receiver.x) && is_finite(receiver.y) &&
           is_finite(receiver.z);
}

template <typename Params>
RAYD_HOST_DEVICE auto path_output_layout(const Params& params, int) -> decltype(params.output_layout) {
    return params.output_layout;
}

template <typename Params> RAYD_HOST_DEVICE int path_output_layout(const Params&, long) {
    return kOutputLayoutCompact;
}

template <typename Params> RAYD_DEVICE int reserve_path_output(const Params& params, std::uint32_t lane) {
    if (path_output_layout(params, 0) == kOutputLayoutSourceLane) {
        atomic_add(params.out_count, 1);
        return static_cast<int>(lane);
    }
    return atomic_add(params.out_count, 1);
}

} // namespace diffraction_paths_algo_detail

/// Combined first-order path export for one lane (former trace_paths_order1_impl).
/// `SplitScene` selects the primary-only vs primary+secondary visibility casts.
template <typename Params, typename Traverser, bool SplitScene>
RAYD_DEVICE void trace_paths_order1_algo(const Params& params, std::uint32_t lane, const Traverser& primary,
                                         const Traverser& secondary) {
    using namespace diffraction_paths_algo_detail;

    if (lane >= static_cast<unsigned int>(params.n_rays) || params.capacity <= 0 || params.tx_count <= 0 ||
        params.rx_count <= 0 || params.state_count <= 0 || params.state_limit <= 0 || params.max_order != 1 ||
        (params.strategy_mask & kStrategyDirect) == 0 || params.receiver_model != kReceiverMatchedIso) {
        return;
    }

    const int state_limit = params.state_limit;
    const int rx_count = params.rx_count;
    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(state_limit));
    const int pair_idx = static_cast<int>(lane / static_cast<unsigned int>(state_limit));
    const int rx_idx = pair_idx % rx_count;
    const int tx_idx = pair_idx / rx_count;
    if (tx_idx >= params.tx_count || state_idx >= params.state_count || !state_active(params, state_idx)) {
        return;
    }

    const Vec3f source = state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const Vec3f edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const Vec3f edge_dir = math::normalize_f32(
        state_vec(params.state_edge_dir_x, params.state_edge_dir_y, params.state_edge_dir_z, state_idx));
    const float t_min = params.state_edge_t_min[state_idx];
    const float t_max = params.state_edge_t_max[state_idx];
    const float edge_length = t_max - t_min;
    const Vec3f receiver = math::make_vec3(params.rx_pos_x[rx_idx], params.rx_pos_y[rx_idx], params.rx_pos_z[rx_idx]);

    if (!is_finite(source.x) || !is_finite(source.y) || !is_finite(source.z) || !is_finite(edge_pos.x) ||
        !is_finite(edge_pos.y) || !is_finite(edge_pos.z) || !is_finite(receiver.x) || !is_finite(receiver.y) ||
        !is_finite(receiver.z) || !(edge_length > kSmallEps)) {
        return;
    }
    const Vec3f edge_origin = math::add(edge_pos, math::scale(edge_dir, t_min));
    const float parameter =
        utd::first_order_diffraction_parameter(to_utd(source), to_utd(receiver), to_utd(edge_origin), to_utd(edge_dir));
    if (!is_finite(parameter)) {
        return;
    }
    const float clamped_parameter = fminf(fmaxf(parameter, 0.f), edge_length);
    const Vec3f edge_point = math::add(edge_origin, math::scale(edge_dir, clamped_parameter));
    if (!visible_segment<SplitScene>(primary, secondary, source, edge_point) ||
        !visible_segment<SplitScene>(primary, secondary, edge_point, receiver)) {
        return;
    }

    const utd::PairInputs pair = direct_pair_inputs(params, state_idx, source, edge_pos, edge_dir, t_min, t_max);
    const utd::PairOutputs utd_out =
        utd::compute_pair_contribution(pair, to_utd(receiver), params.k, paths_material_params(params));
    const float field_norm = utd::cplx_abs_sqr(utd_out.vectorField.x) + utd::cplx_abs_sqr(utd_out.vectorField.y) +
                             utd::cplx_abs_sqr(utd_out.vectorField.z);
    if (!(field_norm > 1.0e-30f) || !is_finite(field_norm)) {
        return;
    }
    const float amplitude_scale = sqrtf(fmaxf(params.state_src_power[state_idx], 0.f));

    const int out_idx = reserve_path_output(params, lane);
    if (out_idx < 0 || out_idx >= params.capacity) {
        return;
    }

    const float path_length =
        math::length_f32(math::subtract(edge_point, source)) + math::length_f32(math::subtract(receiver, edge_point));

    params.out_valid[out_idx] = 1u;
    params.out_tx_id[out_idx] = tx_idx;
    params.out_rx_id[out_idx] = rx_idx;
    params.out_order[out_idx] = 1;
    params.out_edge0[out_idx] = params.state_edge_index[state_idx];
    params.out_edge1[out_idx] = -1;
    params.out_edge2[out_idx] = -1;
    params.out_delay[out_idx] = path_length / kSpeedOfLight;
    params.out_field_x_re[out_idx] = utd_out.vectorField.x.re * amplitude_scale;
    params.out_field_x_im[out_idx] = utd_out.vectorField.x.im * amplitude_scale;
    params.out_field_y_re[out_idx] = utd_out.vectorField.y.re * amplitude_scale;
    params.out_field_y_im[out_idx] = utd_out.vectorField.y.im * amplitude_scale;
    params.out_field_z_re[out_idx] = utd_out.vectorField.z.re * amplitude_scale;
    params.out_field_z_im[out_idx] = utd_out.vectorField.z.im * amplitude_scale;
    write_point(params.out_p0_x, params.out_p0_y, params.out_p0_z, out_idx, edge_point);
    write_point(params.out_p1_x, params.out_p1_y, params.out_p1_z, out_idx, math::make_vec3(0.f, 0.f, 0.f));
    write_point(params.out_p2_x, params.out_p2_y, params.out_p2_z, out_idx, math::make_vec3(0.f, 0.f, 0.f));
}

/// Two-phase source-visibility prepass for one lane (former
/// trace_paths_order1_source_visibility_primary_impl). Primary handle only.
template <typename Params, typename Traverser>
RAYD_DEVICE void trace_paths_source_visibility_algo(const Params& params, std::uint32_t lane, const Traverser& primary,
                                                    const Traverser& secondary) {
    using namespace diffraction_paths_algo_detail;

    if (lane >= static_cast<unsigned int>(params.n_rays) || params.temp_visibility == nullptr) {
        return;
    }
    params.temp_visibility[lane] = 0u;

    int state_idx = -1;
    int rx_idx = -1;
    int tx_idx = -1;
    if (!paths_order1_lane(params, lane, state_idx, rx_idx, tx_idx)) {
        return;
    }
    (void)tx_idx;

    const Vec3f source = state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const Vec3f edge_point = paths_edge_point(params, state_idx, rx_idx);
    if (!is_finite(source.x) || !is_finite(source.y) || !is_finite(source.z) || !is_finite(edge_point.x) ||
        !is_finite(edge_point.y) || !is_finite(edge_point.z)) {
        return;
    }

    params.temp_visibility[lane] = visible_segment<false>(primary, secondary, source, edge_point) ? 1u : 0u;
}

/// Two-phase target-export pass for one lane (former
/// trace_paths_order1_target_export_primary_impl). Primary handle only.
template <typename Params, typename Traverser>
RAYD_DEVICE void trace_paths_target_export_algo(const Params& params, std::uint32_t lane, const Traverser& primary,
                                                const Traverser& secondary) {
    using namespace diffraction_paths_algo_detail;

    if (lane >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }
    if (params.temp_visibility != nullptr && params.temp_visibility[lane] == 0u) {
        return;
    }

    int state_idx = -1;
    int rx_idx = -1;
    int tx_idx = -1;
    if (!paths_order1_lane(params, lane, state_idx, rx_idx, tx_idx)) {
        return;
    }

    const Vec3f source = state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const Vec3f edge_point = paths_edge_point(params, state_idx, rx_idx);
    const Vec3f receiver = math::make_vec3(params.rx_pos_x[rx_idx], params.rx_pos_y[rx_idx], params.rx_pos_z[rx_idx]);

    if (!finite_paths_points(source, edge_point, receiver)) {
        return;
    }
    if (!visible_segment<false>(primary, secondary, edge_point, receiver)) {
        return;
    }

    const Vec3f edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const Vec3f edge_dir = math::normalize_f32(
        state_vec(params.state_edge_dir_x, params.state_edge_dir_y, params.state_edge_dir_z, state_idx));
    const utd::PairInputs pair =
        direct_pair_inputs(params, state_idx, source, edge_pos, edge_dir, params.state_edge_t_min[state_idx],
                           params.state_edge_t_max[state_idx]);
    const utd::PairOutputs utd_out =
        utd::compute_pair_contribution(pair, to_utd(receiver), params.k, paths_material_params(params));
    const float field_norm = utd::cplx_abs_sqr(utd_out.vectorField.x) + utd::cplx_abs_sqr(utd_out.vectorField.y) +
                             utd::cplx_abs_sqr(utd_out.vectorField.z);
    if (!(field_norm > 1.0e-30f) || !is_finite(field_norm)) {
        return;
    }
    const float amplitude_scale = sqrtf(fmaxf(params.state_src_power[state_idx], 0.f));

    const int out_idx = reserve_path_output(params, lane);
    if (out_idx < 0 || out_idx >= params.capacity) {
        return;
    }

    const float path_length =
        math::length_f32(math::subtract(edge_point, source)) + math::length_f32(math::subtract(receiver, edge_point));

    params.out_valid[out_idx] = 1u;
    params.out_tx_id[out_idx] = tx_idx;
    params.out_rx_id[out_idx] = rx_idx;
    params.out_order[out_idx] = 1;
    params.out_edge0[out_idx] = params.state_edge_index[state_idx];
    params.out_edge1[out_idx] = -1;
    params.out_edge2[out_idx] = -1;
    params.out_delay[out_idx] = path_length / kSpeedOfLight;
    params.out_field_x_re[out_idx] = utd_out.vectorField.x.re * amplitude_scale;
    params.out_field_x_im[out_idx] = utd_out.vectorField.x.im * amplitude_scale;
    params.out_field_y_re[out_idx] = utd_out.vectorField.y.re * amplitude_scale;
    params.out_field_y_im[out_idx] = utd_out.vectorField.y.im * amplitude_scale;
    params.out_field_z_re[out_idx] = utd_out.vectorField.z.re * amplitude_scale;
    params.out_field_z_im[out_idx] = utd_out.vectorField.z.im * amplitude_scale;
    write_point(params.out_p0_x, params.out_p0_y, params.out_p0_z, out_idx, edge_point);
    write_point(params.out_p1_x, params.out_p1_y, params.out_p1_z, out_idx, math::make_vec3(0.f, 0.f, 0.f));
    write_point(params.out_p2_x, params.out_p2_y, params.out_p2_z, out_idx, math::make_vec3(0.f, 0.f, 0.f));
}

} // namespace rayd::shared::multipath
