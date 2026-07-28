// Companion fragment to diffraction_accumulation_ad_device.cuh: the unit-VJP
// enumerations shared by both backends. Include this AFTER the core fragment
// and AFTER defining the backend gradient-write helpers, plus the call-site
// macros below (P = params, PR = primal, G = grad_contribution, F = gradient
// field, S = its stride field, I = index, T = tangent):
//
//   RAYD_DFR_AD_ADD_UNIT_VJP(P, PR, G, F, S, I, T)         strided state grads
//   RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(P, PR, G, F, I, T)      dense tri grads
//   RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(P, PR, G, F, S, I, T)   strided state grads
//   RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(P, PR, G, F, I, T) dense tri grads
//
// Each macro forwards to the backend's own add_unit_vjp / add_chain_unit_vjp
// helper spelled exactly as before the dedup, so gradient-write codegen stays
// bitwise identical per backend.
#pragma once

static __forceinline__ __device__ void chain_vjp_by_unit_jvps(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float grad_contribution) {
    ChainTangent tangent = {};
    tangent.first.edge_pos = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_pos_x, grad_state_edge_pos_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_pos = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_pos_y, grad_state_edge_pos_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_pos = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_pos_z, grad_state_edge_pos_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_dir_raw = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_dir_x, grad_state_edge_dir_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_dir_raw = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_dir_y, grad_state_edge_dir_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_dir_raw = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_dir_z, grad_state_edge_dir_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_t_min = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_t_min, grad_state_edge_t_min_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_t_max = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_edge_t_max, grad_state_edge_t_max_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.source = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_src_x, grad_state_src_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.source = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_src_y, grad_state_src_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.source = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_src_z, grad_state_src_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.src_power = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_src_power, grad_state_src_power_stride, p.first_idx, tangent);
    tangent = {};
    tangent.first.exterior_angle = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_state_exterior_angle, grad_state_exterior_angle_stride, p.first_idx, tangent);
    if (p.first.material_active && p.first.material_idx >= 0) {
        tangent = {};
        tangent.first.material_gain = 1.f;
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_material_gain, grad_material_gain_stride, p.first.material_idx, tangent);
    }

    tangent = {};
    tangent.second.edge_pos = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_pos_x, grad_recursive_state_edge_pos_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_pos = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_pos_y, grad_recursive_state_edge_pos_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_pos = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_pos_z, grad_recursive_state_edge_pos_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_dir_raw = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_dir_x, grad_recursive_state_edge_dir_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_dir_raw = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_dir_y, grad_recursive_state_edge_dir_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_dir_raw = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_dir_z, grad_recursive_state_edge_dir_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_t_min = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_t_min, grad_recursive_state_edge_t_min_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_t_max = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_t_max, grad_recursive_state_edge_t_max_stride, p.second_idx, tangent);
    tangent = {};
    tangent.second.exterior_angle = 1.f;
    RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_exterior_angle, grad_recursive_state_exterior_angle_stride, p.second_idx, tangent);
    if (p.second.material_active && p.second.material_idx >= 0) {
        tangent = {};
        tangent.second.material_gain = 1.f;
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_material_gain, grad_material_gain_stride, p.second.material_idx, tangent);
    }

    if (p.has_third) {
        tangent = {};
        tangent.third.edge_pos = dfr_make3(1.f, 0.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_pos_x, grad_recursive_state_edge_pos_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_pos = dfr_make3(0.f, 1.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_pos_y, grad_recursive_state_edge_pos_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_pos = dfr_make3(0.f, 0.f, 1.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_pos_z, grad_recursive_state_edge_pos_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_dir_raw = dfr_make3(1.f, 0.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_dir_x, grad_recursive_state_edge_dir_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_dir_raw = dfr_make3(0.f, 1.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_dir_y, grad_recursive_state_edge_dir_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_dir_raw = dfr_make3(0.f, 0.f, 1.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_dir_z, grad_recursive_state_edge_dir_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_t_min = 1.f;
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_t_min, grad_recursive_state_edge_t_min_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_t_max = 1.f;
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_edge_t_max, grad_recursive_state_edge_t_max_stride, p.third_idx, tangent);
        tangent = {};
        tangent.third.exterior_angle = 1.f;
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_recursive_state_exterior_angle, grad_recursive_state_exterior_angle_stride, p.third_idx, tangent);
        if (p.third.material_active && p.third.material_idx >= 0) {
            tangent = {};
            tangent.third.material_gain = 1.f;
            RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_material_gain, grad_material_gain_stride, p.third.material_idx, tangent);
        }
    }
    if (p.suffix_material_active && p.suffix_material_idx >= 0) {
        tangent = {};
        tangent.suffix_material_gain = 1.f;
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(params, p, grad_contribution, grad_material_gain, grad_material_gain_stride, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = dfr_make3(1.f, 0.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_p0_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = dfr_make3(0.f, 1.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_p0_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = dfr_make3(0.f, 0.f, 1.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_p0_z, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = dfr_make3(1.f, 0.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_fn_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = dfr_make3(0.f, 1.f, 0.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_fn_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = dfr_make3(0.f, 0.f, 1.f);
        RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_fn_z, p.suffix_material_idx, tangent);
    }
}

static __forceinline__ __device__ void vjp_by_unit_jvps(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution) {
    DfrTangent tangent = {};
    tangent.edge_pos = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_pos_x, grad_state_edge_pos_stride, p.state_idx, tangent);
    tangent = {};
    tangent.edge_pos = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_pos_y, grad_state_edge_pos_stride, p.state_idx, tangent);
    tangent = {};
    tangent.edge_pos = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_pos_z, grad_state_edge_pos_stride, p.state_idx, tangent);

    tangent = {};
    tangent.edge_dir_raw = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_dir_x, grad_state_edge_dir_stride, p.state_idx, tangent);
    tangent = {};
    tangent.edge_dir_raw = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_dir_y, grad_state_edge_dir_stride, p.state_idx, tangent);
    tangent = {};
    tangent.edge_dir_raw = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_dir_z, grad_state_edge_dir_stride, p.state_idx, tangent);

    tangent = {};
    tangent.edge_t_min = 1.f;
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_t_min, grad_state_edge_t_min_stride, p.state_idx, tangent);
    tangent = {};
    tangent.edge_t_max = 1.f;
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_edge_t_max, grad_state_edge_t_max_stride, p.state_idx, tangent);

    tangent = {};
    tangent.source = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_src_x, grad_state_src_stride, p.state_idx, tangent);
    tangent = {};
    tangent.source = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_src_y, grad_state_src_stride, p.state_idx, tangent);
    tangent = {};
    tangent.source = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_src_z, grad_state_src_stride, p.state_idx, tangent);

    tangent = {};
    tangent.wi_raw = dfr_make3(1.f, 0.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_wi_x, grad_state_wi_stride, p.state_idx, tangent);
    tangent = {};
    tangent.wi_raw = dfr_make3(0.f, 1.f, 0.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_wi_y, grad_state_wi_stride, p.state_idx, tangent);
    tangent = {};
    tangent.wi_raw = dfr_make3(0.f, 0.f, 1.f);
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_wi_z, grad_state_wi_stride, p.state_idx, tangent);

    tangent = {};
    tangent.src_power = 1.f;
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_src_power, grad_state_src_power_stride, p.state_idx, tangent);
    tangent = {};
    tangent.exterior_angle = 1.f;
    RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_state_exterior_angle, grad_state_exterior_angle_stride, p.state_idx, tangent);
    if (p.material_active && p.material_idx >= 0) {
        tangent = {};
        tangent.material_gain = 1.f;
        RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_material_gain, grad_material_gain_stride, p.material_idx, tangent);
    }
    if (p.suffix_material_active && p.suffix_material_idx >= 0) {
        tangent = {};
        tangent.suffix_material_gain = 1.f;
        RAYD_DFR_AD_ADD_UNIT_VJP(params, p, grad_contribution, grad_material_gain, grad_material_gain_stride, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = dfr_make3(1.f, 0.f, 0.f);
        RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_p0_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = dfr_make3(0.f, 1.f, 0.f);
        RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_p0_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = dfr_make3(0.f, 0.f, 1.f);
        RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_p0_z, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = dfr_make3(1.f, 0.f, 0.f);
        RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_fn_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = dfr_make3(0.f, 1.f, 0.f);
        RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_fn_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = dfr_make3(0.f, 0.f, 1.f);
        RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(params, p, grad_contribution, grad_tri_fn_z, p.suffix_material_idx, tangent);
    }
}
