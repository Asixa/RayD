// Copyright Xingyu Chen.
// Declares internal diffraction support for paths params.

#pragma once

#include <src/diffraction/common.h>
#include <rayd/diffraction/contracts.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <optix.h>
#endif

namespace rayd::torch_backend {

inline constexpr int kDiffractionPathLayoutCompact = 0;
inline constexpr int kDiffractionPathLayoutSourceLane = 1;

/// Launch parameters for first-order diffraction path export.
struct DfrPathParams {
    OptixTraversableHandle primary_handle;
    OptixTraversableHandle secondary_handle;
    int split_mode;

    int n_rays;
    int capacity;
    int output_layout;

    const float *tx_pos_x;
    const float *tx_pos_y;
    const float *tx_pos_z;
    const float *tx_pos_aos;
    int tx_pos_stride0;
    int tx_pos_stride1;
    int tx_count;
    const float *tx_pol_aos;
    int tx_pol_stride0;
    int tx_pol_stride1;
    int tx_pol_count;

    const float *rx_pos_x;
    const float *rx_pos_y;
    const float *rx_pos_z;
    const float *rx_pos_aos;
    int rx_pos_stride0;
    int rx_pos_stride1;
    int rx_count;

    const uint8_t *active_mask;
    int state_count;
    int state_limit;
    const int *state_edge_index;
    int state_edge_index_stride;
    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_pos_aos;
    int state_edge_pos_stride0;
    int state_edge_pos_stride1;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_dir_aos;
    int state_edge_dir_stride0;
    int state_edge_dir_stride1;
    const float *state_edge_t_min;
    int state_edge_t_min_stride;
    const float *state_edge_t_max;
    int state_edge_t_max_stride;
    const float *state_n0_x;
    const float *state_n0_y;
    const float *state_n0_z;
    const float *state_n0_aos;
    int state_n0_stride0;
    int state_n0_stride1;
    const float *state_n1_x;
    const float *state_n1_y;
    const float *state_n1_z;
    const float *state_n1_aos;
    int state_n1_stride0;
    int state_n1_stride1;
    const int *state_prim0;
    int state_prim0_stride;
    const int *state_prim1;
    int state_prim1_stride;
    const float *state_exterior_angle;
    int state_exterior_angle_stride;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_src_aos;
    int state_src_stride0;
    int state_src_stride1;
    const float *state_src_power;
    int state_src_power_stride;

    const float *material_eta_r;
    int material_eta_r_stride;
    const float *material_sigma;
    int material_sigma_stride;
    const float *material_mu_r;
    int material_mu_r_stride;
    const float *material_gain;
    int material_gain_stride;
    const uint8_t *material_valid;
    int material_valid_stride;
    int material_count;

    float wavelength;
    float k;
    float omega;
    // ADR-017 (channel-native) ISB boundary-taper width scale threaded into
    // each order-1 PairInputs fill. 0 (the default fill) reproduces the hard
    // GO step bit-for-bit; > 0 notches the incident-boundary odd part over the
    // congruent half-width in the shared UTD header. Off callers pass 0.
    float isb_taper_width_scale;
    int seed;
    int max_order;
    int strategy_mask;
    int sample_count;
    int return_geom;
    int receiver_model;

    uint8_t *temp_visibility;

    int *out_count;
    uint8_t *out_valid;
    int *out_tx_id;
    int *out_rx_id;
    int *out_order;
    int *out_edge0;
    int *out_edge1;
    int *out_edge2;
    float *out_delay;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    float *out_p0_x;
    float *out_p0_y;
    float *out_p0_z;
    float *out_p0_aos;
    float *out_p1_x;
    float *out_p1_y;
    float *out_p1_z;
    float *out_p2_x;
    float *out_p2_y;
    float *out_p2_z;
};

static_assert(std::is_standard_layout_v<DfrPathParams>);
static_assert(std::is_trivially_copyable_v<DfrPathParams>);
static_assert(sizeof(int) == sizeof(std::int32_t));

#define RAYD_ASSERT_DFR_PATH_PREFIX(Member, ContractMember)                  \
    static_assert(offsetof(DfrPathParams, Member) - offsetof(DfrPathParams, out_count) == \
                  offsetof(shared::optix::DiffractionPathOutputPrefix, ContractMember))

RAYD_ASSERT_DFR_PATH_PREFIX(out_count, count);
RAYD_ASSERT_DFR_PATH_PREFIX(out_valid, valid);
RAYD_ASSERT_DFR_PATH_PREFIX(out_tx_id, tx_id);
RAYD_ASSERT_DFR_PATH_PREFIX(out_rx_id, rx_id);
RAYD_ASSERT_DFR_PATH_PREFIX(out_order, order);
RAYD_ASSERT_DFR_PATH_PREFIX(out_edge0, edge0);
RAYD_ASSERT_DFR_PATH_PREFIX(out_edge1, edge1);
RAYD_ASSERT_DFR_PATH_PREFIX(out_edge2, edge2);
RAYD_ASSERT_DFR_PATH_PREFIX(out_delay, delay);
RAYD_ASSERT_DFR_PATH_PREFIX(out_field_x_re, field_x_re);
RAYD_ASSERT_DFR_PATH_PREFIX(out_field_x_im, field_x_im);
RAYD_ASSERT_DFR_PATH_PREFIX(out_field_y_re, field_y_re);
RAYD_ASSERT_DFR_PATH_PREFIX(out_field_y_im, field_y_im);
RAYD_ASSERT_DFR_PATH_PREFIX(out_field_z_re, field_z_re);
RAYD_ASSERT_DFR_PATH_PREFIX(out_field_z_im, field_z_im);
RAYD_ASSERT_DFR_PATH_PREFIX(out_p0_x, p0_x);
RAYD_ASSERT_DFR_PATH_PREFIX(out_p0_y, p0_y);
RAYD_ASSERT_DFR_PATH_PREFIX(out_p0_z, p0_z);

#undef RAYD_ASSERT_DFR_PATH_PREFIX

#define RAYD_ASSERT_DFR_PATH_TAIL(Member, ContractMember)                    \
    static_assert(offsetof(DfrPathParams, Member) - offsetof(DfrPathParams, out_p1_x) == \
                  offsetof(shared::optix::DiffractionPathGeometryTail, ContractMember))

RAYD_ASSERT_DFR_PATH_TAIL(out_p1_x, p1_x);
RAYD_ASSERT_DFR_PATH_TAIL(out_p1_y, p1_y);
RAYD_ASSERT_DFR_PATH_TAIL(out_p1_z, p1_z);
RAYD_ASSERT_DFR_PATH_TAIL(out_p2_x, p2_x);
RAYD_ASSERT_DFR_PATH_TAIL(out_p2_y, p2_y);
RAYD_ASSERT_DFR_PATH_TAIL(out_p2_z, p2_z);

#undef RAYD_ASSERT_DFR_PATH_TAIL

} // namespace rayd::torch_backend

