// Copyright Xingyu Chen.
// Declares internal diffraction support for accumulation params Dr.Jit.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <rayd/diffraction/contracts.h>

namespace rayd {

/// Launch parameters for the native order-1 diffraction accumulation pipeline.
struct DfrAccumParams {
    std::uint64_t primary_handle;   ///< OptiX IAS handle; 0 on the CUDA path.
    std::uint64_t secondary_handle; ///< Split-scene IAS handle; 0 on the CUDA path.
    int split_mode;

    int n_rays; ///< Number of sample lanes launched by OptiX.

    const uint8_t* active_mask;
    int state_count;
    const int* state_edge_index;
    const float* state_edge_pos_x;
    const float* state_edge_pos_y;
    const float* state_edge_pos_z;
    const float* state_edge_dir_x;
    const float* state_edge_dir_y;
    const float* state_edge_dir_z;
    const float* state_edge_t_min;
    const float* state_edge_t_max;
    const float* state_n0_x;
    const float* state_n0_y;
    const float* state_n0_z;
    const float* state_n1_x;
    const float* state_n1_y;
    const float* state_n1_z;
    const int* state_prim0;
    const int* state_prim1;
    const float* state_exterior_angle;
    const float* state_src_x;
    const float* state_src_y;
    const float* state_src_z;
    const float* state_src_power;
    const float* state_wi_x;
    const float* state_wi_y;
    const float* state_wi_z;
    const float* state_d0_x;
    const float* state_d0_y;
    const float* state_d0_z;
    const int* state_prefix_depth;
    const float* utd_epx;
    const float* utd_epy;
    const float* utd_epz;
    const float* utd_edx;
    const float* utd_edy;
    const float* utd_edz;
    const float* utd_n0x;
    const float* utd_n0y;
    const float* utd_n0z;
    const float* utd_nnx;
    const float* utd_nny;
    const float* utd_nnz;
    const float* utd_wn;
    const float* utd_elm;
    const float* utd_elx;
    const float* utd_spx;
    const float* utd_spy;
    const float* utd_spz;
    const float* utd_ifr;
    const float* utd_ifi;
    const float* utd_inr;
    const float* utd_ini;
    const float* utd_r0r;
    const float* utd_r0i;
    const float* utd_rnr;
    const float* utd_rni;
    const float* utd_vxr;
    const float* utd_vxi;
    const float* utd_vyr;
    const float* utd_vyi;
    const float* utd_vzr;
    const float* utd_vzi;
    const float* utd_dxr;
    const float* utd_dxi;
    const float* utd_dyr;
    const float* utd_dyi;
    const float* utd_dzr;
    const float* utd_dzi;
    const float* utd_jur;
    const float* utd_jui;
    const float* utd_jvr;
    const float* utd_jvi;
    const float* utd_djur;
    const float* utd_djui;
    const float* utd_djvr;
    const float* utd_djvi;
    const float* utd_bux;
    const float* utd_buy;
    const float* utd_buz;
    const float* utd_bvx;
    const float* utd_bvy;
    const float* utd_bvz;
    const float* utd_bkx;
    const float* utd_bky;
    const float* utd_bkz;
    const float* utd_f0m00r;
    const float* utd_f0m00i;
    const float* utd_f0m01r;
    const float* utd_f0m01i;
    const float* utd_f0m10r;
    const float* utd_f0m10i;
    const float* utd_f0m11r;
    const float* utd_f0m11i;
    const float* utd_f1m00r;
    const float* utd_f1m00i;
    const float* utd_f1m01r;
    const float* utd_f1m01i;
    const float* utd_f1m10r;
    const float* utd_f1m10i;
    const float* utd_f1m11r;
    const float* utd_f1m11i;
    const float* utd_f0er;
    const float* utd_f0mu;
    const float* utd_f0sg;
    const float* utd_f0g;
    const float* utd_f0uf;
    const float* utd_f0pr;
    const float* utd_f1er;
    const float* utd_f1mu;
    const float* utd_f1sg;
    const float* utd_f1g;
    const float* utd_f1uf;
    const float* utd_f1pr;
    const float* utd_select;
    const float* const* coherent_utd_slots;
    int coherent_utd_slot_count;
    const int* coherent_owner_code;
    const int* coherent_adjacent_face0;
    const int* coherent_adjacent_face1;

    int recursive_state_count;
    const uint8_t* recursive_active_mask;
    const int* recursive_state_edge_index;
    const float* recursive_state_edge_pos_x;
    const float* recursive_state_edge_pos_y;
    const float* recursive_state_edge_pos_z;
    const float* recursive_state_edge_dir_x;
    const float* recursive_state_edge_dir_y;
    const float* recursive_state_edge_dir_z;
    const float* recursive_state_edge_t_min;
    const float* recursive_state_edge_t_max;
    const float* recursive_state_n0_x;
    const float* recursive_state_n0_y;
    const float* recursive_state_n0_z;
    const float* recursive_state_n1_x;
    const float* recursive_state_n1_y;
    const float* recursive_state_n1_z;
    const int* recursive_state_prim0;
    const int* recursive_state_prim1;
    const float* recursive_state_exterior_angle;

    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    float grid_cell_area;

    const float* tri_p0_x;
    const float* tri_p0_y;
    const float* tri_p0_z;
    const float* tri_e1_x;
    const float* tri_e1_y;
    const float* tri_e1_z;
    const float* tri_e2_x;
    const float* tri_e2_y;
    const float* tri_e2_z;
    const float* tri_fn_x;
    const float* tri_fn_y;
    const float* tri_fn_z;
    const int* face_offsets;
    int n_meshes;
    int n_triangles;
    const uint32_t* suffix_candidate_prim_id;
    int suffix_candidate_count;

    const float* material_eta_r;
    const float* material_sigma;
    const float* material_mu_r;
    const float* material_gain;
    const uint8_t* material_valid;
    int material_count;

    float wavelength;
    float k;
    int seed;
    int samples;
    int max_order;
    int direct_samples;
    int keller_samples;
    int suffix_samples;
    int strategy_mask;
    int sample_sequence;
    int receiver_model;
    int select_diffraction_point;
    int prefilter_visibility;
    int collect_edge_use;
    int collect_debug_counts;
    float omega;
    float tx_pol_x;
    float tx_pol_y;
    float tx_pol_z;

    float* out_power;
    float* out_field_x_re;
    float* out_field_x_im;
    float* out_field_y_re;
    float* out_field_y_im;
    float* out_field_z_re;
    float* out_field_z_im;
    int* out_direct_count;
    int* out_keller_count;
    int* out_suffix_count;
    int* out_vis_rejects;
    int* out_edge_vis_rejects;
    int* out_utd_rejects;
    int* out_edge_uses;

    float* out_direct_field_x_re;
    float* out_direct_field_x_im;
    float* out_direct_field_y_re;
    float* out_direct_field_y_im;
    float* out_direct_field_z_re;
    float* out_direct_field_z_im;
    float* out_multi_field_x_re;
    float* out_multi_field_x_im;
    float* out_multi_field_y_re;
    float* out_multi_field_y_im;
    float* out_multi_field_z_re;
    float* out_multi_field_z_im;
    int* out_multi_count;
    int* out_visibility_reject_count;
    int* out_utd_reject_count;

    uint8_t* temp_visibility;
    uint8_t* tape_active;
    int* tape_state_idx;
    int* tape_cell;
    int* tape_material_idx;
    float* tape_edge_u;
};

static_assert(std::is_standard_layout_v<DfrAccumParams>);
static_assert(std::is_trivially_copyable_v<DfrAccumParams>);
static_assert(sizeof(int) == sizeof(std::int32_t));

#define RAYD_ASSERT_DFR_GRID(Member, ContractMember)                                                                   \
    static_assert(offsetof(DfrAccumParams, Member) - offsetof(DfrAccumParams, grid_axis) ==                            \
                  offsetof(shared::optix::DiffractionGridParams, ContractMember))

RAYD_ASSERT_DFR_GRID(grid_axis, axis);
RAYD_ASSERT_DFR_GRID(grid_position, position);
RAYD_ASSERT_DFR_GRID(grid_coord0_min, coord0_min);
RAYD_ASSERT_DFR_GRID(grid_coord0_max, coord0_max);
RAYD_ASSERT_DFR_GRID(grid_coord1_min, coord1_min);
RAYD_ASSERT_DFR_GRID(grid_coord1_max, coord1_max);
RAYD_ASSERT_DFR_GRID(grid_resolution0, resolution0);
RAYD_ASSERT_DFR_GRID(grid_resolution1, resolution1);
RAYD_ASSERT_DFR_GRID(grid_cell_area, cell_area);

#undef RAYD_ASSERT_DFR_GRID

#define RAYD_ASSERT_DFR_ACCUM_OUTPUT(Member, ContractMember)                                                           \
    static_assert(offsetof(DfrAccumParams, Member) - offsetof(DfrAccumParams, out_power) ==                            \
                  offsetof(shared::optix::DiffractionAccumOutputPointers, ContractMember))

RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_power, power);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_field_x_re, field_x_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_field_x_im, field_x_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_field_y_re, field_y_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_field_y_im, field_y_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_field_z_re, field_z_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_field_z_im, field_z_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_count, direct_count);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_keller_count, keller_count);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_suffix_count, suffix_count);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_vis_rejects, visibility_rejects);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_edge_vis_rejects, edge_visibility_rejects);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_utd_rejects, utd_rejects);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_edge_uses, edge_uses);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_field_x_re, direct_field_x_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_field_x_im, direct_field_x_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_field_y_re, direct_field_y_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_field_y_im, direct_field_y_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_field_z_re, direct_field_z_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_direct_field_z_im, direct_field_z_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_field_x_re, multi_field_x_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_field_x_im, multi_field_x_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_field_y_re, multi_field_y_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_field_y_im, multi_field_y_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_field_z_re, multi_field_z_re);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_field_z_im, multi_field_z_im);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_multi_count, multi_count);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_visibility_reject_count, visibility_reject_count);
RAYD_ASSERT_DFR_ACCUM_OUTPUT(out_utd_reject_count, utd_reject_count);

#undef RAYD_ASSERT_DFR_ACCUM_OUTPUT

#define RAYD_ASSERT_DFR_ACCUM_TAPE(Member, ContractMember)                                                             \
    static_assert(offsetof(DfrAccumParams, Member) - offsetof(DfrAccumParams, tape_active) ==                          \
                  offsetof(shared::optix::DiffractionAccumTapePointers, ContractMember))

RAYD_ASSERT_DFR_ACCUM_TAPE(tape_active, active);
RAYD_ASSERT_DFR_ACCUM_TAPE(tape_state_idx, state_index);
RAYD_ASSERT_DFR_ACCUM_TAPE(tape_cell, cell);
RAYD_ASSERT_DFR_ACCUM_TAPE(tape_material_idx, material_index);
RAYD_ASSERT_DFR_ACCUM_TAPE(tape_edge_u, edge_u);

#undef RAYD_ASSERT_DFR_ACCUM_TAPE

} // namespace rayd
