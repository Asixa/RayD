#pragma once

#include <cstdint>

namespace rayd {

struct DfrDirectAccumADParams {
    int n_rays;
    int state_count;
    int material_count;
    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    float grid_cell_area;
    int direct_samples;
    int keller_samples;
    int seed;

    const uint8_t *tape_active;
    const int *tape_state_idx;
    const int *tape_cell;
    const int *tape_material_idx;
    const float *tape_edge_u;

    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_t_min;
    const float *state_edge_t_max;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_wi_x;
    const float *state_wi_y;
    const float *state_wi_z;
    const float *state_src_power;
    const float *state_exterior_angle;
    const float *material_gain;

    const float *dot_state_edge_pos_x;
    const float *dot_state_edge_pos_y;
    const float *dot_state_edge_pos_z;
    const float *dot_state_edge_dir_x;
    const float *dot_state_edge_dir_y;
    const float *dot_state_edge_dir_z;
    const float *dot_state_edge_t_min;
    const float *dot_state_edge_t_max;
    const float *dot_state_src_x;
    const float *dot_state_src_y;
    const float *dot_state_src_z;
    const float *dot_state_wi_x;
    const float *dot_state_wi_y;
    const float *dot_state_wi_z;
    const float *dot_state_src_power;
    const float *dot_state_exterior_angle;
    const float *dot_material_gain;

    float *dot_out_power;
    float *dot_out_field_x_re;

    const float *grad_out_power;
    const float *grad_out_field_x_re;

    float *grad_state_edge_pos_x;
    float *grad_state_edge_pos_y;
    float *grad_state_edge_pos_z;
    float *grad_state_edge_dir_x;
    float *grad_state_edge_dir_y;
    float *grad_state_edge_dir_z;
    float *grad_state_edge_t_min;
    float *grad_state_edge_t_max;
    float *grad_state_src_x;
    float *grad_state_src_y;
    float *grad_state_src_z;
    float *grad_state_wi_x;
    float *grad_state_wi_y;
    float *grad_state_wi_z;
    float *grad_state_src_power;
    float *grad_state_exterior_angle;
    float *grad_material_gain;
};

void dfr_direct_accum_jvp_gpu(const DfrDirectAccumADParams &params);
void dfr_direct_accum_vjp_gpu(const DfrDirectAccumADParams &params);

} // namespace rayd
