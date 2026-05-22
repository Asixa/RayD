#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <rayd/optix.h>
#endif

namespace rayd {

/// Launch parameters for the native order-1 diffraction accumulation pipeline.
struct DiffractionAccumParams {
    OptixTraversableHandle primary_handle;
    OptixTraversableHandle secondary_handle;
    int split_mode;

    int n_rays;  ///< Number of sample lanes launched by OptiX.

    const uint8_t *active_mask;
    int state_count;
    const int *state_edge_index;
    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_line_min;
    const float *state_edge_line_max;
    const float *state_face0_normal_x;
    const float *state_face0_normal_y;
    const float *state_face0_normal_z;
    const float *state_face1_normal_x;
    const float *state_face1_normal_y;
    const float *state_face1_normal_z;
    const int *state_face0_prim_id;
    const int *state_face1_prim_id;
    const float *state_exterior_angle;
    const float *state_source_x;
    const float *state_source_y;
    const float *state_source_z;
    const float *state_source_power;
    const float *state_incident_dir_x;
    const float *state_incident_dir_y;
    const float *state_incident_dir_z;
    const float *state_initial_dir_x;
    const float *state_initial_dir_y;
    const float *state_initial_dir_z;
    const int *state_prefix_reflection_depth;

    int recursive_state_count;
    const uint8_t *recursive_active_mask;
    const int *recursive_state_edge_index;
    const float *recursive_state_edge_pos_x;
    const float *recursive_state_edge_pos_y;
    const float *recursive_state_edge_pos_z;
    const float *recursive_state_edge_dir_x;
    const float *recursive_state_edge_dir_y;
    const float *recursive_state_edge_dir_z;
    const float *recursive_state_edge_line_min;
    const float *recursive_state_edge_line_max;
    const float *recursive_state_face0_normal_x;
    const float *recursive_state_face0_normal_y;
    const float *recursive_state_face0_normal_z;
    const float *recursive_state_face1_normal_x;
    const float *recursive_state_face1_normal_y;
    const float *recursive_state_face1_normal_z;
    const int *recursive_state_face0_prim_id;
    const int *recursive_state_face1_prim_id;
    const float *recursive_state_exterior_angle;

    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    float grid_cell_area;

    const float *material_eta_r;
    const float *material_sigma;
    const float *material_mu_r;
    const float *material_gain;
    const uint8_t *material_valid;
    int material_count;

    float wavelength;
    float k;
    int seed;
    int samples;
    int max_order;
    int direct_samples;
    int keller_samples;
    int strategy_mask;
    int sample_sequence;
    int receiver_model;
    int collect_edge_use;
    int collect_debug_counts;

    float *out_diffraction_power;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    int *out_direct_count;
    int *out_keller_count;
    int *out_suffix_count;
    int *out_visibility_reject_count;
    int *out_inter_edge_visibility_reject_count;
    int *out_utd_reject_count;
    int *out_edge_use_count;
};

} // namespace rayd
