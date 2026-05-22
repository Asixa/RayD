#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <rayd/optix.h>
#endif

namespace rayd {

/// Launch parameters for compact first-order diffraction path export.
struct DiffractionPathParams {
    OptixTraversableHandle primary_handle;
    OptixTraversableHandle secondary_handle;
    int split_mode;

    int n_rays;
    int capacity;

    const float *tx_pos_x;
    const float *tx_pos_y;
    const float *tx_pos_z;
    int tx_count;

    const float *rx_pos_x;
    const float *rx_pos_y;
    const float *rx_pos_z;
    int rx_count;

    const uint8_t *active_mask;
    int active_width;
    int state_count;
    int state_limit;
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

    const float *material_gain;
    const uint8_t *material_valid;
    int material_count;

    float wavelength;
    float k;
    int seed;
    int max_order;
    int strategy_mask;
    int sample_count;
    int return_geometry;
    int receiver_model;

    int *out_count;
    uint8_t *out_valid;
    int *out_tx_index;
    int *out_rx_index;
    int *out_order;
    int *out_edge_index_0;
    int *out_edge_index_1;
    int *out_edge_index_2;
    float *out_delay;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    float *out_point_0_x;
    float *out_point_0_y;
    float *out_point_0_z;
    float *out_point_1_x;
    float *out_point_1_y;
    float *out_point_1_z;
    float *out_point_2_x;
    float *out_point_2_y;
    float *out_point_2_z;
};

} // namespace rayd
