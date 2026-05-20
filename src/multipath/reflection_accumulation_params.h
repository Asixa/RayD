#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <rayd/optix.h>
#endif

namespace rayd {

struct ReflectionAccumulationParams {
    OptixTraversableHandle primary_handle;
    OptixTraversableHandle secondary_handle;
    int split_mode;

    const float *tri_p0_x;
    const float *tri_p0_y;
    const float *tri_p0_z;
    const float *tri_e1_x;
    const float *tri_e1_y;
    const float *tri_e1_z;
    const float *tri_e2_x;
    const float *tri_e2_y;
    const float *tri_e2_z;
    const float *tri_fn_x;
    const float *tri_fn_y;
    const float *tri_fn_z;

    const int *face_offsets;
    int n_meshes;
    int n_triangles;

    const float *ray_ox;
    const float *ray_oy;
    const float *ray_oz;
    const float *ray_dx;
    const float *ray_dy;
    const float *ray_dz;
    const float *ray_tmax;
    const uint8_t *active_mask;
    int n_rays;

    const float *tx_x;
    const float *tx_y;
    const float *tx_z;

    int max_bounces;
    float wavelength;
    float k;
    float solid_angle_per_ray;
    float cell_area;
    int seed;
    int rr_depth;
    float rr_prob;
    float stop_threshold;

    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;

    const float *material_eta_r;
    const float *material_sigma;
    const float *material_gain;
    const float *material_mu_r;
    const uint8_t *material_valid;
    int material_count;

    int collect_wedges;
    int collect_wedge_prefixes;
    int wedge_capacity;

    float *out_reflection_power;
    int *out_reflection_count;
    int *out_wedge_count;
    int *out_wedge_ray_index;
    float *out_wedge_hit_x;
    float *out_wedge_hit_y;
    float *out_wedge_hit_z;
    float *out_wedge_normal_x;
    float *out_wedge_normal_y;
    float *out_wedge_normal_z;
    int *out_wedge_prim_id;
    float *out_wedge_dir_x;
    float *out_wedge_dir_y;
    float *out_wedge_dir_z;
    int *out_wedge_bounce_depth;
};

} // namespace rayd
