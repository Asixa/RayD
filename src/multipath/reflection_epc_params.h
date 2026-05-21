#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <rayd/optix.h>
#endif

namespace rayd {

constexpr int ReflectionEpcMaxBounces = 8;

struct ReflectionEpcParams {
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
    const float *rx_x;
    const float *rx_y;
    const float *rx_z;
    int rx_count;
    const uint8_t *active_mask;
    int n_rays;
    int max_bounces;

    uint8_t *out_valid;
    int *out_bounce_count;
    float *out_path_length;
    float *out_point_x;
    float *out_point_y;
    float *out_point_z;
    int *out_prim_ids;
    int *out_first_blocked_segment;
    int *out_first_blocked_prim;
};

} // namespace rayd
