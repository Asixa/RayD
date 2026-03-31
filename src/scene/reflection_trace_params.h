#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <rayd/optix.h>
#endif

namespace rayd {

struct ReflectionTraceParams {
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
    int max_bounces;

    int *out_bounce_count;
    int *out_shape_ids;
    int *out_prim_ids;
    float *out_t;
    float *out_bary_u;
    float *out_bary_v;
    float *out_hit_x;
    float *out_hit_y;
    float *out_hit_z;
    float *out_norm_x;
    float *out_norm_y;
    float *out_norm_z;
    float *out_img_x;
    float *out_img_y;
    float *out_img_z;
};

} // namespace rayd
