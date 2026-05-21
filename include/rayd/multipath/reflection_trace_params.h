#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <rayd/optix.h>
#endif

namespace rayd {

/// Launch parameters for the native reflection-trace pipeline. All array fields are
/// flat device pointers (structure-of-arrays); the host fills inputs and pre-sizes outputs.
struct ReflectionTraceParams {
    OptixTraversableHandle primary_handle;   ///< Primary scene IAS handle.
    OptixTraversableHandle secondary_handle; ///< Secondary IAS handle (split static/dynamic scene).
    int split_mode;                          ///< 0 = single scene, nonzero = traverse both handles.

    // Scene-global triangles in edge-vector form: p0 + s*e1 + t*e2, with face normal fn.
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

    const int *face_offsets;  ///< Per-mesh face prefix-sum for globalizing primitive ids.
    int n_meshes;
    int n_triangles;

    // Input rays (SoA) and per-ray active mask.
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

    // Outputs: per-ray bounce_count plus ray-major (n_rays * max_bounces) per-slot
    // arrays, and the trailing-segment state past the last reflection.
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
    float *out_trailing_t;
    int *out_trailing_prim;
    float *out_trailing_dir_x;
    float *out_trailing_dir_y;
    float *out_trailing_dir_z;
    float *out_trailing_origin_x;
    float *out_trailing_origin_y;
    float *out_trailing_origin_z;
};

} // namespace rayd
