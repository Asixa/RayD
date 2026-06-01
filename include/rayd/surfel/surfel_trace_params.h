#pragma once

#include <cstdint>

namespace rayd {

/// Host/device ABI for the native surfel trace pipeline.
/// All pointers refer to CUDA device arrays owned by Dr.Jit or RayD.
struct SurfelTraceParams {
    uint64_t handle = 0;

    const float *ray_ox = nullptr;
    const float *ray_oy = nullptr;
    const float *ray_oz = nullptr;
    const float *ray_dx = nullptr;
    const float *ray_dy = nullptr;
    const float *ray_dz = nullptr;
    const float *ray_tmax = nullptr;
    const uint8_t *active_mask = nullptr;
    int ray_count = 0;

    const int *triangle_to_surfel_id = nullptr;
    int triangle_count = 0;
    int surfel_count = 0;

    const float *center_x = nullptr;
    const float *center_y = nullptr;
    const float *center_z = nullptr;
    const float *tangent_u_x = nullptr;
    const float *tangent_u_y = nullptr;
    const float *tangent_u_z = nullptr;
    const float *tangent_v_x = nullptr;
    const float *tangent_v_y = nullptr;
    const float *tangent_v_z = nullptr;
    const float *opacity = nullptr;
    const float *value = nullptr;
    const float *appearance_values = nullptr;
    int appearance_channel_count = 1;
    int color_model = 0;
    int appearance_sh_degree = 0;
    int sh_degree = 0;
    int render_channel_count = 1;
    float background_rgb[3] = {0.0f, 0.0f, 0.0f};

    float alpha_min = 1.0f / 255.0f;
    float alpha_cap = 0.99f;
    float ray_epsilon = 1.0e-3f;
    float tmax_fallback = 1.0e8f;
    int max_candidate_hits = 8;
    int face_forward = 1;

    int *out_triangle_id = nullptr;
    float *out_proxy_t = nullptr;
    uint8_t *out_valid = nullptr;

    int composite_hit_capacity = 8;
    int *scratch_surfel_id = nullptr;
    float *scratch_t = nullptr;
    float *scratch_alpha = nullptr;
    float *scratch_value = nullptr;
    float *out_intensity = nullptr;
    float *out_channels = nullptr;
    float *out_alpha = nullptr;
    float *out_transmittance = nullptr;
    float *out_depth = nullptr;
};

} // namespace rayd
