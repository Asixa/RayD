// Copyright Xingyu Chen.
// Declares the private Dr.Jit SDF CUDA launcher.

#pragma once

#include <cstdint>

namespace rayd {

struct SdfJitLaunchParams {
    const float* values = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;

    const float* position_x = nullptr;
    const float* position_y = nullptr;
    const float* position_z = nullptr;
    const float* rotation = nullptr;
    const float* scale_x = nullptr;
    const float* scale_y = nullptr;
    const float* scale_z = nullptr;

    const float* origin_x = nullptr;
    const float* origin_y = nullptr;
    const float* origin_z = nullptr;
    const float* direction_x = nullptr;
    const float* direction_y = nullptr;
    const float* direction_z = nullptr;
    const float* ray_tmax = nullptr;
    const std::uint8_t* active = nullptr;
    int ray_count = 0;

    int max_steps = 64;
    float relaxation = 0.9f;
    float eps_hit = -1.0f;

    float* out_t = nullptr;
    std::uint8_t* out_hit = nullptr;
    float* out_position_x = nullptr;
    float* out_position_y = nullptr;
    float* out_position_z = nullptr;
    float* out_normal_x = nullptr;
    float* out_normal_y = nullptr;
    float* out_normal_z = nullptr;
    int* out_steps = nullptr;
    int* out_base_x = nullptr;
    int* out_base_y = nullptr;
    int* out_base_z = nullptr;
    float* out_denominator = nullptr;
};

void launch_sdf_intersect_jit(const SdfJitLaunchParams& params, void* stream);

} // namespace rayd
