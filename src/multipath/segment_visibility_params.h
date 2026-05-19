#pragma once

#include <cstdint>

namespace rayd {

constexpr int SegmentVisibilityMaxSamples = 16;

struct SegmentVisibilityParams {
    uint64_t handle = 0;

    const int *face_offsets = nullptr;
    int n_meshes = 0;

    const float *start_x = nullptr;
    const float *start_y = nullptr;
    const float *start_z = nullptr;
    const float *end_x = nullptr;
    const float *end_y = nullptr;
    const float *end_z = nullptr;
    const float *end_b_x = nullptr;
    const float *end_b_y = nullptr;
    const float *end_b_z = nullptr;

    const float *edge_dir_x = nullptr;
    const float *edge_dir_y = nullptr;
    const float *edge_dir_z = nullptr;
    const float *edge_line_min = nullptr;
    const float *edge_line_max = nullptr;

    const int *ignore_prim_ids = nullptr;
    int ignore_k = 0;
    const uint8_t *active_mask = nullptr;

    int n_rays = 0;
    int sample_count = 0;
    float sample_fractions[SegmentVisibilityMaxSamples] = {};

    uint8_t *out_visible = nullptr;
    uint8_t *out_visible_b = nullptr;
};

} // namespace rayd
