// Copyright Xingyu Chen.
// Defines shared visibility support for segment params.

#pragma once

#include <cstdint>
#include <type_traits>

namespace rayd::shared::optix {

constexpr int SegmentVisibilityMaxSamples = 16;

struct SegmentVisibilityParams {
    std::uint64_t handle = 0;
    const int* face_offsets = nullptr;
    int n_meshes = 0;

    const float* start_aos = nullptr;
    const float* end_aos = nullptr;
    const float* end_b_aos = nullptr;
    const float* start_x = nullptr;
    const float* start_y = nullptr;
    const float* start_z = nullptr;
    const float* end_x = nullptr;
    const float* end_y = nullptr;
    const float* end_z = nullptr;
    const float* end_b_x = nullptr;
    const float* end_b_y = nullptr;
    const float* end_b_z = nullptr;

    const float* edge_dir_x = nullptr;
    const float* edge_dir_y = nullptr;
    const float* edge_dir_z = nullptr;
    const float* edge_t_min = nullptr;
    const float* edge_t_max = nullptr;

    const float* chain_point_x = nullptr;
    const float* chain_point_y = nullptr;
    const float* chain_point_z = nullptr;
    const int* chain_length = nullptr;
    int max_points = 0;
    int max_segments = 0;

    const int* ignore_prim_ids = nullptr;
    int ignore_k = 0;
    const std::uint8_t* active_mask = nullptr;
    int n_rays = 0;
    int sample_count = 0;
    float sample_fractions[SegmentVisibilityMaxSamples] = {};

    std::uint8_t* out_visible = nullptr;
    std::uint8_t* out_visible_b = nullptr;
    int* out_first_blocked_segment = nullptr;
    int* out_first_blocked_prim = nullptr;
    float* out_t = nullptr;
};

static_assert(std::is_standard_layout_v<SegmentVisibilityParams>);
static_assert(std::is_trivially_copyable_v<SegmentVisibilityParams>);

} // namespace rayd::shared::optix
