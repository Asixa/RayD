#pragma once

#include <rayd/shared/visibility/segment_params.h>

#include <cstdint>
#include <type_traits>

namespace rayd::torch_backend {

inline constexpr int AxialEdgeVisibilitySampleCount = 4;

struct AxialEdgeVisibilityParams {
    shared::optix::SegmentVisibilityParams trace = {};
    const float *tx = nullptr;
    const float *edge_position = nullptr;
    const float *edge_direction = nullptr;
    const float *edge_t_min = nullptr;
    const float *edge_t_max = nullptr;
    const std::uint8_t *active = nullptr;
    int state_count = 0;
    float sample_fractions[AxialEdgeVisibilitySampleCount] = {};
    std::uint8_t *out_any_visible = nullptr;
};

static_assert(std::is_standard_layout_v<AxialEdgeVisibilityParams>);
static_assert(std::is_trivially_copyable_v<AxialEdgeVisibilityParams>);

} // namespace rayd::torch_backend
