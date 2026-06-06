#pragma once

#include <optix.h>

#include <cstdint>

namespace raydtorch {

struct EdgeOptixQueryParams {
    OptixTraversableHandle traversable = 0;
    const float *vertices = nullptr;
    const int *edge_v0 = nullptr;
    const int *edge_v1 = nullptr;
    const float *point = nullptr;
    int *out_edge_id = nullptr;
    int32_t edge_count = 0;
    int32_t point_count = 0;
    float search_radius = 0.0f;
};

} // namespace raydtorch
