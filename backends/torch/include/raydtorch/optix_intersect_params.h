#pragma once

#include <optix.h>

#include <cstdint>

namespace raydtorch {

struct OptixIntersectParams {
    OptixTraversableHandle traversable = 0;
    const float *ray_o = nullptr;
    const float *ray_d = nullptr;
    const float *ray_tmax = nullptr;
    const bool *active = nullptr;
    float *out_t = nullptr;
    int *out_prim_id = nullptr;
    float *out_bary_uv = nullptr;
    int32_t ray_count = 0;
};

} // namespace raydtorch
