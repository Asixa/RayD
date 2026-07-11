#pragma once

#include <cuda_runtime_api.h>

namespace rayd::shared::edge {

/// Enqueue packed edge AABB generation on a caller-owned CUDA stream.
void launch_edge_aabb(
    int edge_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float inflation,
    float *out_aabbs,
    cudaStream_t stream);

} // namespace rayd::shared::edge
