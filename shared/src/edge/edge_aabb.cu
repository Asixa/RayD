#include <rayd/shared/edge/edge_aabb.h>

#include <cuda_runtime.h>

namespace rayd::shared::edge {
namespace {

__global__ void compute_edge_aabbs_kernel(
    int edge_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float inflation,
    float *out_aabbs) {
    const int edge = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (edge >= edge_count)
        return;

    const float p0_x = edge_p0_x[edge];
    const float p0_y = edge_p0_y[edge];
    const float p0_z = edge_p0_z[edge];
    const float p1_x = p0_x + edge_e1_x[edge];
    const float p1_y = p0_y + edge_e1_y[edge];
    const float p1_z = p0_z + edge_e1_z[edge];
    const float radius = fmaxf(inflation, 0.0f);
    const int base = edge * 6;
    out_aabbs[base + 0] = fminf(p0_x, p1_x) - radius;
    out_aabbs[base + 1] = fminf(p0_y, p1_y) - radius;
    out_aabbs[base + 2] = fminf(p0_z, p1_z) - radius;
    out_aabbs[base + 3] = fmaxf(p0_x, p1_x) + radius;
    out_aabbs[base + 4] = fmaxf(p0_y, p1_y) + radius;
    out_aabbs[base + 5] = fmaxf(p0_z, p1_z) + radius;
}

} // namespace

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
    cudaStream_t stream) {
    if (edge_count == 0)
        return;

    constexpr int block_size = 256;
    const int block_count = (edge_count + block_size - 1) / block_size;
    compute_edge_aabbs_kernel<<<block_count, block_size, 0, stream>>>(
        edge_count,
        edge_p0_x,
        edge_p0_y,
        edge_p0_z,
        edge_e1_x,
        edge_e1_y,
        edge_e1_z,
        inflation,
        out_aabbs);
}

} // namespace rayd::shared::edge
