#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace rayd::torch_backend::camera_detail {

__global__ void sample_to_world_kernel(
    int64_t count, const float *sample, int64_t s0, int64_t s1, float *world,
    float tan_x, float tan_y, float depth) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float u = sample[i * s0];
    const float v = sample[i * s0 + s1];
    world[i * 3] = (u * 2.0f - 1.0f) * tan_x * depth;
    world[i * 3 + 1] = (1.0f - v * 2.0f) * tan_y * depth;
    world[i * 3 + 2] = depth;
}

__global__ void sample_to_world_backward_kernel(
    int64_t count, const float *grad_world, int64_t s0, int64_t s1,
    float *grad_sample, float tan_x, float tan_y, float depth) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    grad_sample[i * 2] = grad_world[i * s0] * (2.0f * tan_x * depth);
    grad_sample[i * 2 + 1] = grad_world[i * s0 + s1] * (-2.0f * tan_y * depth);
}

__global__ void world_to_sample_kernel(
    int64_t count, const float *point, int64_t s0, int64_t s1, float *sample,
    float tan_x, float tan_y) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = point[i * s0];
    const float y = point[i * s0 + s1];
    const float z = fmaxf(point[i * s0 + 2 * s1], 1.0e-12f);
    sample[i * 2] = x / (z * tan_x) * 0.5f + 0.5f;
    sample[i * 2 + 1] = 0.5f - y / (z * tan_y) * 0.5f;
}

__global__ void world_to_sample_backward_kernel(
    int64_t count, const float *point, int64_t p0, int64_t p1,
    const float *grad_sample, int64_t g0, int64_t g1, float *grad_point,
    float tan_x, float tan_y) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = point[i * p0];
    const float y = point[i * p0 + p1];
    const float z_raw = point[i * p0 + 2 * p1];
    const float z = fmaxf(z_raw, 1.0e-12f);
    const float gu = grad_sample[i * g0];
    const float gv = grad_sample[i * g0 + g1];
    const float inv_z = 1.0f / z;
    const float inv_z2 = inv_z * inv_z;
    grad_point[i * 3] = gu * (0.5f * inv_z / tan_x);
    grad_point[i * 3 + 1] = gv * (-0.5f * inv_z / tan_y);
    grad_point[i * 3 + 2] = z_raw > 1.0e-12f
        ? gu * (-0.5f * x * inv_z2 / tan_x) + gv * (0.5f * y * inv_z2 / tan_y)
        : 0.0f;
}

__global__ void sample_ray_kernel(
    int64_t count, const float *sample, int64_t s0, int64_t s1,
    float *origin, float *direction, float tan_x, float tan_y) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = (sample[i * s0] * 2.0f - 1.0f) * tan_x;
    const float y = (1.0f - sample[i * s0 + s1] * 2.0f) * tan_y;
    const float inv_norm = rsqrtf(x * x + y * y + 1.0f);
    origin[i * 3] = origin[i * 3 + 1] = origin[i * 3 + 2] = 0.0f;
    direction[i * 3] = x * inv_norm;
    direction[i * 3 + 1] = y * inv_norm;
    direction[i * 3 + 2] = inv_norm;
}

__global__ void sample_ray_backward_kernel(
    int64_t count, const float *sample, int64_t s0, int64_t s1,
    const float *grad_direction, int64_t g0, int64_t g1, float *grad_sample,
    float tan_x, float tan_y) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (grad_direction == nullptr) {
        grad_sample[i * 2] = grad_sample[i * 2 + 1] = 0.0f;
        return;
    }
    const float x = (sample[i * s0] * 2.0f - 1.0f) * tan_x;
    const float y = (1.0f - sample[i * s0 + s1] * 2.0f) * tan_y;
    const float inv_norm = rsqrtf(x * x + y * y + 1.0f);
    const float dx = x * inv_norm;
    const float dy = y * inv_norm;
    const float dz = inv_norm;
    const float gx = grad_direction[i * g0];
    const float gy = grad_direction[i * g0 + g1];
    const float gz = grad_direction[i * g0 + 2 * g1];
    const float dot = gx * dx + gy * dy + gz * dz;
    grad_sample[i * 2] = (gx - dx * dot) * inv_norm * (2.0f * tan_x);
    grad_sample[i * 2 + 1] = (gy - dy * dot) * inv_norm * (-2.0f * tan_y);
}

template <typename Kernel, typename... Args>
void launch_1d(cudaStream_t stream, int64_t count, Kernel kernel, Args... args) {
    if (count == 0) return;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    kernel<<<blocks, threads, 0, stream>>>(count, args...);
}

} // namespace rayd::torch_backend::camera_detail
