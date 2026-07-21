#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void cuda_check(cudaError_t result, const char *expression) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expression + ": " + cudaGetErrorString(result));
}

__global__ void deterministic_normalize_vec3_oracle_kernel(
    int64_t count,
    const float *__restrict__ values,
    float epsilon,
    float *__restrict__ output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int64_t base = index * 3;
    const float x = values[base + 0];
    const float y = values[base + 1];
    const float z = values[base + 2];
    const float length = fmaxf(sqrtf(x * x + y * y + z * z), epsilon);
    output[base + 0] = x / length;
    output[base + 1] = y / length;
    output[base + 2] = z / length;
}

__global__ void segment_restart_epsilon_oracle_kernel(
    int64_t count,
    const float *__restrict__ positions,
    float scene_diagonal,
    bool use_l2_norm,
    float *__restrict__ output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int64_t base = index * 3;
    const float x = positions[base + 0];
    const float y = positions[base + 1];
    const float z = positions[base + 2];
    const float position_norm = use_l2_norm
        ? sqrtf(x * x + y * y + z * z)
        : fmaxf(fmaxf(fabsf(x), fabsf(y)), fabsf(z));
    output[index] = fmaxf(
        position_norm * 1.0e-6f,
        fmaxf(scene_diagonal * 1.0e-6f, 1.0e-6f));
}

__global__ void segment_restart_point_oracle_kernel(
    int64_t count,
    const float *__restrict__ positions,
    const float *__restrict__ directions,
    const float *__restrict__ epsilon,
    float *__restrict__ output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int64_t base = index * 3;
    output[base + 0] = __fadd_rn(
        positions[base + 0], __fmul_rn(directions[base + 0], epsilon[index]));
    output[base + 1] = __fadd_rn(
        positions[base + 1], __fmul_rn(directions[base + 1], epsilon[index]));
    output[base + 2] = __fadd_rn(
        positions[base + 2], __fmul_rn(directions[base + 2], epsilon[index]));
}

} // namespace

at::Tensor channel_deterministic_normalize_vec3_oracle(
    const at::Tensor &values,
    float epsilon) {
    auto output = at::empty_like(values);
    const int64_t count = values.size(0);
    if (count == 0)
        return output;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device()).stream();
    deterministic_normalize_vec3_oracle_kernel<<<blocks, threads, 0, stream>>>(
        count,
        values.data_ptr<float>(),
        epsilon,
        output.data_ptr<float>());
    cuda_check(cudaGetLastError(), "deterministic normalize test oracle");
    return output;
}

at::Tensor channel_segment_restart_epsilon_oracle(
    const at::Tensor &positions,
    float scene_diagonal,
    bool use_l2_norm) {
    auto output = at::empty({positions.size(0)}, positions.options());
    const int64_t count = positions.size(0);
    if (count == 0)
        return output;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(positions.get_device()).stream();
    segment_restart_epsilon_oracle_kernel<<<blocks, threads, 0, stream>>>(
        count,
        positions.data_ptr<float>(),
        scene_diagonal,
        use_l2_norm,
        output.data_ptr<float>());
    cuda_check(cudaGetLastError(), "segment restart epsilon test oracle");
    return output;
}

at::Tensor channel_segment_restart_point_oracle(
    const at::Tensor &positions,
    const at::Tensor &directions,
    const at::Tensor &epsilon) {
    auto output = at::empty_like(positions);
    const int64_t count = positions.size(0);
    if (count == 0)
        return output;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(positions.get_device()).stream();
    segment_restart_point_oracle_kernel<<<blocks, threads, 0, stream>>>(
        count,
        positions.data_ptr<float>(),
        directions.data_ptr<float>(),
        epsilon.data_ptr<float>(),
        output.data_ptr<float>());
    cuda_check(cudaGetLastError(), "segment restart point test oracle");
    return output;
}
