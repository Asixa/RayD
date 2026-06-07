#include <raydtorch/diffraction/accum_reduce.h>
#include <raydtorch/common/optix_context.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_i32_count(int64_t count, const char *name) {
    if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + ": count is outside int32 launch range.");
    }
}

struct AddFloat4 {
    __host__ __device__ float4 operator()(float4 a, float4 b) const {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
};

__global__ void scatter_dfr_accum_reduced_kernel(
    const int *__restrict__ num_runs,
    const int *__restrict__ unique_cells,
    const float4 *__restrict__ reduced_values,
    float *__restrict__ out_power,
    float *__restrict__ out_field_x_re,
    int *__restrict__ out_direct_count,
    int *__restrict__ out_keller_count,
    int *__restrict__ out_edge_uses) {
    const int n = *num_runs;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        const int cell = unique_cells[idx];
        if (cell < 0) {
            continue;
        }
        const float4 value = reduced_values[idx];
        if (out_power != nullptr) {
            atomicAdd(out_power + cell, value.x);
        }
        if (out_field_x_re != nullptr) {
            atomicAdd(out_field_x_re + cell, value.y);
        }
        const int direct_count = static_cast<int>(value.z + 0.5f);
        const int keller_count = static_cast<int>(value.w + 0.5f);
        if (out_direct_count != nullptr && direct_count != 0) {
            atomicAdd(out_direct_count, direct_count);
        }
        if (out_keller_count != nullptr && keller_count != 0) {
            atomicAdd(out_keller_count, keller_count);
        }
        if (out_edge_uses != nullptr) {
            const int edge_uses = direct_count + keller_count;
            if (edge_uses != 0) {
                atomicAdd(out_edge_uses, edge_uses);
            }
        }
    }
}

} // namespace

void reduce_dfr_accum_staged_cuda(
    int64_t sample_count,
    const at::Tensor &stage_cell,
    const at::Tensor &stage_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_direct_count,
    at::Tensor &out_keller_count,
    at::Tensor &out_edge_uses) {
    require_i32_count(sample_count, "reduce_dfr_accum_staged_cuda(sample_count)");
    if (sample_count == 0) {
        return;
    }

    const int sample_count_i = static_cast<int>(sample_count);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cudaStream_t stream = torch_ctx.stream;
    at::TensorOptions key_options = stage_cell.options();
    at::TensorOptions value_options = stage_value.options().dtype(at::kFloat);
    at::TensorOptions byte_options = stage_cell.options().dtype(at::kByte);

    at::Tensor sorted_cells = at::empty({sample_count}, key_options);
    at::Tensor sorted_values = at::empty({sample_count, 4}, value_options);
    auto *values_in = reinterpret_cast<float4 *>(stage_value.data_ptr<float>());
    auto *values_sorted = reinterpret_cast<float4 *>(sorted_values.data_ptr<float>());

    size_t sort_temp_bytes = 0;
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_temp_bytes,
            stage_cell.data_ptr<int>(),
            sorted_cells.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(dfr accum size)");
    at::Tensor sort_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(sort_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr<uint8_t>(),
            sort_temp_bytes,
            stage_cell.data_ptr<int>(),
            sorted_cells.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(dfr accum)");

    at::Tensor unique_cells = at::empty({sample_count}, key_options);
    at::Tensor reduced_values = at::empty({sample_count, 4}, value_options);
    at::Tensor num_runs = at::empty({1}, key_options);
    auto *reduced_values_ptr = reinterpret_cast<float4 *>(reduced_values.data_ptr<float>());
    size_t reduce_temp_bytes = 0;
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            nullptr,
            reduce_temp_bytes,
            sorted_cells.data_ptr<int>(),
            unique_cells.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddFloat4{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(dfr accum size)");
    at::Tensor reduce_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(reduce_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            reduce_temp.data_ptr<uint8_t>(),
            reduce_temp_bytes,
            sorted_cells.data_ptr<int>(),
            unique_cells.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddFloat4{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(dfr accum)");

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((sample_count + block_size - 1) / block_size);
    scatter_dfr_accum_reduced_kernel<<<block_count, block_size, 0, stream>>>(
        num_runs.data_ptr<int>(),
        unique_cells.data_ptr<int>(),
        reduced_values_ptr,
        out_power.data_ptr<float>(),
        out_field_x_re.data_ptr<float>(),
        out_direct_count.data_ptr<int>(),
        out_keller_count.data_ptr<int>(),
        out_edge_uses.data_ptr<int>());
    cuda_check(cudaGetLastError(), "scatter_dfr_accum_reduced_kernel");
}

} // namespace raydtorch
