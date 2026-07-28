#include <rayd/torch/diffraction/paths_init.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

namespace {

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__global__ void init_dfr_path_outputs_kernel(
    int64_t capacity,
    int *__restrict__ out_count,
    uint8_t *__restrict__ out_valid,
    int *__restrict__ out_tx_id,
    int *__restrict__ out_rx_id,
    int *__restrict__ out_order,
    int *__restrict__ out_edge0,
    int *__restrict__ out_edge1,
    int *__restrict__ out_edge2,
    float *__restrict__ out_delay,
    float *__restrict__ out_field_x_re,
    float *__restrict__ out_field_x_im,
    float *__restrict__ out_field_y_re,
    float *__restrict__ out_field_y_im,
    float *__restrict__ out_field_z_re,
    float *__restrict__ out_field_z_im,
    float *__restrict__ out_p0,
    float *__restrict__ out_p1,
    float *__restrict__ out_p2) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx == 0) {
        out_count[0] = 0;
    }
    if (idx >= capacity) {
        return;
    }
    out_valid[idx] = 0u;
    out_tx_id[idx] = -1;
    out_rx_id[idx] = -1;
    out_order[idx] = 0;
    out_edge0[idx] = -1;
    out_edge1[idx] = -1;
    out_edge2[idx] = -1;
    out_delay[idx] = 0.0f;
    out_field_x_re[idx] = 0.0f;
    out_field_x_im[idx] = 0.0f;
    out_field_y_re[idx] = 0.0f;
    out_field_y_im[idx] = 0.0f;
    out_field_z_re[idx] = 0.0f;
    out_field_z_im[idx] = 0.0f;
    const int64_t vec = idx * 3;
    out_p0[vec + 0] = 0.0f;
    out_p0[vec + 1] = 0.0f;
    out_p0[vec + 2] = 0.0f;
    out_p1[vec + 0] = 0.0f;
    out_p1[vec + 1] = 0.0f;
    out_p1[vec + 2] = 0.0f;
    out_p2[vec + 0] = 0.0f;
    out_p2[vec + 1] = 0.0f;
    out_p2[vec + 2] = 0.0f;
}

} // namespace

void init_dfr_path_outputs_cuda(
    int64_t capacity,
    at::Tensor &out_count,
    at::Tensor &out_valid,
    at::Tensor &out_tx_id,
    at::Tensor &out_rx_id,
    at::Tensor &out_order,
    at::Tensor &out_edge0,
    at::Tensor &out_edge1,
    at::Tensor &out_edge2,
    at::Tensor &out_delay,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_p0,
    at::Tensor &out_p1,
    at::Tensor &out_p2) {
    c10::cuda::CUDAGuard guard(out_count.device());
    const int64_t launch_count = capacity > 0 ? capacity : 1;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((launch_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(out_count.get_device());
    init_dfr_path_outputs_kernel<<<blocks, threads, 0, stream>>>(
        capacity,
        out_count.data_ptr<int>(),
        reinterpret_cast<uint8_t *>(out_valid.data_ptr<bool>()),
        out_tx_id.data_ptr<int>(),
        out_rx_id.data_ptr<int>(),
        out_order.data_ptr<int>(),
        out_edge0.data_ptr<int>(),
        out_edge1.data_ptr<int>(),
        out_edge2.data_ptr<int>(),
        out_delay.data_ptr<float>(),
        out_field_x_re.data_ptr<float>(),
        out_field_x_im.data_ptr<float>(),
        out_field_y_re.data_ptr<float>(),
        out_field_y_im.data_ptr<float>(),
        out_field_z_re.data_ptr<float>(),
        out_field_z_im.data_ptr<float>(),
        out_p0.data_ptr<float>(),
        out_p1.data_ptr<float>(),
        out_p2.data_ptr<float>());
    cuda_check(cudaGetLastError(), "init_dfr_path_outputs_kernel");
}

} // namespace rayd::torch_backend


// ---- merged from src/diffraction/accum_reduce_part.cu ----

#include <rayd/torch/diffraction/accum_reduce.h>
#include <rayd/torch/runtime/optix_context.h>
#include <rayd/torch/diffraction/accum_params.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

namespace {

void require_i32_count(int64_t count, const char *name) {
    if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + ": count is outside int32 launch range.");
    }
}

__global__ void init_dfr_accum_outputs_kernel(DfrAccumInitArgs args, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (i < args.cell_count) {
#pragma unroll
            for (int field = 0; field < 7; ++field) {
                if (args.fields[field] != nullptr)
                    args.fields[field][i] = 0.0f;
            }
        }
        if (i < 7 && args.counters[i] != nullptr)
            *args.counters[i] = 0;
        if (i < args.state_count && args.state_prefix_depth != nullptr)
            args.state_prefix_depth[i] = 0;
        if (i < args.recursive_state_count && args.recursive_prefix_depth != nullptr)
            args.recursive_prefix_depth[i] = 0;
        if (i < args.launch_count) {
            if (args.temp_visibility != nullptr)
                args.temp_visibility[i] = 0u;
            if (args.tape_active != nullptr) {
                args.tape_active[i] = 0u;
                args.tape_state_idx[i] = -1;
                args.tape_cell[i] = -1;
                args.tape_material_idx[i] = -1;
                args.tape_edge_u[i] = 0.0f;
            }
            if (args.stage_cell != nullptr) {
                args.stage_cell[i] = -1;
                args.stage_value[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }
}

struct AddFloat4 {
    __host__ __device__ float4 operator()(float4 a, float4 b) const {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
};

struct AddCoherentValue {
    __host__ __device__ DfrCoherentStagedValue operator()(
        DfrCoherentStagedValue x,
        DfrCoherentStagedValue y) const {
        DfrCoherentStagedValue out;
        out.a = make_float4(
            x.a.x + y.a.x,
            x.a.y + y.a.y,
            x.a.z + y.a.z,
            x.a.w + y.a.w);
        out.b = make_float4(
            x.b.x + y.b.x,
            x.b.y + y.b.y,
            x.b.z + y.b.z,
            x.b.w + y.b.w);
        return out;
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

__global__ void scatter_dfr_coherent_accum_reduced_kernel(
    int cell_count,
    const int *__restrict__ num_runs,
    const int *__restrict__ unique_keys,
    const DfrCoherentStagedValue *__restrict__ reduced_values,
    float *__restrict__ out_direct_field_x_re,
    float *__restrict__ out_direct_field_x_im,
    float *__restrict__ out_direct_field_y_re,
    float *__restrict__ out_direct_field_y_im,
    float *__restrict__ out_direct_field_z_re,
    float *__restrict__ out_direct_field_z_im,
    float *__restrict__ out_multi_field_x_re,
    float *__restrict__ out_multi_field_x_im,
    float *__restrict__ out_multi_field_y_re,
    float *__restrict__ out_multi_field_y_im,
    float *__restrict__ out_multi_field_z_re,
    float *__restrict__ out_multi_field_z_im,
    int *__restrict__ out_direct_count,
    int *__restrict__ out_multi_count) {
    const int n = *num_runs;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        const int key = unique_keys[idx];
        if (key < 0) {
            continue;
        }
        const bool is_multi = key >= cell_count;
        const int cell = is_multi ? key - cell_count : key;
        if (cell < 0 || cell >= cell_count) {
            continue;
        }
        const DfrCoherentStagedValue value = reduced_values[idx];
        float *x_re = is_multi ? out_multi_field_x_re : out_direct_field_x_re;
        float *x_im = is_multi ? out_multi_field_x_im : out_direct_field_x_im;
        float *y_re = is_multi ? out_multi_field_y_re : out_direct_field_y_re;
        float *y_im = is_multi ? out_multi_field_y_im : out_direct_field_y_im;
        float *z_re = is_multi ? out_multi_field_z_re : out_direct_field_z_re;
        float *z_im = is_multi ? out_multi_field_z_im : out_direct_field_z_im;
        int *count_out = is_multi ? out_multi_count : out_direct_count;
        if (x_re != nullptr) atomicAdd(x_re + cell, value.a.x);
        if (x_im != nullptr) atomicAdd(x_im + cell, value.a.y);
        if (y_re != nullptr) atomicAdd(y_re + cell, value.a.z);
        if (y_im != nullptr) atomicAdd(y_im + cell, value.a.w);
        if (z_re != nullptr) atomicAdd(z_re + cell, value.b.x);
        if (z_im != nullptr) atomicAdd(z_im + cell, value.b.y);
        const int count = static_cast<int>(value.b.z + 0.5f);
        if (count_out != nullptr && count != 0) {
            atomicAdd(count_out + cell, count);
        }
    }
}

} // namespace

void init_dfr_accum_outputs_cuda(const DfrAccumInitArgs &args, cudaStream_t stream) {
    int n = 7;
    n = std::max(n, static_cast<int>(args.cell_count));
    n = std::max(n, static_cast<int>(args.launch_count));
    n = std::max(n, static_cast<int>(args.state_count));
    n = std::max(n, static_cast<int>(args.recursive_state_count));
    if (n <= 0)
        return;
    const int block = 256;
    const int grid = std::min((n + block - 1) / block, 4096);
    init_dfr_accum_outputs_kernel<<<grid, block, 0, stream>>>(args, n);
    cuda_check(cudaGetLastError(), "init_dfr_accum_outputs_kernel");
}

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

void reduce_dfr_coherent_accum_staged_cuda(
    int64_t sample_count,
    int64_t cell_count,
    const at::Tensor &stage_key,
    const at::Tensor &stage_value,
    at::Tensor &out_direct_field_x_re,
    at::Tensor &out_direct_field_x_im,
    at::Tensor &out_direct_field_y_re,
    at::Tensor &out_direct_field_y_im,
    at::Tensor &out_direct_field_z_re,
    at::Tensor &out_direct_field_z_im,
    at::Tensor &out_multi_field_x_re,
    at::Tensor &out_multi_field_x_im,
    at::Tensor &out_multi_field_y_re,
    at::Tensor &out_multi_field_y_im,
    at::Tensor &out_multi_field_z_re,
    at::Tensor &out_multi_field_z_im,
    at::Tensor &out_direct_count,
    at::Tensor &out_multi_count) {
    require_i32_count(sample_count, "reduce_dfr_coherent_accum_staged_cuda(sample_count)");
    require_i32_count(cell_count, "reduce_dfr_coherent_accum_staged_cuda(cell_count)");
    if (sample_count == 0) {
        return;
    }

    const int sample_count_i = static_cast<int>(sample_count);
    const int cell_count_i = static_cast<int>(cell_count);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cudaStream_t stream = torch_ctx.stream;
    at::TensorOptions key_options = stage_key.options();
    at::TensorOptions value_options = stage_value.options().dtype(at::kFloat);
    at::TensorOptions byte_options = stage_key.options().dtype(at::kByte);

    at::Tensor sorted_keys = at::empty({sample_count}, key_options);
    at::Tensor sorted_values = at::empty({sample_count, 8}, value_options);
    auto *values_in =
        reinterpret_cast<DfrCoherentStagedValue *>(stage_value.data_ptr<float>());
    auto *values_sorted =
        reinterpret_cast<DfrCoherentStagedValue *>(sorted_values.data_ptr<float>());

    size_t sort_temp_bytes = 0;
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_temp_bytes,
            stage_key.data_ptr<int>(),
            sorted_keys.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(dfr coherent accum size)");
    at::Tensor sort_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(sort_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr<uint8_t>(),
            sort_temp_bytes,
            stage_key.data_ptr<int>(),
            sorted_keys.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(dfr coherent accum)");

    at::Tensor unique_keys = at::empty({sample_count}, key_options);
    at::Tensor reduced_values = at::empty({sample_count, 8}, value_options);
    at::Tensor num_runs = at::empty({1}, key_options);
    auto *reduced_values_ptr =
        reinterpret_cast<DfrCoherentStagedValue *>(reduced_values.data_ptr<float>());

    size_t reduce_temp_bytes = 0;
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            nullptr,
            reduce_temp_bytes,
            sorted_keys.data_ptr<int>(),
            unique_keys.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddCoherentValue{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(dfr coherent accum size)");
    at::Tensor reduce_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(reduce_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            reduce_temp.data_ptr<uint8_t>(),
            reduce_temp_bytes,
            sorted_keys.data_ptr<int>(),
            unique_keys.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddCoherentValue{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(dfr coherent accum)");

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((sample_count + block_size - 1) / block_size);
    scatter_dfr_coherent_accum_reduced_kernel<<<block_count, block_size, 0, stream>>>(
        cell_count_i,
        num_runs.data_ptr<int>(),
        unique_keys.data_ptr<int>(),
        reduced_values_ptr,
        out_direct_field_x_re.data_ptr<float>(),
        out_direct_field_x_im.data_ptr<float>(),
        out_direct_field_y_re.data_ptr<float>(),
        out_direct_field_y_im.data_ptr<float>(),
        out_direct_field_z_re.data_ptr<float>(),
        out_direct_field_z_im.data_ptr<float>(),
        out_multi_field_x_re.data_ptr<float>(),
        out_multi_field_x_im.data_ptr<float>(),
        out_multi_field_y_re.data_ptr<float>(),
        out_multi_field_y_im.data_ptr<float>(),
        out_multi_field_z_re.data_ptr<float>(),
        out_multi_field_z_im.data_ptr<float>(),
        out_direct_count.data_ptr<int>(),
        out_multi_count.data_ptr<int>());
    cuda_check(cudaGetLastError(), "scatter_dfr_coherent_accum_reduced_kernel");
}

} // namespace rayd::torch_backend


// ---- merged from src/diffraction/accum_ad_part.cu ----

#include <rayd/torch/diffraction/accum_ad.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <rayd/torch/math.cuh>
#include <rayd/torch/native_compat.h>

namespace rayd::torch_backend {

namespace {

constexpr float kDfrEps = 1e-6f;

static __forceinline__ __device__ float read_f32_strided_or_zero(
    const float *ptr,
    int stride,
    int index) {
    return ptr != nullptr ? ptr[index * stride] : 0.f;
}

static __forceinline__ __device__ int read_i32_strided_or_default(
    const int *ptr,
    int stride,
    int index,
    int default_value) {
    return ptr != nullptr ? ptr[index * stride] : default_value;
}

static __forceinline__ __device__ bool read_u8_strided_or_false(
    const uint8_t *ptr,
    int stride,
    int index) {
    return ptr != nullptr && ptr[index * stride] != 0u;
}

static __forceinline__ __device__ float3 read_vec_strided_or_zero(
    const float *x,
    const float *y,
    const float *z,
    int stride,
    int index) {
    return make_f3(read_f32_strided_or_zero(x, stride, index),
                   read_f32_strided_or_zero(y, stride, index),
                   read_f32_strided_or_zero(z, stride, index));
}

static __forceinline__ __device__ float read_grid_or_zero(
    const float *ptr,
    int rank,
    int stride0,
    int stride1,
    int resolution0,
    int cell) {
    if (ptr == nullptr) {
        return 0.f;
    }
    if (rank == 2) {
        const int x = cell % resolution0;
        const int y = cell / resolution0;
        return ptr[y * stride0 + x * stride1];
    }
    return ptr[cell * stride0];
}

static __forceinline__ __device__ void atomic_add_strided(
    float *ptr,
    int stride,
    int index,
    float value) {
    if (ptr != nullptr) {
        atomicAdd(ptr + index * stride, value);
    }
}

static __forceinline__ __device__ void atomic_add_vec_strided(
    float *x,
    float *y,
    float *z,
    int stride,
    int index,
    float3 value) {
    atomic_add_strided(x, stride, index, value.x);
    atomic_add_strided(y, stride, index, value.y);
    atomic_add_strided(z, stride, index, value.z);
}

static __forceinline__ __device__ float material_gain_for_prim(
    const DfrDirectAccumADParams &params,
    int prim) {
    return params.material_gain != nullptr
               ? read_f32_strided_or_zero(params.material_gain, params.material_gain_stride, prim)
               : 1.f;
}

static __forceinline__ __device__ float material_gain_for_prim(
    const DfrChainAccumADParams &params,
    int prim) {
    return params.material_gain != nullptr
               ? read_f32_strided_or_zero(params.material_gain, params.material_gain_stride, prim)
               : 1.f;
}

// Nullable strided storage-access layer for the shared AD device body. Every
// macro expands to the exact pre-dedup Torch expression: reads go through the
// strided-or-default helpers above, gradient outputs carry their stride in a
// DfrGradSlot, and the dense tri gradients keep an inert stride of 1.
#define RAYD_DFR_AD_READ_F32(P, F, S, I) \
    read_f32_strided_or_zero((P).F, (P).S, (I))
#define RAYD_DFR_AD_READ_I32(P, F, S, I) \
    read_i32_strided_or_default((P).F, (P).S, (I), -1)
#define RAYD_DFR_AD_READ_I32_OR(P, F, S, I, D) \
    read_i32_strided_or_default((P).F, (P).S, (I), (D))
#define RAYD_DFR_AD_READ_VEC(P, X, Y, Z, S, I) \
    read_vec_strided_or_zero((P).X, (P).Y, (P).Z, (P).S, (I))
#define RAYD_DFR_AD_READ_VEC_GUARDED(P, X, Y, Z, S, I) \
    read_vec_strided_or_zero((P).X, (P).Y, (P).Z, (P).S, (I))
#define RAYD_DFR_AD_READ_DOT_F32(P, F, S, I) \
    read_f32_strided_or_zero((P).F, (P).S, (I))
#define RAYD_DFR_AD_READ_DOT_VEC(P, X, Y, Z, S, I) \
    read_vec_strided_or_zero((P).X, (P).Y, (P).Z, (P).S, (I))
#define RAYD_DFR_AD_MATERIAL_VALID_ENTRY(P, I) \
    read_u8_strided_or_false((P).material_valid, (P).material_valid_stride, (I))
#define RAYD_DFR_AD_MATERIAL_GAIN_TAPE(P, I) \
    read_f32_strided_or_zero((P).material_gain, (P).material_gain_stride, (I))
#define RAYD_DFR_AD_MATERIAL_GAIN_EVENT(P, I) material_gain_for_prim((P), (I))
#define RAYD_DFR_AD_MATERIAL_GAIN_OR_ONE(P, I) material_gain_for_prim((P), (I))
#define RAYD_DFR_AD_SUFFIX_FACE_PRIM(P, F, S, HAS_THIRD, SECOND, THIRD) \
    read_i32_strided_or_default((P).F, (P).S, (HAS_THIRD) ? (THIRD) : (SECOND), -1)

#include <rayd/shared/multipath/diffraction_accumulation_ad_device.cuh>

static __forceinline__ __device__ void add_chain_unit_vjp(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float grad_contribution,
    float *ptr,
    int stride,
    int index,
    const ChainTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = chain_contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index * stride, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void add_unit_vjp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution,
    float *ptr,
    int index,
    const DfrTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void add_unit_vjp_strided(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution,
    float *ptr,
    int stride,
    int index,
    const DfrTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index * stride, grad_contribution * partial);
    }
}

// The strided gradient-write layer: every unit-VJP call site expands to the
// pre-dedup Torch call, including the dense (stride-free for direct, literal-1
// stride for chain) tri gradient writes.
#define RAYD_DFR_AD_ADD_UNIT_VJP(P, PR, G, F, S, I, T) \
    add_unit_vjp_strided((P), (PR), (G), (P).F, (P).S, (I), (T))
#define RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(P, PR, G, F, I, T) \
    add_unit_vjp((P), (PR), (G), (P).F, (I), (T))
#define RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(P, PR, G, F, S, I, T) \
    add_chain_unit_vjp((P), (PR), (G), (P).F, (P).S, (I), (T))
#define RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(P, PR, G, F, I, T) \
    add_chain_unit_vjp((P), (PR), (G), (P).F, 1, (I), (T))

#include <rayd/shared/multipath/diffraction_accumulation_ad_vjp_device.cuh>

__global__ void dfr_direct_accum_jvp_kernel(DfrDirectAccumADParams params) {
    const int lane =
        params.lane_offset + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }
    const float dot_contribution = direct_jvp(params, p);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kDfrEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_direct_accum_vjp_kernel(DfrDirectAccumADParams params) {
    const int lane =
        params.lane_offset + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }

    float grad_contribution =
        read_grid_or_zero(params.grad_out_power,
                          params.grad_out_power_rank,
                          params.grad_out_power_stride0,
                          params.grad_out_power_stride1,
                          params.grid_resolution0,
                          p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kDfrEps) {
        grad_contribution +=
            read_grid_or_zero(params.grad_out_field_x_re,
                              params.grad_out_field_x_re_rank,
                              params.grad_out_field_x_re_stride0,
                              params.grad_out_field_x_re_stride1,
                              params.grid_resolution0,
                              p.cell) *
            0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }

    if (p.is_keller || p.is_suffix) {
        vjp_by_unit_jvps(params, p, grad_contribution);
        return;
    }

    const float grad_src_power = grad_contribution * p.common_no_src;
    atomic_add_strided(params.grad_state_src_power,
                       params.grad_state_src_power_stride,
                       p.state_idx,
                       grad_src_power);

    if (p.material_active && p.material_idx >= 0) {
        const float grad_gain =
            grad_contribution * p.contribution / fmaxf(p.material_gain, kDfrEps);
        atomic_add_strided(params.grad_material_gain,
                           params.grad_material_gain_stride,
                           p.material_idx,
                           grad_gain);
    }

    float grad_edge_length = 0.f;
    if (p.edge_length_active && p.edge_length > kDfrEps) {
        grad_edge_length = grad_contribution * p.contribution / p.edge_length;
    }
    if (p.wedge_active) {
        const float grad_wedge =
            grad_contribution * p.contribution / fmaxf(p.wedge_scale, kDfrEps);
        atomic_add_strided(params.grad_state_exterior_angle,
                           params.grad_state_exterior_angle_stride,
                           p.state_idx,
                           grad_wedge / (2.f * kPi));
    }

    const float3 source_delta = p.edge_point - p.source;
    const float3 target_delta = p.target - p.edge_point;
    const float3 d_contribution_d_edge =
        p.contribution *
        ((-2.f / p.source_dist2) * source_delta +
         (2.f / p.target_dist2) * target_delta);
    const float3 d_contribution_d_source =
        p.contribution * ((2.f / p.source_dist2) * source_delta);

    const float3 grad_edge_point = grad_contribution * d_contribution_d_edge;
    const float3 grad_source = grad_contribution * d_contribution_d_source;
    atomic_add_vec_strided(params.grad_state_src_x,
                           params.grad_state_src_y,
                           params.grad_state_src_z,
                           params.grad_state_src_stride,
                           p.state_idx,
                           grad_source);
    atomic_add_vec_strided(params.grad_state_edge_pos_x,
                           params.grad_state_edge_pos_y,
                           params.grad_state_edge_pos_z,
                           params.grad_state_edge_pos_stride,
                           p.state_idx,
                           grad_edge_point);

    const float grad_edge_t = dot3(grad_edge_point, p.edge_dir);
    atomic_add_strided(params.grad_state_edge_t_min,
                       params.grad_state_edge_t_min_stride,
                       p.state_idx,
                       (1.f - p.edge_u) * grad_edge_t - grad_edge_length);
    atomic_add_strided(params.grad_state_edge_t_max,
                       params.grad_state_edge_t_max_stride,
                       p.state_idx,
                       p.edge_u * grad_edge_t + grad_edge_length);

    const float3 grad_edge_dir = p.edge_t * grad_edge_point;
    const float3 grad_edge_dir_raw =
        (1.f / p.edge_dir_norm) *
        (grad_edge_dir - dot3(p.edge_dir, grad_edge_dir) * p.edge_dir);
    atomic_add_vec_strided(params.grad_state_edge_dir_x,
                           params.grad_state_edge_dir_y,
                           params.grad_state_edge_dir_z,
                           params.grad_state_edge_dir_stride,
                           p.state_idx,
                           grad_edge_dir_raw);
}

__global__ void dfr_chain_accum_jvp_kernel(DfrChainAccumADParams params) {
    const int lane =
        params.lane_offset + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    ChainPrimal p;
    if (!load_chain_primal(params, lane, p)) {
        return;
    }
    const ChainTangent tangent = chain_read_tangent(params, p);
    const float dot_contribution = chain_contribution_jvp(params, p, tangent);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kDfrEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_chain_accum_vjp_kernel(DfrChainAccumADParams params) {
    const int lane =
        params.lane_offset + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    ChainPrimal p;
    if (!load_chain_primal(params, lane, p)) {
        return;
    }
    float grad_contribution =
        read_grid_or_zero(params.grad_out_power,
                          params.grad_out_power_rank,
                          params.grad_out_power_stride0,
                          params.grad_out_power_stride1,
                          params.grid_resolution0,
                          p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kDfrEps) {
        grad_contribution +=
            read_grid_or_zero(params.grad_out_field_x_re,
                              params.grad_out_field_x_re_rank,
                              params.grad_out_field_x_re_stride0,
                              params.grad_out_field_x_re_stride1,
                              params.grid_resolution0,
                              p.cell) *
            0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }
    chain_vjp_by_unit_jvps(params, p, grad_contribution);
}

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

template <typename Params, typename Kernel>
void launch_ad_kernel(const char *name,
                      Kernel kernel,
                      const Params &params) {
    // The replay window is [lane_offset, n_rays); one lane per tape row.
    const int lane_count = params.n_rays - params.lane_offset;
    if (lane_count <= 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    const int block_size = 128;
    const int block_count = (lane_count + block_size - 1) / block_size;
    audit_cuda_kernel_launch(name,
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(lane_count));
    kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error("dfr_direct_accum_ad_gpu(): failed to launch kernel");
}

} // namespace

void dfr_direct_accum_jvp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_jvp_kernel",
                     dfr_direct_accum_jvp_kernel,
                     params);
}

void dfr_direct_accum_vjp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_vjp_kernel",
                     dfr_direct_accum_vjp_kernel,
                     params);
}

void dfr_chain_accum_jvp_gpu(const DfrChainAccumADParams &params) {
    launch_ad_kernel("dfr_chain_accum_jvp_kernel",
                     dfr_chain_accum_jvp_kernel,
                     params);
}

void dfr_chain_accum_vjp_gpu(const DfrChainAccumADParams &params) {
    launch_ad_kernel("dfr_chain_accum_vjp_kernel",
                     dfr_chain_accum_vjp_kernel,
                     params);
}

} // namespace rayd::torch_backend
