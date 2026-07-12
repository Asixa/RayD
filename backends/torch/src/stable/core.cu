#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <array>

namespace rayd::torch_backend::stable {
using Tensor = torch::stable::Tensor;

namespace {

__global__ void intersection_valid_from_t_kernel(
    const float *__restrict__ t,
    int64_t count,
    int64_t stride,
    bool *__restrict__ valid) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        valid[index] = isfinite(t[index * stride]);
}

__global__ void intersection_valid_from_shape_kernel(
    const int *__restrict__ shape_id,
    int64_t count,
    int64_t stride,
    bool *__restrict__ valid) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        valid[index] = shape_id[index * stride] >= 0;
}

void require_rank1_cuda(
    const Tensor &tensor,
    torch::headeronly::ScalarType dtype,
    const char *name) {
    STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    STD_TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    STD_TORCH_CHECK(tensor.dim() == 1, name, " must have rank 1");
}

cudaStream_t current_stream(const Tensor &tensor) {
    void *stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream));
    return static_cast<cudaStream_t>(stream);
}

} // namespace

Tensor intersection_valid(const Tensor &t, const Tensor &shape_id) {
    require_rank1_cuda(t, torch::headeronly::ScalarType::Float, "t");
    require_rank1_cuda(shape_id, torch::headeronly::ScalarType::Int, "shape_id");
    STD_TORCH_CHECK(
        shape_id.size(0) == 0 || shape_id.size(0) == t.size(0),
        "shape_id must be empty or match t length");
    STD_TORCH_CHECK(
        shape_id.size(0) == 0 ||
            shape_id.get_device_index() == t.get_device_index(),
        "shape_id must be on the same CUDA device as t");

    const torch::stable::accelerator::DeviceGuard guard(t.get_device_index());
    const std::array<int64_t, 1> shape{t.size(0)};
    Tensor valid = torch::stable::new_empty(
        t,
        torch::headeronly::IntHeaderOnlyArrayRef(shape),
        torch::headeronly::ScalarType::Bool);
    if (t.size(0) == 0)
        return valid;

    constexpr int threads = 256;
    const int blocks = static_cast<int>((t.size(0) + threads - 1) / threads);
    cudaStream_t stream = current_stream(t);
    if (shape_id.size(0) == t.size(0)) {
        intersection_valid_from_shape_kernel<<<blocks, threads, 0, stream>>>(
            shape_id.const_data_ptr<int>(),
            t.size(0),
            shape_id.stride(0),
            valid.mutable_data_ptr<bool>());
    } else {
        intersection_valid_from_t_kernel<<<blocks, threads, 0, stream>>>(
            t.const_data_ptr<float>(),
            t.size(0),
            t.stride(0),
            valid.mutable_data_ptr<bool>());
    }
    const cudaError_t error = cudaGetLastError();
    STD_TORCH_CHECK(
        error == cudaSuccess,
        "CUDA error in intersection_valid: ",
        cudaGetErrorString(error));
    return valid;
}

} // namespace rayd::torch_backend::stable

STABLE_TORCH_LIBRARY_FRAGMENT(rayd_torch_stable, m) {
    m.def("intersection_valid(Tensor t, Tensor shape_id) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(rayd_torch_stable, CUDA, m) {
    m.impl(
        "intersection_valid",
        TORCH_BOX(&rayd::torch_backend::stable::intersection_valid));
}
