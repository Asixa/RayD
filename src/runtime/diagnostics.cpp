#include <rayd/torch/runtime/diagnostics.h>
#include <rayd/torch/bindings/tensor_contract.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace rayd::torch_backend {

py::tuple reflection_trace_stats_op(at::Tensor valid, at::Tensor t) {
    require_cuda(valid, "valid");
    require_dtype(valid, at::kBool, "valid");
    require_contiguous(valid, "valid");
    require_rank(valid, 2, "valid");
    require_cuda(t, "t");
    require_dtype(t, at::kFloat, "t");
    require_contiguous(t, "t");
    require_rank(t, 2, "t");
    if (t.size(0) != valid.size(0) || t.size(1) != valid.size(1)) {
        throw std::runtime_error("t must have the same shape as valid.");
    }
    if (t.get_device() != valid.get_device()) {
        throw std::runtime_error("t must be on the same CUDA device as valid.");
    }
    c10::cuda::CUDAGuard guard(static_cast<int>(valid.get_device()));
    auto [counts, checksum] = reflection_trace_stats_cuda(valid, t);
    return py::make_tuple(counts, checksum);
}

py::tuple diffraction_path_stats_op(at::Tensor count, at::Tensor valid, at::Tensor delay) {
    require_cuda(count, "count");
    require_dtype(count, at::kInt, "count");
    require_contiguous(count, "count");
    require_rank(count, 1, "count");
    if (count.numel() == 0) {
        throw std::runtime_error("count must not be empty.");
    }
    require_cuda(valid, "valid");
    require_dtype(valid, at::kBool, "valid");
    require_contiguous(valid, "valid");
    require_rank(valid, 1, "valid");
    require_cuda(delay, "delay");
    require_dtype(delay, at::kFloat, "delay");
    require_contiguous(delay, "delay");
    require_rank(delay, 1, "delay");
    if (delay.size(0) != valid.size(0)) {
        throw std::runtime_error("delay must have the same length as valid.");
    }
    if (count.get_device() != valid.get_device()) {
        throw std::runtime_error("count must be on the same CUDA device as valid.");
    }
    if (delay.get_device() != valid.get_device()) {
        throw std::runtime_error("delay must be on the same CUDA device as valid.");
    }
    c10::cuda::CUDAGuard guard(static_cast<int>(valid.get_device()));
    auto [valid_count, checksum] = diffraction_path_stats_cuda(count, valid, delay);
    return py::make_tuple(valid_count, checksum);
}

py::tuple default_dfr_material_op(int64_t count, at::Tensor like) {
    if (count < 0) {
        throw std::runtime_error("count must be non-negative.");
    }
    require_cuda(like, "like");
    require_dtype(like, at::kFloat, "like");
    c10::cuda::CUDAGuard guard(static_cast<int>(like.get_device()));
    auto [eta_r, sigma, mu_r, gain, valid] = default_dfr_material_cuda(count, like);
    return py::make_tuple(eta_r, sigma, mu_r, gain, valid);
}

at::Tensor intersection_valid_op(at::Tensor t, at::Tensor shape_id) {
    require_cuda(t, "t");
    require_dtype(t, at::kFloat, "t");
    require_rank(t, 1, "t");
    require_cuda(shape_id, "shape_id");
    require_dtype(shape_id, at::kInt, "shape_id");
    require_rank(shape_id, 1, "shape_id");
    if (shape_id.numel() != 0) {
        if (shape_id.size(0) != t.size(0)) {
            throw std::runtime_error("shape_id must be empty or match t length.");
        }
        if (shape_id.get_device() != t.get_device()) {
            throw std::runtime_error("shape_id must be on the same CUDA device as t.");
        }
    }
    c10::cuda::CUDAGuard guard(static_cast<int>(t.get_device()));
    return intersection_valid_cuda(t, shape_id);
}

} // namespace rayd::torch_backend
