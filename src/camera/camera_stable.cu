#include <src/camera/camera_kernels.cuh>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <array>
#include <optional>
#include <tuple>

namespace rayd::torch_backend::stable {
using Tensor = torch::stable::Tensor;

namespace {

void check_tensor(const Tensor &tensor, const char *name, int64_t width) {
    STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Float, name, " must be float32");
    STD_TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == width, name, " must have shape [N, ", width, "]");
}

cudaStream_t current_stream(const Tensor &tensor) {
    void *stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream));
    return static_cast<cudaStream_t>(stream);
}

void check_launch(const char *name) {
    const cudaError_t error = cudaGetLastError();
    STD_TORCH_CHECK(error == cudaSuccess, "CUDA error in ", name, ": ", cudaGetErrorString(error));
}

Tensor output_like(const Tensor &input, int64_t width) {
    const std::array<int64_t, 2> shape{input.size(0), width};
    return torch::stable::new_empty(input, torch::headeronly::IntHeaderOnlyArrayRef(shape));
}

} // namespace

Tensor sample_to_world(const Tensor &sample, double tan_x, double tan_y, double depth) {
    check_tensor(sample, "sample", 2);
    const torch::stable::accelerator::DeviceGuard guard(sample.get_device_index());
    Tensor out = output_like(sample, 3);
    camera_detail::launch_1d(current_stream(sample), sample.size(0), camera_detail::sample_to_world_kernel,
        sample.const_data_ptr<float>(), sample.stride(0), sample.stride(1), out.mutable_data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y), static_cast<float>(depth));
    check_launch("sample_to_world_kernel");
    return out;
}

Tensor sample_to_world_backward(const Tensor &grad, int64_t count, double tan_x, double tan_y, double depth) {
    check_tensor(grad, "grad_world", 3);
    STD_TORCH_CHECK(count == grad.size(0), "sample_count must match grad_world.size(0)");
    const torch::stable::accelerator::DeviceGuard guard(grad.get_device_index());
    Tensor out = output_like(grad, 2);
    camera_detail::launch_1d(current_stream(grad), count, camera_detail::sample_to_world_backward_kernel,
        grad.const_data_ptr<float>(), grad.stride(0), grad.stride(1), out.mutable_data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y), static_cast<float>(depth));
    check_launch("sample_to_world_backward_kernel");
    return out;
}

Tensor world_to_sample(const Tensor &point, double tan_x, double tan_y) {
    check_tensor(point, "point", 3);
    const torch::stable::accelerator::DeviceGuard guard(point.get_device_index());
    Tensor out = output_like(point, 2);
    camera_detail::launch_1d(current_stream(point), point.size(0), camera_detail::world_to_sample_kernel,
        point.const_data_ptr<float>(), point.stride(0), point.stride(1), out.mutable_data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_launch("world_to_sample_kernel");
    return out;
}

Tensor world_to_sample_backward(const Tensor &point, const Tensor &grad, double tan_x, double tan_y) {
    check_tensor(point, "point", 3);
    check_tensor(grad, "grad_sample", 2);
    STD_TORCH_CHECK(point.size(0) == grad.size(0), "point and grad_sample batch sizes must match");
    const torch::stable::accelerator::DeviceGuard guard(point.get_device_index());
    Tensor out = output_like(point, 3);
    camera_detail::launch_1d(current_stream(point), point.size(0), camera_detail::world_to_sample_backward_kernel,
        point.const_data_ptr<float>(), point.stride(0), point.stride(1), grad.const_data_ptr<float>(),
        grad.stride(0), grad.stride(1), out.mutable_data_ptr<float>(), static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_launch("world_to_sample_backward_kernel");
    return out;
}

std::tuple<Tensor, Tensor> sample_ray(const Tensor &sample, double tan_x, double tan_y) {
    check_tensor(sample, "sample", 2);
    const torch::stable::accelerator::DeviceGuard guard(sample.get_device_index());
    Tensor origin = output_like(sample, 3);
    Tensor direction = output_like(sample, 3);
    camera_detail::launch_1d(current_stream(sample), sample.size(0), camera_detail::sample_ray_kernel,
        sample.const_data_ptr<float>(), sample.stride(0), sample.stride(1), origin.mutable_data_ptr<float>(),
        direction.mutable_data_ptr<float>(), static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_launch("sample_ray_kernel");
    return {origin, direction};
}

Tensor sample_ray_backward(const Tensor &sample, std::optional<Tensor> grad, double tan_x, double tan_y) {
    check_tensor(sample, "sample", 2);
    if (grad) check_tensor(*grad, "grad_direction", 3);
    const torch::stable::accelerator::DeviceGuard guard(sample.get_device_index());
    Tensor out = output_like(sample, 2);
    const float *grad_ptr = grad ? grad->const_data_ptr<float>() : nullptr;
    camera_detail::launch_1d(current_stream(sample), sample.size(0), camera_detail::sample_ray_backward_kernel,
        sample.const_data_ptr<float>(), sample.stride(0), sample.stride(1), grad_ptr,
        grad ? grad->stride(0) : 0, grad ? grad->stride(1) : 0, out.mutable_data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_launch("sample_ray_backward_kernel");
    return out;
}

} // namespace rayd::torch_backend::stable

STABLE_TORCH_LIBRARY(rayd_torch_stable, m) {
    m.def("camera_sample_to_world(Tensor sample, float tan_x, float tan_y, float depth) -> Tensor");
    m.def("camera_sample_to_world_backward(Tensor grad_world, int sample_count, float tan_x, float tan_y, float depth) -> Tensor");
    m.def("camera_world_to_sample(Tensor point, float tan_x, float tan_y) -> Tensor");
    m.def("camera_world_to_sample_backward(Tensor point, Tensor grad_sample, float tan_x, float tan_y) -> Tensor");
    m.def("camera_sample_ray(Tensor sample, float tan_x, float tan_y) -> (Tensor, Tensor)");
    m.def("camera_sample_ray_backward(Tensor sample, Tensor? grad_direction, float tan_x, float tan_y) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(rayd_torch_stable, CUDA, m) {
    m.impl("camera_sample_to_world", TORCH_BOX(&rayd::torch_backend::stable::sample_to_world));
    m.impl("camera_sample_to_world_backward", TORCH_BOX(&rayd::torch_backend::stable::sample_to_world_backward));
    m.impl("camera_world_to_sample", TORCH_BOX(&rayd::torch_backend::stable::world_to_sample));
    m.impl("camera_world_to_sample_backward", TORCH_BOX(&rayd::torch_backend::stable::world_to_sample_backward));
    m.impl("camera_sample_ray", TORCH_BOX(&rayd::torch_backend::stable::sample_ray));
    m.impl("camera_sample_ray_backward", TORCH_BOX(&rayd::torch_backend::stable::sample_ray_backward));
}
