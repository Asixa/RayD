#include <rayd/torch/common/camera.h>
#include <rayd/torch/common/camera_kernels.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <stdexcept>
#include <string>

namespace rayd::torch_backend {
namespace {

void check_cuda(cudaError_t result, const char *expr) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

cudaStream_t stream_for(const at::Tensor &tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.get_device());
}

} // namespace

at::Tensor camera_sample_to_world_cuda(const at::Tensor &sample, double tan_x, double tan_y, double depth) {
    c10::cuda::CUDAGuard guard(sample.device());
    at::Tensor out = at::empty({sample.size(0), 3}, sample.options());
    camera_detail::launch_1d(stream_for(sample), sample.size(0), camera_detail::sample_to_world_kernel,
        sample.data_ptr<float>(), sample.stride(0), sample.stride(1), out.data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y), static_cast<float>(depth));
    check_cuda(cudaGetLastError(), "sample_to_world_kernel");
    return out;
}

at::Tensor camera_sample_to_world_backward_cuda(const at::Tensor &grad, int64_t count, double tan_x, double tan_y, double depth) {
    c10::cuda::CUDAGuard guard(grad.device());
    at::Tensor out = at::empty({count, 2}, grad.options());
    camera_detail::launch_1d(stream_for(grad), count, camera_detail::sample_to_world_backward_kernel,
        grad.data_ptr<float>(), grad.stride(0), grad.stride(1), out.data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y), static_cast<float>(depth));
    check_cuda(cudaGetLastError(), "sample_to_world_backward_kernel");
    return out;
}

at::Tensor camera_world_to_sample_cuda(const at::Tensor &point, double tan_x, double tan_y) {
    c10::cuda::CUDAGuard guard(point.device());
    at::Tensor out = at::empty({point.size(0), 2}, point.options());
    camera_detail::launch_1d(stream_for(point), point.size(0), camera_detail::world_to_sample_kernel,
        point.data_ptr<float>(), point.stride(0), point.stride(1), out.data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_cuda(cudaGetLastError(), "world_to_sample_kernel");
    return out;
}

at::Tensor camera_world_to_sample_backward_cuda(const at::Tensor &point, const at::Tensor &grad, double tan_x, double tan_y) {
    c10::cuda::CUDAGuard guard(point.device());
    at::Tensor out = at::empty({point.size(0), 3}, point.options());
    camera_detail::launch_1d(stream_for(point), point.size(0), camera_detail::world_to_sample_backward_kernel,
        point.data_ptr<float>(), point.stride(0), point.stride(1), grad.data_ptr<float>(),
        grad.stride(0), grad.stride(1), out.data_ptr<float>(), static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_cuda(cudaGetLastError(), "world_to_sample_backward_kernel");
    return out;
}

std::tuple<at::Tensor, at::Tensor> camera_sample_ray_cuda(const at::Tensor &sample, double tan_x, double tan_y) {
    c10::cuda::CUDAGuard guard(sample.device());
    at::Tensor origin = at::empty({sample.size(0), 3}, sample.options());
    at::Tensor direction = at::empty({sample.size(0), 3}, sample.options());
    camera_detail::launch_1d(stream_for(sample), sample.size(0), camera_detail::sample_ray_kernel,
        sample.data_ptr<float>(), sample.stride(0), sample.stride(1), origin.data_ptr<float>(),
        direction.data_ptr<float>(), static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_cuda(cudaGetLastError(), "sample_ray_kernel");
    return {origin, direction};
}

at::Tensor camera_sample_ray_backward_cuda(const at::Tensor &sample, const at::Tensor *grad, double tan_x, double tan_y) {
    c10::cuda::CUDAGuard guard(sample.device());
    at::Tensor out = at::empty({sample.size(0), 2}, sample.options());
    const float *grad_ptr = grad == nullptr || grad->numel() == 0 ? nullptr : grad->data_ptr<float>();
    camera_detail::launch_1d(stream_for(sample), sample.size(0), camera_detail::sample_ray_backward_kernel,
        sample.data_ptr<float>(), sample.stride(0), sample.stride(1), grad_ptr,
        grad_ptr ? grad->stride(0) : 0, grad_ptr ? grad->stride(1) : 0, out.data_ptr<float>(),
        static_cast<float>(tan_x), static_cast<float>(tan_y));
    check_cuda(cudaGetLastError(), "sample_ray_backward_kernel");
    return out;
}

} // namespace rayd::torch_backend
