#include <raydtorch/multipath_kernels.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace raydtorch {

namespace {

__global__ void dfr_direct_forward_kernel(
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ src,
    int64_t count,
    float *__restrict__ power,
    float *__restrict__ field_x_re,
    float *__restrict__ field_x_im) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float sx = src[idx * 3 + 0];
    const float sy = src[idx * 3 + 1];
    const float sz = src[idx * 3 + 2];
    const float ax = edge_pos[idx * 3 + 0] + edge_dir[idx * 3 + 0];
    const float ay = edge_pos[idx * 3 + 1] + edge_dir[idx * 3 + 1];
    const float az = edge_pos[idx * 3 + 2] + edge_dir[idx * 3 + 2];
    const float p = ax * sx + ay * sy + az * sz;
    power[idx] = p;
    field_x_re[idx] = p;
    field_x_im[idx] = 0.5f * p;
}

__global__ void dfr_direct_backward_kernel(
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ src,
    const float *__restrict__ grad_power,
    const float *__restrict__ grad_field_x_re,
    const float *__restrict__ grad_field_x_im,
    int64_t count,
    float *__restrict__ grad_edge_pos,
    float *__restrict__ grad_edge_dir,
    float *__restrict__ grad_src) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float total = grad_power[idx] + grad_field_x_re[idx] + 0.5f * grad_field_x_im[idx];
    for (int axis = 0; axis < 3; ++axis) {
        const float s = src[idx * 3 + axis];
        const float state = edge_pos[idx * 3 + axis] + edge_dir[idx * 3 + axis];
        grad_edge_pos[idx * 3 + axis] = total * s;
        grad_edge_dir[idx * 3 + axis] = total * s;
        grad_src[idx * 3 + axis] = total * state;
    }
}

__global__ void dfr_direct_jvp_kernel(
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ src,
    const float *__restrict__ tangent_edge_pos,
    const float *__restrict__ tangent_edge_dir,
    const float *__restrict__ tangent_src,
    int64_t count,
    float *__restrict__ tangent_power,
    float *__restrict__ tangent_field_x_re,
    float *__restrict__ tangent_field_x_im) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float tangent = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        const float state = edge_pos[idx * 3 + axis] + edge_dir[idx * 3 + axis];
        const float tangent_state =
            tangent_edge_pos[idx * 3 + axis] + tangent_edge_dir[idx * 3 + axis];
        tangent += tangent_state * src[idx * 3 + axis] + state * tangent_src[idx * 3 + axis];
    }
    tangent_power[idx] = tangent;
    tangent_field_x_re[idx] = tangent;
    tangent_field_x_im[idx] = 0.5f * tangent;
}

} // namespace

DfrDirectForwardOutputs dfr_direct_forward_cuda(
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &src) {
    const int64_t count = edge_pos.size(0);
    DfrDirectForwardOutputs out;
    out.power = at::empty({count}, edge_pos.options());
    out.field_x_re = at::empty({count}, edge_pos.options());
    out.field_x_im = at::empty({count}, edge_pos.options());
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
    dfr_direct_forward_kernel<<<blocks, threads, 0, stream>>>(
        edge_pos.data_ptr<float>(),
        edge_dir.data_ptr<float>(),
        src.data_ptr<float>(),
        count,
        out.power.data_ptr<float>(),
        out.field_x_re.data_ptr<float>(),
        out.field_x_im.data_ptr<float>());
    return out;
}

DfrDirectBackwardOutputs dfr_direct_backward_cuda(
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &src,
    const at::Tensor &grad_power,
    const at::Tensor &grad_field_x_re,
    const at::Tensor &grad_field_x_im) {
    const int64_t count = edge_pos.size(0);
    DfrDirectBackwardOutputs out;
    out.grad_edge_pos = at::empty_like(edge_pos);
    out.grad_edge_dir = at::empty_like(edge_dir);
    out.grad_src = at::empty_like(src);
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
    dfr_direct_backward_kernel<<<blocks, threads, 0, stream>>>(
        edge_pos.data_ptr<float>(),
        edge_dir.data_ptr<float>(),
        src.data_ptr<float>(),
        grad_power.data_ptr<float>(),
        grad_field_x_re.data_ptr<float>(),
        grad_field_x_im.data_ptr<float>(),
        count,
        out.grad_edge_pos.data_ptr<float>(),
        out.grad_edge_dir.data_ptr<float>(),
        out.grad_src.data_ptr<float>());
    return out;
}

DfrDirectJvpOutputs dfr_direct_jvp_cuda(
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &src,
    const at::Tensor &tangent_edge_pos,
    const at::Tensor &tangent_edge_dir,
    const at::Tensor &tangent_src) {
    const int64_t count = edge_pos.size(0);
    DfrDirectJvpOutputs out;
    out.tangent_power = at::empty({count}, edge_pos.options());
    out.tangent_field_x_re = at::empty({count}, edge_pos.options());
    out.tangent_field_x_im = at::empty({count}, edge_pos.options());
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
    dfr_direct_jvp_kernel<<<blocks, threads, 0, stream>>>(
        edge_pos.data_ptr<float>(),
        edge_dir.data_ptr<float>(),
        src.data_ptr<float>(),
        tangent_edge_pos.data_ptr<float>(),
        tangent_edge_dir.data_ptr<float>(),
        tangent_src.data_ptr<float>(),
        count,
        out.tangent_power.data_ptr<float>(),
        out.tangent_field_x_re.data_ptr<float>(),
        out.tangent_field_x_im.data_ptr<float>());
    return out;
}

} // namespace raydtorch
