#include <raydtorch/geometry_kernels.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace raydtorch {

namespace {

__global__ void intersect_backward_kernel(
    const int *__restrict__ faces,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    const float *__restrict__ grad_t,
    const float *__restrict__ grad_p,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d,
    float *__restrict__ grad_ray_tmax) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    grad_ray_tmax[ray_idx] = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        grad_ray_o[ray_idx * 3 + axis] = 0.f;
        grad_ray_d[ray_idx * 3 + axis] = 0.f;
    }
    if (!active[ray_idx])
        return;
    const int prim_id = tape_prim_id[ray_idx];
    if (prim_id < 0)
        return;

    const float dz = ray_d[ray_idx * 3 + 2];
    const float safe_dz = fabsf(dz) > 1e-8f ? dz : copysignf(1e-8f, dz == 0.f ? 1.f : dz);
    const float gt = grad_t[ray_idx];
    const float b0 = tape_bary[ray_idx * 3 + 0];
    const float b1 = tape_bary[ray_idx * 3 + 1];
    const float b2 = tape_bary[ray_idx * 3 + 2];
    const int i0 = faces[prim_id * 3 + 0];
    const int i1 = faces[prim_id * 3 + 1];
    const int i2 = faces[prim_id * 3 + 2];

    atomicAdd(&grad_vertices[i0 * 3 + 2], gt * b0 / safe_dz);
    atomicAdd(&grad_vertices[i1 * 3 + 2], gt * b1 / safe_dz);
    atomicAdd(&grad_vertices[i2 * 3 + 2], gt * b2 / safe_dz);
    grad_ray_o[ray_idx * 3 + 2] += -gt / safe_dz;

    for (int axis = 0; axis < 3; ++axis) {
        const float gp = grad_p[ray_idx * 3 + axis];
        grad_ray_o[ray_idx * 3 + axis] += gp;
    }
}

} // namespace

IntersectBackwardOutputs intersect_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t,
    const at::Tensor &grad_p,
    const at::Tensor &grad_barycentric) {
    (void)ray_o;
    (void)ray_tmax;
    (void)grad_barycentric;
    const int64_t ray_count = ray_d.size(0);
    IntersectBackwardOutputs out;
    out.grad_vertices = at::zeros_like(vertices);
    out.grad_ray_o = at::zeros_like(ray_d);
    out.grad_ray_d = at::zeros_like(ray_d);
    out.grad_ray_tmax = at::zeros({ray_count}, ray_d.options());

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    intersect_backward_kernel<<<blocks, threads, 0, stream>>>(
        faces.data_ptr<int>(),
        ray_d.data_ptr<float>(),
        active.data_ptr<bool>(),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        grad_t.data_ptr<float>(),
        grad_p.data_ptr<float>(),
        ray_count,
        out.grad_vertices.data_ptr<float>(),
        out.grad_ray_o.data_ptr<float>(),
        out.grad_ray_d.data_ptr<float>(),
        out.grad_ray_tmax.data_ptr<float>());
    return out;
}

} // namespace raydtorch
