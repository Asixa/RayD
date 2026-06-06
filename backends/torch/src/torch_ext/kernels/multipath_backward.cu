#include <raydtorch/geometry_kernels.h>
#include <raydtorch/multipath_kernels.h>

namespace raydtorch {

ReflectionBackwardOutputs reflection_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t) {
    const int64_t ray_count = ray_o.size(0);
    at::Tensor zeros_vec3 = at::zeros({ray_count, 3}, ray_o.options());
    at::Tensor zeros_uv = at::zeros({ray_count, 2}, ray_o.options());
    at::Tensor zeros_bary = at::zeros({ray_count, 3}, ray_o.options());
    IntersectBackwardOutputs hit_grad = intersect_backward_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t.reshape({ray_count}).contiguous(),
        zeros_vec3,
        zeros_vec3,
        zeros_vec3,
        zeros_uv,
        zeros_bary);
    return {
        hit_grad.grad_vertices,
        hit_grad.grad_ray_o,
        hit_grad.grad_ray_d,
        hit_grad.grad_ray_tmax,
    };
}

ReflectionJvpOutputs reflection_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_ray_o,
    const at::Tensor &tangent_ray_d,
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    IntersectJvpOutputs hit_jvp = intersect_jvp_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d);
    return {
        hit_jvp.tangent_t.reshape({ray_count, 1}),
        at::zeros_like(image_sources),
    };
}

} // namespace raydtorch
