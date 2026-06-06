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

ReflEpcBackwardOutputs refl_epc_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor &grad_field_real,
    const at::Tensor &grad_field_imag,
    const at::Tensor &grad_path_length) {
    const int64_t ray_count = source.size(0);
    at::Tensor ray_d = (receiver - source).contiguous();
    at::Tensor ray_tmax = at::ones({ray_count}, source.options());
    at::Tensor denom = 1.f + tape_t;
    at::Tensor inv_denom = 1.f / denom;
    at::Tensor real_dt =
        -at::sin(tape_t) * inv_denom - at::cos(tape_t) * inv_denom * inv_denom;
    at::Tensor imag_dt =
        at::cos(tape_t) * inv_denom - at::sin(tape_t) * inv_denom * inv_denom;
    at::Tensor grad_t =
        grad_path_length.reshape({ray_count}) +
        grad_field_real.reshape({ray_count}) * real_dt +
        grad_field_imag.reshape({ray_count}) * imag_dt;
    ReflectionBackwardOutputs hit_grad = reflection_backward_cuda(
        vertices,
        faces,
        source,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t.contiguous());
    return {
        hit_grad.grad_vertices,
        hit_grad.grad_ray_o - hit_grad.grad_ray_d,
        hit_grad.grad_ray_d,
    };
}

ReflEpcJvpOutputs refl_epc_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_source,
    const at::Tensor &tangent_receiver) {
    const int64_t ray_count = source.size(0);
    at::Tensor ray_d = (receiver - source).contiguous();
    at::Tensor tangent_ray_d = (tangent_receiver - tangent_source).contiguous();
    ReflectionJvpOutputs hit_jvp = reflection_jvp_cuda(
        vertices,
        faces,
        source,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_source,
        tangent_ray_d,
        at::zeros({ray_count, 1, 3}, source.options()));
    at::Tensor tangent_t = hit_jvp.tangent_t.reshape({ray_count});
    at::Tensor denom = 1.f + tape_t;
    at::Tensor inv_denom = 1.f / denom;
    at::Tensor real_dt =
        -at::sin(tape_t) * inv_denom - at::cos(tape_t) * inv_denom * inv_denom;
    at::Tensor imag_dt =
        at::cos(tape_t) * inv_denom - at::sin(tape_t) * inv_denom * inv_denom;
    return {
        (real_dt * tangent_t).contiguous(),
        (imag_dt * tangent_t).contiguous(),
        tangent_t.contiguous(),
    };
}

} // namespace raydtorch
