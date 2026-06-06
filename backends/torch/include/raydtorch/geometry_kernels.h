#pragma once

#include <ATen/ATen.h>

namespace raydtorch {

struct IntersectForwardOutputs {
    at::Tensor t;
    at::Tensor p;
    at::Tensor n;
    at::Tensor geo_n;
    at::Tensor uv;
    at::Tensor barycentric;
    at::Tensor shape_id;
    at::Tensor prim_id;
    at::Tensor local_prim_id;
    at::Tensor global_prim_id;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    at::Tensor tape_t;
};

IntersectForwardOutputs intersect_forward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active);

struct IntersectBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
    at::Tensor grad_ray_tmax;
};

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
    const at::Tensor &grad_barycentric);

} // namespace raydtorch
