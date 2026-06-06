#pragma once

#include <ATen/ATen.h>

#include <raydtorch/scene_cache.h>

namespace raydtorch {

struct VisibilityForwardOutputs {
    at::Tensor visible;
    at::Tensor tape_prim_id;
    at::Tensor tape_t;
};

VisibilityForwardOutputs visibility_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &start,
    const at::Tensor &end,
    const at::Tensor &active);

struct ReflectionBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
    at::Tensor grad_ray_tmax;
};

ReflectionBackwardOutputs reflection_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t);

struct ReflectionJvpOutputs {
    at::Tensor tangent_t;
    at::Tensor tangent_image_sources;
};

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
    const at::Tensor &image_sources);

} // namespace raydtorch
