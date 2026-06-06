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

struct ReflEpcBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_source;
    at::Tensor grad_receiver;
};

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
    const at::Tensor &grad_path_length);

struct ReflEpcJvpOutputs {
    at::Tensor tangent_field_real;
    at::Tensor tangent_field_imag;
    at::Tensor tangent_path_length;
};

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
    const at::Tensor &tangent_receiver);

struct DfrDirectForwardOutputs {
    at::Tensor power;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
};

DfrDirectForwardOutputs dfr_direct_forward_cuda(
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &src);

struct DfrDirectBackwardOutputs {
    at::Tensor grad_edge_pos;
    at::Tensor grad_edge_dir;
    at::Tensor grad_src;
};

DfrDirectBackwardOutputs dfr_direct_backward_cuda(
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &src,
    const at::Tensor &grad_power,
    const at::Tensor &grad_field_x_re,
    const at::Tensor &grad_field_x_im);

struct DfrDirectJvpOutputs {
    at::Tensor tangent_power;
    at::Tensor tangent_field_x_re;
    at::Tensor tangent_field_x_im;
};

DfrDirectJvpOutputs dfr_direct_jvp_cuda(
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &src,
    const at::Tensor &tangent_edge_pos,
    const at::Tensor &tangent_edge_dir,
    const at::Tensor &tangent_src);

} // namespace raydtorch
