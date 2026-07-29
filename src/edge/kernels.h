// Copyright Xingyu Chen.
// Declares internal edge support for kernels.

#pragma once

#include <ATen/ATen.h>

#include <src/scene/cache.h>

namespace rayd::torch_backend {

struct EdgeForwardOutputs {
    at::Tensor distance;
    at::Tensor edge_point;
    at::Tensor edge_t;
    at::Tensor shape_id;
    at::Tensor edge_id;
    at::Tensor global_edge_id;
    at::Tensor tape_edge_id;
    at::Tensor tape_s;
    at::Tensor tape_d;
};

EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point);
EdgeForwardOutputs edge_forward_bvh_cuda(SceneCache &scene, const at::Tensor &point);

struct EdgeForwardPublicOutputs {
    at::Tensor distance;
    at::Tensor edge_point;
    at::Tensor edge_t;
    at::Tensor shape_id;
    at::Tensor edge_id;
    at::Tensor global_edge_id;
};

EdgeForwardPublicOutputs edge_forward_noad_cuda(const SceneCache &scene, const at::Tensor &point);
EdgeForwardPublicOutputs edge_forward_noad_bvh_cuda(SceneCache &scene, const at::Tensor &point);

struct EdgeTopKForwardOutputs {
    at::Tensor is_valid;
    at::Tensor distances;
    at::Tensor points;
    at::Tensor edge_t;
    at::Tensor edge_points;
    at::Tensor shape_ids;
    at::Tensor edge_ids;
    at::Tensor global_edge_ids;
    at::Tensor is_boundary;
    at::Tensor tape_edge_id;
    at::Tensor tape_s;
    at::Tensor tape_d;
};

EdgeTopKForwardOutputs edge_topk_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &point,
    int64_t k,
    const at::Tensor &active);

struct EdgeRayForwardOutputs {
    at::Tensor distance;
    at::Tensor ray_t;
    at::Tensor point;
    at::Tensor edge_t;
    at::Tensor edge_point;
    at::Tensor shape_id;
    at::Tensor edge_id;
    at::Tensor global_edge_id;
    at::Tensor tape_edge_id;
};

EdgeRayForwardOutputs edge_ray_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active);

EdgeRayForwardOutputs edge_ray_forward_bvh_cuda(
    SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active);

void compute_edge_optix_aabbs_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z,
    float radius,
    at::Tensor &out_aabbs);

struct EdgeBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_point;
};

EdgeBackwardOutputs edge_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &grad_distance,
    const at::Tensor &grad_edge_point,
    const at::Tensor &grad_edge_t);

EdgeBackwardOutputs edge_backward_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor *grad_distance,
    const at::Tensor *grad_edge_point,
    const at::Tensor *grad_edge_t,
    const at::Tensor *grad_edge_t_alias);

struct EdgeJvpOutputs {
    at::Tensor tangent_distance;
    at::Tensor tangent_edge_point;
    at::Tensor tangent_edge_t;
    at::Tensor tangent_tape_s;
    at::Tensor tangent_tape_d;
};

EdgeJvpOutputs edge_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_point);

EdgeJvpOutputs edge_jvp_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_point);

struct EdgeRayBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
};

EdgeRayBackwardOutputs edge_ray_backward_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &tape_edge_id,
    const at::Tensor &ray_t,
    const at::Tensor &edge_t,
    const at::Tensor *grad_distance,
    const at::Tensor *grad_ray_t,
    const at::Tensor *grad_point,
    const at::Tensor *grad_edge_t,
    const at::Tensor *grad_edge_point);

struct EdgeRayJvpOutputs {
    at::Tensor tangent_distance;
    at::Tensor tangent_ray_t;
    at::Tensor tangent_point;
    at::Tensor tangent_edge_t;
    at::Tensor tangent_edge_point;
};

EdgeRayJvpOutputs edge_ray_jvp_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &tape_edge_id,
    const at::Tensor &ray_t,
    const at::Tensor &edge_t,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d);

} // namespace rayd::torch_backend
