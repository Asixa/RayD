#pragma once

#include <ATen/ATen.h>

#include <raydtorch/scene_cache.h>

namespace raydtorch {

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

struct EdgeJvpOutputs {
    at::Tensor tangent_distance;
    at::Tensor tangent_edge_point;
    at::Tensor tangent_edge_t;
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

} // namespace raydtorch
