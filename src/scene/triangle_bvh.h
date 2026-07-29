// Copyright Xingyu Chen.
// Declares internal scene support for triangle bvh.

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rayd::torch_backend {

struct SceneCache;

void compute_triangle_bvh_bounds_cuda(
    const SceneCache &scene,
    at::Tensor &primitive_min_x,
    at::Tensor &primitive_min_y,
    at::Tensor &primitive_min_z,
    at::Tensor &primitive_max_x,
    at::Tensor &primitive_max_y,
    at::Tensor &primitive_max_z,
    at::Tensor &packed_bounds,
    cudaStream_t stream);

void launch_intersect_cuda_bvh(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    at::Tensor &out_t,
    int *out_shape_id,
    int *out_local_prim_id,
    int *out_global_prim_id,
    float *out_bary_uv,
    cudaStream_t stream);

} // namespace rayd::torch_backend
