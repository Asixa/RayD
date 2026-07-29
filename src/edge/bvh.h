// Copyright Xingyu Chen.
// Declares internal edge support for bvh.

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#include <rayd/edge/bvh_types.h>

#include <cstddef>
#include <cstdint>

namespace rayd::torch_backend {

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

size_t edge_bvh_bounds_reduce_scratch_bytes(int64_t edge_count, cudaStream_t stream);
void reduce_edge_bvh_bounds_cuda(
    int64_t edge_count,
    const at::Tensor &packed_bounds,
    at::Tensor &out_bound,
    at::Tensor &scratch,
    cudaStream_t stream);

size_t edge_bvh_sort_scratch_bytes(int64_t edge_count, cudaStream_t stream);
void sort_edge_bvh_morton_cuda(
    int64_t edge_count,
    const at::Tensor &morton_codes_in,
    at::Tensor &morton_codes_out,
    const at::Tensor &primitive_ids_in,
    at::Tensor &primitive_ids_out,
    at::Tensor &scratch,
    cudaStream_t stream);

void encode_raw_edge_bvh_cuda(
    int64_t primitive_count,
    at::Tensor &left_child,
    at::Tensor &right_child,
    const at::Tensor &leaf_primitive,
    at::Tensor &leaf_primitives,
    cudaStream_t stream);

} // namespace rayd::torch_backend
