// Copyright Xingyu Chen.
// Declares internal diffraction support for accum reduce.

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rayd::torch_backend {

// One fused launch that initializes every diffraction-accumulation output and
// temp buffer the forward op previously zero-filled with ~16 separate
// at::zeros / at::full kernels. Null pointers are skipped; counts are int32
// because the forward op already validates launch ranges.
struct DfrAccumInitArgs {
    int32_t cell_count = 0;
    int32_t launch_count = 0;
    int32_t state_count = 0;
    int32_t recursive_state_count = 0;
    float* fields[7] = {};                 // length cell_count, zeroed
    int* counters[7] = {};                 // length 1 each, zeroed
    int* state_prefix_depth = nullptr;     // length state_count, zeroed
    int* recursive_prefix_depth = nullptr; // length recursive_state_count, zeroed
    uint8_t* temp_visibility = nullptr;    // length launch_count, zeroed
    uint8_t* tape_active = nullptr;        // length launch_count, zeroed
    int* tape_state_idx = nullptr;         // length launch_count, -1
    int* tape_cell = nullptr;              // length launch_count, -1
    int* tape_material_idx = nullptr;      // length launch_count, -1
    float* tape_edge_u = nullptr;          // length launch_count, zeroed
    int* stage_cell = nullptr;             // length launch_count, -1
    float4* stage_value = nullptr;         // length launch_count, zeroed
};

void init_dfr_accum_outputs_cuda(const DfrAccumInitArgs& args, cudaStream_t stream);

void reduce_dfr_accum_staged_cuda(int64_t sample_count, const at::Tensor& stage_cell, const at::Tensor& stage_value,
                                  at::Tensor& out_power, at::Tensor& out_field_x_re, at::Tensor& out_direct_count,
                                  at::Tensor& out_keller_count, at::Tensor& out_edge_uses);

void reduce_dfr_coherent_accum_staged_cuda(int64_t sample_count, int64_t cell_count, const at::Tensor& stage_key,
                                           const at::Tensor& stage_value, at::Tensor& out_direct_field_x_re,
                                           at::Tensor& out_direct_field_x_im, at::Tensor& out_direct_field_y_re,
                                           at::Tensor& out_direct_field_y_im, at::Tensor& out_direct_field_z_re,
                                           at::Tensor& out_direct_field_z_im, at::Tensor& out_multi_field_x_re,
                                           at::Tensor& out_multi_field_x_im, at::Tensor& out_multi_field_y_re,
                                           at::Tensor& out_multi_field_y_im, at::Tensor& out_multi_field_z_re,
                                           at::Tensor& out_multi_field_z_im, at::Tensor& out_direct_count,
                                           at::Tensor& out_multi_count);

} // namespace rayd::torch_backend
