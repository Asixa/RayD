#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace raydtorch {

void reduce_dfr_accum_staged_cuda(
    int64_t sample_count,
    const at::Tensor &stage_cell,
    const at::Tensor &stage_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_direct_count,
    at::Tensor &out_keller_count,
    at::Tensor &out_edge_uses);

} // namespace raydtorch
