#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <vector>

namespace rayd::torch_backend {

/// Batched segment-pair visibility. ignore_prim_ids contains global primitive
/// IDs flattened as [ray_count, ignore_k]. Returns {visible_a, visible_b}.
std::vector<at::Tensor> visible_pair_forward_impl(
    std::int64_t scene_handle,
    at::Tensor start,
    at::Tensor end_a,
    at::Tensor end_b,
    c10::optional<at::Tensor> ignore_prim_ids,
    c10::optional<at::Tensor> active);

/// Batched source-to-axial-edge visibility. sample_fractions are embedded in
/// the launch parameters and may contain at most 16 values. Returns
/// {any_visible}.
std::vector<at::Tensor> visible_edge_forward_impl(
    std::int64_t scene_handle,
    at::Tensor source,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    std::vector<double> sample_fractions,
    c10::optional<at::Tensor> active);

/// Batched chain visibility. points is [chain_count, max_points, 3], while
/// ignore_prim_per_segment contains global primitive IDs flattened as
/// [chain_count, max_points - 1, ignore_k]. Returns
/// {all_visible, first_blocked_segment, first_blocked_prim}.
std::vector<at::Tensor> visible_chain_forward_impl(
    std::int64_t scene_handle,
    at::Tensor points,
    at::Tensor chain_length,
    c10::optional<at::Tensor> ignore_prim_per_segment,
    c10::optional<at::Tensor> active);

} // namespace rayd::torch_backend
