// Copyright Xingyu Chen.
// Declares the Torch penetration API.

#pragma once

#include <rayd/scene.h>

#include <cstdint>
#include <optional>

namespace rayd::torch {

enum class SegmentPenetrationPolicy : std::uint8_t {
    EnumeratedFullDistance = 0,
    MonteCarloTargetInset = 1,
};

// What a row whose ordered march finds more than `hit_capacity` crossings does to the batch.
// FailBatch ORs `failure_bit` and makes the whole batch inert. DropRow leaves only that row inert,
// flags it in `overflow`, and counts it in the optional `dropped_row_count`; the other rows stay valid.
enum class SegmentPenetrationOverflowPolicy : std::uint8_t {
    FailBatch = 0,
    DropRow = 1,
};

struct SegmentPenetrationRequest {
    const SceneResource& scene;
    at::Tensor origins;
    at::Tensor targets;
    std::optional<at::Tensor> input_active;
    bool input_active_any = true;
    std::int64_t hit_capacity = 0;
    SegmentPenetrationPolicy policy;
    double scene_diagonal = 0.0;
    at::Tensor capacity_failure_state;
    std::int32_t failure_bit = 0;
    SegmentPenetrationOverflowPolicy overflow_policy = SegmentPenetrationOverflowPolicy::FailBatch;
    // CUDA int32 [1] counter, used only with DropRow: every dropped row adds one. RayD never clears it.
    std::optional<at::Tensor> dropped_row_count;
};

struct SegmentPenetrationResult {
    at::Tensor valid;
    at::Tensor num_hits;
    at::Tensor reached_target;
    at::Tensor overflow;
    at::Tensor distance;
    at::Tensor direction;
    at::Tensor t;
    at::Tensor position;
    at::Tensor normal;
    at::Tensor geometric_normal;
    at::Tensor global_primitive_id;
};

struct SegmentPenetrationTapeResult {
    SegmentPenetrationResult result;
    at::Tensor tape_primitive_id;
    at::Tensor tape_barycentric;
    at::Tensor tape_restart_epsilon;
    at::Tensor tape_restart_branch;
    at::Tensor tape_restart_tie_mask;
    at::Tensor tape_direction_denominator_branch;
};

SegmentPenetrationResult segment_penetration_forward(const SegmentPenetrationRequest& request);
SegmentPenetrationTapeResult segment_penetration_forward_tape(const SegmentPenetrationRequest& request);

struct SegmentPenetrationBackwardRequest {
    const SegmentPenetrationRequest& primal;
    const SegmentPenetrationTapeResult& tape;
    std::optional<at::Tensor> grad_distance;
    std::optional<at::Tensor> grad_direction;
    std::optional<at::Tensor> grad_t;
    std::optional<at::Tensor> grad_position;
    std::optional<at::Tensor> grad_normal;
    std::optional<at::Tensor> grad_geometric_normal;
    bool need_grad_vertices = false;
    bool need_grad_origins = false;
    bool need_grad_targets = false;
};

struct SegmentPenetrationBackwardResult {
    at::Tensor grad_vertices;
    at::Tensor grad_origins;
    at::Tensor grad_targets;
};

SegmentPenetrationBackwardResult segment_penetration_backward(const SegmentPenetrationBackwardRequest& request);

struct SegmentPenetrationJvpRequest {
    const SegmentPenetrationRequest& primal;
    const SegmentPenetrationTapeResult& tape;
    std::optional<at::Tensor> tangent_vertices;
    std::optional<at::Tensor> tangent_origins;
    std::optional<at::Tensor> tangent_targets;
};

struct SegmentPenetrationJvpResult {
    at::Tensor tangent_distance;
    at::Tensor tangent_direction;
    at::Tensor tangent_t;
    at::Tensor tangent_position;
    at::Tensor tangent_normal;
    at::Tensor tangent_geometric_normal;
};

SegmentPenetrationJvpResult segment_penetration_jvp(const SegmentPenetrationJvpRequest& request);

} // namespace rayd::torch
