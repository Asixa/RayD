#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>

#include <rayd/torch/runtime/optix_pipeline.h>
#include <rayd/torch/scene/cache.h>
#include <rayd/torch/integration.h>

namespace rayd::torch_backend {

struct SegmentPenetrationNativeTape {
    at::Tensor primitive_id;
    at::Tensor barycentric;
    at::Tensor restart_epsilon;
    at::Tensor restart_branch;
    at::Tensor restart_tie_mask;
    at::Tensor direction_denominator_branch;
};

struct SegmentPenetrationNativeOutputs {
    rayd::torch::SegmentPenetrationResult result;
    SegmentPenetrationNativeTape tape;
};

struct SegmentPenetrationBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_origins;
    at::Tensor grad_targets;
};

struct SegmentPenetrationJvpOutputs {
    at::Tensor tangent_distance;
    at::Tensor tangent_direction;
    at::Tensor tangent_t;
    at::Tensor tangent_position;
    at::Tensor tangent_normal;
    at::Tensor tangent_geometric_normal;
};

OptixPipelineConfig segment_penetration_pipeline_config();

void segment_penetration_initialize_cuda(
    SegmentPenetrationNativeOutputs &outputs,
    const at::Tensor *input_active,
    const at::Tensor &capacity_failure_state,
    std::int32_t failure_bit,
    bool input_active_any);

void segment_penetration_sanitize_cuda(
    SegmentPenetrationNativeOutputs &outputs,
    const at::Tensor &capacity_failure_state);

SegmentPenetrationBackwardOutputs segment_penetration_backward_cuda(
    const SceneCache &scene,
    const rayd::torch::SegmentPenetrationBackwardRequest &request);

SegmentPenetrationJvpOutputs segment_penetration_jvp_cuda(
    const SceneCache &scene,
    const rayd::torch::SegmentPenetrationJvpRequest &request);

} // namespace rayd::torch_backend
