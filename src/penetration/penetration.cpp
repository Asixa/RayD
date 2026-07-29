// Copyright Xingyu Chen.
// Implements penetration support for penetration.

#include <src/penetration/segment_penetration_kernels.h>

#include <src/runtime/optix_context.h>
#include <src/bindings/tensor_contract.h>
#include <src/penetration/segment_penetration_params.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "../bindings/integration_internal.h"

namespace rayd::torch_backend {

namespace {

const at::Tensor* present(const std::optional<at::Tensor>& value) {
    if (!value.has_value())
        return nullptr;
    if (!value->defined())
        throw std::runtime_error("optional segment penetration tensors must be defined.");
    return &*value;
}

void require_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
    if (tensor.device() != reference.device())
        throw std::runtime_error(std::string(name) + " must share the segment CUDA device.");
}

void require_same_device(const at::Tensor& reference, const at::Tensor* tensor, const char* name) {
    if (tensor != nullptr)
        require_same_device(reference, *tensor, name);
}

void require_shape(const at::Tensor& tensor, at::IntArrayRef shape, const char* name) {
    if (!tensor.sizes().equals(shape))
        throw std::runtime_error(std::string(name) + " has the wrong shape.");
}

void require_contiguous_tensor(const at::Tensor& tensor, at::ScalarType dtype, at::IntArrayRef shape,
                               const at::Tensor& device_reference, const char* name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, dtype, name);
    require_shape(tensor, shape, name);
    require_same_device(device_reference, tensor, name);
}

void require_strided_float(const at::Tensor* tensor, at::IntArrayRef shape, const at::Tensor& device_reference,
                           const char* name) {
    if (tensor == nullptr)
        return;
    require_cuda(*tensor, name);
    require_dtype(*tensor, at::kFloat, name);
    require_shape(*tensor, shape, name);
    require_same_device(device_reference, *tensor, name);
}

std::int32_t checked_i32(std::int64_t value, const char* name) {
    if (value < 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error(std::string(name) + " does not fit in non-negative int32.");
    return static_cast<std::int32_t>(value);
}

void validate_failure_bit(std::int32_t failure_bit) {
    const std::uint32_t bits = static_cast<std::uint32_t>(failure_bit);
    if (bits == 0u || (bits & (bits - 1u)) != 0u)
        throw std::runtime_error("failure_bit must contain exactly one non-zero bit.");
}

struct ValidatedRequest {
    SceneCache& scene;
    const at::Tensor* input_active;
    std::int32_t segment_count;
    std::int32_t hit_capacity;
    std::int32_t policy;
    float scene_diagonal;
};

ValidatedRequest validate_request(const rayd::torch::SegmentPenetrationRequest& request) {
    SceneCache& scene = rayd::torch::detail::IntegrationAccess::scene_cache(request.scene);
    if (scene.trace_backend == TraceBackend::Cuda)
        throw std::runtime_error("ADR-0033 segment penetration is unsupported by the CUDA ray-tracing backend; "
                                 "select trace_backend='optix'.");
    require_vec3f(request.origins, "origins");
    require_vec3f(request.targets, "targets");
    if (request.targets.size(0) != request.origins.size(0))
        throw std::runtime_error("origins and targets must have the same segment count.");
    if (request.origins.get_device() != scene.device_index)
        throw std::runtime_error("origins must share the SceneResource CUDA device.");
    require_same_device(request.origins, request.targets, "targets");
    const at::Tensor* input_active = present(request.input_active);
    if (input_active != nullptr) {
        require_mask(*input_active, "input_active");
        require_shape(*input_active, {request.origins.size(0)}, "input_active");
        require_same_device(request.origins, *input_active, "input_active");
    }
    if (request.origins.size(0) > 0 && !request.input_active_any && input_active == nullptr) {
        throw std::runtime_error("input_active_any=false requires an explicit device input_active mask.");
    }
    const auto policy = static_cast<std::uint8_t>(request.policy);
    if (policy != static_cast<std::uint8_t>(rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance) &&
        policy != static_cast<std::uint8_t>(rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset)) {
        throw std::runtime_error("SegmentPenetrationPolicy is invalid.");
    }
    if (!std::isfinite(request.scene_diagonal) || request.scene_diagonal < 0.0 ||
        request.scene_diagonal > std::numeric_limits<float>::max()) {
        throw std::runtime_error("scene_diagonal must be a finite non-negative float32 value.");
    }
    require_cuda(request.capacity_failure_state, "capacity_failure_state");
    require_contiguous(request.capacity_failure_state, "capacity_failure_state");
    require_dtype(request.capacity_failure_state, at::kInt, "capacity_failure_state");
    require_shape(request.capacity_failure_state, {1}, "capacity_failure_state");
    require_same_device(request.origins, request.capacity_failure_state, "capacity_failure_state");
    validate_failure_bit(request.failure_bit);
    const std::int32_t segment_count = checked_i32(request.origins.size(0), "segment_count");
    const std::int32_t hit_capacity = checked_i32(request.hit_capacity, "hit_capacity");
    if (hit_capacity != 0 && segment_count > std::numeric_limits<std::int32_t>::max() / hit_capacity) {
        throw std::runtime_error("segment_count * hit_capacity exceeds int32 row indexing.");
    }
    if (scene.meshes.empty())
        throw std::runtime_error("segment penetration requires a non-empty SceneResource.");
    checked_i32(scene.meshes.size(), "mesh_count");
    return {
        scene,
        input_active,
        segment_count,
        hit_capacity,
        static_cast<std::int32_t>(policy),
        static_cast<float>(request.scene_diagonal),
    };
}

SegmentPenetrationNativeOutputs allocate_outputs(const rayd::torch::SegmentPenetrationRequest& request,
                                                 bool export_tape) {
    const int64_t segment_count = request.origins.size(0);
    const int64_t hit_capacity = request.hit_capacity;
    const auto float_options = request.origins.options();
    const auto int_options = float_options.dtype(at::kInt);
    const auto bool_options = float_options.dtype(at::kBool);
    const auto byte_options = float_options.dtype(at::kByte);
    SegmentPenetrationNativeOutputs outputs{
        {
            at::empty({segment_count, hit_capacity}, bool_options),
            at::empty({segment_count}, int_options),
            at::empty({segment_count}, bool_options),
            at::empty({segment_count}, bool_options),
            at::empty({segment_count}, float_options),
            at::empty({segment_count, 3}, float_options),
            at::empty({segment_count, hit_capacity}, float_options),
            at::empty({segment_count, hit_capacity, 3}, float_options),
            at::empty({segment_count, hit_capacity, 3}, float_options),
            at::empty({segment_count, hit_capacity, 3}, float_options),
            at::empty({segment_count, hit_capacity}, int_options),
        },
        {},
    };
    if (export_tape) {
        outputs.tape = {
            at::empty({segment_count, hit_capacity}, int_options),
            at::empty({segment_count, hit_capacity, 2}, float_options),
            at::empty({segment_count, hit_capacity}, float_options),
            at::empty({segment_count, hit_capacity}, byte_options),
            at::empty({segment_count, hit_capacity}, byte_options),
            at::empty({segment_count}, bool_options),
        };
    }
    return outputs;
}

SegmentPenetrationNativeOutputs forward_native(const rayd::torch::SegmentPenetrationRequest& request,
                                               bool export_tape) {
    ValidatedRequest validated = validate_request(request);
    c10::cuda::CUDAGuard guard(validated.scene.device_index);
    SegmentPenetrationNativeOutputs outputs = allocate_outputs(request, export_tape);
    segment_penetration_initialize_cuda(outputs, validated.input_active, request.capacity_failure_state,
                                        request.failure_bit, request.input_active_any);

    if (validated.segment_count > 0 && request.input_active_any) {
        SegmentPenetrationParams params;
        params.traversable = validated.scene.triangle_ias.traversable;
        params.origins = request.origins.data_ptr<float>();
        params.targets = request.targets.data_ptr<float>();
        params.input_active = validated.input_active == nullptr
                                  ? nullptr
                                  : reinterpret_cast<const std::uint8_t*>(validated.input_active->data_ptr<bool>());
        params.vertices = validated.scene.global_vertices.data_ptr<float>();
        params.faces = validated.scene.global_faces.data_ptr<int>();
        params.face_offsets = validated.scene.face_offsets.data_ptr<int>();
        params.capacity_failure_state = request.capacity_failure_state.data_ptr<int>();
        params.valid = reinterpret_cast<std::uint8_t*>(outputs.result.valid.data_ptr<bool>());
        params.num_hits = outputs.result.num_hits.data_ptr<int>();
        params.reached_target = reinterpret_cast<std::uint8_t*>(outputs.result.reached_target.data_ptr<bool>());
        params.overflow = reinterpret_cast<std::uint8_t*>(outputs.result.overflow.data_ptr<bool>());
        params.distance = outputs.result.distance.data_ptr<float>();
        params.direction = outputs.result.direction.data_ptr<float>();
        params.t = outputs.result.t.data_ptr<float>();
        params.position = outputs.result.position.data_ptr<float>();
        params.normal = outputs.result.normal.data_ptr<float>();
        params.geometric_normal = outputs.result.geometric_normal.data_ptr<float>();
        params.global_primitive_id = outputs.result.global_primitive_id.data_ptr<int>();
        if (export_tape) {
            params.tape_primitive_id = outputs.tape.primitive_id.data_ptr<int>();
            params.tape_barycentric = outputs.tape.barycentric.data_ptr<float>();
            params.tape_restart_epsilon = outputs.tape.restart_epsilon.data_ptr<float>();
            params.tape_restart_branch = outputs.tape.restart_branch.data_ptr<std::uint8_t>();
            params.tape_restart_tie_mask = outputs.tape.restart_tie_mask.data_ptr<std::uint8_t>();
            params.tape_direction_denominator_branch =
                reinterpret_cast<std::uint8_t*>(outputs.tape.direction_denominator_branch.data_ptr<bool>());
        }
        params.segment_count = validated.segment_count;
        params.hit_capacity = validated.hit_capacity;
        params.mesh_count = checked_i32(validated.scene.meshes.size(), "mesh_count");
        params.policy = validated.policy;
        params.failure_bit = request.failure_bit;
        params.scene_diagonal = validated.scene_diagonal;
        // The context, the pipeline, and the stream all belong to the validated
        // scene device, not to whichever device happens to be ambient.
        const int device_index = static_cast<int>(validated.scene.device_index);
        OptixDeviceContextEntry& optix = get_optix_context(device_index);
        shared_optix_launch_pipeline(optix.optix_context, device_index, params.mesh_count,
                                     segment_penetration_pipeline_config())
            ->launch(0, params, static_cast<unsigned int>(validated.segment_count),
                     at::cuda::getCurrentCUDAStream(device_index).stream());
    }
    segment_penetration_sanitize_cuda(outputs, request.capacity_failure_state);
    return outputs;
}

void validate_tape(const rayd::torch::SegmentPenetrationRequest& primal,
                   const rayd::torch::SegmentPenetrationTapeResult& tape) {
    const int64_t segment_count = primal.origins.size(0);
    const int64_t hit_capacity = primal.hit_capacity;
    const at::Tensor& device = primal.origins;
    require_contiguous_tensor(tape.result.valid, at::kBool, {segment_count, hit_capacity}, device, "tape.result.valid");
    require_contiguous_tensor(tape.result.num_hits, at::kInt, {segment_count}, device, "tape.result.num_hits");
    require_contiguous_tensor(tape.result.reached_target, at::kBool, {segment_count}, device,
                              "tape.result.reached_target");
    require_contiguous_tensor(tape.result.overflow, at::kBool, {segment_count}, device, "tape.result.overflow");
    require_contiguous_tensor(tape.result.distance, at::kFloat, {segment_count}, device, "tape.result.distance");
    require_contiguous_tensor(tape.result.direction, at::kFloat, {segment_count, 3}, device, "tape.result.direction");
    require_contiguous_tensor(tape.result.t, at::kFloat, {segment_count, hit_capacity}, device, "tape.result.t");
    for (const auto& entry : {std::pair<const at::Tensor*, const char*>{&tape.result.position, "tape.result.position"},
                              {&tape.result.normal, "tape.result.normal"},
                              {&tape.result.geometric_normal, "tape.result.geometric_normal"}}) {
        require_contiguous_tensor(*entry.first, at::kFloat, {segment_count, hit_capacity, 3}, device, entry.second);
    }
    require_contiguous_tensor(tape.result.global_primitive_id, at::kInt, {segment_count, hit_capacity}, device,
                              "tape.result.global_primitive_id");
    require_contiguous_tensor(tape.tape_primitive_id, at::kInt, {segment_count, hit_capacity}, device,
                              "tape_primitive_id");
    require_contiguous_tensor(tape.tape_barycentric, at::kFloat, {segment_count, hit_capacity, 2}, device,
                              "tape_barycentric");
    require_contiguous_tensor(tape.tape_restart_epsilon, at::kFloat, {segment_count, hit_capacity}, device,
                              "tape_restart_epsilon");
    require_contiguous_tensor(tape.tape_restart_branch, at::kByte, {segment_count, hit_capacity}, device,
                              "tape_restart_branch");
    require_contiguous_tensor(tape.tape_restart_tie_mask, at::kByte, {segment_count, hit_capacity}, device,
                              "tape_restart_tie_mask");
    require_contiguous_tensor(tape.tape_direction_denominator_branch, at::kBool, {segment_count}, device,
                              "tape_direction_denominator_branch");
}

} // namespace

} // namespace rayd::torch_backend

namespace rayd::torch {

SegmentPenetrationResult segment_penetration_forward(const SegmentPenetrationRequest& request) {
    return torch_backend::forward_native(request, false).result;
}

SegmentPenetrationTapeResult segment_penetration_forward_tape(const SegmentPenetrationRequest& request) {
    auto outputs = torch_backend::forward_native(request, true);
    return {
        outputs.result,
        outputs.tape.primitive_id,
        outputs.tape.barycentric,
        outputs.tape.restart_epsilon,
        outputs.tape.restart_branch,
        outputs.tape.restart_tie_mask,
        outputs.tape.direction_denominator_branch,
    };
}

SegmentPenetrationBackwardResult segment_penetration_backward(const SegmentPenetrationBackwardRequest& request) {
    auto validated = torch_backend::validate_request(request.primal);
    torch_backend::validate_tape(request.primal, request.tape);
    const int64_t segment_count = request.primal.origins.size(0);
    const int64_t hit_capacity = request.primal.hit_capacity;
    const at::Tensor& device = request.primal.origins;
    const auto optional = [](const std::optional<at::Tensor>& value) -> const at::Tensor* {
        return value.has_value() && value->defined() ? &*value : nullptr;
    };
    torch_backend::require_strided_float(optional(request.grad_distance), {segment_count}, device, "grad_distance");
    torch_backend::require_strided_float(optional(request.grad_direction), {segment_count, 3}, device,
                                         "grad_direction");
    torch_backend::require_strided_float(optional(request.grad_t), {segment_count, hit_capacity}, device, "grad_t");
    torch_backend::require_strided_float(optional(request.grad_position), {segment_count, hit_capacity, 3}, device,
                                         "grad_position");
    torch_backend::require_strided_float(optional(request.grad_normal), {segment_count, hit_capacity, 3}, device,
                                         "grad_normal");
    torch_backend::require_strided_float(optional(request.grad_geometric_normal), {segment_count, hit_capacity, 3},
                                         device, "grad_geometric_normal");
    c10::cuda::CUDAGuard guard(validated.scene.device_index);
    auto outputs = torch_backend::segment_penetration_backward_cuda(validated.scene, request);
    return {outputs.grad_vertices, outputs.grad_origins, outputs.grad_targets};
}

SegmentPenetrationJvpResult segment_penetration_jvp(const SegmentPenetrationJvpRequest& request) {
    auto validated = torch_backend::validate_request(request.primal);
    torch_backend::validate_tape(request.primal, request.tape);
    const int64_t segment_count = request.primal.origins.size(0);
    const at::Tensor& device = request.primal.origins;
    const auto optional = [](const std::optional<at::Tensor>& value) -> const at::Tensor* {
        return value.has_value() && value->defined() ? &*value : nullptr;
    };
    torch_backend::require_strided_float(optional(request.tangent_vertices), validated.scene.global_vertices.sizes(),
                                         device, "tangent_vertices");
    torch_backend::require_strided_float(optional(request.tangent_origins), {segment_count, 3}, device,
                                         "tangent_origins");
    torch_backend::require_strided_float(optional(request.tangent_targets), {segment_count, 3}, device,
                                         "tangent_targets");
    c10::cuda::CUDAGuard guard(validated.scene.device_index);
    auto outputs = torch_backend::segment_penetration_jvp_cuda(validated.scene, request);
    return {
        outputs.tangent_distance, outputs.tangent_direction, outputs.tangent_t,
        outputs.tangent_position, outputs.tangent_normal,    outputs.tangent_geometric_normal,
    };
}

} // namespace rayd::torch

// OptiX penetration pipeline setup.

#include <src/penetration/segment_penetration_kernels.h>
#include <src/penetration/segment_penetration_params.h>

#include <src/runtime/rt_internal.h>
#include <rayd/penetration/segment_torch_ptx.h>

namespace rayd::torch_backend {

OptixPipelineConfig segment_penetration_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_segment_penetration_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_segment_penetration_optix_ptx);
    config.raygen_entries = {"__raygen__segment_penetration"};
    config.miss_entry = "__miss__segment_penetration";
    config.closesthit_entry = "__closesthit__segment_penetration";
    config.num_payload_values = shared::optix::SceneIntersectionPayloadCount;
    config.params_size = sizeof(SegmentPenetrationParams);
    return config;
}

} // namespace rayd::torch_backend
