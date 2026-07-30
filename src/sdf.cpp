// Copyright Xingyu Chen.
// Implements Torch SDF validation and dispatch.

#include <src/sdf_internal.h>
#include <src/bindings/tensor_contract.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Host entry points for the ADR-0037 SDF intersection. These are pure C++: no
// pybind type crosses this boundary and no GIL is taken, so the dispatcher can
// call them from any thread (ADR-0037 section 9).
//
// Validation is structural only. Value conditions that would need a device read
// (positive `scale`, finite `values`) are the device path's job and surface as
// misses, so nothing here synchronizes or copies from the device.
//
// Every entry guards the grid's CUDA device before the device path runs, since
// that path launches on the input device's current stream and would otherwise
// be submitted to whichever device happens to be ambient. The guard only sets
// the current device: it changes no shape, order, or value.

namespace rayd::torch_backend {

namespace {

// The kernels index `values` and the ray batch with int32 lane arithmetic, so a
// batch or grid that does not fit is rejected rather than silently wrapped.
constexpr int64_t kMaxIndex = 2147483647;

const at::Tensor* present(const c10::optional<at::Tensor>& value) {
    if (!value.has_value() || !value->defined() || value->numel() == 0)
        return nullptr;
    return &*value;
}

void require_grid_values(const at::Tensor& values) {
    require_cuda(values, "values");
    require_contiguous(values, "values");
    require_dtype(values, at::kFloat, "values");
    require_rank(values, 3, "values");
    for (int64_t axis = 0; axis < 3; ++axis) {
        if (values.size(axis) < 2)
            throw std::runtime_error("values must have at least 2 samples on every axis (got " +
                                     std::to_string(values.size(axis)) + " on axis " + std::to_string(axis) + ").");
    }
    if (values.numel() > kMaxIndex)
        throw std::runtime_error("values has more elements than the kernel index range allows.");
}

void require_same_device(const at::Tensor& tensor, int device_index, const char* name) {
    if (tensor.get_device() != device_index)
        throw std::runtime_error(std::string(name) + " must be on the same CUDA device as the SDF grid values.");
}

int64_t require_batch_and_rays(const SdfBatchTensors& batch, const at::Tensor& origins, const at::Tensor& directions) {
    require_cuda(batch.values, "values");
    require_contiguous(batch.values, "values");
    require_dtype(batch.values, at::kFloat, "values");
    require_rank(batch.values, 4, "values");
    if (batch.values.size(0) < 2)
        throw std::runtime_error("SDF batch values must contain at least two grids.");
    for (int64_t axis = 1; axis < 4; ++axis) {
        if (batch.values.size(axis) < 2)
            throw std::runtime_error("SDF batch values must have at least 2 samples on every spatial axis.");
    }
    const int64_t grid_count = batch.values.size(0);
    for (const auto& field :
         {std::pair<const at::Tensor*, int64_t>{&batch.position, 3}, {&batch.rotation, 4}, {&batch.scale, 3}}) {
        require_cuda(*field.first, "batch placement");
        require_contiguous(*field.first, "batch placement");
        require_dtype(*field.first, at::kFloat, "batch placement");
        require_rank(*field.first, 2, "batch placement");
        if (field.first->size(0) != grid_count || field.first->size(1) != field.second)
            throw std::runtime_error("SDF batch placement tensors must match the grid count and field width.");
    }
    require_vec3f(origins, "origins");
    require_vec3f(directions, "directions");
    if (origins.size(0) != directions.size(0))
        throw std::runtime_error("origins and directions must have the same ray count.");
    if (origins.size(0) > kMaxIndex || grid_count > kMaxIndex / std::max<int64_t>(origins.size(0), 1))
        throw std::runtime_error("the SDF batch grid-ray product is larger than the kernel index range allows.");
    const int device_index = batch.values.get_device();
    require_same_device(batch.position, device_index, "position");
    require_same_device(batch.rotation, device_index, "rotation");
    require_same_device(batch.scale, device_index, "scale");
    require_same_device(origins, device_index, "origins");
    require_same_device(directions, device_index, "directions");
    return origins.size(0);
}

void require_placement_vector(const at::Tensor& tensor, int64_t length, const char* name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
    if (tensor.size(0) != length)
        throw std::runtime_error(std::string(name) + " must have exactly " + std::to_string(length) + " elements.");
}

int64_t require_grid_and_rays(const SdfGridTensors& grid, const at::Tensor& origins, const at::Tensor& directions) {
    require_grid_values(grid.values);
    require_placement_vector(grid.position, 3, "position");
    require_placement_vector(grid.rotation, 4, "rotation");
    require_placement_vector(grid.scale, 3, "scale");
    require_vec3f(origins, "origins");
    require_vec3f(directions, "directions");
    if (origins.size(0) != directions.size(0))
        throw std::runtime_error("origins and directions must have the same ray count.");
    if (origins.size(0) > kMaxIndex)
        throw std::runtime_error("the ray batch is larger than the kernel index range allows.");
    const int device_index = grid.values.get_device();
    require_same_device(grid.position, device_index, "position");
    require_same_device(grid.rotation, device_index, "rotation");
    require_same_device(grid.scale, device_index, "scale");
    require_same_device(origins, device_index, "origins");
    require_same_device(directions, device_index, "directions");
    return origins.size(0);
}

void require_trace_params(const SdfTraceParams& params) {
    if (!(params.tmax > 0.0))
        throw std::runtime_error("tmax must be positive.");
    if (params.max_steps < 1)
        throw std::runtime_error("max_steps must be at least 1.");
    if (!(params.relaxation > 0.0) || params.relaxation > 1.0)
        throw std::runtime_error("relaxation must lie in (0, 1].");
}

// The tape is the whole of the frozen decision set: hit distance, hit mask, and
// base voxel index. Backward and JVP consume it and never re-march.
void require_tape(const SdfTapeTensors& tape, int64_t ray_count, int device_index) {
    require_cuda(tape.t, "tape_t");
    require_contiguous(tape.t, "tape_t");
    require_dtype(tape.t, at::kFloat, "tape_t");
    require_rank(tape.t, 1, "tape_t");
    require_mask(tape.hit, "tape_hit");
    require_cuda(tape.base, "tape_base");
    require_contiguous(tape.base, "tape_base");
    require_dtype(tape.base, at::kInt, "tape_base");
    require_rank(tape.base, 2, "tape_base");
    require_last_dim(tape.base, 3, "tape_base");
    if (tape.t.size(0) != ray_count || tape.hit.size(0) != ray_count || tape.base.size(0) != ray_count)
        throw std::runtime_error("the tape must match the ray batch size.");
    require_same_device(tape.t, device_index, "tape_t");
    require_same_device(tape.hit, device_index, "tape_hit");
    require_same_device(tape.base, device_index, "tape_base");
}

// Gradients and tangents are read with contiguous row arithmetic, so a strided
// view is rejected instead of being silently misread.
void require_row_tensor(const at::Tensor* tensor, int64_t ray_count, int64_t width, int device_index,
                        const char* name) {
    if (tensor == nullptr)
        return;
    require_cuda(*tensor, name);
    require_contiguous(*tensor, name);
    require_dtype(*tensor, at::kFloat, name);
    if (width == 0) {
        require_rank(*tensor, 1, name);
    } else {
        require_rank(*tensor, 2, name);
        require_last_dim(*tensor, width, name);
    }
    if (tensor->size(0) != ray_count)
        throw std::runtime_error(std::string(name) + " must match the ray batch size.");
    require_same_device(*tensor, device_index, name);
}

void require_like(const at::Tensor* tensor, const at::Tensor& reference, int device_index, const char* name) {
    if (tensor == nullptr)
        return;
    require_cuda(*tensor, name);
    require_contiguous(*tensor, name);
    require_dtype(*tensor, at::kFloat, name);
    if (tensor->sizes() != reference.sizes())
        throw std::runtime_error(std::string(name) + " must have the same shape as its input.");
    require_same_device(*tensor, device_index, name);
}

} // namespace

std::vector<at::Tensor> sdf_intersect_forward_impl(at::Tensor values, at::Tensor position, at::Tensor rotation,
                                                   at::Tensor scale, at::Tensor origins, at::Tensor directions,
                                                   double tmax, int64_t max_steps, double relaxation, double eps_hit) {
    const SdfGridTensors grid{std::move(values), std::move(position), std::move(rotation), std::move(scale)};
    require_grid_and_rays(grid, origins, directions);
    // A non-positive `eps_hit` is the ADR-0037 section 7 sentinel meaning
    // "derive from the resident scale on the device", so it is not rejected
    // here; the Python layer rejects an explicit non-positive request.
    const SdfTraceParams params{tmax, max_steps, relaxation, eps_hit};
    require_trace_params(params);
    c10::cuda::CUDAGuard guard(grid.values.device());
    SdfIntersectForwardOutputs out = sdf_intersect_forward_cuda(grid, origins, directions, params);
    return {
        out.t, out.hit_mask, out.hit_position, out.normal, out.steps, out.tape_t, out.tape_base,
    };
}

std::vector<at::Tensor> sdf_batch_intersect_forward_impl(at::Tensor values, at::Tensor position, at::Tensor rotation,
                                                         at::Tensor scale, at::Tensor origins, at::Tensor directions,
                                                         double tmax, int64_t max_steps, double relaxation,
                                                         double eps_hit) {
    const SdfBatchTensors batch{std::move(values), std::move(position), std::move(rotation), std::move(scale)};
    require_batch_and_rays(batch, origins, directions);
    const SdfTraceParams params{tmax, max_steps, relaxation, eps_hit};
    require_trace_params(params);
    c10::cuda::CUDAGuard guard(batch.values.device());
    SdfBatchForwardOutputs out = sdf_batch_intersect_forward_cuda(batch, origins, directions, params);
    return {out.t, out.hit_mask, out.hit_position, out.normal, out.steps};
}

std::vector<c10::optional<at::Tensor>> sdf_intersect_backward_impl(
    at::Tensor values, at::Tensor position, at::Tensor rotation, at::Tensor scale, at::Tensor origins,
    at::Tensor directions, at::Tensor tape_t, at::Tensor tape_hit, at::Tensor tape_base,
    c10::optional<at::Tensor> grad_t, c10::optional<at::Tensor> grad_hit_position,
    c10::optional<at::Tensor> grad_normal, bool need_grad_values, bool need_grad_position, bool need_grad_rotation,
    bool need_grad_scale, bool need_grad_origins, bool need_grad_directions) {
    const SdfGridTensors grid{std::move(values), std::move(position), std::move(rotation), std::move(scale)};
    const int64_t ray_count = require_grid_and_rays(grid, origins, directions);
    const int device_index = grid.values.get_device();
    const SdfTapeTensors tape{std::move(tape_t), std::move(tape_hit), std::move(tape_base)};
    require_tape(tape, ray_count, device_index);
    const SdfIntersectGradRequest request{
        present(grad_t),  present(grad_hit_position), present(grad_normal),
        need_grad_values, need_grad_position,         need_grad_rotation,
        need_grad_scale,  need_grad_origins,          need_grad_directions,
    };
    require_row_tensor(request.grad_t, ray_count, 0, device_index, "grad_t");
    require_row_tensor(request.grad_hit_position, ray_count, 3, device_index, "grad_hit_position");
    require_row_tensor(request.grad_normal, ray_count, 3, device_index, "grad_normal");
    c10::cuda::CUDAGuard guard(grid.values.device());
    SdfIntersectBackwardOutputs out = sdf_intersect_backward_cuda(grid, origins, directions, tape, request);
    std::vector<c10::optional<at::Tensor>> result;
    result.reserve(6);
    for (const at::Tensor& grad : {out.grad_values, out.grad_position, out.grad_rotation, out.grad_scale,
                                   out.grad_origins, out.grad_directions}) {
        if (grad.defined())
            result.emplace_back(grad);
        else
            result.emplace_back(c10::nullopt);
    }
    return result;
}

std::vector<at::Tensor> sdf_intersect_jvp_impl(
    at::Tensor values, at::Tensor position, at::Tensor rotation, at::Tensor scale, at::Tensor origins,
    at::Tensor directions, at::Tensor tape_t, at::Tensor tape_hit, at::Tensor tape_base,
    c10::optional<at::Tensor> tangent_values, c10::optional<at::Tensor> tangent_position,
    c10::optional<at::Tensor> tangent_rotation, c10::optional<at::Tensor> tangent_scale,
    c10::optional<at::Tensor> tangent_origins, c10::optional<at::Tensor> tangent_directions) {
    const SdfGridTensors grid{std::move(values), std::move(position), std::move(rotation), std::move(scale)};
    const int64_t ray_count = require_grid_and_rays(grid, origins, directions);
    const int device_index = grid.values.get_device();
    const SdfTapeTensors tape{std::move(tape_t), std::move(tape_hit), std::move(tape_base)};
    require_tape(tape, ray_count, device_index);
    const SdfIntersectTangentInputs tangents{
        present(tangent_values), present(tangent_position), present(tangent_rotation),
        present(tangent_scale),  present(tangent_origins),  present(tangent_directions),
    };
    require_like(tangents.values, grid.values, device_index, "tangent_values");
    require_like(tangents.position, grid.position, device_index, "tangent_position");
    require_like(tangents.rotation, grid.rotation, device_index, "tangent_rotation");
    require_like(tangents.scale, grid.scale, device_index, "tangent_scale");
    require_row_tensor(tangents.origins, ray_count, 3, device_index, "tangent_origins");
    require_row_tensor(tangents.directions, ray_count, 3, device_index, "tangent_directions");
    c10::cuda::CUDAGuard guard(grid.values.device());
    SdfIntersectJvpOutputs out = sdf_intersect_jvp_cuda(grid, origins, directions, tape, tangents);
    return {out.tangent_t, out.tangent_hit_position, out.tangent_normal};
}

} // namespace rayd::torch_backend
