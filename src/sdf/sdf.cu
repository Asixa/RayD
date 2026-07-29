// Copyright Xingyu Chen.
// Implements sdf support for sdf.

#include <src/sdf/kernels.h>
#include <src/sdf/derivatives.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <limits>

// Runs one detached sphere trace per ray and records the derivative tape.

namespace rayd::torch_backend {

namespace {

namespace sm = sdf_math;

// ADR-0037 section 5: the `operations.json` `distance` miss sentinel, and the
// only non-finite value any output of this operation may carry. It is spelled
// locally rather than taken from `rt/numeric_policy.h`, which is inside a
// committed-PTX include closure that this device code must stay out of
// (ADR-0037 section 9).
constexpr float kMissDistance = std::numeric_limits<float>::infinity();

__global__ void sdf_intersect_forward_kernel(
    const float* __restrict__ values, int nx, int ny, int nz, const float* __restrict__ box_position,
    const float* __restrict__ box_rotation, const float* __restrict__ box_scale, const float* __restrict__ origins,
    const float* __restrict__ directions, float tmax, int max_steps, float relaxation, float eps_hit_request,
    int64_t ray_count, float* __restrict__ out_t, bool* __restrict__ out_hit, float* __restrict__ out_position,
    float* __restrict__ out_normal, int* __restrict__ out_steps, float* __restrict__ out_tape_t,
    int* __restrict__ out_tape_base) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= ray_count)
        return;

    // Every lane writes its whole row exactly once, so a missed lane is bitwise
    // inert by construction rather than by a later fixup (ADR-0037 section 5).
    out_t[ray] = kMissDistance;
    out_hit[ray] = false;
    out_steps[ray] = 0;
    out_tape_t[ray] = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        out_position[ray * 3 + axis] = 0.0f;
        out_normal[ray * 3 + axis] = 0.0f;
        out_tape_base[ray * 3 + axis] = 0;
    }

    const sm::Lane lane =
        sm::make_lane(box_position, box_rotation, box_scale,
                      sm::vmath::make_vec3(origins[ray * 3 + 0], origins[ray * 3 + 1], origins[ray * 3 + 2]),
                      sm::vmath::make_vec3(directions[ray * 3 + 0], directions[ray * 3 + 1], directions[ray * 3 + 2]),
                      sm::core::GridExtent{nx, ny, nz});
    if (!lane.usable)
        return;

    const sm::core::Interval interval =
        sm::core::clip_ray_to_box(lane.local_origin, lane.local_direction, lane.scale, tmax);
    if (!interval.valid)
        return;

    sm::GridSampler sampler = sm::make_sampler(values, lane);
    sm::core::MarchConfig config{
        interval.t_lo, interval.t_hi, sm::core::resolve_eps_hit(eps_hit_request, lane.scale, lane.cells),
        relaxation,    max_steps,
    };
    const sm::core::MarchResult march = sm::core::sphere_trace(sampler, config);
    out_steps[ray] = march.steps;
    if (!march.hit)
        return;

    const sm::FrozenHit hit = sm::evaluate_frozen(values, lane, sampler.base, march.t);
    out_t[ray] = march.t;
    out_hit[ray] = true;
    out_tape_t[ray] = march.t;
    out_tape_base[ray * 3 + 0] = sampler.base.i;
    out_tape_base[ray * 3 + 1] = sampler.base.j;
    out_tape_base[ray * 3 + 2] = sampler.base.k;
    out_position[ray * 3 + 0] = hit.world_point.x;
    out_position[ray * 3 + 1] = hit.world_point.y;
    out_position[ray * 3 + 2] = hit.world_point.z;
    out_normal[ray * 3 + 0] = hit.normal.x;
    out_normal[ray * 3 + 1] = hit.normal.y;
    out_normal[ray * 3 + 2] = hit.normal.z;
}

} // namespace

SdfIntersectForwardOutputs sdf_intersect_forward_cuda(const SdfGridTensors& grid, const at::Tensor& origins,
                                                      const at::Tensor& directions, const SdfTraceParams& params) {
    const int64_t ray_count = origins.size(0);
    const auto float_options = origins.options();
    SdfIntersectForwardOutputs out;
    out.t = at::empty({ray_count}, float_options);
    out.hit_mask = at::empty({ray_count}, float_options.dtype(at::kBool));
    out.hit_position = at::empty({ray_count, 3}, float_options);
    out.normal = at::empty({ray_count, 3}, float_options);
    out.steps = at::empty({ray_count}, float_options.dtype(at::kInt));
    out.tape_t = at::empty({ray_count}, float_options);
    out.tape_base = at::empty({ray_count, 3}, float_options.dtype(at::kInt));
    if (ray_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(origins.get_device()).stream();
    sdf_intersect_forward_kernel<<<blocks, threads, 0, stream>>>(
        grid.values.data_ptr<float>(), static_cast<int>(grid.values.size(0)), static_cast<int>(grid.values.size(1)),
        static_cast<int>(grid.values.size(2)), grid.position.data_ptr<float>(), grid.rotation.data_ptr<float>(),
        grid.scale.data_ptr<float>(), origins.data_ptr<float>(), directions.data_ptr<float>(),
        static_cast<float>(params.tmax), static_cast<int>(params.max_steps), static_cast<float>(params.relaxation),
        static_cast<float>(params.eps_hit), ray_count, out.t.data_ptr<float>(), out.hit_mask.data_ptr<bool>(),
        out.hit_position.data_ptr<float>(), out.normal.data_ptr<float>(), out.steps.data_ptr<int>(),
        out.tape_t.data_ptr<float>(), out.tape_base.data_ptr<int>());
    return out;
}

} // namespace rayd::torch_backend

// ---- merged from src/sdf/sdf_backward_part.cu ----

#include <src/sdf/kernels.h>
#include <src/sdf/derivatives.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// ADR-0037 derivatives: the frozen-winner implicit function theorem, in both
// modes. Neither kernel re-marches. Both consume the tape's hit distance, hit
// mask and base voxel index as constants, rebuild the interpolant on that voxel,
// and differentiate the resulting closed-form expressions.
//
// Both modes carry the same six inputs (`values`, `position`, `rotation`,
// `scale`, `origins`, `directions`) and the same three differentiable outputs
// (`t`, `hit_position`, `normal`), so they are exact duals on every lane.
// Missed lanes contribute no atomic and leave every output at positive zero.

namespace rayd::torch_backend {

namespace {

namespace sm = sdf_math;
namespace vm = shared::math;

using vm::Vec3f;

__device__ Vec3f read_vec3(const float* base, int64_t row) {
    if (base == nullptr)
        return vm::make_vec3(0.0f, 0.0f, 0.0f);
    return vm::make_vec3(base[row * 3 + 0], base[row * 3 + 1], base[row * 3 + 2]);
}

// A placement tensor is one shared `[3]` row, not a per-ray one.
__device__ Vec3f read_vec3_shared(const float* base) {
    return read_vec3(base, 0);
}

__device__ void write_vec3(float* base, int64_t row, Vec3f value) {
    base[row * 3 + 0] = value.x;
    base[row * 3 + 1] = value.y;
    base[row * 3 + 2] = value.z;
}

// Componentwise `v_i * cells_i / scale_i`, the chain-rule factor between index
// space and local space that both modes apply twice.
__device__ Vec3f per_axis_ratio(Vec3f value, Vec3f cells, Vec3f box_scale) {
    return vm::make_vec3(value.x * cells.x / box_scale.x, value.y * cells.y / box_scale.y,
                         value.z * cells.z / box_scale.z);
}

__global__ void sdf_intersect_backward_kernel(
    const float* __restrict__ values, int nx, int ny, int nz, const float* __restrict__ box_position,
    const float* __restrict__ box_rotation, const float* __restrict__ box_scale, const float* __restrict__ origins,
    const float* __restrict__ directions, const float* __restrict__ tape_t, const bool* __restrict__ tape_hit,
    const int* __restrict__ tape_base, const float* __restrict__ grad_t, const float* __restrict__ grad_hit_position,
    const float* __restrict__ grad_normal, int64_t ray_count, float* __restrict__ grad_values,
    float* __restrict__ grad_box_position, float* __restrict__ grad_box_rotation, float* __restrict__ grad_box_scale,
    float* __restrict__ grad_origins, float* __restrict__ grad_directions) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= ray_count || !tape_hit[ray])
        return;

    const sm::Lane lane = sm::make_lane(box_position, box_rotation, box_scale, read_vec3(origins, ray),
                                        read_vec3(directions, ray), sm::core::GridExtent{nx, ny, nz});
    if (!lane.usable)
        return;

    const sm::core::BaseIndex base{
        tape_base[ray * 3 + 0],
        tape_base[ray * 3 + 1],
        tape_base[ray * 3 + 2],
    };
    const float hit_distance = tape_t[ray];
    const sm::FrozenHit hit = sm::evaluate_frozen(values, lane, base, hit_distance);
    const Vec3f center = vm::make_vec3(box_position[0], box_position[1], box_position[2]);
    const Vec3f offset = vm::subtract(hit.world_point, center);

    Vec3f grad_world_point = read_vec3(grad_hit_position, ray);
    Vec3f grad_center = vm::make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_box_scale_local = vm::make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_index_gradient = vm::make_vec3(0.0f, 0.0f, 0.0f);
    sm::Mat3 grad_rotation_matrix = sm::zero_mat3();

    // The normal is recomputed differentiably at the frozen hit, so its
    // gradient reaches the field values twice (through the interpolant's
    // gradient and, via the hit point, through the hit distance) and needs the
    // interpolant's second derivative (ADR-0037 section 6).
    if (grad_normal != nullptr) {
        const Vec3f grad_world_gradient =
            sm::normalize_floor_jacobian(hit.gradient_length, hit.normal, read_vec3(grad_normal, ray));
        const Vec3f grad_local_gradient = sm::core::world_to_local_direction(lane.placement, grad_world_gradient);
        sm::add_outer(grad_rotation_matrix, grad_world_gradient, hit.local_gradient, 1.0f);
        grad_index_gradient = per_axis_ratio(grad_local_gradient, lane.cells, lane.scale);
        grad_box_scale_local = vm::subtract(grad_box_scale_local,
                                            vm::make_vec3(grad_local_gradient.x * hit.local_gradient.x / lane.scale.x,
                                                          grad_local_gradient.y * hit.local_gradient.y / lane.scale.y,
                                                          grad_local_gradient.z * hit.local_gradient.z / lane.scale.z));

        const Vec3f grad_coordinate = sm::hessian_multiply(sm::index_hessian(values, hit.cell), grad_index_gradient);
        const Vec3f grad_local_point = per_axis_ratio(grad_coordinate, lane.cells, lane.scale);
        grad_box_scale_local =
            vm::subtract(grad_box_scale_local, vm::make_vec3(grad_coordinate.x * lane.cells.x * hit.local_point.x /
                                                                 (lane.scale.x * lane.scale.x),
                                                             grad_coordinate.y * lane.cells.y * hit.local_point.y /
                                                                 (lane.scale.y * lane.scale.y),
                                                             grad_coordinate.z * lane.cells.z * hit.local_point.z /
                                                                 (lane.scale.z * lane.scale.z)));
        sm::add_outer(grad_rotation_matrix, offset, grad_local_point, 1.0f);
        const Vec3f grad_offset = sm::core::local_to_world_direction(lane.placement, grad_local_point);
        grad_world_point = vm::add(grad_world_point, grad_offset);
        grad_center = vm::subtract(grad_center, grad_offset);
    }

    // The hit point depends on the hit distance, so the normal and position
    // gradients fold back into the single scalar the IFT is applied to.
    const float grad_hit_distance =
        (grad_t == nullptr ? 0.0f : grad_t[ray]) + vm::dot(grad_world_point, lane.unit_direction);
    Vec3f grad_unit_direction = vm::scale(grad_world_point, hit_distance);
    Vec3f grad_origin = grad_world_point;

    // dt*/dtheta = -(dF/dtheta) / g_clamped, applied once to every input.
    const float factor = -grad_hit_distance / hit.denominator;
    grad_origin = vm::add(grad_origin, vm::scale(hit.world_gradient, factor));
    grad_center = vm::subtract(grad_center, vm::scale(hit.world_gradient, factor));
    grad_box_scale_local = vm::add(grad_box_scale_local, vm::scale(sm::scale_partial(lane, hit), factor));
    grad_unit_direction = vm::add(grad_unit_direction, vm::scale(hit.world_gradient, factor * hit_distance));
    sm::add_outer(grad_rotation_matrix, offset, hit.local_gradient, factor);

    if (grad_values != nullptr) {
        for (int corner = 0; corner < 8; ++corner) {
            float contribution = factor * hit.cell.weight[corner];
            if (grad_normal != nullptr)
                contribution += vm::dot(grad_index_gradient, sm::corner_weight_gradient(hit.cell, corner));
            atomicAdd(&grad_values[hit.cell.index[corner]], contribution);
        }
    }
    if (grad_origins != nullptr)
        write_vec3(grad_origins, ray, grad_origin);
    if (grad_directions != nullptr)
        write_vec3(grad_directions, ray,
                   sm::normalize_floor_jacobian(lane.direction_length, lane.unit_direction, grad_unit_direction));
    if (grad_box_position != nullptr) {
        atomicAdd(&grad_box_position[0], grad_center.x);
        atomicAdd(&grad_box_position[1], grad_center.y);
        atomicAdd(&grad_box_position[2], grad_center.z);
    }
    if (grad_box_scale != nullptr) {
        atomicAdd(&grad_box_scale[0], grad_box_scale_local.x);
        atomicAdd(&grad_box_scale[1], grad_box_scale_local.y);
        atomicAdd(&grad_box_scale[2], grad_box_scale_local.z);
    }
    if (grad_box_rotation != nullptr) {
        const sm::Quat grad_quaternion = sm::quaternion_vjp(sm::make_quat(box_rotation), grad_rotation_matrix);
        atomicAdd(&grad_box_rotation[0], grad_quaternion.w);
        atomicAdd(&grad_box_rotation[1], grad_quaternion.x);
        atomicAdd(&grad_box_rotation[2], grad_quaternion.y);
        atomicAdd(&grad_box_rotation[3], grad_quaternion.z);
    }
}

__global__ void sdf_intersect_jvp_kernel(
    const float* __restrict__ values, int nx, int ny, int nz, const float* __restrict__ box_position,
    const float* __restrict__ box_rotation, const float* __restrict__ box_scale, const float* __restrict__ origins,
    const float* __restrict__ directions, const float* __restrict__ tape_t, const bool* __restrict__ tape_hit,
    const int* __restrict__ tape_base, const float* __restrict__ tangent_values,
    const float* __restrict__ tangent_position, const float* __restrict__ tangent_rotation,
    const float* __restrict__ tangent_scale, const float* __restrict__ tangent_origins,
    const float* __restrict__ tangent_directions, int64_t ray_count, float* __restrict__ out_tangent_t,
    float* __restrict__ out_tangent_position, float* __restrict__ out_tangent_normal) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= ray_count || !tape_hit[ray])
        return;

    const sm::Lane lane = sm::make_lane(box_position, box_rotation, box_scale, read_vec3(origins, ray),
                                        read_vec3(directions, ray), sm::core::GridExtent{nx, ny, nz});
    if (!lane.usable)
        return;

    const sm::core::BaseIndex base{
        tape_base[ray * 3 + 0],
        tape_base[ray * 3 + 1],
        tape_base[ray * 3 + 2],
    };
    const float hit_distance = tape_t[ray];
    const sm::FrozenHit hit = sm::evaluate_frozen(values, lane, base, hit_distance);
    const Vec3f center = vm::make_vec3(box_position[0], box_position[1], box_position[2]);
    const Vec3f offset = vm::subtract(hit.world_point, center);

    const Vec3f tangent_center = read_vec3_shared(tangent_position);
    const Vec3f tangent_box_scale = read_vec3_shared(tangent_scale);
    const Vec3f tangent_origin = read_vec3(tangent_origins, ray);
    const Vec3f tangent_direction = read_vec3(tangent_directions, ray);
    const sm::Quat tangent_quaternion =
        tangent_rotation == nullptr ? sm::Quat{0.0f, 0.0f, 0.0f, 0.0f} : sm::make_quat(tangent_rotation);
    const sm::Mat3 rotation_rate = sm::rotation_differential(sm::make_quat(box_rotation), tangent_quaternion);
    const Vec3f tangent_unit_direction =
        sm::normalize_floor_jacobian(lane.direction_length, lane.unit_direction, tangent_direction);

    float field_rate = vm::dot(hit.world_gradient, vm::subtract(tangent_origin, tangent_center)) +
                       hit_distance * vm::dot(hit.world_gradient, tangent_unit_direction) +
                       vm::dot(sm::scale_partial(lane, hit), tangent_box_scale) +
                       vm::dot(offset, sm::multiply(rotation_rate, hit.local_gradient));
    if (tangent_values != nullptr)
        for (int corner = 0; corner < 8; ++corner)
            field_rate += hit.cell.weight[corner] * tangent_values[hit.cell.index[corner]];
    const float tangent_hit_distance = -field_rate / hit.denominator;

    const Vec3f tangent_world_point =
        vm::add(tangent_origin, vm::add(vm::scale(lane.unit_direction, tangent_hit_distance),
                                        vm::scale(tangent_unit_direction, hit_distance)));
    const Vec3f tangent_offset = vm::subtract(tangent_world_point, tangent_center);
    const Vec3f tangent_local_point = vm::add(sm::transpose_multiply(rotation_rate, offset),
                                              sm::core::world_to_local_direction(lane.placement, tangent_offset));
    const Vec3f tangent_coordinate = vm::subtract(
        per_axis_ratio(tangent_local_point, lane.cells, lane.scale),
        vm::make_vec3(tangent_box_scale.x * lane.cells.x * hit.local_point.x / (lane.scale.x * lane.scale.x),
                      tangent_box_scale.y * lane.cells.y * hit.local_point.y / (lane.scale.y * lane.scale.y),
                      tangent_box_scale.z * lane.cells.z * hit.local_point.z / (lane.scale.z * lane.scale.z)));

    Vec3f tangent_index_gradient = sm::hessian_multiply(sm::index_hessian(values, hit.cell), tangent_coordinate);
    if (tangent_values != nullptr)
        for (int corner = 0; corner < 8; ++corner)
            tangent_index_gradient =
                vm::add(tangent_index_gradient, vm::scale(sm::corner_weight_gradient(hit.cell, corner),
                                                          tangent_values[hit.cell.index[corner]]));
    const Vec3f tangent_local_gradient =
        vm::subtract(per_axis_ratio(tangent_index_gradient, lane.cells, lane.scale),
                     vm::make_vec3(tangent_box_scale.x * hit.local_gradient.x / lane.scale.x,
                                   tangent_box_scale.y * hit.local_gradient.y / lane.scale.y,
                                   tangent_box_scale.z * hit.local_gradient.z / lane.scale.z));
    const Vec3f tangent_world_gradient =
        vm::add(sm::multiply(rotation_rate, hit.local_gradient),
                sm::core::local_to_world_direction(lane.placement, tangent_local_gradient));

    out_tangent_t[ray] = tangent_hit_distance;
    write_vec3(out_tangent_position, ray, tangent_world_point);
    write_vec3(out_tangent_normal, ray,
               sm::normalize_floor_jacobian(hit.gradient_length, hit.normal, tangent_world_gradient));
}

const float* raw(const at::Tensor* tensor) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0)
        return nullptr;
    return tensor->data_ptr<float>();
}

float* raw(at::Tensor& tensor) {
    if (!tensor.defined())
        return nullptr;
    return tensor.data_ptr<float>();
}

} // namespace

SdfIntersectBackwardOutputs sdf_intersect_backward_cuda(const SdfGridTensors& grid, const at::Tensor& origins,
                                                        const at::Tensor& directions, const SdfTapeTensors& tape,
                                                        const SdfIntersectGradRequest& request) {
    const int64_t ray_count = origins.size(0);
    SdfIntersectBackwardOutputs out;
    if (request.need_grad_values)
        out.grad_values = at::zeros_like(grid.values);
    if (request.need_grad_position)
        out.grad_position = at::zeros_like(grid.position);
    if (request.need_grad_rotation)
        out.grad_rotation = at::zeros_like(grid.rotation);
    if (request.need_grad_scale)
        out.grad_scale = at::zeros_like(grid.scale);
    if (request.need_grad_origins)
        out.grad_origins = at::zeros_like(origins);
    if (request.need_grad_directions)
        out.grad_directions = at::zeros_like(directions);
    // With no upstream gradient, or with no input that wants one, every answer
    // is exactly the zero tensor already allocated, so the launch is skipped
    // rather than run to add zeros.
    const bool any_upstream = raw(request.grad_t) != nullptr || raw(request.grad_hit_position) != nullptr ||
                              raw(request.grad_normal) != nullptr;
    const bool any_wanted = request.need_grad_values || request.need_grad_position || request.need_grad_rotation ||
                            request.need_grad_scale || request.need_grad_origins || request.need_grad_directions;
    if (ray_count == 0 || !any_upstream || !any_wanted)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(origins.get_device()).stream();
    sdf_intersect_backward_kernel<<<blocks, threads, 0, stream>>>(
        grid.values.data_ptr<float>(), static_cast<int>(grid.values.size(0)), static_cast<int>(grid.values.size(1)),
        static_cast<int>(grid.values.size(2)), grid.position.data_ptr<float>(), grid.rotation.data_ptr<float>(),
        grid.scale.data_ptr<float>(), origins.data_ptr<float>(), directions.data_ptr<float>(), tape.t.data_ptr<float>(),
        tape.hit.data_ptr<bool>(), tape.base.data_ptr<int>(), raw(request.grad_t), raw(request.grad_hit_position),
        raw(request.grad_normal), ray_count, raw(out.grad_values), raw(out.grad_position), raw(out.grad_rotation),
        raw(out.grad_scale), raw(out.grad_origins), raw(out.grad_directions));
    return out;
}

SdfIntersectJvpOutputs sdf_intersect_jvp_cuda(const SdfGridTensors& grid, const at::Tensor& origins,
                                              const at::Tensor& directions, const SdfTapeTensors& tape,
                                              const SdfIntersectTangentInputs& tangents) {
    const int64_t ray_count = origins.size(0);
    SdfIntersectJvpOutputs out;
    out.tangent_t = at::zeros({ray_count}, origins.options());
    out.tangent_hit_position = at::zeros({ray_count, 3}, origins.options());
    out.tangent_normal = at::zeros({ray_count, 3}, origins.options());
    if (ray_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(origins.get_device()).stream();
    sdf_intersect_jvp_kernel<<<blocks, threads, 0, stream>>>(
        grid.values.data_ptr<float>(), static_cast<int>(grid.values.size(0)), static_cast<int>(grid.values.size(1)),
        static_cast<int>(grid.values.size(2)), grid.position.data_ptr<float>(), grid.rotation.data_ptr<float>(),
        grid.scale.data_ptr<float>(), origins.data_ptr<float>(), directions.data_ptr<float>(), tape.t.data_ptr<float>(),
        tape.hit.data_ptr<bool>(), tape.base.data_ptr<int>(), raw(tangents.values), raw(tangents.position),
        raw(tangents.rotation), raw(tangents.scale), raw(tangents.origins), raw(tangents.directions), ray_count,
        out.tangent_t.data_ptr<float>(), out.tangent_hit_position.data_ptr<float>(),
        out.tangent_normal.data_ptr<float>());
    return out;
}

} // namespace rayd::torch_backend
