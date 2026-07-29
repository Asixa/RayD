// Copyright Xingyu Chen.
// Implements penetration kernels and derivatives.

#include <src/penetration/segment_penetration_kernels.h>

#include <rayd/math.h>
#include <src/runtime/optix_context.h>
#include <src/penetration/segment_penetration_params.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

namespace {

void cuda_check(cudaError_t result, const char* expression) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(std::string("CUDA error in ") + expression + ": " + cudaGetErrorString(result));
}

std::uint8_t* byte_mask_ptr(const at::Tensor& tensor) {
    return reinterpret_cast<std::uint8_t*>(tensor.data_ptr<bool>());
}

const std::uint8_t* optional_byte_mask_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : reinterpret_cast<const std::uint8_t*>(tensor->data_ptr<bool>());
}

int64_t optional_stride(const at::Tensor* tensor, int64_t dimension) {
    return tensor == nullptr ? 0 : tensor->stride(dimension);
}

__device__ float3 load3(const float* values, int64_t row, int64_t stride0, int64_t stride1) {
    return values == nullptr ? make_float3(0.0f, 0.0f, 0.0f)
                             : make_float3(values[row * stride0 + 0 * stride1], values[row * stride0 + 1 * stride1],
                                           values[row * stride0 + 2 * stride1]);
}

__device__ float3 load3_3d(const float* values, int64_t row, int64_t slot, int64_t stride0, int64_t stride1,
                           int64_t stride2) {
    return values == nullptr ? make_float3(0.0f, 0.0f, 0.0f)
                             : make_float3(values[row * stride0 + slot * stride1 + 0 * stride2],
                                           values[row * stride0 + slot * stride1 + 1 * stride2],
                                           values[row * stride0 + slot * stride1 + 2 * stride2]);
}

__device__ void store3(float* values, int64_t row, float3 value) {
    values[row * 3 + 0] = value.x;
    values[row * 3 + 1] = value.y;
    values[row * 3 + 2] = value.z;
}

__device__ void atomic_add3(float* values, int index, float3 value) {
    atomicAdd(values + index * 3 + 0, value.x);
    atomicAdd(values + index * 3 + 1, value.y);
    atomicAdd(values + index * 3 + 2, value.z);
}

__device__ float determinant(float3 c0, float3 c1, float3 c2) {
    return dot3(c0, cross3(c1, c2));
}

__device__ float3 solve_columns(float3 c0, float3 c1, float3 c2, float3 rhs) {
    float det = determinant(c0, c1, c2);
    if (fabsf(det) < 1.0e-12f)
        det = copysignf(1.0e-12f, det == 0.0f ? 1.0f : det);
    const float inverse = 1.0f / det;
    return make_float3(determinant(rhs, c1, c2) * inverse, determinant(c0, rhs, c2) * inverse,
                       determinant(c0, c1, rhs) * inverse);
}

__device__ float3 solve_transpose(float3 c0, float3 c1, float3 c2, float3 rhs) {
    return solve_columns(make_float3(c0.x, c1.x, c2.x), make_float3(c0.y, c1.y, c2.y), make_float3(c0.z, c1.z, c2.z),
                         rhs);
}

__device__ float3 normalized_vjp(float3 value, float floor, float3 gradient) {
    const float norm = sqrtf(fmaxf(dot3(value, value), 0.0f));
    if (norm <= floor)
        return mul3(1.0f / floor, gradient);
    const float3 normalized = mul3(1.0f / norm, value);
    return mul3(1.0f / norm, sub3(gradient, mul3(dot3(normalized, gradient), normalized)));
}

__device__ float3 normalized_jvp(float3 value, float floor, float3 tangent) {
    const float norm = sqrtf(fmaxf(dot3(value, value), 0.0f));
    if (norm <= floor)
        return mul3(1.0f / floor, tangent);
    const float3 normalized = mul3(1.0f / norm, value);
    return mul3(1.0f / norm, sub3(tangent, mul3(dot3(normalized, tangent), normalized)));
}

__device__ int population3(std::uint8_t mask) {
    return static_cast<int>(mask & 1u) + static_cast<int>((mask >> 1) & 1u) + static_cast<int>((mask >> 2) & 1u);
}

__global__ void initialize_kernel(std::uint8_t* valid, int* num_hits, std::uint8_t* reached_target,
                                  std::uint8_t* overflow, float* distance, float* direction, float* t, float* position,
                                  float* normal, float* geometric_normal, int* global_primitive_id,
                                  int* tape_primitive_id, float* tape_barycentric, float* tape_restart_epsilon,
                                  std::uint8_t* tape_restart_branch, std::uint8_t* tape_restart_tie_mask,
                                  std::uint8_t* tape_direction_denominator_branch, const std::uint8_t* input_active,
                                  int* capacity_failure_state, int failure_bit, bool input_active_any,
                                  int64_t segment_count, int64_t hit_capacity) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t slot_count = segment_count * hit_capacity;
    if (index < segment_count) {
        num_hits[index] = 0;
        reached_target[index] = 0u;
        overflow[index] = 0u;
        distance[index] = 0.0f;
        direction[index * 3 + 0] = 0.0f;
        direction[index * 3 + 1] = 0.0f;
        direction[index * 3 + 2] = 0.0f;
        if (tape_direction_denominator_branch != nullptr)
            tape_direction_denominator_branch[index] = 0u;
        if (!input_active_any && input_active != nullptr && input_active[index] != 0u)
            atomicOr(capacity_failure_state, failure_bit);
    }
    if (index >= slot_count)
        return;
    valid[index] = 0u;
    t[index] = -1.0f;
    global_primitive_id[index] = -1;
    for (int axis = 0; axis < 3; ++axis) {
        position[index * 3 + axis] = 0.0f;
        normal[index * 3 + axis] = 0.0f;
        geometric_normal[index * 3 + axis] = 0.0f;
    }
    if (tape_primitive_id != nullptr) {
        tape_primitive_id[index] = -1;
        tape_barycentric[index * 2 + 0] = 0.0f;
        tape_barycentric[index * 2 + 1] = 0.0f;
        tape_restart_epsilon[index] = 0.0f;
        tape_restart_branch[index] = SegmentPenetrationRestartConstant;
        tape_restart_tie_mask[index] = 0u;
    }
}

__global__ void sanitize_kernel(std::uint8_t* valid, int* num_hits, std::uint8_t* reached_target, float* distance,
                                float* direction, float* t, float* position, float* normal, float* geometric_normal,
                                int* global_primitive_id, int* tape_primitive_id, float* tape_barycentric,
                                float* tape_restart_epsilon, std::uint8_t* tape_restart_branch,
                                std::uint8_t* tape_restart_tie_mask, std::uint8_t* tape_direction_denominator_branch,
                                const int* capacity_failure_state, int64_t segment_count, int64_t hit_capacity) {
    if (capacity_failure_state[0] == 0)
        return;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t slot_count = segment_count * hit_capacity;
    if (index < segment_count) {
        num_hits[index] = 0;
        reached_target[index] = 0u;
        distance[index] = 0.0f;
        direction[index * 3 + 0] = 0.0f;
        direction[index * 3 + 1] = 0.0f;
        direction[index * 3 + 2] = 0.0f;
        if (tape_direction_denominator_branch != nullptr)
            tape_direction_denominator_branch[index] = 0u;
    }
    if (index >= slot_count)
        return;
    valid[index] = 0u;
    t[index] = -1.0f;
    global_primitive_id[index] = -1;
    for (int axis = 0; axis < 3; ++axis) {
        position[index * 3 + axis] = 0.0f;
        normal[index * 3 + axis] = 0.0f;
        geometric_normal[index * 3 + axis] = 0.0f;
    }
    if (tape_primitive_id != nullptr) {
        tape_primitive_id[index] = -1;
        tape_barycentric[index * 2 + 0] = 0.0f;
        tape_barycentric[index * 2 + 1] = 0.0f;
        tape_restart_epsilon[index] = 0.0f;
        tape_restart_branch[index] = SegmentPenetrationRestartConstant;
        tape_restart_tie_mask[index] = 0u;
    }
}

__global__ void backward_kernel(
    const float* vertices, const int* faces, const float* origins, const float* targets,
    const std::uint8_t* input_active, const int* failure_state, const std::uint8_t* valid, const float* t,
    const float* position, const float* geometric_normal, const int* primitive_id, const float* barycentric,
    const float* restart_epsilon, const std::uint8_t* restart_branch, const std::uint8_t* restart_tie_mask,
    const std::uint8_t* direction_denominator_branch, const float* grad_distance, int64_t grad_distance_stride0,
    const float* grad_direction, int64_t grad_direction_stride0, int64_t grad_direction_stride1, const float* grad_t,
    int64_t grad_t_stride0, int64_t grad_t_stride1, const float* grad_position, int64_t grad_position_stride0,
    int64_t grad_position_stride1, int64_t grad_position_stride2, const float* grad_normal, int64_t grad_normal_stride0,
    int64_t grad_normal_stride1, int64_t grad_normal_stride2, const float* grad_geometric_normal,
    int64_t grad_geometric_normal_stride0, int64_t grad_geometric_normal_stride1, int64_t grad_geometric_normal_stride2,
    int policy, int64_t segment_count, int64_t hit_capacity, float* grad_vertices, float* grad_origins,
    float* grad_targets) {
    const int64_t segment = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (segment >= segment_count)
        return;
    if (failure_state[0] != 0 || (input_active != nullptr && input_active[segment] == 0u))
        return;

    const float3 source = load3(origins, segment, 3, 1);
    const float3 target = load3(targets, segment, 3, 1);
    const float3 delta = sub3(target, source);
    const float distance = sqrtf(fmaxf(dot3(delta, delta), 0.0f));
    const float direction_floor = policy == SegmentPenetrationEnumeratedFullDistance ? 1.0e-9f : 1.0e-6f;
    const float denominator = direction_denominator_branch[segment] != 0u ? distance : direction_floor;
    const float3 direction = mul3(1.0f / denominator, delta);
    float3 direction_gradient = load3(grad_direction, segment, grad_direction_stride0, grad_direction_stride1);
    float3 current_origin_gradient = make_float3(0.0f, 0.0f, 0.0f);

    for (int64_t slot = hit_capacity; slot-- > 0;) {
        const int64_t row = segment * hit_capacity + slot;
        if (valid[row] == 0u)
            continue;
        const int primitive = primitive_id[row];
        const int i0 = faces[primitive * 3 + 0];
        const int i1 = faces[primitive * 3 + 1];
        const int i2 = faces[primitive * 3 + 2];
        const float3 v0 = load3(vertices, i0, 3, 1);
        const float3 v1 = load3(vertices, i1, 3, 1);
        const float3 v2 = load3(vertices, i2, 3, 1);
        const float3 e1 = sub3(v1, v0);
        const float3 e2 = sub3(v2, v0);
        const float hit_t = t[row];
        const float3 hit_position = load3(position, row, 3, 1);

        float3 position_gradient =
            load3_3d(grad_position, segment, slot, grad_position_stride0, grad_position_stride1, grad_position_stride2);
        position_gradient = add3(position_gradient, current_origin_gradient);
        direction_gradient = add3(direction_gradient, mul3(restart_epsilon[row], current_origin_gradient));
        const float restart_gradient = dot3(current_origin_gradient, direction);
        if (restart_branch[row] == SegmentPenetrationRestartPosition) {
            if (policy == SegmentPenetrationEnumeratedFullDistance) {
                const float position_norm = sqrtf(fmaxf(dot3(hit_position, hit_position), 0.0f));
                if (position_norm > 0.0f) {
                    position_gradient =
                        add3(position_gradient, mul3(restart_gradient * 1.0e-6f / position_norm, hit_position));
                }
            } else {
                const std::uint8_t tie_mask = restart_tie_mask[row];
                const int tie_count = population3(tie_mask);
                if (tie_count > 0) {
                    const float share = restart_gradient * 1.0e-6f / static_cast<float>(tie_count);
                    position_gradient.x += (tie_mask & 1u) != 0u ? share * copysignf(1.0f, hit_position.x) : 0.0f;
                    position_gradient.y += (tie_mask & 2u) != 0u ? share * copysignf(1.0f, hit_position.y) : 0.0f;
                    position_gradient.z += (tie_mask & 4u) != 0u ? share * copysignf(1.0f, hit_position.z) : 0.0f;
                }
            }
        }

        float3 selected_normal_gradient =
            load3_3d(grad_normal, segment, slot, grad_normal_stride0, grad_normal_stride1, grad_normal_stride2);
        const float3 geo = load3(geometric_normal, row, 3, 1);
        if (policy == SegmentPenetrationEnumeratedFullDistance) {
            selected_normal_gradient = normalized_vjp(geo, 1.0e-9f, selected_normal_gradient);
        }
        const float3 normal_gradient =
            add3(selected_normal_gradient, load3_3d(grad_geometric_normal, segment, slot, grad_geometric_normal_stride0,
                                                    grad_geometric_normal_stride1, grad_geometric_normal_stride2));
        float normal_length = 0.0f;
        const float3 face_n = ::rayd::shared::math::triangle_unit_normal(e1, e2, &normal_length);
        const float3 raw_normal_gradient =
            mul3(1.0f / normal_length, sub3(normal_gradient, mul3(dot3(face_n, normal_gradient), face_n)));
        const float3 grad_e1 = cross3(e2, raw_normal_gradient);
        const float3 grad_e2 = cross3(raw_normal_gradient, e1);
        float3 grad_v0 = mul3(-1.0f, add3(grad_e1, grad_e2));
        float3 grad_v1 = grad_e1;
        float3 grad_v2 = grad_e2;

        const float hit_gradient =
            (grad_t == nullptr ? 0.0f : grad_t[segment * grad_t_stride0 + slot * grad_t_stride1]) +
            dot3(position_gradient, direction);
        current_origin_gradient = position_gradient;
        direction_gradient = add3(direction_gradient, mul3(hit_t, position_gradient));
        const float3 lambda = solve_transpose(mul3(-1.0f, direction), e1, e2, make_float3(hit_gradient, 0.0f, 0.0f));
        current_origin_gradient = add3(current_origin_gradient, lambda);
        direction_gradient = add3(direction_gradient, mul3(hit_t, lambda));
        const float u = barycentric[row * 2 + 0];
        const float v = barycentric[row * 2 + 1];
        grad_v0 = sub3(grad_v0, mul3(1.0f - u - v, lambda));
        grad_v1 = sub3(grad_v1, mul3(u, lambda));
        grad_v2 = sub3(grad_v2, mul3(v, lambda));
        if (grad_vertices != nullptr) {
            atomic_add3(grad_vertices, i0, grad_v0);
            atomic_add3(grad_vertices, i1, grad_v1);
            atomic_add3(grad_vertices, i2, grad_v2);
        }
    }

    float3 delta_gradient;
    if (direction_denominator_branch[segment] != 0u) {
        delta_gradient =
            mul3(1.0f / distance, sub3(direction_gradient, mul3(dot3(direction, direction_gradient), direction)));
    } else {
        delta_gradient = mul3(1.0f / direction_floor, direction_gradient);
    }
    const float distance_gradient = grad_distance == nullptr ? 0.0f : grad_distance[segment * grad_distance_stride0];
    if (distance > 0.0f)
        delta_gradient = add3(delta_gradient, mul3(distance_gradient / distance, delta));
    if (grad_origins != nullptr)
        store3(grad_origins, segment, sub3(current_origin_gradient, delta_gradient));
    if (grad_targets != nullptr)
        store3(grad_targets, segment, delta_gradient);
}

__global__ void jvp_kernel(const float* vertices, const int* faces, const float* origins, const float* targets,
                           const std::uint8_t* input_active, const int* failure_state, const std::uint8_t* valid,
                           const float* t, const float* position, const float* geometric_normal,
                           const int* primitive_id, const float* barycentric, const float* restart_epsilon,
                           const std::uint8_t* restart_branch, const std::uint8_t* restart_tie_mask,
                           const std::uint8_t* direction_denominator_branch, const float* tangent_vertices,
                           int64_t tangent_vertices_stride0, int64_t tangent_vertices_stride1,
                           const float* tangent_origins, int64_t tangent_origins_stride0,
                           int64_t tangent_origins_stride1, const float* tangent_targets,
                           int64_t tangent_targets_stride0, int64_t tangent_targets_stride1, int policy,
                           int64_t segment_count, int64_t hit_capacity, float* tangent_distance,
                           float* tangent_direction, float* tangent_t, float* tangent_position, float* tangent_normal,
                           float* tangent_geometric_normal) {
    const int64_t segment = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (segment >= segment_count)
        return;
    if (failure_state[0] != 0 || (input_active != nullptr && input_active[segment] == 0u))
        return;

    const float3 source = load3(origins, segment, 3, 1);
    const float3 target = load3(targets, segment, 3, 1);
    const float3 delta = sub3(target, source);
    const float distance = sqrtf(fmaxf(dot3(delta, delta), 0.0f));
    const float3 dsource = load3(tangent_origins, segment, tangent_origins_stride0, tangent_origins_stride1);
    const float3 dtarget = load3(tangent_targets, segment, tangent_targets_stride0, tangent_targets_stride1);
    const float3 ddelta = sub3(dtarget, dsource);
    const float ddistance = distance > 0.0f ? dot3(delta, ddelta) / distance : 0.0f;
    const float direction_floor = policy == SegmentPenetrationEnumeratedFullDistance ? 1.0e-9f : 1.0e-6f;
    const float denominator = direction_denominator_branch[segment] != 0u ? distance : direction_floor;
    const float3 direction = mul3(1.0f / denominator, delta);
    const float3 ddirection = direction_denominator_branch[segment] != 0u
                                  ? mul3(1.0f / distance, sub3(ddelta, mul3(ddistance, direction)))
                                  : mul3(1.0f / direction_floor, ddelta);
    tangent_distance[segment] = ddistance;
    store3(tangent_direction, segment, ddirection);
    float3 current_origin_tangent = dsource;

    for (int64_t slot = 0; slot < hit_capacity; ++slot) {
        const int64_t row = segment * hit_capacity + slot;
        if (valid[row] == 0u)
            continue;
        const int primitive = primitive_id[row];
        const int i0 = faces[primitive * 3 + 0];
        const int i1 = faces[primitive * 3 + 1];
        const int i2 = faces[primitive * 3 + 2];
        const float3 v0 = load3(vertices, i0, 3, 1);
        const float3 v1 = load3(vertices, i1, 3, 1);
        const float3 v2 = load3(vertices, i2, 3, 1);
        const float3 dv0 = load3(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
        const float3 dv1 = load3(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
        const float3 dv2 = load3(tangent_vertices, i2, tangent_vertices_stride0, tangent_vertices_stride1);
        const float3 e1 = sub3(v1, v0);
        const float3 e2 = sub3(v2, v0);
        const float3 de1 = sub3(dv1, dv0);
        const float3 de2 = sub3(dv2, dv0);
        const float u = barycentric[row * 2 + 0];
        const float v = barycentric[row * 2 + 1];
        const float hit_t = t[row];
        const float3 vertex_tangent = add3(add3(mul3(1.0f - u - v, dv0), mul3(u, dv1)), mul3(v, dv2));
        const float3 rhs = sub3(add3(current_origin_tangent, mul3(hit_t, ddirection)), vertex_tangent);
        const float hit_tangent = solve_columns(mul3(-1.0f, direction), e1, e2, rhs).x;
        const float3 position_tangent =
            add3(current_origin_tangent, add3(mul3(hit_tangent, direction), mul3(hit_t, ddirection)));
        const float3 geo_tangent = ::rayd::shared::math::triangle_unit_normal_jvp(e1, e2, de1, de2);
        const float3 selected_normal_tangent =
            policy == SegmentPenetrationEnumeratedFullDistance
                ? normalized_jvp(load3(geometric_normal, row, 3, 1), 1.0e-9f, geo_tangent)
                : geo_tangent;
        tangent_t[row] = hit_tangent;
        store3(tangent_position, row, position_tangent);
        store3(tangent_normal, row, selected_normal_tangent);
        store3(tangent_geometric_normal, row, geo_tangent);

        float restart_tangent = 0.0f;
        if (restart_branch[row] == SegmentPenetrationRestartPosition) {
            const float3 hit_position = load3(position, row, 3, 1);
            if (policy == SegmentPenetrationEnumeratedFullDistance) {
                const float position_norm = sqrtf(fmaxf(dot3(hit_position, hit_position), 0.0f));
                restart_tangent =
                    position_norm > 0.0f ? 1.0e-6f * dot3(hit_position, position_tangent) / position_norm : 0.0f;
            } else {
                const std::uint8_t tie_mask = restart_tie_mask[row];
                const int tie_count = population3(tie_mask);
                if (tie_count > 0) {
                    const float inverse_count = 1.0f / static_cast<float>(tie_count);
                    restart_tangent =
                        1.0e-6f * inverse_count *
                        (((tie_mask & 1u) != 0u ? copysignf(1.0f, hit_position.x) * position_tangent.x : 0.0f) +
                         ((tie_mask & 2u) != 0u ? copysignf(1.0f, hit_position.y) * position_tangent.y : 0.0f) +
                         ((tie_mask & 4u) != 0u ? copysignf(1.0f, hit_position.z) * position_tangent.z : 0.0f));
                }
            }
        }
        current_origin_tangent =
            add3(position_tangent, add3(mul3(restart_epsilon[row], ddirection), mul3(restart_tangent, direction)));
    }
}

} // namespace

void segment_penetration_initialize_cuda(SegmentPenetrationNativeOutputs& outputs, const at::Tensor* input_active,
                                         const at::Tensor& capacity_failure_state, std::int32_t failure_bit,
                                         bool input_active_any) {
    const int64_t segment_count = outputs.result.num_hits.size(0);
    const int64_t hit_capacity = outputs.result.valid.size(1);
    const int64_t work_count = (std::max)(segment_count, segment_count * hit_capacity);
    if (work_count == 0)
        return;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((work_count + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(capacity_failure_state.get_device()).stream();
    initialize_kernel<<<blocks, threads, 0, stream>>>(
        byte_mask_ptr(outputs.result.valid), outputs.result.num_hits.data_ptr<int>(),
        byte_mask_ptr(outputs.result.reached_target), byte_mask_ptr(outputs.result.overflow),
        outputs.result.distance.data_ptr<float>(), outputs.result.direction.data_ptr<float>(),
        outputs.result.t.data_ptr<float>(), outputs.result.position.data_ptr<float>(),
        outputs.result.normal.data_ptr<float>(), outputs.result.geometric_normal.data_ptr<float>(),
        outputs.result.global_primitive_id.data_ptr<int>(),
        outputs.tape.primitive_id.defined() ? outputs.tape.primitive_id.data_ptr<int>() : nullptr,
        outputs.tape.barycentric.defined() ? outputs.tape.barycentric.data_ptr<float>() : nullptr,
        outputs.tape.restart_epsilon.defined() ? outputs.tape.restart_epsilon.data_ptr<float>() : nullptr,
        outputs.tape.restart_branch.defined() ? outputs.tape.restart_branch.data_ptr<std::uint8_t>() : nullptr,
        outputs.tape.restart_tie_mask.defined() ? outputs.tape.restart_tie_mask.data_ptr<std::uint8_t>() : nullptr,
        outputs.tape.direction_denominator_branch.defined() ? byte_mask_ptr(outputs.tape.direction_denominator_branch)
                                                            : nullptr,
        optional_byte_mask_ptr(input_active), capacity_failure_state.data_ptr<int>(), failure_bit, input_active_any,
        segment_count, hit_capacity);
    cuda_check(cudaGetLastError(), "segment_penetration initialize kernel");
}

void segment_penetration_sanitize_cuda(SegmentPenetrationNativeOutputs& outputs,
                                       const at::Tensor& capacity_failure_state) {
    const int64_t segment_count = outputs.result.num_hits.size(0);
    const int64_t hit_capacity = outputs.result.valid.size(1);
    const int64_t work_count = (std::max)(segment_count, segment_count * hit_capacity);
    if (work_count == 0)
        return;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((work_count + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(capacity_failure_state.get_device()).stream();
    sanitize_kernel<<<blocks, threads, 0, stream>>>(
        byte_mask_ptr(outputs.result.valid), outputs.result.num_hits.data_ptr<int>(),
        byte_mask_ptr(outputs.result.reached_target), outputs.result.distance.data_ptr<float>(),
        outputs.result.direction.data_ptr<float>(), outputs.result.t.data_ptr<float>(),
        outputs.result.position.data_ptr<float>(), outputs.result.normal.data_ptr<float>(),
        outputs.result.geometric_normal.data_ptr<float>(), outputs.result.global_primitive_id.data_ptr<int>(),
        outputs.tape.primitive_id.defined() ? outputs.tape.primitive_id.data_ptr<int>() : nullptr,
        outputs.tape.barycentric.defined() ? outputs.tape.barycentric.data_ptr<float>() : nullptr,
        outputs.tape.restart_epsilon.defined() ? outputs.tape.restart_epsilon.data_ptr<float>() : nullptr,
        outputs.tape.restart_branch.defined() ? outputs.tape.restart_branch.data_ptr<std::uint8_t>() : nullptr,
        outputs.tape.restart_tie_mask.defined() ? outputs.tape.restart_tie_mask.data_ptr<std::uint8_t>() : nullptr,
        outputs.tape.direction_denominator_branch.defined() ? byte_mask_ptr(outputs.tape.direction_denominator_branch)
                                                            : nullptr,
        capacity_failure_state.data_ptr<int>(), segment_count, hit_capacity);
    cuda_check(cudaGetLastError(), "segment_penetration sanitize kernel");
}

SegmentPenetrationBackwardOutputs segment_penetration_backward_cuda(
    const SceneCache& scene, const rayd::torch::SegmentPenetrationBackwardRequest& request) {
    const int64_t segment_count = request.primal.origins.size(0);
    const int64_t hit_capacity = request.primal.hit_capacity;
    const auto* active = request.primal.input_active.has_value() ? &*request.primal.input_active : nullptr;
    const auto optional = [](const std::optional<at::Tensor>& value) -> const at::Tensor* {
        return value.has_value() && value->defined() ? &*value : nullptr;
    };
    const at::Tensor* grad_distance = optional(request.grad_distance);
    const at::Tensor* grad_direction = optional(request.grad_direction);
    const at::Tensor* grad_t = optional(request.grad_t);
    const at::Tensor* grad_position = optional(request.grad_position);
    const at::Tensor* grad_normal = optional(request.grad_normal);
    const at::Tensor* grad_geometric_normal = optional(request.grad_geometric_normal);
    SegmentPenetrationBackwardOutputs outputs;
    outputs.grad_vertices = request.need_grad_vertices ? at::empty_like(scene.global_vertices) : at::Tensor();
    outputs.grad_origins = request.need_grad_origins ? at::empty_like(request.primal.origins) : at::Tensor();
    outputs.grad_targets = request.need_grad_targets ? at::empty_like(request.primal.targets) : at::Tensor();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(request.primal.origins.get_device()).stream();
    for (const at::Tensor* tensor : {&outputs.grad_vertices, &outputs.grad_origins, &outputs.grad_targets}) {
        if (tensor->defined() && tensor->numel() != 0)
            cuda_check(cudaMemsetAsync(tensor->data_ptr<float>(), 0,
                                       static_cast<size_t>(tensor->numel()) * sizeof(float), stream),
                       "cudaMemsetAsync(segment penetration gradient)");
    }
    if (segment_count == 0 ||
        (!request.need_grad_vertices && !request.need_grad_origins && !request.need_grad_targets)) {
        return outputs;
    }
    constexpr int threads = 128;
    const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
    backward_kernel<<<blocks, threads, 0, stream>>>(
        scene.global_vertices.data_ptr<float>(), scene.global_faces.data_ptr<int>(),
        request.primal.origins.data_ptr<float>(), request.primal.targets.data_ptr<float>(),
        optional_byte_mask_ptr(active), request.primal.capacity_failure_state.data_ptr<int>(),
        reinterpret_cast<const std::uint8_t*>(request.tape.result.valid.data_ptr<bool>()),
        request.tape.result.t.data_ptr<float>(), request.tape.result.position.data_ptr<float>(),
        request.tape.result.geometric_normal.data_ptr<float>(), request.tape.tape_primitive_id.data_ptr<int>(),
        request.tape.tape_barycentric.data_ptr<float>(), request.tape.tape_restart_epsilon.data_ptr<float>(),
        request.tape.tape_restart_branch.data_ptr<std::uint8_t>(),
        request.tape.tape_restart_tie_mask.data_ptr<std::uint8_t>(),
        reinterpret_cast<const std::uint8_t*>(request.tape.tape_direction_denominator_branch.data_ptr<bool>()),
        grad_distance == nullptr ? nullptr : grad_distance->data_ptr<float>(), optional_stride(grad_distance, 0),
        grad_direction == nullptr ? nullptr : grad_direction->data_ptr<float>(), optional_stride(grad_direction, 0),
        optional_stride(grad_direction, 1), grad_t == nullptr ? nullptr : grad_t->data_ptr<float>(),
        optional_stride(grad_t, 0), optional_stride(grad_t, 1),
        grad_position == nullptr ? nullptr : grad_position->data_ptr<float>(), optional_stride(grad_position, 0),
        optional_stride(grad_position, 1), optional_stride(grad_position, 2),
        grad_normal == nullptr ? nullptr : grad_normal->data_ptr<float>(), optional_stride(grad_normal, 0),
        optional_stride(grad_normal, 1), optional_stride(grad_normal, 2),
        grad_geometric_normal == nullptr ? nullptr : grad_geometric_normal->data_ptr<float>(),
        optional_stride(grad_geometric_normal, 0), optional_stride(grad_geometric_normal, 1),
        optional_stride(grad_geometric_normal, 2), static_cast<int>(request.primal.policy), segment_count, hit_capacity,
        request.need_grad_vertices ? outputs.grad_vertices.data_ptr<float>() : nullptr,
        request.need_grad_origins ? outputs.grad_origins.data_ptr<float>() : nullptr,
        request.need_grad_targets ? outputs.grad_targets.data_ptr<float>() : nullptr);
    cuda_check(cudaGetLastError(), "segment_penetration backward kernel");
    return outputs;
}

SegmentPenetrationJvpOutputs segment_penetration_jvp_cuda(const SceneCache& scene,
                                                          const rayd::torch::SegmentPenetrationJvpRequest& request) {
    const int64_t segment_count = request.primal.origins.size(0);
    const int64_t hit_capacity = request.primal.hit_capacity;
    const auto* active = request.primal.input_active.has_value() ? &*request.primal.input_active : nullptr;
    const auto optional = [](const std::optional<at::Tensor>& value) -> const at::Tensor* {
        return value.has_value() && value->defined() ? &*value : nullptr;
    };
    const at::Tensor* tangent_vertices = optional(request.tangent_vertices);
    const at::Tensor* tangent_origins = optional(request.tangent_origins);
    const at::Tensor* tangent_targets = optional(request.tangent_targets);
    SegmentPenetrationJvpOutputs outputs{
        at::empty({segment_count}, request.primal.origins.options()),
        at::empty({segment_count, 3}, request.primal.origins.options()),
        at::empty({segment_count, hit_capacity}, request.primal.origins.options()),
        at::empty({segment_count, hit_capacity, 3}, request.primal.origins.options()),
        at::empty({segment_count, hit_capacity, 3}, request.primal.origins.options()),
        at::empty({segment_count, hit_capacity, 3}, request.primal.origins.options()),
    };
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(request.primal.origins.get_device()).stream();
    for (const at::Tensor* tensor :
         {&outputs.tangent_distance, &outputs.tangent_direction, &outputs.tangent_t, &outputs.tangent_position,
          &outputs.tangent_normal, &outputs.tangent_geometric_normal}) {
        if (tensor->numel() != 0)
            cuda_check(cudaMemsetAsync(tensor->data_ptr<float>(), 0,
                                       static_cast<size_t>(tensor->numel()) * sizeof(float), stream),
                       "cudaMemsetAsync(segment penetration tangent)");
    }
    if (segment_count == 0)
        return outputs;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
    jvp_kernel<<<blocks, threads, 0, stream>>>(
        scene.global_vertices.data_ptr<float>(), scene.global_faces.data_ptr<int>(),
        request.primal.origins.data_ptr<float>(), request.primal.targets.data_ptr<float>(),
        optional_byte_mask_ptr(active), request.primal.capacity_failure_state.data_ptr<int>(),
        reinterpret_cast<const std::uint8_t*>(request.tape.result.valid.data_ptr<bool>()),
        request.tape.result.t.data_ptr<float>(), request.tape.result.position.data_ptr<float>(),
        request.tape.result.geometric_normal.data_ptr<float>(), request.tape.tape_primitive_id.data_ptr<int>(),
        request.tape.tape_barycentric.data_ptr<float>(), request.tape.tape_restart_epsilon.data_ptr<float>(),
        request.tape.tape_restart_branch.data_ptr<std::uint8_t>(),
        request.tape.tape_restart_tie_mask.data_ptr<std::uint8_t>(),
        reinterpret_cast<const std::uint8_t*>(request.tape.tape_direction_denominator_branch.data_ptr<bool>()),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        optional_stride(tangent_vertices, 0), optional_stride(tangent_vertices, 1),
        tangent_origins == nullptr ? nullptr : tangent_origins->data_ptr<float>(), optional_stride(tangent_origins, 0),
        optional_stride(tangent_origins, 1), tangent_targets == nullptr ? nullptr : tangent_targets->data_ptr<float>(),
        optional_stride(tangent_targets, 0), optional_stride(tangent_targets, 1),
        static_cast<int>(request.primal.policy), segment_count, hit_capacity,
        outputs.tangent_distance.data_ptr<float>(), outputs.tangent_direction.data_ptr<float>(),
        outputs.tangent_t.data_ptr<float>(), outputs.tangent_position.data_ptr<float>(),
        outputs.tangent_normal.data_ptr<float>(), outputs.tangent_geometric_normal.data_ptr<float>());
    cuda_check(cudaGetLastError(), "segment_penetration jvp kernel");
    return outputs;
}

} // namespace rayd::torch_backend
