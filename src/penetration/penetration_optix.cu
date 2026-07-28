#include <optix.h>
#include <optix_device.h>

#include <cuda_runtime.h>

#include <rayd/shared/contracts.h>
#include <src/penetration/segment_penetration_params.h>

namespace rayd::torch_backend {

extern "C" {
__constant__ SegmentPenetrationParams params;
}

namespace {

__forceinline__ __device__ float3 load3(const float *values, int index) {
    return make_float3(
        values[index * 3 + 0],
        values[index * 3 + 1],
        values[index * 3 + 2]);
}

__forceinline__ __device__ void store3(float *values, int index, float3 value) {
    values[index * 3 + 0] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

__forceinline__ __device__ float dot_ordered(float3 a, float3 b) {
    return (a.x * b.x + a.y * b.y) + a.z * b.z;
}

__forceinline__ __device__ float length_ordered(float3 value) {
    return sqrtf(fmaxf(dot_ordered(value, value), 0.0f));
}

__forceinline__ __device__ float3 sub_ordered(float3 a, float3 b) {
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z);
}

__forceinline__ __device__ float3 add_scaled_ordered(float3 a, float3 b, float scale) {
    return make_float3(
        a.x + b.x * scale,
        a.y + b.y * scale,
        a.z + b.z * scale);
}

__forceinline__ __device__ float3 restart_point_ordered(
    float3 position,
    float3 direction,
    float epsilon) {
    // The frozen Channel baseline evaluated direction * epsilon and the
    // following position + offset as two Torch operations. Preserve that
    // float32 rounding boundary inside this otherwise fused OptiX program.
    return make_float3(
        __fadd_rn(position.x, __fmul_rn(direction.x, epsilon)),
        __fadd_rn(position.y, __fmul_rn(direction.y, epsilon)),
        __fadd_rn(position.z, __fmul_rn(direction.z, epsilon)));
}

__forceinline__ __device__ float3 divide_ordered(float3 value, float denominator) {
    return make_float3(
        value.x / denominator,
        value.y / denominator,
        value.z / denominator);
}

__forceinline__ __device__ float3 cross_ordered(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float3 face_normal(int primitive_id) {
    const int i0 = params.faces[primitive_id * 3 + 0];
    const int i1 = params.faces[primitive_id * 3 + 1];
    const int i2 = params.faces[primitive_id * 3 + 2];
    const float3 p0 = load3(params.vertices, i0);
    const float3 p1 = load3(params.vertices, i1);
    const float3 p2 = load3(params.vertices, i2);
    const float3 normal = cross_ordered(sub_ordered(p1, p0), sub_ordered(p2, p0));
    const float inverse_length = rsqrtf(fmaxf(dot_ordered(normal, normal), 1.0e-20f));
    return make_float3(
        normal.x * inverse_length,
        normal.y * inverse_length,
        normal.z * inverse_length);
}

__forceinline__ __device__ float3 enumerated_normal(float3 geometric_normal) {
    const float denominator = fmaxf(length_ordered(geometric_normal), 1.0e-9f);
    return divide_ordered(geometric_normal, denominator);
}

__forceinline__ __device__ float restart_floor() {
    return fmaxf(params.scene_diagonal * 1.0e-6f, 1.0e-6f);
}

__forceinline__ __device__ float enumerated_restart(
    float3 position,
    std::uint8_t *branch) {
    const float candidate = length_ordered(position) * 1.0e-6f;
    const float floor = restart_floor();
    *branch = candidate > floor
        ? SegmentPenetrationRestartPosition
        : SegmentPenetrationRestartConstant;
    return fmaxf(candidate, floor);
}

__forceinline__ __device__ float monte_carlo_restart(
    float3 position,
    std::uint8_t *branch,
    std::uint8_t *tie_mask) {
    const float ax = fabsf(position.x);
    const float ay = fabsf(position.y);
    const float az = fabsf(position.z);
    const float position_norm = fmaxf(fmaxf(ax, ay), az);
    const float candidate = position_norm * 1.0e-6f;
    const float floor = restart_floor();
    const bool position_branch = candidate > floor;
    *branch = position_branch
        ? SegmentPenetrationRestartPosition
        : SegmentPenetrationRestartConstant;
    *tie_mask = position_branch
        ? static_cast<std::uint8_t>(
              (ax == position_norm ? 1u : 0u) |
              (ay == position_norm ? 2u : 0u) |
              (az == position_norm ? 4u : 0u))
        : 0u;
    return fmaxf(candidate, floor);
}

} // namespace

extern "C" __global__ void __closesthit__segment_penetration() {
    const float2 barycentric = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(static_cast<unsigned int>(optixGetInstanceId()));
    optixSetPayload_2(__float_as_uint(barycentric.x));
    optixSetPayload_3(__float_as_uint(barycentric.y));
    optixSetPayload_4(static_cast<unsigned int>(optixGetPrimitiveIndex()));
}

extern "C" __global__ void __miss__segment_penetration() {
}

extern "C" __global__ void __raygen__segment_penetration() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.segment_count))
        return;
    if (params.capacity_failure_state[0] != 0)
        return;
    if (params.input_active != nullptr && params.input_active[lane] == 0u)
        return;

    const float3 source = load3(params.origins, static_cast<int>(lane));
    const float3 target = load3(params.targets, static_cast<int>(lane));
    const float3 delta = sub_ordered(target, source);
    if (!isfinite(delta.x) || !isfinite(delta.y) || !isfinite(delta.z)) {
        atomicOr(params.capacity_failure_state, params.failure_bit);
        return;
    }
    const float distance = length_ordered(delta);
    if (!isfinite(distance)) {
        atomicOr(params.capacity_failure_state, params.failure_bit);
        return;
    }

    const float direction_floor =
        params.policy == SegmentPenetrationEnumeratedFullDistance ? 1.0e-9f : 1.0e-6f;
    const bool direction_uses_length = distance > direction_floor;
    const float3 direction = divide_ordered(delta, fmaxf(distance, direction_floor));
    params.distance[lane] = distance;
    store3(params.direction, static_cast<int>(lane), direction);
    if (params.tape_direction_denominator_branch != nullptr) {
        params.tape_direction_denominator_branch[lane] =
            direction_uses_length ? 1u : 0u;
    }

    float remaining = distance;
    if (params.policy == SegmentPenetrationEnumeratedFullDistance) {
        if (distance <= 1.0e-9f) {
            params.reached_target[lane] = 1u;
            return;
        }
    } else {
        const float target_norm = fmaxf(
            fmaxf(fabsf(target.x), fabsf(target.y)),
            fabsf(target.z));
        const float target_epsilon = fmaxf(
            target_norm * 1.0e-6f,
            restart_floor());
        remaining = fmaxf(distance - target_epsilon, 0.0f);
        if (remaining <= 0.0f) {
            params.reached_target[lane] = 1u;
            return;
        }
    }

    float3 current_origin = source;
    float traveled = 0.0f;
    for (int probe = 0; probe <= params.hit_capacity; ++probe) {
        if (remaining <= 0.0f) {
            params.reached_target[lane] = 1u;
            return;
        }

        unsigned int payload_t = 0x7f800000u;
        unsigned int payload_shape = 0xFFFFFFFFu;
        unsigned int payload_u = 0u;
        unsigned int payload_v = 0u;
        unsigned int payload_local_primitive = 0xFFFFFFFFu;
        const float trace_tmax =
            params.policy == SegmentPenetrationMonteCarloTargetInset
            ? nextafterf(remaining, __uint_as_float(0x7f800000u))
            : remaining;
        optixTrace(
            params.traversable,
            current_origin,
            direction,
            rayd::shared::SmallEpsilon,
            trace_tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            payload_t,
            payload_shape,
            payload_u,
            payload_v,
            payload_local_primitive);

        const float hit_t = __uint_as_float(payload_t);
        const int shape_id = static_cast<int>(payload_shape);
        const int local_primitive = static_cast<int>(payload_local_primitive);
        const int global_primitive =
            shape_id >= 0 && shape_id < params.mesh_count && local_primitive >= 0
            ? params.face_offsets[shape_id] + local_primitive
            : rayd::shared::InvalidSignedId;
        const bool accepted = params.policy == SegmentPenetrationEnumeratedFullDistance
            ? global_primitive >= 0 && isfinite(hit_t) && hit_t < remaining
            : global_primitive >= 0 && hit_t > 0.0f && hit_t <= remaining;
        if (!accepted) {
            params.reached_target[lane] = 1u;
            return;
        }
        if (probe == params.hit_capacity) {
            params.overflow[lane] = 1u;
            atomicOr(params.capacity_failure_state, params.failure_bit);
            return;
        }

        const int row = static_cast<int>(lane) * params.hit_capacity + probe;
        const float3 position = add_scaled_ordered(current_origin, direction, hit_t);
        const float3 geometric_normal = face_normal(global_primitive);
        const float3 selected_normal =
            params.policy == SegmentPenetrationEnumeratedFullDistance
            ? enumerated_normal(geometric_normal)
            : geometric_normal;
        params.valid[row] = 1u;
        params.num_hits[lane] = probe + 1;
        params.t[row] = hit_t;
        store3(params.position, row, position);
        store3(params.normal, row, selected_normal);
        store3(params.geometric_normal, row, geometric_normal);
        params.global_primitive_id[row] = global_primitive;

        std::uint8_t restart_branch = SegmentPenetrationRestartConstant;
        std::uint8_t restart_tie_mask = 0u;
        const float epsilon = params.policy == SegmentPenetrationEnumeratedFullDistance
            ? enumerated_restart(position, &restart_branch)
            : monte_carlo_restart(position, &restart_branch, &restart_tie_mask);
        if (params.tape_primitive_id != nullptr) {
            params.tape_primitive_id[row] = global_primitive;
            params.tape_barycentric[row * 2 + 0] = __uint_as_float(payload_u);
            params.tape_barycentric[row * 2 + 1] = __uint_as_float(payload_v);
            params.tape_restart_epsilon[row] = epsilon;
            params.tape_restart_branch[row] = restart_branch;
            params.tape_restart_tie_mask[row] = restart_tie_mask;
        }

        current_origin = restart_point_ordered(position, direction, epsilon);
        if (params.policy == SegmentPenetrationEnumeratedFullDistance) {
            traveled = (traveled + hit_t) + epsilon;
            remaining = fmaxf(distance - traveled, 0.0f);
        } else {
            remaining = (remaining - hit_t) - epsilon;
        }
    }
}

} // namespace rayd::torch_backend
