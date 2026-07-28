#include <src/scene/geometry_kernels.h>
#include <src/reflection/kernels.h>
#include <src/runtime/math.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <vector>

namespace rayd::torch_backend {

namespace {

const bool *optional_bool_ptr(const at::Tensor &active) {
    if (!active.defined() || active.numel() == 0)
        return nullptr;
    return active.data_ptr<bool>();
}

void zero_float_tensor_async(const at::Tensor &tensor, cudaStream_t stream) {
    if (tensor.defined() && tensor.numel() > 0) {
        cudaMemsetAsync(tensor.data_ptr<float>(), 0, static_cast<size_t>(tensor.numel()) * sizeof(float), stream);
    }
}

int64_t optional_stride(const at::Tensor *tensor, int64_t dim) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0 || tensor->dim() <= dim)
        return 0;
    return tensor->stride(dim);
}

__device__ float read_scalar_or_zero(const float *base, int64_t index, int64_t stride0) {
    return base == nullptr ? 0.f : base[index * stride0];
}

__device__ float3 read_vec3_or_zero(const float *base, int64_t index, int64_t stride0, int64_t stride1) {
    return base == nullptr ? make_float3(0.f, 0.f, 0.f)
                           : make_float3(base[index * stride0 + 0 * stride1],
                                         base[index * stride0 + 1 * stride1],
                                         base[index * stride0 + 2 * stride1]);
}

__device__ void write_vec3_or_skip(float *base, int64_t index, float3 value) {
    if (base == nullptr)
        return;
    base[index * 3 + 0] = value.x;
    base[index * 3 + 1] = value.y;
    base[index * 3 + 2] = value.z;
}

__device__ float det3(float3 c0, float3 c1, float3 c2) {
    return dot3(c0, cross3(c1, c2));
}

__device__ float3 solve_columns(float3 c0, float3 c1, float3 c2, float3 rhs) {
    float determinant = det3(c0, c1, c2);
    if (fabsf(determinant) < 1e-12f)
        determinant = copysignf(1e-12f, determinant == 0.f ? 1.f : determinant);
    const float inv_det = 1.f / determinant;
    return make_float3(
        det3(rhs, c1, c2) * inv_det,
        det3(c0, rhs, c2) * inv_det,
        det3(c0, c1, rhs) * inv_det);
}

__device__ float3 solve_transpose_columns(float3 c0, float3 c1, float3 c2, float3 rhs) {
    const float3 r0 = make_float3(c0.x, c1.x, c2.x);
    const float3 r1 = make_float3(c0.y, c1.y, c2.y);
    const float3 r2 = make_float3(c0.z, c1.z, c2.z);
    return solve_columns(r0, r1, r2, rhs);
}

__device__ float3 bary3_from_tape(const float *tape_bary, int tape_bary_width, int64_t ray_idx) {
    if (tape_bary_width == 2) {
        const float u = tape_bary[ray_idx * 2 + 0];
        const float v = tape_bary[ray_idx * 2 + 1];
        return make_float3(1.f - u - v, u, v);
    }
    return make_f3(tape_bary + ray_idx * 3);
}

__device__ int64_t ray_bounce_index(int64_t ray_idx, int64_t bounce, int64_t max_bounces) {
    return ray_idx * max_bounces + bounce;
}

__device__ int64_t ray_bounce_vec3_index(int64_t ray_idx, int64_t bounce, int64_t max_bounces) {
    return (ray_idx * max_bounces + bounce) * 3;
}

__device__ int64_t state_vec3_index(int64_t bounce, int64_t ray_idx, int64_t ray_count) {
    return (bounce * ray_count + ray_idx) * 3;
}

__device__ float3 read_ray_vec3(const float *base, int64_t ray_idx) {
    return make_f3(base + ray_idx * 3);
}

__device__ float3 read_ray_bounce_vec3(const float *base, int64_t ray_idx, int64_t bounce, int64_t max_bounces) {
    return make_f3(base + ray_bounce_vec3_index(ray_idx, bounce, max_bounces));
}

__device__ void write_ray_vec3(float *base, int64_t ray_idx, float3 value) {
    base[ray_idx * 3 + 0] = value.x;
    base[ray_idx * 3 + 1] = value.y;
    base[ray_idx * 3 + 2] = value.z;
}

__device__ void write_ray_bounce_vec3(float *base, int64_t ray_idx, int64_t bounce, int64_t max_bounces, float3 value) {
    const int64_t idx = ray_bounce_vec3_index(ray_idx, bounce, max_bounces);
    base[idx + 0] = value.x;
    base[idx + 1] = value.y;
    base[idx + 2] = value.z;
}

__device__ float3 read_state_vec3(const float *base, int64_t bounce, int64_t ray_idx, int64_t ray_count) {
    return make_f3(base + state_vec3_index(bounce, ray_idx, ray_count));
}

__device__ void write_state_vec3(float *base, int64_t bounce, int64_t ray_idx, int64_t ray_count, float3 value) {
    const int64_t idx = state_vec3_index(bounce, ray_idx, ray_count);
    base[idx + 0] = value.x;
    base[idx + 1] = value.y;
    base[idx + 2] = value.z;
}

__device__ float read_grad_t_or_zero(
    const float *base,
    int grad_dim,
    int64_t stride0,
    int64_t stride1,
    int64_t ray_idx,
    int64_t bounce) {
    if (base == nullptr)
        return 0.f;
    if (grad_dim <= 1)
        return base[ray_idx * stride0];
    return base[ray_idx * stride0 + bounce * stride1];
}

__device__ float3 read_grad_image_or_zero(
    const float *base,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t ray_idx,
    int64_t bounce) {
    return base == nullptr ? make_float3(0.f, 0.f, 0.f)
                           : make_float3(base[ray_idx * stride0 + bounce * stride1 + 0 * stride2],
                                         base[ray_idx * stride0 + bounce * stride1 + 1 * stride2],
                                         base[ray_idx * stride0 + bounce * stride1 + 2 * stride2]);
}

__device__ float3 normal_from_edges(float3 e1, float3 e2, float *length_out) {
    const float3 q = cross3(e1, e2);
    const float length = sqrtf(fmaxf(dot3(q, q), 1e-20f));
    if (length_out != nullptr)
        *length_out = length;
    return mul3(1.f / length, q);
}

__device__ float3 normal_jvp(float3 e1, float3 e2, float3 de1, float3 de2) {
    float length = 0.f;
    const float3 n = normal_from_edges(e1, e2, &length);
    const float3 dq = add3(cross3(de1, e2), cross3(e1, de2));
    return mul3(1.f / length, sub3(dq, mul3(dot3(n, dq), n)));
}

__global__ void reflection_chain_state_kernel(
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const float *__restrict__ tape_hit_points,
    const float *__restrict__ tape_normals,
    const float *__restrict__ image_sources,
    int64_t ray_count,
    int64_t max_bounces,
    float *__restrict__ origins,
    float *__restrict__ directions,
    float *__restrict__ image_states) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    float3 origin = read_ray_vec3(ray_o, ray_idx);
    float3 direction = read_ray_vec3(ray_d, ray_idx);
    float3 image_state = origin;
    for (int64_t bounce = 0; bounce < max_bounces; ++bounce) {
        write_state_vec3(origins, bounce, ray_idx, ray_count, origin);
        write_state_vec3(directions, bounce, ray_idx, ray_count, direction);
        write_state_vec3(image_states, bounce, ray_idx, ray_count, image_state);
        if (bounce + 1 >= max_bounces)
            continue;
        const float3 normal = read_ray_bounce_vec3(tape_normals, ray_idx, bounce, max_bounces);
        const float3 hit = read_ray_bounce_vec3(tape_hit_points, ray_idx, bounce, max_bounces);
        const float dir_dot_n = dot3(direction, normal);
        const float3 next_direction = sub3(direction, mul3(2.f * dir_dot_n, normal));
        origin = add3(hit, mul3(static_cast<float>(kRayBias), next_direction));
        direction = next_direction;
        image_state = read_ray_bounce_vec3(image_sources, ray_idx, bounce, max_bounces);
    }
}

__global__ void reflection_chain_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_hit_points,
    const float *__restrict__ tape_normals,
    const float *__restrict__ origins,
    const float *__restrict__ directions,
    const float *__restrict__ image_states,
    const float *__restrict__ grad_t,
    int grad_t_dim,
    int64_t grad_t_stride0,
    int64_t grad_t_stride1,
    const float *__restrict__ grad_image_sources,
    int64_t grad_image_stride0,
    int64_t grad_image_stride1,
    int64_t grad_image_stride2,
    int64_t ray_count,
    int64_t max_bounces,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d,
    float *__restrict__ grad_ray_tmax) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    write_ray_vec3(grad_ray_o, ray_idx, make_float3(0.f, 0.f, 0.f));
    write_ray_vec3(grad_ray_d, ray_idx, make_float3(0.f, 0.f, 0.f));
    grad_ray_tmax[ray_idx] = 0.f;

    float3 grad_origin_next = make_float3(0.f, 0.f, 0.f);
    float3 grad_direction_next = make_float3(0.f, 0.f, 0.f);
    float3 grad_image_next = make_float3(0.f, 0.f, 0.f);

    for (int64_t bounce = max_bounces - 1; bounce >= 0; --bounce) {
        const int64_t rb = ray_bounce_index(ray_idx, bounce, max_bounces);
        const int prim_id = tape_prim_id[rb];
        const bool active_b = prim_id >= 0 && (active == nullptr || active[ray_idx]);
        if (!active_b) {
            grad_origin_next = make_float3(0.f, 0.f, 0.f);
            grad_direction_next = make_float3(0.f, 0.f, 0.f);
            grad_image_next = make_float3(0.f, 0.f, 0.f);
            continue;
        }

        const float3 normal = read_ray_bounce_vec3(tape_normals, ray_idx, bounce, max_bounces);
        const float3 hit = read_ray_bounce_vec3(tape_hit_points, ray_idx, bounce, max_bounces);
        const float3 direction = read_state_vec3(directions, bounce, ray_idx, ray_count);
        const float3 origin = read_state_vec3(origins, bounce, ray_idx, ray_count);
        const float3 image_before = read_state_vec3(image_states, bounce, ray_idx, ray_count);

        const float3 grad_image_out = add3(
            read_grad_image_or_zero(
                grad_image_sources,
                grad_image_stride0,
                grad_image_stride1,
                grad_image_stride2,
                ray_idx,
                bounce),
            grad_image_next);
        const float3 image_delta = sub3(image_before, hit);
        const float image_dist = dot3(image_delta, normal);
        const float image_gdotn = dot3(grad_image_out, normal);
        const float3 grad_image_prev = sub3(grad_image_out, mul3(2.f * image_gdotn, normal));

        float3 grad_p = mul3(2.f * image_gdotn, normal);
        float3 grad_signed_n =
            mul3(-2.f, add3(mul3(image_gdotn, image_delta), mul3(image_dist, grad_image_out)));

        grad_p = add3(grad_p, grad_origin_next);
        const float3 grad_reflected =
            add3(grad_direction_next, mul3(static_cast<float>(kRayBias), grad_origin_next));
        const float dir_dot_n = dot3(direction, normal);
        const float refl_gdotn = dot3(grad_reflected, normal);
        const float3 grad_direction_current =
            sub3(grad_reflected, mul3(2.f * refl_gdotn, normal));
        grad_signed_n = sub3(
            grad_signed_n,
            mul3(2.f, add3(mul3(refl_gdotn, direction), mul3(dir_dot_n, grad_reflected))));

        const int i0 = faces[prim_id * 3 + 0];
        const int i1 = faces[prim_id * 3 + 1];
        const int i2 = faces[prim_id * 3 + 2];
        const float3 v0 = make_f3(vertices + i0 * 3);
        const float3 v1 = make_f3(vertices + i1 * 3);
        const float3 v2 = make_f3(vertices + i2 * 3);
        const float3 e1 = sub3(v1, v0);
        const float3 e2 = sub3(v2, v0);
        const float3 raw_normal = normal_from_edges(e1, e2, nullptr);
        const float sign = dot3(raw_normal, normal) >= 0.f ? 1.f : -1.f;
        const float3 grad_raw_n = mul3(sign, grad_signed_n);

        float3 g_vertices0 = make_float3(0.f, 0.f, 0.f);
        float3 g_vertices1 = make_float3(0.f, 0.f, 0.f);
        float3 g_vertices2 = make_float3(0.f, 0.f, 0.f);
        const float normal_length = sqrtf(fmaxf(dot3(cross3(e1, e2), cross3(e1, e2)), 1e-20f));
        const float3 gq = mul3(
            1.f / normal_length,
            sub3(grad_raw_n, mul3(dot3(raw_normal, grad_raw_n), raw_normal)));
        const float3 ge1_normal = cross3(e2, gq);
        const float3 ge2_normal = cross3(gq, e1);
        g_vertices0 = sub3(g_vertices0, add3(ge1_normal, ge2_normal));
        g_vertices1 = add3(g_vertices1, ge1_normal);
        g_vertices2 = add3(g_vertices2, ge2_normal);

        const float3 d = direction;
        const float3 c0 = mul3(-1.f, d);
        const float3 bary = bary3_from_tape(tape_bary + rb * tape_bary_width, tape_bary_width, 0);
        const float gt = read_grad_t_or_zero(
            grad_t,
            grad_t_dim,
            grad_t_stride0,
            grad_t_stride1,
            ray_idx,
            bounce);
        const float t_bar_from_p = dot3(grad_p, d);
        float3 grad_ray_o_hit = grad_p;
        const float3 gy = make_float3(gt + t_bar_from_p, 0.f, 0.f);
        const float3 lambda = solve_transpose_columns(c0, e1, e2, gy);
        grad_ray_o_hit = add3(grad_ray_o_hit, lambda);
        const float solved_t = solve_columns(c0, e1, e2, sub3(origin, v0)).x;
        const float3 grad_ray_d_hit = mul3(solved_t, add3(lambda, grad_p));

        g_vertices0 = sub3(g_vertices0, mul3(bary.x, lambda));
        g_vertices1 = sub3(g_vertices1, mul3(bary.y, lambda));
        g_vertices2 = sub3(g_vertices2, mul3(bary.z, lambda));
        atomic_add3(grad_vertices, i0, g_vertices0);
        atomic_add3(grad_vertices, i1, g_vertices1);
        atomic_add3(grad_vertices, i2, g_vertices2);

        grad_origin_next = grad_ray_o_hit;
        grad_direction_next = add3(grad_ray_d_hit, grad_direction_current);
        grad_image_next = grad_image_prev;
    }

    write_ray_vec3(grad_ray_o, ray_idx, add3(grad_origin_next, grad_image_next));
    write_ray_vec3(grad_ray_d, ray_idx, grad_direction_next);
}

__global__ void reflection_chain_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_hit_points,
    const float *__restrict__ tape_normals,
    const float *__restrict__ tangent_vertices,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    const float *__restrict__ tangent_ray_o,
    int64_t tangent_ray_o_stride0,
    int64_t tangent_ray_o_stride1,
    const float *__restrict__ tangent_ray_d,
    int64_t tangent_ray_d_stride0,
    int64_t tangent_ray_d_stride1,
    const float *__restrict__ image_sources,
    int64_t ray_count,
    int64_t max_bounces,
    float *__restrict__ tangent_t,
    float *__restrict__ tangent_image_sources) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    float3 origin = read_ray_vec3(ray_o, ray_idx);
    float3 direction = read_ray_vec3(ray_d, ray_idx);
    float3 tangent_origin =
        read_vec3_or_zero(tangent_ray_o, ray_idx, tangent_ray_o_stride0, tangent_ray_o_stride1);
    float3 tangent_direction =
        read_vec3_or_zero(tangent_ray_d, ray_idx, tangent_ray_d_stride0, tangent_ray_d_stride1);
    float3 image_state = origin;
    float3 tangent_image_state = tangent_origin;

    for (int64_t bounce = 0; bounce < max_bounces; ++bounce) {
        const int64_t rb = ray_bounce_index(ray_idx, bounce, max_bounces);
        const int prim_id = tape_prim_id[rb];
        const bool active_b = prim_id >= 0 && (active == nullptr || active[ray_idx]);
        const float3 normal = read_ray_bounce_vec3(tape_normals, ray_idx, bounce, max_bounces);
        const float3 hit = read_ray_bounce_vec3(tape_hit_points, ray_idx, bounce, max_bounces);

        float tangent_hit_t = 0.f;
        float3 tangent_hit = make_float3(0.f, 0.f, 0.f);
        float3 tangent_normal = make_float3(0.f, 0.f, 0.f);
        if (active_b) {
            const int i0 = faces[prim_id * 3 + 0];
            const int i1 = faces[prim_id * 3 + 1];
            const int i2 = faces[prim_id * 3 + 2];
            const float3 v0 = make_f3(vertices + i0 * 3);
            const float3 v1 = make_f3(vertices + i1 * 3);
            const float3 v2 = make_f3(vertices + i2 * 3);
            const float3 dv0 =
                read_vec3_or_zero(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
            const float3 dv1 =
                read_vec3_or_zero(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
            const float3 dv2 =
                read_vec3_or_zero(tangent_vertices, i2, tangent_vertices_stride0, tangent_vertices_stride1);
            const float3 e1 = sub3(v1, v0);
            const float3 e2 = sub3(v2, v0);
            const float3 de1 = sub3(dv1, dv0);
            const float3 de2 = sub3(dv2, dv0);
            const float3 bary = bary3_from_tape(tape_bary + rb * tape_bary_width, tape_bary_width, 0);
            const float solved_t = solve_columns(
                                      mul3(-1.f, direction),
                                      e1,
                                      e2,
                                      sub3(origin, v0))
                                      .x;
            const float3 vertex_tangent =
                add3(add3(mul3(bary.x, dv0), mul3(bary.y, dv1)), mul3(bary.z, dv2));
            const float3 rhs = sub3(
                add3(tangent_origin, mul3(solved_t, tangent_direction)),
                vertex_tangent);
            const float3 dy = solve_columns(mul3(-1.f, direction), e1, e2, rhs);
            tangent_hit_t = dy.x;
            tangent_hit = add3(tangent_origin, add3(mul3(dy.x, direction), mul3(solved_t, tangent_direction)));
            const float3 raw_normal = normal_from_edges(e1, e2, nullptr);
            const float sign = dot3(raw_normal, normal) >= 0.f ? 1.f : -1.f;
            tangent_normal = mul3(sign, normal_jvp(e1, e2, de1, de2));
        }
        tangent_t[ray_bounce_index(ray_idx, bounce, max_bounces)] = active_b ? tangent_hit_t : 0.f;

        const float3 image_delta = sub3(image_state, hit);
        const float3 tangent_image_delta = sub3(tangent_image_state, tangent_hit);
        const float image_dist = dot3(image_delta, normal);
        const float tangent_image_dist =
            dot3(tangent_image_delta, normal) + dot3(image_delta, tangent_normal);
        const float3 next_image_state = sub3(image_state, mul3(2.f * image_dist, normal));
        float3 next_tangent_image_state =
            sub3(tangent_image_state,
                 mul3(2.f, add3(mul3(tangent_image_dist, normal), mul3(image_dist, tangent_normal))));
        if (!active_b) {
            next_tangent_image_state = make_float3(0.f, 0.f, 0.f);
        }
        write_ray_bounce_vec3(
            tangent_image_sources,
            ray_idx,
            bounce,
            max_bounces,
            next_tangent_image_state);

        const float dir_dot_n = dot3(direction, normal);
        const float tangent_dir_dot_n =
            dot3(tangent_direction, normal) + dot3(direction, tangent_normal);
        const float3 next_direction = sub3(direction, mul3(2.f * dir_dot_n, normal));
        float3 next_tangent_direction =
            sub3(tangent_direction,
                 mul3(2.f, add3(mul3(tangent_dir_dot_n, normal), mul3(dir_dot_n, tangent_normal))));
        const float3 next_origin = add3(hit, mul3(static_cast<float>(kRayBias), next_direction));
        float3 next_tangent_origin =
            add3(tangent_hit, mul3(static_cast<float>(kRayBias), next_tangent_direction));
        if (!active_b) {
            next_tangent_origin = make_float3(0.f, 0.f, 0.f);
            next_tangent_direction = make_float3(0.f, 0.f, 0.f);
        }

        origin = next_origin;
        direction = next_direction;
        tangent_origin = next_tangent_origin;
        tangent_direction = next_tangent_direction;
        image_state = next_image_state;
        tangent_image_state = next_tangent_image_state;
        if (bounce + 1 < max_bounces) {
            image_state = read_ray_bounce_vec3(image_sources, ray_idx, bounce, max_bounces);
        }
    }
}

__global__ void refl_epc_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_t,
    const float *__restrict__ grad_field_real,
    const float *__restrict__ grad_field_imag,
    const float *__restrict__ grad_path_length,
    int64_t grad_field_real_stride0,
    int64_t grad_field_imag_stride0,
    int64_t grad_path_length_stride0,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_source,
    float *__restrict__ grad_receiver) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    write_vec3_or_skip(grad_source, ray_idx, make_float3(0.f, 0.f, 0.f));
    write_vec3_or_skip(grad_receiver, ray_idx, make_float3(0.f, 0.f, 0.f));
    if (active != nullptr && !active[ray_idx])
        return;
    const int prim_id = tape_prim_id[ray_idx];
    if (prim_id < 0)
        return;

    const float t = tape_t[ray_idx];
    const float inv_denom = 1.f / (1.f + t);
    const float s = sinf(t);
    const float c = cosf(t);
    const float real_dt = -s * inv_denom - c * inv_denom * inv_denom;
    const float imag_dt = c * inv_denom - s * inv_denom * inv_denom;
    const float gt =
        read_scalar_or_zero(grad_path_length, ray_idx, grad_path_length_stride0) +
        read_scalar_or_zero(grad_field_real, ray_idx, grad_field_real_stride0) * real_dt +
        read_scalar_or_zero(grad_field_imag, ray_idx, grad_field_imag_stride0) * imag_dt;
    if (gt == 0.f)
        return;

    const int i0 = faces[prim_id * 3 + 0];
    const int i1 = faces[prim_id * 3 + 1];
    const int i2 = faces[prim_id * 3 + 2];
    const float3 v0 = make_f3(vertices + i0 * 3);
    const float3 v1 = make_f3(vertices + i1 * 3);
    const float3 v2 = make_f3(vertices + i2 * 3);
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 o = make_f3(source + ray_idx * 3);
    const float3 r = make_f3(receiver + ray_idx * 3);
    const float3 d = sub3(r, o);
    const float3 c0 = mul3(-1.f, d);
    const float3 lambda = solve_transpose_columns(c0, e1, e2, make_float3(gt, 0.f, 0.f));

    if (grad_source != nullptr || grad_receiver != nullptr) {
        const float solved_t = solve_columns(c0, e1, e2, sub3(o, v0)).x;
        const float3 grad_ray_d = mul3(solved_t, lambda);
        write_vec3_or_skip(grad_source, ray_idx, sub3(lambda, grad_ray_d));
        write_vec3_or_skip(grad_receiver, ray_idx, grad_ray_d);
    }

    if (grad_vertices == nullptr)
        return;
    const float3 bary = bary3_from_tape(tape_bary, tape_bary_width, ray_idx);
    atomic_add3(grad_vertices, i0, mul3(-bary.x, lambda));
    atomic_add3(grad_vertices, i1, mul3(-bary.y, lambda));
    atomic_add3(grad_vertices, i2, mul3(-bary.z, lambda));
}

__global__ void refl_epc_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_t,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_source,
    const float *__restrict__ tangent_receiver,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_source_stride0,
    int64_t tangent_source_stride1,
    int64_t tangent_receiver_stride0,
    int64_t tangent_receiver_stride1,
    int64_t ray_count,
    float *__restrict__ tangent_field_real,
    float *__restrict__ tangent_field_imag,
    float *__restrict__ tangent_path_length) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    tangent_field_real[ray_idx] = 0.f;
    tangent_field_imag[ray_idx] = 0.f;
    tangent_path_length[ray_idx] = 0.f;
    if (active != nullptr && !active[ray_idx])
        return;
    const int prim_id = tape_prim_id[ray_idx];
    if (prim_id < 0)
        return;

    const int i0 = faces[prim_id * 3 + 0];
    const int i1 = faces[prim_id * 3 + 1];
    const int i2 = faces[prim_id * 3 + 2];
    const float3 v0 = make_f3(vertices + i0 * 3);
    const float3 v1 = make_f3(vertices + i1 * 3);
    const float3 v2 = make_f3(vertices + i2 * 3);
    const float3 dv0 = read_vec3_or_zero(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 dv1 = read_vec3_or_zero(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 dv2 = read_vec3_or_zero(tangent_vertices, i2, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 o = make_f3(source + ray_idx * 3);
    const float3 r = make_f3(receiver + ray_idx * 3);
    const float3 d = sub3(r, o);
    const float3 do_t = read_vec3_or_zero(tangent_source, ray_idx, tangent_source_stride0, tangent_source_stride1);
    const float3 dr_t = read_vec3_or_zero(tangent_receiver, ray_idx, tangent_receiver_stride0, tangent_receiver_stride1);
    const float3 dd_t = sub3(dr_t, do_t);
    const float3 bary = bary3_from_tape(tape_bary, tape_bary_width, ray_idx);
    const float3 c0 = mul3(-1.f, d);
    const float solved_t = solve_columns(c0, e1, e2, sub3(o, v0)).x;
    const float3 vertex_tangent =
        add3(add3(mul3(bary.x, dv0), mul3(bary.y, dv1)), mul3(bary.z, dv2));
    const float3 rhs = sub3(add3(do_t, mul3(solved_t, dd_t)), vertex_tangent);
    const float tangent_t = solve_columns(c0, e1, e2, rhs).x;

    const float t = tape_t[ray_idx];
    const float inv_denom = 1.f / (1.f + t);
    const float s = sinf(t);
    const float c = cosf(t);
    const float real_dt = -s * inv_denom - c * inv_denom * inv_denom;
    const float imag_dt = c * inv_denom - s * inv_denom * inv_denom;
    tangent_field_real[ray_idx] = real_dt * tangent_t;
    tangent_field_imag[ray_idx] = imag_dt * tangent_t;
    tangent_path_length[ray_idx] = tangent_t;
}

} // namespace

ReflectionBackwardOutputs reflection_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t) {
    (void)ray_tmax;
    at::Tensor grad_t_flat = grad_t.dim() == 1 ? grad_t : grad_t.select(1, 0);
    IntersectBackwardOutputs hit_grad = intersect_backward_t_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t_flat,
        grad_t_flat.stride(0),
        true,
        true,
        true,
        true);
    return {
        hit_grad.grad_vertices,
        hit_grad.grad_ray_o,
        hit_grad.grad_ray_d,
        hit_grad.grad_ray_tmax,
    };
}

ReflectionBackwardOutputs reflection_chain_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor &image_sources,
    const at::Tensor *grad_t,
    const at::Tensor *grad_image_sources) {
    (void)ray_tmax;
    const int64_t ray_count = ray_o.size(0);
    const int64_t max_bounces = tape_prim_id.size(1);
    if (max_bounces == 1 && grad_t != nullptr && grad_t->numel() != 0 && grad_image_sources == nullptr) {
        at::Tensor grad_t_flat = grad_t->dim() == 1 ? *grad_t : grad_t->select(1, 0);
        IntersectBackwardOutputs hit_grad = intersect_backward_t_cuda(
            vertices,
            faces,
            ray_o,
            ray_d,
            active,
            tape_prim_id.select(1, 0),
            tape_barycentric.select(1, 0),
            grad_t_flat,
            grad_t_flat.stride(0),
            true,
            true,
            true,
            true);
        return {
            hit_grad.grad_vertices,
            hit_grad.grad_ray_o,
            hit_grad.grad_ray_d,
            hit_grad.grad_ray_tmax,
        };
    }

    ReflectionBackwardOutputs out;
    out.grad_vertices = at::empty_like(vertices);
    out.grad_ray_o = at::empty_like(ray_o);
    out.grad_ray_d = at::empty_like(ray_d);
    out.grad_ray_tmax = at::empty({ray_count}, ray_o.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0) {
        return out;
    }

    at::Tensor origins = at::empty({max_bounces, ray_count, 3}, ray_o.options());
    at::Tensor directions = at::empty({max_bounces, ray_count, 3}, ray_o.options());
    at::Tensor image_states = at::empty({max_bounces, ray_count, 3}, ray_o.options());
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_chain_state_kernel<<<blocks, threads, 0, stream>>>(
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        tape_hit_points.data_ptr<float>(),
        tape_normals.data_ptr<float>(),
        image_sources.data_ptr<float>(),
        ray_count,
        max_bounces,
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        image_states.data_ptr<float>());
    reflection_chain_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(2)),
        tape_hit_points.data_ptr<float>(),
        tape_normals.data_ptr<float>(),
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        image_states.data_ptr<float>(),
        grad_t == nullptr ? nullptr : grad_t->data_ptr<float>(),
        grad_t == nullptr ? 0 : static_cast<int>(grad_t->dim()),
        optional_stride(grad_t, 0),
        optional_stride(grad_t, 1),
        grad_image_sources == nullptr ? nullptr : grad_image_sources->data_ptr<float>(),
        optional_stride(grad_image_sources, 0),
        optional_stride(grad_image_sources, 1),
        optional_stride(grad_image_sources, 2),
        ray_count,
        max_bounces,
        out.grad_vertices.data_ptr<float>(),
        out.grad_ray_o.data_ptr<float>(),
        out.grad_ray_d.data_ptr<float>(),
        out.grad_ray_tmax.data_ptr<float>());
    return out;
}

ReflectionJvpOutputs reflection_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_ray_o,
    const at::Tensor &tangent_ray_d,
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    IntersectJvpOutputs hit_jvp = intersect_jvp_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d);
    return {
        hit_jvp.tangent_t.reshape({ray_count, 1}),
        at::zeros_like(image_sources),
    };
}

ReflectionJvpOutputs reflection_chain_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    const int64_t max_bounces = tape_prim_id.size(1);
    ReflectionJvpOutputs out;
    out.tangent_t = at::empty({ray_count, max_bounces}, ray_o.options());
    out.tangent_image_sources = at::empty_like(image_sources);
    if (ray_count == 0) {
        return out;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_chain_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(2)),
        tape_hit_points.data_ptr<float>(),
        tape_normals.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        tangent_ray_o == nullptr ? nullptr : tangent_ray_o->data_ptr<float>(),
        optional_stride(tangent_ray_o, 0),
        optional_stride(tangent_ray_o, 1),
        tangent_ray_d == nullptr ? nullptr : tangent_ray_d->data_ptr<float>(),
        optional_stride(tangent_ray_d, 0),
        optional_stride(tangent_ray_d, 1),
        image_sources.data_ptr<float>(),
        ray_count,
        max_bounces,
        out.tangent_t.data_ptr<float>(),
        out.tangent_image_sources.data_ptr<float>());
    return out;
}

ReflEpcBackwardOutputs refl_epc_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor *grad_field_real,
    const at::Tensor *grad_field_imag,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    const int64_t ray_count = source.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    ReflEpcBackwardOutputs out;
    out.grad_vertices = need_grad_vertices ? at::empty_like(vertices) : at::empty({0, 3}, vertices.options());
    out.grad_source = need_grad_source ? at::empty_like(source) : at::empty({0, 3}, source.options());
    out.grad_receiver = need_grad_receiver ? at::empty_like(receiver) : at::empty({0, 3}, receiver.options());
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0 || (!need_grad_vertices && !need_grad_source && !need_grad_receiver)) {
        return out;
    }
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    refl_epc_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        tape_t.data_ptr<float>(),
        grad_field_real == nullptr ? nullptr : grad_field_real->data_ptr<float>(),
        grad_field_imag == nullptr ? nullptr : grad_field_imag->data_ptr<float>(),
        grad_path_length == nullptr ? nullptr : grad_path_length->data_ptr<float>(),
        optional_stride(grad_field_real, 0),
        optional_stride(grad_field_imag, 0),
        optional_stride(grad_path_length, 0),
        ray_count,
        need_grad_vertices ? out.grad_vertices.data_ptr<float>() : nullptr,
        need_grad_source ? out.grad_source.data_ptr<float>() : nullptr,
        need_grad_receiver ? out.grad_receiver.data_ptr<float>() : nullptr);
    return out;
}

ReflEpcJvpOutputs refl_epc_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver) {
    const int64_t ray_count = source.size(0);
    ReflEpcJvpOutputs out;
    out.tangent_field_real = at::empty({ray_count}, source.options());
    out.tangent_field_imag = at::empty({ray_count}, source.options());
    out.tangent_path_length = at::empty({ray_count}, source.options());
    if (ray_count == 0) {
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    refl_epc_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        tape_t.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        tangent_source == nullptr ? nullptr : tangent_source->data_ptr<float>(),
        tangent_receiver == nullptr ? nullptr : tangent_receiver->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        optional_stride(tangent_source, 0),
        optional_stride(tangent_source, 1),
        optional_stride(tangent_receiver, 0),
        optional_stride(tangent_receiver, 1),
        ray_count,
        out.tangent_field_real.data_ptr<float>(),
        out.tangent_field_imag.data_ptr<float>(),
        out.tangent_path_length.data_ptr<float>());
    return out;
}

} // namespace rayd::torch_backend


// ---- merged from src/reflection/dedup_part.cu ----

#include <src/reflection/dedup.h>
#include <rayd/shared/reflection/dedup.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <string>

#include <src/runtime/native_compat.h>


namespace rayd::torch_backend {

namespace {

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) {
        allocate(count);
    }

    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer &operator=(CudaBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }

        count_ = count;
        if (count_ == 0) {
            return;
        }

        const cudaError_t error =
            cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_);
        require(error == cudaSuccess,
                std::string("reflection_dedup_gpu(): cudaMalloc failed: ") +
                    cudaGetErrorString(error));
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

// Runs the sort/scan passes the shared sequence delegates back to this
// backend. Keeping the CUB calls here keeps their template kernels
// instantiated in this translation unit, exactly as before the sequence
// orchestration moved to the shared layer.
cudaError_t run_dedup_pass(const shared::multipath::ReflectionDedupSequenceParams &params,
                           shared::multipath::ReflectionDedupDevicePass pass) {
    using shared::multipath::ReflectionDedupDevicePass;
    size_t sort_temp_bytes = params.sort_temp_bytes;
    size_t scan_temp_bytes = params.scan_temp_bytes;
    size_t cluster_sort_temp_bytes = params.cluster_sort_temp_bytes;
    switch (pass) {
    case ReflectionDedupDevicePass::kFirstSort:
        return cub::DeviceRadixSort::SortPairs(params.sort_temp,
                                               sort_temp_bytes,
                                               params.keys_in,
                                               params.keys_out,
                                               params.ray_indices_in,
                                               params.ray_indices_out,
                                               params.ray_count,
                                               0,
                                               64,
                                               params.stream);
    case ReflectionDedupDevicePass::kFirstScan:
        return cub::DeviceScan::InclusiveSum(params.scan_temp,
                                             scan_temp_bytes,
                                             params.boundary_flags,
                                             params.hash_group_ids,
                                             params.ray_count,
                                             params.stream);
    case ReflectionDedupDevicePass::kSecondSort:
        return cub::DeviceRadixSort::SortPairs(params.cluster_sort_temp,
                                               cluster_sort_temp_bytes,
                                               params.cluster_keys_in,
                                               params.cluster_keys_out,
                                               params.cluster_ray_indices_in,
                                               params.cluster_ray_indices_out,
                                               params.ray_count,
                                               0,
                                               64,
                                               params.stream);
    case ReflectionDedupDevicePass::kSecondScan:
        return cub::DeviceScan::InclusiveSum(params.scan_temp,
                                             scan_temp_bytes,
                                             params.boundary_flags,
                                             params.unique_path_ids,
                                             params.ray_count,
                                             params.stream);
    }
    return cudaErrorInvalidValue;
}

// Per-step error strings stay in this backend verbatim; the shared sequence
// only reports which step produced the failing CUDA result.
const char *sequence_step_message(shared::multipath::ReflectionDedupSequenceStep step) {
    using shared::multipath::ReflectionDedupSequenceStep;
    switch (step) {
    case ReflectionDedupSequenceStep::kBuildKeys:
        return "reflection_dedup_gpu(): failed to launch build-keys kernel";
    case ReflectionDedupSequenceStep::kFirstSort:
        return "reflection_dedup_gpu(): failed to run first radix sort";
    case ReflectionDedupSequenceStep::kFirstBoundaries:
        return "reflection_dedup_gpu(): failed to launch first boundary kernel";
    case ReflectionDedupSequenceStep::kFirstScan:
        return "reflection_dedup_gpu(): failed to run first scan";
    case ReflectionDedupSequenceStep::kFirstZeroBase:
        return "reflection_dedup_gpu(): failed to launch first id-fix kernel";
    case ReflectionDedupSequenceStep::kSubCluster:
        return "reflection_dedup_gpu(): failed to launch sub-cluster kernel";
    case ReflectionDedupSequenceStep::kSecondSort:
        return "reflection_dedup_gpu(): failed to run second radix sort";
    case ReflectionDedupSequenceStep::kSecondBoundaries:
        return "reflection_dedup_gpu(): failed to launch second boundary kernel";
    case ReflectionDedupSequenceStep::kSecondScan:
        return "reflection_dedup_gpu(): failed to run second scan";
    case ReflectionDedupSequenceStep::kSecondZeroBase:
        return "reflection_dedup_gpu(): failed to launch second id-fix kernel";
    case ReflectionDedupSequenceStep::kCompact:
        return "reflection_dedup_gpu(): failed to launch compact kernel";
    case ReflectionDedupSequenceStep::kNone:
        break;
    }
    return "reflection_dedup_gpu(): dedup sequence failed";
}

void check_sequence_status(const shared::multipath::ReflectionDedupSequenceStatus &status) {
    check_cuda_call(status.error, sequence_step_message(status.step));
}

} // namespace

int reflection_dedup_gpu(
    int device_index,
    int n_rays,
    int max_bounces,
    const int *bounce_count,
    const int *shape_ids,
    const int *prim_ids,
    const float *t,
    const float *bary_u,
    const float *bary_v,
    const float *hit_x,
    const float *hit_y,
    const float *hit_z,
    const float *norm_x,
    const float *norm_y,
    const float *norm_z,
    const float *img_x,
    const float *img_y,
    const float *img_z,
    const int *face_offsets,
    int n_meshes,
    const int *canonical_prim_table,
    int canonical_table_size,
    float image_source_tolerance,
    int *out_bounce_count,
    int *out_shape_ids,
    int *out_prim_ids,
    float *out_t,
    float *out_bary_u,
    float *out_bary_v,
    float *out_hit_x,
    float *out_hit_y,
    float *out_hit_z,
    float *out_norm_x,
    float *out_norm_y,
    float *out_norm_z,
    float *out_img_x,
    float *out_img_y,
    float *out_img_z,
    int *out_discovery_count,
    int *out_representative_ray_index) {
    require(n_rays >= 0, "reflection_dedup_gpu(): n_rays must be non-negative.");
    require(max_bounces > 0,
            "reflection_dedup_gpu(): max_bounces must be positive.");

    if (n_rays == 0) {
        return 0;
    }

    c10::cuda::CUDAGuard guard(device_index);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream(device_index));

    constexpr int block_size = 256;
    const int block_count = (n_rays + block_size - 1) / block_size;

    CudaBuffer<uint64_t> keys_in(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> keys_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> ray_indices_in(static_cast<size_t>(n_rays));
    CudaBuffer<int> ray_indices_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> boundary_flags(static_cast<size_t>(n_rays));
    CudaBuffer<int> hash_group_ids(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> cluster_keys_in(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> cluster_keys_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> cluster_ray_indices_in(static_cast<size_t>(n_rays));
    CudaBuffer<int> cluster_ray_indices_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> unique_path_ids(static_cast<size_t>(n_rays));
    CudaBuffer<int> unique_count_device(1);

    check_cuda_call(cudaMemsetAsync(out_discovery_count,
                                    0,
                                    sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear discovery counts");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(out_representative_ray_index,
                                    0xFF,
                                    sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear representative indices");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(unique_count_device.get(),
                                    0,
                                    sizeof(int),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear unique counter");
    audit_cuda_memset_async();

    size_t sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    sort_temp_size,
                                                    keys_in.get(),
                                                    keys_out.get(),
                                                    ray_indices_in.get(),
                                                    ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to size first radix sort");
    CudaBuffer<char> sort_temp(std::max<size_t>(sort_temp_size, 1));

    size_t scan_temp_size = 0;
    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(nullptr,
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  hash_group_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to size first scan");
    CudaBuffer<char> scan_temp(std::max<size_t>(scan_temp_size, 1));

    size_t cluster_sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    cluster_sort_temp_size,
                                                    cluster_keys_in.get(),
                                                    cluster_keys_out.get(),
                                                    cluster_ray_indices_in.get(),
                                                    cluster_ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to size second radix sort");
    CudaBuffer<char> cluster_sort_temp(std::max<size_t>(cluster_sort_temp_size, 1));

    audit_cuda_kernel_launch("reflection_dedup_build_keys_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    audit_cub_sort();
    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    audit_cub_scan();
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    audit_cuda_kernel_launch("reflection_dedup_sub_cluster_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    audit_cub_sort();
    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    audit_cub_scan();
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    audit_cuda_kernel_launch("reflection_dedup_compact_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));

    shared::multipath::ReflectionDedupSequenceParams sequence{};
    sequence.ray_count = n_rays;
    sequence.max_bounces = max_bounces;
    sequence.bounce_count = bounce_count;
    sequence.shape_ids = shape_ids;
    sequence.prim_ids = prim_ids;
    sequence.face_offsets = face_offsets;
    sequence.mesh_count = n_meshes;
    sequence.canonical_table = canonical_prim_table;
    sequence.canonical_table_size = canonical_table_size;
    sequence.image_source_tolerance = image_source_tolerance;
    sequence.raw_t = t;
    sequence.raw_bary_u = bary_u;
    sequence.raw_bary_v = bary_v;
    sequence.raw_hit_x = hit_x;
    sequence.raw_hit_y = hit_y;
    sequence.raw_hit_z = hit_z;
    sequence.raw_norm_x = norm_x;
    sequence.raw_norm_y = norm_y;
    sequence.raw_norm_z = norm_z;
    sequence.raw_image_x = img_x;
    sequence.raw_image_y = img_y;
    sequence.raw_image_z = img_z;
    sequence.keys_in = keys_in.get();
    sequence.keys_out = keys_out.get();
    sequence.ray_indices_in = ray_indices_in.get();
    sequence.ray_indices_out = ray_indices_out.get();
    sequence.boundary_flags = boundary_flags.get();
    sequence.hash_group_ids = hash_group_ids.get();
    sequence.cluster_keys_in = cluster_keys_in.get();
    sequence.cluster_keys_out = cluster_keys_out.get();
    sequence.cluster_ray_indices_in = cluster_ray_indices_in.get();
    sequence.cluster_ray_indices_out = cluster_ray_indices_out.get();
    sequence.unique_path_ids = unique_path_ids.get();
    sequence.sort_temp = sort_temp.get();
    sequence.sort_temp_bytes = sort_temp_size;
    sequence.scan_temp = scan_temp.get();
    sequence.scan_temp_bytes = scan_temp_size;
    sequence.cluster_sort_temp = cluster_sort_temp.get();
    sequence.cluster_sort_temp_bytes = cluster_sort_temp_size;
    sequence.out_unique_count = unique_count_device.get();
    sequence.out_bounce_count = out_bounce_count;
    sequence.out_shape_ids = out_shape_ids;
    sequence.out_prim_ids = out_prim_ids;
    sequence.out_t = out_t;
    sequence.out_bary_u = out_bary_u;
    sequence.out_bary_v = out_bary_v;
    sequence.out_hit_x = out_hit_x;
    sequence.out_hit_y = out_hit_y;
    sequence.out_hit_z = out_hit_z;
    sequence.out_norm_x = out_norm_x;
    sequence.out_norm_y = out_norm_y;
    sequence.out_norm_z = out_norm_z;
    sequence.out_image_x = out_img_x;
    sequence.out_image_y = out_img_y;
    sequence.out_image_z = out_img_z;
    sequence.out_discovery_count = out_discovery_count;
    sequence.out_representative_ray_index = out_representative_ray_index;
    sequence.run_pass = &run_dedup_pass;
    sequence.stream = stream;
    check_sequence_status(shared::multipath::launch_reflection_dedup_sequence(sequence));

    int unique_count = 0;
    audit_cuda_memcpy_async();
    check_cuda_call(cudaMemcpyAsync(&unique_count,
                                    unique_count_device.get(),
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "reflection_dedup_gpu(): failed to copy unique count");
    audit_cuda_stream_synchronize();
    check_cuda_call(cudaStreamSynchronize(stream),
                    "reflection_dedup_gpu(): failed to finish dedup stream");
    return unique_count;
}

} // namespace rayd::torch_backend


// ---- merged from src/reflection/epc_field_part.cu ----

#include <src/reflection/epc_field.h>
#include <rayd/shared/contracts.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include <src/reflection/complex.cuh>
#include <src/runtime/math.cuh>
#include <src/runtime/native_compat.h>



namespace rayd::torch_backend {

namespace {

constexpr float kReflEps = shared::SmallEpsilon;

static __forceinline__ __device__ bool slot_reflection_coefficients(
    const ReflEpcFieldParams params,
    int slot,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    const float eta_r_value = params.slot_eta_r != nullptr ? params.slot_eta_r[slot] : 1.f;
    const float sigma_value = params.slot_sigma != nullptr ? params.slot_sigma[slot] : 0.f;
    const float gain = params.slot_gain != nullptr ? params.slot_gain[slot] : 1.f;
    const float mu_r_value = params.slot_mu_r != nullptr ? params.slot_mu_r[slot] : 1.f;
    return shared::field::fresnel_reflection_coefficients(
        eta_r_value,
        sigma_value,
        mu_r_value,
        gain,
        params.omega,
        cos_theta,
        r_te,
        r_tm,
        kReflEps);
}

static __forceinline__ __device__ void store_zero_field(
    const ReflEpcFieldParams params,
    int ray_index) {
    if (params.out_valid != nullptr) {
        params.out_valid[ray_index] = 0u;
    }
    if (params.out_field_x_re != nullptr) {
        params.out_field_x_re[ray_index] = 0.f;
        params.out_field_x_im[ray_index] = 0.f;
    }
    if (params.out_field_y_re != nullptr) {
        params.out_field_y_re[ray_index] = 0.f;
        params.out_field_y_im[ray_index] = 0.f;
    }
    if (params.out_field_z_re != nullptr) {
        params.out_field_z_re[ray_index] = 0.f;
        params.out_field_z_im[ray_index] = 0.f;
    }
}

__global__ void reflection_epc_forward_setup_kernel(ReflEpcForwardSetupParams params) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int slot_count = params.n_rays * params.max_bounces;
    const int total = params.n_rays > slot_count ? params.n_rays : slot_count;
    if (idx >= total) {
        return;
    }

    if (idx < params.n_rays) {
        const int base3 = idx * 3;
        const float sx = params.source_aos[base3 + 0];
        const float sy = params.source_aos[base3 + 1];
        const float sz = params.source_aos[base3 + 2];
        const float rx = params.receiver_aos[base3 + 0];
        const float ry = params.receiver_aos[base3 + 1];
        const float rz = params.receiver_aos[base3 + 2];
        const float dx = rx - sx;
        const float dy = ry - sy;
        const float dz = rz - sz;

        params.source_x[idx] = sx;
        params.source_y[idx] = sy;
        params.source_z[idx] = sz;
        params.receiver_x[idx] = rx;
        params.receiver_y[idx] = ry;
        params.receiver_z[idx] = rz;
        params.ray_dx[idx] = dx;
        params.ray_dy[idx] = dy;
        params.ray_dz[idx] = dz;
        params.ray_tmax[idx] = sqrtf(dx * dx + dy * dy + dz * dz);

        params.epc_valid[idx] = 0u;
        params.epc_bounce_count[idx] = 0;
        params.epc_path_length[idx] = __uint_as_float(0x7f800000u);
        params.first_blocked_segment[idx] = -1;
        params.first_blocked_prim[idx] = -1;
        params.first_blocked_group[idx] = -1;

        const int bary = idx * 3;
        params.tape_barycentric[bary + 0] = 0.f;
        params.tape_barycentric[bary + 1] = 0.f;
        params.tape_barycentric[bary + 2] = 0.f;
    }

    if (idx < slot_count) {
        params.point_x[idx] = 0.f;
        params.point_y[idx] = 0.f;
        params.point_z[idx] = 0.f;
        params.trace_prim_ids[idx] = -1;
        params.resolved_prim_ids[idx] = -1;
        params.surface_group_ids[idx] = -1;
        params.plane_normal_x[idx] = 0.f;
        params.plane_normal_y[idx] = 0.f;
        params.plane_normal_z[idx] = 0.f;
    }
}

// Identifier/storage layer for the shared EPC field device body. Every macro
// expands to the exact pre-dedup expression of this backend: nullable reads
// with defaults, first-prim-id prologue exports, and null-guarded output
// writes.
#define RAYD_REFL_EPC_MAKE3(x, y, z) make_f3(x, y, z)
#define RAYD_REFL_EPC_EPS kReflEps
#define RAYD_REFL_EPC_FIELD_PROLOGUE(P, RAY, BASE)                                 \
    if ((P).out_first_resolved_prim_id != nullptr) {                               \
        (P).out_first_resolved_prim_id[(RAY)] =                                    \
            (P).resolved_prim_ids != nullptr ? (P).resolved_prim_ids[(BASE)] : -1; \
    }                                                                              \
    if ((P).out_first_trace_prim_id != nullptr) {                                  \
        (P).out_first_trace_prim_id[(RAY)] =                                       \
            (P).trace_prim_ids != nullptr ? (P).trace_prim_ids[(BASE)] : -1;       \
    }
#define RAYD_REFL_EPC_LOAD_TX_POLARIZATION(P, RAY)                                 \
    float3 tx_polarization = make_f3(1.f, 0.f, 0.f);                               \
    if ((P).tx_pol_x != nullptr) {                                                 \
        const int tx_pol_index = (P).tx_pol_count == 1 ? 0 : (RAY);                \
        tx_polarization = make_f3((P).tx_pol_x[tx_pol_index],                      \
                                  (P).tx_pol_y[tx_pol_index],                      \
                                  (P).tx_pol_z[tx_pol_index]);                     \
    }
#define RAYD_REFL_EPC_STORE_FIELD(P, RAY, FIELD)                                   \
    if ((P).out_valid != nullptr) {                                                \
        (P).out_valid[(RAY)] = 1u;                                                 \
    }                                                                              \
    if ((P).out_field_x_re != nullptr) {                                           \
        (P).out_field_x_re[(RAY)] = (FIELD).x.r;                                   \
        (P).out_field_x_im[(RAY)] = (FIELD).x.i;                                   \
    }                                                                              \
    if ((P).out_field_y_re != nullptr) {                                           \
        (P).out_field_y_re[(RAY)] = (FIELD).y.r;                                   \
        (P).out_field_y_im[(RAY)] = (FIELD).y.i;                                   \
    }                                                                              \
    if ((P).out_field_z_re != nullptr) {                                           \
        (P).out_field_z_re[(RAY)] = (FIELD).z.r;                                   \
        (P).out_field_z_im[(RAY)] = (FIELD).z.i;                                   \
    }

#include <rayd/shared/reflection/epc_field_device.cuh>

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

} // namespace

void reflection_epc_forward_setup_gpu(const ReflEpcForwardSetupParams &params, int device_index) {
    require(params.n_rays >= 0,
            "reflection_epc_forward_setup_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_forward_setup_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    c10::cuda::CUDAGuard guard(device_index);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream(device_index));
    const int slot_count = params.n_rays * params.max_bounces;
    const int total = std::max(params.n_rays, slot_count);
    const int block_size = 128;
    const int block_count = (total + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_forward_setup_kernel",
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(total));
    reflection_epc_forward_setup_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error(
        "reflection_epc_forward_setup_gpu(): failed to launch setup kernel");
}

void reflection_epc_field_gpu(const ReflEpcFieldParams &params, int device_index) {
    require(params.n_rays >= 0,
            "reflection_epc_field_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_field_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    c10::cuda::CUDAGuard guard(device_index);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream(device_index));

    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_field_kernel",
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(params.n_rays));
    reflection_epc_field_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error(
        "reflection_epc_field_gpu(): failed to launch field kernel");
}

} // namespace rayd::torch_backend


// ---- merged from src/reflection/epc_geometry_ad_part.cu ----

// Fixed-winner geometry adjoint / tangent of the reflection EPC path export
// (direct-plane mode) and of the scene's unit face-normal table.
//
// The forward being differentiated is the specular chain the EPC discovery
// raygen solves for an already-selected plane sequence: mirror the source
// through each plane, walk back from the receiver intersecting each plane,
// sum the segment lengths. That chain lives in
// include/rayd/shared/reflection/epc_chain.h together with its
// reverse-mode companion, so the math here has exactly one implementation.
// Which primitive each bounce hits, the containment test and the visibility
// casts are frozen discovery decisions: invalid rows contribute nothing and
// no ray is traced, so no OptiX is involved.

#include <src/reflection/kernels.h>
#include <src/runtime/math.cuh>
#include <rayd/shared/reflection/epc_params.h>
#include <rayd/shared/reflection/epc_chain.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace rayd::torch_backend {

namespace {

namespace shared_math = rayd::shared::math;
namespace shared_reflection = rayd::shared::reflection;

using shared_math::Vec3f;
using shared::optix::ReflEpcMaxBounces;

__device__ Vec3f load_shared_vec3(const float *base, int64_t index) {
    return shared_math::make_vec3(
        base[index * 3 + 0], base[index * 3 + 1], base[index * 3 + 2]);
}

__device__ Vec3f load_strided_vec3_or_zero(
    const float *base,
    int64_t index0,
    int64_t index1,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2) {
    if (base == nullptr) {
        return shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    }
    const int64_t offset = index0 * stride0 + index1 * stride1;
    return shared_math::make_vec3(
        base[offset], base[offset + stride2], base[offset + 2 * stride2]);
}

__device__ void store_vec3(float *base, int64_t index, Vec3f value) {
    base[index * 3 + 0] = value.x;
    base[index * 3 + 1] = value.y;
    base[index * 3 + 2] = value.z;
}

__device__ void atomic_add_shared_vec3(float *base, int index, Vec3f value) {
    atomicAdd(&base[index * 3 + 0], value.x);
    atomicAdd(&base[index * 3 + 1], value.y);
    atomicAdd(&base[index * 3 + 2], value.z);
}

int64_t optional_stride_or_zero(const at::Tensor *tensor, int64_t dim) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0 ||
        tensor->dim() <= dim) {
        return 0;
    }
    return tensor->stride(dim);
}

const float *optional_data_ptr(const at::Tensor *tensor) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0) {
        return nullptr;
    }
    return tensor->data_ptr<float>();
}

// Re-solve the frozen-winner chain for one ray and load its plane inputs.
// Returns false when the row is invalid or the chain guard rejects it (the
// row then contributes exactly zero, matching the frozen discovery record).
__device__ bool load_row_chain(
    const float *source,
    const float *receiver,
    const float *plane_points,
    const float *plane_normals,
    const bool *valid,
    const int *bounce_count,
    int64_t ray_index,
    int max_bounces,
    Vec3f *row_plane_points,
    Vec3f *row_plane_normals,
    shared_reflection::EpcChain<ReflEpcMaxBounces> &chain) {
    if (!valid[ray_index]) {
        return false;
    }
    const int bounces = bounce_count[ray_index];
    if (bounces < 1 || bounces > max_bounces || bounces > ReflEpcMaxBounces) {
        return false;
    }
    const int64_t base = ray_index * max_bounces;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        row_plane_points[bounce] = load_shared_vec3(plane_points, base + bounce);
        row_plane_normals[bounce] = load_shared_vec3(plane_normals, base + bounce);
    }
    return shared_reflection::solve_epc_chain<ReflEpcMaxBounces>(
        row_plane_points,
        row_plane_normals,
        bounces,
        load_shared_vec3(source, ray_index),
        load_shared_vec3(receiver, ray_index),
        chain);
}

__global__ void reflection_epc_paths_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const int *__restrict__ sequence,
    const float *__restrict__ plane_points,
    const float *__restrict__ plane_normals,
    const bool *__restrict__ valid,
    const int *__restrict__ bounce_count,
    const float *__restrict__ grad_points,
    const float *__restrict__ grad_normals,
    const float *__restrict__ grad_path_length,
    int64_t grad_points_stride0,
    int64_t grad_points_stride1,
    int64_t grad_points_stride2,
    int64_t grad_normals_stride0,
    int64_t grad_normals_stride1,
    int64_t grad_normals_stride2,
    int64_t grad_path_length_stride0,
    int64_t ray_count,
    int max_bounces,
    int64_t triangle_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_source,
    float *__restrict__ grad_receiver) {
    const int64_t ray_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_index >= ray_count) {
        return;
    }
    const Vec3f zero = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    if (grad_source != nullptr) {
        store_vec3(grad_source, ray_index, zero);
    }
    if (grad_receiver != nullptr) {
        store_vec3(grad_receiver, ray_index, zero);
    }

    Vec3f row_plane_points[ReflEpcMaxBounces];
    Vec3f row_plane_normals[ReflEpcMaxBounces];
    shared_reflection::EpcChain<ReflEpcMaxBounces> chain;
    if (!load_row_chain(
            source,
            receiver,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            ray_index,
            max_bounces,
            row_plane_points,
            row_plane_normals,
            chain)) {
        return;
    }
    const int bounces = chain.bounces;

    Vec3f grad_hits[ReflEpcMaxBounces];
    Vec3f grad_unit_normals[ReflEpcMaxBounces];
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_hits[bounce] = load_strided_vec3_or_zero(
            grad_points,
            ray_index,
            bounce,
            grad_points_stride0,
            grad_points_stride1,
            grad_points_stride2);
        grad_unit_normals[bounce] = load_strided_vec3_or_zero(
            grad_normals,
            ray_index,
            bounce,
            grad_normals_stride0,
            grad_normals_stride1,
            grad_normals_stride2);
    }
    const float grad_length =
        grad_path_length == nullptr
            ? 0.0f
            : grad_path_length[ray_index * grad_path_length_stride0];

    Vec3f grad_source_row;
    Vec3f grad_receiver_row;
    Vec3f grad_plane_points[ReflEpcMaxBounces];
    Vec3f grad_plane_normals[ReflEpcMaxBounces];
    shared_reflection::adj_solve_epc_chain<ReflEpcMaxBounces>(
        chain,
        row_plane_points,
        row_plane_normals,
        load_shared_vec3(source, ray_index),
        load_shared_vec3(receiver, ray_index),
        grad_hits,
        grad_unit_normals,
        grad_length,
        grad_source_row,
        grad_receiver_row,
        grad_plane_points,
        grad_plane_normals);

    if (grad_source != nullptr) {
        store_vec3(grad_source, ray_index, grad_source_row);
    }
    if (grad_receiver != nullptr) {
        store_vec3(grad_receiver, ray_index, grad_receiver_row);
    }
    if (grad_vertices == nullptr) {
        return;
    }

    // Chain each bounce's plane cotangents to the winner triangle: the anchor
    // is v0(prim) and the plane normal is the unit face normal, exactly how
    // the consumer builds the direct-plane arrays from the scene export.
    const int64_t base = ray_index * max_bounces;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const int prim = sequence[base + bounce];
        if (prim < 0 || prim >= triangle_count) {
            continue;
        }
        const int i0 = faces[prim * 3 + 0];
        const int i1 = faces[prim * 3 + 1];
        const int i2 = faces[prim * 3 + 2];
        const Vec3f v0 = load_shared_vec3(vertices, i0);
        const Vec3f v1 = load_shared_vec3(vertices, i1);
        const Vec3f v2 = load_shared_vec3(vertices, i2);
        Vec3f grad_v0 = grad_plane_points[bounce];
        Vec3f grad_v1 = zero;
        Vec3f grad_v2 = zero;
        shared_reflection::adj_face_normal(
            v0,
            v1,
            v2,
            shared_reflection::face_unit_normal(v0, v1, v2),
            grad_plane_normals[bounce],
            grad_v0,
            grad_v1,
            grad_v2);
        atomic_add_shared_vec3(grad_vertices, i0, grad_v0);
        atomic_add_shared_vec3(grad_vertices, i1, grad_v1);
        atomic_add_shared_vec3(grad_vertices, i2, grad_v2);
    }
}

__global__ void reflection_epc_paths_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const int *__restrict__ sequence,
    const float *__restrict__ plane_points,
    const float *__restrict__ plane_normals,
    const bool *__restrict__ valid,
    const int *__restrict__ bounce_count,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_source,
    const float *__restrict__ tangent_receiver,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_source_stride0,
    int64_t tangent_source_stride1,
    int64_t tangent_receiver_stride0,
    int64_t tangent_receiver_stride1,
    int64_t ray_count,
    int max_bounces,
    int64_t triangle_count,
    float *__restrict__ tangent_points,
    float *__restrict__ tangent_normals,
    float *__restrict__ tangent_path_length) {
    const int64_t ray_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_index >= ray_count) {
        return;
    }
    const Vec3f zero = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    const int64_t base = ray_index * max_bounces;
    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        store_vec3(tangent_points, base + bounce, zero);
        store_vec3(tangent_normals, base + bounce, zero);
    }
    tangent_path_length[ray_index] = 0.0f;

    Vec3f row_plane_points[ReflEpcMaxBounces];
    Vec3f row_plane_normals[ReflEpcMaxBounces];
    shared_reflection::EpcChain<ReflEpcMaxBounces> chain;
    if (!load_row_chain(
            source,
            receiver,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            ray_index,
            max_bounces,
            row_plane_points,
            row_plane_normals,
            chain)) {
        return;
    }
    const int bounces = chain.bounces;

    // Tangent of each plane under vertex tangents: anchor tangent is the
    // winner triangle's v0 tangent, normal tangent is the unit face-normal
    // tangent (the transpose of the vertex chaining in the backward kernel).
    Vec3f tangent_plane_points[ReflEpcMaxBounces];
    Vec3f tangent_plane_normals[ReflEpcMaxBounces];
    for (int bounce = 0; bounce < bounces; ++bounce) {
        tangent_plane_points[bounce] = zero;
        tangent_plane_normals[bounce] = zero;
        if (tangent_vertices == nullptr) {
            continue;
        }
        const int prim = sequence[base + bounce];
        if (prim < 0 || prim >= triangle_count) {
            continue;
        }
        const int i0 = faces[prim * 3 + 0];
        const int i1 = faces[prim * 3 + 1];
        const int i2 = faces[prim * 3 + 2];
        const Vec3f tangent_v0 = load_strided_vec3_or_zero(
            tangent_vertices, i0, 0, tangent_vertices_stride0, 0,
            tangent_vertices_stride1);
        const Vec3f tangent_v1 = load_strided_vec3_or_zero(
            tangent_vertices, i1, 0, tangent_vertices_stride0, 0,
            tangent_vertices_stride1);
        const Vec3f tangent_v2 = load_strided_vec3_or_zero(
            tangent_vertices, i2, 0, tangent_vertices_stride0, 0,
            tangent_vertices_stride1);
        tangent_plane_points[bounce] = tangent_v0;
        tangent_plane_normals[bounce] = shared_reflection::jvp_face_normal(
            load_shared_vec3(vertices, i0),
            load_shared_vec3(vertices, i1),
            load_shared_vec3(vertices, i2),
            tangent_v0,
            tangent_v1,
            tangent_v2);
    }

    Vec3f tangent_hits[ReflEpcMaxBounces];
    Vec3f tangent_unit_normals[ReflEpcMaxBounces];
    float tangent_length = 0.0f;
    shared_reflection::jvp_solve_epc_chain<ReflEpcMaxBounces>(
        chain,
        row_plane_points,
        row_plane_normals,
        load_shared_vec3(source, ray_index),
        load_shared_vec3(receiver, ray_index),
        load_strided_vec3_or_zero(
            tangent_source, ray_index, 0, tangent_source_stride0, 0,
            tangent_source_stride1),
        load_strided_vec3_or_zero(
            tangent_receiver, ray_index, 0, tangent_receiver_stride0, 0,
            tangent_receiver_stride1),
        tangent_plane_points,
        tangent_plane_normals,
        tangent_hits,
        tangent_unit_normals,
        tangent_length);

    for (int bounce = 0; bounce < bounces; ++bounce) {
        store_vec3(tangent_points, base + bounce, tangent_hits[bounce]);
        store_vec3(tangent_normals, base + bounce, tangent_unit_normals[bounce]);
    }
    tangent_path_length[ray_index] = tangent_length;
}

__global__ void scene_face_normals_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ grad_face_normals,
    int64_t grad_stride0,
    int64_t grad_stride1,
    int64_t triangle_count,
    float *__restrict__ grad_vertices) {
    const int64_t face_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face_index >= triangle_count) {
        return;
    }
    const Vec3f grad_normal = load_strided_vec3_or_zero(
        grad_face_normals, face_index, 0, grad_stride0, 0, grad_stride1);
    if (grad_normal.x == 0.0f && grad_normal.y == 0.0f && grad_normal.z == 0.0f) {
        return;
    }
    const int i0 = faces[face_index * 3 + 0];
    const int i1 = faces[face_index * 3 + 1];
    const int i2 = faces[face_index * 3 + 2];
    const Vec3f v0 = load_shared_vec3(vertices, i0);
    const Vec3f v1 = load_shared_vec3(vertices, i1);
    const Vec3f v2 = load_shared_vec3(vertices, i2);
    Vec3f grad_v0 = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_v1 = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_v2 = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    shared_reflection::adj_face_normal(
        v0,
        v1,
        v2,
        shared_reflection::face_unit_normal(v0, v1, v2),
        grad_normal,
        grad_v0,
        grad_v1,
        grad_v2);
    atomic_add_shared_vec3(grad_vertices, i0, grad_v0);
    atomic_add_shared_vec3(grad_vertices, i1, grad_v1);
    atomic_add_shared_vec3(grad_vertices, i2, grad_v2);
}

__global__ void scene_face_normals_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ tangent_vertices,
    int64_t tangent_stride0,
    int64_t tangent_stride1,
    int64_t triangle_count,
    float *__restrict__ tangent_face_normals) {
    const int64_t face_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face_index >= triangle_count) {
        return;
    }
    const int i0 = faces[face_index * 3 + 0];
    const int i1 = faces[face_index * 3 + 1];
    const int i2 = faces[face_index * 3 + 2];
    const Vec3f tangent = shared_reflection::jvp_face_normal(
        load_shared_vec3(vertices, i0),
        load_shared_vec3(vertices, i1),
        load_shared_vec3(vertices, i2),
        load_strided_vec3_or_zero(
            tangent_vertices, i0, 0, tangent_stride0, 0, tangent_stride1),
        load_strided_vec3_or_zero(
            tangent_vertices, i1, 0, tangent_stride0, 0, tangent_stride1),
        load_strided_vec3_or_zero(
            tangent_vertices, i2, 0, tangent_stride0, 0, tangent_stride1));
    store_vec3(tangent_face_normals, face_index, tangent);
}

} // namespace

ReflEpcPathsBackwardOutputs reflection_epc_paths_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count,
    const at::Tensor *grad_points,
    const at::Tensor *grad_normals,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    const int64_t ray_count = source.size(0);
    const int max_bounces = static_cast<int>(sequence.size(1));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    ReflEpcPathsBackwardOutputs out;
    out.grad_vertices = need_grad_vertices
        ? at::empty(vertices.sizes(), vertices.options())
        : at::Tensor();
    out.grad_source =
        need_grad_source ? at::empty(source.sizes(), source.options()) : at::Tensor();
    out.grad_receiver = need_grad_receiver
        ? at::empty(receiver.sizes(), receiver.options())
        : at::Tensor();
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0 ||
        (!need_grad_vertices && !need_grad_source && !need_grad_receiver)) {
        return out;
    }

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_epc_paths_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        sequence.data_ptr<int>(),
        plane_points.data_ptr<float>(),
        plane_normals.data_ptr<float>(),
        valid.data_ptr<bool>(),
        bounce_count.data_ptr<int>(),
        optional_data_ptr(grad_points),
        optional_data_ptr(grad_normals),
        optional_data_ptr(grad_path_length),
        optional_stride_or_zero(grad_points, 0),
        optional_stride_or_zero(grad_points, 1),
        optional_stride_or_zero(grad_points, 2),
        optional_stride_or_zero(grad_normals, 0),
        optional_stride_or_zero(grad_normals, 1),
        optional_stride_or_zero(grad_normals, 2),
        optional_stride_or_zero(grad_path_length, 0),
        ray_count,
        max_bounces,
        faces.size(0),
        need_grad_vertices ? out.grad_vertices.data_ptr<float>() : nullptr,
        need_grad_source ? out.grad_source.data_ptr<float>() : nullptr,
        need_grad_receiver ? out.grad_receiver.data_ptr<float>() : nullptr);
    return out;
}

ReflEpcPathsJvpOutputs reflection_epc_paths_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver) {
    const int64_t ray_count = source.size(0);
    const int64_t max_bounces = sequence.size(1);
    ReflEpcPathsJvpOutputs out;
    out.tangent_points =
        at::empty({ray_count, max_bounces, 3}, source.options());
    out.tangent_normals =
        at::empty({ray_count, max_bounces, 3}, source.options());
    out.tangent_path_length = at::empty({ray_count}, source.options());
    if (ray_count == 0) {
        return out;
    }

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_epc_paths_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        sequence.data_ptr<int>(),
        plane_points.data_ptr<float>(),
        plane_normals.data_ptr<float>(),
        valid.data_ptr<bool>(),
        bounce_count.data_ptr<int>(),
        optional_data_ptr(tangent_vertices),
        optional_data_ptr(tangent_source),
        optional_data_ptr(tangent_receiver),
        optional_stride_or_zero(tangent_vertices, 0),
        optional_stride_or_zero(tangent_vertices, 1),
        optional_stride_or_zero(tangent_source, 0),
        optional_stride_or_zero(tangent_source, 1),
        optional_stride_or_zero(tangent_receiver, 0),
        optional_stride_or_zero(tangent_receiver, 1),
        ray_count,
        static_cast<int>(max_bounces),
        faces.size(0),
        out.tangent_points.data_ptr<float>(),
        out.tangent_normals.data_ptr<float>(),
        out.tangent_path_length.data_ptr<float>());
    return out;
}

at::Tensor scene_face_normals_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &grad_face_normals) {
    const int64_t triangle_count = faces.size(0);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    at::Tensor grad_vertices = at::empty(vertices.sizes(), vertices.options());
    zero_float_tensor_async(grad_vertices, stream);
    if (triangle_count == 0) {
        return grad_vertices;
    }

    const int threads = 128;
    const int blocks = static_cast<int>((triangle_count + threads - 1) / threads);
    scene_face_normals_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        grad_face_normals.data_ptr<float>(),
        grad_face_normals.stride(0),
        grad_face_normals.stride(1),
        triangle_count,
        grad_vertices.data_ptr<float>());
    return grad_vertices;
}

at::Tensor scene_face_normals_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &tangent_vertices) {
    const int64_t triangle_count = faces.size(0);
    at::Tensor tangent_face_normals =
        at::empty({triangle_count, 3}, vertices.options());
    if (triangle_count == 0) {
        return tangent_face_normals;
    }

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((triangle_count + threads - 1) / threads);
    scene_face_normals_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        tangent_vertices.data_ptr<float>(),
        tangent_vertices.stride(0),
        tangent_vertices.stride(1),
        triangle_count,
        tangent_face_normals.data_ptr<float>());
    return tangent_face_normals;
}

} // namespace rayd::torch_backend


// ---- merged from src/reflection/accum_reduce_part.cu ----

#include <src/reflection/accum_reduce.h>
#include <src/runtime/optix_context.h>
#include <src/reflection/accum_params.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

namespace {

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_i32_count(int64_t count, const char *name) {
    if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + ": count is outside int32 launch range.");
    }
}

struct AddReflAccumValue {
    __host__ __device__ ReflAccumStagedValue operator()(
        ReflAccumStagedValue x,
        ReflAccumStagedValue y) const {
        ReflAccumStagedValue out;
        out.a = make_float4(
            x.a.x + y.a.x,
            x.a.y + y.a.y,
            x.a.z + y.a.z,
            x.a.w + y.a.w);
        out.b = make_float4(
            x.b.x + y.b.x,
            x.b.y + y.b.y,
            x.b.z + y.b.z,
            x.b.w + y.b.w);
        return out;
    }
};

__global__ void scatter_refl_accum_reduced_kernel(
    const int *__restrict__ num_runs,
    const int *__restrict__ unique_cells,
    const ReflAccumStagedValue *__restrict__ reduced_values,
    float *__restrict__ out_power,
    float *__restrict__ out_field_x_re,
    float *__restrict__ out_field_x_im,
    float *__restrict__ out_field_y_re,
    float *__restrict__ out_field_y_im,
    float *__restrict__ out_field_z_re,
    float *__restrict__ out_field_z_im,
    int *__restrict__ out_reflection_count) {
    const int n = *num_runs;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        const int cell = unique_cells[idx];
        if (cell < 0) {
            continue;
        }
        const ReflAccumStagedValue value = reduced_values[idx];
        if (out_power != nullptr) {
            atomicAdd(out_power + cell, value.a.x);
        }
        if (out_field_x_re != nullptr) {
            atomicAdd(out_field_x_re + cell, value.a.y);
        }
        if (out_field_x_im != nullptr) {
            atomicAdd(out_field_x_im + cell, value.a.z);
        }
        if (out_field_y_re != nullptr) {
            atomicAdd(out_field_y_re + cell, value.a.w);
        }
        if (out_field_y_im != nullptr) {
            atomicAdd(out_field_y_im + cell, value.b.x);
        }
        if (out_field_z_re != nullptr) {
            atomicAdd(out_field_z_re + cell, value.b.y);
        }
        if (out_field_z_im != nullptr) {
            atomicAdd(out_field_z_im + cell, value.b.z);
        }
        const int count = static_cast<int>(value.b.w + 0.5f);
        if (out_reflection_count != nullptr && count != 0) {
            atomicAdd(out_reflection_count, count);
        }
    }
}

} // namespace

void reduce_refl_accum_staged_cuda(
    int64_t sample_count,
    const at::Tensor &stage_cell,
    const at::Tensor &stage_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_reflection_count) {
    require_i32_count(sample_count, "reduce_refl_accum_staged_cuda(sample_count)");
    if (sample_count == 0) {
        return;
    }

    const int sample_count_i = static_cast<int>(sample_count);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cudaStream_t stream = torch_ctx.stream;
    at::TensorOptions key_options = stage_cell.options();
    at::TensorOptions value_options = stage_value.options().dtype(at::kFloat);
    at::TensorOptions byte_options = stage_cell.options().dtype(at::kByte);

    at::Tensor sorted_cells = at::empty({sample_count}, key_options);
    at::Tensor sorted_values = at::empty({sample_count, 8}, value_options);
    auto *values_in = reinterpret_cast<ReflAccumStagedValue *>(stage_value.data_ptr<float>());
    auto *values_sorted =
        reinterpret_cast<ReflAccumStagedValue *>(sorted_values.data_ptr<float>());

    size_t sort_temp_bytes = 0;
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_temp_bytes,
            stage_cell.data_ptr<int>(),
            sorted_cells.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(refl accum size)");
    at::Tensor sort_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(sort_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr<uint8_t>(),
            sort_temp_bytes,
            stage_cell.data_ptr<int>(),
            sorted_cells.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(refl accum)");

    at::Tensor unique_cells = at::empty({sample_count}, key_options);
    at::Tensor reduced_values = at::empty({sample_count, 8}, value_options);
    at::Tensor num_runs = at::empty({1}, key_options);
    auto *reduced_values_ptr =
        reinterpret_cast<ReflAccumStagedValue *>(reduced_values.data_ptr<float>());

    size_t reduce_temp_bytes = 0;
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            nullptr,
            reduce_temp_bytes,
            sorted_cells.data_ptr<int>(),
            unique_cells.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddReflAccumValue{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(refl accum size)");
    at::Tensor reduce_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(reduce_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            reduce_temp.data_ptr<uint8_t>(),
            reduce_temp_bytes,
            sorted_cells.data_ptr<int>(),
            unique_cells.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddReflAccumValue{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(refl accum)");

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((sample_count + block_size - 1) / block_size);
    scatter_refl_accum_reduced_kernel<<<block_count, block_size, 0, stream>>>(
        num_runs.data_ptr<int>(),
        unique_cells.data_ptr<int>(),
        reduced_values_ptr,
        out_power.data_ptr<float>(),
        out_field_x_re.data_ptr<float>(),
        out_field_x_im.data_ptr<float>(),
        out_field_y_re.data_ptr<float>(),
        out_field_y_im.data_ptr<float>(),
        out_field_z_re.data_ptr<float>(),
        out_field_z_im.data_ptr<float>(),
        out_reflection_count.data_ptr<int>());
    cuda_check(cudaGetLastError(), "scatter_refl_accum_reduced_kernel");
}

} // namespace rayd::torch_backend
