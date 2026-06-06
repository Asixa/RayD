#include <raydtorch/geometry_kernels.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace raydtorch {

namespace {

__device__ float3 make_f3(const float *ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__global__ void intersect_forward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    int64_t face_count,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const float *__restrict__ ray_tmax,
    const bool *__restrict__ active,
    int64_t ray_count,
    float *__restrict__ out_t,
    float *__restrict__ out_p,
    float *__restrict__ out_n,
    float *__restrict__ out_geo_n,
    float *__restrict__ out_uv,
    float *__restrict__ out_bary,
    int *__restrict__ out_shape_id,
    int *__restrict__ out_prim_id,
    int *__restrict__ out_local_prim_id,
    int *__restrict__ out_global_prim_id) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    float best_t = CUDART_INF_F;
    float best_u = 0.f;
    float best_v = 0.f;
    int best_face = -1;
    const bool lane_active = active[ray_idx];
    const float3 o = make_f3(ray_o + ray_idx * 3);
    const float3 d = make_f3(ray_d + ray_idx * 3);

    if (lane_active) {
        for (int face_idx = 0; face_idx < face_count; ++face_idx) {
            const int i0 = faces[face_idx * 3 + 0];
            const int i1 = faces[face_idx * 3 + 1];
            const int i2 = faces[face_idx * 3 + 2];
            const float3 p0 = make_f3(vertices + i0 * 3);
            const float3 p1 = make_f3(vertices + i1 * 3);
            const float3 p2 = make_f3(vertices + i2 * 3);
            const float3 e1 = sub(p1, p0);
            const float3 e2 = sub(p2, p0);
            const float3 pvec = cross3(d, e2);
            const float det = dot3(e1, pvec);
            if (fabsf(det) < 1e-8f)
                continue;
            const float inv_det = 1.f / det;
            const float3 tvec = sub(o, p0);
            const float u = dot3(tvec, pvec) * inv_det;
            if (u < 0.f || u > 1.f)
                continue;
            const float3 qvec = cross3(tvec, e1);
            const float v = dot3(d, qvec) * inv_det;
            if (v < 0.f || u + v > 1.f)
                continue;
            const float t = dot3(e2, qvec) * inv_det;
            if (t > 1e-6f && t < best_t && t < ray_tmax[ray_idx]) {
                best_t = t;
                best_u = u;
                best_v = v;
                best_face = face_idx;
            }
        }
    }

    out_t[ray_idx] = best_t;
    out_shape_id[ray_idx] = best_face >= 0 ? 0 : -1;
    out_prim_id[ray_idx] = best_face;
    out_local_prim_id[ray_idx] = best_face;
    out_global_prim_id[ray_idx] = best_face;
    out_bary[ray_idx * 3 + 0] = best_face >= 0 ? 1.f - best_u - best_v : 0.f;
    out_bary[ray_idx * 3 + 1] = best_face >= 0 ? best_u : 0.f;
    out_bary[ray_idx * 3 + 2] = best_face >= 0 ? best_v : 0.f;
    out_uv[ray_idx * 2 + 0] = best_face >= 0 ? best_u : 0.f;
    out_uv[ray_idx * 2 + 1] = best_face >= 0 ? best_v : 0.f;

    const float safe_t = best_face >= 0 ? best_t : 0.f;
    const float3 p = add(o, mul(safe_t, d));
    out_p[ray_idx * 3 + 0] = best_face >= 0 ? p.x : 0.f;
    out_p[ray_idx * 3 + 1] = best_face >= 0 ? p.y : 0.f;
    out_p[ray_idx * 3 + 2] = best_face >= 0 ? p.z : 0.f;

    float3 normal = make_float3(0.f, 0.f, 0.f);
    if (best_face >= 0) {
        const int i0 = faces[best_face * 3 + 0];
        const int i1 = faces[best_face * 3 + 1];
        const int i2 = faces[best_face * 3 + 2];
        const float3 p0 = make_f3(vertices + i0 * 3);
        const float3 p1 = make_f3(vertices + i1 * 3);
        const float3 p2 = make_f3(vertices + i2 * 3);
        normal = cross3(sub(p1, p0), sub(p2, p0));
        const float inv_len = rsqrtf(fmaxf(dot3(normal, normal), 1e-20f));
        normal = mul(inv_len, normal);
    }
    out_n[ray_idx * 3 + 0] = normal.x;
    out_n[ray_idx * 3 + 1] = normal.y;
    out_n[ray_idx * 3 + 2] = normal.z;
    out_geo_n[ray_idx * 3 + 0] = normal.x;
    out_geo_n[ray_idx * 3 + 1] = normal.y;
    out_geo_n[ray_idx * 3 + 2] = normal.z;
}

} // namespace

IntersectForwardOutputs intersect_forward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    const int64_t ray_count = ray_o.size(0);
    const int64_t face_count = faces.size(0);
    auto fopts = vertices.options();
    auto iopts = faces.options();

    IntersectForwardOutputs out;
    out.t = at::empty({ray_count}, fopts);
    out.p = at::empty({ray_count, 3}, fopts);
    out.n = at::empty({ray_count, 3}, fopts);
    out.geo_n = at::empty({ray_count, 3}, fopts);
    out.uv = at::empty({ray_count, 2}, fopts);
    out.barycentric = at::empty({ray_count, 3}, fopts);
    out.shape_id = at::empty({ray_count}, iopts);
    out.prim_id = at::empty({ray_count}, iopts);
    out.local_prim_id = at::empty({ray_count}, iopts);
    out.global_prim_id = at::empty({ray_count}, iopts);
    out.tape_prim_id = out.global_prim_id;
    out.tape_barycentric = out.barycentric;
    out.tape_t = out.t;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    intersect_forward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        face_count,
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        ray_tmax.data_ptr<float>(),
        active.data_ptr<bool>(),
        ray_count,
        out.t.data_ptr<float>(),
        out.p.data_ptr<float>(),
        out.n.data_ptr<float>(),
        out.geo_n.data_ptr<float>(),
        out.uv.data_ptr<float>(),
        out.barycentric.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.prim_id.data_ptr<int>(),
        out.local_prim_id.data_ptr<int>(),
        out.global_prim_id.data_ptr<int>());

    return out;
}

} // namespace raydtorch
