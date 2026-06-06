#include <raydtorch/geometry_kernels.h>
#include <raydtorch/optix_context.h>
#include <raydtorch/optix_intersect_params.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optix_stubs.h>

#include <stdexcept>
#include <string>

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

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__global__ void intersect_recompute_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    int64_t ray_count,
    const float *__restrict__ optix_t,
    const int *__restrict__ optix_prim_id,
    const float *__restrict__ optix_bary_uv,
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

    const bool lane_active = active[ray_idx];
    const float3 o = make_f3(ray_o + ray_idx * 3);
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const int prim_id = lane_active ? optix_prim_id[ray_idx] : -1;
    const float hit_t = optix_t[ray_idx];
    const float u = optix_bary_uv[ray_idx * 2 + 0];
    const float v = optix_bary_uv[ray_idx * 2 + 1];
    const bool hit = prim_id >= 0;

    out_shape_id[ray_idx] = hit ? 0 : -1;
    out_prim_id[ray_idx] = hit ? prim_id : -1;
    out_local_prim_id[ray_idx] = hit ? prim_id : -1;
    out_global_prim_id[ray_idx] = hit ? prim_id : -1;
    out_bary[ray_idx * 3 + 0] = hit ? 1.f - u - v : 0.f;
    out_bary[ray_idx * 3 + 1] = hit ? u : 0.f;
    out_bary[ray_idx * 3 + 2] = hit ? v : 0.f;
    out_uv[ray_idx * 2 + 0] = hit ? u : 0.f;
    out_uv[ray_idx * 2 + 1] = hit ? v : 0.f;

    const float safe_t = hit ? hit_t : 0.f;
    const float3 p = add(o, mul(safe_t, d));
    out_p[ray_idx * 3 + 0] = hit ? p.x : 0.f;
    out_p[ray_idx * 3 + 1] = hit ? p.y : 0.f;
    out_p[ray_idx * 3 + 2] = hit ? p.z : 0.f;

    float3 normal = make_float3(0.f, 0.f, 0.f);
    if (hit) {
        const int i0 = faces[prim_id * 3 + 0];
        const int i1 = faces[prim_id * 3 + 1];
        const int i2 = faces[prim_id * 3 + 2];
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
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    const MeshRecord &mesh = scene.meshes[0];
    const OptixTriangleAccel &accel = scene.triangle_accels[0];
    const at::Tensor &vertices = mesh.vertices;
    const at::Tensor &faces = mesh.faces;
    const int64_t ray_count = ray_o.size(0);
    auto fopts = vertices.options();
    auto iopts = faces.options();

    IntersectForwardOutputs out;
    out.t = at::empty({ray_count}, fopts);
    at::Tensor optix_bary_uv = at::empty({ray_count, 2}, fopts);
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

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_intersect_pipeline(optix_entry);

    OptixIntersectParams params = {};
    params.traversable = accel.traversable;
    params.ray_o = ray_o.data_ptr<float>();
    params.ray_d = ray_d.data_ptr<float>();
    params.ray_tmax = ray_tmax.data_ptr<float>();
    params.active = active.data_ptr<bool>();
    params.out_t = out.t.data_ptr<float>();
    params.out_prim_id = out.global_prim_id.data_ptr<int>();
    params.out_bary_uv = optix_bary_uv.data_ptr<float>();
    params.ray_count = static_cast<int32_t>(ray_count);

    at::TensorOptions byte_options =
        at::TensorOptions().device(vertices.device()).dtype(at::kByte);
    at::Tensor params_buffer =
        at::empty({static_cast<int64_t>(sizeof(OptixIntersectParams))}, byte_options);
    cuda_check(
        cudaMemcpyAsync(
            params_buffer.data_ptr<uint8_t>(),
            &params,
            sizeof(OptixIntersectParams),
            cudaMemcpyHostToDevice,
            torch_ctx.stream),
        "cudaMemcpyAsync(OptiX intersect params)");

    raydtorch_OPTIX_CHECK(optixLaunch(
        optix_entry.intersect_pipeline,
        torch_ctx.stream,
        reinterpret_cast<CUdeviceptr>(params_buffer.data_ptr<uint8_t>()),
        sizeof(OptixIntersectParams),
        &optix_entry.intersect_sbt,
        static_cast<unsigned int>(ray_count),
        1,
        1));

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    intersect_recompute_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        active.data_ptr<bool>(),
        ray_count,
        out.t.data_ptr<float>(),
        out.global_prim_id.data_ptr<int>(),
        optix_bary_uv.data_ptr<float>(),
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
