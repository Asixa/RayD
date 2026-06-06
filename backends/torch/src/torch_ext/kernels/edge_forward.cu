#include <raydtorch/edge_kernels.h>
#include <raydtorch/edge_optix_params.h>
#include <raydtorch/optix_context.h>

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

__device__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__global__ void edge_recompute_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    const float *__restrict__ point,
    int64_t point_count,
    const int *__restrict__ candidate_edge_id,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    float *__restrict__ tape_s,
    float *__restrict__ tape_d) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;

    const int best_edge = candidate_edge_id[point_idx];
    if (best_edge < 0) {
        distance[point_idx] = CUDART_INF_F;
        edge_t[point_idx] = 0.f;
        tape_s[point_idx] = 0.f;
        tape_edge_id[point_idx] = -1;
        shape_id[point_idx] = -1;
        edge_id[point_idx] = -1;
        global_edge_id[point_idx] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            edge_point[point_idx * 3 + axis] = 0.f;
            tape_d[point_idx * 3 + axis] = 0.f;
        }
        return;
    }

    const float3 p = make_f3(point + point_idx * 3);
    const int i0 = edge_v0[best_edge];
    const int i1 = edge_v1[best_edge];
    const float3 a = make_f3(vertices + i0 * 3);
    const float3 b = make_f3(vertices + i1 * 3);
    const float3 ab = sub3(b, a);
    const float denom = fmaxf(dot3(ab, ab), 1e-20f);
    float best_s = dot3(sub3(p, a), ab) / denom;
    best_s = fminf(1.f, fmaxf(0.f, best_s));
    const float3 best_edge_point = add3(a, mul3(best_s, ab));
    const float3 best_d = sub3(p, best_edge_point);
    const float best_dist2 = dot3(best_d, best_d);

    distance[point_idx] = sqrtf(best_dist2);
    edge_t[point_idx] = best_s;
    tape_s[point_idx] = best_s;
    tape_edge_id[point_idx] = best_edge;
    shape_id[point_idx] = best_edge >= 0 ? edge_shape_id[best_edge] : -1;
    edge_id[point_idx] = best_edge >= 0 ? edge_local_id[best_edge] : -1;
    global_edge_id[point_idx] = best_edge;
    edge_point[point_idx * 3 + 0] = best_edge_point.x;
    edge_point[point_idx * 3 + 1] = best_edge_point.y;
    edge_point[point_idx * 3 + 2] = best_edge_point.z;
    tape_d[point_idx * 3 + 0] = best_d.x;
    tape_d[point_idx * 3 + 1] = best_d.y;
    tape_d[point_idx * 3 + 2] = best_d.z;
}

} // namespace

EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point) {
    const MeshRecord &mesh = scene.meshes[0];
    const int64_t point_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = point.options();
    auto iopts = scene.edge_v0.options();

    EdgeForwardOutputs out;
    out.distance = at::empty({point_count}, fopts);
    out.edge_point = at::empty({point_count, 3}, fopts);
    out.edge_t = at::empty({point_count}, fopts);
    out.shape_id = at::empty({point_count}, iopts);
    out.edge_id = at::empty({point_count}, iopts);
    out.global_edge_id = at::empty({point_count}, iopts);
    out.tape_edge_id = out.global_edge_id;
    out.tape_s = out.edge_t;
    out.tape_d = at::empty({point_count, 3}, fopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    EdgeOptixQueryParams params = {};
    params.traversable = scene.edge_accel.traversable;
    params.vertices = mesh.vertices.data_ptr<float>();
    params.edge_v0 = scene.edge_v0.data_ptr<int>();
    params.edge_v1 = scene.edge_v1.data_ptr<int>();
    params.point = point.data_ptr<float>();
    params.out_edge_id = out.global_edge_id.data_ptr<int>();
    params.edge_count = static_cast<int32_t>(edge_count);
    params.point_count = static_cast<int32_t>(point_count);
    params.search_radius = scene.edge_accel.search_radius;

    at::TensorOptions byte_options =
        at::TensorOptions().device(point.device()).dtype(at::kByte);
    at::Tensor params_buffer =
        at::empty({static_cast<int64_t>(sizeof(EdgeOptixQueryParams))}, byte_options);
    cuda_check(
        cudaMemcpyAsync(
            params_buffer.data_ptr<uint8_t>(),
            &params,
            sizeof(EdgeOptixQueryParams),
            cudaMemcpyHostToDevice,
            torch_ctx.stream),
        "cudaMemcpyAsync(edge OptiX params)");
    raydtorch_OPTIX_CHECK(optixLaunch(
        optix_entry.edge_pipeline,
        torch_ctx.stream,
        reinterpret_cast<CUdeviceptr>(params_buffer.data_ptr<uint8_t>()),
        sizeof(EdgeOptixQueryParams),
        &optix_entry.edge_sbt,
        static_cast<unsigned int>(point_count),
        1,
        1));

    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    edge_recompute_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        mesh.vertices.data_ptr<float>(),
        scene.edge_v0.data_ptr<int>(),
        scene.edge_v1.data_ptr<int>(),
        scene.edge_shape_id.data_ptr<int>(),
        scene.edge_local_id.data_ptr<int>(),
        point.data_ptr<float>(),
        point_count,
        out.global_edge_id.data_ptr<int>(),
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        out.tape_s.data_ptr<float>(),
        out.tape_d.data_ptr<float>());
    return out;
}

} // namespace raydtorch
