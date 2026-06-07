#include <raydtorch/edge/kernels.h>
#include <raydtorch/edge/optix_params.h>
#include <raydtorch/common/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optix_stubs.h>

#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

__device__ float3 make_aos_f3(const float *ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ float3 edge_start(
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    int edge) {
    return make_float3(p0_x[edge], p0_y[edge], p0_z[edge]);
}

__device__ float3 edge_vector(
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    int edge) {
    return make_float3(e1_x[edge], e1_y[edge], e1_z[edge]);
}

__device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__global__ void init_edge_point_outputs_kernel(
    int64_t point_count,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    float *__restrict__ tape_s,
    float *__restrict__ tape_d,
    bool *__restrict__ unresolved) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    distance[point_idx] = CUDART_INF_F;
    edge_t[point_idx] = 0.f;
    shape_id[point_idx] = -1;
    edge_id[point_idx] = -1;
    global_edge_id[point_idx] = -1;
    tape_edge_id[point_idx] = -1;
    tape_s[point_idx] = 0.f;
    unresolved[point_idx] = true;
    for (int axis = 0; axis < 3; ++axis) {
        edge_point[point_idx * 3 + axis] = 0.f;
        tape_d[point_idx * 3 + axis] = 0.f;
    }
}

__global__ void finalize_edge_point_stage_kernel(
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    const float *__restrict__ point,
    const int *__restrict__ stage_edge_id,
    const float *__restrict__ stage_distance_sq,
    const float *__restrict__ stage_edge_t,
    const bool *__restrict__ stage_valid,
    int64_t point_count,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    float *__restrict__ tape_s,
    float *__restrict__ tape_d,
    bool *__restrict__ unresolved) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count || !unresolved[point_idx])
        return;
    const int edge = stage_edge_id[point_idx];
    if (!stage_valid[point_idx] || edge < 0)
        return;

    const float s = fminf(1.f, fmaxf(0.f, stage_edge_t[point_idx]));
    const float3 p = make_aos_f3(point + point_idx * 3);
    const float3 a = edge_start(p0_x, p0_y, p0_z, edge);
    const float3 e = edge_vector(e1_x, e1_y, e1_z, edge);
    const float3 q = add3(a, mul3(s, e));
    const float3 d = sub3(p, q);

    distance[point_idx] = sqrtf(fmaxf(stage_distance_sq[point_idx], 0.f));
    edge_t[point_idx] = s;
    shape_id[point_idx] = edge_shape_id[edge];
    edge_id[point_idx] = edge_local_id[edge];
    global_edge_id[point_idx] = edge;
    tape_edge_id[point_idx] = edge;
    tape_s[point_idx] = s;
    edge_point[point_idx * 3 + 0] = q.x;
    edge_point[point_idx * 3 + 1] = q.y;
    edge_point[point_idx * 3 + 2] = q.z;
    tape_d[point_idx * 3 + 0] = d.x;
    tape_d[point_idx * 3 + 1] = d.y;
    tape_d[point_idx * 3 + 2] = d.z;
    unresolved[point_idx] = false;
}

__global__ void init_edge_ray_outputs_kernel(
    const bool *__restrict__ active,
    int64_t ray_count,
    float *__restrict__ distance,
    float *__restrict__ ray_t,
    float *__restrict__ point,
    float *__restrict__ edge_t,
    float *__restrict__ edge_point,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    bool *__restrict__ unresolved) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    distance[ray_idx] = CUDART_INF_F;
    ray_t[ray_idx] = 0.f;
    edge_t[ray_idx] = 0.f;
    shape_id[ray_idx] = -1;
    edge_id[ray_idx] = -1;
    global_edge_id[ray_idx] = -1;
    tape_edge_id[ray_idx] = -1;
    unresolved[ray_idx] = active[ray_idx];
    for (int axis = 0; axis < 3; ++axis) {
        point[ray_idx * 3 + axis] = 0.f;
        edge_point[ray_idx * 3 + axis] = 0.f;
    }
}

__global__ void finalize_edge_ray_stage_kernel(
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const int *__restrict__ stage_edge_id,
    const float *__restrict__ stage_distance_sq,
    const float *__restrict__ stage_ray_t,
    const float *__restrict__ stage_edge_t,
    const bool *__restrict__ stage_valid,
    int64_t ray_count,
    float *__restrict__ distance,
    float *__restrict__ ray_t,
    float *__restrict__ point,
    float *__restrict__ edge_t,
    float *__restrict__ edge_point,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    bool *__restrict__ unresolved) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count || !unresolved[ray_idx])
        return;
    const int edge = stage_edge_id[ray_idx];
    if (!stage_valid[ray_idx] || edge < 0)
        return;

    const float rt = stage_ray_t[ray_idx];
    const float s = fminf(1.f, fmaxf(0.f, stage_edge_t[ray_idx]));
    const float3 ro = make_aos_f3(ray_o + ray_idx * 3);
    const float3 rd = make_aos_f3(ray_d + ray_idx * 3);
    const float3 rp = add3(ro, mul3(rt, rd));
    const float3 a = edge_start(p0_x, p0_y, p0_z, edge);
    const float3 e = edge_vector(e1_x, e1_y, e1_z, edge);
    const float3 ep = add3(a, mul3(s, e));

    distance[ray_idx] = sqrtf(fmaxf(stage_distance_sq[ray_idx], 0.f));
    ray_t[ray_idx] = rt;
    edge_t[ray_idx] = s;
    shape_id[ray_idx] = edge_shape_id[edge];
    edge_id[ray_idx] = edge_local_id[edge];
    global_edge_id[ray_idx] = edge;
    tape_edge_id[ray_idx] = edge;
    point[ray_idx * 3 + 0] = rp.x;
    point[ray_idx * 3 + 1] = rp.y;
    point[ray_idx * 3 + 2] = rp.z;
    edge_point[ray_idx * 3 + 0] = ep.x;
    edge_point[ray_idx * 3 + 1] = ep.y;
    edge_point[ray_idx * 3 + 2] = ep.z;
    unresolved[ray_idx] = false;
}

void launch_edge_query(
    const OptixDeviceContextEntry &optix_entry,
    cudaStream_t stream,
    const EdgeOptixQueryParams &params,
    EdgeOptixLaunchKind kind,
    int64_t query_count,
    const at::Device &device) {
    at::TensorOptions byte_options = at::TensorOptions().device(device).dtype(at::kByte);
    at::Tensor params_buffer =
        at::empty({static_cast<int64_t>(sizeof(EdgeOptixQueryParams))}, byte_options);
    cuda_check(
        cudaMemcpyAsync(
            params_buffer.data_ptr<uint8_t>(),
            &params,
            sizeof(EdgeOptixQueryParams),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(edge OptiX params)");
    raydtorch_OPTIX_CHECK(optixLaunch(
        optix_entry.edge_pipeline,
        stream,
        reinterpret_cast<CUdeviceptr>(params_buffer.data_ptr<uint8_t>()),
        sizeof(EdgeOptixQueryParams),
        &edge_sbt(optix_entry, kind),
        static_cast<unsigned int>(query_count),
        1,
        1));
}

} // namespace

EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point) {
    const int64_t point_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = point.options();
    auto iopts = scene.edge_v0.options();
    auto bopts = at::TensorOptions().device(point.device()).dtype(at::kBool);

    EdgeForwardOutputs out;
    out.distance = at::empty({point_count}, fopts);
    out.edge_point = at::empty({point_count, 3}, fopts);
    out.edge_t = at::empty({point_count}, fopts);
    out.shape_id = at::empty({point_count}, iopts);
    out.edge_id = at::empty({point_count}, iopts);
    out.global_edge_id = at::empty({point_count}, iopts);
    out.tape_edge_id = at::empty({point_count}, iopts);
    out.tape_s = at::empty({point_count}, fopts);
    out.tape_d = at::empty({point_count, 3}, fopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    at::Tensor unresolved = at::empty({point_count}, bopts);
    init_edge_point_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point_count,
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        out.tape_s.data_ptr<float>(),
        out.tape_d.data_ptr<float>(),
        unresolved.data_ptr<bool>());

    if (edge_count <= 0 || point_count <= 0 || scene.edge_accels.empty())
        return out;

    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    at::Tensor query_x = point.select(1, 0).contiguous();
    at::Tensor query_y = point.select(1, 1).contiguous();
    at::Tensor query_z = point.select(1, 2).contiguous();

    for (const OptixEdgeAccel &accel : scene.edge_accels) {
        at::Tensor stage_edge_id = at::empty({point_count}, iopts);
        at::Tensor stage_distance_sq = at::empty({point_count}, fopts);
        at::Tensor stage_edge_t = at::empty({point_count}, fopts);
        at::Tensor stage_valid = at::empty({point_count}, bopts);

        EdgeOptixQueryParams params = {};
        params.handle = static_cast<uint64_t>(accel.traversable);
        params.edge_p0_x = scene.edge_p0_x.data_ptr<float>();
        params.edge_p0_y = scene.edge_p0_y.data_ptr<float>();
        params.edge_p0_z = scene.edge_p0_z.data_ptr<float>();
        params.edge_e1_x = scene.edge_e1_x.data_ptr<float>();
        params.edge_e1_y = scene.edge_e1_y.data_ptr<float>();
        params.edge_e1_z = scene.edge_e1_z.data_ptr<float>();
        params.edge_mask = scene.edge_mask.data_ptr<uint8_t>();
        params.edge_count = static_cast<int>(edge_count);
        params.search_radius = accel.search_radius;
        params.query_x = query_x.data_ptr<float>();
        params.query_y = query_y.data_ptr<float>();
        params.query_z = query_z.data_ptr<float>();
        params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
        params.query_count = static_cast<int>(point_count);
        params.out_edge_ids = stage_edge_id.data_ptr<int>();
        params.out_distance_sq = stage_distance_sq.data_ptr<float>();
        params.out_edge_t = stage_edge_t.data_ptr<float>();
        params.out_valid = reinterpret_cast<uint8_t *>(stage_valid.data_ptr<bool>());

        launch_edge_query(
            optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Point, point_count, point.device());
        finalize_edge_point_stage_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
            scene.edge_p0_x.data_ptr<float>(),
            scene.edge_p0_y.data_ptr<float>(),
            scene.edge_p0_z.data_ptr<float>(),
            scene.edge_e1_x.data_ptr<float>(),
            scene.edge_e1_y.data_ptr<float>(),
            scene.edge_e1_z.data_ptr<float>(),
            scene.edge_shape_id.data_ptr<int>(),
            scene.edge_local_id.data_ptr<int>(),
            point.data_ptr<float>(),
            stage_edge_id.data_ptr<int>(),
            stage_distance_sq.data_ptr<float>(),
            stage_edge_t.data_ptr<float>(),
            stage_valid.data_ptr<bool>(),
            point_count,
            out.distance.data_ptr<float>(),
            out.edge_point.data_ptr<float>(),
            out.edge_t.data_ptr<float>(),
            out.shape_id.data_ptr<int>(),
            out.edge_id.data_ptr<int>(),
            out.global_edge_id.data_ptr<int>(),
            out.tape_edge_id.data_ptr<int>(),
            out.tape_s.data_ptr<float>(),
            out.tape_d.data_ptr<float>(),
            unresolved.data_ptr<bool>());
    }
    return out;
}

EdgeRayForwardOutputs edge_ray_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    const int64_t ray_count = ray_o.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = ray_o.options();
    auto iopts = scene.edge_v0.options();
    auto bopts = at::TensorOptions().device(ray_o.device()).dtype(at::kBool);

    EdgeRayForwardOutputs out;
    out.distance = at::empty({ray_count}, fopts);
    out.ray_t = at::empty({ray_count}, fopts);
    out.point = at::empty({ray_count, 3}, fopts);
    out.edge_t = at::empty({ray_count}, fopts);
    out.edge_point = at::empty({ray_count, 3}, fopts);
    out.shape_id = at::empty({ray_count}, iopts);
    out.edge_id = at::empty({ray_count}, iopts);
    out.global_edge_id = at::empty({ray_count}, iopts);
    out.tape_edge_id = at::empty({ray_count}, iopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    at::Tensor unresolved = at::empty({ray_count}, bopts);
    at::Tensor active_c = active.contiguous();
    init_edge_ray_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        active_c.data_ptr<bool>(),
        ray_count,
        out.distance.data_ptr<float>(),
        out.ray_t.data_ptr<float>(),
        out.point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        unresolved.data_ptr<bool>());

    if (edge_count <= 0 || ray_count <= 0 || scene.edge_accels.empty())
        return out;

    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    at::Tensor query_x = ray_o.select(1, 0).contiguous();
    at::Tensor query_y = ray_o.select(1, 1).contiguous();
    at::Tensor query_z = ray_o.select(1, 2).contiguous();
    at::Tensor ray_dx = ray_d.select(1, 0).contiguous();
    at::Tensor ray_dy = ray_d.select(1, 1).contiguous();
    at::Tensor ray_dz = ray_d.select(1, 2).contiguous();

    for (const OptixEdgeAccel &accel : scene.edge_accels) {
        at::Tensor stage_edge_id = at::empty({ray_count}, iopts);
        at::Tensor stage_distance_sq = at::empty({ray_count}, fopts);
        at::Tensor stage_ray_t = at::empty({ray_count}, fopts);
        at::Tensor stage_edge_t = at::empty({ray_count}, fopts);
        at::Tensor stage_valid = at::empty({ray_count}, bopts);

        EdgeOptixQueryParams params = {};
        params.handle = static_cast<uint64_t>(accel.traversable);
        params.edge_p0_x = scene.edge_p0_x.data_ptr<float>();
        params.edge_p0_y = scene.edge_p0_y.data_ptr<float>();
        params.edge_p0_z = scene.edge_p0_z.data_ptr<float>();
        params.edge_e1_x = scene.edge_e1_x.data_ptr<float>();
        params.edge_e1_y = scene.edge_e1_y.data_ptr<float>();
        params.edge_e1_z = scene.edge_e1_z.data_ptr<float>();
        params.edge_mask = scene.edge_mask.data_ptr<uint8_t>();
        params.edge_count = static_cast<int>(edge_count);
        params.search_radius = accel.search_radius;
        params.query_x = query_x.data_ptr<float>();
        params.query_y = query_y.data_ptr<float>();
        params.query_z = query_z.data_ptr<float>();
        params.ray_dx = ray_dx.data_ptr<float>();
        params.ray_dy = ray_dy.data_ptr<float>();
        params.ray_dz = ray_dz.data_ptr<float>();
        params.ray_tmax = ray_tmax.data_ptr<float>();
        params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
        params.query_count = static_cast<int>(ray_count);
        params.out_edge_ids = stage_edge_id.data_ptr<int>();
        params.out_distance_sq = stage_distance_sq.data_ptr<float>();
        params.out_ray_t = stage_ray_t.data_ptr<float>();
        params.out_edge_t = stage_edge_t.data_ptr<float>();
        params.out_valid = reinterpret_cast<uint8_t *>(stage_valid.data_ptr<bool>());

        launch_edge_query(
            optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Ray, ray_count, ray_o.device());
        finalize_edge_ray_stage_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
            scene.edge_p0_x.data_ptr<float>(),
            scene.edge_p0_y.data_ptr<float>(),
            scene.edge_p0_z.data_ptr<float>(),
            scene.edge_e1_x.data_ptr<float>(),
            scene.edge_e1_y.data_ptr<float>(),
            scene.edge_e1_z.data_ptr<float>(),
            scene.edge_shape_id.data_ptr<int>(),
            scene.edge_local_id.data_ptr<int>(),
            ray_o.data_ptr<float>(),
            ray_d.data_ptr<float>(),
            stage_edge_id.data_ptr<int>(),
            stage_distance_sq.data_ptr<float>(),
            stage_ray_t.data_ptr<float>(),
            stage_edge_t.data_ptr<float>(),
            stage_valid.data_ptr<bool>(),
            ray_count,
            out.distance.data_ptr<float>(),
            out.ray_t.data_ptr<float>(),
            out.point.data_ptr<float>(),
            out.edge_t.data_ptr<float>(),
            out.edge_point.data_ptr<float>(),
            out.shape_id.data_ptr<int>(),
            out.edge_id.data_ptr<int>(),
            out.global_edge_id.data_ptr<int>(),
            out.tape_edge_id.data_ptr<int>(),
            unresolved.data_ptr<bool>());
    }
    return out;
}

} // namespace raydtorch
