#include <rayd/torch/edge/kernels.h>
#include <rayd/torch/edge/optix_params.h>
#include <rayd/torch/runtime/optix_context.h>
#include <rayd/shared/edge/edge_distance_math.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optix_stubs.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

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

__device__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_edge_cache_dtype(const at::Tensor &tensor, at::ScalarType dtype, const char *name) {
    if (tensor.scalar_type() != dtype)
        throw std::runtime_error(
            std::string("edge cache tensor ") + name + " has dtype " +
            std::string(c10::toString(tensor.scalar_type())) + ", expected " +
            std::string(c10::toString(dtype)) + ".");
}

void require_edge_cache_dtypes(const SceneCache &scene) {
    require_edge_cache_dtype(scene.edge_v0, at::kInt, "edge_v0");
    require_edge_cache_dtype(scene.edge_v1, at::kInt, "edge_v1");
    require_edge_cache_dtype(scene.edge_shape_id, at::kInt, "edge_shape_id");
    require_edge_cache_dtype(scene.edge_local_id, at::kInt, "edge_local_id");
    require_edge_cache_dtype(scene.edge_p0_x, at::kFloat, "edge_p0_x");
    require_edge_cache_dtype(scene.edge_p0_y, at::kFloat, "edge_p0_y");
    require_edge_cache_dtype(scene.edge_p0_z, at::kFloat, "edge_p0_z");
    require_edge_cache_dtype(scene.edge_e1_x, at::kFloat, "edge_e1_x");
    require_edge_cache_dtype(scene.edge_e1_y, at::kFloat, "edge_e1_y");
    require_edge_cache_dtype(scene.edge_e1_z, at::kFloat, "edge_e1_z");
    require_edge_cache_dtype(scene.edge_mask, at::kByte, "edge_mask");
}

void fill_edge_tiers(const SceneCache &scene, EdgeOptixQueryParams &params) {
    if (scene.edge_accels.size() > static_cast<std::size_t>(EdgeOptixMaxTiers)) {
        throw std::runtime_error(
            "edge query has more GAS tiers than EdgeOptixMaxTiers; increase the OptiX params "
            "tier array size.");
    }

    params.tier_count = static_cast<int>(scene.edge_accels.size());
    if (params.tier_count <= 0) {
        return;
    }

    params.handle = static_cast<uint64_t>(scene.edge_accels.front().traversable);
    params.search_radius = scene.edge_accels.back().search_radius;
    for (int tier = 0; tier < params.tier_count; ++tier) {
        const OptixEdgeAccel &accel = scene.edge_accels[static_cast<std::size_t>(tier)];
        params.tier_handles[tier] = static_cast<uint64_t>(accel.traversable);
        params.tier_search_radii[tier] = accel.search_radius;
    }
}

void fill_edge_geometry_params(
    const SceneCache &scene,
    int64_t edge_count,
    EdgeOptixQueryParams &params) {
    fill_edge_tiers(scene, params);
    params.edge_p0_x = scene.edge_p0_x.data_ptr<float>();
    params.edge_p0_y = scene.edge_p0_y.data_ptr<float>();
    params.edge_p0_z = scene.edge_p0_z.data_ptr<float>();
    params.edge_e1_x = scene.edge_e1_x.data_ptr<float>();
    params.edge_e1_y = scene.edge_e1_y.data_ptr<float>();
    params.edge_e1_z = scene.edge_e1_z.data_ptr<float>();
    params.edge_mask = scene.edge_mask.data_ptr<uint8_t>();
    params.edge_count = static_cast<int>(edge_count);
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

__global__ void init_edge_point_public_outputs_kernel(
    int64_t point_count,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    bool *__restrict__ unresolved) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    distance[point_idx] = CUDART_INF_F;
    edge_t[point_idx] = 0.f;
    shape_id[point_idx] = -1;
    edge_id[point_idx] = -1;
    global_edge_id[point_idx] = -1;
    unresolved[point_idx] = true;
    for (int axis = 0; axis < 3; ++axis) {
        edge_point[point_idx * 3 + axis] = 0.f;
    }
}

__global__ void split_vec3_to_soa_kernel(
    const float *__restrict__ value,
    int64_t count,
    int64_t stride0,
    int64_t stride1,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float *row = value + static_cast<int64_t>(idx) * stride0;
    x[idx] = row[0 * stride1];
    y[idx] = row[1 * stride1];
    z[idx] = row[2 * stride1];
}

__global__ void split_ray_vec3_to_soa_kernel(
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    int64_t count,
    int64_t ray_o_stride0,
    int64_t ray_o_stride1,
    int64_t ray_d_stride0,
    int64_t ray_d_stride1,
    float *__restrict__ query_x,
    float *__restrict__ query_y,
    float *__restrict__ query_z,
    float *__restrict__ ray_dx,
    float *__restrict__ ray_dy,
    float *__restrict__ ray_dz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float *origin = ray_o + static_cast<int64_t>(idx) * ray_o_stride0;
    const float *direction = ray_d + static_cast<int64_t>(idx) * ray_d_stride0;
    query_x[idx] = origin[0 * ray_o_stride1];
    query_y[idx] = origin[1 * ray_o_stride1];
    query_z[idx] = origin[2 * ray_o_stride1];
    ray_dx[idx] = direction[0 * ray_d_stride1];
    ray_dy[idx] = direction[1 * ray_d_stride1];
    ray_dz[idx] = direction[2 * ray_d_stride1];
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
    unresolved[ray_idx] = active == nullptr || active[ray_idx];
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

constexpr int kEdgeBruteTile = 128;

// Exact nearest-edge fallback for queries the tight OptiX radius tiers did not
// resolve. The previous full-radius GAS tier inflated every edge AABB by the
// scene diagonal, so a far query visited all edge AABBs through the OptiX
// intersection program; a tiled scan over the edge SoA does the same exact
// search at a fraction of the cost. The full-radius cutoff is preserved:
// queries farther than max_radius from every edge stay unresolved.
__global__ void edge_point_bruteforce_kernel(
    const float *__restrict__ query_x,
    const float *__restrict__ query_y,
    const float *__restrict__ query_z,
    const bool *__restrict__ unresolved_mask,
    int point_count,
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    const uint8_t *__restrict__ edge_mask,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    int edge_count,
    float max_radius,
    float *__restrict__ final_distance,
    float *__restrict__ final_edge_point,
    float *__restrict__ final_edge_t,
    int *__restrict__ final_shape_id,
    int *__restrict__ final_edge_id,
    int *__restrict__ final_global_edge_id,
    int *__restrict__ final_tape_edge_id,
    float *__restrict__ final_tape_s,
    float *__restrict__ final_tape_d,
    bool *__restrict__ unresolved_out) {
    __shared__ float s_p0x[kEdgeBruteTile];
    __shared__ float s_p0y[kEdgeBruteTile];
    __shared__ float s_p0z[kEdgeBruteTile];
    __shared__ float s_e1x[kEdgeBruteTile];
    __shared__ float s_e1y[kEdgeBruteTile];
    __shared__ float s_e1z[kEdgeBruteTile];
    __shared__ uint8_t s_mask[kEdgeBruteTile];

    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = query < point_count && unresolved_mask[query];
    if (__syncthreads_count(active ? 1 : 0) == 0)
        return;

    float qx = 0.f;
    float qy = 0.f;
    float qz = 0.f;
    if (active) {
        qx = query_x[query];
        qy = query_y[query];
        qz = query_z[query];
    }
    float best_d2 = CUDART_INF_F;
    float best_t = 0.f;
    int best_edge = -1;

    for (int tile = 0; tile < edge_count; tile += kEdgeBruteTile) {
        const int tile_n = min(kEdgeBruteTile, edge_count - tile);
        for (int j = threadIdx.x; j < tile_n; j += blockDim.x) {
            s_p0x[j] = p0_x[tile + j];
            s_p0y[j] = p0_y[tile + j];
            s_p0z[j] = p0_z[tile + j];
            s_e1x[j] = e1_x[tile + j];
            s_e1y[j] = e1_y[tile + j];
            s_e1z[j] = e1_z[tile + j];
            s_mask[j] = edge_mask == nullptr ? 1u : edge_mask[tile + j];
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < tile_n; ++j) {
                if (s_mask[j] == 0u)
                    continue;
                const shared::edge::PointSegmentDistance candidate =
                    shared::edge::point_segment_distance(
                        shared::math::make_vec3(qx, qy, qz),
                        shared::math::make_vec3(s_p0x[j], s_p0y[j], s_p0z[j]),
                        shared::math::make_vec3(s_e1x[j], s_e1y[j], s_e1z[j]));
                const float t = candidate.edge_parameter;
                const float d2 = candidate.squared_distance;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_t = t;
                    best_edge = tile + j;
                }
            }
        }
        __syncthreads();
    }

    if (!active || best_edge < 0)
        return;
    const float distance = sqrtf(fmaxf(best_d2, 0.f));
    if (distance > max_radius)
        return;

    const float3 a = edge_start(p0_x, p0_y, p0_z, best_edge);
    const float3 e = edge_vector(e1_x, e1_y, e1_z, best_edge);
    const float3 q = add3(a, mul3(best_t, e));
    final_distance[query] = distance;
    final_edge_t[query] = best_t;
    final_shape_id[query] = edge_shape_id[best_edge];
    final_edge_id[query] = edge_local_id[best_edge];
    final_global_edge_id[query] = best_edge;
    final_edge_point[query * 3 + 0] = q.x;
    final_edge_point[query * 3 + 1] = q.y;
    final_edge_point[query * 3 + 2] = q.z;
    if (final_tape_edge_id != nullptr) {
        final_tape_edge_id[query] = best_edge;
    }
    if (final_tape_s != nullptr) {
        final_tape_s[query] = best_t;
    }
    if (final_tape_d != nullptr) {
        final_tape_d[query * 3 + 0] = qx - q.x;
        final_tape_d[query * 3 + 1] = qy - q.y;
        final_tape_d[query * 3 + 2] = qz - q.z;
    }
    unresolved_out[query] = false;
}

void launch_edge_query(
    OptixDeviceContextEntry &optix_entry,
    cudaStream_t stream,
    const EdgeOptixQueryParams &params,
    EdgeOptixLaunchKind kind,
    int64_t query_count) {
    at::Tensor params_buffer = at::empty(
        {static_cast<int64_t>(sizeof(EdgeOptixQueryParams))},
        at::TensorOptions()
            .device(at::Device(at::kCUDA, optix_entry.device_index))
            .dtype(at::kByte));
    cuda_check(
        cudaMemcpyAsync(
            params_buffer.data_ptr<uint8_t>(),
            &params,
            sizeof(EdgeOptixQueryParams),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(edge OptiX params)");
    rayd_torch_OPTIX_CHECK(optixLaunch(
        edge_pipeline(optix_entry, kind),
        stream,
        reinterpret_cast<CUdeviceptr>(params_buffer.data_ptr<uint8_t>()),
        sizeof(EdgeOptixQueryParams),
        &edge_sbt(optix_entry, kind),
        static_cast<unsigned int>(query_count),
        1,
        1));
}

} // namespace

// Device contract: the public edge ops in edge/ops_edge.cpp make the scene
// device current with a c10::cuda::CUDAGuard before dispatching here, so the
// host launchers below resolve their stream on the scene device through
// current_torch_cuda_context() and add no guard of their own.
EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point) {
    require_edge_cache_dtypes(scene);
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

    at::Tensor query_x = at::empty({point_count}, fopts);
    at::Tensor query_y = at::empty({point_count}, fopts);
    at::Tensor query_z = at::empty({point_count}, fopts);
    split_vec3_to_soa_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point.data_ptr<float>(),
        point_count,
        point.stride(0),
        point.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>());

    EdgeOptixQueryParams params = {};
    fill_edge_geometry_params(scene, edge_count, params);
    params.query_x = query_x.data_ptr<float>();
    params.query_y = query_y.data_ptr<float>();
    params.query_z = query_z.data_ptr<float>();
    params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
    params.query_count = static_cast<int>(point_count);
    params.edge_shape_id = scene.edge_shape_id.data_ptr<int>();
    params.edge_local_id = scene.edge_local_id.data_ptr<int>();
    params.write_point_outputs = 1;
    params.final_distance = out.distance.data_ptr<float>();
    params.final_edge_point = out.edge_point.data_ptr<float>();
    params.final_edge_t = out.edge_t.data_ptr<float>();
    params.final_shape_id = out.shape_id.data_ptr<int>();
    params.final_edge_id = out.edge_id.data_ptr<int>();
    params.final_global_edge_id = out.global_edge_id.data_ptr<int>();
    params.final_tape_edge_id = out.tape_edge_id.data_ptr<int>();
    params.final_tape_s = out.tape_s.data_ptr<float>();
    params.final_tape_d = out.tape_d.data_ptr<float>();
    params.final_unresolved = reinterpret_cast<uint8_t *>(unresolved.data_ptr<bool>());

    // The widest (scene-diagonal) tier degenerates to an O(edges) scan through
    // the OptiX intersection program for far queries, and even the middle tier
    // inflates every edge AABB enough that mid-distance queries visit a large
    // fraction of the scene. Keep only the tightest tier on the OptiX path and
    // resolve the remainder with the tiled fallback scan.
    const float full_search_radius = params.search_radius;
    params.tier_count = params.tier_count > 1 ? 1 : 0;
    if (params.tier_count > 0) {
        launch_edge_query(
            optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Point, point_count);
    }
    const int brute_blocks =
        static_cast<int>((point_count + kEdgeBruteTile - 1) / kEdgeBruteTile);
    edge_point_bruteforce_kernel<<<brute_blocks, kEdgeBruteTile, 0, torch_ctx.stream>>>(
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        unresolved.data_ptr<bool>(),
        static_cast<int>(point_count),
        scene.edge_p0_x.data_ptr<float>(),
        scene.edge_p0_y.data_ptr<float>(),
        scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),
        scene.edge_e1_y.data_ptr<float>(),
        scene.edge_e1_z.data_ptr<float>(),
        scene.edge_mask.data_ptr<uint8_t>(),
        scene.edge_shape_id.data_ptr<int>(),
        scene.edge_local_id.data_ptr<int>(),
        static_cast<int>(edge_count),
        full_search_radius,
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
    return out;
}

EdgeForwardPublicOutputs edge_forward_noad_cuda(const SceneCache &scene, const at::Tensor &point) {
    require_edge_cache_dtypes(scene);
    const int64_t point_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = point.options();
    auto iopts = scene.edge_v0.options();
    auto bopts = at::TensorOptions().device(point.device()).dtype(at::kBool);

    EdgeForwardPublicOutputs out;
    out.distance = at::empty({point_count}, fopts);
    out.edge_point = at::empty({point_count, 3}, fopts);
    out.edge_t = at::empty({point_count}, fopts);
    out.shape_id = at::empty({point_count}, iopts);
    out.edge_id = at::empty({point_count}, iopts);
    out.global_edge_id = at::empty({point_count}, iopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    at::Tensor unresolved = at::empty({point_count}, bopts);
    init_edge_point_public_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point_count,
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        unresolved.data_ptr<bool>());

    if (edge_count <= 0 || point_count <= 0 || scene.edge_accels.empty())
        return out;

    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    at::Tensor query_x = at::empty({point_count}, fopts);
    at::Tensor query_y = at::empty({point_count}, fopts);
    at::Tensor query_z = at::empty({point_count}, fopts);
    split_vec3_to_soa_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point.data_ptr<float>(),
        point_count,
        point.stride(0),
        point.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>());

    EdgeOptixQueryParams params = {};
    fill_edge_geometry_params(scene, edge_count, params);
    params.query_x = query_x.data_ptr<float>();
    params.query_y = query_y.data_ptr<float>();
    params.query_z = query_z.data_ptr<float>();
    params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
    params.query_count = static_cast<int>(point_count);
    params.edge_shape_id = scene.edge_shape_id.data_ptr<int>();
    params.edge_local_id = scene.edge_local_id.data_ptr<int>();
    params.write_point_outputs = 1;
    params.final_distance = out.distance.data_ptr<float>();
    params.final_edge_point = out.edge_point.data_ptr<float>();
    params.final_edge_t = out.edge_t.data_ptr<float>();
    params.final_shape_id = out.shape_id.data_ptr<int>();
    params.final_edge_id = out.edge_id.data_ptr<int>();
    params.final_global_edge_id = out.global_edge_id.data_ptr<int>();
    params.final_unresolved = reinterpret_cast<uint8_t *>(unresolved.data_ptr<bool>());

    // Same tightest-tier + fallback split as the tape-producing path above.
    const float full_search_radius = params.search_radius;
    params.tier_count = params.tier_count > 1 ? 1 : 0;
    if (params.tier_count > 0) {
        launch_edge_query(
            optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Point, point_count);
    }
    const int brute_blocks =
        static_cast<int>((point_count + kEdgeBruteTile - 1) / kEdgeBruteTile);
    edge_point_bruteforce_kernel<<<brute_blocks, kEdgeBruteTile, 0, torch_ctx.stream>>>(
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        unresolved.data_ptr<bool>(),
        static_cast<int>(point_count),
        scene.edge_p0_x.data_ptr<float>(),
        scene.edge_p0_y.data_ptr<float>(),
        scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),
        scene.edge_e1_y.data_ptr<float>(),
        scene.edge_e1_z.data_ptr<float>(),
        scene.edge_mask.data_ptr<uint8_t>(),
        scene.edge_shape_id.data_ptr<int>(),
        scene.edge_local_id.data_ptr<int>(),
        static_cast<int>(edge_count),
        full_search_radius,
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        nullptr,
        nullptr,
        nullptr,
        unresolved.data_ptr<bool>());
    return out;
}

EdgeRayForwardOutputs edge_ray_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    require_edge_cache_dtypes(scene);
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
    at::Tensor active_c = active.numel() == 0 ? active : active.contiguous();
    init_edge_ray_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        active_c.numel() == 0 ? nullptr : active_c.data_ptr<bool>(),
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

    at::Tensor query_x = at::empty({ray_count}, fopts);
    at::Tensor query_y = at::empty({ray_count}, fopts);
    at::Tensor query_z = at::empty({ray_count}, fopts);
    at::Tensor ray_dx = at::empty({ray_count}, fopts);
    at::Tensor ray_dy = at::empty({ray_count}, fopts);
    at::Tensor ray_dz = at::empty({ray_count}, fopts);
    split_ray_vec3_to_soa_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        ray_count,
        ray_o.stride(0),
        ray_o.stride(1),
        ray_d.stride(0),
        ray_d.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        ray_dx.data_ptr<float>(),
        ray_dy.data_ptr<float>(),
        ray_dz.data_ptr<float>());

    at::Tensor stage_edge_id = at::empty({ray_count}, iopts);
    at::Tensor stage_distance_sq = at::empty({ray_count}, fopts);
    at::Tensor stage_ray_t = at::empty({ray_count}, fopts);
    at::Tensor stage_edge_t = at::empty({ray_count}, fopts);
    at::Tensor stage_valid = at::empty({ray_count}, bopts);

    EdgeOptixQueryParams params = {};
    fill_edge_geometry_params(scene, edge_count, params);
    params.query_x = query_x.data_ptr<float>();
    params.query_y = query_y.data_ptr<float>();
    params.query_z = query_z.data_ptr<float>();
    params.ray_dx = ray_dx.data_ptr<float>();
    params.ray_dy = ray_dy.data_ptr<float>();
    params.ray_dz = ray_dz.data_ptr<float>();
    params.ray_tmax = ray_tmax.numel() == 0 ? nullptr : ray_tmax.data_ptr<float>();
    params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
    params.query_count = static_cast<int>(ray_count);
    params.out_edge_ids = stage_edge_id.data_ptr<int>();
    params.out_distance_sq = stage_distance_sq.data_ptr<float>();
    params.out_ray_t = stage_ray_t.data_ptr<float>();
    params.out_edge_t = stage_edge_t.data_ptr<float>();
    params.out_valid = reinterpret_cast<uint8_t *>(stage_valid.data_ptr<bool>());

    launch_edge_query(optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Ray, ray_count);
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
    return out;
}

} // namespace rayd::torch_backend


// ---- merged from src/edge/edge_backward_part.cu ----

#include <rayd/torch/edge/kernels.h>
#include <rayd/torch/math.cuh>
#include <rayd/shared/edge/edge_distance_math.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace rayd::torch_backend {

namespace {

__device__ shared::math::Vec3f to_shared_vec3(float3 value) {
    return shared::math::make_vec3(value.x, value.y, value.z);
}

__device__ float3 from_shared_vec3(shared::math::Vec3f value) {
    return make_float3(value.x, value.y, value.z);
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

__global__ void edge_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const float *__restrict__ point,
    const int *__restrict__ tape_edge_id,
    const float *__restrict__ tape_s,
    const float *__restrict__ tape_d,
    const float *__restrict__ grad_distance,
    const float *__restrict__ grad_edge_point,
    const float *__restrict__ grad_edge_t,
    const float *__restrict__ grad_edge_t_alias,
    int64_t grad_distance_stride0,
    int64_t grad_edge_point_stride0,
    int64_t grad_edge_point_stride1,
    int64_t grad_edge_t_stride0,
    int64_t grad_edge_t_alias_stride0,
    int64_t point_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_point) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    for (int axis = 0; axis < 3; ++axis)
        grad_point[point_idx * 3 + axis] = 0.f;
    const int edge_id = tape_edge_id[point_idx];
    if (edge_id < 0)
        return;

    const int i0 = edge_v0[edge_id];
    const int i1 = edge_v1[edge_id];
    const float3 p = make_f3(point + point_idx * 3);
    const float3 a = make_f3(vertices + i0 * 3);
    const float3 b = make_f3(vertices + i1 * 3);
    const float s = tape_s[point_idx];
    const float3 d = make_f3(tape_d + point_idx * 3);
    const float distance_bar = read_scalar_or_zero(grad_distance, point_idx, grad_distance_stride0);
    const float3 gep = read_vec3_or_zero(grad_edge_point, point_idx, grad_edge_point_stride0, grad_edge_point_stride1);
    const float edge_parameter_bar =
        read_scalar_or_zero(grad_edge_t, point_idx, grad_edge_t_stride0) +
        read_scalar_or_zero(grad_edge_t_alias, point_idx, grad_edge_t_alias_stride0);
    const shared::edge::PointSegmentVjp vjp =
        shared::edge::point_segment_vjp_fixed_winner(
            to_shared_vec3(p),
            to_shared_vec3(a),
            to_shared_vec3(b),
            s,
            to_shared_vec3(d),
            distance_bar,
            to_shared_vec3(gep),
            edge_parameter_bar);
    const float3 point_bar = from_shared_vec3(vjp.point);
    grad_point[point_idx * 3 + 0] = point_bar.x;
    grad_point[point_idx * 3 + 1] = point_bar.y;
    grad_point[point_idx * 3 + 2] = point_bar.z;
    atomic_add3(grad_vertices, i0, from_shared_vec3(vjp.edge_start));
    atomic_add3(grad_vertices, i1, from_shared_vec3(vjp.edge_end));
}

__global__ void edge_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const float *__restrict__ point,
    const int *__restrict__ tape_edge_id,
    const float *__restrict__ tape_s,
    const float *__restrict__ tape_d,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_point,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_point_stride0,
    int64_t tangent_point_stride1,
    int64_t point_count,
    float *__restrict__ tangent_distance,
    float *__restrict__ tangent_edge_point,
    float *__restrict__ tangent_edge_t,
    float *__restrict__ tangent_tape_s,
    float *__restrict__ tangent_tape_d) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    tangent_distance[point_idx] = 0.f;
    tangent_edge_t[point_idx] = 0.f;
    tangent_tape_s[point_idx] = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        tangent_edge_point[point_idx * 3 + axis] = 0.f;
        tangent_tape_d[point_idx * 3 + axis] = 0.f;
    }
    const int edge_id = tape_edge_id[point_idx];
    if (edge_id < 0)
        return;

    const int i0 = edge_v0[edge_id];
    const int i1 = edge_v1[edge_id];
    const float3 a = make_f3(vertices + i0 * 3);
    const float3 b = make_f3(vertices + i1 * 3);
    const float3 p = make_f3(point + point_idx * 3);
    const float3 da = read_vec3_or_zero(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 db = read_vec3_or_zero(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 dp = read_vec3_or_zero(tangent_point, point_idx, tangent_point_stride0, tangent_point_stride1);
    const float s = tape_s[point_idx];
    const float3 d = make_f3(tape_d + point_idx * 3);
    const shared::edge::PointSegmentJvp jvp =
        shared::edge::point_segment_jvp_fixed_winner(
            to_shared_vec3(p),
            to_shared_vec3(a),
            to_shared_vec3(b),
            s,
            to_shared_vec3(d),
            to_shared_vec3(dp),
            to_shared_vec3(da),
            to_shared_vec3(db));
    const float3 dep = from_shared_vec3(jvp.edge_point);

    tangent_distance[point_idx] = jvp.distance;
    tangent_edge_t[point_idx] = jvp.edge_parameter;
    tangent_edge_point[point_idx * 3 + 0] = dep.x;
    tangent_edge_point[point_idx * 3 + 1] = dep.y;
    tangent_edge_point[point_idx * 3 + 2] = dep.z;
}

__global__ void edge_ray_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const float *__restrict__ ray_tmax,
    const int *__restrict__ tape_edge_id,
    const float *__restrict__ ray_t,
    const float *__restrict__ edge_t,
    const float *__restrict__ grad_distance,
    const float *__restrict__ grad_ray_t,
    const float *__restrict__ grad_point,
    const float *__restrict__ grad_edge_t,
    const float *__restrict__ grad_edge_point,
    int64_t grad_distance_stride0,
    int64_t grad_ray_t_stride0,
    int64_t grad_point_stride0,
    int64_t grad_point_stride1,
    int64_t grad_edge_t_stride0,
    int64_t grad_edge_point_stride0,
    int64_t grad_edge_point_stride1,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    for (int axis = 0; axis < 3; ++axis) {
        grad_ray_o[ray_idx * 3 + axis] = 0.f;
        grad_ray_d[ray_idx * 3 + axis] = 0.f;
    }
    const int edge_id = tape_edge_id[ray_idx];
    if (edge_id < 0)
        return;

    const int i0 = edge_v0[edge_id];
    const int i1 = edge_v1[edge_id];
    const float3 ro = make_f3(ray_o + ray_idx * 3);
    const float3 rd = make_f3(ray_d + ray_idx * 3);
    const float3 a = make_f3(vertices + i0 * 3);
    const float3 b = make_f3(vertices + i1 * 3);
    const bool has_max = ray_tmax != nullptr &&
                         isfinite(ray_tmax[ray_idx]) &&
                         ray_tmax[ray_idx] > 0.f;
    const shared::edge::RaySegmentVjp vjp =
        shared::edge::ray_segment_vjp_fixed_winner(
            to_shared_vec3(ro),
            to_shared_vec3(rd),
            to_shared_vec3(a),
            to_shared_vec3(b),
            ray_t[ray_idx],
            edge_t[ray_idx],
            has_max,
            has_max ? ray_tmax[ray_idx] : shared::edge::EdgeDistanceFloatMax,
            read_scalar_or_zero(grad_distance, ray_idx, grad_distance_stride0),
            read_scalar_or_zero(grad_ray_t, ray_idx, grad_ray_t_stride0),
            to_shared_vec3(read_vec3_or_zero(
                grad_point, ray_idx, grad_point_stride0, grad_point_stride1)),
            read_scalar_or_zero(grad_edge_t, ray_idx, grad_edge_t_stride0),
            to_shared_vec3(read_vec3_or_zero(
                grad_edge_point,
                ray_idx,
                grad_edge_point_stride0,
                grad_edge_point_stride1)));
    const float3 ray_origin_bar = from_shared_vec3(vjp.ray_origin);
    const float3 ray_direction_bar = from_shared_vec3(vjp.ray_direction);
    grad_ray_o[ray_idx * 3 + 0] = ray_origin_bar.x;
    grad_ray_o[ray_idx * 3 + 1] = ray_origin_bar.y;
    grad_ray_o[ray_idx * 3 + 2] = ray_origin_bar.z;
    grad_ray_d[ray_idx * 3 + 0] = ray_direction_bar.x;
    grad_ray_d[ray_idx * 3 + 1] = ray_direction_bar.y;
    grad_ray_d[ray_idx * 3 + 2] = ray_direction_bar.z;
    atomic_add3(grad_vertices, i0, from_shared_vec3(vjp.edge_start));
    atomic_add3(grad_vertices, i1, from_shared_vec3(vjp.edge_end));
}

__global__ void edge_ray_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const float *__restrict__ ray_tmax,
    const int *__restrict__ tape_edge_id,
    const float *__restrict__ ray_t,
    const float *__restrict__ edge_t,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_ray_o,
    const float *__restrict__ tangent_ray_d,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_ray_o_stride0,
    int64_t tangent_ray_o_stride1,
    int64_t tangent_ray_d_stride0,
    int64_t tangent_ray_d_stride1,
    int64_t ray_count,
    float *__restrict__ tangent_distance,
    float *__restrict__ tangent_ray_t,
    float *__restrict__ tangent_point,
    float *__restrict__ tangent_edge_t,
    float *__restrict__ tangent_edge_point) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    tangent_distance[ray_idx] = 0.f;
    tangent_ray_t[ray_idx] = 0.f;
    tangent_edge_t[ray_idx] = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        tangent_point[ray_idx * 3 + axis] = 0.f;
        tangent_edge_point[ray_idx * 3 + axis] = 0.f;
    }
    const int edge_id = tape_edge_id[ray_idx];
    if (edge_id < 0)
        return;

    const int i0 = edge_v0[edge_id];
    const int i1 = edge_v1[edge_id];
    const bool has_max = ray_tmax != nullptr &&
                         isfinite(ray_tmax[ray_idx]) &&
                         ray_tmax[ray_idx] > 0.f;
    const shared::edge::RaySegmentJvp jvp =
        shared::edge::ray_segment_jvp_fixed_winner(
            to_shared_vec3(make_f3(ray_o + ray_idx * 3)),
            to_shared_vec3(make_f3(ray_d + ray_idx * 3)),
            to_shared_vec3(make_f3(vertices + i0 * 3)),
            to_shared_vec3(make_f3(vertices + i1 * 3)),
            ray_t[ray_idx],
            edge_t[ray_idx],
            has_max,
            has_max ? ray_tmax[ray_idx] : shared::edge::EdgeDistanceFloatMax,
            to_shared_vec3(read_vec3_or_zero(
                tangent_ray_o,
                ray_idx,
                tangent_ray_o_stride0,
                tangent_ray_o_stride1)),
            to_shared_vec3(read_vec3_or_zero(
                tangent_ray_d,
                ray_idx,
                tangent_ray_d_stride0,
                tangent_ray_d_stride1)),
            to_shared_vec3(read_vec3_or_zero(
                tangent_vertices,
                i0,
                tangent_vertices_stride0,
                tangent_vertices_stride1)),
            to_shared_vec3(read_vec3_or_zero(
                tangent_vertices,
                i1,
                tangent_vertices_stride0,
                tangent_vertices_stride1)));
    const float3 point_tangent = from_shared_vec3(jvp.ray_point);
    const float3 edge_point_tangent = from_shared_vec3(jvp.edge_point);
    tangent_distance[ray_idx] = jvp.distance;
    tangent_ray_t[ray_idx] = jvp.ray_parameter;
    tangent_edge_t[ray_idx] = jvp.edge_parameter;
    tangent_point[ray_idx * 3 + 0] = point_tangent.x;
    tangent_point[ray_idx * 3 + 1] = point_tangent.y;
    tangent_point[ray_idx * 3 + 2] = point_tangent.z;
    tangent_edge_point[ray_idx * 3 + 0] = edge_point_tangent.x;
    tangent_edge_point[ray_idx * 3 + 1] = edge_point_tangent.y;
    tangent_edge_point[ray_idx * 3 + 2] = edge_point_tangent.z;
}

} // namespace

// Device contract: the public edge ops in edge/ops_edge.cpp make the scene
// device current with a c10::cuda::CUDAGuard before dispatching here, so every
// host launcher below resolves its stream on the scene device and adds no
// guard of its own.
EdgeBackwardOutputs edge_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &grad_distance,
    const at::Tensor &grad_edge_point,
    const at::Tensor &grad_edge_t) {
    return edge_backward_optional_cuda(
        vertices,
        edge_v0,
        edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        &grad_distance,
        &grad_edge_point,
        &grad_edge_t,
        nullptr);
}

EdgeBackwardOutputs edge_backward_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor *grad_distance,
    const at::Tensor *grad_edge_point,
    const at::Tensor *grad_edge_t,
    const at::Tensor *grad_edge_t_alias) {
    const int64_t point_count = point.size(0);
    EdgeBackwardOutputs out;
    out.grad_vertices = at::empty_like(vertices);
    out.grad_point = at::empty_like(point);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device()).stream();
    cudaMemsetAsync(out.grad_vertices.data_ptr<float>(), 0, static_cast<size_t>(out.grad_vertices.nbytes()), stream);
    if (point_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    edge_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        point.data_ptr<float>(),
        tape_edge_id.data_ptr<int>(),
        tape_s.data_ptr<float>(),
        tape_d.data_ptr<float>(),
        grad_distance == nullptr ? nullptr : grad_distance->data_ptr<float>(),
        grad_edge_point == nullptr ? nullptr : grad_edge_point->data_ptr<float>(),
        grad_edge_t == nullptr ? nullptr : grad_edge_t->data_ptr<float>(),
        grad_edge_t_alias == nullptr ? nullptr : grad_edge_t_alias->data_ptr<float>(),
        optional_stride(grad_distance, 0),
        optional_stride(grad_edge_point, 0),
        optional_stride(grad_edge_point, 1),
        optional_stride(grad_edge_t, 0),
        optional_stride(grad_edge_t_alias, 0),
        point_count,
        out.grad_vertices.data_ptr<float>(),
        out.grad_point.data_ptr<float>());
    return out;
}

EdgeJvpOutputs edge_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_point) {
    return edge_jvp_optional_cuda(
        vertices,
        edge_v0,
        edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        &tangent_vertices,
        &tangent_point);
}

EdgeJvpOutputs edge_jvp_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_point) {
    const int64_t point_count = point.size(0);
    EdgeJvpOutputs out;
    out.tangent_distance = at::empty({point_count}, point.options());
    out.tangent_edge_point = at::empty_like(point);
    out.tangent_edge_t = at::empty({point_count}, point.options());
    out.tangent_tape_s = at::empty_like(tape_s);
    out.tangent_tape_d = at::empty_like(tape_d);
    if (point_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device()).stream();
    edge_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        point.data_ptr<float>(),
        tape_edge_id.data_ptr<int>(),
        tape_s.data_ptr<float>(),
        tape_d.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        tangent_point == nullptr ? nullptr : tangent_point->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        optional_stride(tangent_point, 0),
        optional_stride(tangent_point, 1),
        point_count,
        out.tangent_distance.data_ptr<float>(),
        out.tangent_edge_point.data_ptr<float>(),
        out.tangent_edge_t.data_ptr<float>(),
        out.tangent_tape_s.data_ptr<float>(),
        out.tangent_tape_d.data_ptr<float>());
    return out;
}

EdgeRayBackwardOutputs edge_ray_backward_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &tape_edge_id,
    const at::Tensor &ray_t,
    const at::Tensor &edge_t,
    const at::Tensor *grad_distance,
    const at::Tensor *grad_ray_t,
    const at::Tensor *grad_point,
    const at::Tensor *grad_edge_t,
    const at::Tensor *grad_edge_point) {
    const int64_t ray_count = ray_o.size(0);
    EdgeRayBackwardOutputs out;
    out.grad_vertices = at::empty_like(vertices);
    out.grad_ray_o = at::empty_like(ray_o);
    out.grad_ray_d = at::empty_like(ray_d);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    cudaMemsetAsync(
        out.grad_vertices.data_ptr<float>(),
        0,
        static_cast<size_t>(out.grad_vertices.nbytes()),
        stream);
    if (ray_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    edge_ray_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        ray_tmax.numel() == 0 ? nullptr : ray_tmax.data_ptr<float>(),
        tape_edge_id.data_ptr<int>(),
        ray_t.data_ptr<float>(),
        edge_t.data_ptr<float>(),
        grad_distance == nullptr ? nullptr : grad_distance->data_ptr<float>(),
        grad_ray_t == nullptr ? nullptr : grad_ray_t->data_ptr<float>(),
        grad_point == nullptr ? nullptr : grad_point->data_ptr<float>(),
        grad_edge_t == nullptr ? nullptr : grad_edge_t->data_ptr<float>(),
        grad_edge_point == nullptr ? nullptr : grad_edge_point->data_ptr<float>(),
        optional_stride(grad_distance, 0),
        optional_stride(grad_ray_t, 0),
        optional_stride(grad_point, 0),
        optional_stride(grad_point, 1),
        optional_stride(grad_edge_t, 0),
        optional_stride(grad_edge_point, 0),
        optional_stride(grad_edge_point, 1),
        ray_count,
        out.grad_vertices.data_ptr<float>(),
        out.grad_ray_o.data_ptr<float>(),
        out.grad_ray_d.data_ptr<float>());
    return out;
}

EdgeRayJvpOutputs edge_ray_jvp_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &tape_edge_id,
    const at::Tensor &ray_t,
    const at::Tensor &edge_t,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d) {
    const int64_t ray_count = ray_o.size(0);
    EdgeRayJvpOutputs out;
    out.tangent_distance = at::empty({ray_count}, ray_o.options());
    out.tangent_ray_t = at::empty({ray_count}, ray_o.options());
    out.tangent_point = at::empty_like(ray_o);
    out.tangent_edge_t = at::empty({ray_count}, ray_o.options());
    out.tangent_edge_point = at::empty_like(ray_o);
    if (ray_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    edge_ray_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        ray_tmax.numel() == 0 ? nullptr : ray_tmax.data_ptr<float>(),
        tape_edge_id.data_ptr<int>(),
        ray_t.data_ptr<float>(),
        edge_t.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        tangent_ray_o == nullptr ? nullptr : tangent_ray_o->data_ptr<float>(),
        tangent_ray_d == nullptr ? nullptr : tangent_ray_d->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        optional_stride(tangent_ray_o, 0),
        optional_stride(tangent_ray_o, 1),
        optional_stride(tangent_ray_d, 0),
        optional_stride(tangent_ray_d, 1),
        ray_count,
        out.tangent_distance.data_ptr<float>(),
        out.tangent_ray_t.data_ptr<float>(),
        out.tangent_point.data_ptr<float>(),
        out.tangent_edge_t.data_ptr<float>(),
        out.tangent_edge_point.data_ptr<float>());
    return out;
}

} // namespace rayd::torch_backend


// ---- merged from src/edge/edge_topk_part.cu ----

#include <rayd/torch/edge/kernels.h>
#include <rayd/torch/runtime/optix_context.h>
#include <rayd/torch/scene/cache.h>

#include <rayd/shared/edge/bvh_query.h>
#include <rayd/shared/edge/edge_distance_math.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

namespace {

constexpr int kThreads = 128;

__global__ void prepare_topk_queries_kernel(
    const float *__restrict__ point,
    const bool *__restrict__ active,
    int64_t query_count,
    int64_t point_stride0,
    int64_t point_stride1,
    float *__restrict__ query_x,
    float *__restrict__ query_y,
    float *__restrict__ query_z,
    std::uint8_t *__restrict__ query_active) {
    const int64_t query = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count)
        return;
    const float *row = point + query * point_stride0;
    const float x = row[0 * point_stride1];
    const float y = row[1 * point_stride1];
    const float z = row[2 * point_stride1];
    query_x[query] = x;
    query_y[query] = y;
    query_z[query] = z;
    query_active[query] =
        (active == nullptr || active[query]) && isfinite(x) && isfinite(y) && isfinite(z)
        ? 1u
        : 0u;
}

__device__ __forceinline__ bool candidate_precedes(
    float distance,
    int edge,
    float slot_distance,
    int slot_edge) {
    return distance < slot_distance ||
           (distance == slot_distance && edge < slot_edge);
}

// A valid compact tree fits the product traversal stack. If corrupt or unusually
// deep input ever overflows it, recompute only that query exactly instead of
// exposing a partial top-k result.
__global__ void repair_topk_overflow_kernel(
    const float *__restrict__ query_x,
    const float *__restrict__ query_y,
    const float *__restrict__ query_z,
    const std::uint8_t *__restrict__ query_active,
    int64_t query_count,
    int k,
    const float *__restrict__ edge_p0_x,
    const float *__restrict__ edge_p0_y,
    const float *__restrict__ edge_p0_z,
    const float *__restrict__ edge_e1_x,
    const float *__restrict__ edge_e1_y,
    const float *__restrict__ edge_e1_z,
    const std::uint8_t *__restrict__ edge_mask,
    int64_t edge_count,
    std::uint8_t *__restrict__ overflow,
    int *__restrict__ candidate_edge,
    float *__restrict__ candidate_distance_sq,
    float *__restrict__ candidate_edge_t) {
    const int64_t query = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count || overflow[query] == 0u)
        return;

    int edges[shared::edge::kBvhTopKMax];
    float distances[shared::edge::kBvhTopKMax];
    float edge_parameters[shared::edge::kBvhTopKMax];
#pragma unroll
    for (int rank = 0; rank < shared::edge::kBvhTopKMax; ++rank) {
        edges[rank] = -1;
        distances[rank] = CUDART_INF_F;
        edge_parameters[rank] = 0.0f;
    }

    if (query_active[query] != 0u) {
        const shared::math::Vec3f point =
            shared::math::make_vec3(query_x[query], query_y[query], query_z[query]);
        for (int edge = 0; edge < edge_count; ++edge) {
            if (edge_mask != nullptr && edge_mask[edge] == 0u)
                continue;
            const shared::edge::PointSegmentDistance candidate =
                shared::edge::point_segment_distance(
                    point,
                    shared::math::make_vec3(
                        edge_p0_x[edge], edge_p0_y[edge], edge_p0_z[edge]),
                    shared::math::make_vec3(
                        edge_e1_x[edge], edge_e1_y[edge], edge_e1_z[edge]));
            int displaced_edge = edge;
            float displaced_distance = candidate.squared_distance;
            float displaced_parameter = candidate.edge_parameter;
            for (int rank = 0; rank < k; ++rank) {
                if (!candidate_precedes(
                        displaced_distance,
                        displaced_edge,
                        distances[rank],
                        edges[rank]))
                    continue;
                const int next_edge = edges[rank];
                const float next_distance = distances[rank];
                const float next_parameter = edge_parameters[rank];
                edges[rank] = displaced_edge;
                distances[rank] = displaced_distance;
                edge_parameters[rank] = displaced_parameter;
                displaced_edge = next_edge;
                displaced_distance = next_distance;
                displaced_parameter = next_parameter;
                if (displaced_edge < 0)
                    break;
            }
        }
    }

    const int64_t base = query * k;
    for (int rank = 0; rank < k; ++rank) {
        candidate_edge[base + rank] = edges[rank];
        candidate_distance_sq[base + rank] = distances[rank];
        candidate_edge_t[base + rank] = edge_parameters[rank];
    }
    overflow[query] = 0u;
}

__global__ void finalize_topk_kernel(
    const float *__restrict__ query_x,
    const float *__restrict__ query_y,
    const float *__restrict__ query_z,
    const std::uint8_t *__restrict__ query_active,
    int64_t query_count,
    int k,
    const int *__restrict__ candidate_edge,
    const float *__restrict__ candidate_distance_sq,
    const float *__restrict__ candidate_edge_t,
    const float *__restrict__ edge_p0_x,
    const float *__restrict__ edge_p0_y,
    const float *__restrict__ edge_p0_z,
    const float *__restrict__ edge_e1_x,
    const float *__restrict__ edge_e1_y,
    const float *__restrict__ edge_e1_z,
    const int *__restrict__ edge_face1,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    int64_t edge_count,
    bool *__restrict__ is_valid,
    float *__restrict__ distances,
    float *__restrict__ points,
    float *__restrict__ edge_t,
    float *__restrict__ edge_points,
    int *__restrict__ shape_ids,
    int *__restrict__ edge_ids,
    int *__restrict__ global_edge_ids,
    bool *__restrict__ is_boundary,
    int *__restrict__ tape_edge_id,
    float *__restrict__ tape_s,
    float *__restrict__ tape_d) {
    const int64_t slot = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_count = query_count * k;
    if (slot >= output_count)
        return;
    const int64_t query = slot / k;
    const int edge = candidate_edge[slot];
    if (query_active[query] == 0u || edge < 0 || edge >= edge_count)
        return;

    const float s = fminf(fmaxf(candidate_edge_t[slot], 0.0f), 1.0f);
    const float px = edge_p0_x[edge] + s * edge_e1_x[edge];
    const float py = edge_p0_y[edge] + s * edge_e1_y[edge];
    const float pz = edge_p0_z[edge] + s * edge_e1_z[edge];
    const float qx = query_x[query];
    const float qy = query_y[query];
    const float qz = query_z[query];

    is_valid[slot] = true;
    distances[slot] = sqrtf(fmaxf(candidate_distance_sq[slot], 0.0f));
    edge_t[slot] = s;
    shape_ids[slot] = edge_shape_id[edge];
    edge_ids[slot] = edge_local_id[edge];
    global_edge_ids[slot] = edge;
    is_boundary[slot] = edge_face1[edge] < 0;
    tape_edge_id[slot] = edge;
    tape_s[slot] = s;

    const int64_t vector_slot = slot * 3;
    points[vector_slot + 0] = qx;
    points[vector_slot + 1] = qy;
    points[vector_slot + 2] = qz;
    edge_points[vector_slot + 0] = px;
    edge_points[vector_slot + 1] = py;
    edge_points[vector_slot + 2] = pz;
    tape_d[vector_slot + 0] = qx - px;
    tape_d[vector_slot + 1] = qy - py;
    tape_d[vector_slot + 2] = qz - pz;
}

} // namespace

// Device contract: the public edge ops in edge/ops_edge.cpp make the scene
// device current with a c10::cuda::CUDAGuard before dispatching here, so every
// host launcher below resolves its stream on the scene device and adds no
// guard of its own.
EdgeTopKForwardOutputs edge_topk_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &point,
    int64_t k,
    const at::Tensor &active) {
    const int64_t query_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    const int64_t output_count = query_count * k;
    const auto float_options = point.options();
    const auto int_options = scene.edge_v0.options();
    const auto bool_options =
        at::TensorOptions().device(point.device()).dtype(at::kBool);
    const auto byte_options =
        at::TensorOptions().device(point.device()).dtype(at::kByte);

    EdgeTopKForwardOutputs out;
    out.is_valid = at::zeros({query_count, k}, bool_options);
    out.distances = at::full(
        {query_count, k}, std::numeric_limits<float>::infinity(), float_options);
    out.points = at::zeros({query_count, k, 3}, float_options);
    out.edge_t = at::zeros({query_count, k}, float_options);
    out.edge_points = at::zeros({query_count, k, 3}, float_options);
    out.shape_ids = at::full({query_count, k}, -1, int_options);
    out.edge_ids = at::full({query_count, k}, -1, int_options);
    out.global_edge_ids = at::full({query_count, k}, -1, int_options);
    out.is_boundary = at::zeros({query_count, k}, bool_options);
    out.tape_edge_id = at::full({query_count, k}, -1, int_options);
    out.tape_s = at::zeros({query_count, k}, float_options);
    out.tape_d = at::zeros({query_count, k, 3}, float_options);
    if (query_count == 0 || edge_count == 0)
        return out;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device()).stream();
    const int query_blocks = static_cast<int>((query_count + kThreads - 1) / kThreads);
    at::Tensor query_x = at::empty({query_count}, float_options);
    at::Tensor query_y = at::empty({query_count}, float_options);
    at::Tensor query_z = at::empty({query_count}, float_options);
    at::Tensor query_active = at::empty({query_count}, byte_options);
    prepare_topk_queries_kernel<<<query_blocks, kThreads, 0, stream>>>(
        point.data_ptr<float>(),
        active.numel() == 0 ? nullptr : active.data_ptr<bool>(),
        query_count,
        point.stride(0),
        point.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        query_active.data_ptr<std::uint8_t>());

    at::Tensor candidate_edge = at::empty({query_count, k}, int_options);
    at::Tensor candidate_distance_sq = at::empty({query_count, k}, float_options);
    at::Tensor candidate_edge_t = at::empty({query_count, k}, float_options);
    at::Tensor traversal_stack = at::empty(
        {shared::edge::kBvhTraversalStackDepth, query_count}, int_options);
    at::Tensor overflow = at::empty({query_count}, byte_options);

    shared::edge::PointBvhQueryParams params = {};
    params.edges = scene_edge_view(scene);
    params.node_bounds = scene_edge_bvh_bounds_view(scene);
    params.topology = scene_edge_bvh_topology_view(scene);
    params.points = {
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        static_cast<std::size_t>(query_count),
    };
    params.output = {
        candidate_edge.data_ptr<int>(),
        candidate_distance_sq.data_ptr<float>(),
        candidate_edge_t.data_ptr<float>(),
        nullptr,
        static_cast<std::size_t>(query_count),
        static_cast<std::size_t>(k),
        static_cast<std::size_t>(k),
        static_cast<std::size_t>(output_count),
    };
    params.scratch = {
        traversal_stack.data_ptr<int>(),
        overflow.data_ptr<std::uint8_t>(),
        static_cast<std::size_t>(query_count),
        static_cast<std::size_t>(shared::edge::kBvhTraversalStackDepth),
        static_cast<std::size_t>(traversal_stack.numel()),
        static_cast<std::size_t>(overflow.numel()),
    };
    params.active_mask = query_active.data_ptr<std::uint8_t>();
    params.edge_mask = scene.edge_mask.numel() == 0
        ? nullptr
        : scene.edge_mask.data_ptr<std::uint8_t>();
    params.stream = stream;
    shared::edge::launch_point_bvh_query_async(params);

    repair_topk_overflow_kernel<<<query_blocks, kThreads, 0, stream>>>(
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        query_active.data_ptr<std::uint8_t>(),
        query_count,
        static_cast<int>(k),
        scene.edge_p0_x.data_ptr<float>(),
        scene.edge_p0_y.data_ptr<float>(),
        scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),
        scene.edge_e1_y.data_ptr<float>(),
        scene.edge_e1_z.data_ptr<float>(),
        params.edge_mask,
        edge_count,
        overflow.data_ptr<std::uint8_t>(),
        candidate_edge.data_ptr<int>(),
        candidate_distance_sq.data_ptr<float>(),
        candidate_edge_t.data_ptr<float>());

    const int output_blocks = static_cast<int>((output_count + kThreads - 1) / kThreads);
    finalize_topk_kernel<<<output_blocks, kThreads, 0, stream>>>(
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        query_active.data_ptr<std::uint8_t>(),
        query_count,
        static_cast<int>(k),
        candidate_edge.data_ptr<int>(),
        candidate_distance_sq.data_ptr<float>(),
        candidate_edge_t.data_ptr<float>(),
        scene.edge_p0_x.data_ptr<float>(),
        scene.edge_p0_y.data_ptr<float>(),
        scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),
        scene.edge_e1_y.data_ptr<float>(),
        scene.edge_e1_z.data_ptr<float>(),
        scene.edge_face1.data_ptr<int>(),
        scene.edge_shape_id.data_ptr<int>(),
        scene.edge_local_id.data_ptr<int>(),
        edge_count,
        out.is_valid.data_ptr<bool>(),
        out.distances.data_ptr<float>(),
        out.points.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.edge_points.data_ptr<float>(),
        out.shape_ids.data_ptr<int>(),
        out.edge_ids.data_ptr<int>(),
        out.global_edge_ids.data_ptr<int>(),
        out.is_boundary.data_ptr<bool>(),
        out.tape_edge_id.data_ptr<int>(),
        out.tape_s.data_ptr<float>(),
        out.tape_d.data_ptr<float>());
    cuda_check(cudaGetLastError(), "edge_topk_forward_cuda() kernel launch");
    return out;
}

__global__ void prepare_ray_query_kernel(
    const float *ray_o, const float *ray_d, const float *ray_tmax,
    const bool *active, int64_t count,
    int64_t o_s0, int64_t o_s1, int64_t d_s0, int64_t d_s1,
    float *ox, float *oy, float *oz, float *dx, float *dy, float *dz,
    float *tmax, std::uint8_t *query_active) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= count)
        return;
    const float *o = ray_o + ray * o_s0;
    const float *d = ray_d + ray * d_s0;
    ox[ray] = o[0 * o_s1]; oy[ray] = o[1 * o_s1]; oz[ray] = o[2 * o_s1];
    dx[ray] = d[0 * d_s1]; dy[ray] = d[1 * d_s1]; dz[ray] = d[2 * d_s1];
    tmax[ray] = ray_tmax == nullptr ? CUDART_INF_F : ray_tmax[ray];
    query_active[ray] =
        (active == nullptr || active[ray]) && isfinite(ox[ray]) && isfinite(oy[ray]) &&
        isfinite(oz[ray]) && isfinite(dx[ray]) && isfinite(dy[ray]) && isfinite(dz[ray])
        ? 1u : 0u;
}

__global__ void repair_ray_overflow_kernel(
    const float *ox, const float *oy, const float *oz,
    const float *dx, const float *dy, const float *dz, const float *tmax,
    const std::uint8_t *active, int64_t query_count,
    const float *p0x, const float *p0y, const float *p0z,
    const float *e1x, const float *e1y, const float *e1z,
    const std::uint8_t *edge_mask, int64_t edge_count,
    std::uint8_t *overflow, int *out_edge, float *out_distance_sq,
    float *out_edge_t, float *out_ray_t) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= query_count || overflow[ray] == 0u)
        return;
    int best_edge = -1;
    float best_distance = CUDART_INF_F;
    float best_edge_t = 0.0f;
    float best_ray_t = 0.0f;
    if (active[ray] != 0u) {
        const auto o = shared::math::make_vec3(ox[ray], oy[ray], oz[ray]);
        const auto d = shared::math::make_vec3(dx[ray], dy[ray], dz[ray]);
        const float limit = fmaxf(tmax[ray], 0.0f);
        for (int edge = 0; edge < edge_count; ++edge) {
            if (edge_mask != nullptr && edge_mask[edge] == 0u)
                continue;
            const auto p0 = shared::math::make_vec3(p0x[edge], p0y[edge], p0z[edge]);
            const auto e1 = shared::math::make_vec3(e1x[edge], e1y[edge], e1z[edge]);
            float distance;
            float edge_t;
            float ray_t;
            if (isinf(tmax[ray])) {
                const auto candidate = shared::edge::ray_segment_distance(o, d, p0, e1);
                distance = candidate.squared_distance;
                edge_t = candidate.edge_parameter;
                ray_t = candidate.ray_parameter;
            } else {
                const auto candidate = shared::edge::segment_segment_distance(
                    o, shared::math::scale(d, limit), p0, e1);
                distance = candidate.squared_distance;
                edge_t = candidate.edge_parameter;
                ray_t = limit > 0.0f ? candidate.query_parameter * limit : 0.0f;
            }
            if (distance < best_distance || (distance == best_distance && edge < best_edge)) {
                best_edge = edge; best_distance = distance;
                best_edge_t = edge_t; best_ray_t = ray_t;
            }
        }
    }
    out_edge[ray] = best_edge;
    out_distance_sq[ray] = best_distance;
    out_edge_t[ray] = best_edge_t;
    out_ray_t[ray] = best_ray_t;
    overflow[ray] = 0u;
}

__global__ void finalize_ray_query_kernel(
    const float *ox, const float *oy, const float *oz,
    const float *dx, const float *dy, const float *dz,
    const std::uint8_t *active, int64_t count,
    const int *candidate_edge, const float *distance_sq,
    const float *candidate_ray_t, const float *candidate_edge_t,
    const float *p0x, const float *p0y, const float *p0z,
    const float *e1x, const float *e1y, const float *e1z,
    const int *edge_shape, const int *edge_local,
    float *out_distance, float *out_ray_t, float *out_point,
    float *out_edge_t, float *out_edge_point, int *out_shape,
    int *out_local, int *out_global, int *out_tape) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= count)
        return;
    const int edge = candidate_edge[ray];
    out_distance[ray] = CUDART_INF_F; out_ray_t[ray] = 0.0f; out_edge_t[ray] = 0.0f;
    out_shape[ray] = -1; out_local[ray] = -1; out_global[ray] = -1; out_tape[ray] = -1;
    for (int axis = 0; axis < 3; ++axis) {
        out_point[ray * 3 + axis] = 0.0f;
        out_edge_point[ray * 3 + axis] = 0.0f;
    }
    if (active[ray] == 0u || edge < 0)
        return;
    const float rt = candidate_ray_t[ray];
    const float et = candidate_edge_t[ray];
    out_distance[ray] = sqrtf(fmaxf(distance_sq[ray], 0.0f));
    out_ray_t[ray] = rt; out_edge_t[ray] = et;
    out_shape[ray] = edge_shape[edge]; out_local[ray] = edge_local[edge];
    out_global[ray] = edge; out_tape[ray] = edge;
    out_point[ray * 3 + 0] = ox[ray] + rt * dx[ray];
    out_point[ray * 3 + 1] = oy[ray] + rt * dy[ray];
    out_point[ray * 3 + 2] = oz[ray] + rt * dz[ray];
    out_edge_point[ray * 3 + 0] = p0x[edge] + et * e1x[edge];
    out_edge_point[ray * 3 + 1] = p0y[edge] + et * e1y[edge];
    out_edge_point[ray * 3 + 2] = p0z[edge] + et * e1z[edge];
}

EdgeForwardOutputs edge_forward_bvh_cuda(SceneCache &scene, const at::Tensor &point) {
    if (point.size(0) != 0 && scene.edge_v0.numel() != 0)
        ensure_custom_edge_bvh(scene);
    at::Tensor active = at::empty({0}, point.options().dtype(at::kBool));
    EdgeTopKForwardOutputs top = edge_topk_forward_cuda(scene, point, 1, active);
    return {
        top.distances.select(1, 0),
        top.edge_points.select(1, 0),
        top.edge_t.select(1, 0),
        top.shape_ids.select(1, 0),
        top.edge_ids.select(1, 0),
        top.global_edge_ids.select(1, 0),
        top.tape_edge_id.select(1, 0),
        top.tape_s.select(1, 0),
        top.tape_d.select(1, 0),
    };
}

EdgeForwardPublicOutputs edge_forward_noad_bvh_cuda(
    SceneCache &scene,
    const at::Tensor &point) {
    EdgeForwardOutputs out = edge_forward_bvh_cuda(scene, point);
    return {out.distance, out.edge_point, out.edge_t, out.shape_id,
            out.edge_id, out.global_edge_id};
}

EdgeRayForwardOutputs edge_ray_forward_bvh_cuda(
    SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    if (ray_o.size(0) != 0 && scene.edge_v0.numel() != 0)
        ensure_custom_edge_bvh(scene);
    const int64_t count = ray_o.size(0);
    const auto fopts = ray_o.options();
    const auto iopts = scene.edge_v0.options();
    const auto bopts = at::TensorOptions().device(ray_o.device()).dtype(at::kByte);
    EdgeRayForwardOutputs out;
    out.distance = at::empty({count}, fopts); out.ray_t = at::empty({count}, fopts);
    out.point = at::empty({count, 3}, fopts); out.edge_t = at::empty({count}, fopts);
    out.edge_point = at::empty({count, 3}, fopts); out.shape_id = at::empty({count}, iopts);
    out.edge_id = at::empty({count}, iopts); out.global_edge_id = at::empty({count}, iopts);
    out.tape_edge_id = at::empty({count}, iopts);
    if (count == 0)
        return out;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    at::Tensor ox=at::empty({count},fopts), oy=at::empty({count},fopts), oz=at::empty({count},fopts);
    at::Tensor dx=at::empty({count},fopts), dy=at::empty({count},fopts), dz=at::empty({count},fopts);
    at::Tensor tmax=at::empty({count},fopts), query_active=at::empty({count},bopts);
    prepare_ray_query_kernel<<<blocks,kThreads,0,stream>>>(
        ray_o.data_ptr<float>(),ray_d.data_ptr<float>(),
        ray_tmax.numel()==0?nullptr:ray_tmax.data_ptr<float>(),
        active.numel()==0?nullptr:active.data_ptr<bool>(),count,
        ray_o.stride(0),ray_o.stride(1),ray_d.stride(0),ray_d.stride(1),
        ox.data_ptr<float>(),oy.data_ptr<float>(),oz.data_ptr<float>(),
        dx.data_ptr<float>(),dy.data_ptr<float>(),dz.data_ptr<float>(),
        tmax.data_ptr<float>(),query_active.data_ptr<uint8_t>());
    at::Tensor edge=at::empty({count},iopts), dist=at::empty({count},fopts);
    at::Tensor et=at::empty({count},fopts), rt=at::empty({count},fopts);
    at::Tensor stack=at::empty({shared::edge::kBvhTraversalStackDepth,count},iopts);
    at::Tensor overflow=at::empty({count},bopts);
    shared::edge::RayBvhQueryParams params={};
    params.edges=scene_edge_view(scene); params.node_bounds=scene_edge_bvh_bounds_view(scene);
    params.topology=scene_edge_bvh_topology_view(scene);
    params.rays={ox.data_ptr<float>(),oy.data_ptr<float>(),oz.data_ptr<float>(),
                 dx.data_ptr<float>(),dy.data_ptr<float>(),dz.data_ptr<float>(),
                 tmax.data_ptr<float>(),static_cast<size_t>(count)};
    params.output={edge.data_ptr<int>(),dist.data_ptr<float>(),et.data_ptr<float>(),rt.data_ptr<float>(),
                   static_cast<size_t>(count),1,1,static_cast<size_t>(count)};
    params.scratch={stack.data_ptr<int>(),overflow.data_ptr<uint8_t>(),static_cast<size_t>(count),
                    static_cast<size_t>(shared::edge::kBvhTraversalStackDepth),
                    static_cast<size_t>(stack.numel()),static_cast<size_t>(overflow.numel())};
    params.active_mask=query_active.data_ptr<uint8_t>(); params.edge_mask=scene.edge_mask.data_ptr<uint8_t>();
    params.stream=stream; shared::edge::launch_ray_bvh_query_async(params);
    repair_ray_overflow_kernel<<<blocks,kThreads,0,stream>>>(
        ox.data_ptr<float>(),oy.data_ptr<float>(),oz.data_ptr<float>(),dx.data_ptr<float>(),dy.data_ptr<float>(),dz.data_ptr<float>(),
        tmax.data_ptr<float>(),query_active.data_ptr<uint8_t>(),count,
        scene.edge_p0_x.data_ptr<float>(),scene.edge_p0_y.data_ptr<float>(),scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),scene.edge_e1_y.data_ptr<float>(),scene.edge_e1_z.data_ptr<float>(),
        params.edge_mask,scene.edge_v0.numel(),overflow.data_ptr<uint8_t>(),edge.data_ptr<int>(),dist.data_ptr<float>(),et.data_ptr<float>(),rt.data_ptr<float>());
    finalize_ray_query_kernel<<<blocks,kThreads,0,stream>>>(
        ox.data_ptr<float>(),oy.data_ptr<float>(),oz.data_ptr<float>(),dx.data_ptr<float>(),dy.data_ptr<float>(),dz.data_ptr<float>(),
        query_active.data_ptr<uint8_t>(),count,edge.data_ptr<int>(),dist.data_ptr<float>(),rt.data_ptr<float>(),et.data_ptr<float>(),
        scene.edge_p0_x.data_ptr<float>(),scene.edge_p0_y.data_ptr<float>(),scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),scene.edge_e1_y.data_ptr<float>(),scene.edge_e1_z.data_ptr<float>(),
        scene.edge_shape_id.data_ptr<int>(),scene.edge_local_id.data_ptr<int>(),out.distance.data_ptr<float>(),out.ray_t.data_ptr<float>(),
        out.point.data_ptr<float>(),out.edge_t.data_ptr<float>(),out.edge_point.data_ptr<float>(),out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),out.global_edge_id.data_ptr<int>(),out.tape_edge_id.data_ptr<int>());
    cuda_check(cudaGetLastError(),"edge_ray_forward_bvh_cuda");
    return out;
}

} // namespace rayd::torch_backend
