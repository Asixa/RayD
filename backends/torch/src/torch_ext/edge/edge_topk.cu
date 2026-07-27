#include <rayd/torch/edge/kernels.h>
#include <rayd/torch/common/optix_context.h>
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

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

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
