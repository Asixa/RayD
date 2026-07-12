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
    out.distances = at::full({query_count, k}, CUDART_INF_F, float_options);
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

} // namespace rayd::torch_backend
