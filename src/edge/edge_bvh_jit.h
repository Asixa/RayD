// Copyright Xingyu Chen.
// Declares the private CUDA launch boundary for the Dr.Jit edge BVH.

#pragma once

#include <cuda_runtime_api.h>

namespace rayd {

struct EdgeBvhCudaContext {
    int device;
    cudaStream_t stream;
};

void build_edge_bvh_gpu(const EdgeBvhCudaContext& context, int primitive_count, const float* edge_p0_x,
                        const float* edge_p0_y, const float* edge_p0_z, const float* edge_e1_x, const float* edge_e1_y,
                        const float* edge_e1_z, float* primitive_bbox_min_x, float* primitive_bbox_min_y,
                        float* primitive_bbox_min_z, float* primitive_bbox_max_x, float* primitive_bbox_max_y,
                        float* primitive_bbox_max_z, float* node_bbox_min_x, float* node_bbox_min_y,
                        float* node_bbox_min_z, float* node_bbox_max_x, float* node_bbox_max_y, float* node_bbox_max_z,
                        int* left_child, int* right_child, int* leaf_primitive, int* is_leaf, int* primitive_leaf_node);

/// Device views of one compacted edge BVH as the shared compact-BVH traversal consumes it. The two optional
/// pointers select a masked query: per-node active primitive counts and a per-edge byte mask, both null when
/// every edge is active.
struct EdgeBvhQueryView {
    const float* edge_p0[3];
    const float* edge_e1[3];
    const float* node_bbox_min[3];
    const float* node_bbox_max[3];
    const int* left_child;
    const int* right_child;
    const int* leaf_primitives;
    const int* node_active_count;
    const unsigned char* edge_mask;
    int primitive_count;
    int node_count;
    int leaf_primitive_count;
};

/// Caller-owned per-query inputs and outputs of a native closest-edge query. \p direction and \p t_max are null for
/// point queries. Outputs hold \p k slots per query ordered by (squared distance, edge id); empty slots carry id -1.
/// \p stack holds kBvhTraversalStackDepth entries per query.
struct EdgeBvhQueryBatch {
    int query_count;
    int k;
    const float* origin[3];
    const float* direction[3];
    const float* t_max;
    const unsigned char* active;
    int* out_edge_ids;
    float* out_distance_sq;
    int* stack;
};

/// Run the shared compact-BVH closest-edge traversal asynchronously on \p context.stream.
void query_edge_bvh_gpu(const EdgeBvhCudaContext& context, const EdgeBvhQueryView& bvh, const EdgeBvhQueryBatch& batch);

void compute_edge_optix_aabbs_gpu(const EdgeBvhCudaContext& context, int primitive_count, const float* edge_p0_x,
                                  const float* edge_p0_y, const float* edge_p0_z, const float* edge_e1_x,
                                  const float* edge_e1_y, const float* edge_e1_z, float inflation, float* out_aabbs);

void mark_edge_bvh_dirty_ancestors_gpu(const EdgeBvhCudaContext& context, int node_count, int leaf_count,
                                       const int* leaf_nodes, const int* node_parent, int* out_dirty_marks,
                                       bool clear_marks);

void compact_and_refit_edge_bvh_level_gpu(const EdgeBvhCudaContext& context, int level_count, const int* level_nodes,
                                          const int* dirty_marks, int* scratch_selected_nodes,
                                          int* scratch_selected_count, const int* left_child, const int* right_child,
                                          float* node_bbox_min_x, float* node_bbox_min_y, float* node_bbox_min_z,
                                          float* node_bbox_max_x, float* node_bbox_max_y, float* node_bbox_max_z);

} // namespace rayd