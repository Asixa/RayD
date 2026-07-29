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