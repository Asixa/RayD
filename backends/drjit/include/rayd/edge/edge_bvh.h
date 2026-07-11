#pragma once

#include <cstddef>

namespace rayd {

// GPU entry points for the edge BVH (implemented in src/edge/edge_bvh.cu). Inputs
// and outputs are flat device pointers in structure-of-arrays layout; the caller
// owns and pre-sizes every buffer. Edges are passed as p0 + e1 (start and edge vector).

/// \brief Build the edge BVH (LBVH via Morton codes) and emit per-primitive and per-node bounds.
///
/// Writes the node hierarchy (children, leaf assignment, leaf flag) and both
/// primitive- and node-level AABBs into the caller-owned output buffers.
void build_edge_bvh_gpu(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float *primitive_bbox_min_x,
    float *primitive_bbox_min_y,
    float *primitive_bbox_min_z,
    float *primitive_bbox_max_x,
    float *primitive_bbox_max_y,
    float *primitive_bbox_max_z,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z,
    int *left_child,
    int *right_child,
    int *leaf_primitive,
    int *is_leaf,
    int *primitive_leaf_node);

/// Compute per-edge AABBs inflated by \p inflation, packed for an OptiX custom-primitive build.
void compute_edge_optix_aabbs_gpu(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float inflation,
    float *out_aabbs);

/// Mark every ancestor of the given dirty leaf nodes for refit; \p clear_marks resets first.
void mark_edge_bvh_dirty_ancestors_gpu(
    int node_count,
    int leaf_count,
    const int *leaf_nodes,
    const int *node_parent,
    int *out_dirty_marks,
    bool clear_marks);

/// Refit one BVH level by gathering dirty nodes and recomputing bounds from children.
void compact_and_refit_edge_bvh_level_gpu(
    int level_count,
    const int *level_nodes,
    const int *dirty_marks,
    int *scratch_selected_nodes,
    int *scratch_selected_count,
    const int *left_child,
    const int *right_child,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z);

} // namespace rayd
