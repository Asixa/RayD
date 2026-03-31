#pragma once

#include <cstddef>

namespace rayd {

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

void build_edge_ploc_bvh_gpu(
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

void collapse_edge_bvh_gpu(
    int primitive_count,
    int raw_node_count,
    const int *raw_left_child,
    const int *raw_right_child,
    const int *raw_leaf_primitive,
    int *out_left_child,
    int *out_right_child,
    int *out_leaf_primitives,
    int *out_primitive_leaf_node);

void compact_edge_bvh_gpu(
    int primitive_count,
    int raw_node_count,
    const int *raw_left_child,
    const int *raw_right_child,
    const int *raw_leaf_primitive,
    const float *raw_node_bbox_min_x,
    const float *raw_node_bbox_min_y,
    const float *raw_node_bbox_min_z,
    const float *raw_node_bbox_max_x,
    const float *raw_node_bbox_max_y,
    const float *raw_node_bbox_max_z,
    int compacted_node_count,
    const int *compacted_left_child,
    const int *compacted_right_child,
    const int *compacted_new_to_old,
    const int *compacted_leaf_begin,
    const int *compacted_leaf_count,
    float *out_node_bbox_min_x,
    float *out_node_bbox_min_y,
    float *out_node_bbox_min_z,
    float *out_node_bbox_max_x,
    float *out_node_bbox_max_y,
    float *out_node_bbox_max_z,
    int *out_left_child,
    int *out_right_child,
    int *out_leaf_primitives,
    int *out_primitive_leaf_node);

} // namespace rayd
