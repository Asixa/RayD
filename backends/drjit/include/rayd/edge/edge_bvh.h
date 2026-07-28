#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace rayd {

// GPU entry points for the edge BVH (implemented in src/edge/edge_bvh.cu). Inputs
// and outputs are flat device pointers in structure-of-arrays layout; the caller
// owns and pre-sizes every buffer. Edges are passed as p0 + e1 (start and edge vector).

/// \brief Explicit CUDA device and stream binding for the edge BVH entry points.
///
/// These entry points used to run on whichever CUDA device and stream happened
/// to be current, so the ABI depended on ambient per-thread state: scratch
/// allocations, kernel launches, and synchronization silently followed a device
/// the caller never named, and a caller that switched the Dr.Jit device between
/// calls got no diagnostic. The caller now captures its binding once and passes
/// it in, so every allocation and launch inside one call is bound to it.
///
/// \c device is a *raw* CUDA device ordinal (`jit_cuda_device_raw()`), not a
/// Dr.Jit device index. \c stream is the caller's CUDA stream
/// (`jit_cuda_stream()`); the null stream is accepted and selects the legacy
/// default stream.
struct EdgeBvhCudaContext {
    int device;
    cudaStream_t stream;
};

/// \brief Build the edge BVH (LBVH via Morton codes) and emit per-primitive and per-node bounds.
///
/// Writes the node hierarchy (children, leaf assignment, leaf flag) and both
/// primitive- and node-level AABBs into the caller-owned output buffers.
///
/// The build owns its internal streams and drains them before returning;
/// \p context supplies the device they are created on and the stream the build
/// orders itself against.
void build_edge_bvh_gpu(
    const EdgeBvhCudaContext &context,
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
///
/// Launches on \c context.stream and drains that stream only, so the inputs the
/// caller evaluated there and the OptiX build that consumes \p out_aabbs stay
/// ordered without a device-wide synchronize.
void compute_edge_optix_aabbs_gpu(
    const EdgeBvhCudaContext &context,
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
///
/// Asynchronous on \c context.stream; the caller drains it.
void mark_edge_bvh_dirty_ancestors_gpu(
    const EdgeBvhCudaContext &context,
    int node_count,
    int leaf_count,
    const int *leaf_nodes,
    const int *node_parent,
    int *out_dirty_marks,
    bool clear_marks);

/// Refit one BVH level by gathering dirty nodes and recomputing bounds from children.
///
/// Asynchronous on \c context.stream; the caller drains it.
void compact_and_refit_edge_bvh_level_gpu(
    const EdgeBvhCudaContext &context,
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
