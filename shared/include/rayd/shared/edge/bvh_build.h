#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/shared/edge/bvh_types.h>

namespace rayd::shared::edge {

/// Inputs and caller-owned outputs for an asynchronous edge BVH build.
struct BvhBuildParams {
    EdgeSoAView edges;
    MutableAabbSoAView primitive_bounds;
    MutableAabbSoAView node_bounds;
    MutableBvhTopologyView topology;
    DeviceScratchView scratch;
    cudaStream_t stream;
};

/// Dirty primitive selection for an asynchronous topology-preserving refit.
struct BvhRefitSelection {
    const std::int32_t *primitive_ids;
    std::size_t count;
};

/// Inputs and caller-owned outputs for an asynchronous edge BVH refit.
struct BvhRefitParams {
    EdgeSoAView edges;
    BvhTopologyView topology;
    MutableAabbSoAView primitive_bounds;
    MutableAabbSoAView node_bounds;
    BvhRefitSelection dirty_primitives;
    DeviceScratchView scratch;
    cudaStream_t stream;
};

/// Caller-owned inputs and output for marking dirty BVH ancestors.
struct DirtyAncestorMarkParams {
    const std::int32_t *leaf_nodes;
    const std::int32_t *node_parent;
    std::int32_t *dirty_marks;
    std::int32_t leaf_count;
    cudaStream_t stream;
};

/// Caller-owned scratch selection for compacting one dirty BVH level.
struct DirtyLevelCompactParams {
    const std::int32_t *level_nodes;
    const std::int32_t *dirty_marks;
    std::int32_t *selected_nodes;
    std::int32_t *selected_count;
    std::int32_t level_count;
    cudaStream_t stream;
};

/// Caller-owned inputs and outputs for refitting selected internal nodes.
struct InternalNodeRefitParams {
    const std::int32_t *selected_count;
    const std::int32_t *selected_nodes;
    const std::int32_t *left_child;
    const std::int32_t *right_child;
    MutableAabbSoAView node_bounds;
    std::int32_t max_selected_count;
    cudaStream_t stream;
};

/// Launch only the dirty-ancestor marking kernel. No allocation, clearing, or synchronization.
void launch_mark_dirty_ancestors_async(const DirtyAncestorMarkParams &params);

/// Launch only one dirty-level compaction kernel. The caller clears selected_count.
void launch_compact_dirty_level_async(const DirtyLevelCompactParams &params);

/// Launch only one selected-internal-node refit kernel.
void launch_refit_selected_internal_nodes_async(const InternalNodeRefitParams &params);

static_assert(std::is_standard_layout_v<BvhBuildParams>);
static_assert(std::is_trivially_copyable_v<BvhBuildParams>);
static_assert(std::is_standard_layout_v<BvhRefitSelection>);
static_assert(std::is_trivially_copyable_v<BvhRefitSelection>);
static_assert(std::is_standard_layout_v<BvhRefitParams>);
static_assert(std::is_trivially_copyable_v<BvhRefitParams>);
static_assert(std::is_standard_layout_v<DirtyAncestorMarkParams>);
static_assert(std::is_trivially_copyable_v<DirtyAncestorMarkParams>);
static_assert(std::is_standard_layout_v<DirtyLevelCompactParams>);
static_assert(std::is_trivially_copyable_v<DirtyLevelCompactParams>);
static_assert(std::is_standard_layout_v<InternalNodeRefitParams>);
static_assert(std::is_trivially_copyable_v<InternalNodeRefitParams>);

} // namespace rayd::shared::edge
