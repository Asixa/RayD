#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/shared/edge/bvh_types.h>

namespace rayd::shared::edge {

/// Compute primitive AABBs and a packed bounds scratch array.
struct PrimitiveBoundsParams {
    EdgeSoAView edges;
    MutableAabbSoAView primitive_bounds;
    BvhBounds3 *packed_bounds;
    cudaStream_t stream;
};

/// Fill a caller-owned integer buffer with the identity permutation.
struct SequenceInitParams {
    std::int32_t *values;
    std::int32_t count;
    cudaStream_t stream;
};

/// Compute Morton codes from primitive bounds and caller-provided scene bounds.
struct MortonCodeParams {
    AabbSoAView primitive_bounds;
    BvhBounds3 scene_bounds;
    std::uint32_t *morton_codes;
    cudaStream_t stream;
};

/// Emit Karras LBVH child/parent topology from a sorted Morton stream.
struct RadixTreeParams {
    const std::uint32_t *morton_codes;
    const std::int32_t *sorted_primitives;
    std::int32_t *left_child;
    std::int32_t *right_child;
    std::int32_t *parent;
    std::int32_t primitive_count;
    cudaStream_t stream;
};

/// Finalize leaf records and atomically merge bounds toward the root.
struct LeafBoundsFinalizeParams {
    const std::int32_t *sorted_primitives;
    const std::int32_t *parent;
    AabbSoAView primitive_bounds;
    const std::int32_t *left_child;
    const std::int32_t *right_child;
    MutableAabbSoAView node_bounds;
    std::int32_t *leaf_primitive;
    std::int32_t *is_leaf;
    std::int32_t *primitive_leaf_node;
    std::int32_t *merge_counters;
    std::int32_t primitive_count;
    cudaStream_t stream;
};

/// Initialize per-leaf inflated SAH costs.
struct LeafCostParams {
    AabbSoAView node_bounds;
    float *node_cost;
    float inflation;
    std::int32_t primitive_count;
    cudaStream_t stream;
};

/// Initialize internal SAH costs using a caller-cleared arrival counter.
struct InternalCostParams {
    const std::int32_t *left_child;
    const std::int32_t *right_child;
    const std::int32_t *parent;
    AabbSoAView node_bounds;
    float *node_cost;
    std::int32_t *arrival_counter;
    float inflation;
    std::int32_t primitive_count;
    cudaStream_t stream;
};

/// Reorganize one host-prepared level of independent GPU treelets.
struct TreeletOptimizeParams {
    const std::int32_t *selected_nodes;
    const std::int32_t *is_leaf;
    std::int32_t *left_child;
    std::int32_t *right_child;
    std::int32_t *parent;
    MutableAabbSoAView node_bounds;
    std::int32_t *leaf_primitive;
    float *node_cost;
    float inflation;
    std::int32_t selected_count;
    cudaStream_t stream;
};

/// Inputs and caller-owned outputs for an asynchronous edge BVH build.
struct BvhBuildParams {
    EdgeSoAView edges;
    MutableAabbSoAView primitive_bounds;
    MutableAabbSoAView node_bounds;
    MutableRawBvhTopologyView topology;
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
    RawBvhTopologyView topology;
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

/// Launch build stages. These functions allocate no memory, perform no copies,
/// and do not synchronize or consume CUDA errors.
void launch_compute_primitive_bounds_async(const PrimitiveBoundsParams &params);
void launch_init_sequence_async(const SequenceInitParams &params);
void launch_compute_morton_codes_async(const MortonCodeParams &params);
void launch_build_radix_tree_async(const RadixTreeParams &params);
void launch_finalize_leaves_and_bounds_async(const LeafBoundsFinalizeParams &params);
void launch_initialize_leaf_costs_async(const LeafCostParams &params);
void launch_initialize_internal_costs_async(const InternalCostParams &params);
void launch_optimize_selected_treelets_async(const TreeletOptimizeParams &params);

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

static_assert(std::is_standard_layout_v<PrimitiveBoundsParams>);
static_assert(std::is_trivially_copyable_v<PrimitiveBoundsParams>);
static_assert(std::is_standard_layout_v<SequenceInitParams>);
static_assert(std::is_trivially_copyable_v<SequenceInitParams>);
static_assert(std::is_standard_layout_v<MortonCodeParams>);
static_assert(std::is_trivially_copyable_v<MortonCodeParams>);
static_assert(std::is_standard_layout_v<RadixTreeParams>);
static_assert(std::is_trivially_copyable_v<RadixTreeParams>);
static_assert(std::is_standard_layout_v<LeafBoundsFinalizeParams>);
static_assert(std::is_trivially_copyable_v<LeafBoundsFinalizeParams>);
static_assert(std::is_standard_layout_v<LeafCostParams>);
static_assert(std::is_trivially_copyable_v<LeafCostParams>);
static_assert(std::is_standard_layout_v<InternalCostParams>);
static_assert(std::is_trivially_copyable_v<InternalCostParams>);
static_assert(std::is_standard_layout_v<TreeletOptimizeParams>);
static_assert(std::is_trivially_copyable_v<TreeletOptimizeParams>);

} // namespace rayd::shared::edge
