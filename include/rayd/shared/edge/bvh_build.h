#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <rayd/shared/edge/bvh_types.h>
#include <rayd/shared/bvh/build.h>
#include <rayd/shared/bvh/refit.h>

namespace rayd::shared::edge {

// The primitive-agnostic build and refit parameter structs live in
// <rayd/shared/bvh/build.h> and <rayd/shared/bvh/refit.h>. They are re-exported
// here so every existing rayd::shared::edge:: name keeps resolving. Only the
// primitive-bounds and whole-build contracts carry the edge SoA view.
using bvh::SequenceInitParams;
using bvh::MortonCodeParams;
using bvh::RadixTreeParams;
using bvh::LeafBoundsFinalizeParams;
using bvh::LeafCostParams;
using bvh::InternalCostParams;
using bvh::TreeletOptimizeParams;
using bvh::DirtyAncestorMarkParams;
using bvh::DirtyLevelCompactParams;
using bvh::InternalNodeRefitParams;

/// Compute primitive AABBs and a packed bounds scratch array.
struct PrimitiveBoundsParams {
    EdgeSoAView edges;
    MutableAabbSoAView primitive_bounds;
    BvhBounds3 *packed_bounds;
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

/// Compute edge primitive AABBs. Allocates nothing, performs no copies, and does
/// not synchronize or consume CUDA errors.
void launch_compute_primitive_bounds_async(const PrimitiveBoundsParams &params);

// Thin edge adapters that forward to the shared primitive-agnostic launchers so
// both backend orchestrators keep calling shared::edge::launch_*. These allocate
// no memory, perform no copies, and do not synchronize or consume CUDA errors.
void launch_init_sequence_async(const SequenceInitParams &params);
void launch_compute_morton_codes_async(const MortonCodeParams &params);
void launch_build_radix_tree_async(const RadixTreeParams &params);
void launch_finalize_leaves_and_bounds_async(const LeafBoundsFinalizeParams &params);
void launch_initialize_leaf_costs_async(const LeafCostParams &params);
void launch_initialize_internal_costs_async(const InternalCostParams &params);
void launch_optimize_selected_treelets_async(const TreeletOptimizeParams &params);
void launch_mark_dirty_ancestors_async(const DirtyAncestorMarkParams &params);
void launch_compact_dirty_level_async(const DirtyLevelCompactParams &params);
void launch_refit_selected_internal_nodes_async(const InternalNodeRefitParams &params);

static_assert(std::is_standard_layout_v<PrimitiveBoundsParams>);
static_assert(std::is_trivially_copyable_v<PrimitiveBoundsParams>);
static_assert(std::is_standard_layout_v<BvhBuildParams>);
static_assert(std::is_trivially_copyable_v<BvhBuildParams>);
static_assert(std::is_standard_layout_v<BvhRefitSelection>);
static_assert(std::is_trivially_copyable_v<BvhRefitSelection>);
static_assert(std::is_standard_layout_v<BvhRefitParams>);
static_assert(std::is_trivially_copyable_v<BvhRefitParams>);

} // namespace rayd::shared::edge
