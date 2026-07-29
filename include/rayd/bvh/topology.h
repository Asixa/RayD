// Copyright Xingyu Chen.
// Defines shared bvh support for topology.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <rayd/math.h>

namespace rayd::shared::bvh {

/// Compact vector and bounds types used by caller-owned LBVH scratch buffers.
using BvhFloat3 = math::Vec3f;

struct BvhBounds3 {
    BvhFloat3 min;
    BvhFloat3 max;
};

// Product treelet constants are shared so a second backend cannot silently
// compile a different optimizer shape.
inline constexpr std::int32_t kBvhTreeletMaxLeaves = 7;
inline constexpr std::int32_t kBvhTreeletMinPrimitives = 65536;
// The host-prepared GPU treelet pass is verified through 500k primitives. At
// larger sizes, retain the valid LBVH instead of risking a non-transactional
// topology rewrite; large-scene treelet support can be raised after a dedicated
// coverage gate proves primitive preservation.
inline constexpr std::int32_t kBvhTreeletMaxPrimitives = 500000;
inline constexpr std::int32_t kBvhTreeletMinSubtreeLeaves = 32;
inline constexpr float kBvhTreeletCostInflationRatio = 1e-4f;
inline constexpr std::int32_t kBvhLeafSize = 4;
inline constexpr std::int32_t kBvhTraversalStackDepth = 64;
inline constexpr std::int32_t kBvhTopKMax = 16;

/// Read-only structure-of-arrays view of axis-aligned bounds.
struct AabbSoAView {
    const float *min_x;
    const float *min_y;
    const float *min_z;
    const float *max_x;
    const float *max_y;
    const float *max_z;
    std::size_t count;
};

/// Mutable structure-of-arrays view of caller-owned axis-aligned bounds.
struct MutableAabbSoAView {
    float *min_x;
    float *min_y;
    float *min_z;
    float *max_x;
    float *max_y;
    float *max_z;
    std::size_t count;
};

/// Read-only topology produced directly by the GPU LBVH builder.
struct RawBvhTopologyView {
    const std::int32_t *left_child;
    const std::int32_t *right_child;
    const std::int32_t *leaf_primitive;
    const std::int32_t *is_leaf;
    const std::int32_t *primitive_leaf_node;
    std::size_t node_count;
    std::size_t primitive_count;
};

/// Mutable caller-owned topology buffers for the GPU LBVH builder.
struct MutableRawBvhTopologyView {
    std::int32_t *left_child;
    std::int32_t *right_child;
    std::int32_t *leaf_primitive;
    std::int32_t *is_leaf;
    std::int32_t *primitive_leaf_node;
    std::size_t node_count;
    std::size_t primitive_count;
};

/// Read-only compacted preorder topology used by product traversal.
/// Internal nodes store non-negative child indices. Leaves encode
/// `left_child[node] = -leaf_begin - 1` and `right_child[node] = leaf_count`.
struct CompactBvhTopologyView {
    const std::int32_t *left_child;
    const std::int32_t *right_child;
    const std::int32_t *leaf_primitives;
    /// Optional number of active primitives below each node. When present it
    /// must be synchronized with the mask used by the query; pass null while
    /// mask-derived counts are stale.
    const std::int32_t *node_active_count;
    std::size_t node_count;
    std::size_t primitive_count;
    std::size_t leaf_primitive_count;
};

/// Type-erased caller-owned temporary device storage.
struct DeviceScratchView {
    void *data;
    std::size_t size_bytes;
};

#define RAYD_SHARED_BVH_ASSERT_POD(Type)                                      \
    static_assert(std::is_standard_layout_v<Type>);                           \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_BVH_ASSERT_POD(BvhFloat3);
RAYD_SHARED_BVH_ASSERT_POD(BvhBounds3);
RAYD_SHARED_BVH_ASSERT_POD(AabbSoAView);
RAYD_SHARED_BVH_ASSERT_POD(MutableAabbSoAView);
RAYD_SHARED_BVH_ASSERT_POD(RawBvhTopologyView);
RAYD_SHARED_BVH_ASSERT_POD(MutableRawBvhTopologyView);
RAYD_SHARED_BVH_ASSERT_POD(CompactBvhTopologyView);
RAYD_SHARED_BVH_ASSERT_POD(DeviceScratchView);

#undef RAYD_SHARED_BVH_ASSERT_POD

} // namespace rayd::shared::bvh
