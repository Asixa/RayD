#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rayd::shared::edge {

/// Read-only structure-of-arrays view of edge segments stored as p0 + direction.
struct EdgeSoAView {
    const float *p0_x;
    const float *p0_y;
    const float *p0_z;
    const float *direction_x;
    const float *direction_y;
    const float *direction_z;
    std::size_t count;
};

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

/// Read-only view of the flattened binary BVH topology.
struct BvhTopologyView {
    const std::int32_t *left_child;
    const std::int32_t *right_child;
    const std::int32_t *leaf_primitive;
    const std::int32_t *is_leaf;
    const std::int32_t *primitive_leaf_node;
    std::size_t node_count;
    std::size_t primitive_count;
};

/// Mutable view of caller-owned flattened binary BVH topology buffers.
struct MutableBvhTopologyView {
    std::int32_t *left_child;
    std::int32_t *right_child;
    std::int32_t *leaf_primitive;
    std::int32_t *is_leaf;
    std::int32_t *primitive_leaf_node;
    std::size_t node_count;
    std::size_t primitive_count;
};

/// Type-erased caller-owned temporary device storage.
struct DeviceScratchView {
    void *data;
    std::size_t size_bytes;
};

#define RAYD_SHARED_EDGE_ASSERT_POD(Type)                                     \
    static_assert(std::is_standard_layout_v<Type>);                           \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_EDGE_ASSERT_POD(EdgeSoAView);
RAYD_SHARED_EDGE_ASSERT_POD(AabbSoAView);
RAYD_SHARED_EDGE_ASSERT_POD(MutableAabbSoAView);
RAYD_SHARED_EDGE_ASSERT_POD(BvhTopologyView);
RAYD_SHARED_EDGE_ASSERT_POD(MutableBvhTopologyView);
RAYD_SHARED_EDGE_ASSERT_POD(DeviceScratchView);

#undef RAYD_SHARED_EDGE_ASSERT_POD

} // namespace rayd::shared::edge
