// Copyright Xingyu Chen.
// Defines shared edge support for bvh types.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <src/bvh_topology.h>

namespace rayd::shared::edge {

// The primitive-agnostic BVH types, treelet constants, and the compact
// leaf-encoding contract (`left_child[node] = -leaf_begin - 1`) now live in
// <src/bvh_topology.h>. They are re-exported here so every existing
// rayd::shared::edge:: name keeps resolving. Only EdgeSoAView is edge-specific.
using bvh::AabbSoAView;
using bvh::BvhBounds3;
using bvh::BvhFloat3;
using bvh::CompactBvhTopologyView;
using bvh::DeviceScratchView;
using bvh::MutableAabbSoAView;
using bvh::MutableRawBvhTopologyView;
using bvh::RawBvhTopologyView;

using bvh::kBvhLeafSize;
using bvh::kBvhTopKMax;
using bvh::kBvhTraversalStackDepth;
using bvh::kBvhTreeletCostInflationRatio;
using bvh::kBvhTreeletMaxLeaves;
using bvh::kBvhTreeletMaxPrimitives;
using bvh::kBvhTreeletMinPrimitives;
using bvh::kBvhTreeletMinSubtreeLeaves;

/// Read-only structure-of-arrays view of edge segments stored as p0 + direction.
struct EdgeSoAView {
    const float* p0_x;
    const float* p0_y;
    const float* p0_z;
    const float* direction_x;
    const float* direction_y;
    const float* direction_z;
    std::size_t count;
};

#define RAYD_SHARED_EDGE_ASSERT_POD(Type)                                                                              \
    static_assert(std::is_standard_layout_v<Type>);                                                                    \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_EDGE_ASSERT_POD(EdgeSoAView);

#undef RAYD_SHARED_EDGE_ASSERT_POD

} // namespace rayd::shared::edge
