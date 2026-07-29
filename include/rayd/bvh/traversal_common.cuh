// Copyright Xingyu Chen.
// Defines shared bvh support for traversal common.

#pragma once

#include <cstddef>

namespace rayd::shared::bvh {

// Primitive-agnostic depth-major traversal-stack helpers. The scratch view is
// supplied by the caller so any BVH consumer (edge queries today, other
// primitives later) can share the identical coalesced push/pop indexing. The
// Scratch type must expose node_indices, stack_depth, query_stride, and
// capacity. Behaviour is bitwise identical to an inlined per-query stack.

/// Store `node` at the given stack depth for `query`. Returns false without
/// writing when the depth exceeds capacity or the view is unallocated.
template <typename Scratch>
__device__ __forceinline__ bool stack_push(const Scratch &scratch,
                                           std::size_t query,
                                           std::size_t depth,
                                           int node) {
    if (scratch.node_indices == nullptr || depth >= scratch.stack_depth) {
        return false;
    }
    const std::size_t slot = depth * scratch.query_stride + query;
    if (query >= scratch.query_stride || slot >= scratch.capacity) {
        return false;
    }
    scratch.node_indices[slot] = node;
    return true;
}

/// Load the node deferred at `depth_index` for `query`.
template <typename Scratch>
__device__ __forceinline__ int stack_load(const Scratch &scratch,
                                          std::size_t query,
                                          std::size_t depth_index) {
    return scratch.node_indices[depth_index * scratch.query_stride + query];
}

/// Deterministic near/far ordering for a two-child descent: the smaller bound
/// wins, ties break on the lower child index so traversal order is stable.
__device__ __forceinline__ bool near_child_is_left(float left_bound,
                                                   float right_bound,
                                                   int left,
                                                   int right) {
    return left_bound < right_bound || (left_bound == right_bound && left < right);
}

} // namespace rayd::shared::bvh
