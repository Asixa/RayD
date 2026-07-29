// Copyright Xingyu Chen.
// Defines shared bvh support for host topology.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// Primitive-agnostic host BVH topology algorithms. These operate purely on
// std::vector<int> child/leaf arrays and, for compaction, a caller-supplied
// scalar vector type, so both backends can share the identical preorder,
// subtree-count, height, and compaction logic without depending on Dr.Jit,
// Torch, or any device type. Every routine here is behaviour-preserving with
// respect to the per-backend copies it replaces.

namespace rayd::shared::bvh {

/// Memoized number of leaves under \p node_index. \p subtree_leaf_counts must be
/// sized to the node count and initialized to -1.
inline int compute_subtree_leaf_count(int node_index, const std::vector<int>& left_child,
                                      const std::vector<int>& right_child, const std::vector<int>& is_leaf,
                                      std::vector<int>& subtree_leaf_counts) {
    int& count = subtree_leaf_counts[static_cast<size_t>(node_index)];
    if (count >= 0) {
        return count;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        count = 1;
        return count;
    }

    count = compute_subtree_leaf_count(left_child[static_cast<size_t>(node_index)], left_child, right_child, is_leaf,
                                       subtree_leaf_counts) +
            compute_subtree_leaf_count(right_child[static_cast<size_t>(node_index)], left_child, right_child, is_leaf,
                                       subtree_leaf_counts);
    return count;
}

/// Memoized number of primitives under \p node_index. Leaves store their count
/// in right_child. \p subtree_primitive_counts must be sized and initialized to -1.
inline int compute_subtree_primitive_count(int node_index, const std::vector<int>& left_child,
                                           const std::vector<int>& right_child, const std::vector<int>& is_leaf,
                                           std::vector<int>& subtree_primitive_counts) {
    int& count = subtree_primitive_counts[static_cast<size_t>(node_index)];
    if (count >= 0) {
        return count;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        count = std::max(right_child[static_cast<size_t>(node_index)], 0);
        return count;
    }

    count = compute_subtree_primitive_count(left_child[static_cast<size_t>(node_index)], left_child, right_child,
                                            is_leaf, subtree_primitive_counts) +
            compute_subtree_primitive_count(right_child[static_cast<size_t>(node_index)], left_child, right_child,
                                            is_leaf, subtree_primitive_counts);
    return count;
}

/// Memoized height of \p node_index (leaves are height 0). \p heights must be
/// sized to the node count and initialized to -1.
inline int compute_node_height(int node_index, const std::vector<int>& left_child, const std::vector<int>& right_child,
                               const std::vector<int>& is_leaf, std::vector<int>& heights) {
    int& height = heights[static_cast<size_t>(node_index)];
    if (height >= 0) {
        return height;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        height = 0;
        return height;
    }

    height = 1 + std::max(compute_node_height(left_child[static_cast<size_t>(node_index)], left_child, right_child,
                                              is_leaf, heights),
                          compute_node_height(right_child[static_cast<size_t>(node_index)], left_child, right_child,
                                              is_leaf, heights));
    return height;
}

/// Append the primitive ids of every leaf under \p node_index in left-to-right order.
inline void collect_subtree_primitives(int node_index, const std::vector<int>& left_child,
                                       const std::vector<int>& right_child, const std::vector<int>& leaf_primitive,
                                       const std::vector<int>& is_leaf, std::vector<int>& primitives) {
    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        primitives.push_back(leaf_primitive[static_cast<size_t>(node_index)]);
        return;
    }

    collect_subtree_primitives(left_child[static_cast<size_t>(node_index)], left_child, right_child, leaf_primitive,
                               is_leaf, primitives);
    collect_subtree_primitives(right_child[static_cast<size_t>(node_index)], left_child, right_child, leaf_primitive,
                               is_leaf, primitives);
}

/// Host-side dense BVH after compaction: topology, leaf primitives, and per-node
/// bounds. \p Vec3 is the caller's scalar 3-vector (bounds are only copied here,
/// never operated on, so any backend vector type works unchanged).
template <typename Vec3> struct HostCompactedBvh {
    std::vector<int> left_child;
    std::vector<int> right_child;
    std::vector<int> is_leaf;
    std::vector<int> primitive_leaf_nodes;
    std::vector<int> leaf_primitives;
    std::vector<Vec3> node_bbox_min;
    std::vector<Vec3> node_bbox_max;
};

/// Emit the compacted preorder BVH rooted at \p old_node_index. Subtrees with at
/// most \p leaf_size leaves collapse into a single leaf whose primitives are
/// packed contiguously. Leaves encode `left_child = -leaf_begin - 1` and
/// `right_child = leaf_count`. Returns the new node index.
template <typename Vec3>
int emit_compacted_preorder(int old_node_index, const std::vector<int>& left_child, const std::vector<int>& right_child,
                            const std::vector<int>& leaf_primitive, const std::vector<int>& is_leaf,
                            const std::vector<int>& subtree_leaf_counts, const std::vector<Vec3>& node_bbox_min,
                            const std::vector<Vec3>& node_bbox_max, int leaf_size, HostCompactedBvh<Vec3>& compacted) {
    const int new_node_index = static_cast<int>(compacted.is_leaf.size());
    compacted.left_child.push_back(-1);
    compacted.right_child.push_back(-1);
    compacted.is_leaf.push_back(0);
    compacted.node_bbox_min.push_back(node_bbox_min[static_cast<size_t>(old_node_index)]);
    compacted.node_bbox_max.push_back(node_bbox_max[static_cast<size_t>(old_node_index)]);

    const bool collapse_to_leaf = is_leaf[static_cast<size_t>(old_node_index)] > 0 ||
                                  subtree_leaf_counts[static_cast<size_t>(old_node_index)] <= leaf_size;
    if (collapse_to_leaf) {
        compacted.is_leaf[static_cast<size_t>(new_node_index)] = 1;

        std::vector<int> primitives;
        primitives.reserve(static_cast<size_t>(subtree_leaf_counts[static_cast<size_t>(old_node_index)]));
        collect_subtree_primitives(old_node_index, left_child, right_child, leaf_primitive, is_leaf, primitives);

        const int leaf_begin = static_cast<int>(compacted.leaf_primitives.size());
        compacted.left_child[static_cast<size_t>(new_node_index)] = -leaf_begin - 1;
        compacted.right_child[static_cast<size_t>(new_node_index)] = static_cast<int>(primitives.size());
        for (int primitive : primitives) {
            compacted.primitive_leaf_nodes[static_cast<size_t>(primitive)] = new_node_index;
            compacted.leaf_primitives.push_back(primitive);
        }
        return new_node_index;
    }

    const int new_left_child = emit_compacted_preorder(left_child[static_cast<size_t>(old_node_index)], left_child,
                                                       right_child, leaf_primitive, is_leaf, subtree_leaf_counts,
                                                       node_bbox_min, node_bbox_max, leaf_size, compacted);
    const int new_right_child = emit_compacted_preorder(right_child[static_cast<size_t>(old_node_index)], left_child,
                                                        right_child, leaf_primitive, is_leaf, subtree_leaf_counts,
                                                        node_bbox_min, node_bbox_max, leaf_size, compacted);
    compacted.left_child[static_cast<size_t>(new_node_index)] = new_left_child;
    compacted.right_child[static_cast<size_t>(new_node_index)] = new_right_child;
    return new_node_index;
}

} // namespace rayd::shared::bvh
