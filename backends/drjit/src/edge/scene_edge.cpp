#include <algorithm>
#include <array>
#include <limits>
#include <tuple>
#include <vector>

#include <drjit-core/jit.h>
#include <drjit/while_loop.h>

#include <rayd/edge/edge_bvh.h>
#include <rayd/edge/edge_bvh_config.h>

#include <rayd/edge/scene_edge.h>
#include <rayd/utils.h>

namespace rayd {

namespace {

constexpr size_t EdgeBVHTraversalStackSize =
    static_cast<size_t>(shared::edge::kBvhTraversalStackDepth);
constexpr int EdgeBVHPackedBoundsStride = 6;
constexpr int EdgeBVHPackedChildrenStride = 2;
constexpr size_t EdgeBVHDirtyRefitMinPrimitives = 65536;
using TraversalStack = Int;

/// Per-query running top-k during BVH traversal, kept as 16 unrolled (distance, primitive)
/// slots so Dr.Jit can hold the candidate heap in registers rather than indexed memory.
struct TopKTraversalState {
    Float distance0, distance1, distance2, distance3;
    Float distance4, distance5, distance6, distance7;
    Float distance8, distance9, distance10, distance11;
    Float distance12, distance13, distance14, distance15;
    Int primitive0, primitive1, primitive2, primitive3;
    Int primitive4, primitive5, primitive6, primitive7;
    Int primitive8, primitive9, primitive10, primitive11;
    Int primitive12, primitive13, primitive14, primitive15;

    DRJIT_STRUCT(TopKTraversalState,
                 distance0, distance1, distance2, distance3,
                 distance4, distance5, distance6, distance7,
                 distance8, distance9, distance10, distance11,
                 distance12, distance13, distance14, distance15,
                 primitive0, primitive1, primitive2, primitive3,
                 primitive4, primitive5, primitive6, primitive7,
                 primitive8, primitive9, primitive10, primitive11,
                 primitive12, primitive13, primitive14, primitive15)
};

/// Strategy for refitting after edits: refit every node, or only ancestors of dirty leaves.
enum class EdgeBVHRefitStrategy {
    Auto,
    Full,
    DirtyAncestors
};

/// Refit strategy from RAYD_EDGE_BVH_REFIT_STRATEGY (auto/full/dirty_ancestors), cached once.
EdgeBVHRefitStrategy active_edge_bvh_refit_strategy() {
    static const EdgeBVHRefitStrategy value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_REFIT_STRATEGY");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return EdgeBVHRefitStrategy::Auto;
        }
        if (normalized == "full") {
            return EdgeBVHRefitStrategy::Full;
        }
        if (normalized == "dirty_ancestors") {
            return EdgeBVHRefitStrategy::DirtyAncestors;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_REFIT_STRATEGY. Expected one of: auto, full, dirty_ancestors.");
    }();
    return value;
}

/// In Auto mode, choose the dirty-ancestor refit only for large trees with a small dirty fraction.
bool should_use_dirty_ancestor_refit(EdgeBVHRefitStrategy strategy,
                                     size_t primitive_count,
                                     size_t dirty_primitive_count) {
    if (strategy == EdgeBVHRefitStrategy::DirtyAncestors) {
        return true;
    }
    if (strategy != EdgeBVHRefitStrategy::Auto || dirty_primitive_count == 0) {
        return false;
    }

    return primitive_count >= EdgeBVHDirtyRefitMinPrimitives &&
           dirty_primitive_count * 64u <= primitive_count;
}

ScalarVector3f scalar_min(const ScalarVector3f &a, const ScalarVector3f &b) {
    return ScalarVector3f(std::min(a.x(), b.x()),
                          std::min(a.y(), b.y()),
                          std::min(a.z(), b.z()));
}

ScalarVector3f scalar_max(const ScalarVector3f &a, const ScalarVector3f &b) {
    return ScalarVector3f(std::max(a.x(), b.x()),
                          std::max(a.y(), b.y()),
                          std::max(a.z(), b.z()));
}

ScalarVector3f empty_bbox_min() {
    const float inf = std::numeric_limits<float>::infinity();
    return ScalarVector3f(inf, inf, inf);
}

ScalarVector3f empty_bbox_max() {
    const float inf = std::numeric_limits<float>::infinity();
    return ScalarVector3f(-inf, -inf, -inf);
}

float bbox_surface_area(const ScalarVector3f &bbox_min, const ScalarVector3f &bbox_max) {
    const ScalarVector3f extent = scalar_max(bbox_max - bbox_min, ScalarVector3f(0.f, 0.f, 0.f));
    return 2.f * (extent.x() * extent.y() +
                  extent.x() * extent.z() +
                  extent.y() * extent.z());
}

Vector3f zero_vector3(int size) {
    if (size <= 0) {
        return Vector3f();
    }

    return Vector3f(zeros<Float>(size),
                            zeros<Float>(size),
                            zeros<Float>(size));
}

Vector3f empty_vector3(int size) {
    if (size <= 0) {
        return Vector3f();
    }

    return Vector3f(empty<Float>(size),
                    empty<Float>(size),
                    empty<Float>(size));
}

Int load_ints(const std::vector<int> &values) {
    if (values.empty()) {
        return Int();
    }
    return load<Int>(values.data(), values.size());
}

Vector3f load_vector3(const std::vector<ScalarVector3f> &values) {
    const size_t count = values.size();
    if (count == 0) {
        return Vector3f();
    }

    std::vector<float> x(count), y(count), z(count);
    for (size_t index = 0; index < count; ++index) {
        x[index] = values[index].x();
        y[index] = values[index].y();
        z[index] = values[index].z();
    }

    return Vector3f(load<Float>(x.data(), count),
                            load<Float>(y.data(), count),
                            load<Float>(z.data(), count));
}

std::vector<int> copy_ints_to_host(const Int &values) {
    const size_t count = values.size();
    if (count == 0) {
        return {};
    }

    std::vector<int> result(count);
    drjit::store(result.data(), values);
    return result;
}

std::vector<ScalarVector3f> copy_vector3_to_host(const Vector3f &values) {
    const size_t count = static_cast<size_t>(slices(values));
    if (count == 0) {
        return {};
    }

    std::vector<float> x(count), y(count), z(count);
    drjit::store(x.data(), values.x());
    drjit::store(y.data(), values.y());
    drjit::store(z.data(), values.z());

    std::vector<ScalarVector3f> result(count);
    for (size_t index = 0; index < count; ++index) {
        result[index] = ScalarVector3f(x[index], y[index], z[index]);
    }
    return result;
}

std::vector<int> build_preorder_mapping(const std::vector<int> &left_child,
                                        const std::vector<int> &right_child,
                                        const std::vector<int> &is_leaf) {
    const int node_count = static_cast<int>(is_leaf.size());
    if (node_count == 0) {
        return {};
    }

    std::vector<int> new_to_old(static_cast<size_t>(node_count), -1);
    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(node_count));
    stack.push_back(0);

    int next_index = 0;
    while (!stack.empty()) {
        const int old_index = stack.back();
        stack.pop_back();
        new_to_old[static_cast<size_t>(next_index++)] = old_index;

        if (is_leaf[static_cast<size_t>(old_index)] == 0) {
            stack.push_back(right_child[static_cast<size_t>(old_index)]);
            stack.push_back(left_child[static_cast<size_t>(old_index)]);
        }
    }

    require(next_index == node_count,
            "SceneEdge::build(): GPU LBVH traversal did not cover every node.");
    return new_to_old;
}

int compute_subtree_leaf_count(int node_index,
                               const std::vector<int> &left_child,
                               const std::vector<int> &right_child,
                               const std::vector<int> &is_leaf,
                               std::vector<int> &subtree_leaf_counts) {
    int &count = subtree_leaf_counts[static_cast<size_t>(node_index)];
    if (count >= 0) {
        return count;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        count = 1;
        return count;
    }

    count = compute_subtree_leaf_count(left_child[static_cast<size_t>(node_index)],
                                       left_child,
                                       right_child,
                                       is_leaf,
                                       subtree_leaf_counts) +
            compute_subtree_leaf_count(right_child[static_cast<size_t>(node_index)],
                                       left_child,
                                       right_child,
                                       is_leaf,
                                       subtree_leaf_counts);
    return count;
}

int compute_subtree_primitive_count(int node_index,
                                    const std::vector<int> &left_child,
                                    const std::vector<int> &right_child,
                                    const std::vector<int> &is_leaf,
                                    std::vector<int> &subtree_primitive_counts) {
    int &count = subtree_primitive_counts[static_cast<size_t>(node_index)];
    if (count >= 0) {
        return count;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        count = std::max(right_child[static_cast<size_t>(node_index)], 0);
        return count;
    }

    count = compute_subtree_primitive_count(left_child[static_cast<size_t>(node_index)],
                                            left_child,
                                            right_child,
                                            is_leaf,
                                            subtree_primitive_counts) +
            compute_subtree_primitive_count(right_child[static_cast<size_t>(node_index)],
                                            left_child,
                                            right_child,
                                            is_leaf,
                                            subtree_primitive_counts);
    return count;
}

int compute_node_height(int node_index,
                        const std::vector<int> &left_child,
                        const std::vector<int> &right_child,
                        const std::vector<int> &is_leaf,
                        std::vector<int> &heights) {
    int &height = heights[static_cast<size_t>(node_index)];
    if (height >= 0) {
        return height;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        height = 0;
        return height;
    }

    height = 1 + std::max(compute_node_height(left_child[static_cast<size_t>(node_index)],
                                              left_child,
                                              right_child,
                                              is_leaf,
                                              heights),
                          compute_node_height(right_child[static_cast<size_t>(node_index)],
                                              left_child,
                                              right_child,
                                              is_leaf,
                                              heights));
    return height;
}

float bbox_overlap_surface_area(const ScalarVector3f &a_min,
                                const ScalarVector3f &a_max,
                                const ScalarVector3f &b_min,
                                const ScalarVector3f &b_max) {
    const ScalarVector3f overlap_min = scalar_max(a_min, b_min);
    const ScalarVector3f overlap_max = scalar_min(a_max, b_max);
    const ScalarVector3f overlap_extent = overlap_max - overlap_min;
    if (overlap_extent.x() <= 0.f || overlap_extent.y() <= 0.f || overlap_extent.z() <= 0.f) {
        return 0.f;
    }
    return bbox_surface_area(overlap_min, overlap_max);
}

void collect_subtree_primitives(int node_index,
                                const std::vector<int> &left_child,
                                const std::vector<int> &right_child,
                                const std::vector<int> &leaf_primitive,
                                const std::vector<int> &is_leaf,
                                std::vector<int> &primitives) {
    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        primitives.push_back(leaf_primitive[static_cast<size_t>(node_index)]);
        return;
    }

    collect_subtree_primitives(left_child[static_cast<size_t>(node_index)],
                               left_child,
                               right_child,
                               leaf_primitive,
                               is_leaf,
                               primitives);
    collect_subtree_primitives(right_child[static_cast<size_t>(node_index)],
                               left_child,
                               right_child,
                               leaf_primitive,
                               is_leaf,
                               primitives);
}

/// Host-side dense BVH after compaction: topology, leaf primitives, and per-node bounds.
struct CompactedEdgeBVH {
    std::vector<int> left_child;
    std::vector<int> right_child;
    std::vector<int> is_leaf;
    std::vector<int> primitive_leaf_nodes;
    std::vector<int> leaf_primitives;
    std::vector<ScalarVector3f> node_bbox_min;
    std::vector<ScalarVector3f> node_bbox_max;
};

int emit_compacted_bvh_preorder(int old_node_index,
                                const std::vector<int> &left_child,
                                const std::vector<int> &right_child,
                                const std::vector<int> &leaf_primitive,
                                const std::vector<int> &is_leaf,
                                const std::vector<int> &subtree_leaf_counts,
                                const std::vector<ScalarVector3f> &node_bbox_min,
                                const std::vector<ScalarVector3f> &node_bbox_max,
                                CompactedEdgeBVH &compacted) {
    const int new_node_index = static_cast<int>(compacted.is_leaf.size());
    compacted.left_child.push_back(-1);
    compacted.right_child.push_back(-1);
    compacted.is_leaf.push_back(0);
    compacted.node_bbox_min.push_back(node_bbox_min[static_cast<size_t>(old_node_index)]);
    compacted.node_bbox_max.push_back(node_bbox_max[static_cast<size_t>(old_node_index)]);

    const bool collapse_to_leaf =
        is_leaf[static_cast<size_t>(old_node_index)] > 0 ||
        subtree_leaf_counts[static_cast<size_t>(old_node_index)] <= EdgeBVHLeafSize;
    if (collapse_to_leaf) {
        compacted.is_leaf[static_cast<size_t>(new_node_index)] = 1;

        std::vector<int> primitives;
        primitives.reserve(static_cast<size_t>(subtree_leaf_counts[static_cast<size_t>(old_node_index)]));
        collect_subtree_primitives(old_node_index,
                                   left_child,
                                   right_child,
                                   leaf_primitive,
                                   is_leaf,
                                   primitives);

        const int leaf_begin = static_cast<int>(compacted.leaf_primitives.size());
        compacted.left_child[static_cast<size_t>(new_node_index)] = -leaf_begin - 1;
        compacted.right_child[static_cast<size_t>(new_node_index)] = static_cast<int>(primitives.size());
        for (int primitive : primitives) {
            compacted.primitive_leaf_nodes[static_cast<size_t>(primitive)] = new_node_index;
            compacted.leaf_primitives.push_back(primitive);
        }
        return new_node_index;
    }

    const int new_left_child = emit_compacted_bvh_preorder(left_child[static_cast<size_t>(old_node_index)],
                                                           left_child,
                                                           right_child,
                                                           leaf_primitive,
                                                           is_leaf,
                                                           subtree_leaf_counts,
                                                           node_bbox_min,
                                                           node_bbox_max,
                                                           compacted);
    const int new_right_child = emit_compacted_bvh_preorder(right_child[static_cast<size_t>(old_node_index)],
                                                            left_child,
                                                            right_child,
                                                            leaf_primitive,
                                                            is_leaf,
                                                            subtree_leaf_counts,
                                                            node_bbox_min,
                                                            node_bbox_max,
                                                            compacted);
    compacted.left_child[static_cast<size_t>(new_node_index)] = new_left_child;
    compacted.right_child[static_cast<size_t>(new_node_index)] = new_right_child;
    return new_node_index;
}

float bbox_cost_inflated(const ScalarVector3f &bbox_min,
                         const ScalarVector3f &bbox_max,
                         float inflation) {
    const ScalarVector3f extent =
        scalar_max(bbox_max - bbox_min, ScalarVector3f(0.f, 0.f, 0.f)) +
        ScalarVector3f(inflation, inflation, inflation);
    return 2.f * (extent.x() * extent.y() +
                  extent.x() * extent.z() +
                  extent.y() * extent.z());
}

int popcount_u32(uint32_t value) {
    int count = 0;
    while (value != 0u) {
        count += static_cast<int>(value & 1u);
        value >>= 1u;
    }
    return count;
}

int first_set_bit_u32(uint32_t value) {
    int index = 0;
    while ((value & 1u) == 0u) {
        value >>= 1u;
        ++index;
    }
    return index;
}

/// Result of building one treelet branch over a subset of leaves: the new root node,
/// its bounds, and its SAH cost (used to compare against the original arrangement).
struct TreeletBuildResult {
    int node_index = -1;
    ScalarVector3f bbox_min;
    ScalarVector3f bbox_max;
    float cost = 0.f;
};

/// Recursively rebuild the optimal sub-tree for a leaf subset (bitmask) during treelet optimization.
TreeletBuildResult rebuild_treelet_branch(
    uint32_t subset,
    const std::array<int, EdgeBVHTreeletMaxLeaves> &frontier_nodes,
    const std::array<uint8_t, 1 << EdgeBVHTreeletMaxLeaves> &optimal_partitions,
    const std::array<int, EdgeBVHTreeletMaxLeaves - 2> &reusable_nodes,
    size_t reusable_count,
    size_t &next_reusable_node,
    std::vector<int> &left_child,
    std::vector<int> &right_child,
    std::vector<int> &leaf_primitive,
    std::vector<int> &is_leaf,
    std::vector<ScalarVector3f> &node_bbox_min,
    std::vector<ScalarVector3f> &node_bbox_max,
    std::vector<float> &subtree_costs,
    float inflation) {
    require(subset != 0u,
            "SceneEdge::build(): attempted to rebuild an empty treelet subset.");

    if (popcount_u32(subset) == 1) {
        const int frontier_index = first_set_bit_u32(subset);
        const int node_index = frontier_nodes[static_cast<size_t>(frontier_index)];
        return TreeletBuildResult{
            node_index,
            node_bbox_min[static_cast<size_t>(node_index)],
            node_bbox_max[static_cast<size_t>(node_index)],
            subtree_costs[static_cast<size_t>(node_index)]
        };
    }

    require(next_reusable_node < reusable_count,
            "SceneEdge::build(): treelet rebuild ran out of internal nodes.");
    const int node_index = reusable_nodes[next_reusable_node++];
    const uint32_t left_subset = optimal_partitions[static_cast<size_t>(subset)];
    const uint32_t right_subset = subset ^ left_subset;
    require(left_subset != 0u && right_subset != 0u,
            "SceneEdge::build(): invalid treelet partition.");

    const TreeletBuildResult left_result = rebuild_treelet_branch(left_subset,
                                                                  frontier_nodes,
                                                                  optimal_partitions,
                                                                  reusable_nodes,
                                                                  reusable_count,
                                                                  next_reusable_node,
                                                                  left_child,
                                                                  right_child,
                                                                  leaf_primitive,
                                                                  is_leaf,
                                                                  node_bbox_min,
                                                                  node_bbox_max,
                                                                  subtree_costs,
                                                                  inflation);
    const TreeletBuildResult right_result = rebuild_treelet_branch(right_subset,
                                                                   frontier_nodes,
                                                                   optimal_partitions,
                                                                   reusable_nodes,
                                                                   reusable_count,
                                                                   next_reusable_node,
                                                                   left_child,
                                                                   right_child,
                                                                   leaf_primitive,
                                                                   is_leaf,
                                                                   node_bbox_min,
                                                                   node_bbox_max,
                                                                   subtree_costs,
                                                                   inflation);

    left_child[static_cast<size_t>(node_index)] = left_result.node_index;
    right_child[static_cast<size_t>(node_index)] = right_result.node_index;
    leaf_primitive[static_cast<size_t>(node_index)] = -1;
    is_leaf[static_cast<size_t>(node_index)] = 0;
    node_bbox_min[static_cast<size_t>(node_index)] =
        scalar_min(left_result.bbox_min, right_result.bbox_min);
    node_bbox_max[static_cast<size_t>(node_index)] =
        scalar_max(left_result.bbox_max, right_result.bbox_max);
    subtree_costs[static_cast<size_t>(node_index)] =
        bbox_cost_inflated(node_bbox_min[static_cast<size_t>(node_index)],
                           node_bbox_max[static_cast<size_t>(node_index)],
                           inflation) +
        left_result.cost + right_result.cost;

    return TreeletBuildResult{
        node_index,
        node_bbox_min[static_cast<size_t>(node_index)],
        node_bbox_max[static_cast<size_t>(node_index)],
        subtree_costs[static_cast<size_t>(node_index)]
    };
}

/// Reorganize the treelet rooted at \p node_index if a cheaper arrangement exists; returns whether it changed.
bool optimize_treelet_at_node(int node_index,
                              std::vector<int> &left_child,
                              std::vector<int> &right_child,
                              std::vector<int> &leaf_primitive,
                              std::vector<int> &is_leaf,
                              std::vector<ScalarVector3f> &node_bbox_min,
                              std::vector<ScalarVector3f> &node_bbox_max,
                              std::vector<float> &subtree_costs,
                              float inflation) {
    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        return false;
    }

    std::array<int, EdgeBVHTreeletMaxLeaves> frontier_nodes{};
    std::array<int, EdgeBVHTreeletMaxLeaves - 2> reusable_nodes{};
    size_t frontier_count = 0;
    size_t reusable_count = 0;

    frontier_nodes[frontier_count++] = left_child[static_cast<size_t>(node_index)];
    frontier_nodes[frontier_count++] = right_child[static_cast<size_t>(node_index)];

    while (frontier_count < EdgeBVHTreeletMaxLeaves) {
        int expand_slot = -1;
        float max_cost = -1.f;
        for (size_t frontier_index = 0; frontier_index < frontier_count; ++frontier_index) {
            const int frontier_node = frontier_nodes[frontier_index];
            if (is_leaf[static_cast<size_t>(frontier_node)] > 0) {
                continue;
            }

            const float candidate_cost =
                bbox_cost_inflated(node_bbox_min[static_cast<size_t>(frontier_node)],
                                   node_bbox_max[static_cast<size_t>(frontier_node)],
                                   inflation);
            if (candidate_cost > max_cost) {
                max_cost = candidate_cost;
                expand_slot = static_cast<int>(frontier_index);
            }
        }

        if (expand_slot < 0) {
            break;
        }

        const int expanded_node = frontier_nodes[static_cast<size_t>(expand_slot)];
        reusable_nodes[reusable_count++] = expanded_node;
        frontier_nodes[static_cast<size_t>(expand_slot)] =
            left_child[static_cast<size_t>(expanded_node)];
        frontier_nodes[frontier_count++] = right_child[static_cast<size_t>(expanded_node)];
    }

    if (frontier_count < 3) {
        return false;
    }

    constexpr size_t MaxTreeletSubsets = 1u << EdgeBVHTreeletMaxLeaves;
    std::array<ScalarVector3f, MaxTreeletSubsets> subset_bbox_min{};
    std::array<ScalarVector3f, MaxTreeletSubsets> subset_bbox_max{};
    std::array<float, MaxTreeletSubsets> subset_bbox_cost{};
    std::array<float, MaxTreeletSubsets> optimal_cost{};
    std::array<uint8_t, MaxTreeletSubsets> optimal_partitions{};

    const uint32_t full_mask = (1u << frontier_count) - 1u;
    for (uint32_t subset = 1u; subset <= full_mask; ++subset) {
        ScalarVector3f bbox_min = empty_bbox_min();
        ScalarVector3f bbox_max = empty_bbox_max();
        for (size_t frontier_index = 0; frontier_index < frontier_count; ++frontier_index) {
            if ((subset & (1u << frontier_index)) == 0u) {
                continue;
            }

            const int frontier_node = frontier_nodes[frontier_index];
            bbox_min = scalar_min(bbox_min, node_bbox_min[static_cast<size_t>(frontier_node)]);
            bbox_max = scalar_max(bbox_max, node_bbox_max[static_cast<size_t>(frontier_node)]);
        }

        subset_bbox_min[static_cast<size_t>(subset)] = bbox_min;
        subset_bbox_max[static_cast<size_t>(subset)] = bbox_max;
        subset_bbox_cost[static_cast<size_t>(subset)] =
            bbox_cost_inflated(bbox_min, bbox_max, inflation);
    }

    for (size_t frontier_index = 0; frontier_index < frontier_count; ++frontier_index) {
        const uint32_t subset = 1u << frontier_index;
        const int frontier_node = frontier_nodes[frontier_index];
        optimal_cost[static_cast<size_t>(subset)] =
            subtree_costs[static_cast<size_t>(frontier_node)];
        optimal_partitions[static_cast<size_t>(subset)] = 0u;
    }

    for (size_t subset_size = 2; subset_size <= frontier_count; ++subset_size) {
        for (uint32_t subset = 1u; subset <= full_mask; ++subset) {
            if (popcount_u32(subset) != static_cast<int>(subset_size)) {
                continue;
            }

            float best_children_cost = std::numeric_limits<float>::infinity();
            uint32_t best_partition = 0u;
            for (uint32_t left_subset = (subset - 1u) & subset;
                 left_subset > 0u;
                 left_subset = (left_subset - 1u) & subset) {
                const uint32_t right_subset = subset ^ left_subset;
                if (right_subset == 0u || left_subset > right_subset) {
                    continue;
                }

                const float candidate_cost =
                    optimal_cost[static_cast<size_t>(left_subset)] +
                    optimal_cost[static_cast<size_t>(right_subset)];
                if (candidate_cost < best_children_cost) {
                    best_children_cost = candidate_cost;
                    best_partition = left_subset;
                }
            }

            require(best_partition != 0u,
            "SceneEdge::build(): failed to find a valid treelet partition.");
            optimal_cost[static_cast<size_t>(subset)] =
                subset_bbox_cost[static_cast<size_t>(subset)] + best_children_cost;
            optimal_partitions[static_cast<size_t>(subset)] =
                static_cast<uint8_t>(best_partition);
        }
    }

    if (!(optimal_cost[static_cast<size_t>(full_mask)] <
          subtree_costs[static_cast<size_t>(node_index)] - 1e-6f)) {
        return false;
    }

    const uint32_t left_subset = optimal_partitions[static_cast<size_t>(full_mask)];
    const uint32_t right_subset = full_mask ^ left_subset;
    require(left_subset != 0u && right_subset != 0u,
            "SceneEdge::build(): invalid root treelet partition.");

    size_t next_reusable_node = 0;
    const TreeletBuildResult left_result = rebuild_treelet_branch(left_subset,
                                                                  frontier_nodes,
                                                                  optimal_partitions,
                                                                  reusable_nodes,
                                                                  reusable_count,
                                                                  next_reusable_node,
                                                                  left_child,
                                                                  right_child,
                                                                  leaf_primitive,
                                                                  is_leaf,
                                                                  node_bbox_min,
                                                                  node_bbox_max,
                                                                  subtree_costs,
                                                                  inflation);
    const TreeletBuildResult right_result = rebuild_treelet_branch(right_subset,
                                                                   frontier_nodes,
                                                                   optimal_partitions,
                                                                   reusable_nodes,
                                                                   reusable_count,
                                                                   next_reusable_node,
                                                                   left_child,
                                                                   right_child,
                                                                   leaf_primitive,
                                                                   is_leaf,
                                                                   node_bbox_min,
                                                                   node_bbox_max,
                                                                   subtree_costs,
                                                                   inflation);

    require(next_reusable_node == reusable_count,
            "SceneEdge::build(): treelet rebuild did not consume every internal node.");

    left_child[static_cast<size_t>(node_index)] = left_result.node_index;
    right_child[static_cast<size_t>(node_index)] = right_result.node_index;
    leaf_primitive[static_cast<size_t>(node_index)] = -1;
    is_leaf[static_cast<size_t>(node_index)] = 0;
    node_bbox_min[static_cast<size_t>(node_index)] =
        scalar_min(left_result.bbox_min, right_result.bbox_min);
    node_bbox_max[static_cast<size_t>(node_index)] =
        scalar_max(left_result.bbox_max, right_result.bbox_max);
    subtree_costs[static_cast<size_t>(node_index)] =
        bbox_cost_inflated(node_bbox_min[static_cast<size_t>(node_index)],
                           node_bbox_max[static_cast<size_t>(node_index)],
                           inflation) +
        left_result.cost + right_result.cost;
    return true;
}

/// Bottom-up host treelet optimization over the whole tree; returns the (possibly improved) subtree cost.
float optimize_treelets_recursive(int node_index,
                                  std::vector<int> &left_child,
                                  std::vector<int> &right_child,
                                  std::vector<int> &leaf_primitive,
                                  std::vector<int> &is_leaf,
                                  std::vector<ScalarVector3f> &node_bbox_min,
                                  std::vector<ScalarVector3f> &node_bbox_max,
                                  std::vector<float> &subtree_costs,
                                  const std::vector<int> &subtree_leaf_counts,
                                  float inflation) {
    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        subtree_costs[static_cast<size_t>(node_index)] =
            bbox_cost_inflated(node_bbox_min[static_cast<size_t>(node_index)],
                               node_bbox_max[static_cast<size_t>(node_index)],
                               inflation);
        return subtree_costs[static_cast<size_t>(node_index)];
    }

    const int left_index = left_child[static_cast<size_t>(node_index)];
    const int right_index = right_child[static_cast<size_t>(node_index)];
    const float left_cost = optimize_treelets_recursive(left_index,
                                                        left_child,
                                                        right_child,
                                                        leaf_primitive,
                                                        is_leaf,
                                                        node_bbox_min,
                                                        node_bbox_max,
                                                        subtree_costs,
                                                        subtree_leaf_counts,
                                                        inflation);
    const float right_cost = optimize_treelets_recursive(right_index,
                                                         left_child,
                                                         right_child,
                                                         leaf_primitive,
                                                         is_leaf,
                                                         node_bbox_min,
                                                         node_bbox_max,
                                                         subtree_costs,
                                                         subtree_leaf_counts,
                                                         inflation);

    node_bbox_min[static_cast<size_t>(node_index)] =
        scalar_min(node_bbox_min[static_cast<size_t>(left_index)],
                   node_bbox_min[static_cast<size_t>(right_index)]);
    node_bbox_max[static_cast<size_t>(node_index)] =
        scalar_max(node_bbox_max[static_cast<size_t>(left_index)],
                   node_bbox_max[static_cast<size_t>(right_index)]);
    subtree_costs[static_cast<size_t>(node_index)] =
        bbox_cost_inflated(node_bbox_min[static_cast<size_t>(node_index)],
                           node_bbox_max[static_cast<size_t>(node_index)],
                           inflation) +
        left_cost + right_cost;

    if (subtree_leaf_counts[static_cast<size_t>(node_index)] >= EdgeBVHTreeletMinSubtreeLeaves) {
        optimize_treelet_at_node(node_index,
                                 left_child,
                                 right_child,
                                 leaf_primitive,
                                 is_leaf,
                                 node_bbox_min,
                                 node_bbox_max,
                                 subtree_costs,
                                 inflation);
    }
    return subtree_costs[static_cast<size_t>(node_index)];
}

// Per-lane explicit traversal stack: one fixed-size stack per query, packed into a
// single Dr.Jit array. The push/pop helpers operate on all lanes at once, gated by a mask.

/// Allocate a per-query traversal stack (query_count * EdgeBVHTraversalStackSize entries).
TraversalStack make_empty_stack(int query_count) {
    if (query_count <= 0) {
        return Int();
    }
    return full<Int>(-1, query_count * static_cast<int>(EdgeBVHTraversalStackSize));
}

/// Push \p value onto each active lane's stack and advance its size.
void stack_push(TraversalStack &stack,
                const Int &stack_base,
                Int &stack_size,
                const Int &value,
                const Mask &active) {
    const int query_count = static_cast<int>(slices(stack_size));
    const Int write_index = stack_base + stack_size;
    scatter(stack, value, write_index, active);

    stack_size = stack_size +
                 select(active,
                        full<Int>(1, query_count),
                        zeros<Int>(query_count));
}

/// Pop one entry from each active non-empty lane's stack; lanes that cannot pop return -1.
Int stack_pop(TraversalStack &stack,
                      const Int &stack_base,
                      Int &stack_size,
                      const Mask &active) {
    const int query_count = static_cast<int>(slices(stack_size));
    const Mask can_pop = active && (stack_size > 0);
    const Int safe_pop_index =
        select(can_pop,
               stack_base + stack_size - full<Int>(1, query_count),
               zeros<Int>(query_count));
    const Int value = gather<Int>(stack, safe_pop_index, can_pop);

    stack_size = stack_size -
                 select(can_pop,
                        full<Int>(1, query_count),
                        zeros<Int>(query_count));
    return select(can_pop, value, full<Int>(-1, query_count));
}

/// Initialize all top-k slots to (distance = Infinity, primitive = -1) for every query.
TopKTraversalState make_empty_topk_state(int query_count) {
    const Float inf = full<Float>(Infinity, query_count);
    const Int none = full<Int>(-1, query_count);
    return TopKTraversalState{
        inf, inf, inf, inf,
        inf, inf, inf, inf,
        inf, inf, inf, inf,
        inf, inf, inf, inf,
        none, none, none, none,
        none, none, none, none,
        none, none, none, none,
        none, none, none, none,
    };
}

std::array<Float *, 16> topk_distance_slots(TopKTraversalState &state) {
    return { &state.distance0, &state.distance1, &state.distance2, &state.distance3,
             &state.distance4, &state.distance5, &state.distance6, &state.distance7,
             &state.distance8, &state.distance9, &state.distance10, &state.distance11,
             &state.distance12, &state.distance13, &state.distance14, &state.distance15 };
}

std::array<Int *, 16> topk_primitive_slots(TopKTraversalState &state) {
    return { &state.primitive0, &state.primitive1, &state.primitive2, &state.primitive3,
             &state.primitive4, &state.primitive5, &state.primitive6, &state.primitive7,
             &state.primitive8, &state.primitive9, &state.primitive10, &state.primitive11,
             &state.primitive12, &state.primitive13, &state.primitive14, &state.primitive15 };
}

/// Insertion-sort a candidate into each lane's top-k slots, keeping them ordered nearest-first.
void topk_insert_candidate(TopKTraversalState &state,
                           int k,
                           const Int &primitive_index,
                           const Float &candidate_distance_sq,
                           const Mask &active) {
    auto distances = topk_distance_slots(state);
    auto primitives = topk_primitive_slots(state);

    Mask carry_active = active;
    Int carry_primitive = primitive_index;
    Float carry_distance_sq = candidate_distance_sq;

    for (int rank = 0; rank < k; ++rank) {
        const Float slot_distance_sq = *distances[static_cast<size_t>(rank)];
        const Int slot_primitive = *primitives[static_cast<size_t>(rank)];
        const Mask insert =
            carry_active && (carry_distance_sq < slot_distance_sq);

        *distances[static_cast<size_t>(rank)] =
            select(insert, carry_distance_sq, slot_distance_sq);
        *primitives[static_cast<size_t>(rank)] =
            select(insert, carry_primitive, slot_primitive);

        const Mask ejected_active = insert && (slot_primitive >= 0);
        carry_distance_sq = select(insert, slot_distance_sq, carry_distance_sq);
        carry_primitive = select(insert, slot_primitive, carry_primitive);
        carry_active = (carry_active && !insert) || ejected_active;
    }
}

DRJIT_INLINE Mask node_is_leaf(const Int &encoded_left_child) {
    return encoded_left_child < 0;
}

DRJIT_INLINE Int node_leaf_begin(const Int &encoded_left_child) {
    return -encoded_left_child - full<Int>(1, slices(encoded_left_child));
}

} // namespace

void SceneEdge::scatter_node_bounds(const Int &node_indices,
                                    const Vector3f &bbox_min,
                                    const Vector3f &bbox_max) {
    scatter(node_bbox_min_, bbox_min, node_indices);
    scatter(node_bbox_max_, bbox_max, node_indices);
}

Int SceneEdge::gather_node_left_child(const Int &node_indices,
                                               const Mask &active) const {
    return gather<Int>(left_child_, node_indices, active);
}

Int SceneEdge::gather_node_right_child(const Int &node_indices,
                                                const Mask &active) const {
    return gather<Int>(right_child_, node_indices, active);
}

Int SceneEdge::gather_node_active_count(const Int &node_indices,
                                                const Mask &active) const {
    return gather<Int>(node_active_count_, node_indices, active);
}

Vector3f SceneEdge::gather_node_bbox_min(const Int &node_indices,
                                                  const Mask &active) const {
    return gather<Vector3f>(node_bbox_min_, node_indices, active);
}

Vector3f SceneEdge::gather_node_bbox_max(const Int &node_indices,
                                                  const Mask &active) const {
    return gather<Vector3f>(node_bbox_max_, node_indices, active);
}

void SceneEdge::build(const SecondaryEdgeInfoAD &edge_info) {
    build(edge_info, true);
}

void SceneEdge::build(const SecondaryEdgeInfoAD &edge_info,
                      bool allow_refit) {
    build_bvh(edge_info, allow_refit);
    if (primitive_count_ > 0) {
        set_all_active_state();
    } else {
        all_active_ = true;
        active_primitive_count_ = 0;
    }
}

void SceneEdge::build(const SecondaryEdgeInfoAD &edge_info,
                      const Mask &mask) {
    build(edge_info, mask, true);
}

void SceneEdge::build(const SecondaryEdgeInfoAD &edge_info,
                      const Mask &mask,
                      bool allow_refit) {
    const int edge_count = edge_info.size();
    require(static_cast<int>(mask.size()) == edge_count,
            "SceneEdge::build(): mask size must match the edge count.");

    build_bvh(edge_info, allow_refit);
    if (edge_count > 0) {
        set_mask(mask);
    } else {
        all_active_ = true;
        active_primitive_count_ = 0;
    }
}

void SceneEdge::set_mask(const Mask &mask) {
    require(ready_, "SceneEdge::set_mask(): BVH is not built.");
    require(static_cast<int>(mask.size()) == primitive_count_,
            "SceneEdge::set_mask(): mask size must match the BVH edge count.");
    update_active_counts_from_mask(mask);
}

void SceneEdge::build_bvh(const SecondaryEdgeInfoAD &edge_info,
                          bool allow_refit) {
    primitive_count_ = edge_info.size();
    node_count_ = 0;
    ready_ = false;
    refit_levels_.clear();
    active_primitive_count_ = 0;
    full_refit_node_count_ = 0;
    refit_enabled_ = allow_refit;

    if (primitive_count_ == 0) {
        edge_p0_ = Vector3f();
        edge_e1_ = Vector3f();
        primitive_bbox_min_ = Vector3f();
        primitive_bbox_max_ = Vector3f();
        node_bbox_min_ = Vector3f();
        node_bbox_max_ = Vector3f();
        left_child_ = Int();
        right_child_ = Int();
        leaf_primitives_ = Int();
        primitive_leaf_node_ = Int();
        leaf_nodes_ = Int();
        primitive_active_flags_ = Int();
        node_active_count_ = Int();
        node_subtree_primitive_count_ = Int();
        node_parent_ = Int();
        dirty_node_marks_ = Int();
        dirty_level_nodes_ = Int();
        dirty_level_count_ = Int();
        ready_ = true;
        return;
    }

    edge_p0_ = detach<false>(edge_info.start);
    edge_e1_ = detach<false>(edge_info.edge);

    const int build_node_count = std::max(2 * primitive_count_ - 1, 1);
    node_count_ = build_node_count;
    primitive_bbox_min_ = empty_vector3(primitive_count_);
    primitive_bbox_max_ = empty_vector3(primitive_count_);
    node_bbox_min_ = empty_vector3(node_count_);
    node_bbox_max_ = empty_vector3(node_count_);
    left_child_ = empty<Int>(node_count_);
    right_child_ = empty<Int>(node_count_);
    leaf_primitives_ = Int();
    primitive_leaf_node_ = empty<Int>(primitive_count_);
    leaf_nodes_ = Int();
    primitive_active_flags_ = Int();
    node_active_count_ = Int();
    node_subtree_primitive_count_ = Int();
    node_parent_ = Int();
    dirty_node_marks_ = Int();
    dirty_level_nodes_ = Int();
    dirty_level_count_ = Int();
    Int build_leaf_primitive = empty<Int>(build_node_count);
    Int build_is_leaf = empty<Int>(build_node_count);
    std::vector<int> left_child;
    std::vector<int> right_child;
    std::vector<int> is_leaf;
    std::vector<int> leaf_primitive;
    std::vector<int> optimized_left_child;
    std::vector<int> optimized_right_child;
    std::vector<int> optimized_is_leaf;
    std::vector<int> optimized_leaf_primitive;
    std::vector<ScalarVector3f> node_bbox_min;
    std::vector<ScalarVector3f> node_bbox_max;

    // The native builder uses independent non-blocking CUDA streams. Finish the
    // Dr.Jit input producers before exposing their pointers. Its outputs use
    // uninitialized storage because the native build fully writes every element.
    drjit::eval(edge_p0_, edge_e1_);
    drjit::sync_thread();

    build_edge_bvh_gpu(
        primitive_count_,
        edge_p0_[0].data(),
        edge_p0_[1].data(),
        edge_p0_[2].data(),
        edge_e1_[0].data(),
        edge_e1_[1].data(),
        edge_e1_[2].data(),
        primitive_bbox_min_[0].data(),
        primitive_bbox_min_[1].data(),
        primitive_bbox_min_[2].data(),
        primitive_bbox_max_[0].data(),
        primitive_bbox_max_[1].data(),
        primitive_bbox_max_[2].data(),
        node_bbox_min_[0].data(),
        node_bbox_min_[1].data(),
        node_bbox_min_[2].data(),
        node_bbox_max_[0].data(),
        node_bbox_max_[1].data(),
        node_bbox_max_[2].data(),
        left_child_.data(),
        right_child_.data(),
        build_leaf_primitive.data(),
        build_is_leaf.data(),
        primitive_leaf_node_.data());

    left_child = copy_ints_to_host(left_child_);
    right_child = copy_ints_to_host(right_child_);
    is_leaf = copy_ints_to_host(build_is_leaf);
    leaf_primitive = copy_ints_to_host(build_leaf_primitive);
    optimized_left_child = left_child;
    optimized_right_child = right_child;
    optimized_is_leaf = is_leaf;
    optimized_leaf_primitive = leaf_primitive;
    if (node_bbox_min.empty()) {
        node_bbox_min = copy_vector3_to_host(node_bbox_min_);
        node_bbox_max = copy_vector3_to_host(node_bbox_max_);
    }

    std::vector<int> final_left_child;
    std::vector<int> final_right_child;
    std::vector<int> final_is_leaf;

    std::vector<int> final_subtree_leaf_counts(static_cast<size_t>(build_node_count), -1);
    compute_subtree_leaf_count(
        0, optimized_left_child, optimized_right_child, optimized_is_leaf, final_subtree_leaf_counts);
    CompactedEdgeBVH compacted;
    compacted.primitive_leaf_nodes.assign(static_cast<size_t>(primitive_count_), -1);
    emit_compacted_bvh_preorder(0,
                                optimized_left_child,
                                optimized_right_child,
                                optimized_leaf_primitive,
                                optimized_is_leaf,
                                final_subtree_leaf_counts,
                                node_bbox_min,
                                node_bbox_max,
                                compacted);

    require(compacted.leaf_primitives.size() == static_cast<size_t>(primitive_count_),
            "SceneEdge::build(): compacted BVH lost edge primitives.");

    node_count_ = static_cast<int>(compacted.left_child.size());
    node_bbox_min_ = load_vector3(compacted.node_bbox_min);
    node_bbox_max_ = load_vector3(compacted.node_bbox_max);
    left_child_ = load_ints(compacted.left_child);
    right_child_ = load_ints(compacted.right_child);
    leaf_primitives_ = load_ints(compacted.leaf_primitives);
    primitive_leaf_node_ = allow_refit
        ? load_ints(compacted.primitive_leaf_nodes)
        : Int();
    final_left_child = compacted.left_child;
    final_right_child = compacted.right_child;
    final_is_leaf = compacted.is_leaf;

    std::vector<int> heights(static_cast<size_t>(node_count_), -1);
    const int max_height = compute_node_height(
        0, final_left_child, final_right_child, final_is_leaf, heights);

    require(max_height + 1 <= static_cast<int>(EdgeBVHTraversalStackSize),
            "SceneEdge::build(): BVH depth exceeds traversal stack capacity.");

    std::vector<int> node_parent(static_cast<size_t>(node_count_), -1);
    for (int node_index = 0; node_index < node_count_; ++node_index) {
        if (final_is_leaf[static_cast<size_t>(node_index)] != 0) {
            continue;
        }

        const int left = final_left_child[static_cast<size_t>(node_index)];
        const int right = final_right_child[static_cast<size_t>(node_index)];
        if (left >= 0) {
            node_parent[static_cast<size_t>(left)] = node_index;
        }
        if (right >= 0) {
            node_parent[static_cast<size_t>(right)] = node_index;
        }
    }

    std::vector<std::vector<int>> refit_levels(static_cast<size_t>(max_height + 1));
    for (int node_index = 0; node_index < node_count_; ++node_index) {
        const int height = heights[static_cast<size_t>(node_index)];
        if (height <= 0 || final_is_leaf[static_cast<size_t>(node_index)] != 0) {
            continue;
        }
        refit_levels[static_cast<size_t>(height)].push_back(node_index);
    }
    for (int height = 1; height <= max_height; ++height) {
        if (!refit_levels[static_cast<size_t>(height)].empty()) {
            refit_levels_.push_back(
                load<Int>(refit_levels[static_cast<size_t>(height)].data(),
                                  refit_levels[static_cast<size_t>(height)].size()));
        }
    }
    full_refit_node_count_ = 0;
    for (const Int &level : refit_levels_) {
        full_refit_node_count_ += static_cast<int>(level.size());
    }
    std::vector<int> leaf_nodes;
    leaf_nodes.reserve(static_cast<size_t>(primitive_count_));
    for (int node_index = 0; node_index < node_count_; ++node_index) {
        if (final_is_leaf[static_cast<size_t>(node_index)] != 0) {
            leaf_nodes.push_back(node_index);
        }
    }
    std::vector<int> subtree_primitive_counts(static_cast<size_t>(node_count_), -1);
    compute_subtree_primitive_count(0,
                                    final_left_child,
                                    final_right_child,
                                    final_is_leaf,
                                    subtree_primitive_counts);
    leaf_nodes_ = load_ints(leaf_nodes);
    primitive_active_flags_ = full<Int>(1, primitive_count_);
    node_active_count_ = load_ints(subtree_primitive_counts);
    node_subtree_primitive_count_ = load_ints(subtree_primitive_counts);
    node_parent_ = allow_refit ? load_ints(node_parent) : Int();
    dirty_node_marks_ = Int();
    dirty_level_nodes_ = Int();
    dirty_level_count_ = Int();

    if (!allow_refit) {
        primitive_bbox_min_ = Vector3f();
        primitive_bbox_max_ = Vector3f();
        if (primitive_count_ > EdgeBVHTreeletMaxPrimitives) {
            drjit::eval(edge_p0_,
                        edge_e1_,
                        node_bbox_min_,
                        node_bbox_max_,
                        left_child_,
                        right_child_,
                        leaf_primitives_,
                        leaf_nodes_,
                        primitive_active_flags_,
                        node_active_count_,
                        node_subtree_primitive_count_);
            drjit::sync_thread();
            jit_flush_malloc_cache();
        }
    }

    ready_ = true;
}

void SceneEdge::update_active_counts_from_mask(const Mask &mask) {
    require(ready_, "SceneEdge::update_active_counts_from_mask(): BVH is not built.");
    require(static_cast<int>(mask.size()) == primitive_count_,
            "SceneEdge::update_active_counts_from_mask(): mask size must match the BVH edge count.");

    if (primitive_count_ == 0 || node_count_ == 0) {
        all_active_ = true;
        active_primitive_count_ = 0;
        primitive_active_flags_ = Int();
        node_active_count_ = Int();
        return;
    }

    primitive_active_flags_ =
        select(mask, full<Int>(1, primitive_count_), zeros<Int>(primitive_count_));
    node_active_count_ = zeros<Int>(node_count_);

    const int leaf_node_count = static_cast<int>(leaf_nodes_.size());
    if (leaf_node_count > 0) {
        const Int encoded_leaf_begin = gather<Int>(left_child_, leaf_nodes_);
        const Int leaf_begin = node_leaf_begin(encoded_leaf_begin);
        const Int leaf_size = gather<Int>(right_child_, leaf_nodes_);
        Int leaf_active_count = zeros<Int>(leaf_node_count);
        for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
            const Mask slot_valid = leaf_size > slot;
            const Int primitive_offset =
                leaf_begin + full<Int>(slot, leaf_node_count);
            const Int primitive_index =
                gather<Int>(leaf_primitives_, primitive_offset, slot_valid);
            const Int slot_active =
                gather<Int>(primitive_active_flags_, primitive_index, slot_valid);
            leaf_active_count += select(slot_valid,
                                        slot_active,
                                        zeros<Int>(leaf_node_count));
        }
        scatter(node_active_count_, leaf_active_count, leaf_nodes_);
    }

    for (const Int &level : refit_levels_) {
        const Int left = gather<Int>(left_child_, level);
        const Int right = gather<Int>(right_child_, level);
        const Int left_count = gather<Int>(node_active_count_, left);
        const Int right_count = gather<Int>(node_active_count_, right);
        scatter(node_active_count_, left_count + right_count, level);
    }

    const std::vector<int> root_active_count =
        copy_ints_to_host(gather<Int>(node_active_count_, zeros<Int>(1)));
    active_primitive_count_ = root_active_count.empty() ? 0 : root_active_count.front();
    all_active_ = active_primitive_count_ == primitive_count_;
    if (all_active_) {
        set_all_active_state();
    }
}

void SceneEdge::materialize() const {
    require(ready_, "SceneEdge::materialize(): BVH is not built.");
    drjit::eval(edge_p0_,
                edge_e1_,
                primitive_bbox_min_,
                primitive_bbox_max_,
                node_bbox_min_,
                node_bbox_max_,
                left_child_,
                right_child_,
                leaf_primitives_,
                primitive_leaf_node_,
                leaf_nodes_,
                primitive_active_flags_,
                node_active_count_,
                node_subtree_primitive_count_,
                node_parent_,
                dirty_node_marks_,
                dirty_level_nodes_,
                dirty_level_count_);
}

SceneEdgeBVHStats SceneEdge::stats() const {
    require(ready_, "SceneEdge::stats(): BVH is not built.");

    SceneEdgeBVHStats result;
    result.primitive_count = primitive_count_;
    result.refit_level_count = static_cast<int>(refit_levels_.size());
    result.leaf_size_histogram.assign(static_cast<size_t>(EdgeBVHLeafSize + 1), 0);

    if (node_count_ <= 0) {
        return result;
    }

    const std::vector<int> left_child = copy_ints_to_host(left_child_);
    const std::vector<int> right_child = copy_ints_to_host(right_child_);
    const std::vector<ScalarVector3f> node_bbox_min = copy_vector3_to_host(node_bbox_min_);
    const std::vector<ScalarVector3f> node_bbox_max = copy_vector3_to_host(node_bbox_max_);

    std::vector<int> is_leaf(static_cast<size_t>(node_count_), 0);
    std::vector<int> heights(static_cast<size_t>(node_count_), -1);
    for (int node_index = 0; node_index < node_count_; ++node_index) {
        is_leaf[static_cast<size_t>(node_index)] =
            left_child[static_cast<size_t>(node_index)] < 0 &&
            right_child[static_cast<size_t>(node_index)] > 0 ? 1 : 0;
    }

    result.max_height = compute_node_height(0, left_child, right_child, is_leaf, heights);
    result.root_surface_area = static_cast<double>(bbox_surface_area(node_bbox_min[0], node_bbox_max[0]));

    int leaf_primitive_sum = 0;
    for (int node_index = 0; node_index < node_count_; ++node_index) {
        if (heights[static_cast<size_t>(node_index)] < 0) {
            continue;
        }

        ++result.node_count;
        if (is_leaf[static_cast<size_t>(node_index)] > 0) {
            ++result.leaf_node_count;
            const int leaf_size = right_child[static_cast<size_t>(node_index)];
            if (leaf_size > 0) {
                leaf_primitive_sum += leaf_size;
                result.min_leaf_size =
                    result.leaf_node_count == 1 ? leaf_size : std::min(result.min_leaf_size, leaf_size);
                result.max_leaf_size = std::max(result.max_leaf_size, leaf_size);
                if (leaf_size >= static_cast<int>(result.leaf_size_histogram.size())) {
                    result.leaf_size_histogram.resize(static_cast<size_t>(leaf_size + 1), 0);
                }
                result.leaf_size_histogram[static_cast<size_t>(leaf_size)] += 1;
            }
            continue;
        }

        ++result.internal_node_count;
        const int left_index = left_child[static_cast<size_t>(node_index)];
        const int right_index = right_child[static_cast<size_t>(node_index)];
        result.internal_surface_area_sum += static_cast<double>(
            bbox_surface_area(node_bbox_min[static_cast<size_t>(node_index)],
                              node_bbox_max[static_cast<size_t>(node_index)]));
        result.sibling_overlap_surface_area_sum += static_cast<double>(
            bbox_overlap_surface_area(node_bbox_min[static_cast<size_t>(left_index)],
                                      node_bbox_max[static_cast<size_t>(left_index)],
                                      node_bbox_min[static_cast<size_t>(right_index)],
                                      node_bbox_max[static_cast<size_t>(right_index)]));
    }

    if (result.leaf_node_count > 0) {
        result.avg_leaf_size =
            static_cast<double>(leaf_primitive_sum) / static_cast<double>(result.leaf_node_count);
    }
    if (result.internal_node_count > 0) {
        result.sibling_overlap_surface_area_avg =
            result.sibling_overlap_surface_area_sum / static_cast<double>(result.internal_node_count);
    }
    if (result.internal_surface_area_sum > 0.0) {
        result.normalized_sibling_overlap =
            result.sibling_overlap_surface_area_sum / result.internal_surface_area_sum;
    }
    return result;
}

void SceneEdge::set_all_active_state() {
    primitive_active_flags_ = primitive_count_ > 0 ? full<Int>(1, primitive_count_) : Int();
    node_active_count_ = node_subtree_primitive_count_;
    all_active_ = true;
    active_primitive_count_ = primitive_count_;
}

Int SceneEdge::refit_leaf_nodes_from_primitive_indices(const SecondaryEdgeInfoAD &edge_info,
                                                               const Int &primitive_indices) {
    const int dirty_primitive_count = static_cast<int>(primitive_indices.size());
    if (primitive_count_ == 0 || dirty_primitive_count == 0) {
        return Int();
    }

    const Vector3f scene_p0 = detach<false>(edge_info.start);
    const Vector3f scene_e1 = detach<false>(edge_info.edge);
    const Vector3f edge_p0 = gather<Vector3f>(scene_p0, primitive_indices);
    const Vector3f edge_e1 = gather<Vector3f>(scene_e1, primitive_indices);
    const Vector3f edge_p1 = edge_p0 + edge_e1;
    const Vector3f bbox_min = minimum(edge_p0, edge_p1);
    const Vector3f bbox_max = maximum(edge_p0, edge_p1);
    const Int leaf_nodes = gather<Int>(primitive_leaf_node_, primitive_indices);

    scatter(edge_p0_, edge_p0, primitive_indices);
    scatter(edge_e1_, edge_e1, primitive_indices);
    scatter(primitive_bbox_min_, bbox_min, primitive_indices);
    scatter(primitive_bbox_max_, bbox_max, primitive_indices);

    const Int encoded_leaf_begin = gather<Int>(left_child_, leaf_nodes);
    const Int leaf_begin = node_leaf_begin(encoded_leaf_begin);
    const Int leaf_count = gather<Int>(right_child_, leaf_nodes);
    Vector3f leaf_bbox_min = zero_vector3(dirty_primitive_count);
    Vector3f leaf_bbox_max = zero_vector3(dirty_primitive_count);
    Mask initialized = zeros<Mask>(dirty_primitive_count);
    for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
        const Mask lane_active = leaf_count > slot;
        const Int slot_offset =
            leaf_begin + full<Int>(slot, dirty_primitive_count);
        const Int leaf_primitive =
            gather<Int>(leaf_primitives_, slot_offset, lane_active);
        const Vector3f slot_bbox_min =
            gather<Vector3f>(primitive_bbox_min_, leaf_primitive, lane_active);
        const Vector3f slot_bbox_max =
            gather<Vector3f>(primitive_bbox_max_, leaf_primitive, lane_active);

        leaf_bbox_min = select(lane_active && !initialized, slot_bbox_min, leaf_bbox_min);
        leaf_bbox_max = select(lane_active && !initialized, slot_bbox_max, leaf_bbox_max);
        leaf_bbox_min =
            select(lane_active && initialized, minimum(leaf_bbox_min, slot_bbox_min), leaf_bbox_min);
        leaf_bbox_max =
            select(lane_active && initialized, maximum(leaf_bbox_max, slot_bbox_max), leaf_bbox_max);
        initialized |= lane_active;
    }

    scatter_node_bounds(leaf_nodes, leaf_bbox_min, leaf_bbox_max);
    return leaf_nodes;
}

void SceneEdge::refit_internal_nodes_full() {
    for (const Int &level : refit_levels_) {
        const Int left = gather<Int>(left_child_, level);
        const Int right = gather<Int>(right_child_, level);
        const Vector3f left_bbox_min = gather<Vector3f>(node_bbox_min_, left);
        const Vector3f left_bbox_max = gather<Vector3f>(node_bbox_max_, left);
        const Vector3f right_bbox_min = gather<Vector3f>(node_bbox_min_, right);
        const Vector3f right_bbox_max = gather<Vector3f>(node_bbox_max_, right);
        scatter_node_bounds(level,
                            minimum(left_bbox_min, right_bbox_min),
                            maximum(left_bbox_max, right_bbox_max));
    }
}

void SceneEdge::refit_internal_nodes_dirty(const std::vector<Int> &dirty_leaf_chunks) {
    if (refit_levels_.empty()) {
        return;
    }

    if (dirty_leaf_chunks.empty()) {
        return;
    }

    if (node_parent_.size() != static_cast<size_t>(node_count_) ||
        full_refit_node_count_ <= 0) {
        refit_internal_nodes_full();
        return;
    }

    if (dirty_node_marks_.size() != static_cast<size_t>(node_count_) ||
        dirty_level_nodes_.size() != static_cast<size_t>(node_count_) ||
        dirty_level_count_.size() != 1) {
        // Native dirty-refit launchers clear/fill these buffers before reading
        // them, so static scenes need not retain the scratch allocation.
        dirty_node_marks_ = empty<Int>(node_count_);
        dirty_level_nodes_ = empty<Int>(node_count_);
        dirty_level_count_ = empty<Int>(1);
    }

    bool cleared_marks = false;
    for (const Int &leaf_nodes : dirty_leaf_chunks) {
        const int leaf_count = static_cast<int>(leaf_nodes.size());
        if (leaf_count == 0) {
            continue;
        }

        mark_edge_bvh_dirty_ancestors_gpu(node_count_,
                                          leaf_count,
                                          leaf_nodes.data(),
                                          node_parent_.data(),
                                          dirty_node_marks_.data(),
                                          !cleared_marks);
        cleared_marks = true;
    }

    if (!cleared_marks) {
        return;
    }

    for (const Int &level : refit_levels_) {
        const int level_count = static_cast<int>(level.size());
        if (level_count <= 0) {
            continue;
        }

        compact_and_refit_edge_bvh_level_gpu(
            level_count,
            level.data(),
            dirty_node_marks_.data(),
            dirty_level_nodes_.data(),
            dirty_level_count_.data(),
            left_child_.data(),
            right_child_.data(),
            node_bbox_min_.x().data(),
            node_bbox_min_.y().data(),
            node_bbox_min_.z().data(),
            node_bbox_max_.x().data(),
            node_bbox_max_.y().data(),
            node_bbox_max_.z().data());
    }
}

void SceneEdge::refit(const SecondaryEdgeInfoAD &edge_info,
                      const std::vector<EdgeDirtyRange> &dirty_ranges) {
    require(ready_, "SceneEdge::refit(): BVH is not built.");
    require(refit_enabled_, "SceneEdge::refit(): BVH was built without dynamic-refit state.");
    if (primitive_count_ == 0 || dirty_ranges.empty()) {
        return;
    }

    std::vector<Int> dirty_leaf_chunks;
    dirty_leaf_chunks.reserve(dirty_ranges.size());
    size_t dirty_primitive_count = 0;
    for (const EdgeDirtyRange &range : dirty_ranges) {
        if (range.count <= 0) {
            continue;
        }

        dirty_primitive_count += static_cast<size_t>(range.count);
        const Int primitive_indices = arange<Int>(range.count) + range.offset;
        const Int leaf_nodes =
            refit_leaf_nodes_from_primitive_indices(edge_info, primitive_indices);
        if (leaf_nodes.size() > 0) {
            dirty_leaf_chunks.push_back(leaf_nodes);
        }
    }

    const EdgeBVHRefitStrategy refit_strategy = active_edge_bvh_refit_strategy();
    const bool use_dirty_ancestors =
        should_use_dirty_ancestor_refit(refit_strategy,
                                        static_cast<size_t>(primitive_count_),
                                        dirty_primitive_count);
    if (use_dirty_ancestors) {
        refit_internal_nodes_dirty(dirty_leaf_chunks);
    } else {
        refit_internal_nodes_full();
    }

}

void SceneEdge::refit(const SecondaryEdgeInfoAD &edge_info,
                          const Int &primitive_indices) {
    require(ready_, "SceneEdge::refit(): BVH is not built.");
    require(refit_enabled_, "SceneEdge::refit(): BVH was built without dynamic-refit state.");

    if (primitive_count_ == 0 || primitive_indices.size() == 0) {
        return;
    }

    const Int leaf_nodes =
        refit_leaf_nodes_from_primitive_indices(edge_info, primitive_indices);
    std::vector<Int> dirty_leaf_chunks;
    if (leaf_nodes.size() > 0) {
        dirty_leaf_chunks.push_back(leaf_nodes);
    }

    const EdgeBVHRefitStrategy refit_strategy = active_edge_bvh_refit_strategy();
    const size_t dirty_primitive_count = static_cast<size_t>(primitive_indices.size());
    const bool use_dirty_ancestors =
        should_use_dirty_ancestor_refit(refit_strategy,
                                        static_cast<size_t>(primitive_count_),
                                        dirty_primitive_count);
    if (use_dirty_ancestors) {
        refit_internal_nodes_dirty(dirty_leaf_chunks);
    } else {
        refit_internal_nodes_full();
    }

}

Int SceneEdge::map_to_global(const Int &bvh_ids,
                                     const Mask &valid) const {
    const int query_count = static_cast<int>(bvh_ids.size());
    if (query_count == 0) {
        return Int();
    }

    Int result = full<Int>(-1, query_count);
    return select(valid, bvh_ids, result);
}

ClosestEdgeCandidate SceneEdge::nearest_edge_point_detached(const Vector3f &point,
                                                                const Mask &active) const {
    const int query_count = static_cast<int>(slices(point));

    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0 || active_primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    const Int stack_base =
        arange<Int>(query_count) * static_cast<int>(EdgeBVHTraversalStackSize);

    auto [current_node,
          stack_size,
          stack,
          best_distance_sq,
          best_primitive] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<Int>(query_count), full<Int>(-1, query_count)),
                          zeros<Int>(query_count),
                          make_empty_stack(query_count),
                          full<Float>(Infinity, query_count),
                          full<Int>(-1, query_count)),
        [](const Int &current_node,
           const Int &stack_size,
           const TraversalStack &,
           const Float &,
           const Int &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &point, &stack_base, query_count](Int &current_node,
                                                 Int &stack_size,
                                                 TraversalStack &stack,
                                                 Float &best_distance_sq,
                                                 Int &best_primitive) {
            const Mask need_pop = (current_node < 0) && (stack_size > 0);
            const Int popped_node = stack_pop(stack, stack_base, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const Mask lane_active = current_node >= 0;
            const Mask node_active =
                all_active_ ? lane_active :
                              lane_active &&
                                  (gather_node_active_count(current_node, lane_active) > 0);
            const Vector3f bbox_min = gather_node_bbox_min(current_node, lane_active);
            const Vector3f bbox_max = gather_node_bbox_max(current_node, lane_active);
            const Float node_bound = point_aabb_distance_sq(point, bbox_min, bbox_max);
            const Mask visit = node_active && (node_bound <= best_distance_sq);

            const Int encoded_left = gather_node_left_child(current_node, lane_active);
            const Mask leaf_node = lane_active && node_is_leaf(encoded_left);
            const Mask leaf_visit = visit && leaf_node;
            const Int leaf_begin = node_leaf_begin(encoded_left);
            const Int leaf_count = gather_node_right_child(current_node, lane_active);
            for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
                const Mask slot_lane = leaf_visit && (leaf_count > slot);
                const Int primitive_offset = leaf_begin + full<Int>(slot, query_count);
                const Int primitive_index =
                    gather<Int>(leaf_primitives_, primitive_offset, slot_lane);
                const Mask slot_visit =
                    all_active_ ? slot_lane :
                                  slot_lane &&
                                      (gather<Int>(primitive_active_flags_,
                                                           primitive_index,
                                                           slot_lane) > 0);
                const Vector3f edge_p0 = gather<Vector3f>(edge_p0_, primitive_index, slot_visit);
                const Vector3f edge_e1 = gather<Vector3f>(edge_e1_, primitive_index, slot_visit);

                Float edge_t;
                Vector3f edge_point;
                Float candidate_distance_sq;
                std::tie(edge_t, edge_point, candidate_distance_sq) =
                    closest_point_on_segment<true>(point, edge_p0, edge_e1);
                DRJIT_MARK_USED(edge_t);
                DRJIT_MARK_USED(edge_point);

                const Mask better = slot_visit && (candidate_distance_sq < best_distance_sq);
                best_distance_sq = select(better, candidate_distance_sq, best_distance_sq);
                best_primitive = select(better, primitive_index, best_primitive);
            }

            const Mask internal_visit = visit && !leaf_node;
            const Int left = select(internal_visit, encoded_left, full<Int>(-1, query_count));
            const Int right = gather_node_right_child(current_node, internal_visit);

            const Vector3f left_bbox_min = gather_node_bbox_min(left, internal_visit);
            const Vector3f left_bbox_max = gather_node_bbox_max(left, internal_visit);
            const Vector3f right_bbox_min = gather_node_bbox_min(right, internal_visit);
            const Vector3f right_bbox_max = gather_node_bbox_max(right, internal_visit);
            const Float left_bound = point_aabb_distance_sq(point, left_bbox_min, left_bbox_max);
            const Float right_bound = point_aabb_distance_sq(point, right_bbox_min, right_bbox_max);

            const Mask left_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(left, internal_visit) > 0);
            const Mask right_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(right, internal_visit) > 0);
            const Mask left_visit = left_nonempty && (left_bound <= best_distance_sq);
            const Mask right_visit = right_nonempty && (right_bound <= best_distance_sq);
            const Mask both_children = left_visit && right_visit;
            const Mask only_left = left_visit && !right_visit;
            const Mask only_right = right_visit && !left_visit;
            const Mask left_first = left_bound <= right_bound;

            const Int near_child = select(left_first, left, right);
            const Int far_child = select(left_first, right, left);
            stack_push(stack, stack_base, stack_size, far_child, both_children);

            Int next_node = full<Int>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "nearest_edge_point_bvh");

    result.global_edge_id = best_primitive;
    result.distance_sq = best_distance_sq;
    return result;
}

ClosestEdgeTopKCandidate SceneEdge::nearest_edges_point_detached(
    const Vector3f &point,
    int k,
    const Mask &active) const {
    const int query_count = static_cast<int>(slices(point));
    const int output_count = query_count * k;

    ClosestEdgeTopKCandidate result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<Mask>(false, output_count);
    result.global_edge_ids = full<Int>(-1, output_count);
    result.distance_sq = full<Float>(Infinity, output_count);
    if (primitive_count_ == 0 || active_primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    const Int query_index = arange<Int>(query_count);
    const Int stack_base =
        query_index * static_cast<int>(EdgeBVHTraversalStackSize);

    auto [current_node,
          stack_size,
          stack,
          topk] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<Int>(query_count), full<Int>(-1, query_count)),
                          zeros<Int>(query_count),
                          make_empty_stack(query_count),
                          make_empty_topk_state(query_count)),
        [](const Int &current_node,
           const Int &stack_size,
           const TraversalStack &,
           const TopKTraversalState &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &point, &stack_base, query_count, k](Int &current_node,
                                                    Int &stack_size,
                                                    TraversalStack &stack,
                                                    TopKTraversalState &topk) {
            const Mask need_pop = (current_node < 0) && (stack_size > 0);
            const Int popped_node = stack_pop(stack, stack_base, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const Mask lane_active = current_node >= 0;
            const Mask node_active =
                all_active_ ? lane_active :
                              lane_active &&
                                  (gather_node_active_count(current_node, lane_active) > 0);
            auto distance_slots = topk_distance_slots(topk);
            const Float worst_distance_sq =
                *distance_slots[static_cast<size_t>(k - 1)];
            const Vector3f bbox_min = gather_node_bbox_min(current_node, lane_active);
            const Vector3f bbox_max = gather_node_bbox_max(current_node, lane_active);
            const Float node_bound = point_aabb_distance_sq(point, bbox_min, bbox_max);
            const Mask visit = node_active && (node_bound <= worst_distance_sq);

            const Int encoded_left = gather_node_left_child(current_node, lane_active);
            const Mask leaf_node = lane_active && node_is_leaf(encoded_left);
            const Mask leaf_visit = visit && leaf_node;
            const Int leaf_begin = node_leaf_begin(encoded_left);
            const Int leaf_count = gather_node_right_child(current_node, lane_active);
            for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
                const Mask slot_lane = leaf_visit && (leaf_count > slot);
                const Int primitive_offset = leaf_begin + full<Int>(slot, query_count);
                const Int primitive_index =
                    gather<Int>(leaf_primitives_, primitive_offset, slot_lane);
                const Mask slot_visit =
                    all_active_ ? slot_lane :
                                  slot_lane &&
                                      (gather<Int>(primitive_active_flags_,
                                                           primitive_index,
                                                           slot_lane) > 0);
                const Vector3f edge_p0 = gather<Vector3f>(edge_p0_, primitive_index, slot_visit);
                const Vector3f edge_e1 = gather<Vector3f>(edge_e1_, primitive_index, slot_visit);

                Float edge_t;
                Vector3f edge_point;
                Float candidate_distance_sq;
                std::tie(edge_t, edge_point, candidate_distance_sq) =
                    closest_point_on_segment<true>(point, edge_p0, edge_e1);
                DRJIT_MARK_USED(edge_t);
                DRJIT_MARK_USED(edge_point);

                topk_insert_candidate(topk,
                                      k,
                                      primitive_index,
                                      candidate_distance_sq,
                                      slot_visit);
            }

            const Mask internal_visit = visit && !leaf_node;
            const Int left = select(internal_visit, encoded_left, full<Int>(-1, query_count));
            const Int right = gather_node_right_child(current_node, internal_visit);

            const Vector3f left_bbox_min = gather_node_bbox_min(left, internal_visit);
            const Vector3f left_bbox_max = gather_node_bbox_max(left, internal_visit);
            const Vector3f right_bbox_min = gather_node_bbox_min(right, internal_visit);
            const Vector3f right_bbox_max = gather_node_bbox_max(right, internal_visit);
            const Float left_bound = point_aabb_distance_sq(point, left_bbox_min, left_bbox_max);
            const Float right_bound = point_aabb_distance_sq(point, right_bbox_min, right_bbox_max);
            const Float updated_worst_distance_sq =
                *distance_slots[static_cast<size_t>(k - 1)];

            const Mask left_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(left, internal_visit) > 0);
            const Mask right_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(right, internal_visit) > 0);
            const Mask left_visit = left_nonempty && (left_bound <= updated_worst_distance_sq);
            const Mask right_visit = right_nonempty && (right_bound <= updated_worst_distance_sq);
            const Mask both_children = left_visit && right_visit;
            const Mask only_left = left_visit && !right_visit;
            const Mask only_right = right_visit && !left_visit;
            const Mask left_first = left_bound <= right_bound;

            const Int near_child = select(left_first, left, right);
            const Int far_child = select(left_first, right, left);
            stack_push(stack, stack_base, stack_size, far_child, both_children);

            Int next_node = full<Int>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "nearest_edges_point_bvh");

    const Int top_base = query_index * k;
    auto distance_slots = topk_distance_slots(topk);
    auto primitive_slots = topk_primitive_slots(topk);
    for (int rank = 0; rank < k; ++rank) {
        const Int output_slot =
            top_base + full<Int>(rank, query_count);
        const Int primitive = *primitive_slots[static_cast<size_t>(rank)];
        const Float distance_sq = *distance_slots[static_cast<size_t>(rank)];
        const Mask valid = primitive >= 0;
        scatter(result.global_edge_ids, primitive, output_slot, valid);
        scatter(result.distance_sq, distance_sq, output_slot, valid);
        scatter(result.is_valid, valid, output_slot, valid);
    }
    return result;
}

ClosestEdgeCandidate SceneEdge::nearest_edge_finite_ray_detached(const Vector3f &origin,
                                                                      const Vector3f &segment,
                                                                      const Mask &active) const {
    const int query_count = static_cast<int>(slices(origin));

    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0 || active_primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    const Int stack_base =
        arange<Int>(query_count) * static_cast<int>(EdgeBVHTraversalStackSize);

    auto [current_node,
          stack_size,
          stack,
          best_distance_sq,
          best_primitive] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<Int>(query_count), full<Int>(-1, query_count)),
                          zeros<Int>(query_count),
                          make_empty_stack(query_count),
                          full<Float>(Infinity, query_count),
                          full<Int>(-1, query_count)),
        [](const Int &current_node,
           const Int &stack_size,
           const TraversalStack &,
           const Float &,
           const Int &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &origin, &segment, &stack_base, query_count](Int &current_node,
                                                            Int &stack_size,
                                                            TraversalStack &stack,
                                                            Float &best_distance_sq,
                                                            Int &best_primitive) {
            const Mask need_pop = (current_node < 0) && (stack_size > 0);
            const Int popped_node = stack_pop(stack, stack_base, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const Mask lane_active = current_node >= 0;
            const Mask node_active =
                all_active_ ? lane_active :
                              lane_active &&
                                  (gather_node_active_count(current_node, lane_active) > 0);
            const Vector3f bbox_min = gather_node_bbox_min(current_node, lane_active);
            const Vector3f bbox_max = gather_node_bbox_max(current_node, lane_active);
            const Float node_bound = segment_aabb_lower_bound_sq(origin, segment, bbox_min, bbox_max);
            const Mask visit = node_active && (node_bound <= best_distance_sq);

            const Int encoded_left = gather_node_left_child(current_node, lane_active);
            const Mask leaf_node = lane_active && node_is_leaf(encoded_left);
            const Mask leaf_visit = visit && leaf_node;
            const Int leaf_begin = node_leaf_begin(encoded_left);
            const Int leaf_count = gather_node_right_child(current_node, lane_active);
            for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
                const Mask slot_lane = leaf_visit && (leaf_count > slot);
                const Int primitive_offset = leaf_begin + full<Int>(slot, query_count);
                const Int primitive_index =
                    gather<Int>(leaf_primitives_, primitive_offset, slot_lane);
                const Mask slot_visit =
                    all_active_ ? slot_lane :
                                  slot_lane &&
                                      (gather<Int>(primitive_active_flags_,
                                                           primitive_index,
                                                           slot_lane) > 0);
                const Vector3f edge_p0 = gather<Vector3f>(edge_p0_, primitive_index, slot_visit);
                const Vector3f edge_e1 = gather<Vector3f>(edge_e1_, primitive_index, slot_visit);

                Float query_t;
                Vector3f query_point;
                Float edge_t;
                Vector3f edge_point;
                Float candidate_distance_sq;
                std::tie(query_t, query_point, edge_t, edge_point, candidate_distance_sq) =
                    closest_segment_segment<true>(origin, segment, edge_p0, edge_e1);
                DRJIT_MARK_USED(query_t);
                DRJIT_MARK_USED(query_point);
                DRJIT_MARK_USED(edge_t);
                DRJIT_MARK_USED(edge_point);

                const Mask better = slot_visit && (candidate_distance_sq < best_distance_sq);
                best_distance_sq = select(better, candidate_distance_sq, best_distance_sq);
                best_primitive = select(better, primitive_index, best_primitive);
            }

            const Mask internal_visit = visit && !leaf_node;
            const Int left = select(internal_visit, encoded_left, full<Int>(-1, query_count));
            const Int right = gather_node_right_child(current_node, internal_visit);

            const Vector3f left_bbox_min = gather_node_bbox_min(left, internal_visit);
            const Vector3f left_bbox_max = gather_node_bbox_max(left, internal_visit);
            const Vector3f right_bbox_min = gather_node_bbox_min(right, internal_visit);
            const Vector3f right_bbox_max = gather_node_bbox_max(right, internal_visit);
            const Float left_bound = segment_aabb_lower_bound_sq(origin, segment, left_bbox_min, left_bbox_max);
            const Float right_bound = segment_aabb_lower_bound_sq(origin, segment, right_bbox_min, right_bbox_max);

            const Mask left_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(left, internal_visit) > 0);
            const Mask right_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(right, internal_visit) > 0);
            const Mask left_visit = left_nonempty && (left_bound <= best_distance_sq);
            const Mask right_visit = right_nonempty && (right_bound <= best_distance_sq);
            const Mask both_children = left_visit && right_visit;
            const Mask only_left = left_visit && !right_visit;
            const Mask only_right = right_visit && !left_visit;
            const Mask left_first = left_bound <= right_bound;

            const Int near_child = select(left_first, left, right);
            const Int far_child = select(left_first, right, left);
            stack_push(stack, stack_base, stack_size, far_child, both_children);

            Int next_node = full<Int>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "nearest_edge_finite_ray_bvh");

    result.global_edge_id = best_primitive;
    result.distance_sq = best_distance_sq;
    return result;
}

ClosestEdgeCandidate SceneEdge::nearest_edge_infinite_ray_detached(const Vector3f &origin,
                                                                       const Vector3f &direction,
                                                                       const Mask &active) const {
    const int query_count = static_cast<int>(slices(origin));

    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0 || active_primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    const Int stack_base =
        arange<Int>(query_count) * static_cast<int>(EdgeBVHTraversalStackSize);

    auto [current_node,
          stack_size,
          stack,
          best_distance_sq,
          best_primitive] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<Int>(query_count), full<Int>(-1, query_count)),
                          zeros<Int>(query_count),
                          make_empty_stack(query_count),
                          full<Float>(Infinity, query_count),
                          full<Int>(-1, query_count)),
        [](const Int &current_node,
           const Int &stack_size,
           const TraversalStack &,
           const Float &,
           const Int &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &origin, &direction, &stack_base, query_count](Int &current_node,
                                                              Int &stack_size,
                                                              TraversalStack &stack,
                                                              Float &best_distance_sq,
                                                              Int &best_primitive) {
            const Mask need_pop = (current_node < 0) && (stack_size > 0);
            const Int popped_node = stack_pop(stack, stack_base, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const Mask lane_active = current_node >= 0;
            const Mask node_active =
                all_active_ ? lane_active :
                              lane_active &&
                                  (gather_node_active_count(current_node, lane_active) > 0);
            const Vector3f bbox_min = gather_node_bbox_min(current_node, lane_active);
            const Vector3f bbox_max = gather_node_bbox_max(current_node, lane_active);
            const Float node_bound = ray_aabb_lower_bound_sq(origin, direction, bbox_min, bbox_max);
            const Mask visit = node_active && (node_bound <= best_distance_sq);

            const Int encoded_left = gather_node_left_child(current_node, lane_active);
            const Mask leaf_node = lane_active && node_is_leaf(encoded_left);
            const Mask leaf_visit = visit && leaf_node;
            const Int leaf_begin = node_leaf_begin(encoded_left);
            const Int leaf_count = gather_node_right_child(current_node, lane_active);
            for (int slot = 0; slot < EdgeBVHLeafSize; ++slot) {
                const Mask slot_lane = leaf_visit && (leaf_count > slot);
                const Int primitive_offset = leaf_begin + full<Int>(slot, query_count);
                const Int primitive_index =
                    gather<Int>(leaf_primitives_, primitive_offset, slot_lane);
                const Mask slot_visit =
                    all_active_ ? slot_lane :
                                  slot_lane &&
                                      (gather<Int>(primitive_active_flags_,
                                                           primitive_index,
                                                           slot_lane) > 0);
                const Vector3f edge_p0 = gather<Vector3f>(edge_p0_, primitive_index, slot_visit);
                const Vector3f edge_e1 = gather<Vector3f>(edge_e1_, primitive_index, slot_visit);

                Float query_t;
                Vector3f query_point;
                Float edge_t;
                Vector3f edge_point;
                Float candidate_distance_sq;
                std::tie(query_t, query_point, edge_t, edge_point, candidate_distance_sq) =
                    closest_ray_segment<true>(origin, direction, edge_p0, edge_e1);
                DRJIT_MARK_USED(query_t);
                DRJIT_MARK_USED(query_point);
                DRJIT_MARK_USED(edge_t);
                DRJIT_MARK_USED(edge_point);

                const Mask better = slot_visit && (candidate_distance_sq < best_distance_sq);
                best_distance_sq = select(better, candidate_distance_sq, best_distance_sq);
                best_primitive = select(better, primitive_index, best_primitive);
            }

            const Mask internal_visit = visit && !leaf_node;
            const Int left = select(internal_visit, encoded_left, full<Int>(-1, query_count));
            const Int right = gather_node_right_child(current_node, internal_visit);

            const Vector3f left_bbox_min = gather_node_bbox_min(left, internal_visit);
            const Vector3f left_bbox_max = gather_node_bbox_max(left, internal_visit);
            const Vector3f right_bbox_min = gather_node_bbox_min(right, internal_visit);
            const Vector3f right_bbox_max = gather_node_bbox_max(right, internal_visit);
            const Float left_bound = ray_aabb_lower_bound_sq(origin, direction, left_bbox_min, left_bbox_max);
            const Float right_bound = ray_aabb_lower_bound_sq(origin, direction, right_bbox_min, right_bbox_max);

            const Mask left_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(left, internal_visit) > 0);
            const Mask right_nonempty =
                all_active_ ? internal_visit :
                              internal_visit &&
                                  (gather_node_active_count(right, internal_visit) > 0);
            const Mask left_visit = left_nonempty && (left_bound <= best_distance_sq);
            const Mask right_visit = right_nonempty && (right_bound <= best_distance_sq);
            const Mask both_children = left_visit && right_visit;
            const Mask only_left = left_visit && !right_visit;
            const Mask only_right = right_visit && !left_visit;
            const Mask left_first = left_bound <= right_bound;

            const Int near_child = select(left_first, left, right);
            const Int far_child = select(left_first, right, left);
            stack_push(stack, stack_base, stack_size, far_child, both_children);

            Int next_node = full<Int>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "nearest_edge_infinite_ray_bvh");

    result.global_edge_id = best_primitive;
    result.distance_sq = best_distance_sq;
    return result;
}

template <bool Detached>
ClosestEdgeCandidate SceneEdge::nearest_edge(const Vector3fT<Detached> &point,
                                                  MaskT<Detached> &active) const {
    require(ready_, "SceneEdge::nearest_edge(point): BVH is not built.");
    drjit::scoped_set_flag symbolic_loops_scope(JitFlag::SymbolicLoops, false);

    const int query_count = static_cast<int>(slices(point));
    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const Mask active_detached = detach<false>(active);
    result = nearest_edge_point_detached(detach<false>(point), active_detached);
    if constexpr (!Detached) {
        active &= MaskAD(result.global_edge_id >= 0);
    } else {
        active &= (result.global_edge_id >= 0);
    }
    return result;
}

template <bool Detached>
ClosestEdgeTopKCandidate SceneEdge::nearest_edges(const Vector3fT<Detached> &point,
                                                       int k,
                                                       MaskT<Detached> &active) const {
    require(ready_, "SceneEdge::nearest_edges(point): BVH is not built.");
    require(k > 0, "SceneEdge::nearest_edges(point): k must be positive.");
    require(k <= 16, "SceneEdge::nearest_edges(point): k must be <= 16.");
    drjit::scoped_set_flag symbolic_loops_scope(JitFlag::SymbolicLoops, false);

    const int query_count = static_cast<int>(slices(point));
    ClosestEdgeTopKCandidate result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<Mask>(false, query_count * k);
    result.global_edge_ids = full<Int>(-1, query_count * k);
    result.distance_sq = full<Float>(Infinity, query_count * k);
    if (primitive_count_ == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const Mask active_detached = detach<false>(active);
    result = nearest_edges_point_detached(detach<false>(point), k, active_detached);
    const Int first_slot = arange<Int>(query_count) * k;
    const Mask has_any =
        gather<Mask>(result.is_valid, first_slot, active_detached);
    if constexpr (!Detached) {
        active &= MaskAD(has_any);
    } else {
        active &= has_any;
    }
    return result;
}

template <bool Detached>
ClosestEdgeCandidate SceneEdge::nearest_edge(const RayT<Detached> &ray,
                                                  MaskT<Detached> &active) const {
    require(ready_, "SceneEdge::nearest_edge(ray): BVH is not built.");
    drjit::scoped_set_flag symbolic_loops_scope(JitFlag::SymbolicLoops, false);

    const int query_count = static_cast<int>(slices(ray.o));
    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const Mask active_detached = detach<false>(active);
    if (drjit::none(active_detached)) {
        return result;
    }

    const Vector3f origin = detach<false>(ray.o);
    const Vector3f direction = detach<false>(ray.d);
    const Float tmax = detach<false>(ray.tmax);
    const Mask finite_mask = active_detached && drjit::isfinite(tmax);
    const Mask infinite_mask = active_detached && !drjit::isfinite(tmax);

    if (drjit::any(finite_mask)) {
        const ClosestEdgeCandidate finite_result =
            nearest_edge_finite_ray_detached(origin, direction * tmax, finite_mask);
        result.global_edge_id = select(finite_mask, finite_result.global_edge_id, result.global_edge_id);
        result.distance_sq = select(finite_mask, finite_result.distance_sq, result.distance_sq);
    }

    if (drjit::any(infinite_mask)) {
        const ClosestEdgeCandidate infinite_result =
            nearest_edge_infinite_ray_detached(origin, direction, infinite_mask);
        result.global_edge_id = select(infinite_mask, infinite_result.global_edge_id, result.global_edge_id);
        result.distance_sq = select(infinite_mask, infinite_result.distance_sq, result.distance_sq);
    }

    if constexpr (!Detached) {
        active &= MaskAD(result.global_edge_id >= 0);
    } else {
        active &= (result.global_edge_id >= 0);
    }
    return result;
}

template ClosestEdgeCandidate SceneEdge::nearest_edge<true>(const Vector3f &point,
                                                                 Mask &active) const;
template ClosestEdgeCandidate SceneEdge::nearest_edge<false>(const Vector3fAD &point,
                                                                  MaskAD &active) const;
template ClosestEdgeTopKCandidate SceneEdge::nearest_edges<true>(
    const Vector3f &point,
    int k,
    Mask &active) const;
template ClosestEdgeTopKCandidate SceneEdge::nearest_edges<false>(
    const Vector3fAD &point,
    int k,
    MaskAD &active) const;
template ClosestEdgeCandidate SceneEdge::nearest_edge<true>(const Ray &ray,
                                                                 Mask &active) const;
template ClosestEdgeCandidate SceneEdge::nearest_edge<false>(const RayAD &ray,
                                                                  MaskAD &active) const;

} // namespace rayd

