#include <algorithm>
#include <array>
#include <limits>
#include <tuple>
#include <vector>

#include <drjit/while_loop.h>

#include "edge_bvh.h"

#include <rayd/scene/scene_edge.h>
#include <rayd/utils.h>

namespace rayd {

namespace {

constexpr size_t EdgeBVHTraversalStackSize = 32;
constexpr int EdgeBVHHybridClusterLeafCount = 32;
constexpr int EdgeBVHHybridClusterMaxHeight = 12;
constexpr int EdgeBVHHybridTopLevelMinPrimitives = 65536;
constexpr int EdgeBVHSAHBins = 12;
constexpr int EdgeBVHTreeletMaxLeaves = 7;
constexpr int EdgeBVHTreeletMinPrimitives = 65536;
constexpr int EdgeBVHTreeletMinSubtreeLeaves = 32;
constexpr float EdgeBVHTreeletCostInflationRatio = 1e-4f;
enum class EdgeBVHPostBuildStrategy {
    None,
    HybridTopLevelSAH,
    Treelet
};
constexpr EdgeBVHPostBuildStrategy EdgeBVHActivePostBuildStrategy =
    EdgeBVHPostBuildStrategy::Treelet;
using TraversalStack = Array<IntDetached, EdgeBVHTraversalStackSize>;

struct TopLevelBuildRecord {
    int node_index = -1;
    ScalarVector3f bbox_min;
    ScalarVector3f bbox_max;
    ScalarVector3f centroid;
};

struct SAHBin {
    ScalarVector3f bbox_min;
    ScalarVector3f bbox_max;
    int count = 0;
};

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

int dominant_axis(const ScalarVector3f &extent) {
    if (extent.x() >= extent.y() && extent.x() >= extent.z()) {
        return 0;
    }
    if (extent.y() >= extent.z()) {
        return 1;
    }
    return 2;
}

float axis_value(const ScalarVector3f &v, int axis) {
    if (axis == 0) {
        return v.x();
    }
    if (axis == 1) {
        return v.y();
    }
    return v.z();
}

Vector3fDetached zero_vector3(int size) {
    if (size <= 0) {
        return Vector3fDetached();
    }

    return Vector3fDetached(zeros<FloatDetached>(size),
                            zeros<FloatDetached>(size),
                            zeros<FloatDetached>(size));
}

IntDetached load_ints(const std::vector<int> &values) {
    if (values.empty()) {
        return IntDetached();
    }
    return load<IntDetached>(values.data(), values.size());
}

Vector3fDetached load_vector3(const std::vector<ScalarVector3f> &values) {
    const size_t count = values.size();
    if (count == 0) {
        return Vector3fDetached();
    }

    std::vector<float> x(count), y(count), z(count);
    for (size_t index = 0; index < count; ++index) {
        x[index] = values[index].x();
        y[index] = values[index].y();
        z[index] = values[index].z();
    }

    return Vector3fDetached(load<FloatDetached>(x.data(), count),
                            load<FloatDetached>(y.data(), count),
                            load<FloatDetached>(z.data(), count));
}

std::vector<int> copy_ints_to_host(const IntDetached &values) {
    const size_t count = values.size();
    if (count == 0) {
        return {};
    }

    std::vector<int> result(count);
    drjit::store(result.data(), values);
    return result;
}

std::vector<ScalarVector3f> copy_vector3_to_host(const Vector3fDetached &values) {
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
            "SceneEdge::configure(): GPU LBVH traversal did not cover every node.");
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

int compute_subtree_height(int node_index,
                           const std::vector<int> &left_child,
                           const std::vector<int> &right_child,
                           const std::vector<int> &is_leaf,
                           std::vector<int> &subtree_heights) {
    int &height = subtree_heights[static_cast<size_t>(node_index)];
    if (height >= 0) {
        return height;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        height = 0;
        return height;
    }

    height = 1 + std::max(compute_subtree_height(left_child[static_cast<size_t>(node_index)],
                                                 left_child,
                                                 right_child,
                                                 is_leaf,
                                                 subtree_heights),
                          compute_subtree_height(right_child[static_cast<size_t>(node_index)],
                                                 left_child,
                                                 right_child,
                                                 is_leaf,
                                                 subtree_heights));
    return height;
}

void select_lbvh_clusters(int node_index,
                          const std::vector<int> &left_child,
                          const std::vector<int> &right_child,
                          const std::vector<int> &is_leaf,
                          const std::vector<int> &subtree_leaf_counts,
                          const std::vector<int> &subtree_heights,
                          std::vector<uint8_t> &is_cluster_root,
                          std::vector<int> &cluster_roots) {
    if (is_leaf[static_cast<size_t>(node_index)] > 0 ||
        subtree_leaf_counts[static_cast<size_t>(node_index)] <= EdgeBVHHybridClusterLeafCount ||
        subtree_heights[static_cast<size_t>(node_index)] <= EdgeBVHHybridClusterMaxHeight) {
        is_cluster_root[static_cast<size_t>(node_index)] = 1;
        cluster_roots.push_back(node_index);
        return;
    }

    select_lbvh_clusters(left_child[static_cast<size_t>(node_index)],
                         left_child,
                         right_child,
                         is_leaf,
                         subtree_leaf_counts,
                         subtree_heights,
                         is_cluster_root,
                         cluster_roots);
    select_lbvh_clusters(right_child[static_cast<size_t>(node_index)],
                         left_child,
                         right_child,
                         is_leaf,
                         subtree_leaf_counts,
                         subtree_heights,
                         is_cluster_root,
                         cluster_roots);
}

void collect_top_level_nodes(int node_index,
                             const std::vector<int> &left_child,
                             const std::vector<int> &right_child,
                             const std::vector<uint8_t> &is_cluster_root,
                             std::vector<int> &top_level_nodes) {
    if (is_cluster_root[static_cast<size_t>(node_index)] > 0) {
        return;
    }

    top_level_nodes.push_back(node_index);
    collect_top_level_nodes(left_child[static_cast<size_t>(node_index)],
                            left_child,
                            right_child,
                            is_cluster_root,
                            top_level_nodes);
    collect_top_level_nodes(right_child[static_cast<size_t>(node_index)],
                            left_child,
                            right_child,
                            is_cluster_root,
                            top_level_nodes);
}

bool choose_sah_split(std::vector<TopLevelBuildRecord> &records,
                      int begin,
                      int end,
                      int &split_index) {
    float best_cost = std::numeric_limits<float>::infinity();
    int best_axis = -1;
    int best_bin = -1;
    float best_axis_min = 0.f;
    float best_axis_max = 0.f;

    for (int axis = 0; axis < 3; ++axis) {
        float centroid_min = std::numeric_limits<float>::infinity();
        float centroid_max = -std::numeric_limits<float>::infinity();
        for (int index = begin; index < end; ++index) {
            const float value = axis_value(records[static_cast<size_t>(index)].centroid, axis);
            centroid_min = std::min(centroid_min, value);
            centroid_max = std::max(centroid_max, value);
        }

        if (!(centroid_max > centroid_min)) {
            continue;
        }

        std::array<SAHBin, EdgeBVHSAHBins> bins;
        for (SAHBin &bin : bins) {
            bin.bbox_min = empty_bbox_min();
            bin.bbox_max = empty_bbox_max();
            bin.count = 0;
        }

        const float scale = static_cast<float>(EdgeBVHSAHBins) / (centroid_max - centroid_min);
        for (int index = begin; index < end; ++index) {
            const float value = axis_value(records[static_cast<size_t>(index)].centroid, axis);
            int bin_index = static_cast<int>((value - centroid_min) * scale);
            bin_index = std::max(0, std::min(bin_index, EdgeBVHSAHBins - 1));

            SAHBin &bin = bins[static_cast<size_t>(bin_index)];
            bin.count += 1;
            bin.bbox_min = scalar_min(bin.bbox_min, records[static_cast<size_t>(index)].bbox_min);
            bin.bbox_max = scalar_max(bin.bbox_max, records[static_cast<size_t>(index)].bbox_max);
        }

        std::array<float, EdgeBVHSAHBins - 1> left_areas{};
        std::array<float, EdgeBVHSAHBins - 1> right_areas{};
        std::array<int, EdgeBVHSAHBins - 1> left_counts{};
        std::array<int, EdgeBVHSAHBins - 1> right_counts{};
        ScalarVector3f left_bbox_min = empty_bbox_min();
        ScalarVector3f left_bbox_max = empty_bbox_max();
        int left_count = 0;
        for (int bin_index = 0; bin_index < EdgeBVHSAHBins - 1; ++bin_index) {
            left_count += bins[static_cast<size_t>(bin_index)].count;
            if (bins[static_cast<size_t>(bin_index)].count > 0) {
                left_bbox_min = scalar_min(left_bbox_min, bins[static_cast<size_t>(bin_index)].bbox_min);
                left_bbox_max = scalar_max(left_bbox_max, bins[static_cast<size_t>(bin_index)].bbox_max);
            }
            left_counts[static_cast<size_t>(bin_index)] = left_count;
            left_areas[static_cast<size_t>(bin_index)] = bbox_surface_area(left_bbox_min, left_bbox_max);
        }

        ScalarVector3f right_bbox_min = empty_bbox_min();
        ScalarVector3f right_bbox_max = empty_bbox_max();
        int right_count = 0;
        for (int bin_index = EdgeBVHSAHBins - 1; bin_index >= 1; --bin_index) {
            right_count += bins[static_cast<size_t>(bin_index)].count;
            if (bins[static_cast<size_t>(bin_index)].count > 0) {
                right_bbox_min = scalar_min(right_bbox_min, bins[static_cast<size_t>(bin_index)].bbox_min);
                right_bbox_max = scalar_max(right_bbox_max, bins[static_cast<size_t>(bin_index)].bbox_max);
            }
            right_counts[static_cast<size_t>(bin_index - 1)] = right_count;
            right_areas[static_cast<size_t>(bin_index - 1)] = bbox_surface_area(right_bbox_min, right_bbox_max);
        }

        for (int bin_index = 0; bin_index < EdgeBVHSAHBins - 1; ++bin_index) {
            if (left_counts[static_cast<size_t>(bin_index)] == 0 ||
                right_counts[static_cast<size_t>(bin_index)] == 0) {
                continue;
            }

            const float cost =
                left_areas[static_cast<size_t>(bin_index)] *
                    static_cast<float>(left_counts[static_cast<size_t>(bin_index)]) +
                right_areas[static_cast<size_t>(bin_index)] *
                    static_cast<float>(right_counts[static_cast<size_t>(bin_index)]);
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_bin = bin_index;
                best_axis_min = centroid_min;
                best_axis_max = centroid_max;
            }
        }
    }

    if (best_axis >= 0) {
        const float split_value =
            best_axis_min +
            (best_axis_max - best_axis_min) *
                (static_cast<float>(best_bin + 1) / static_cast<float>(EdgeBVHSAHBins));
        auto middle = std::partition(
            records.begin() + begin,
            records.begin() + end,
            [best_axis, split_value](const TopLevelBuildRecord &record) {
                return axis_value(record.centroid, best_axis) < split_value;
            });

        split_index = static_cast<int>(middle - records.begin());
        const int min_side_count = std::max(1, (end - begin) / 8);
        if (split_index - begin >= min_side_count &&
            end - split_index >= min_side_count) {
            return true;
        }
    }

    ScalarVector3f centroid_min = empty_bbox_min();
    ScalarVector3f centroid_max = empty_bbox_max();
    for (int index = begin; index < end; ++index) {
        centroid_min = scalar_min(centroid_min, records[static_cast<size_t>(index)].centroid);
        centroid_max = scalar_max(centroid_max, records[static_cast<size_t>(index)].centroid);
    }

    const int axis = dominant_axis(centroid_max - centroid_min);
    split_index = begin + (end - begin) / 2;
    std::nth_element(records.begin() + begin,
                     records.begin() + split_index,
                     records.begin() + end,
                     [axis](const TopLevelBuildRecord &a, const TopLevelBuildRecord &b) {
                         return axis_value(a.centroid, axis) < axis_value(b.centroid, axis);
                     });
    return split_index > begin && split_index < end;
}

int build_top_level_bvh_recursive(std::vector<TopLevelBuildRecord> &records,
                                  int begin,
                                  int end,
                                  const std::vector<int> &top_level_nodes,
                                  size_t &next_top_level_node,
                                  std::vector<int> &left_child,
                                  std::vector<int> &right_child,
                                  std::vector<int> &leaf_primitive,
                                  std::vector<int> &is_leaf,
                                  std::vector<ScalarVector3f> &node_bbox_min,
                                  std::vector<ScalarVector3f> &node_bbox_max) {
    if (end - begin == 1) {
        return records[static_cast<size_t>(begin)].node_index;
    }

    require(next_top_level_node < top_level_nodes.size(),
            "SceneEdge::configure(): not enough nodes for hybrid top-level rebuild.");

    int split_index = begin + (end - begin) / 2;
    choose_sah_split(records, begin, end, split_index);

    const int node_index = top_level_nodes[next_top_level_node++];
    const int left_index = build_top_level_bvh_recursive(records,
                                                         begin,
                                                         split_index,
                                                         top_level_nodes,
                                                         next_top_level_node,
                                                         left_child,
                                                         right_child,
                                                         leaf_primitive,
                                                         is_leaf,
                                                         node_bbox_min,
                                                         node_bbox_max);
    const int right_index = build_top_level_bvh_recursive(records,
                                                          split_index,
                                                          end,
                                                          top_level_nodes,
                                                          next_top_level_node,
                                                          left_child,
                                                          right_child,
                                                          leaf_primitive,
                                                          is_leaf,
                                                          node_bbox_min,
                                                          node_bbox_max);

    left_child[static_cast<size_t>(node_index)] = left_index;
    right_child[static_cast<size_t>(node_index)] = right_index;
    leaf_primitive[static_cast<size_t>(node_index)] = -1;
    is_leaf[static_cast<size_t>(node_index)] = 0;
    node_bbox_min[static_cast<size_t>(node_index)] =
        scalar_min(node_bbox_min[static_cast<size_t>(left_index)],
                   node_bbox_min[static_cast<size_t>(right_index)]);
    node_bbox_max[static_cast<size_t>(node_index)] =
        scalar_max(node_bbox_max[static_cast<size_t>(left_index)],
                   node_bbox_max[static_cast<size_t>(right_index)]);
    return node_index;
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

struct TreeletBuildResult {
    int node_index = -1;
    ScalarVector3f bbox_min;
    ScalarVector3f bbox_max;
    float cost = 0.f;
};

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
            "SceneEdge::configure(): attempted to rebuild an empty treelet subset.");

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
            "SceneEdge::configure(): treelet rebuild ran out of internal nodes.");
    const int node_index = reusable_nodes[next_reusable_node++];
    const uint32_t left_subset = optimal_partitions[static_cast<size_t>(subset)];
    const uint32_t right_subset = subset ^ left_subset;
    require(left_subset != 0u && right_subset != 0u,
            "SceneEdge::configure(): invalid treelet partition.");

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
            "SceneEdge::configure(): failed to find a valid treelet partition.");
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
            "SceneEdge::configure(): invalid root treelet partition.");

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
            "SceneEdge::configure(): treelet rebuild did not consume every internal node.");

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

TraversalStack make_empty_stack(int query_count) {
    TraversalStack stack;
    for (size_t index = 0; index < EdgeBVHTraversalStackSize; ++index) {
        stack[index] = full<IntDetached>(-1, query_count);
    }
    return stack;
}

void stack_push(TraversalStack &stack,
                IntDetached &stack_size,
                const IntDetached &value,
                const MaskDetached &active) {
    const int query_count = static_cast<int>(slices(stack_size));
    for (size_t index = 0; index < EdgeBVHTraversalStackSize; ++index) {
        const MaskDetached write = active && (stack_size == static_cast<int>(index));
        stack[index] = select(write, value, stack[index]);
    }

    stack_size = stack_size +
                 select(active,
                        full<IntDetached>(1, query_count),
                        zeros<IntDetached>(query_count));
}

IntDetached stack_pop(TraversalStack &stack,
                      IntDetached &stack_size,
                      const MaskDetached &active) {
    const int query_count = static_cast<int>(slices(stack_size));
    const MaskDetached can_pop = active && (stack_size > 0);
    const IntDetached pop_index = stack_size - full<IntDetached>(1, query_count);

    IntDetached value = full<IntDetached>(-1, query_count);
    for (size_t index = 0; index < EdgeBVHTraversalStackSize; ++index) {
        value = select(can_pop && (pop_index == static_cast<int>(index)),
                       stack[index],
                       value);
    }

    stack_size = stack_size -
                 select(can_pop,
                        full<IntDetached>(1, query_count),
                        zeros<IntDetached>(query_count));
    return value;
}

} // namespace

void SceneEdge::configure(const SecondaryEdgeInfo &edge_info) {
    primitive_count_ = edge_info.size();
    node_count_ = 0;
    ready_ = false;
    refit_levels_.clear();

    if (primitive_count_ == 0) {
        edge_p0_ = Vector3fDetached();
        edge_e1_ = Vector3fDetached();
        primitive_bbox_min_ = Vector3fDetached();
        primitive_bbox_max_ = Vector3fDetached();
        node_bbox_min_ = Vector3fDetached();
        node_bbox_max_ = Vector3fDetached();
        left_child_ = IntDetached();
        right_child_ = IntDetached();
        leaf_primitive_ = IntDetached();
        is_leaf_ = IntDetached();
        primitive_leaf_node_ = IntDetached();
        ready_ = true;
        return;
    }

    edge_p0_ = detach<false>(edge_info.start);
    edge_e1_ = detach<false>(edge_info.edge);
    drjit::eval(edge_p0_, edge_e1_);
    drjit::sync_thread();

    node_count_ = std::max(2 * primitive_count_ - 1, 1);
    primitive_bbox_min_ = zero_vector3(primitive_count_);
    primitive_bbox_max_ = zero_vector3(primitive_count_);
    node_bbox_min_ = zero_vector3(node_count_);
    node_bbox_max_ = zero_vector3(node_count_);
    left_child_ = full<IntDetached>(-1, node_count_);
    right_child_ = full<IntDetached>(-1, node_count_);
    leaf_primitive_ = full<IntDetached>(-1, node_count_);
    is_leaf_ = zeros<IntDetached>(node_count_);
    primitive_leaf_node_ = full<IntDetached>(-1, primitive_count_);

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
        leaf_primitive_.data(),
        is_leaf_.data(),
        primitive_leaf_node_.data());

    drjit::sync_thread();

    const std::vector<int> left_child = copy_ints_to_host(left_child_);
    const std::vector<int> right_child = copy_ints_to_host(right_child_);
    const std::vector<int> is_leaf = copy_ints_to_host(is_leaf_);
    const std::vector<int> leaf_primitive = copy_ints_to_host(leaf_primitive_);
    std::vector<ScalarVector3f> node_bbox_min = copy_vector3_to_host(node_bbox_min_);
    std::vector<ScalarVector3f> node_bbox_max = copy_vector3_to_host(node_bbox_max_);

    std::vector<int> optimized_left_child = left_child;
    std::vector<int> optimized_right_child = right_child;
    std::vector<int> optimized_is_leaf = is_leaf;
    std::vector<int> optimized_leaf_primitive = leaf_primitive;

    if (EdgeBVHActivePostBuildStrategy == EdgeBVHPostBuildStrategy::HybridTopLevelSAH &&
        primitive_count_ >= EdgeBVHHybridTopLevelMinPrimitives) {
        std::vector<int> subtree_leaf_counts(static_cast<size_t>(node_count_), -1);
        std::vector<int> subtree_heights(static_cast<size_t>(node_count_), -1);
        compute_subtree_leaf_count(
            0, optimized_left_child, optimized_right_child, optimized_is_leaf, subtree_leaf_counts);
        compute_subtree_height(
            0, optimized_left_child, optimized_right_child, optimized_is_leaf, subtree_heights);

        std::vector<uint8_t> is_cluster_root(static_cast<size_t>(node_count_), 0);
        std::vector<int> cluster_roots;
        cluster_roots.reserve(static_cast<size_t>(
            std::max(1, (primitive_count_ + EdgeBVHHybridClusterLeafCount - 1) /
                            EdgeBVHHybridClusterLeafCount)));
        select_lbvh_clusters(0,
                             optimized_left_child,
                             optimized_right_child,
                             optimized_is_leaf,
                             subtree_leaf_counts,
                             subtree_heights,
                             is_cluster_root,
                             cluster_roots);

        if (cluster_roots.size() > 1) {
            std::vector<int> top_level_nodes;
            top_level_nodes.reserve(cluster_roots.size() - 1);
            collect_top_level_nodes(
                0, optimized_left_child, optimized_right_child, is_cluster_root, top_level_nodes);

            require(top_level_nodes.size() + 1 == cluster_roots.size(),
            "SceneEdge::configure(): invalid hybrid top-level node count.");

            std::vector<TopLevelBuildRecord> records(cluster_roots.size());
            for (size_t index = 0; index < cluster_roots.size(); ++index) {
                const int node_index = cluster_roots[index];
                records[index].node_index = node_index;
                records[index].bbox_min = node_bbox_min[static_cast<size_t>(node_index)];
                records[index].bbox_max = node_bbox_max[static_cast<size_t>(node_index)];
                records[index].centroid =
                    (records[index].bbox_min + records[index].bbox_max) * 0.5f;
            }

            size_t next_top_level_node = 0;
            const int hybrid_root = build_top_level_bvh_recursive(records,
                                                                  0,
                                                                  static_cast<int>(records.size()),
                                                                  top_level_nodes,
                                                                  next_top_level_node,
                                                                  optimized_left_child,
                                                                  optimized_right_child,
                                                                  optimized_leaf_primitive,
                                                                  optimized_is_leaf,
                                                                  node_bbox_min,
                                                                  node_bbox_max);

            require(hybrid_root == 0,
            "SceneEdge::configure(): hybrid top-level rebuild changed the root.");
            require(next_top_level_node == top_level_nodes.size(),
            "SceneEdge::configure(): hybrid top-level rebuild left unused nodes.");
        }
    } else if (EdgeBVHActivePostBuildStrategy == EdgeBVHPostBuildStrategy::Treelet &&
               primitive_count_ >= EdgeBVHTreeletMinPrimitives) {
        std::vector<int> subtree_leaf_counts(static_cast<size_t>(node_count_), -1);
        compute_subtree_leaf_count(
            0, optimized_left_child, optimized_right_child, optimized_is_leaf, subtree_leaf_counts);
        const ScalarVector3f scene_extent =
            scalar_max(node_bbox_max[0] - node_bbox_min[0], ScalarVector3f(0.f, 0.f, 0.f));
        const float scene_scale =
            std::max(scene_extent.x(), std::max(scene_extent.y(), scene_extent.z()));
        const float inflation =
            std::max(scene_scale * EdgeBVHTreeletCostInflationRatio, 1e-6f);
        std::vector<float> subtree_costs(static_cast<size_t>(node_count_), 0.f);
        optimize_treelets_recursive(0,
                                    optimized_left_child,
                                    optimized_right_child,
                                    optimized_leaf_primitive,
                                    optimized_is_leaf,
                                    node_bbox_min,
                                    node_bbox_max,
                                    subtree_costs,
                                    subtree_leaf_counts,
                                    inflation);
    }

    const std::vector<int> new_to_old =
        build_preorder_mapping(optimized_left_child, optimized_right_child, optimized_is_leaf);
    std::vector<int> old_to_new(static_cast<size_t>(node_count_), -1);
    for (int new_index = 0; new_index < node_count_; ++new_index) {
        old_to_new[static_cast<size_t>(new_to_old[static_cast<size_t>(new_index)])] = new_index;
    }

    std::vector<int> remapped_left_child(static_cast<size_t>(node_count_), -1);
    std::vector<int> remapped_right_child(static_cast<size_t>(node_count_), -1);
    std::vector<int> remapped_leaf_primitive(static_cast<size_t>(node_count_), -1);
    std::vector<int> remapped_is_leaf(static_cast<size_t>(node_count_), 0);
    std::vector<int> primitive_leaf_nodes(static_cast<size_t>(primitive_count_), -1);
    std::vector<ScalarVector3f> remapped_node_bbox_min(static_cast<size_t>(node_count_));
    std::vector<ScalarVector3f> remapped_node_bbox_max(static_cast<size_t>(node_count_));
    for (int new_index = 0; new_index < node_count_; ++new_index) {
        const int old_index = new_to_old[static_cast<size_t>(new_index)];
        remapped_is_leaf[static_cast<size_t>(new_index)] = optimized_is_leaf[static_cast<size_t>(old_index)];
        remapped_leaf_primitive[static_cast<size_t>(new_index)] =
            optimized_leaf_primitive[static_cast<size_t>(old_index)];
        remapped_node_bbox_min[static_cast<size_t>(new_index)] =
            node_bbox_min[static_cast<size_t>(old_index)];
        remapped_node_bbox_max[static_cast<size_t>(new_index)] =
            node_bbox_max[static_cast<size_t>(old_index)];

        if (remapped_is_leaf[static_cast<size_t>(new_index)] > 0) {
            const int primitive = remapped_leaf_primitive[static_cast<size_t>(new_index)];
            primitive_leaf_nodes[static_cast<size_t>(primitive)] = new_index;
        } else {
            remapped_left_child[static_cast<size_t>(new_index)] =
                old_to_new[static_cast<size_t>(optimized_left_child[static_cast<size_t>(old_index)])];
            remapped_right_child[static_cast<size_t>(new_index)] =
                old_to_new[static_cast<size_t>(optimized_right_child[static_cast<size_t>(old_index)])];
        }
    }

    node_bbox_min_ = load_vector3(remapped_node_bbox_min);
    node_bbox_max_ = load_vector3(remapped_node_bbox_max);
    left_child_ = load_ints(remapped_left_child);
    right_child_ = load_ints(remapped_right_child);
    leaf_primitive_ = load_ints(remapped_leaf_primitive);
    is_leaf_ = load_ints(remapped_is_leaf);
    primitive_leaf_node_ = load_ints(primitive_leaf_nodes);

    std::vector<int> heights(static_cast<size_t>(node_count_), -1);
    const int max_height = compute_node_height(
        0, remapped_left_child, remapped_right_child, remapped_is_leaf, heights);

    require(max_height + 1 <= static_cast<int>(EdgeBVHTraversalStackSize),
            "SceneEdge::configure(): BVH depth exceeds traversal stack capacity.");

    std::vector<std::vector<int>> refit_levels(static_cast<size_t>(max_height + 1));
    for (int node_index = 0; node_index < node_count_; ++node_index) {
        if (remapped_is_leaf[static_cast<size_t>(node_index)] == 0) {
            refit_levels[static_cast<size_t>(heights[static_cast<size_t>(node_index)])].push_back(node_index);
        }
    }
    for (int height = 1; height <= max_height; ++height) {
        if (!refit_levels[static_cast<size_t>(height)].empty()) {
            refit_levels_.push_back(
                load<IntDetached>(refit_levels[static_cast<size_t>(height)].data(),
                                  refit_levels[static_cast<size_t>(height)].size()));
        }
    }

    drjit::eval(edge_p0_,
                edge_e1_,
                primitive_bbox_min_,
                primitive_bbox_max_,
                node_bbox_min_,
                node_bbox_max_,
                left_child_,
                right_child_,
                leaf_primitive_,
                is_leaf_,
                primitive_leaf_node_);
    drjit::sync_thread();
    ready_ = true;
}

void SceneEdge::refit(const SecondaryEdgeInfo &edge_info,
                          const std::vector<EdgeDirtyRange> &dirty_ranges) {
    require(ready_, "SceneEdge::refit(): BVH is not configured.");
    if (primitive_count_ == 0 || dirty_ranges.empty()) {
        return;
    }

    const Vector3fDetached scene_p0 = detach<false>(edge_info.start);
    const Vector3fDetached scene_e1 = detach<false>(edge_info.edge);
    for (const EdgeDirtyRange &range : dirty_ranges) {
        if (range.count <= 0) {
            continue;
        }

        const IntDetached primitive_indices = arange<IntDetached>(range.count) + range.offset;
        const Vector3fDetached edge_p0 = gather<Vector3fDetached>(scene_p0, primitive_indices);
        const Vector3fDetached edge_e1 = gather<Vector3fDetached>(scene_e1, primitive_indices);
        const Vector3fDetached edge_p1 = edge_p0 + edge_e1;
        const Vector3fDetached bbox_min = minimum(edge_p0, edge_p1);
        const Vector3fDetached bbox_max = maximum(edge_p0, edge_p1);
        const IntDetached leaf_nodes = gather<IntDetached>(primitive_leaf_node_, primitive_indices);

        scatter(edge_p0_, edge_p0, primitive_indices);
        scatter(edge_e1_, edge_e1, primitive_indices);
        scatter(primitive_bbox_min_, bbox_min, primitive_indices);
        scatter(primitive_bbox_max_, bbox_max, primitive_indices);
        scatter(node_bbox_min_, bbox_min, leaf_nodes);
        scatter(node_bbox_max_, bbox_max, leaf_nodes);
    }

    for (const IntDetached &level : refit_levels_) {
        const IntDetached left = gather<IntDetached>(left_child_, level);
        const IntDetached right = gather<IntDetached>(right_child_, level);
        const Vector3fDetached left_bbox_min = gather<Vector3fDetached>(node_bbox_min_, left);
        const Vector3fDetached left_bbox_max = gather<Vector3fDetached>(node_bbox_max_, left);
        const Vector3fDetached right_bbox_min = gather<Vector3fDetached>(node_bbox_min_, right);
        const Vector3fDetached right_bbox_max = gather<Vector3fDetached>(node_bbox_max_, right);
        scatter(node_bbox_min_, minimum(left_bbox_min, right_bbox_min), level);
        scatter(node_bbox_max_, maximum(left_bbox_max, right_bbox_max), level);
    }

    drjit::eval(edge_p0_,
                edge_e1_,
                primitive_bbox_min_,
                primitive_bbox_max_,
                node_bbox_min_,
                node_bbox_max_);
    drjit::sync_thread();
}

ClosestEdgeCandidate SceneEdge::closest_edge_point_detached(const Vector3fDetached &point,
                                                                const MaskDetached &active) const {
    const int query_count = static_cast<int>(slices(point));

    ClosestEdgeCandidate result;
    result.global_edge_id = full<IntDetached>(-1, query_count);
    result.distance_sq = full<FloatDetached>(Infinity, query_count);
    if (primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    auto [current_node,
          stack_size,
          stack,
          best_distance_sq,
          best_primitive] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<IntDetached>(query_count), full<IntDetached>(-1, query_count)),
                          zeros<IntDetached>(query_count),
                          make_empty_stack(query_count),
                          full<FloatDetached>(Infinity, query_count),
                          full<IntDetached>(-1, query_count)),
        [](const IntDetached &current_node,
           const IntDetached &stack_size,
           const TraversalStack &,
           const FloatDetached &,
           const IntDetached &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &point, query_count](IntDetached &current_node,
                                    IntDetached &stack_size,
                                    TraversalStack &stack,
                                    FloatDetached &best_distance_sq,
                                    IntDetached &best_primitive) {
            const MaskDetached need_pop = (current_node < 0) && (stack_size > 0);
            const IntDetached popped_node = stack_pop(stack, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const MaskDetached lane_active = current_node >= 0;
            const Vector3fDetached bbox_min = gather<Vector3fDetached>(node_bbox_min_, current_node, lane_active);
            const Vector3fDetached bbox_max = gather<Vector3fDetached>(node_bbox_max_, current_node, lane_active);
            const FloatDetached node_bound = point_aabb_distance_sq(point, bbox_min, bbox_max);
            const MaskDetached visit = lane_active && (node_bound <= best_distance_sq);

            const MaskDetached leaf_node =
                lane_active && (gather<IntDetached>(is_leaf_, current_node, lane_active) > 0);
            const MaskDetached leaf_visit = visit && leaf_node;
            const IntDetached primitive_index = gather<IntDetached>(leaf_primitive_, current_node, leaf_visit);
            const Vector3fDetached edge_p0 = gather<Vector3fDetached>(edge_p0_, primitive_index, leaf_visit);
            const Vector3fDetached edge_e1 = gather<Vector3fDetached>(edge_e1_, primitive_index, leaf_visit);

            FloatDetached edge_t;
            Vector3fDetached edge_point;
            FloatDetached candidate_distance_sq;
            std::tie(edge_t, edge_point, candidate_distance_sq) =
                closest_point_on_segment<true>(point, edge_p0, edge_e1);
            DRJIT_MARK_USED(edge_t);
            DRJIT_MARK_USED(edge_point);

            const MaskDetached better = leaf_visit && (candidate_distance_sq < best_distance_sq);
            best_distance_sq = select(better, candidate_distance_sq, best_distance_sq);
            best_primitive = select(better, primitive_index, best_primitive);

            const MaskDetached internal_visit = visit && !leaf_node;
            const IntDetached left = gather<IntDetached>(left_child_, current_node, internal_visit);
            const IntDetached right = gather<IntDetached>(right_child_, current_node, internal_visit);

            const Vector3fDetached left_bbox_min = gather<Vector3fDetached>(node_bbox_min_, left, internal_visit);
            const Vector3fDetached left_bbox_max = gather<Vector3fDetached>(node_bbox_max_, left, internal_visit);
            const Vector3fDetached right_bbox_min = gather<Vector3fDetached>(node_bbox_min_, right, internal_visit);
            const Vector3fDetached right_bbox_max = gather<Vector3fDetached>(node_bbox_max_, right, internal_visit);
            const FloatDetached left_bound = point_aabb_distance_sq(point, left_bbox_min, left_bbox_max);
            const FloatDetached right_bound = point_aabb_distance_sq(point, right_bbox_min, right_bbox_max);

            const MaskDetached left_visit = internal_visit && (left_bound <= best_distance_sq);
            const MaskDetached right_visit = internal_visit && (right_bound <= best_distance_sq);
            const MaskDetached both_children = left_visit && right_visit;
            const MaskDetached only_left = left_visit && !right_visit;
            const MaskDetached only_right = right_visit && !left_visit;
            const MaskDetached left_first = left_bound <= right_bound;

            const IntDetached near_child = select(left_first, left, right);
            const IntDetached far_child = select(left_first, right, left);
            stack_push(stack, stack_size, far_child, both_children);

            IntDetached next_node = full<IntDetached>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "closest_edge_point_bvh");

    result.global_edge_id = best_primitive;
    result.distance_sq = best_distance_sq;
    return result;
}

ClosestEdgeCandidate SceneEdge::closest_edge_finite_ray_detached(const Vector3fDetached &origin,
                                                                     const Vector3fDetached &segment,
                                                                     const MaskDetached &active) const {
    const int query_count = static_cast<int>(slices(origin));

    ClosestEdgeCandidate result;
    result.global_edge_id = full<IntDetached>(-1, query_count);
    result.distance_sq = full<FloatDetached>(Infinity, query_count);
    if (primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    auto [current_node,
          stack_size,
          stack,
          best_distance_sq,
          best_primitive] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<IntDetached>(query_count), full<IntDetached>(-1, query_count)),
                          zeros<IntDetached>(query_count),
                          make_empty_stack(query_count),
                          full<FloatDetached>(Infinity, query_count),
                          full<IntDetached>(-1, query_count)),
        [](const IntDetached &current_node,
           const IntDetached &stack_size,
           const TraversalStack &,
           const FloatDetached &,
           const IntDetached &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &origin, &segment, query_count](IntDetached &current_node,
                                               IntDetached &stack_size,
                                               TraversalStack &stack,
                                               FloatDetached &best_distance_sq,
                                               IntDetached &best_primitive) {
            const MaskDetached need_pop = (current_node < 0) && (stack_size > 0);
            const IntDetached popped_node = stack_pop(stack, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const MaskDetached lane_active = current_node >= 0;
            const Vector3fDetached bbox_min = gather<Vector3fDetached>(node_bbox_min_, current_node, lane_active);
            const Vector3fDetached bbox_max = gather<Vector3fDetached>(node_bbox_max_, current_node, lane_active);
            const FloatDetached node_bound = segment_aabb_lower_bound_sq(origin, segment, bbox_min, bbox_max);
            const MaskDetached visit = lane_active && (node_bound <= best_distance_sq);

            const MaskDetached leaf_node =
                lane_active && (gather<IntDetached>(is_leaf_, current_node, lane_active) > 0);
            const MaskDetached leaf_visit = visit && leaf_node;
            const IntDetached primitive_index = gather<IntDetached>(leaf_primitive_, current_node, leaf_visit);
            const Vector3fDetached edge_p0 = gather<Vector3fDetached>(edge_p0_, primitive_index, leaf_visit);
            const Vector3fDetached edge_e1 = gather<Vector3fDetached>(edge_e1_, primitive_index, leaf_visit);

            FloatDetached query_t;
            Vector3fDetached query_point;
            FloatDetached edge_t;
            Vector3fDetached edge_point;
            FloatDetached candidate_distance_sq;
            std::tie(query_t, query_point, edge_t, edge_point, candidate_distance_sq) =
                closest_segment_segment<true>(origin, segment, edge_p0, edge_e1);
            DRJIT_MARK_USED(query_t);
            DRJIT_MARK_USED(query_point);
            DRJIT_MARK_USED(edge_t);
            DRJIT_MARK_USED(edge_point);

            const MaskDetached better = leaf_visit && (candidate_distance_sq < best_distance_sq);
            best_distance_sq = select(better, candidate_distance_sq, best_distance_sq);
            best_primitive = select(better, primitive_index, best_primitive);

            const MaskDetached internal_visit = visit && !leaf_node;
            const IntDetached left = gather<IntDetached>(left_child_, current_node, internal_visit);
            const IntDetached right = gather<IntDetached>(right_child_, current_node, internal_visit);

            const Vector3fDetached left_bbox_min = gather<Vector3fDetached>(node_bbox_min_, left, internal_visit);
            const Vector3fDetached left_bbox_max = gather<Vector3fDetached>(node_bbox_max_, left, internal_visit);
            const Vector3fDetached right_bbox_min = gather<Vector3fDetached>(node_bbox_min_, right, internal_visit);
            const Vector3fDetached right_bbox_max = gather<Vector3fDetached>(node_bbox_max_, right, internal_visit);
            const FloatDetached left_bound = segment_aabb_lower_bound_sq(origin, segment, left_bbox_min, left_bbox_max);
            const FloatDetached right_bound = segment_aabb_lower_bound_sq(origin, segment, right_bbox_min, right_bbox_max);

            const MaskDetached left_visit = internal_visit && (left_bound <= best_distance_sq);
            const MaskDetached right_visit = internal_visit && (right_bound <= best_distance_sq);
            const MaskDetached both_children = left_visit && right_visit;
            const MaskDetached only_left = left_visit && !right_visit;
            const MaskDetached only_right = right_visit && !left_visit;
            const MaskDetached left_first = left_bound <= right_bound;

            const IntDetached near_child = select(left_first, left, right);
            const IntDetached far_child = select(left_first, right, left);
            stack_push(stack, stack_size, far_child, both_children);

            IntDetached next_node = full<IntDetached>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "closest_edge_finite_ray_bvh");

    result.global_edge_id = best_primitive;
    result.distance_sq = best_distance_sq;
    return result;
}

ClosestEdgeCandidate SceneEdge::closest_edge_infinite_ray_detached(const Vector3fDetached &origin,
                                                                       const Vector3fDetached &direction,
                                                                       const MaskDetached &active) const {
    const int query_count = static_cast<int>(slices(origin));

    ClosestEdgeCandidate result;
    result.global_edge_id = full<IntDetached>(-1, query_count);
    result.distance_sq = full<FloatDetached>(Infinity, query_count);
    if (primitive_count_ == 0 || drjit::none(active)) {
        return result;
    }

    auto [current_node,
          stack_size,
          stack,
          best_distance_sq,
          best_primitive] = drjit::while_loop(
        drjit::make_tuple(select(active, zeros<IntDetached>(query_count), full<IntDetached>(-1, query_count)),
                          zeros<IntDetached>(query_count),
                          make_empty_stack(query_count),
                          full<FloatDetached>(Infinity, query_count),
                          full<IntDetached>(-1, query_count)),
        [](const IntDetached &current_node,
           const IntDetached &stack_size,
           const TraversalStack &,
           const FloatDetached &,
           const IntDetached &) {
            return (current_node >= 0) || (stack_size > 0);
        },
        [this, &origin, &direction, query_count](IntDetached &current_node,
                                                 IntDetached &stack_size,
                                                 TraversalStack &stack,
                                                 FloatDetached &best_distance_sq,
                                                 IntDetached &best_primitive) {
            const MaskDetached need_pop = (current_node < 0) && (stack_size > 0);
            const IntDetached popped_node = stack_pop(stack, stack_size, need_pop);
            current_node = select(need_pop, popped_node, current_node);

            const MaskDetached lane_active = current_node >= 0;
            const Vector3fDetached bbox_min = gather<Vector3fDetached>(node_bbox_min_, current_node, lane_active);
            const Vector3fDetached bbox_max = gather<Vector3fDetached>(node_bbox_max_, current_node, lane_active);
            const FloatDetached node_bound = ray_aabb_lower_bound_sq(origin, direction, bbox_min, bbox_max);
            const MaskDetached visit = lane_active && (node_bound <= best_distance_sq);

            const MaskDetached leaf_node =
                lane_active && (gather<IntDetached>(is_leaf_, current_node, lane_active) > 0);
            const MaskDetached leaf_visit = visit && leaf_node;
            const IntDetached primitive_index = gather<IntDetached>(leaf_primitive_, current_node, leaf_visit);
            const Vector3fDetached edge_p0 = gather<Vector3fDetached>(edge_p0_, primitive_index, leaf_visit);
            const Vector3fDetached edge_e1 = gather<Vector3fDetached>(edge_e1_, primitive_index, leaf_visit);

            FloatDetached query_t;
            Vector3fDetached query_point;
            FloatDetached edge_t;
            Vector3fDetached edge_point;
            FloatDetached candidate_distance_sq;
            std::tie(query_t, query_point, edge_t, edge_point, candidate_distance_sq) =
                closest_ray_segment<true>(origin, direction, edge_p0, edge_e1);
            DRJIT_MARK_USED(query_t);
            DRJIT_MARK_USED(query_point);
            DRJIT_MARK_USED(edge_t);
            DRJIT_MARK_USED(edge_point);

            const MaskDetached better = leaf_visit && (candidate_distance_sq < best_distance_sq);
            best_distance_sq = select(better, candidate_distance_sq, best_distance_sq);
            best_primitive = select(better, primitive_index, best_primitive);

            const MaskDetached internal_visit = visit && !leaf_node;
            const IntDetached left = gather<IntDetached>(left_child_, current_node, internal_visit);
            const IntDetached right = gather<IntDetached>(right_child_, current_node, internal_visit);

            const Vector3fDetached left_bbox_min = gather<Vector3fDetached>(node_bbox_min_, left, internal_visit);
            const Vector3fDetached left_bbox_max = gather<Vector3fDetached>(node_bbox_max_, left, internal_visit);
            const Vector3fDetached right_bbox_min = gather<Vector3fDetached>(node_bbox_min_, right, internal_visit);
            const Vector3fDetached right_bbox_max = gather<Vector3fDetached>(node_bbox_max_, right, internal_visit);
            const FloatDetached left_bound = ray_aabb_lower_bound_sq(origin, direction, left_bbox_min, left_bbox_max);
            const FloatDetached right_bound = ray_aabb_lower_bound_sq(origin, direction, right_bbox_min, right_bbox_max);

            const MaskDetached left_visit = internal_visit && (left_bound <= best_distance_sq);
            const MaskDetached right_visit = internal_visit && (right_bound <= best_distance_sq);
            const MaskDetached both_children = left_visit && right_visit;
            const MaskDetached only_left = left_visit && !right_visit;
            const MaskDetached only_right = right_visit && !left_visit;
            const MaskDetached left_first = left_bound <= right_bound;

            const IntDetached near_child = select(left_first, left, right);
            const IntDetached far_child = select(left_first, right, left);
            stack_push(stack, stack_size, far_child, both_children);

            IntDetached next_node = full<IntDetached>(-1, query_count);
            next_node = select(both_children, near_child, next_node);
            next_node = select(only_left, left, next_node);
            next_node = select(only_right, right, next_node);
            current_node = select(lane_active, next_node, current_node);
        },
        "closest_edge_infinite_ray_bvh");

    result.global_edge_id = best_primitive;
    result.distance_sq = best_distance_sq;
    return result;
}

template <bool Detached>
ClosestEdgeCandidate SceneEdge::closest_edge(const Vector3fT<Detached> &point,
                                                 MaskT<Detached> &active) const {
    require(ready_, "SceneEdge::closest_edge(point): BVH is not configured.");

    const int query_count = static_cast<int>(slices(point));
    ClosestEdgeCandidate result;
    result.global_edge_id = full<IntDetached>(-1, query_count);
    result.distance_sq = full<FloatDetached>(Infinity, query_count);
    if (primitive_count_ == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const MaskDetached active_detached = detach<false>(active);
    result = closest_edge_point_detached(detach<false>(point), active_detached);
    if constexpr (!Detached) {
        active &= Mask(result.global_edge_id >= 0);
    } else {
        active &= (result.global_edge_id >= 0);
    }
    return result;
}

template <bool Detached>
ClosestEdgeCandidate SceneEdge::closest_edge(const RayT<Detached> &ray,
                                                 MaskT<Detached> &active) const {
    require(ready_, "SceneEdge::closest_edge(ray): BVH is not configured.");

    const int query_count = static_cast<int>(slices(ray.o));
    ClosestEdgeCandidate result;
    result.global_edge_id = full<IntDetached>(-1, query_count);
    result.distance_sq = full<FloatDetached>(Infinity, query_count);
    if (primitive_count_ == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const MaskDetached active_detached = detach<false>(active);
    if (drjit::none(active_detached)) {
        return result;
    }

    const Vector3fDetached origin = detach<false>(ray.o);
    const Vector3fDetached direction = detach<false>(ray.d);
    const FloatDetached tmax = detach<false>(ray.tmax);
    const MaskDetached finite_mask = active_detached && drjit::isfinite(tmax);
    const MaskDetached infinite_mask = active_detached && !drjit::isfinite(tmax);

    if (drjit::any(finite_mask)) {
        const ClosestEdgeCandidate finite_result =
            closest_edge_finite_ray_detached(origin, direction * tmax, finite_mask);
        result.global_edge_id = select(finite_mask, finite_result.global_edge_id, result.global_edge_id);
        result.distance_sq = select(finite_mask, finite_result.distance_sq, result.distance_sq);
    }

    if (drjit::any(infinite_mask)) {
        const ClosestEdgeCandidate infinite_result =
            closest_edge_infinite_ray_detached(origin, direction, infinite_mask);
        result.global_edge_id = select(infinite_mask, infinite_result.global_edge_id, result.global_edge_id);
        result.distance_sq = select(infinite_mask, infinite_result.distance_sq, result.distance_sq);
    }

    if constexpr (!Detached) {
        active &= Mask(result.global_edge_id >= 0);
    } else {
        active &= (result.global_edge_id >= 0);
    }
    return result;
}

template ClosestEdgeCandidate SceneEdge::closest_edge<true>(const Vector3fDetached &point,
                                                                MaskDetached &active) const;
template ClosestEdgeCandidate SceneEdge::closest_edge<false>(const Vector3f &point,
                                                                 Mask &active) const;
template ClosestEdgeCandidate SceneEdge::closest_edge<true>(const RayDetached &ray,
                                                                MaskDetached &active) const;
template ClosestEdgeCandidate SceneEdge::closest_edge<false>(const Ray &ray,
                                                                 Mask &active) const;

} // namespace rayd

