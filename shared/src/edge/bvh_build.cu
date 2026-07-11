#include <rayd/shared/edge/bvh_build.h>

namespace rayd::shared::edge {
namespace {

constexpr int kBlockSize = 256;

__global__ void mark_dirty_ancestors_kernel(int leaf_count,
                                            const std::int32_t *leaf_nodes,
                                            const std::int32_t *node_parent,
                                            std::int32_t *dirty_marks) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= leaf_count) {
        return;
    }

    const int leaf_node = leaf_nodes[leaf_index];
    if (leaf_node < 0 || atomicExch(dirty_marks + leaf_node, 1) != 0) {
        return;
    }

    int current = node_parent[leaf_node];
    while (current >= 0) {
        if (atomicExch(dirty_marks + current, 1) != 0) {
            break;
        }
        current = node_parent[current];
    }
}

__global__ void compact_dirty_level_kernel(int level_count,
                                           const std::int32_t *level_nodes,
                                           const std::int32_t *dirty_marks,
                                           std::int32_t *selected_nodes,
                                           std::int32_t *selected_count) {
    const int item_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item_index >= level_count) {
        return;
    }

    const int node_index = level_nodes[item_index];
    if (node_index < 0 || dirty_marks[node_index] == 0) {
        return;
    }

    selected_nodes[atomicAdd(selected_count, 1)] = node_index;
}

__global__ void refit_selected_internal_nodes_kernel(
    int max_selected_count,
    const std::int32_t *selected_count,
    const std::int32_t *selected_nodes,
    const std::int32_t *left_child,
    const std::int32_t *right_child,
    MutableAabbSoAView node_bounds) {
    const int item_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int selected = *selected_count;
    if (item_index >= max_selected_count || item_index >= selected) {
        return;
    }

    const int node_index = selected_nodes[item_index];
    const int left = left_child[node_index];
    const int right = right_child[node_index];
    node_bounds.min_x[node_index] = fminf(node_bounds.min_x[left], node_bounds.min_x[right]);
    node_bounds.min_y[node_index] = fminf(node_bounds.min_y[left], node_bounds.min_y[right]);
    node_bounds.min_z[node_index] = fminf(node_bounds.min_z[left], node_bounds.min_z[right]);
    node_bounds.max_x[node_index] = fmaxf(node_bounds.max_x[left], node_bounds.max_x[right]);
    node_bounds.max_y[node_index] = fmaxf(node_bounds.max_y[left], node_bounds.max_y[right]);
    node_bounds.max_z[node_index] = fmaxf(node_bounds.max_z[left], node_bounds.max_z[right]);
}

} // namespace

void launch_mark_dirty_ancestors_async(const DirtyAncestorMarkParams &params) {
    const int count = params.leaf_count;
    if (count == 0) {
        return;
    }
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    mark_dirty_ancestors_kernel<<<blocks, kBlockSize, 0, params.stream>>>(
        count, params.leaf_nodes, params.node_parent, params.dirty_marks);
}

void launch_compact_dirty_level_async(const DirtyLevelCompactParams &params) {
    const int count = params.level_count;
    if (count == 0) {
        return;
    }
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    compact_dirty_level_kernel<<<blocks, kBlockSize, 0, params.stream>>>(
        count, params.level_nodes, params.dirty_marks, params.selected_nodes, params.selected_count);
}

void launch_refit_selected_internal_nodes_async(const InternalNodeRefitParams &params) {
    const int count = params.max_selected_count;
    if (count == 0) {
        return;
    }
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    refit_selected_internal_nodes_kernel<<<blocks, kBlockSize, 0, params.stream>>>(
        count,
        params.selected_count,
        params.selected_nodes,
        params.left_child,
        params.right_child,
        params.node_bounds);
}

} // namespace rayd::shared::edge
