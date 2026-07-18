#include <rayd/shared/bvh/build.h>
#include <rayd/shared/bvh/refit.h>

namespace rayd::shared::bvh {
namespace {

constexpr int kBlockSize = 256;

__host__ __device__ inline BvhFloat3 min3(const BvhFloat3 &a, const BvhFloat3 &b) {
    return { fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}

__host__ __device__ inline BvhFloat3 max3(const BvhFloat3 &a, const BvhFloat3 &b) {
    return { fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

__host__ __device__ inline BvhFloat3 add3(const BvhFloat3 &a, const BvhFloat3 &b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__ inline BvhFloat3 mul3(const BvhFloat3 &a, float scale) {
    return { a.x * scale, a.y * scale, a.z * scale };
}

__host__ __device__ inline BvhBounds3 merge_bounds(const BvhBounds3 &a,
                                                    const BvhBounds3 &b) {
    return { min3(a.min, b.min), max3(a.max, b.max) };
}

__device__ inline BvhBounds3 empty_bounds() {
    constexpr float inf = 1e30f;
    return { { inf, inf, inf }, { -inf, -inf, -inf } };
}

__host__ __device__ inline std::uint32_t expand_bits_10(std::uint32_t value) {
    value &= 0x000003ffu;
    value = (value | (value << 16)) & 0x030000FFu;
    value = (value | (value << 8)) & 0x0300F00Fu;
    value = (value | (value << 4)) & 0x030C30C3u;
    value = (value | (value << 2)) & 0x09249249u;
    return value;
}

__host__ __device__ inline std::uint32_t morton_code_3d(const BvhFloat3 &point,
                                                        const BvhBounds3 &scene_bounds) {
    BvhFloat3 normalized{ 0.5f, 0.5f, 0.5f };
    const BvhFloat3 extent{ scene_bounds.max.x - scene_bounds.min.x,
                            scene_bounds.max.y - scene_bounds.min.y,
                            scene_bounds.max.z - scene_bounds.min.z };
    if (extent.x > 0.f) normalized.x = (point.x - scene_bounds.min.x) / extent.x;
    if (extent.y > 0.f) normalized.y = (point.y - scene_bounds.min.y) / extent.y;
    if (extent.z > 0.f) normalized.z = (point.z - scene_bounds.min.z) / extent.z;
    normalized.x = fminf(fmaxf(normalized.x, 0.f), 1.f);
    normalized.y = fminf(fmaxf(normalized.y, 0.f), 1.f);
    normalized.z = fminf(fmaxf(normalized.z, 0.f), 1.f);
    constexpr std::uint32_t scale = (1u << 10) - 1u;
    const auto x = static_cast<std::uint32_t>(normalized.x * static_cast<float>(scale));
    const auto y = static_cast<std::uint32_t>(normalized.y * static_cast<float>(scale));
    const auto z = static_cast<std::uint32_t>(normalized.z * static_cast<float>(scale));
    return (expand_bits_10(x) << 2u) | (expand_bits_10(y) << 1u) | expand_bits_10(z);
}

__device__ inline int clz_u32(std::uint32_t value) {
    return value == 0u ? 32 : __clz(value);
}

__device__ inline BvhBounds3 load_bounds(int index, AabbSoAView bounds) {
    return { { bounds.min_x[index], bounds.min_y[index], bounds.min_z[index] },
             { bounds.max_x[index], bounds.max_y[index], bounds.max_z[index] } };
}

__device__ inline BvhBounds3 load_bounds(int index, MutableAabbSoAView bounds) {
    return { { bounds.min_x[index], bounds.min_y[index], bounds.min_z[index] },
             { bounds.max_x[index], bounds.max_y[index], bounds.max_z[index] } };
}

__device__ inline void store_bounds(int index,
                                    const BvhBounds3 &value,
                                    MutableAabbSoAView bounds) {
    bounds.min_x[index] = value.min.x;
    bounds.min_y[index] = value.min.y;
    bounds.min_z[index] = value.min.z;
    bounds.max_x[index] = value.max.x;
    bounds.max_y[index] = value.max.y;
    bounds.max_z[index] = value.max.z;
}

__device__ inline float bbox_cost_inflated(const BvhBounds3 &bounds, float inflation) {
    const float dx = fmaxf(bounds.max.x - bounds.min.x, 0.f) + inflation;
    const float dy = fmaxf(bounds.max.y - bounds.min.y, 0.f) + inflation;
    const float dz = fmaxf(bounds.max.z - bounds.min.z, 0.f) + inflation;
    return 2.f * (dx * dy + dx * dz + dy * dz);
}

__device__ inline bool node_in_treelet(int node_index,
                                       const int *treelet_nodes,
                                       int treelet_node_count) {
    for (int i = 0; i < treelet_node_count; ++i) {
        if (treelet_nodes[i] == node_index) return true;
    }
    return false;
}

__device__ inline void update_internal_node(int node_index,
                                            const int *left_child,
                                            const int *right_child,
                                            MutableAabbSoAView node_bounds,
                                            float *node_cost,
                                            float inflation) {
    const int left = left_child[node_index];
    const int right = right_child[node_index];
    const BvhBounds3 merged = merge_bounds(load_bounds(left, node_bounds),
                                           load_bounds(right, node_bounds));
    store_bounds(node_index, merged, node_bounds);
    node_cost[node_index] =
        bbox_cost_inflated(merged, inflation) + node_cost[left] + node_cost[right];
}

struct TreeletPartitionEntry {
    std::uint8_t partition;
    std::uint8_t child_slot;
    int parent_node;
};

__device__ inline void treelet_optimize_node(int root_index,
                                             const int *is_leaf,
                                             int *left_child,
                                             int *right_child,
                                             int *parent,
                                             MutableAabbSoAView node_bounds,
                                             int *leaf_primitive,
                                             float *node_cost,
                                             float inflation) {
    update_internal_node(root_index, left_child, right_child, node_bounds, node_cost, inflation);

    int frontier_nodes[kBvhTreeletMaxLeaves];
    int reusable_nodes[kBvhTreeletMaxLeaves - 2];
    int treelet_nodes[kBvhTreeletMaxLeaves - 1];
    int frontier_count = 0;
    int reusable_count = 0;
    frontier_nodes[frontier_count++] = left_child[root_index];
    frontier_nodes[frontier_count++] = right_child[root_index];

    while (frontier_count < kBvhTreeletMaxLeaves) {
        int expand_slot = -1;
        float max_cost = -1.f;
        for (int i = 0; i < frontier_count; ++i) {
            const int node = frontier_nodes[i];
            if (is_leaf[node] > 0) continue;
            const float cost = bbox_cost_inflated(load_bounds(node, node_bounds), inflation);
            if (cost > max_cost) {
                max_cost = cost;
                expand_slot = i;
            }
        }
        if (expand_slot < 0) break;
        const int expanded = frontier_nodes[expand_slot];
        reusable_nodes[reusable_count++] = expanded;
        frontier_nodes[expand_slot] = left_child[expanded];
        frontier_nodes[frontier_count++] = right_child[expanded];
    }
    if (frontier_count < 3) return;

    constexpr int kMaxSubsets = 1 << kBvhTreeletMaxLeaves;
    float subset_bbox_cost[kMaxSubsets];
    float optimal_cost[kMaxSubsets];
    std::uint8_t optimal_partitions[kMaxSubsets];
    const std::uint32_t full_mask = (1u << frontier_count) - 1u;
    for (std::uint32_t subset = 1u; subset <= full_mask; ++subset) {
        BvhBounds3 bounds = empty_bounds();
        for (int i = 0; i < frontier_count; ++i) {
            if ((subset & (1u << i)) != 0u) {
                bounds = merge_bounds(bounds, load_bounds(frontier_nodes[i], node_bounds));
            }
        }
        subset_bbox_cost[subset] = bbox_cost_inflated(bounds, inflation);
    }
    for (int i = 0; i < frontier_count; ++i) {
        const std::uint32_t subset = 1u << i;
        optimal_cost[subset] = node_cost[frontier_nodes[i]];
        optimal_partitions[subset] = 0u;
    }
    for (int subset_size = 2; subset_size <= frontier_count; ++subset_size) {
        for (std::uint32_t subset = 1u; subset <= full_mask; ++subset) {
            if (__popc(subset) != subset_size) continue;
            float best_children_cost = 1e30f;
            std::uint32_t best_partition = 0u;
            for (std::uint32_t left_subset = (subset - 1u) & subset;
                 left_subset > 0u;
                 left_subset = (left_subset - 1u) & subset) {
                const std::uint32_t right_subset = subset ^ left_subset;
                if (right_subset == 0u || left_subset > right_subset) continue;
                const float candidate = optimal_cost[left_subset] + optimal_cost[right_subset];
                if (candidate < best_children_cost) {
                    best_children_cost = candidate;
                    best_partition = left_subset;
                }
            }
            if (best_partition == 0u) {
                best_partition = subset & (~(subset - 1u));
                best_children_cost = optimal_cost[best_partition] +
                                     optimal_cost[subset ^ best_partition];
            }
            optimal_cost[subset] = subset_bbox_cost[subset] + best_children_cost;
            optimal_partitions[subset] = static_cast<std::uint8_t>(best_partition);
        }
    }
    if (!(optimal_cost[full_mask] < node_cost[root_index] - 1e-6f)) return;

    const auto left_partition = optimal_partitions[full_mask];
    const auto right_partition = static_cast<std::uint8_t>(full_mask ^ left_partition);
    if (left_partition == 0u || right_partition == 0u) return;

    treelet_nodes[0] = root_index;
    int treelet_node_count = 1;
    int next_reusable_node = 0;
    TreeletPartitionEntry stack[2 * kBvhTreeletMaxLeaves];
    int stack_size = 0;
    stack[stack_size++] = { right_partition, 1u, root_index };
    stack[stack_size++] = { left_partition, 0u, root_index };
    while (stack_size > 0) {
        const TreeletPartitionEntry entry = stack[--stack_size];
        const auto partition = entry.partition;
        if (__popc(static_cast<std::uint32_t>(partition)) == 1) {
            const int frontier_index = __ffs(static_cast<unsigned int>(partition)) - 1;
            const int child = frontier_nodes[frontier_index];
            if (entry.child_slot == 0u) left_child[entry.parent_node] = child;
            else right_child[entry.parent_node] = child;
            parent[child] = entry.parent_node;
            continue;
        }
        if (next_reusable_node >= reusable_count) return;
        const int internal = reusable_nodes[next_reusable_node++];
        treelet_nodes[treelet_node_count++] = internal;
        if (entry.child_slot == 0u) left_child[entry.parent_node] = internal;
        else right_child[entry.parent_node] = internal;
        parent[internal] = entry.parent_node;
        leaf_primitive[internal] = -1;
        const auto left_subset = optimal_partitions[partition];
        const auto right_subset = static_cast<std::uint8_t>(partition ^ left_subset);
        if (left_subset == 0u || right_subset == 0u) return;
        stack[stack_size++] = { right_subset, 1u, internal };
        stack[stack_size++] = { left_subset, 0u, internal };
    }

    int post_nodes[2 * kBvhTreeletMaxLeaves];
    std::uint8_t post_states[2 * kBvhTreeletMaxLeaves];
    int post_size = 0;
    post_nodes[post_size] = root_index;
    post_states[post_size++] = 0u;
    while (post_size > 0) {
        const int node = post_nodes[post_size - 1];
        const auto state = post_states[post_size - 1];
        if (state == 0u) {
            post_states[post_size - 1] = 1u;
            const int right = right_child[node];
            if (is_leaf[right] == 0 && node_in_treelet(right, treelet_nodes, treelet_node_count)) {
                post_nodes[post_size] = right;
                post_states[post_size++] = 0u;
            }
            const int left = left_child[node];
            if (is_leaf[left] == 0 && node_in_treelet(left, treelet_nodes, treelet_node_count)) {
                post_nodes[post_size] = left;
                post_states[post_size++] = 0u;
            }
        } else {
            --post_size;
            update_internal_node(node, left_child, right_child, node_bounds, node_cost, inflation);
        }
    }
}

__global__ void init_sequence_kernel(int count, std::int32_t *values) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) values[index] = index;
}

__global__ void compute_morton_codes_kernel(MortonCodeParams params) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= static_cast<int>(params.primitive_bounds.count)) return;
    const BvhBounds3 bounds = load_bounds(primitive, params.primitive_bounds);
    params.morton_codes[primitive] =
        morton_code_3d(mul3(add3(bounds.min, bounds.max), 0.5f), params.scene_bounds);
}

__device__ inline int longest_common_prefix(const std::uint32_t *morton_codes,
                                            const std::int32_t *sorted_primitives,
                                            int primitive_count,
                                            int first,
                                            int second) {
    if (first < 0 || first >= primitive_count || second < 0 || second >= primitive_count) {
        return -1;
    }
    const auto first_code = morton_codes[first];
    const auto second_code = morton_codes[second];
    if (first_code != second_code) return clz_u32(first_code ^ second_code);
    return 32 + clz_u32(static_cast<std::uint32_t>(sorted_primitives[first]) ^
                        static_cast<std::uint32_t>(sorted_primitives[second]));
}

__global__ void build_radix_tree_kernel(RadixTreeParams params) {
    const int node_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node_index >= params.primitive_count - 1) return;
    const int delta_next = longest_common_prefix(params.morton_codes,
                                                  params.sorted_primitives,
                                                  params.primitive_count,
                                                  node_index,
                                                  node_index + 1);
    const int delta_prev = longest_common_prefix(params.morton_codes,
                                                  params.sorted_primitives,
                                                  params.primitive_count,
                                                  node_index,
                                                  node_index - 1);
    const int direction = (delta_next - delta_prev) >= 0 ? 1 : -1;
    const int delta_min = longest_common_prefix(params.morton_codes,
                                                 params.sorted_primitives,
                                                 params.primitive_count,
                                                 node_index,
                                                 node_index - direction);
    int max_length = 2;
    while (longest_common_prefix(params.morton_codes,
                                 params.sorted_primitives,
                                 params.primitive_count,
                                 node_index,
                                 node_index + max_length * direction) > delta_min) {
        max_length *= 2;
    }
    int length = 0;
    int divider = 2;
    for (int step = max_length / divider; step >= 1;) {
        if (longest_common_prefix(params.morton_codes,
                                  params.sorted_primitives,
                                  params.primitive_count,
                                  node_index,
                                  node_index + (length + step) * direction) > delta_min) {
            length += step;
        }
        if (step == 1) break;
        divider *= 2;
        step = max_length / divider;
    }
    const int other = node_index + length * direction;
    const int node_prefix = longest_common_prefix(params.morton_codes,
                                                   params.sorted_primitives,
                                                   params.primitive_count,
                                                   node_index,
                                                   other);
    int split = 0;
    divider = 2;
    for (int step = (length + divider - 1) / divider; step >= 1;) {
        if (longest_common_prefix(params.morton_codes,
                                  params.sorted_primitives,
                                  params.primitive_count,
                                  node_index,
                                  node_index + (split + step) * direction) > node_prefix) {
            split += step;
        }
        if (step == 1) break;
        divider *= 2;
        step = (length + divider - 1) / divider;
    }
    const int direction_min = direction < 0 ? direction : 0;
    const int gamma = node_index + split * direction + direction_min;
    const int range_min = node_index < other ? node_index : other;
    const int range_max = node_index > other ? node_index : other;
    const int leaf_base = params.primitive_count - 1;
    const int left = range_min == gamma ? leaf_base + gamma : gamma;
    const int right = range_max == gamma + 1 ? leaf_base + gamma + 1 : gamma + 1;
    params.left_child[node_index] = left;
    params.right_child[node_index] = right;
    params.parent[left] = node_index;
    params.parent[right] = node_index;
}

__global__ void finalize_leaves_and_bounds_kernel(LeafBoundsFinalizeParams params) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= params.primitive_count) return;
    const int primitive = params.sorted_primitives[leaf_index];
    const int node = params.primitive_count - 1 + leaf_index;
    store_bounds(node, load_bounds(primitive, params.primitive_bounds), params.node_bounds);
    params.leaf_primitive[node] = primitive;
    params.is_leaf[node] = 1;
    params.primitive_leaf_node[primitive] = node;
    int current = params.parent[node];
    while (current >= 0) {
        __threadfence();
        if (atomicAdd(params.merge_counters + current, 1) == 0) return;
        const int left = params.left_child[current];
        const int right = params.right_child[current];
        store_bounds(current,
                     merge_bounds(load_bounds(left, params.node_bounds),
                                  load_bounds(right, params.node_bounds)),
                     params.node_bounds);
        current = params.parent[current];
    }
}

__global__ void initialize_leaf_costs_kernel(LeafCostParams params) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= params.primitive_count) return;
    const int node = params.primitive_count - 1 + leaf_index;
    params.node_cost[node] = bbox_cost_inflated(load_bounds(node, params.node_bounds),
                                                params.inflation);
}

__global__ void initialize_internal_costs_kernel(InternalCostParams params) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= params.primitive_count) return;
    int current = params.parent[params.primitive_count - 1 + leaf_index];
    while (current >= 0) {
        if (atomicAdd(params.arrival_counter + current, 1) == 0) return;
        params.node_cost[current] =
            bbox_cost_inflated(load_bounds(current, params.node_bounds), params.inflation) +
            params.node_cost[params.left_child[current]] +
            params.node_cost[params.right_child[current]];
        __threadfence();
        current = params.parent[current];
    }
}

__global__ void optimize_selected_treelets_kernel(TreeletOptimizeParams params) {
    const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item >= params.selected_count) return;
    treelet_optimize_node(params.selected_nodes[item],
                          params.is_leaf,
                          params.left_child,
                          params.right_child,
                          params.parent,
                          params.node_bounds,
                          params.leaf_primitive,
                          params.node_cost,
                          params.inflation);
}

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

void launch_init_sequence_async(const SequenceInitParams &params) {
    if (params.count == 0) return;
    init_sequence_kernel<<<(params.count + kBlockSize - 1) / kBlockSize,
                           kBlockSize,
                           0,
                           params.stream>>>(params.count, params.values);
}

void launch_compute_morton_codes_async(const MortonCodeParams &params) {
    const int count = static_cast<int>(params.primitive_bounds.count);
    if (count == 0) return;
    compute_morton_codes_kernel<<<(count + kBlockSize - 1) / kBlockSize,
                                  kBlockSize,
                                  0,
                                  params.stream>>>(params);
}

void launch_build_radix_tree_async(const RadixTreeParams &params) {
    const int count = params.primitive_count - 1;
    if (count <= 0) return;
    build_radix_tree_kernel<<<(count + kBlockSize - 1) / kBlockSize,
                              kBlockSize,
                              0,
                              params.stream>>>(params);
}

void launch_finalize_leaves_and_bounds_async(const LeafBoundsFinalizeParams &params) {
    if (params.primitive_count == 0) return;
    finalize_leaves_and_bounds_kernel<<<(params.primitive_count + kBlockSize - 1) / kBlockSize,
                                        kBlockSize,
                                        0,
                                        params.stream>>>(params);
}

void launch_initialize_leaf_costs_async(const LeafCostParams &params) {
    if (params.primitive_count == 0) return;
    initialize_leaf_costs_kernel<<<(params.primitive_count + kBlockSize - 1) / kBlockSize,
                                   kBlockSize,
                                   0,
                                   params.stream>>>(params);
}

void launch_initialize_internal_costs_async(const InternalCostParams &params) {
    if (params.primitive_count == 0) return;
    initialize_internal_costs_kernel<<<(params.primitive_count + kBlockSize - 1) / kBlockSize,
                                       kBlockSize,
                                       0,
                                       params.stream>>>(params);
}

void launch_optimize_selected_treelets_async(const TreeletOptimizeParams &params) {
    if (params.selected_count == 0) return;
    optimize_selected_treelets_kernel<<<(params.selected_count + kBlockSize - 1) / kBlockSize,
                                        kBlockSize,
                                        0,
                                        params.stream>>>(params);
}

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

} // namespace rayd::shared::bvh
