#include "edge_bvh.h"
#include "edge_bvh_config.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>
#include <vector>

namespace rayd {

[[noreturn]] inline void throw_runtime_error_local(const std::string &message) {
    throw std::runtime_error(message);
}

inline void require_local(bool condition, const std::string &message) {
    if (!condition) {
        throw_runtime_error_local(message);
    }
}

namespace {

struct Float3 {
    float x;
    float y;
    float z;

    __host__ __device__ Float3()
        : x(0.f), y(0.f), z(0.f) { }

    __host__ __device__ Float3(float x_, float y_, float z_)
        : x(x_), y(y_), z(z_) { }
};

struct Bounds3 {
    Float3 min;
    Float3 max;

    __host__ __device__ static Bounds3 empty() {
        const float inf = 1e30f;
        return Bounds3{ Float3(inf, inf, inf), Float3(-inf, -inf, -inf) };
    }
};

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) {
        allocate(count);
    }

    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer &operator=(CudaBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }

        count_ = count;
        if (count_ == 0) {
            return;
        }

        const cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_);
        require_local(error == cudaSuccess,
                      std::string("CudaBuffer::allocate(): ") + cudaGetErrorString(error));
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

__host__ __device__ inline Float3 min3(const Float3 &a, const Float3 &b) {
    return Float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline Float3 max3(const Float3 &a, const Float3 &b) {
    return Float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ inline Float3 add3(const Float3 &a, const Float3 &b) {
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Float3 mul3(const Float3 &a, float scale) {
    return Float3(a.x * scale, a.y * scale, a.z * scale);
}

__host__ __device__ inline Bounds3 merge_bounds(const Bounds3 &a, const Bounds3 &b) {
    return Bounds3{ min3(a.min, b.min), max3(a.max, b.max) };
}

struct BoundsUnion {
    __host__ __device__ Bounds3 operator()(const Bounds3 &a, const Bounds3 &b) const {
        return merge_bounds(a, b);
    }
};

__host__ __device__ inline uint32_t expand_bits_10(uint32_t value) {
    value &= 0x000003ffu;
    value = (value | (value << 16)) & 0x030000FFu;
    value = (value | (value << 8)) & 0x0300F00Fu;
    value = (value | (value << 4)) & 0x030C30C3u;
    value = (value | (value << 2)) & 0x09249249u;
    return value;
}

__host__ __device__ inline uint32_t morton_code_3d(const Float3 &point, const Bounds3 &scene_bounds) {
    Float3 normalized(0.5f, 0.5f, 0.5f);
    const Float3 extent(scene_bounds.max.x - scene_bounds.min.x,
                        scene_bounds.max.y - scene_bounds.min.y,
                        scene_bounds.max.z - scene_bounds.min.z);

    if (extent.x > 0.f) {
        normalized.x = (point.x - scene_bounds.min.x) / extent.x;
    }
    if (extent.y > 0.f) {
        normalized.y = (point.y - scene_bounds.min.y) / extent.y;
    }
    if (extent.z > 0.f) {
        normalized.z = (point.z - scene_bounds.min.z) / extent.z;
    }

    normalized.x = fminf(fmaxf(normalized.x, 0.f), 1.f);
    normalized.y = fminf(fmaxf(normalized.y, 0.f), 1.f);
    normalized.z = fminf(fmaxf(normalized.z, 0.f), 1.f);

    constexpr uint32_t scale = (1u << 10) - 1u;
    const uint32_t x = static_cast<uint32_t>(normalized.x * static_cast<float>(scale));
    const uint32_t y = static_cast<uint32_t>(normalized.y * static_cast<float>(scale));
    const uint32_t z = static_cast<uint32_t>(normalized.z * static_cast<float>(scale));
    return (expand_bits_10(x) << 2u) |
           (expand_bits_10(y) << 1u) |
           (expand_bits_10(z) << 0u);
}

__device__ inline int clz_u32(uint32_t value) {
    return value == 0u ? 32 : __clz(value);
}

__device__ inline Bounds3 load_bounds(int node_index,
                                      const float *node_bbox_min_x,
                                      const float *node_bbox_min_y,
                                      const float *node_bbox_min_z,
                                      const float *node_bbox_max_x,
                                      const float *node_bbox_max_y,
                                      const float *node_bbox_max_z) {
    return Bounds3{
        Float3(node_bbox_min_x[node_index], node_bbox_min_y[node_index], node_bbox_min_z[node_index]),
        Float3(node_bbox_max_x[node_index], node_bbox_max_y[node_index], node_bbox_max_z[node_index])
    };
}

__device__ inline void store_bounds(int node_index,
                                    const Bounds3 &bounds,
                                    float *node_bbox_min_x,
                                    float *node_bbox_min_y,
                                    float *node_bbox_min_z,
                                    float *node_bbox_max_x,
                                    float *node_bbox_max_y,
                                    float *node_bbox_max_z) {
    node_bbox_min_x[node_index] = bounds.min.x;
    node_bbox_min_y[node_index] = bounds.min.y;
    node_bbox_min_z[node_index] = bounds.min.z;
    node_bbox_max_x[node_index] = bounds.max.x;
    node_bbox_max_y[node_index] = bounds.max.y;
    node_bbox_max_z[node_index] = bounds.max.z;
}

__device__ inline float bbox_cost_inflated(const Bounds3 &bounds, float inflation) {
    const float dx = fmaxf(bounds.max.x - bounds.min.x, 0.f) + inflation;
    const float dy = fmaxf(bounds.max.y - bounds.min.y, 0.f) + inflation;
    const float dz = fmaxf(bounds.max.z - bounds.min.z, 0.f) + inflation;
    return 2.f * (dx * dy + dx * dz + dy * dz);
}

__device__ inline bool node_in_treelet(int node_index,
                                       const int *treelet_nodes,
                                       int treelet_node_count) {
    for (int index = 0; index < treelet_node_count; ++index) {
        if (treelet_nodes[index] == node_index) {
            return true;
        }
    }
    return false;
}

__device__ inline void update_internal_node(int node_index,
                                            const int *left_child,
                                            const int *right_child,
                                            float *node_bbox_min_x,
                                            float *node_bbox_min_y,
                                            float *node_bbox_min_z,
                                            float *node_bbox_max_x,
                                            float *node_bbox_max_y,
                                            float *node_bbox_max_z,
                                            float *node_cost,
                                            float inflation) {
    const int left_index = left_child[node_index];
    const int right_index = right_child[node_index];
    const Bounds3 left_bounds = load_bounds(left_index,
                                            node_bbox_min_x,
                                            node_bbox_min_y,
                                            node_bbox_min_z,
                                            node_bbox_max_x,
                                            node_bbox_max_y,
                                            node_bbox_max_z);
    const Bounds3 right_bounds = load_bounds(right_index,
                                             node_bbox_min_x,
                                             node_bbox_min_y,
                                             node_bbox_min_z,
                                             node_bbox_max_x,
                                             node_bbox_max_y,
                                             node_bbox_max_z);
    const Bounds3 merged = merge_bounds(left_bounds, right_bounds);
    store_bounds(node_index,
                 merged,
                 node_bbox_min_x,
                 node_bbox_min_y,
                 node_bbox_min_z,
                 node_bbox_max_x,
                 node_bbox_max_y,
                 node_bbox_max_z);
    node_cost[node_index] =
        bbox_cost_inflated(merged, inflation) + node_cost[left_index] + node_cost[right_index];
}

struct TreeletPartitionEntry {
    uint8_t partition;
    uint8_t child_slot;
    int parent_node;
};

__device__ inline void treelet_optimize_node(int root_index,
                                             const int *is_leaf,
                                             int *left_child,
                                             int *right_child,
                                             int *parent,
                                             float *node_bbox_min_x,
                                             float *node_bbox_min_y,
                                             float *node_bbox_min_z,
                                             float *node_bbox_max_x,
                                             float *node_bbox_max_y,
                                             float *node_bbox_max_z,
                                             int *leaf_primitive,
                                             float *node_cost,
                                             float inflation) {
    update_internal_node(root_index,
                         left_child,
                         right_child,
                         node_bbox_min_x,
                         node_bbox_min_y,
                         node_bbox_min_z,
                         node_bbox_max_x,
                         node_bbox_max_y,
                         node_bbox_max_z,
                         node_cost,
                         inflation);

    int frontier_nodes[EdgeBVHTreeletMaxLeaves];
    int reusable_nodes[EdgeBVHTreeletMaxLeaves - 2];
    int treelet_nodes[EdgeBVHTreeletMaxLeaves - 1];
    int frontier_count = 0;
    int reusable_count = 0;

    frontier_nodes[frontier_count++] = left_child[root_index];
    frontier_nodes[frontier_count++] = right_child[root_index];

    while (frontier_count < EdgeBVHTreeletMaxLeaves) {
        int expand_slot = -1;
        float max_cost = -1.f;
        for (int frontier_index = 0; frontier_index < frontier_count; ++frontier_index) {
            const int node_index = frontier_nodes[frontier_index];
            if (is_leaf[node_index] > 0) {
                continue;
            }

            const Bounds3 bounds = load_bounds(node_index,
                                               node_bbox_min_x,
                                               node_bbox_min_y,
                                               node_bbox_min_z,
                                               node_bbox_max_x,
                                               node_bbox_max_y,
                                               node_bbox_max_z);
            const float candidate_cost = bbox_cost_inflated(bounds, inflation);
            if (candidate_cost > max_cost) {
                max_cost = candidate_cost;
                expand_slot = frontier_index;
            }
        }

        if (expand_slot < 0) {
            break;
        }

        const int expanded_node = frontier_nodes[expand_slot];
        reusable_nodes[reusable_count++] = expanded_node;
        frontier_nodes[expand_slot] = left_child[expanded_node];
        frontier_nodes[frontier_count++] = right_child[expanded_node];
    }

    if (frontier_count < 3) {
        return;
    }

    constexpr int MaxTreeletSubsets = 1 << EdgeBVHTreeletMaxLeaves;
    Bounds3 subset_bounds[MaxTreeletSubsets];
    float subset_bbox_cost[MaxTreeletSubsets];
    float optimal_cost[MaxTreeletSubsets];
    uint8_t optimal_partitions[MaxTreeletSubsets];
    const uint32_t full_mask = (1u << frontier_count) - 1u;

    for (uint32_t subset = 1u; subset <= full_mask; ++subset) {
        Bounds3 bounds = Bounds3::empty();
        for (int frontier_index = 0; frontier_index < frontier_count; ++frontier_index) {
            if ((subset & (1u << frontier_index)) == 0u) {
                continue;
            }
            const int node_index = frontier_nodes[frontier_index];
            bounds = merge_bounds(bounds,
                                  load_bounds(node_index,
                                              node_bbox_min_x,
                                              node_bbox_min_y,
                                              node_bbox_min_z,
                                              node_bbox_max_x,
                                              node_bbox_max_y,
                                              node_bbox_max_z));
        }
        subset_bounds[subset] = bounds;
        subset_bbox_cost[subset] = bbox_cost_inflated(bounds, inflation);
    }

    for (int frontier_index = 0; frontier_index < frontier_count; ++frontier_index) {
        const uint32_t subset = 1u << frontier_index;
        optimal_cost[subset] = node_cost[frontier_nodes[frontier_index]];
        optimal_partitions[subset] = 0u;
    }

    for (int subset_size = 2; subset_size <= frontier_count; ++subset_size) {
        for (uint32_t subset = 1u; subset <= full_mask; ++subset) {
            if (__popc(subset) != subset_size) {
                continue;
            }

            float best_children_cost = 1e30f;
            uint32_t best_partition = 0u;
            for (uint32_t left_subset = (subset - 1u) & subset;
                 left_subset > 0u;
                 left_subset = (left_subset - 1u) & subset) {
                const uint32_t right_subset = subset ^ left_subset;
                if (right_subset == 0u || left_subset > right_subset) {
                    continue;
                }

                const float candidate_cost =
                    optimal_cost[left_subset] + optimal_cost[right_subset];
                if (candidate_cost < best_children_cost) {
                    best_children_cost = candidate_cost;
                    best_partition = left_subset;
                }
            }

            if (best_partition == 0u) {
                best_partition = subset & (~(subset - 1u));
                best_children_cost =
                    optimal_cost[best_partition] + optimal_cost[subset ^ best_partition];
            }

            optimal_cost[subset] = subset_bbox_cost[subset] + best_children_cost;
            optimal_partitions[subset] = static_cast<uint8_t>(best_partition);
        }
    }

    if (!(optimal_cost[full_mask] < node_cost[root_index] - 1e-6f)) {
        return;
    }

    const uint8_t left_partition = optimal_partitions[full_mask];
    const uint8_t right_partition = static_cast<uint8_t>(full_mask ^ left_partition);
    if (left_partition == 0u || right_partition == 0u) {
        return;
    }

    treelet_nodes[0] = root_index;
    int treelet_node_count = 1;
    int next_reusable_node = 0;
    TreeletPartitionEntry stack[2 * EdgeBVHTreeletMaxLeaves];
    int stack_size = 0;
    stack[stack_size++] = TreeletPartitionEntry{ right_partition, 1u, root_index };
    stack[stack_size++] = TreeletPartitionEntry{ left_partition, 0u, root_index };

    while (stack_size > 0) {
        const TreeletPartitionEntry entry = stack[--stack_size];
        const uint8_t partition = entry.partition;
        const int partition_size = __popc(static_cast<uint32_t>(partition));
        if (partition_size == 1) {
            const int frontier_index = __ffs(static_cast<unsigned int>(partition)) - 1;
            const int child_node = frontier_nodes[frontier_index];
            if (entry.child_slot == 0u) {
                left_child[entry.parent_node] = child_node;
            } else {
                right_child[entry.parent_node] = child_node;
            }
            parent[child_node] = entry.parent_node;
            continue;
        }

        if (next_reusable_node >= reusable_count) {
            return;
        }

        const int internal_node = reusable_nodes[next_reusable_node++];
        treelet_nodes[treelet_node_count++] = internal_node;
        if (entry.child_slot == 0u) {
            left_child[entry.parent_node] = internal_node;
        } else {
            right_child[entry.parent_node] = internal_node;
        }
        parent[internal_node] = entry.parent_node;
        leaf_primitive[internal_node] = -1;

        const uint8_t left_subset = optimal_partitions[partition];
        const uint8_t right_subset = static_cast<uint8_t>(partition ^ left_subset);
        if (left_subset == 0u || right_subset == 0u) {
            return;
        }

        stack[stack_size++] = TreeletPartitionEntry{ right_subset, 1u, internal_node };
        stack[stack_size++] = TreeletPartitionEntry{ left_subset, 0u, internal_node };
    }

    int post_nodes[2 * EdgeBVHTreeletMaxLeaves];
    uint8_t post_states[2 * EdgeBVHTreeletMaxLeaves];
    int post_size = 0;
    post_nodes[post_size] = root_index;
    post_states[post_size++] = 0u;
    while (post_size > 0) {
        const int node_index = post_nodes[post_size - 1];
        const uint8_t state = post_states[post_size - 1];
        if (state == 0u) {
            post_states[post_size - 1] = 1u;
            const int right_index = right_child[node_index];
            if (is_leaf[right_index] == 0 &&
                node_in_treelet(right_index, treelet_nodes, treelet_node_count)) {
                post_nodes[post_size] = right_index;
                post_states[post_size++] = 0u;
            }
            const int left_index = left_child[node_index];
            if (is_leaf[left_index] == 0 &&
                node_in_treelet(left_index, treelet_nodes, treelet_node_count)) {
                post_nodes[post_size] = left_index;
                post_states[post_size++] = 0u;
            }
        } else {
            --post_size;
            update_internal_node(node_index,
                                 left_child,
                                 right_child,
                                 node_bbox_min_x,
                                 node_bbox_min_y,
                                 node_bbox_min_z,
                                 node_bbox_max_x,
                                 node_bbox_max_y,
                                 node_bbox_max_z,
                                 node_cost,
                                 inflation);
        }
    }
}

__global__ void initialize_leaf_costs_kernel(int primitive_count,
                                             float *node_bbox_min_x,
                                             float *node_bbox_min_y,
                                             float *node_bbox_min_z,
                                             float *node_bbox_max_x,
                                             float *node_bbox_max_y,
                                             float *node_bbox_max_z,
                                             float *node_cost,
                                             float inflation) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= primitive_count) {
        return;
    }

    const int node_index = primitive_count - 1 + leaf_index;
    node_cost[node_index] =
        bbox_cost_inflated(load_bounds(node_index,
                                       node_bbox_min_x,
                                       node_bbox_min_y,
                                       node_bbox_min_z,
                                       node_bbox_max_x,
                                       node_bbox_max_y,
                                       node_bbox_max_z),
                           inflation);
}

__global__ void update_internal_nodes_kernel(int node_count,
                                             const int *node_indices,
                                             const int *left_child,
                                             const int *right_child,
                                             float *node_bbox_min_x,
                                             float *node_bbox_min_y,
                                             float *node_bbox_min_z,
                                             float *node_bbox_max_x,
                                             float *node_bbox_max_y,
                                             float *node_bbox_max_z,
                                             float *node_cost,
                                             float inflation) {
    const int item_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item_index >= node_count) {
        return;
    }

    update_internal_node(node_indices[item_index],
                         left_child,
                         right_child,
                         node_bbox_min_x,
                         node_bbox_min_y,
                         node_bbox_min_z,
                         node_bbox_max_x,
                         node_bbox_max_y,
                         node_bbox_max_z,
                         node_cost,
                         inflation);
}

__global__ void optimize_selected_treelets_kernel(int node_count,
                                                  const int *node_indices,
                                                  const int *is_leaf,
                                                  int *left_child,
                                                  int *right_child,
                                                  int *parent,
                                                  float *node_bbox_min_x,
                                                  float *node_bbox_min_y,
                                                  float *node_bbox_min_z,
                                                  float *node_bbox_max_x,
                                                  float *node_bbox_max_y,
                                                  float *node_bbox_max_z,
                                                  int *leaf_primitive,
                                                  float *node_cost,
                                                  float inflation) {
    const int item_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item_index >= node_count) {
        return;
    }

    treelet_optimize_node(node_indices[item_index],
                          is_leaf,
                          left_child,
                          right_child,
                          parent,
                          node_bbox_min_x,
                          node_bbox_min_y,
                          node_bbox_min_z,
                          node_bbox_max_x,
                          node_bbox_max_y,
                          node_bbox_max_z,
                          leaf_primitive,
                          node_cost,
                          inflation);
}

__global__ void compute_primitive_bounds_kernel(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float *primitive_bbox_min_x,
    float *primitive_bbox_min_y,
    float *primitive_bbox_min_z,
    float *primitive_bbox_max_x,
    float *primitive_bbox_max_y,
    float *primitive_bbox_max_z,
    Bounds3 *primitive_bounds) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= primitive_count) {
        return;
    }

    const Float3 p0(edge_p0_x[primitive], edge_p0_y[primitive], edge_p0_z[primitive]);
    const Float3 p1(p0.x + edge_e1_x[primitive],
                    p0.y + edge_e1_y[primitive],
                    p0.z + edge_e1_z[primitive]);
    const Float3 bbox_min = min3(p0, p1);
    const Float3 bbox_max = max3(p0, p1);

    primitive_bbox_min_x[primitive] = bbox_min.x;
    primitive_bbox_min_y[primitive] = bbox_min.y;
    primitive_bbox_min_z[primitive] = bbox_min.z;
    primitive_bbox_max_x[primitive] = bbox_max.x;
    primitive_bbox_max_y[primitive] = bbox_max.y;
    primitive_bbox_max_z[primitive] = bbox_max.z;
    primitive_bounds[primitive] = Bounds3{ bbox_min, bbox_max };
}

__global__ void init_sequence_kernel(int count, int *values) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    values[index] = index;
}

__global__ void compute_morton_codes_kernel(
    int primitive_count,
    Bounds3 scene_bounds,
    const float *primitive_bbox_min_x,
    const float *primitive_bbox_min_y,
    const float *primitive_bbox_min_z,
    const float *primitive_bbox_max_x,
    const float *primitive_bbox_max_y,
    const float *primitive_bbox_max_z,
    uint32_t *morton_codes) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= primitive_count) {
        return;
    }

    const Float3 bbox_min(primitive_bbox_min_x[primitive],
                          primitive_bbox_min_y[primitive],
                          primitive_bbox_min_z[primitive]);
    const Float3 bbox_max(primitive_bbox_max_x[primitive],
                          primitive_bbox_max_y[primitive],
                          primitive_bbox_max_z[primitive]);
    morton_codes[primitive] = morton_code_3d(mul3(add3(bbox_min, bbox_max), 0.5f), scene_bounds);
}

__device__ inline int longest_common_prefix(const uint32_t *morton_codes,
                                            const int *sorted_primitives,
                                            int primitive_count,
                                            int first,
                                            int second) {
    if (first < 0 || first >= primitive_count || second < 0 || second >= primitive_count) {
        return -1;
    }

    const uint32_t code_first = morton_codes[first];
    const uint32_t code_second = morton_codes[second];
    if (code_first != code_second) {
        return clz_u32(code_first ^ code_second);
    }

    const uint32_t primitive_first = static_cast<uint32_t>(sorted_primitives[first]);
    const uint32_t primitive_second = static_cast<uint32_t>(sorted_primitives[second]);
    return 32 + clz_u32(primitive_first ^ primitive_second);
}

__global__ void build_radix_tree_kernel(
    int primitive_count,
    const uint32_t *morton_codes,
    const int *sorted_primitives,
    int *left_child,
    int *right_child,
    int *parent) {
    const int node_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node_index >= primitive_count - 1) {
        return;
    }

    const int delta_next =
        longest_common_prefix(morton_codes, sorted_primitives, primitive_count, node_index, node_index + 1);
    const int delta_prev =
        longest_common_prefix(morton_codes, sorted_primitives, primitive_count, node_index, node_index - 1);
    const int direction = (delta_next - delta_prev) >= 0 ? 1 : -1;

    const int delta_min =
        longest_common_prefix(morton_codes, sorted_primitives, primitive_count, node_index, node_index - direction);
    int max_length = 2;
    while (longest_common_prefix(morton_codes,
                                 sorted_primitives,
                                 primitive_count,
                                 node_index,
                                 node_index + max_length * direction) > delta_min) {
        max_length *= 2;
    }

    int length = 0;
    int divider = 2;
    for (int step = max_length / divider; step >= 1;) {
        if (longest_common_prefix(morton_codes,
                                  sorted_primitives,
                                  primitive_count,
                                  node_index,
                                  node_index + (length + step) * direction) > delta_min) {
            length += step;
        }
        if (step == 1) {
            break;
        }
        divider *= 2;
        step = max_length / divider;
    }

    const int other = node_index + length * direction;
    const int node_prefix =
        longest_common_prefix(morton_codes, sorted_primitives, primitive_count, node_index, other);

    int split = 0;
    divider = 2;
    for (int step = (length + (divider - 1)) / divider; step >= 1;) {
        if (longest_common_prefix(morton_codes,
                                  sorted_primitives,
                                  primitive_count,
                                  node_index,
                                  node_index + (split + step) * direction) > node_prefix) {
            split += step;
        }
        if (step == 1) {
            break;
        }
        divider *= 2;
        step = (length + (divider - 1)) / divider;
    }

    const int direction_min = direction < 0 ? direction : 0;
    const int gamma = node_index + split * direction + direction_min;
    const int range_min = node_index < other ? node_index : other;
    const int range_max = node_index > other ? node_index : other;
    const int leaf_base = primitive_count - 1;
    const int left_index = range_min == gamma ? leaf_base + gamma : gamma;
    const int right_index = range_max == gamma + 1 ? leaf_base + gamma + 1 : gamma + 1;

    left_child[node_index] = left_index;
    right_child[node_index] = right_index;
    parent[left_index] = node_index;
    parent[right_index] = node_index;
}

__global__ void finalize_leaf_nodes_kernel(
    int primitive_count,
    const int *sorted_primitives,
    const float *primitive_bbox_min_x,
    const float *primitive_bbox_min_y,
    const float *primitive_bbox_min_z,
    const float *primitive_bbox_max_x,
    const float *primitive_bbox_max_y,
    const float *primitive_bbox_max_z,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z,
    int *leaf_primitive,
    int *is_leaf,
    int *primitive_leaf_node) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= primitive_count) {
        return;
    }

    const int primitive = sorted_primitives[leaf_index];
    const int node_index = primitive_count - 1 + leaf_index;

    node_bbox_min_x[node_index] = primitive_bbox_min_x[primitive];
    node_bbox_min_y[node_index] = primitive_bbox_min_y[primitive];
    node_bbox_min_z[node_index] = primitive_bbox_min_z[primitive];
    node_bbox_max_x[node_index] = primitive_bbox_max_x[primitive];
    node_bbox_max_y[node_index] = primitive_bbox_max_y[primitive];
    node_bbox_max_z[node_index] = primitive_bbox_max_z[primitive];
    leaf_primitive[node_index] = primitive;
    is_leaf[node_index] = 1;
    primitive_leaf_node[primitive] = node_index;
}

__global__ void merge_internal_bounds_kernel(int node_count,
                                             const int *node_indices,
                                             const int *left_child,
                                             const int *right_child,
                                             float *node_bbox_min_x,
                                             float *node_bbox_min_y,
                                             float *node_bbox_min_z,
                                             float *node_bbox_max_x,
                                             float *node_bbox_max_y,
                                             float *node_bbox_max_z) {
    const int item_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item_index >= node_count) {
        return;
    }

    const int node_index = node_indices[item_index];
    const int left = left_child[node_index];
    const int right = right_child[node_index];
    node_bbox_min_x[node_index] = fminf(node_bbox_min_x[left], node_bbox_min_x[right]);
    node_bbox_min_y[node_index] = fminf(node_bbox_min_y[left], node_bbox_min_y[right]);
    node_bbox_min_z[node_index] = fminf(node_bbox_min_z[left], node_bbox_min_z[right]);
    node_bbox_max_x[node_index] = fmaxf(node_bbox_max_x[left], node_bbox_max_x[right]);
    node_bbox_max_y[node_index] = fmaxf(node_bbox_max_y[left], node_bbox_max_y[right]);
    node_bbox_max_z[node_index] = fmaxf(node_bbox_max_z[left], node_bbox_max_z[right]);
}

__global__ void finalize_leaves_and_bounds_kernel(
    int primitive_count,
    const int *sorted_primitives,
    const int *parent,
    const float *primitive_bbox_min_x,
    const float *primitive_bbox_min_y,
    const float *primitive_bbox_min_z,
    const float *primitive_bbox_max_x,
    const float *primitive_bbox_max_y,
    const float *primitive_bbox_max_z,
    const int *left_child,
    const int *right_child,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z,
    int *leaf_primitive,
    int *is_leaf,
    int *primitive_leaf_node,
    int *merge_counters) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= primitive_count) {
        return;
    }

    const int primitive = sorted_primitives[leaf_index];
    const int node_index = primitive_count - 1 + leaf_index;

    node_bbox_min_x[node_index] = primitive_bbox_min_x[primitive];
    node_bbox_min_y[node_index] = primitive_bbox_min_y[primitive];
    node_bbox_min_z[node_index] = primitive_bbox_min_z[primitive];
    node_bbox_max_x[node_index] = primitive_bbox_max_x[primitive];
    node_bbox_max_y[node_index] = primitive_bbox_max_y[primitive];
    node_bbox_max_z[node_index] = primitive_bbox_max_z[primitive];
    leaf_primitive[node_index] = primitive;
    is_leaf[node_index] = 1;
    primitive_leaf_node[primitive] = node_index;

    int current = parent[node_index];
    while (current >= 0) {
        if (atomicAdd(merge_counters + current, 1) == 0) {
            return;
        }

        const int left = left_child[current];
        const int right = right_child[current];
        node_bbox_min_x[current] = fminf(node_bbox_min_x[left], node_bbox_min_x[right]);
        node_bbox_min_y[current] = fminf(node_bbox_min_y[left], node_bbox_min_y[right]);
        node_bbox_min_z[current] = fminf(node_bbox_min_z[left], node_bbox_min_z[right]);
        node_bbox_max_x[current] = fmaxf(node_bbox_max_x[left], node_bbox_max_x[right]);
        node_bbox_max_y[current] = fmaxf(node_bbox_max_y[left], node_bbox_max_y[right]);
        node_bbox_max_z[current] = fmaxf(node_bbox_max_z[left], node_bbox_max_z[right]);
        current = parent[current];
    }
}

void check_cuda_call(cudaError_t error, const char *message) {
    require_local(error == cudaSuccess,
                  std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

void synchronize_cuda(const char *message) {
    check_cuda_call(cudaDeviceSynchronize(), message);
}

class CudaStreamHandle {
public:
    explicit CudaStreamHandle(unsigned flags = cudaStreamNonBlocking) {
        check_cuda_call(cudaStreamCreateWithFlags(&stream_, flags),
                        "build_edge_lbvh_gpu(): failed to create CUDA stream");
    }

    ~CudaStreamHandle() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStreamHandle(const CudaStreamHandle &) = delete;
    CudaStreamHandle &operator=(const CudaStreamHandle &) = delete;

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEventHandle {
public:
    explicit CudaEventHandle(unsigned flags = cudaEventDisableTiming) {
        check_cuda_call(cudaEventCreateWithFlags(&event_, flags),
                        "build_edge_lbvh_gpu(): failed to create CUDA event");
    }

    ~CudaEventHandle() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEventHandle(const CudaEventHandle &) = delete;
    CudaEventHandle &operator=(const CudaEventHandle &) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

void memset_int_async(int *ptr,
                      int value,
                      size_t count,
                      cudaStream_t stream,
                      const char *message) {
    if (count == 0) {
        return;
    }

    const unsigned char byte_value = static_cast<unsigned char>(value & 0xff);
    check_cuda_call(cudaMemsetAsync(ptr, byte_value, count * sizeof(int), stream), message);
}

std::vector<int> copy_int_buffer_to_host(const int *device_ptr, size_t count, const char *message) {
    std::vector<int> result(count);
    if (count == 0) {
        return result;
    }

    check_cuda_call(cudaMemcpy(result.data(),
                               device_ptr,
                               count * sizeof(int),
                               cudaMemcpyDeviceToHost),
                    message);
    return result;
}

int compute_subtree_leaf_count_host(int node_index,
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

    count =
        compute_subtree_leaf_count_host(left_child[static_cast<size_t>(node_index)],
                                        left_child,
                                        right_child,
                                        is_leaf,
                                        subtree_leaf_counts) +
        compute_subtree_leaf_count_host(right_child[static_cast<size_t>(node_index)],
                                        left_child,
                                        right_child,
                                        is_leaf,
                                        subtree_leaf_counts);
    return count;
}

int compute_node_height_host(int node_index,
                             const std::vector<int> &left_child,
                             const std::vector<int> &right_child,
                             const std::vector<int> &is_leaf,
                             std::vector<int> &node_heights) {
    int &height = node_heights[static_cast<size_t>(node_index)];
    if (height >= 0) {
        return height;
    }

    if (is_leaf[static_cast<size_t>(node_index)] > 0) {
        height = 0;
        return height;
    }

    height = 1 + std::max(compute_node_height_host(left_child[static_cast<size_t>(node_index)],
                                                   left_child,
                                                   right_child,
                                                   is_leaf,
                                                   node_heights),
                          compute_node_height_host(right_child[static_cast<size_t>(node_index)],
                                                   left_child,
                                                   right_child,
                                                   is_leaf,
                                                   node_heights));
    return height;
}

struct FlatNodeLevels {
    std::vector<int> nodes;
    std::vector<int> level_offsets;
};

struct HostTreeLevels {
    std::vector<int> node_heights;
    std::vector<std::vector<int>> levels;
    int max_height = 0;
};

FlatNodeLevels flatten_node_levels(const std::vector<std::vector<int>> &levels) {
    FlatNodeLevels result;
    result.level_offsets.reserve(levels.size() + 1);
    result.level_offsets.push_back(0);
    for (const std::vector<int> &level : levels) {
        result.nodes.insert(result.nodes.end(), level.begin(), level.end());
        result.level_offsets.push_back(static_cast<int>(result.nodes.size()));
    }
    return result;
}

HostTreeLevels build_host_tree_levels_from_topology(const std::vector<int> &left_child,
                                                    const std::vector<int> &right_child,
                                                    const std::vector<int> &is_leaf) {
    const int node_count = static_cast<int>(is_leaf.size());

    HostTreeLevels result;
    result.node_heights.assign(static_cast<size_t>(node_count), -1);
    result.levels.resize(1);
    if (node_count == 0) {
        return result;
    }

    for (int node_index = 0; node_index < node_count; ++node_index) {
        if (is_leaf[static_cast<size_t>(node_index)] > 0) {
            result.node_heights[static_cast<size_t>(node_index)] = 0;
        }
    }

    for (int node_index = 0; node_index < node_count; ++node_index) {
        if (is_leaf[static_cast<size_t>(node_index)] > 0) {
            continue;
        }

        const int height = compute_node_height_host(node_index,
                                                    left_child,
                                                    right_child,
                                                    is_leaf,
                                                    result.node_heights);
        if (static_cast<int>(result.levels.size()) <= height) {
            result.levels.resize(static_cast<size_t>(height + 1));
        }
        result.levels[static_cast<size_t>(height)].push_back(node_index);
        result.max_height = std::max(result.max_height, height);
    }

    return result;
}

} // namespace

void build_edge_bvh_gpu(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float *primitive_bbox_min_x,
    float *primitive_bbox_min_y,
    float *primitive_bbox_min_z,
    float *primitive_bbox_max_x,
    float *primitive_bbox_max_y,
    float *primitive_bbox_max_z,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z,
    int *left_child,
    int *right_child,
    int *leaf_primitive,
    int *is_leaf,
    int *primitive_leaf_node) {
    require_local(primitive_count > 0, "build_edge_lbvh_gpu(): primitive_count must be positive.");

    try {
        const int node_count = std::max(2 * primitive_count - 1, 1);
        const int block_size = 256;
        const int primitive_blocks = (primitive_count + block_size - 1) / block_size;
        const int internal_count = std::max(primitive_count - 1, 0);
        const int internal_blocks = (internal_count + block_size - 1) / block_size;
        const EdgeBVHBuildStreamMode build_stream_mode = active_edge_bvh_build_stream_mode();
        const EdgeBVHPostBuildStrategy post_build_strategy = active_edge_bvh_post_build_strategy();
        const EdgeBVHFinalizeMode finalize_mode = active_edge_bvh_finalize_mode();
        const EdgeBVHTreeletScheduleMode treelet_schedule_mode =
            active_edge_bvh_treelet_schedule_mode();
        const bool overlap_build_streams =
            build_stream_mode == EdgeBVHBuildStreamMode::Overlap;
        const bool treelet_enabled =
            post_build_strategy == EdgeBVHPostBuildStrategy::GpuTreelet &&
            primitive_count >= EdgeBVHTreeletMinPrimitives &&
            internal_count > 0;

        CudaBuffer<Bounds3> primitive_bounds(static_cast<size_t>(primitive_count));
        CudaBuffer<Bounds3> reduced_bounds(1);
        CudaBuffer<uint32_t> morton_codes_in(static_cast<size_t>(primitive_count));
        CudaBuffer<uint32_t> morton_codes_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> primitive_indices_in(static_cast<size_t>(primitive_count));
        CudaBuffer<int> primitive_indices_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> parent(static_cast<size_t>(node_count));
        CudaBuffer<int> merge_counters(static_cast<size_t>(std::max(internal_count, 1)));
        CudaBuffer<float> node_costs(static_cast<size_t>(node_count));
        CudaStreamHandle bounds_stream_handle;
        CudaStreamHandle sequence_stream_handle;
        CudaEventHandle sequence_ready_event;
        const cudaStream_t bounds_stream = bounds_stream_handle.get();
        const cudaStream_t sequence_stream =
            overlap_build_streams ? sequence_stream_handle.get() : bounds_stream;
        std::vector<int> host_left_child;
        std::vector<int> host_right_child;
        std::vector<int> host_is_leaf;
        std::vector<int> subtree_leaf_counts;
        std::vector<int> node_heights;
        std::vector<std::vector<int>> host_level_groups;
        FlatNodeLevels internal_level_schedule;
        int max_height = 0;

        compute_primitive_bounds_kernel<<<primitive_blocks, block_size, 0, bounds_stream>>>(
            primitive_count,
            edge_p0_x,
            edge_p0_y,
            edge_p0_z,
            edge_e1_x,
            edge_e1_y,
            edge_e1_z,
            primitive_bbox_min_x,
            primitive_bbox_min_y,
            primitive_bbox_min_z,
            primitive_bbox_max_x,
            primitive_bbox_max_y,
            primitive_bbox_max_z,
            primitive_bounds.get());
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch primitive-bounds kernel");

        size_t reduce_temp_size = 0;
        check_cuda_call(
            cub::DeviceReduce::Reduce(nullptr,
                                      reduce_temp_size,
                                      primitive_bounds.get(),
                                      reduced_bounds.get(),
                                      primitive_count,
                                      BoundsUnion(),
                                      Bounds3::empty(),
                                      bounds_stream),
            "build_edge_lbvh_gpu(): failed to size scene-bound reduction");
        CudaBuffer<char> reduce_temp(reduce_temp_size);
        check_cuda_call(
            cub::DeviceReduce::Reduce(reduce_temp.get(),
                                      reduce_temp_size,
                                      primitive_bounds.get(),
                                      reduced_bounds.get(),
                                      primitive_count,
                                      BoundsUnion(),
                                      Bounds3::empty(),
                                      bounds_stream),
            "build_edge_lbvh_gpu(): failed to reduce scene bounds");
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch scene-bound reduction");

        Bounds3 scene_bounds = Bounds3::empty();
        check_cuda_call(cudaMemcpyAsync(&scene_bounds,
                                        reduced_bounds.get(),
                                        sizeof(Bounds3),
                                        cudaMemcpyDeviceToHost,
                                        bounds_stream),
                        "build_edge_lbvh_gpu(): failed to copy scene bounds");
        init_sequence_kernel<<<primitive_blocks, block_size, 0, sequence_stream>>>(
            primitive_count, primitive_indices_in.get());
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch primitive-index initialization");
        if (overlap_build_streams) {
            check_cuda_call(cudaEventRecord(sequence_ready_event.get(), sequence_stream),
                            "build_edge_lbvh_gpu(): failed to record sequence-ready event");
        }
        check_cuda_call(cudaStreamSynchronize(bounds_stream),
                        "build_edge_lbvh_gpu(): failed to reduce scene bounds");

        compute_morton_codes_kernel<<<primitive_blocks, block_size, 0, bounds_stream>>>(
            primitive_count,
            scene_bounds,
            primitive_bbox_min_x,
            primitive_bbox_min_y,
            primitive_bbox_min_z,
            primitive_bbox_max_x,
            primitive_bbox_max_y,
            primitive_bbox_max_z,
            morton_codes_in.get());
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch Morton-code kernel");
        if (overlap_build_streams) {
            check_cuda_call(cudaStreamWaitEvent(bounds_stream, sequence_ready_event.get(), 0),
                            "build_edge_lbvh_gpu(): failed to join primitive-index stream");
        }

        size_t sort_temp_size = 0;
        check_cuda_call(
            cub::DeviceRadixSort::SortPairs(nullptr,
                                            sort_temp_size,
                                            morton_codes_in.get(),
                                            morton_codes_out.get(),
                                            primitive_indices_in.get(),
                                            primitive_indices_out.get(),
                                            primitive_count,
                                            0,
                                            32,
                                            bounds_stream),
            "build_edge_lbvh_gpu(): failed to size radix sort");
        CudaBuffer<char> sort_temp(sort_temp_size);
        check_cuda_call(
            cub::DeviceRadixSort::SortPairs(sort_temp.get(),
                                            sort_temp_size,
                                            morton_codes_in.get(),
                                            morton_codes_out.get(),
                                            primitive_indices_in.get(),
                                            primitive_indices_out.get(),
                                            primitive_count,
                                            0,
                                            32,
                                            bounds_stream),
            "build_edge_lbvh_gpu(): failed to sort Morton codes");
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch Morton sort");

        memset_int_async(left_child,
                         -1,
                         static_cast<size_t>(node_count),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init left_child");
        memset_int_async(right_child,
                         -1,
                         static_cast<size_t>(node_count),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init right_child");
        memset_int_async(leaf_primitive,
                         -1,
                         static_cast<size_t>(node_count),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init leaf_primitive");
        memset_int_async(is_leaf,
                         0,
                         static_cast<size_t>(node_count),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init is_leaf");
        memset_int_async(primitive_leaf_node,
                         -1,
                         static_cast<size_t>(primitive_count),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init primitive_leaf_node");
        memset_int_async(parent.get(),
                         -1,
                         static_cast<size_t>(node_count),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init parent");
        if (finalize_mode == EdgeBVHFinalizeMode::Atomic) {
            memset_int_async(merge_counters.get(),
                             0,
                             static_cast<size_t>(std::max(internal_count, 1)),
                             bounds_stream,
                             "build_edge_lbvh_gpu(): failed to init merge counters");
        }

        if (internal_count > 0) {
            build_radix_tree_kernel<<<internal_blocks, block_size, 0, bounds_stream>>>(
                primitive_count,
                morton_codes_out.get(),
                primitive_indices_out.get(),
                left_child,
                right_child,
                parent.get());
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch radix-tree kernel");
        }

        if (internal_count > 0 &&
            (finalize_mode == EdgeBVHFinalizeMode::LevelByLevel || treelet_enabled)) {
            check_cuda_call(cudaStreamSynchronize(bounds_stream),
                            "build_edge_lbvh_gpu(): failed to prepare host topology");
            host_left_child = copy_int_buffer_to_host(left_child,
                                                      static_cast<size_t>(node_count),
                                                      "build_edge_lbvh_gpu(): failed to copy left_child");
            host_right_child = copy_int_buffer_to_host(right_child,
                                                       static_cast<size_t>(node_count),
                                                       "build_edge_lbvh_gpu(): failed to copy right_child");
            host_is_leaf.assign(static_cast<size_t>(node_count), 0);
            for (int leaf_index = internal_count; leaf_index < node_count; ++leaf_index) {
                host_is_leaf[static_cast<size_t>(leaf_index)] = 1;
            }
            const HostTreeLevels host_levels = build_host_tree_levels_from_topology(host_left_child,
                                                                                    host_right_child,
                                                                                    host_is_leaf);
            node_heights = host_levels.node_heights;
            host_level_groups = host_levels.levels;
            max_height = host_levels.max_height;
            size_t scheduled_internal_count = 0;
            for (size_t level_index = 1; level_index < host_level_groups.size(); ++level_index) {
                scheduled_internal_count += host_level_groups[level_index].size();
            }
            require_local(static_cast<int>(scheduled_internal_count) == internal_count,
                          "build_edge_lbvh_gpu(): failed to levelize every internal node.");
            if (finalize_mode == EdgeBVHFinalizeMode::LevelByLevel) {
                internal_level_schedule = flatten_node_levels(host_level_groups);
            }
            if (treelet_enabled) {
                subtree_leaf_counts.assign(static_cast<size_t>(node_count), 1);
                for (int height = 1; height <= max_height; ++height) {
                    for (int node_index : host_level_groups[static_cast<size_t>(height)]) {
                        subtree_leaf_counts[static_cast<size_t>(node_index)] =
                            subtree_leaf_counts[static_cast<size_t>(host_left_child[static_cast<size_t>(node_index)])] +
                            subtree_leaf_counts[static_cast<size_t>(host_right_child[static_cast<size_t>(node_index)])];
                    }
                }
            }
        }

        if (finalize_mode == EdgeBVHFinalizeMode::LevelByLevel && internal_count > 0) {
            finalize_leaf_nodes_kernel<<<primitive_blocks, block_size, 0, bounds_stream>>>(
                primitive_count,
                primitive_indices_out.get(),
                primitive_bbox_min_x,
                primitive_bbox_min_y,
                primitive_bbox_min_z,
                primitive_bbox_max_x,
                primitive_bbox_max_y,
                primitive_bbox_max_z,
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                leaf_primitive,
                is_leaf,
                primitive_leaf_node);
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch leaf-finalization kernel");
            CudaBuffer<int> internal_nodes_device(internal_level_schedule.nodes.size());
            if (!internal_level_schedule.nodes.empty()) {
                check_cuda_call(cudaMemcpy(internal_nodes_device.get(),
                                           internal_level_schedule.nodes.data(),
                                           internal_level_schedule.nodes.size() * sizeof(int),
                                           cudaMemcpyHostToDevice),
                                "build_edge_lbvh_gpu(): failed to upload internal-node levels");
            }
            for (int height = 1; height <= max_height; ++height) {
                const int level_start =
                    internal_level_schedule.level_offsets[static_cast<size_t>(height)];
                const int level_end =
                    internal_level_schedule.level_offsets[static_cast<size_t>(height + 1)];
                const int level_count = level_end - level_start;
                if (level_count == 0) {
                    continue;
                }
                const int level_blocks = (level_count + block_size - 1) / block_size;
                merge_internal_bounds_kernel<<<level_blocks, block_size, 0, bounds_stream>>>(
                    level_count,
                    internal_nodes_device.get() + level_start,
                    left_child,
                    right_child,
                    node_bbox_min_x,
                    node_bbox_min_y,
                    node_bbox_min_z,
                    node_bbox_max_x,
                    node_bbox_max_y,
                    node_bbox_max_z);
                check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch internal-node merge");
            }
        } else if (finalize_mode == EdgeBVHFinalizeMode::LevelByLevel) {
            finalize_leaf_nodes_kernel<<<primitive_blocks, block_size, 0, bounds_stream>>>(
                primitive_count,
                primitive_indices_out.get(),
                primitive_bbox_min_x,
                primitive_bbox_min_y,
                primitive_bbox_min_z,
                primitive_bbox_max_x,
                primitive_bbox_max_y,
                primitive_bbox_max_z,
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                leaf_primitive,
                is_leaf,
                primitive_leaf_node);
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch leaf-finalization kernel");
        } else if (finalize_mode == EdgeBVHFinalizeMode::Atomic) {
            finalize_leaves_and_bounds_kernel<<<primitive_blocks, block_size, 0, bounds_stream>>>(
                primitive_count,
                primitive_indices_out.get(),
                parent.get(),
                primitive_bbox_min_x,
                primitive_bbox_min_y,
                primitive_bbox_min_z,
                primitive_bbox_max_x,
                primitive_bbox_max_y,
                primitive_bbox_max_z,
                left_child,
                right_child,
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                leaf_primitive,
                is_leaf,
                primitive_leaf_node,
                merge_counters.get());
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch bounds-finalization kernel");
        }

        if (treelet_enabled) {
            check_cuda_call(cudaStreamSynchronize(bounds_stream),
                            "build_edge_lbvh_gpu(): failed to finalize node bounds");
            std::vector<std::vector<int>> recompute_levels(static_cast<size_t>(max_height + 1));
            std::vector<std::vector<int>> optimize_levels(static_cast<size_t>(max_height + 1));
            for (int node_index = 0; node_index < internal_count; ++node_index) {
                const int height = node_heights[static_cast<size_t>(node_index)];
                if (subtree_leaf_counts[static_cast<size_t>(node_index)] >=
                    EdgeBVHTreeletMinSubtreeLeaves) {
                    optimize_levels[static_cast<size_t>(height)].push_back(node_index);
                } else {
                    recompute_levels[static_cast<size_t>(height)].push_back(node_index);
                }
            }
            const FlatNodeLevels recompute_schedule = flatten_node_levels(recompute_levels);
            const FlatNodeLevels optimize_schedule = flatten_node_levels(optimize_levels);
            CudaBuffer<int> recompute_nodes_device(recompute_schedule.nodes.size());
            CudaBuffer<int> optimize_nodes_device(optimize_schedule.nodes.size());
            if (treelet_schedule_mode == EdgeBVHTreeletScheduleMode::FlatLevels) {
                if (!recompute_schedule.nodes.empty()) {
                    check_cuda_call(cudaMemcpy(recompute_nodes_device.get(),
                                               recompute_schedule.nodes.data(),
                                               recompute_schedule.nodes.size() * sizeof(int),
                                               cudaMemcpyHostToDevice),
                                    "build_edge_lbvh_gpu(): failed to upload recompute schedule");
                }
                if (!optimize_schedule.nodes.empty()) {
                    check_cuda_call(cudaMemcpy(optimize_nodes_device.get(),
                                               optimize_schedule.nodes.data(),
                                               optimize_schedule.nodes.size() * sizeof(int),
                                               cudaMemcpyHostToDevice),
                                    "build_edge_lbvh_gpu(): failed to upload treelet schedule");
                }
            }

            const float scene_scale =
                fmaxf(scene_bounds.max.x - scene_bounds.min.x,
                      fmaxf(scene_bounds.max.y - scene_bounds.min.y,
                            scene_bounds.max.z - scene_bounds.min.z));
            const float inflation =
                fmaxf(scene_scale * EdgeBVHTreeletCostInflationRatio, 1e-6f);
            initialize_leaf_costs_kernel<<<primitive_blocks, block_size, 0, bounds_stream>>>(
                primitive_count,
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                node_costs.get(),
                inflation);
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch leaf-cost initialization");

            for (int height = 1; height <= max_height; ++height) {
                const int recompute_start =
                    recompute_schedule.level_offsets[static_cast<size_t>(height)];
                const int recompute_end =
                    recompute_schedule.level_offsets[static_cast<size_t>(height + 1)];
                const int recompute_count = recompute_end - recompute_start;
                if (recompute_count > 0) {
                    const int level_blocks = (recompute_count + block_size - 1) / block_size;
                    if (treelet_schedule_mode == EdgeBVHTreeletScheduleMode::FlatLevels) {
                        update_internal_nodes_kernel<<<level_blocks, block_size, 0, bounds_stream>>>(
                            recompute_count,
                            recompute_nodes_device.get() + recompute_start,
                            left_child,
                            right_child,
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            node_costs.get(),
                            inflation);
                    } else {
                        const std::vector<int> &recompute_nodes =
                            recompute_levels[static_cast<size_t>(height)];
                        CudaBuffer<int> device_nodes(recompute_nodes.size());
                        check_cuda_call(cudaMemcpy(device_nodes.get(),
                                                   recompute_nodes.data(),
                                                   recompute_nodes.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice),
                                        "build_edge_lbvh_gpu(): failed to upload recompute nodes");
                        update_internal_nodes_kernel<<<level_blocks, block_size, 0, bounds_stream>>>(
                            recompute_count,
                            device_nodes.get(),
                            left_child,
                            right_child,
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            node_costs.get(),
                            inflation);
                    }
                    check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch internal-node recompute");
                }

                const int optimize_start =
                    optimize_schedule.level_offsets[static_cast<size_t>(height)];
                const int optimize_end =
                    optimize_schedule.level_offsets[static_cast<size_t>(height + 1)];
                const int optimize_count = optimize_end - optimize_start;
                if (optimize_count > 0) {
                    const int level_blocks = (optimize_count + block_size - 1) / block_size;
                    if (treelet_schedule_mode == EdgeBVHTreeletScheduleMode::FlatLevels) {
                        optimize_selected_treelets_kernel<<<level_blocks, block_size, 0, bounds_stream>>>(
                            optimize_count,
                            optimize_nodes_device.get() + optimize_start,
                            is_leaf,
                            left_child,
                            right_child,
                            parent.get(),
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            leaf_primitive,
                            node_costs.get(),
                            inflation);
                    } else {
                        const std::vector<int> &optimize_nodes =
                            optimize_levels[static_cast<size_t>(height)];
                        CudaBuffer<int> device_nodes(optimize_nodes.size());
                        check_cuda_call(cudaMemcpy(device_nodes.get(),
                                                   optimize_nodes.data(),
                                                   optimize_nodes.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice),
                                        "build_edge_lbvh_gpu(): failed to upload treelet roots");
                        optimize_selected_treelets_kernel<<<level_blocks, block_size, 0, bounds_stream>>>(
                            optimize_count,
                            device_nodes.get(),
                            is_leaf,
                            left_child,
                            right_child,
                            parent.get(),
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            leaf_primitive,
                            node_costs.get(),
                            inflation);
                    }
                    check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch GPU treelet optimization");
                }
            }
        }

        check_cuda_call(cudaStreamSynchronize(bounds_stream),
                        "build_edge_lbvh_gpu(): failed to complete build");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("build_edge_lbvh_gpu(): ") + e.what());
    }
}

__global__ void emit_compacted_nodes_kernel(int node_count,
                                            const int *new_to_old,
                                            const int *compacted_left_child,
                                            const int *compacted_right_child,
                                            const float *raw_node_bbox_min_x,
                                            const float *raw_node_bbox_min_y,
                                            const float *raw_node_bbox_min_z,
                                            const float *raw_node_bbox_max_x,
                                            const float *raw_node_bbox_max_y,
                                            const float *raw_node_bbox_max_z,
                                            float *out_node_bbox_min_x,
                                            float *out_node_bbox_min_y,
                                            float *out_node_bbox_min_z,
                                            float *out_node_bbox_max_x,
                                            float *out_node_bbox_max_y,
                                            float *out_node_bbox_max_z,
                                            int *out_left_child,
                                            int *out_right_child) {
    const int node_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node_index >= node_count) {
        return;
    }

    const int raw_node_index = new_to_old[node_index];
    out_node_bbox_min_x[node_index] = raw_node_bbox_min_x[raw_node_index];
    out_node_bbox_min_y[node_index] = raw_node_bbox_min_y[raw_node_index];
    out_node_bbox_min_z[node_index] = raw_node_bbox_min_z[raw_node_index];
    out_node_bbox_max_x[node_index] = raw_node_bbox_max_x[raw_node_index];
    out_node_bbox_max_y[node_index] = raw_node_bbox_max_y[raw_node_index];
    out_node_bbox_max_z[node_index] = raw_node_bbox_max_z[raw_node_index];
    out_left_child[node_index] = compacted_left_child[node_index];
    out_right_child[node_index] = compacted_right_child[node_index];
}

__global__ void emit_compacted_leaf_primitives_kernel(int node_count,
                                                      const int *new_to_old,
                                                      const int *compacted_leaf_count,
                                                      const int *compacted_leaf_begin,
                                                      const int *raw_left_child,
                                                      const int *raw_right_child,
                                                      const int *raw_leaf_primitive,
                                                      int *out_leaf_primitives,
                                                      int *out_primitive_leaf_node) {
    constexpr int max_stack_size = 8;
    const int node_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node_index >= node_count) {
        return;
    }

    const int leaf_count = compacted_leaf_count[node_index];
    if (leaf_count <= 0) {
        return;
    }

    int stack[max_stack_size];
    int stack_size = 0;
    stack[stack_size++] = new_to_old[node_index];
    int write_offset = compacted_leaf_begin[node_index];

    while (stack_size > 0) {
        const int raw_node_index = stack[--stack_size];
        const int primitive = raw_leaf_primitive[raw_node_index];
        if (primitive >= 0) {
            out_leaf_primitives[write_offset++] = primitive;
            out_primitive_leaf_node[primitive] = node_index;
            continue;
        }

        const int right = raw_right_child[raw_node_index];
        const int left = raw_left_child[raw_node_index];
        if (right >= 0 && stack_size < max_stack_size) {
            stack[stack_size++] = right;
        }
        if (left >= 0 && stack_size < max_stack_size) {
            stack[stack_size++] = left;
        }
    }
}

void compact_edge_bvh_gpu(
    int primitive_count,
    int raw_node_count,
    const int *raw_left_child,
    const int *raw_right_child,
    const int *raw_leaf_primitive,
    const float *raw_node_bbox_min_x,
    const float *raw_node_bbox_min_y,
    const float *raw_node_bbox_min_z,
    const float *raw_node_bbox_max_x,
    const float *raw_node_bbox_max_y,
    const float *raw_node_bbox_max_z,
    int compacted_node_count,
    const int *compacted_left_child,
    const int *compacted_right_child,
    const int *compacted_new_to_old,
    const int *compacted_leaf_begin,
    const int *compacted_leaf_count,
    float *out_node_bbox_min_x,
    float *out_node_bbox_min_y,
    float *out_node_bbox_min_z,
    float *out_node_bbox_max_x,
    float *out_node_bbox_max_y,
    float *out_node_bbox_max_z,
    int *out_left_child,
    int *out_right_child,
    int *out_leaf_primitives,
    int *out_primitive_leaf_node) {
    require_local(primitive_count >= 0, "compact_edge_bvh_gpu(): primitive_count must be non-negative.");
    require_local(raw_node_count >= 0, "compact_edge_bvh_gpu(): raw_node_count must be non-negative.");
    require_local(compacted_node_count >= 0,
                  "compact_edge_bvh_gpu(): compacted_node_count must be non-negative.");

    try {
        const int block_size = 256;
        const int node_blocks = (compacted_node_count + block_size - 1) / block_size;
        CudaBuffer<int> device_compacted_left(static_cast<size_t>(compacted_node_count));
        CudaBuffer<int> device_compacted_right(static_cast<size_t>(compacted_node_count));
        CudaBuffer<int> device_new_to_old(static_cast<size_t>(compacted_node_count));
        CudaBuffer<int> device_leaf_begin(static_cast<size_t>(compacted_node_count));
        CudaBuffer<int> device_leaf_count(static_cast<size_t>(compacted_node_count));

        if (compacted_node_count > 0) {
            check_cuda_call(cudaMemcpy(device_compacted_left.get(),
                                       compacted_left_child,
                                       sizeof(int) * static_cast<size_t>(compacted_node_count),
                                       cudaMemcpyHostToDevice),
                            "compact_edge_bvh_gpu(): failed to upload left-child plan");
            check_cuda_call(cudaMemcpy(device_compacted_right.get(),
                                       compacted_right_child,
                                       sizeof(int) * static_cast<size_t>(compacted_node_count),
                                       cudaMemcpyHostToDevice),
                            "compact_edge_bvh_gpu(): failed to upload right-child plan");
            check_cuda_call(cudaMemcpy(device_new_to_old.get(),
                                       compacted_new_to_old,
                                       sizeof(int) * static_cast<size_t>(compacted_node_count),
                                       cudaMemcpyHostToDevice),
                            "compact_edge_bvh_gpu(): failed to upload preorder map");
            check_cuda_call(cudaMemcpy(device_leaf_begin.get(),
                                       compacted_leaf_begin,
                                       sizeof(int) * static_cast<size_t>(compacted_node_count),
                                       cudaMemcpyHostToDevice),
                            "compact_edge_bvh_gpu(): failed to upload leaf-begin plan");
            check_cuda_call(cudaMemcpy(device_leaf_count.get(),
                                       compacted_leaf_count,
                                       sizeof(int) * static_cast<size_t>(compacted_node_count),
                                       cudaMemcpyHostToDevice),
                            "compact_edge_bvh_gpu(): failed to upload leaf-count plan");
            emit_compacted_nodes_kernel<<<node_blocks, block_size>>>(
                compacted_node_count,
                device_new_to_old.get(),
                device_compacted_left.get(),
                device_compacted_right.get(),
                raw_node_bbox_min_x,
                raw_node_bbox_min_y,
                raw_node_bbox_min_z,
                raw_node_bbox_max_x,
                raw_node_bbox_max_y,
                raw_node_bbox_max_z,
                out_node_bbox_min_x,
                out_node_bbox_min_y,
                out_node_bbox_min_z,
                out_node_bbox_max_x,
                out_node_bbox_max_y,
                out_node_bbox_max_z,
                out_left_child,
                out_right_child);
            check_cuda_last_error("compact_edge_bvh_gpu(): failed to emit compacted nodes");
            emit_compacted_leaf_primitives_kernel<<<node_blocks, block_size>>>(
                compacted_node_count,
                device_new_to_old.get(),
                device_leaf_count.get(),
                device_leaf_begin.get(),
                raw_left_child,
                raw_right_child,
                raw_leaf_primitive,
                out_leaf_primitives,
                out_primitive_leaf_node);
            check_cuda_last_error("compact_edge_bvh_gpu(): failed to emit compacted leaves");
        }

        synchronize_cuda("compact_edge_bvh_gpu(): failed to finish GPU compaction");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("compact_edge_bvh_gpu(): ") + e.what());
    }
}

} // namespace rayd
