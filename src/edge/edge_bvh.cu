// Copyright Xingyu Chen.
// Implements edge support for edge bvh.

#include <src/edge/bvh.h>
#include <src/edge/bvh_build.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

// Windows RPC headers define `small` as `char`; keep it from rewriting
// PyTorch's CUDACachingAllocator constructor parameter.
#ifdef small
#undef small
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

namespace {

struct BoundsUnion {
    __host__ __device__ rayd::shared::edge::BvhBounds3 operator()(const rayd::shared::edge::BvhBounds3& a,
                                                                  const rayd::shared::edge::BvhBounds3& b) const {
        return {{fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)},
                {fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z)}};
    }
};

__global__ void encode_raw_bvh_kernel(int primitive_count, int* left_child, int* right_child, const int* leaf_primitive,
                                      int* leaf_primitives) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= primitive_count)
        return;
    const int node = primitive_count - 1 + leaf_index;
    const int primitive = leaf_primitive[node];
    leaf_primitives[leaf_index] = primitive;
    left_child[node] = -leaf_index - 1;
    right_child[node] = 1;
}

// Leaves of the collapsed edge BVH hold up to this many primitives, like the Dr.Jit backend's compacted BVH.
constexpr int kCollapsedLeafSize = rayd::shared::edge::kBvhLeafSize;

// The number of leaves under `node` when it has at most kCollapsedLeafSize of them, else 0. Nodes from
// `primitive_count - 1` on are the leaves of the binary radix tree.
__device__ int small_subtree_leaf_count(int node, int primitive_count, const int* left_child, const int* right_child) {
    int pending[kCollapsedLeafSize];
    int pending_count = 0;
    int internal_count = 0;
    int leaves = 0;
    int current = node;
    while (true) {
        if (current >= primitive_count - 1) {
            if (++leaves > kCollapsedLeafSize)
                return 0;
            if (pending_count == 0)
                return leaves;
            current = pending[--pending_count];
            continue;
        }
        // A subtree with at most kCollapsedLeafSize leaves has at most kCollapsedLeafSize - 1 internal nodes.
        if (++internal_count >= kCollapsedLeafSize)
            return 0;
        pending[pending_count++] = right_child[current];
        current = left_child[current];
    }
}

__global__ void count_small_subtrees_kernel(int node_count, int primitive_count, const int* left_child,
                                            const int* right_child, int* small_count) {
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node >= node_count)
        return;
    small_count[node] = small_subtree_leaf_count(node, primitive_count, left_child, right_child);
}

// A node becomes a leaf of the collapsed tree when its subtree is small and its parent's is not (or it is the root);
// leaf_size[node] is its primitive count there and 0 for every other node.
__global__ void mark_collapsed_leaves_kernel(int primitive_count, const int* left_child, const int* right_child,
                                             const int* small_count, int* leaf_size) {
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node == 0 && small_count[0] > 0)
        leaf_size[0] = small_count[0];
    if (node >= primitive_count - 1 || small_count[node] > 0)
        return;
    leaf_size[left_child[node]] = small_count[left_child[node]];
    leaf_size[right_child[node]] = small_count[right_child[node]];
}

// Writes each collapsed leaf's primitives, in left-to-right order, at its exclusive offset and encodes the node as a
// leaf of the compact traversal layout. Only collapsed leaves change; the nodes below them become unreachable.
__global__ void write_collapsed_leaves_kernel(int node_count, int primitive_count, int* left_child, int* right_child,
                                              const int* leaf_primitive, const int* leaf_size, const int* leaf_end,
                                              int* leaf_primitives) {
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node >= node_count || leaf_size[node] == 0)
        return;
    const int begin = leaf_end[node] - leaf_size[node];
    int pending[kCollapsedLeafSize];
    int pending_count = 0;
    int written = 0;
    int current = node;
    while (true) {
        if (current >= primitive_count - 1) {
            leaf_primitives[begin + written++] = leaf_primitive[current];
            if (pending_count == 0)
                break;
            current = pending[--pending_count];
            continue;
        }
        pending[pending_count++] = right_child[current];
        current = left_child[current];
    }
    left_child[node] = -begin - 1;
    right_child[node] = written;
}

} // namespace

[[noreturn]] inline void throw_runtime_error_local(const std::string& message) {
    throw std::runtime_error(message);
}

inline void require_local(bool condition, const std::string& message) {
    if (!condition)
        throw_runtime_error_local(message);
}

void compute_edge_optix_aabbs_cuda(int64_t edge_count, const at::Tensor& edge_p0_x, const at::Tensor& edge_p0_y,
                                   const at::Tensor& edge_p0_z, const at::Tensor& edge_e1_x,
                                   const at::Tensor& edge_e1_y, const at::Tensor& edge_e1_z, float radius,
                                   at::Tensor& out_aabbs) {
    require_local(edge_count >= 0, "compute_edge_optix_aabbs_cuda(): edge_count must be non-negative.");
    if (edge_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("compute_edge_optix_aabbs_cuda(): edge_count exceeds int32 range.");
    if (edge_count == 0)
        return;
    require_local(edge_p0_x.data_ptr<float>() != nullptr && edge_p0_y.data_ptr<float>() != nullptr &&
                      edge_p0_z.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): edge start pointer is null.");
    require_local(edge_e1_x.data_ptr<float>() != nullptr && edge_e1_y.data_ptr<float>() != nullptr &&
                      edge_e1_z.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): edge vector pointer is null.");
    require_local(out_aabbs.data_ptr<float>() != nullptr, "compute_edge_optix_aabbs_cuda(): output pointer is null.");

    // The edge SoA and the AABB buffer are scene-owned, so the launch follows
    // the buffers' device instead of whatever device happens to be current.
    c10::cuda::CUDAGuard guard(out_aabbs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(out_aabbs.get_device()).stream();
    rayd::shared::edge::launch_edge_aabb(static_cast<int>(edge_count), edge_p0_x.data_ptr<float>(),
                                         edge_p0_y.data_ptr<float>(), edge_p0_z.data_ptr<float>(),
                                         edge_e1_x.data_ptr<float>(), edge_e1_y.data_ptr<float>(),
                                         edge_e1_z.data_ptr<float>(), radius, out_aabbs.data_ptr<float>(), stream);
}

size_t edge_bvh_bounds_reduce_scratch_bytes(int64_t edge_count, cudaStream_t stream) {
    require_local(edge_count > 0 && edge_count <= std::numeric_limits<int>::max(),
                  "edge_bvh_bounds_reduce_scratch_bytes(): invalid edge count.");
    size_t bytes = 0;
    const rayd::shared::edge::BvhBounds3 empty = {{std::numeric_limits<float>::infinity(),
                                                   std::numeric_limits<float>::infinity(),
                                                   std::numeric_limits<float>::infinity()},
                                                  {-std::numeric_limits<float>::infinity(),
                                                   -std::numeric_limits<float>::infinity(),
                                                   -std::numeric_limits<float>::infinity()}};
    auto error = cub::DeviceReduce::Reduce(nullptr, bytes, static_cast<const rayd::shared::edge::BvhBounds3*>(nullptr),
                                           static_cast<rayd::shared::edge::BvhBounds3*>(nullptr),
                                           static_cast<int>(edge_count), BoundsUnion{}, empty, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB bounds scratch sizing failed: ") + cudaGetErrorString(error));
    return bytes;
}

void reduce_edge_bvh_bounds_cuda(int64_t edge_count, const at::Tensor& packed_bounds, at::Tensor& out_bound,
                                 at::Tensor& scratch, cudaStream_t stream) {
    size_t bytes = static_cast<size_t>(scratch.numel());
    const rayd::shared::edge::BvhBounds3 empty = {{std::numeric_limits<float>::infinity(),
                                                   std::numeric_limits<float>::infinity(),
                                                   std::numeric_limits<float>::infinity()},
                                                  {-std::numeric_limits<float>::infinity(),
                                                   -std::numeric_limits<float>::infinity(),
                                                   -std::numeric_limits<float>::infinity()}};
    auto error =
        cub::DeviceReduce::Reduce(scratch.data_ptr<uint8_t>(), bytes,
                                  reinterpret_cast<const rayd::shared::edge::BvhBounds3*>(
                                      packed_bounds.data_ptr<uint8_t>()),
                                  reinterpret_cast<rayd::shared::edge::BvhBounds3*>(out_bound.data_ptr<uint8_t>()),
                                  static_cast<int>(edge_count), BoundsUnion{}, empty, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB bounds reduction failed: ") + cudaGetErrorString(error));
}

size_t edge_bvh_sort_scratch_bytes(int64_t edge_count, cudaStream_t stream) {
    require_local(edge_count > 0 && edge_count <= std::numeric_limits<int>::max(),
                  "edge_bvh_sort_scratch_bytes(): invalid edge count.");
    size_t bytes = 0;
    auto error =
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                                        static_cast<uint32_t*>(nullptr), static_cast<const int*>(nullptr),
                                        static_cast<int*>(nullptr), static_cast<int>(edge_count), 0, 32, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB sort scratch sizing failed: ") + cudaGetErrorString(error));
    return bytes;
}

void sort_edge_bvh_morton_cuda(int64_t edge_count, const at::Tensor& morton_codes_in, at::Tensor& morton_codes_out,
                               const at::Tensor& primitive_ids_in, at::Tensor& primitive_ids_out, at::Tensor& scratch,
                               cudaStream_t stream) {
    size_t bytes = static_cast<size_t>(scratch.numel());
    auto error = cub::DeviceRadixSort::SortPairs(scratch.data_ptr<uint8_t>(), bytes,
                                                 reinterpret_cast<const uint32_t*>(morton_codes_in.data_ptr<int>()),
                                                 reinterpret_cast<uint32_t*>(morton_codes_out.data_ptr<int>()),
                                                 primitive_ids_in.data_ptr<int>(), primitive_ids_out.data_ptr<int>(),
                                                 static_cast<int>(edge_count), 0, 32, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB Morton sort failed: ") + cudaGetErrorString(error));
}

void encode_raw_edge_bvh_cuda(int64_t primitive_count, at::Tensor& left_child, at::Tensor& right_child,
                              const at::Tensor& leaf_primitive, at::Tensor& leaf_primitives, cudaStream_t stream) {
    if (primitive_count == 0)
        return;
    constexpr int block_size = 256;
    const int blocks = (static_cast<int>(primitive_count) + block_size - 1) / block_size;
    encode_raw_bvh_kernel<<<blocks, block_size, 0, stream>>>(static_cast<int>(primitive_count),
                                                             left_child.data_ptr<int>(), right_child.data_ptr<int>(),
                                                             leaf_primitive.data_ptr<int>(),
                                                             leaf_primitives.data_ptr<int>());
}

void collapse_edge_bvh_leaves_cuda(int64_t primitive_count, at::Tensor& left_child, at::Tensor& right_child,
                                   const at::Tensor& leaf_primitive, at::Tensor& leaf_primitives, cudaStream_t stream) {
    if (primitive_count == 0)
        return;
    constexpr int block_size = 256;
    const int primitives = static_cast<int>(primitive_count);
    const int node_count = 2 * primitives - 1;
    const int node_blocks = (node_count + block_size - 1) / block_size;
    const at::TensorOptions int_options = left_child.options();
    at::Tensor small_count = at::empty({node_count}, int_options);
    at::Tensor leaf_size = at::zeros({node_count}, int_options);
    count_small_subtrees_kernel<<<node_blocks, block_size, 0, stream>>>(node_count, primitives,
                                                                        left_child.data_ptr<int>(),
                                                                        right_child.data_ptr<int>(),
                                                                        small_count.data_ptr<int>());
    mark_collapsed_leaves_kernel<<<node_blocks, block_size, 0, stream>>>(primitives, left_child.data_ptr<int>(),
                                                                         right_child.data_ptr<int>(),
                                                                         small_count.data_ptr<int>(),
                                                                         leaf_size.data_ptr<int>());
    // Every primitive lies under exactly one collapsed leaf, so the leaf sizes sum to the primitive count.
    const at::Tensor leaf_end = at::cumsum(leaf_size, 0, at::kInt);
    write_collapsed_leaves_kernel<<<node_blocks, block_size, 0, stream>>>(node_count, primitives,
                                                                          left_child.data_ptr<int>(),
                                                                          right_child.data_ptr<int>(),
                                                                          leaf_primitive.data_ptr<int>(),
                                                                          leaf_size.data_ptr<int>(),
                                                                          leaf_end.data_ptr<int>(),
                                                                          leaf_primitives.data_ptr<int>());
}

} // namespace rayd::torch_backend
