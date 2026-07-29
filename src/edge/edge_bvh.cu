#include <src/edge/bvh.h>
#include <rayd/detail/edge/edge_aabb.h>
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
    __host__ __device__ rayd::shared::edge::BvhBounds3 operator()(
        const rayd::shared::edge::BvhBounds3 &a,
        const rayd::shared::edge::BvhBounds3 &b) const {
        return {{fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)},
                {fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z)}};
    }
};

__global__ void encode_raw_bvh_kernel(
    int primitive_count,
    int *left_child,
    int *right_child,
    const int *leaf_primitive,
    int *leaf_primitives) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= primitive_count)
        return;
    const int node = primitive_count - 1 + leaf_index;
    const int primitive = leaf_primitive[node];
    leaf_primitives[leaf_index] = primitive;
    left_child[node] = -leaf_index - 1;
    right_child[node] = 1;
}

} // namespace

[[noreturn]] inline void throw_runtime_error_local(const std::string &message) {
    throw std::runtime_error(message);
}

inline void require_local(bool condition, const std::string &message) {
    if (!condition)
        throw_runtime_error_local(message);
}

void compute_edge_optix_aabbs_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z,
    float radius,
    at::Tensor &out_aabbs) {
    require_local(edge_count >= 0, "compute_edge_optix_aabbs_cuda(): edge_count must be non-negative.");
    if (edge_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("compute_edge_optix_aabbs_cuda(): edge_count exceeds int32 range.");
    if (edge_count == 0)
        return;
    require_local(edge_p0_x.data_ptr<float>() != nullptr &&
                      edge_p0_y.data_ptr<float>() != nullptr &&
                      edge_p0_z.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): edge start pointer is null.");
    require_local(edge_e1_x.data_ptr<float>() != nullptr &&
                      edge_e1_y.data_ptr<float>() != nullptr &&
                      edge_e1_z.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): edge vector pointer is null.");
    require_local(out_aabbs.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): output pointer is null.");

    // The edge SoA and the AABB buffer are scene-owned, so the launch follows
    // the buffers' device instead of whatever device happens to be current.
    c10::cuda::CUDAGuard guard(out_aabbs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(out_aabbs.get_device()).stream();
    rayd::shared::edge::launch_edge_aabb(
        static_cast<int>(edge_count),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>(),
        radius,
        out_aabbs.data_ptr<float>(),
        stream);
}

size_t edge_bvh_bounds_reduce_scratch_bytes(int64_t edge_count, cudaStream_t stream) {
    require_local(edge_count > 0 && edge_count <= std::numeric_limits<int>::max(),
                  "edge_bvh_bounds_reduce_scratch_bytes(): invalid edge count.");
    size_t bytes = 0;
    const rayd::shared::edge::BvhBounds3 empty = {
        {std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()}};
    auto error = cub::DeviceReduce::Reduce(
        nullptr, bytes,
        static_cast<const rayd::shared::edge::BvhBounds3 *>(nullptr),
        static_cast<rayd::shared::edge::BvhBounds3 *>(nullptr),
        static_cast<int>(edge_count), BoundsUnion{}, empty, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB bounds scratch sizing failed: ") + cudaGetErrorString(error));
    return bytes;
}

void reduce_edge_bvh_bounds_cuda(
    int64_t edge_count,
    const at::Tensor &packed_bounds,
    at::Tensor &out_bound,
    at::Tensor &scratch,
    cudaStream_t stream) {
    size_t bytes = static_cast<size_t>(scratch.numel());
    const rayd::shared::edge::BvhBounds3 empty = {
        {std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()}};
    auto error = cub::DeviceReduce::Reduce(
        scratch.data_ptr<uint8_t>(), bytes,
        reinterpret_cast<const rayd::shared::edge::BvhBounds3 *>(packed_bounds.data_ptr<uint8_t>()),
        reinterpret_cast<rayd::shared::edge::BvhBounds3 *>(out_bound.data_ptr<uint8_t>()),
        static_cast<int>(edge_count), BoundsUnion{}, empty, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB bounds reduction failed: ") + cudaGetErrorString(error));
}

size_t edge_bvh_sort_scratch_bytes(int64_t edge_count, cudaStream_t stream) {
    require_local(edge_count > 0 && edge_count <= std::numeric_limits<int>::max(),
                  "edge_bvh_sort_scratch_bytes(): invalid edge count.");
    size_t bytes = 0;
    auto error = cub::DeviceRadixSort::SortPairs(
        nullptr, bytes,
        static_cast<const uint32_t *>(nullptr), static_cast<uint32_t *>(nullptr),
        static_cast<const int *>(nullptr), static_cast<int *>(nullptr),
        static_cast<int>(edge_count), 0, 32, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB sort scratch sizing failed: ") + cudaGetErrorString(error));
    return bytes;
}

void sort_edge_bvh_morton_cuda(
    int64_t edge_count,
    const at::Tensor &morton_codes_in,
    at::Tensor &morton_codes_out,
    const at::Tensor &primitive_ids_in,
    at::Tensor &primitive_ids_out,
    at::Tensor &scratch,
    cudaStream_t stream) {
    size_t bytes = static_cast<size_t>(scratch.numel());
    auto error = cub::DeviceRadixSort::SortPairs(
        scratch.data_ptr<uint8_t>(), bytes,
        reinterpret_cast<const uint32_t *>(morton_codes_in.data_ptr<int>()),
        reinterpret_cast<uint32_t *>(morton_codes_out.data_ptr<int>()),
        primitive_ids_in.data_ptr<int>(), primitive_ids_out.data_ptr<int>(),
        static_cast<int>(edge_count), 0, 32, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUB Morton sort failed: ") + cudaGetErrorString(error));
}

void encode_raw_edge_bvh_cuda(
    int64_t primitive_count,
    at::Tensor &left_child,
    at::Tensor &right_child,
    const at::Tensor &leaf_primitive,
    at::Tensor &leaf_primitives,
    cudaStream_t stream) {
    if (primitive_count == 0)
        return;
    constexpr int block_size = 256;
    const int blocks = (static_cast<int>(primitive_count) + block_size - 1) / block_size;
    encode_raw_bvh_kernel<<<blocks, block_size, 0, stream>>>(
        static_cast<int>(primitive_count), left_child.data_ptr<int>(),
        right_child.data_ptr<int>(), leaf_primitive.data_ptr<int>(),
        leaf_primitives.data_ptr<int>());
}

} // namespace rayd::torch_backend
