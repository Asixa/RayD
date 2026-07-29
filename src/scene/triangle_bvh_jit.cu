// Copyright Xingyu Chen.
// Implements scene support for triangle bvh Dr.Jit.

#include <rayd/jit/triangle_bvh_gpu.h>

#include <rayd/jit/native_launch_audit.h>
#include <rayd/bvh/build.h>
#include <rayd/bvh/topology.h>
#include <rayd/bvh/triangle_query.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace rayd {

namespace {

using Bounds3 = shared::bvh::BvhBounds3;

[[noreturn]] void throw_runtime_error_local(const std::string &message) {
    throw std::runtime_error(message);
}

void require_local(bool condition, const std::string &message) {
    if (!condition) {
        throw_runtime_error_local(message);
    }
}

void check_cuda_call(cudaError_t error, const char *message) {
    require_local(error == cudaSuccess,
                  std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

Bounds3 empty_bounds() {
    constexpr float inf = 1e30f;
    return {{inf, inf, inf}, {-inf, -inf, -inf}};
}

struct BoundsUnion {
    __host__ __device__ Bounds3 operator()(const Bounds3 &a, const Bounds3 &b) const {
        return {{fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)},
                {fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z)}};
    }
};

/// The CUDA device that is current on this thread right now.
int current_cuda_device() {
    int device = -1;
    check_cuda_call(cudaGetDevice(&device), "triangle_bvh(): failed to query the current CUDA device");
    return device;
}

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t count) { allocate(count); }
    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    void allocate(size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = count;
        device_ = -1;
        if (count_ == 0) {
            return;
        }
        // Device memory is only valid on the device it was allocated on. Record
        // that device here and re-check it in get(), so scratch allocated under
        // one current device can never be handed to a launch issued under
        // another.
        const int device = current_cuda_device();
        const cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_);
        require_local(error == cudaSuccess,
                      std::string("CudaBuffer::allocate(): ") + cudaGetErrorString(error));
        device_ = device;
    }

    T *get() { require_owning_device(); return ptr_; }
    const T *get() const { require_owning_device(); return ptr_; }
    int device() const { return device_; }

private:
    /// Refuse to hand out the pointer unless the allocating device is current.
    void require_owning_device() const {
        if (ptr_ == nullptr) {
            return;
        }
        const int device = current_cuda_device();
        require_local(device == device_,
                      "CudaBuffer::get(): buffer was allocated on CUDA device " +
                          std::to_string(device_) + " but device " + std::to_string(device) +
                          " is current.");
    }

    T *ptr_ = nullptr;
    size_t count_ = 0;
    int device_ = -1;
};

class CudaStreamHandle {
public:
    CudaStreamHandle() {
        check_cuda_call(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                        "triangle_bvh(): failed to create CUDA stream");
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

constexpr int kBlockSize = 256;

int block_count(int count) { return (count + kBlockSize - 1) / kBlockSize; }

/// Triangle AABB: writes the SoA bounds used by Morton/finalize and a packed
/// Bounds3 array used by the scene-bounds reduction.
__global__ void triangle_bounds_kernel(int primitive_count,
                                        TriBvhTrianglePtrs tri,
                                        shared::bvh::MutableAabbSoAView bounds_soa,
                                        Bounds3 *packed_bounds) {
    const int prim = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (prim >= primitive_count) {
        return;
    }
    const float ax = tri.p0_x[prim];
    const float ay = tri.p0_y[prim];
    const float az = tri.p0_z[prim];
    const float bx = ax + tri.e1_x[prim];
    const float by = ay + tri.e1_y[prim];
    const float bz = az + tri.e1_z[prim];
    const float cx = ax + tri.e2_x[prim];
    const float cy = ay + tri.e2_y[prim];
    const float cz = az + tri.e2_z[prim];
    const float min_x = fminf(ax, fminf(bx, cx));
    const float min_y = fminf(ay, fminf(by, cy));
    const float min_z = fminf(az, fminf(bz, cz));
    const float max_x = fmaxf(ax, fmaxf(bx, cx));
    const float max_y = fmaxf(ay, fmaxf(by, cy));
    const float max_z = fmaxf(az, fmaxf(bz, cz));
    bounds_soa.min_x[prim] = min_x;
    bounds_soa.min_y[prim] = min_y;
    bounds_soa.min_z[prim] = min_z;
    bounds_soa.max_x[prim] = max_x;
    bounds_soa.max_y[prim] = max_y;
    bounds_soa.max_z[prim] = max_z;
    packed_bounds[prim] = {{min_x, min_y, min_z}, {max_x, max_y, max_z}};
}

/// Recompute one leaf node's bounds as the union of its primitives' AABBs.
__global__ void refit_leaf_bounds_kernel(int leaf_node_count,
                                         const int *leaf_nodes,
                                         const int *left_child,
                                         const int *right_child,
                                         const int *leaf_primitives,
                                         TriBvhTrianglePtrs tri,
                                         shared::bvh::MutableAabbSoAView node_bounds) {
    const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item >= leaf_node_count) {
        return;
    }
    const int node = leaf_nodes[item];
    const int leaf_begin = -left_child[node] - 1;
    const int leaf_count = right_child[node];
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    for (int slot = 0; slot < leaf_count; ++slot) {
        const int prim = leaf_primitives[leaf_begin + slot];
        const float ax = tri.p0_x[prim];
        const float ay = tri.p0_y[prim];
        const float az = tri.p0_z[prim];
        const float bx = ax + tri.e1_x[prim];
        const float by = ay + tri.e1_y[prim];
        const float bz = az + tri.e1_z[prim];
        const float cx = ax + tri.e2_x[prim];
        const float cy = ay + tri.e2_y[prim];
        const float cz = az + tri.e2_z[prim];
        min_x = fminf(min_x, fminf(ax, fminf(bx, cx)));
        min_y = fminf(min_y, fminf(ay, fminf(by, cy)));
        min_z = fminf(min_z, fminf(az, fminf(bz, cz)));
        max_x = fmaxf(max_x, fmaxf(ax, fmaxf(bx, cx)));
        max_y = fmaxf(max_y, fmaxf(ay, fmaxf(by, cy)));
        max_z = fmaxf(max_z, fmaxf(az, fmaxf(bz, cz)));
    }
    node_bounds.min_x[node] = min_x;
    node_bounds.min_y[node] = min_y;
    node_bounds.min_z[node] = min_z;
    node_bounds.max_x[node] = max_x;
    node_bounds.max_y[node] = max_y;
    node_bounds.max_z[node] = max_z;
}

/// Refit one level of internal nodes: bounds = union of the two child bounds.
__global__ void refit_internal_level_kernel(int level_count,
                                            const int *level_nodes,
                                            const int *left_child,
                                            const int *right_child,
                                            shared::bvh::MutableAabbSoAView node_bounds) {
    const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (item >= level_count) {
        return;
    }
    const int node = level_nodes[item];
    const int left = left_child[node];
    const int right = right_child[node];
    node_bounds.min_x[node] = fminf(node_bounds.min_x[left], node_bounds.min_x[right]);
    node_bounds.min_y[node] = fminf(node_bounds.min_y[left], node_bounds.min_y[right]);
    node_bounds.min_z[node] = fminf(node_bounds.min_z[left], node_bounds.min_z[right]);
    node_bounds.max_x[node] = fmaxf(node_bounds.max_x[left], node_bounds.max_x[right]);
    node_bounds.max_y[node] = fmaxf(node_bounds.max_y[left], node_bounds.max_y[right]);
    node_bounds.max_z[node] = fmaxf(node_bounds.max_z[left], node_bounds.max_z[right]);
}

shared::bvh::MutableAabbSoAView mutable_soa(TriBvhBoundsPtrs ptrs, int count) {
    return {ptrs.min_x, ptrs.min_y, ptrs.min_z, ptrs.max_x, ptrs.max_y, ptrs.max_z,
            static_cast<size_t>(count)};
}

shared::bvh::AabbSoAView const_soa(TriBvhBoundsPtrs ptrs, int count) {
    return {ptrs.min_x, ptrs.min_y, ptrs.min_z, ptrs.max_x, ptrs.max_y, ptrs.max_z,
            static_cast<size_t>(count)};
}

shared::bvh::AabbSoAView const_soa(TriBvhConstBoundsPtrs ptrs, int count) {
    return {ptrs.min_x, ptrs.min_y, ptrs.min_z, ptrs.max_x, ptrs.max_y, ptrs.max_z,
            static_cast<size_t>(count)};
}

shared::bvh::TriangleSoAView triangle_soa(TriBvhTrianglePtrs tri, int count) {
    return {tri.p0_x, tri.p0_y, tri.p0_z, tri.e1_x, tri.e1_y, tri.e1_z,
            tri.e2_x, tri.e2_y, tri.e2_z, static_cast<size_t>(count)};
}

shared::bvh::TriangleRaySoAView ray_soa(TriBvhRayPtrs rays, int count) {
    return {rays.origin_x, rays.origin_y, rays.origin_z, rays.dir_x, rays.dir_y, rays.dir_z,
            rays.t_max, rays.active, static_cast<size_t>(count)};
}

shared::bvh::TriangleTraversalScratchView scratch_view(int *stack_nodes, int *overflow, int ray_count) {
    shared::bvh::TriangleTraversalScratchView scratch;
    scratch.node_indices = stack_nodes;
    scratch.overflow = overflow;
    scratch.query_stride = static_cast<size_t>(ray_count);
    scratch.stack_depth = static_cast<size_t>(shared::bvh::kBvhTraversalStackDepth);
    scratch.capacity = static_cast<size_t>(ray_count) *
                       static_cast<size_t>(shared::bvh::kBvhTraversalStackDepth);
    scratch.overflow_capacity = static_cast<size_t>(ray_count);
    return scratch;
}

shared::bvh::CompactBvhTopologyView topology_view(const int *left_child,
                                                  const int *right_child,
                                                  const int *leaf_primitives,
                                                  int node_count,
                                                  int primitive_count,
                                                  int leaf_primitive_count) {
    shared::bvh::CompactBvhTopologyView topology;
    topology.left_child = left_child;
    topology.right_child = right_child;
    topology.leaf_primitives = leaf_primitives;
    topology.node_active_count = nullptr;
    topology.node_count = static_cast<size_t>(node_count);
    topology.primitive_count = static_cast<size_t>(primitive_count);
    topology.leaf_primitive_count = static_cast<size_t>(leaf_primitive_count);
    return topology;
}

/// True when any overflow lane fired; also gives the host a repair trigger.
bool any_overflow(const int *overflow, int ray_count, cudaStream_t stream) {
    if (overflow == nullptr || ray_count == 0) {
        return false;
    }
    std::vector<int> host(static_cast<size_t>(ray_count));
    audit_cuda_memcpy_async();
    check_cuda_call(cudaMemcpyAsync(host.data(), overflow, sizeof(int) * host.size(),
                                    cudaMemcpyDeviceToHost, stream),
                    "triangle_bvh(): failed to copy overflow flags");
    audit_cuda_stream_synchronize();
    check_cuda_call(cudaStreamSynchronize(stream), "triangle_bvh(): failed to drain overflow copy");
    for (int value : host) {
        if (value != 0) {
            return true;
        }
    }
    return false;
}

} // namespace

void build_triangle_bvh_gpu(int primitive_count,
                            TriBvhTrianglePtrs triangles,
                            TriBvhBoundsPtrs primitive_bounds,
                            TriBvhBoundsPtrs node_bounds,
                            int *left_child,
                            int *right_child,
                            int *leaf_primitive,
                            int *is_leaf,
                            int *primitive_leaf_node) {
    require_local(primitive_count > 0, "build_triangle_bvh_gpu(): primitive_count must be positive.");
    try {
        const int node_count = std::max(2 * primitive_count - 1, 1);
        const int internal_count = std::max(primitive_count - 1, 0);
        const int primitive_blocks = block_count(primitive_count);

        CudaStreamHandle stream_handle;
        const cudaStream_t stream = stream_handle.get();

        CudaBuffer<Bounds3> packed_bounds(static_cast<size_t>(primitive_count));
        CudaBuffer<Bounds3> reduced_bounds(1);
        CudaBuffer<uint32_t> morton_in(static_cast<size_t>(primitive_count));
        CudaBuffer<uint32_t> morton_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> indices_in(static_cast<size_t>(primitive_count));
        CudaBuffer<int> indices_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> parent(static_cast<size_t>(node_count));
        CudaBuffer<int> merge_counters(static_cast<size_t>(std::max(internal_count, 1)));

        triangle_bounds_kernel<<<primitive_blocks, kBlockSize, 0, stream>>>(
            primitive_count, triangles, mutable_soa(primitive_bounds, primitive_count),
            packed_bounds.get());
        audit_cuda_kernel_launch("triangle_bounds_kernel", static_cast<uint32_t>(primitive_blocks),
                                 1, 1, kBlockSize, 1, 1, static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch triangle-bounds kernel");

        size_t reduce_temp_size = 0;
        audit_cub_reduce();
        check_cuda_call(cub::DeviceReduce::Reduce(nullptr, reduce_temp_size, packed_bounds.get(),
                                                  reduced_bounds.get(), primitive_count,
                                                  BoundsUnion(), empty_bounds(), stream),
                        "build_triangle_bvh_gpu(): failed to size scene-bound reduction");
        CudaBuffer<char> reduce_temp(reduce_temp_size);
        audit_cub_reduce();
        check_cuda_call(cub::DeviceReduce::Reduce(reduce_temp.get(), reduce_temp_size,
                                                  packed_bounds.get(), reduced_bounds.get(),
                                                  primitive_count, BoundsUnion(), empty_bounds(),
                                                  stream),
                        "build_triangle_bvh_gpu(): failed to reduce scene bounds");
        check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch scene-bound reduction");

        Bounds3 scene_bounds = empty_bounds();
        audit_cuda_memcpy_async();
        check_cuda_call(cudaMemcpyAsync(&scene_bounds, reduced_bounds.get(), sizeof(Bounds3),
                                        cudaMemcpyDeviceToHost, stream),
                        "build_triangle_bvh_gpu(): failed to copy scene bounds");

        shared::bvh::launch_init_sequence_async({indices_in.get(), primitive_count, stream});
        audit_cuda_kernel_launch("init_sequence_kernel", static_cast<uint32_t>(primitive_blocks),
                                 1, 1, kBlockSize, 1, 1, static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch index initialization");

        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream), "build_triangle_bvh_gpu(): failed to reduce scene bounds");

        shared::bvh::launch_compute_morton_codes_async({const_soa(primitive_bounds, primitive_count),
                                                        scene_bounds, morton_in.get(), stream});
        audit_cuda_kernel_launch("compute_morton_codes_kernel",
                                 static_cast<uint32_t>(primitive_blocks), 1, 1, kBlockSize, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch Morton-code kernel");

        size_t sort_temp_size = 0;
        audit_cub_sort();
        check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_size, morton_in.get(),
                                                        morton_out.get(), indices_in.get(),
                                                        indices_out.get(), primitive_count, 0, 32,
                                                        stream),
                        "build_triangle_bvh_gpu(): failed to size radix sort");
        CudaBuffer<char> sort_temp(sort_temp_size);
        audit_cub_sort();
        check_cuda_call(cub::DeviceRadixSort::SortPairs(sort_temp.get(), sort_temp_size,
                                                        morton_in.get(), morton_out.get(),
                                                        indices_in.get(), indices_out.get(),
                                                        primitive_count, 0, 32, stream),
                        "build_triangle_bvh_gpu(): failed to sort Morton codes");
        check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch Morton sort");

        auto memset_int = [&](int *ptr, int value, size_t count, const char *message) {
            if (count == 0) {
                return;
            }
            const unsigned char byte_value = static_cast<unsigned char>(value & 0xff);
            audit_cuda_memset_async();
            check_cuda_call(cudaMemsetAsync(ptr, byte_value, count * sizeof(int), stream), message);
        };
        memset_int(left_child, -1, static_cast<size_t>(node_count), "build_triangle_bvh_gpu(): init left_child");
        memset_int(right_child, -1, static_cast<size_t>(node_count), "build_triangle_bvh_gpu(): init right_child");
        memset_int(leaf_primitive, -1, static_cast<size_t>(node_count), "build_triangle_bvh_gpu(): init leaf_primitive");
        memset_int(is_leaf, 0, static_cast<size_t>(node_count), "build_triangle_bvh_gpu(): init is_leaf");
        memset_int(primitive_leaf_node, -1, static_cast<size_t>(primitive_count), "build_triangle_bvh_gpu(): init primitive_leaf_node");
        memset_int(parent.get(), -1, static_cast<size_t>(node_count), "build_triangle_bvh_gpu(): init parent");
        memset_int(merge_counters.get(), 0, static_cast<size_t>(std::max(internal_count, 1)), "build_triangle_bvh_gpu(): init merge counters");

        if (internal_count > 0) {
            shared::bvh::launch_build_radix_tree_async({morton_out.get(), indices_out.get(),
                                                        left_child, right_child, parent.get(),
                                                        primitive_count, stream});
            audit_cuda_kernel_launch("build_radix_tree_kernel",
                                     static_cast<uint32_t>(block_count(internal_count)), 1, 1,
                                     kBlockSize, 1, 1, static_cast<uint64_t>(internal_count));
            check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch radix-tree kernel");
        }

        shared::bvh::launch_finalize_leaves_and_bounds_async({indices_out.get(), parent.get(),
                                                              const_soa(primitive_bounds, primitive_count),
                                                              left_child, right_child,
                                                              mutable_soa(node_bounds, node_count),
                                                              leaf_primitive, is_leaf,
                                                              primitive_leaf_node, merge_counters.get(),
                                                              primitive_count, stream});
        audit_cuda_kernel_launch("finalize_leaves_and_bounds_kernel",
                                 static_cast<uint32_t>(primitive_blocks), 1, 1, kBlockSize, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_triangle_bvh_gpu(): failed to launch bounds-finalization kernel");

        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream), "build_triangle_bvh_gpu(): failed to complete build");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("build_triangle_bvh_gpu(): ") + error.what());
    }
}

void refit_triangle_bvh_gpu(int node_count,
                            TriBvhTrianglePtrs triangles,
                            const int *left_child,
                            const int *right_child,
                            const int *leaf_primitives,
                            const int *leaf_nodes,
                            int leaf_node_count,
                            const int *level_nodes,
                            const int *level_offsets,
                            int level_count,
                            TriBvhBoundsPtrs node_bounds) {
    try {
        CudaStreamHandle stream_handle;
        const cudaStream_t stream = stream_handle.get();
        const shared::bvh::MutableAabbSoAView bounds = mutable_soa(node_bounds, node_count);

        if (leaf_node_count > 0) {
            refit_leaf_bounds_kernel<<<block_count(leaf_node_count), kBlockSize, 0, stream>>>(
                leaf_node_count, leaf_nodes, left_child, right_child, leaf_primitives, triangles,
                bounds);
            audit_cuda_kernel_launch("refit_leaf_bounds_kernel",
                                     static_cast<uint32_t>(block_count(leaf_node_count)), 1, 1,
                                     kBlockSize, 1, 1, static_cast<uint64_t>(leaf_node_count));
            check_cuda_last_error("refit_triangle_bvh_gpu(): failed to launch leaf-bounds refit");
        }

        for (int level = 0; level < level_count; ++level) {
            const int begin = level_offsets[level];
            const int end = level_offsets[level + 1];
            const int count = end - begin;
            if (count <= 0) {
                continue;
            }
            refit_internal_level_kernel<<<block_count(count), kBlockSize, 0, stream>>>(
                count, level_nodes + begin, left_child, right_child, bounds);
            audit_cuda_kernel_launch("refit_internal_level_kernel",
                                     static_cast<uint32_t>(block_count(count)), 1, 1, kBlockSize, 1,
                                     1, static_cast<uint64_t>(count));
            check_cuda_last_error("refit_triangle_bvh_gpu(): failed to launch internal-node refit");
        }

        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream), "refit_triangle_bvh_gpu(): failed to complete refit");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("refit_triangle_bvh_gpu(): ") + error.what());
    }
}

void query_triangle_closest_hit_gpu(int ray_count,
                                    int primitive_count,
                                    int node_count,
                                    int leaf_primitive_count,
                                    float t_min,
                                    TriBvhTrianglePtrs triangles,
                                    TriBvhConstBoundsPtrs node_bounds,
                                    const int *left_child,
                                    const int *right_child,
                                    const int *leaf_primitives,
                                    TriBvhRayPtrs rays,
                                    const int *shape_id,
                                    const int *local_prim_id,
                                    float *out_t,
                                    float *out_bary_u,
                                    float *out_bary_v,
                                    int *out_shape_id,
                                    int *out_local_prim_id,
                                    int *stack_nodes,
                                    int *overflow) {
    if (ray_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        const cudaStream_t stream = stream_handle.get();

        shared::bvh::TriangleClosestHitParams params;
        params.triangles = triangle_soa(triangles, primitive_count);
        params.node_bounds = const_soa(node_bounds, node_count);
        params.topology = topology_view(left_child, right_child, leaf_primitives, node_count,
                                        primitive_count, leaf_primitive_count);
        params.rays = ray_soa(rays, ray_count);
        params.prim_map = {shape_id, local_prim_id, static_cast<size_t>(primitive_count)};
        params.output = {out_t, out_bary_u, out_bary_v, out_shape_id, out_local_prim_id,
                         static_cast<size_t>(ray_count)};
        params.scratch = scratch_view(stack_nodes, overflow, ray_count);
        params.t_min = t_min;
        params.stream = stream;

        shared::bvh::launch_triangle_closest_hit_async(params);
        audit_cuda_kernel_launch("triangle_closest_hit_kernel",
                                 static_cast<uint32_t>(block_count(ray_count)), 1, 1, kBlockSize, 1,
                                 1, static_cast<uint64_t>(ray_count));
        check_cuda_last_error("query_triangle_closest_hit_gpu(): failed to launch closest-hit kernel");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream), "query_triangle_closest_hit_gpu(): failed to drain closest-hit kernel");

        if (any_overflow(overflow, ray_count, stream)) {
            shared::bvh::launch_triangle_closest_hit_repair_async(params);
            audit_cuda_kernel_launch("triangle_closest_hit_repair_kernel",
                                     static_cast<uint32_t>(block_count(ray_count)), 1, 1, kBlockSize,
                                     1, 1, static_cast<uint64_t>(ray_count));
            check_cuda_last_error("query_triangle_closest_hit_gpu(): failed to launch repair kernel");
            audit_cuda_stream_synchronize();
            check_cuda_call(cudaStreamSynchronize(stream), "query_triangle_closest_hit_gpu(): failed to drain repair kernel");
        }
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("query_triangle_closest_hit_gpu(): ") + error.what());
    }
}

void query_triangle_occluded_gpu(int ray_count,
                                 int primitive_count,
                                 int node_count,
                                 int leaf_primitive_count,
                                 float t_min,
                                 TriBvhTrianglePtrs triangles,
                                 TriBvhConstBoundsPtrs node_bounds,
                                 const int *left_child,
                                 const int *right_child,
                                 const int *leaf_primitives,
                                 TriBvhRayPtrs rays,
                                 int *out_hit,
                                 int *stack_nodes,
                                 int *overflow) {
    if (ray_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        const cudaStream_t stream = stream_handle.get();

        shared::bvh::TriangleOccludedParams params;
        params.triangles = triangle_soa(triangles, primitive_count);
        params.node_bounds = const_soa(node_bounds, node_count);
        params.topology = topology_view(left_child, right_child, leaf_primitives, node_count,
                                        primitive_count, leaf_primitive_count);
        params.rays = ray_soa(rays, ray_count);
        params.out_hit = out_hit;
        params.scratch = scratch_view(stack_nodes, overflow, ray_count);
        params.t_min = t_min;
        params.stream = stream;

        shared::bvh::launch_triangle_occluded_async(params);
        audit_cuda_kernel_launch("triangle_occluded_kernel",
                                 static_cast<uint32_t>(block_count(ray_count)), 1, 1, kBlockSize, 1,
                                 1, static_cast<uint64_t>(ray_count));
        check_cuda_last_error("query_triangle_occluded_gpu(): failed to launch occlusion kernel");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream), "query_triangle_occluded_gpu(): failed to drain occlusion kernel");

        if (any_overflow(overflow, ray_count, stream)) {
            shared::bvh::launch_triangle_occluded_repair_async(params);
            audit_cuda_kernel_launch("triangle_occluded_repair_kernel",
                                     static_cast<uint32_t>(block_count(ray_count)), 1, 1, kBlockSize,
                                     1, 1, static_cast<uint64_t>(ray_count));
            check_cuda_last_error("query_triangle_occluded_gpu(): failed to launch occlusion repair");
            audit_cuda_stream_synchronize();
            check_cuda_call(cudaStreamSynchronize(stream), "query_triangle_occluded_gpu(): failed to drain occlusion repair");
        }
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("query_triangle_occluded_gpu(): ") + error.what());
    }
}

void query_triangle_first_blocker_gpu(int ray_count,
                                      int primitive_count,
                                      int node_count,
                                      int leaf_primitive_count,
                                      float t_min,
                                      TriBvhTrianglePtrs triangles,
                                      TriBvhConstBoundsPtrs node_bounds,
                                      const int *left_child,
                                      const int *right_child,
                                      const int *leaf_primitives,
                                      TriBvhRayPtrs rays,
                                      const int *ignore_prim_ids,
                                      int ignore_stride,
                                      int *out_global_prim_id,
                                      int *stack_nodes,
                                      int *overflow) {
    if (ray_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        const cudaStream_t stream = stream_handle.get();

        shared::bvh::TriangleFirstBlockerParams params;
        params.triangles = triangle_soa(triangles, primitive_count);
        params.node_bounds = const_soa(node_bounds, node_count);
        params.topology = topology_view(left_child, right_child, leaf_primitives, node_count,
                                        primitive_count, leaf_primitive_count);
        params.rays = ray_soa(rays, ray_count);
        params.out_global_prim_id = out_global_prim_id;
        params.ignore_prim_ids = ignore_prim_ids;
        params.ignore_stride = ignore_stride;
        params.scratch = scratch_view(stack_nodes, overflow, ray_count);
        params.t_min = t_min;
        params.stream = stream;

        shared::bvh::launch_triangle_first_blocker_async(params);
        audit_cuda_kernel_launch("triangle_first_blocker_kernel",
                                 static_cast<uint32_t>(block_count(ray_count)), 1, 1, kBlockSize, 1,
                                 1, static_cast<uint64_t>(ray_count));
        check_cuda_last_error("query_triangle_first_blocker_gpu(): failed to launch first-blocker kernel");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream), "query_triangle_first_blocker_gpu(): failed to drain first-blocker kernel");
        require_local(!any_overflow(overflow, ray_count, stream),
                      "query_triangle_first_blocker_gpu(): traversal stack overflow.");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("query_triangle_first_blocker_gpu(): ") + error.what());
    }
}

} // namespace rayd
