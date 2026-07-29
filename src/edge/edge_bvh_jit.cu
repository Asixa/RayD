#include <rayd/jit/edge_bvh.h>
#include <rayd/jit/edge_bvh_config.h>
#include <rayd/jit/native_launch_audit.h>
#include <rayd/detail/edge/bvh_build.h>
#include <rayd/detail/edge/edge_aabb.h>

#include <algorithm>
#include <cstdlib>
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


using Bounds3 = shared::edge::BvhBounds3;

inline Bounds3 empty_bounds() {
    constexpr float inf = 1e30f;
    return { { inf, inf, inf }, { -inf, -inf, -inf } };
}

void check_cuda_call(cudaError_t error, const char *message);

/// The CUDA device that is current on this thread right now.
int current_cuda_device() {
    int device = -1;
    check_cuda_call(cudaGetDevice(&device), "edge_bvh: failed to query the current CUDA device");
    return device;
}

/// Make \p device current for the lifetime of the guard, restoring the previous
/// device afterwards.
///
/// Every edge-BVH entry point opens one of these from its caller-supplied
/// context, so the scratch allocations, launches, and stream objects created
/// inside a single call all belong to the same, explicitly named device instead
/// of to whatever happened to be current when the call was made.
class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device) : previous_device_(current_cuda_device()) {
        if (previous_device_ != device) {
            check_cuda_call(cudaSetDevice(device),
                            "edge_bvh: failed to bind the requested CUDA device");
            restore_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (restore_) {
            cudaSetDevice(previous_device_);
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard &) = delete;
    CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
    int previous_device_ = 0;
    bool restore_ = false;
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
        : ptr_(other.ptr_), count_(other.count_), device_(other.device_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.device_ = -1;
    }

    CudaBuffer &operator=(CudaBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            device_ = other.device_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.device_ = -1;
        }
        return *this;
    }

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

        // Record the device the allocation belongs to. Device memory is only
        // valid on the device it was allocated on, so binding it here (under the
        // entry point's CudaDeviceGuard) is what makes the get() check below a
        // real guarantee rather than a guess.
        const int device = current_cuda_device();
        const cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_);
        require_local(error == cudaSuccess,
                      std::string("CudaBuffer::allocate(): ") + cudaGetErrorString(error));
        device_ = device;
    }

    T *get() { require_owning_device(); return ptr_; }
    const T *get() const { require_owning_device(); return ptr_; }
    size_t size() const { return count_; }
    int device() const { return device_; }

private:
    /// Refuse to hand out the pointer unless the allocating device is current,
    /// so it can never reach a launch or copy issued against another device.
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

struct BoundsUnion {
    __host__ __device__ Bounds3 operator()(const Bounds3 &a, const Bounds3 &b) const {
        return {
            { fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z) },
            { fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z) }
        };
    }
};












void check_cuda_call(cudaError_t error, const char *message) {
    require_local(error == cudaSuccess,
                  std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

/// Drain \p stream only.
///
/// Every launch in this file goes to a stream the entry point either owns or was
/// handed, so a device-wide synchronize was both wider than needed (it also
/// blocked on unrelated work from other threads sharing the device) and weaker
/// than it looked (it says nothing about *which* device was drained). Scoping
/// the wait to the stream that carries the work makes the dependency explicit.
void synchronize_cuda(cudaStream_t stream, const char *message) {
    audit_cuda_stream_synchronize();
    check_cuda_call(cudaStreamSynchronize(stream), message);
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
    audit_cuda_memset_async();
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
    audit_cuda_memcpy();
    return result;
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

void compute_edge_optix_aabbs_gpu(
    const EdgeBvhCudaContext &context,
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float inflation,
    float *out_aabbs) {
    require_local(primitive_count >= 0, "compute_edge_optix_aabbs_gpu(): primitive_count must be non-negative.");
    if (primitive_count == 0) {
        return;
    }
    require_local(edge_p0_x != nullptr && edge_p0_y != nullptr && edge_p0_z != nullptr,
                  "compute_edge_optix_aabbs_gpu(): edge start pointer is null.");
    require_local(edge_e1_x != nullptr && edge_e1_y != nullptr && edge_e1_z != nullptr,
                  "compute_edge_optix_aabbs_gpu(): edge vector pointer is null.");
    require_local(out_aabbs != nullptr, "compute_edge_optix_aabbs_gpu(): output pointer is null.");

    try {
        const CudaDeviceGuard device_guard(context.device);
        constexpr int block_size = 256;
        const int block_count = (primitive_count + block_size - 1) / block_size;
        // On the caller's stream: the edge SoA inputs were produced there, and
        // the OptiX custom-primitive build that consumes out_aabbs is issued
        // there too, so both dependencies are stream-ordered rather than resting
        // on a device-wide drain.
        shared::edge::launch_edge_aabb(
            primitive_count,
            edge_p0_x,
            edge_p0_y,
            edge_p0_z,
            edge_e1_x,
            edge_e1_y,
            edge_e1_z,
            inflation,
            out_aabbs,
            context.stream);
        audit_cuda_kernel_launch("compute_edge_optix_aabbs_kernel",
                                 static_cast<uint32_t>(block_count), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("compute_edge_optix_aabbs_gpu(): failed to launch AABB kernel");
        synchronize_cuda(context.stream,
                         "compute_edge_optix_aabbs_gpu(): failed to finish AABB kernel");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("compute_edge_optix_aabbs_gpu(): ") + e.what());
    }
}

void build_edge_bvh_gpu(
    const EdgeBvhCudaContext &context,
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
        // Bind the caller's device before anything is allocated: the scratch
        // buffers, streams, and events below all belong to it.
        const CudaDeviceGuard device_guard(context.device);
        const int node_count = std::max(2 * primitive_count - 1, 1);
        const int block_size = 256;
        const int primitive_blocks = (primitive_count + block_size - 1) / block_size;
        const int internal_count = std::max(primitive_count - 1, 0);
        const int internal_blocks = (internal_count + block_size - 1) / block_size;
        const EdgeBVHBuildStreamMode build_stream_mode = active_edge_bvh_build_stream_mode();
        const EdgeBVHPostBuildStrategy post_build_strategy = active_edge_bvh_post_build_strategy();
        const bool overlap_build_streams =
            build_stream_mode == EdgeBVHBuildStreamMode::Overlap;
        const bool treelet_enabled =
            post_build_strategy == EdgeBVHPostBuildStrategy::GpuTreelet &&
            primitive_count >= EdgeBVHTreeletMinPrimitives &&
            primitive_count <= EdgeBVHTreeletMaxPrimitives &&
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
        CudaEventHandle caller_ready_event;
        CudaEventHandle schedule_uploaded_event;
        const cudaStream_t bounds_stream = bounds_stream_handle.get();
        const cudaStream_t sequence_stream =
            overlap_build_streams ? sequence_stream_handle.get() : bounds_stream;

        // The build runs on its own non-blocking streams, which are *not*
        // implicitly ordered against the caller's stream. Join it once here so
        // the edge SoA inputs the caller produced are visible to every kernel
        // below without depending on the caller having drained its stream first.
        audit_cuda_event_record();
        check_cuda_call(cudaEventRecord(caller_ready_event.get(), context.stream),
                        "build_edge_lbvh_gpu(): failed to record caller-ready event");
        audit_cuda_stream_wait_event();
        check_cuda_call(cudaStreamWaitEvent(bounds_stream, caller_ready_event.get(), 0),
                        "build_edge_lbvh_gpu(): failed to join the caller stream");
        if (overlap_build_streams) {
            audit_cuda_stream_wait_event();
            check_cuda_call(cudaStreamWaitEvent(sequence_stream, caller_ready_event.get(), 0),
                            "build_edge_lbvh_gpu(): failed to join the caller stream");
        }

        // The treelet schedule and its device copy outlive the treelet block:
        // the upload is asynchronous, so the host source and the device buffer
        // must both stay alive until the final drain below.
        FlatNodeLevels optimize_schedule;
        CudaBuffer<int> optimize_nodes_device;
        std::vector<int> host_left_child;
        std::vector<int> host_right_child;
        std::vector<int> host_is_leaf;
        std::vector<int> subtree_leaf_counts;
        std::vector<int> node_heights;
        std::vector<std::vector<int>> host_level_groups;
        int max_height = 0;

        shared::edge::launch_compute_primitive_bounds_async({
            { edge_p0_x, edge_p0_y, edge_p0_z, edge_e1_x, edge_e1_y, edge_e1_z,
              static_cast<size_t>(primitive_count) },
            { primitive_bbox_min_x, primitive_bbox_min_y, primitive_bbox_min_z,
              primitive_bbox_max_x, primitive_bbox_max_y, primitive_bbox_max_z,
              static_cast<size_t>(primitive_count) },
            primitive_bounds.get(),
            bounds_stream
        });
        audit_cuda_kernel_launch("compute_primitive_bounds_kernel",
                                 static_cast<uint32_t>(primitive_blocks), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch primitive-bounds kernel");

        size_t reduce_temp_size = 0;
        audit_cub_reduce();
        check_cuda_call(
            cub::DeviceReduce::Reduce(nullptr,
                                      reduce_temp_size,
                                      primitive_bounds.get(),
                                      reduced_bounds.get(),
                                      primitive_count,
                                      BoundsUnion(),
                                      empty_bounds(),
                                      bounds_stream),
            "build_edge_lbvh_gpu(): failed to size scene-bound reduction");
        CudaBuffer<char> reduce_temp(reduce_temp_size);
        audit_cub_reduce();
        check_cuda_call(
            cub::DeviceReduce::Reduce(reduce_temp.get(),
                                      reduce_temp_size,
                                      primitive_bounds.get(),
                                      reduced_bounds.get(),
                                      primitive_count,
                                      BoundsUnion(),
                                      empty_bounds(),
                                      bounds_stream),
            "build_edge_lbvh_gpu(): failed to reduce scene bounds");
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch scene-bound reduction");

        Bounds3 scene_bounds = empty_bounds();
        audit_cuda_memcpy_async();
        check_cuda_call(cudaMemcpyAsync(&scene_bounds,
                                        reduced_bounds.get(),
                                        sizeof(Bounds3),
                                        cudaMemcpyDeviceToHost,
                                        bounds_stream),
                        "build_edge_lbvh_gpu(): failed to copy scene bounds");
        shared::edge::launch_init_sequence_async({
            primitive_indices_in.get(), primitive_count, sequence_stream
        });
        audit_cuda_kernel_launch("init_sequence_kernel",
                                 static_cast<uint32_t>(primitive_blocks), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch primitive-index initialization");
        if (overlap_build_streams) {
            audit_cuda_event_record();
            check_cuda_call(cudaEventRecord(sequence_ready_event.get(), sequence_stream),
                            "build_edge_lbvh_gpu(): failed to record sequence-ready event");
        }
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(bounds_stream),
                        "build_edge_lbvh_gpu(): failed to reduce scene bounds");

        shared::edge::launch_compute_morton_codes_async({
            {
                primitive_bbox_min_x,
                primitive_bbox_min_y,
                primitive_bbox_min_z,
                primitive_bbox_max_x,
                primitive_bbox_max_y,
                primitive_bbox_max_z,
                static_cast<size_t>(primitive_count)
            },
            scene_bounds,
            morton_codes_in.get(),
            bounds_stream
        });
        audit_cuda_kernel_launch("compute_morton_codes_kernel",
                                 static_cast<uint32_t>(primitive_blocks), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch Morton-code kernel");
        if (overlap_build_streams) {
            audit_cuda_stream_wait_event();
            check_cuda_call(cudaStreamWaitEvent(bounds_stream, sequence_ready_event.get(), 0),
                            "build_edge_lbvh_gpu(): failed to join primitive-index stream");
        }

        size_t sort_temp_size = 0;
        audit_cub_sort();
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
        audit_cub_sort();
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
        memset_int_async(merge_counters.get(),
                         0,
                         static_cast<size_t>(std::max(internal_count, 1)),
                         bounds_stream,
                         "build_edge_lbvh_gpu(): failed to init merge counters");

        if (internal_count > 0) {
            shared::edge::launch_build_radix_tree_async({
                morton_codes_out.get(),
                primitive_indices_out.get(),
                left_child,
                right_child,
                parent.get(),
                primitive_count,
                bounds_stream
            });
            audit_cuda_kernel_launch("build_radix_tree_kernel",
                                     static_cast<uint32_t>(internal_blocks), 1, 1,
                                     block_size, 1, 1,
                                     static_cast<uint64_t>(internal_count));
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch radix-tree kernel");
        }

        if (internal_count > 0 && treelet_enabled) {
            audit_cuda_stream_synchronize();
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
            subtree_leaf_counts.assign(static_cast<size_t>(node_count), 1);
            for (int height = 1; height <= max_height; ++height) {
                for (int node_index : host_level_groups[static_cast<size_t>(height)]) {
                    subtree_leaf_counts[static_cast<size_t>(node_index)] =
                        subtree_leaf_counts[static_cast<size_t>(host_left_child[static_cast<size_t>(node_index)])] +
                        subtree_leaf_counts[static_cast<size_t>(host_right_child[static_cast<size_t>(node_index)])];
                }
            }
        }

        shared::edge::launch_finalize_leaves_and_bounds_async({
            primitive_indices_out.get(),
            parent.get(),
            { primitive_bbox_min_x, primitive_bbox_min_y, primitive_bbox_min_z,
              primitive_bbox_max_x, primitive_bbox_max_y, primitive_bbox_max_z,
              static_cast<size_t>(primitive_count) },
            left_child,
            right_child,
            { node_bbox_min_x, node_bbox_min_y, node_bbox_min_z,
              node_bbox_max_x, node_bbox_max_y, node_bbox_max_z,
              static_cast<size_t>(node_count) },
            leaf_primitive,
            is_leaf,
            primitive_leaf_node,
            merge_counters.get(),
            primitive_count,
            bounds_stream
        });
        audit_cuda_kernel_launch("finalize_leaves_and_bounds_kernel",
                                 static_cast<uint32_t>(primitive_blocks), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(primitive_count));
        check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch bounds-finalization kernel");

        if (treelet_enabled) {
            const float scene_scale =
                fmaxf(scene_bounds.max.x - scene_bounds.min.x,
                      fmaxf(scene_bounds.max.y - scene_bounds.min.y,
                            scene_bounds.max.z - scene_bounds.min.z));
            const float inflation =
                fmaxf(scene_scale * EdgeBVHTreeletCostInflationRatio, 1e-6f);
            shared::edge::launch_initialize_leaf_costs_async({
                { node_bbox_min_x, node_bbox_min_y, node_bbox_min_z,
                  node_bbox_max_x, node_bbox_max_y, node_bbox_max_z,
                  static_cast<size_t>(node_count) },
                node_costs.get(),
                inflation,
                primitive_count,
                bounds_stream
            });
            audit_cuda_kernel_launch("initialize_leaf_costs_kernel",
                                     static_cast<uint32_t>(primitive_blocks), 1, 1,
                                     block_size, 1, 1,
                                     static_cast<uint64_t>(primitive_count));
            check_cuda_last_error("build_edge_lbvh_gpu(): failed to launch leaf-cost initialization");
            CudaBuffer<int> internal_cost_arrival_counter(static_cast<size_t>(node_count));
            memset_int_async(internal_cost_arrival_counter.get(),
                             0,
                             static_cast<size_t>(node_count),
                             bounds_stream,
                             "build_edge_lbvh_gpu(): failed to init internal-cost arrivals");
            if (internal_count > 0) {
                shared::edge::launch_initialize_internal_costs_async({
                    left_child,
                    right_child,
                    parent.get(),
                    { node_bbox_min_x, node_bbox_min_y, node_bbox_min_z,
                      node_bbox_max_x, node_bbox_max_y, node_bbox_max_z,
                      static_cast<size_t>(node_count) },
                    node_costs.get(),
                    internal_cost_arrival_counter.get(),
                    inflation,
                    primitive_count,
                    bounds_stream
                });
                audit_cuda_kernel_launch("initialize_internal_costs_kernel",
                                         static_cast<uint32_t>(primitive_blocks), 1, 1,
                                         block_size, 1, 1,
                                         static_cast<uint64_t>(primitive_count));
                check_cuda_last_error(
                    "build_edge_lbvh_gpu(): failed to launch internal-cost initialization");
            }

            {
                audit_cuda_stream_synchronize();
                check_cuda_call(cudaStreamSynchronize(bounds_stream),
                                "build_edge_lbvh_gpu(): failed to finalize node bounds");
                std::vector<std::vector<int>> optimize_levels(static_cast<size_t>(max_height + 1));
                for (int node_index = 0; node_index < internal_count; ++node_index) {
                    const int height = node_heights[static_cast<size_t>(node_index)];
                    if (subtree_leaf_counts[static_cast<size_t>(node_index)] >=
                        EdgeBVHTreeletMinSubtreeLeaves) {
                        optimize_levels[static_cast<size_t>(height)].push_back(node_index);
                    }
                }
                optimize_schedule = flatten_node_levels(optimize_levels);
                optimize_nodes_device.allocate(optimize_schedule.nodes.size());
                if (!optimize_schedule.nodes.empty()) {
                    // Asynchronous on the build stream rather than blocking on
                    // the default stream: bounds_stream is non-blocking, so a
                    // default-stream copy is not ordered against the cost-init
                    // kernels queued just above, and the blocking form serialized
                    // the whole build on the host for no reason. The event makes
                    // the schedule -> treelet-kernel dependency explicit.
                    audit_cuda_memcpy_async();
                    check_cuda_call(cudaMemcpyAsync(optimize_nodes_device.get(),
                                                    optimize_schedule.nodes.data(),
                                                    optimize_schedule.nodes.size() * sizeof(int),
                                                    cudaMemcpyHostToDevice,
                                                    bounds_stream),
                                    "build_edge_lbvh_gpu(): failed to upload treelet schedule");
                    audit_cuda_event_record();
                    check_cuda_call(cudaEventRecord(schedule_uploaded_event.get(), bounds_stream),
                                    "build_edge_lbvh_gpu(): failed to record schedule-upload event");
                    audit_cuda_stream_wait_event();
                    check_cuda_call(
                        cudaStreamWaitEvent(bounds_stream, schedule_uploaded_event.get(), 0),
                        "build_edge_lbvh_gpu(): failed to order the treelet schedule upload");
                }

                for (int height = 1; height <= max_height; ++height) {
                    const int optimize_start =
                        optimize_schedule.level_offsets[static_cast<size_t>(height)];
                    const int optimize_end =
                        optimize_schedule.level_offsets[static_cast<size_t>(height + 1)];
                    const int optimize_count = optimize_end - optimize_start;
                    if (optimize_count > 0) {
                        const int level_blocks = (optimize_count + block_size - 1) / block_size;
                        shared::edge::launch_optimize_selected_treelets_async({
                            optimize_nodes_device.get() + optimize_start,
                            is_leaf,
                            left_child,
                            right_child,
                            parent.get(),
                            { node_bbox_min_x, node_bbox_min_y, node_bbox_min_z,
                              node_bbox_max_x, node_bbox_max_y, node_bbox_max_z,
                              static_cast<size_t>(node_count) },
                            leaf_primitive,
                            node_costs.get(),
                            inflation,
                            optimize_count,
                            bounds_stream
                        });
                        audit_cuda_kernel_launch("optimize_selected_treelets_kernel",
                                                 static_cast<uint32_t>(level_blocks), 1, 1,
                                                 block_size, 1, 1,
                                                 static_cast<uint64_t>(optimize_count));
                        check_cuda_last_error(
                            "build_edge_lbvh_gpu(): failed to launch GPU treelet optimization");
                    }
                }
            }
        }

        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(bounds_stream),
                        "build_edge_lbvh_gpu(): failed to complete build");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("build_edge_lbvh_gpu(): ") + e.what());
    }
}

void mark_edge_bvh_dirty_ancestors_gpu(
    const EdgeBvhCudaContext &context,
    int node_count,
    int leaf_count,
    const int *leaf_nodes,
    const int *node_parent,
    int *out_dirty_marks,
    bool clear_marks) {
    require_local(node_count >= 0,
                  "mark_edge_bvh_dirty_ancestors_gpu(): node_count must be non-negative.");
    require_local(leaf_count >= 0,
                  "mark_edge_bvh_dirty_ancestors_gpu(): leaf_count must be non-negative.");
    require_local(node_parent != nullptr || node_count == 0,
                  "mark_edge_bvh_dirty_ancestors_gpu(): node_parent pointer is null.");
    require_local(out_dirty_marks != nullptr || node_count == 0,
                  "mark_edge_bvh_dirty_ancestors_gpu(): output mark pointer is null.");
    require_local(leaf_nodes != nullptr || leaf_count == 0,
                  "mark_edge_bvh_dirty_ancestors_gpu(): leaf_nodes pointer is null.");

    try {
        if (node_count == 0) {
            return;
        }

        const CudaDeviceGuard device_guard(context.device);

        if (clear_marks) {
            memset_int_async(out_dirty_marks,
                             0,
                             static_cast<size_t>(node_count),
                             context.stream,
                             "mark_edge_bvh_dirty_ancestors_gpu(): failed to clear dirty marks");
        }

        if (leaf_count == 0) {
            return;
        }

        constexpr int block_size = 256;
        const int block_count = (leaf_count + block_size - 1) / block_size;
        // The clear above and this kernel are ordered by the caller's stream,
        // which also carries the Dr.Jit buffers they touch; the default stream
        // was ordered against neither.
        shared::edge::launch_mark_dirty_ancestors_async({
            leaf_nodes,
            node_parent,
            out_dirty_marks,
            leaf_count,
            context.stream
        });
        audit_cuda_kernel_launch("mark_dirty_ancestors_kernel",
                                 static_cast<uint32_t>(block_count), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(leaf_count));
        check_cuda_last_error(
            "mark_edge_bvh_dirty_ancestors_gpu(): failed to launch dirty-ancestor kernel");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("mark_edge_bvh_dirty_ancestors_gpu(): ") + e.what());
    }
}

void compact_and_refit_edge_bvh_level_gpu(
    const EdgeBvhCudaContext &context,
    int level_count,
    const int *level_nodes,
    const int *dirty_marks,
    int *scratch_selected_nodes,
    int *scratch_selected_count,
    const int *left_child,
    const int *right_child,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z) {
    require_local(level_count >= 0,
                  "compact_and_refit_edge_bvh_level_gpu(): level_count must be non-negative.");
    require_local(level_nodes != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): level_nodes pointer is null.");
    require_local(dirty_marks != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): dirty_marks pointer is null.");
    require_local(scratch_selected_nodes != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): scratch_selected_nodes pointer is null.");
    require_local(scratch_selected_count != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): scratch_selected_count pointer is null.");
    require_local(left_child != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): left_child pointer is null.");
    require_local(right_child != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): right_child pointer is null.");
    require_local(node_bbox_min_x != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): node_bbox_min_x pointer is null.");
    require_local(node_bbox_min_y != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): node_bbox_min_y pointer is null.");
    require_local(node_bbox_min_z != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): node_bbox_min_z pointer is null.");
    require_local(node_bbox_max_x != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): node_bbox_max_x pointer is null.");
    require_local(node_bbox_max_y != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): node_bbox_max_y pointer is null.");
    require_local(node_bbox_max_z != nullptr || level_count == 0,
                  "compact_and_refit_edge_bvh_level_gpu(): node_bbox_max_z pointer is null.");

    try {
        if (level_count == 0) {
            return;
        }

        const CudaDeviceGuard device_guard(context.device);

        memset_int_async(scratch_selected_count,
                         0,
                         1,
                         context.stream,
                         "compact_and_refit_edge_bvh_level_gpu(): failed to clear selected count");

        constexpr int block_size = 256;
        const int block_count = (level_count + block_size - 1) / block_size;
        // Clear, compaction, and refit form a chain through
        // scratch_selected_count/scratch_selected_nodes; keeping all three on the
        // caller's stream is what orders them against each other and against the
        // Dr.Jit work that produced the level and mark buffers.
        shared::edge::launch_compact_dirty_level_async({
            level_nodes,
            dirty_marks,
            scratch_selected_nodes,
            scratch_selected_count,
            level_count,
            context.stream
        });
        audit_cuda_kernel_launch("compact_dirty_level_kernel",
                                 static_cast<uint32_t>(block_count), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(level_count));
        check_cuda_last_error(
            "compact_and_refit_edge_bvh_level_gpu(): failed to launch dirty-level compaction");

        shared::edge::launch_refit_selected_internal_nodes_async({
            scratch_selected_count,
            scratch_selected_nodes,
            left_child,
            right_child,
            {
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                0
            },
            level_count,
            context.stream
        });
        audit_cuda_kernel_launch("refit_selected_internal_nodes_kernel",
                                 static_cast<uint32_t>(block_count), 1, 1,
                                 block_size, 1, 1,
                                 static_cast<uint64_t>(level_count));
        check_cuda_last_error(
            "compact_and_refit_edge_bvh_level_gpu(): failed to launch dirty-level refit");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("compact_and_refit_edge_bvh_level_gpu(): ") + e.what());
    }
}

} // namespace rayd
