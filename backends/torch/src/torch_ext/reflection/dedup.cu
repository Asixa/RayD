#include <rayd/torch/reflection/dedup.h>
#include <rayd/shared/multipath/reflection_dedup.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <string>

#include <rayd/torch/common/native_compat.h>


namespace rayd::torch_backend {

namespace {

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

        const cudaError_t error =
            cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_);
        require(error == cudaSuccess,
                std::string("reflection_dedup_gpu(): cudaMalloc failed: ") +
                    cudaGetErrorString(error));
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

} // namespace

int reflection_dedup_gpu(
    int n_rays,
    int max_bounces,
    const int *bounce_count,
    const int *shape_ids,
    const int *prim_ids,
    const float *t,
    const float *bary_u,
    const float *bary_v,
    const float *hit_x,
    const float *hit_y,
    const float *hit_z,
    const float *norm_x,
    const float *norm_y,
    const float *norm_z,
    const float *img_x,
    const float *img_y,
    const float *img_z,
    const int *face_offsets,
    int n_meshes,
    const int *canonical_prim_table,
    int canonical_table_size,
    float image_source_tolerance,
    int *out_bounce_count,
    int *out_shape_ids,
    int *out_prim_ids,
    float *out_t,
    float *out_bary_u,
    float *out_bary_v,
    float *out_hit_x,
    float *out_hit_y,
    float *out_hit_z,
    float *out_norm_x,
    float *out_norm_y,
    float *out_norm_z,
    float *out_img_x,
    float *out_img_y,
    float *out_img_z,
    int *out_discovery_count,
    int *out_representative_ray_index) {
    require(n_rays >= 0, "reflection_dedup_gpu(): n_rays must be non-negative.");
    require(max_bounces > 0,
            "reflection_dedup_gpu(): max_bounces must be positive.");

    if (n_rays == 0) {
        return 0;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());

    constexpr int block_size = 256;
    const int block_count = (n_rays + block_size - 1) / block_size;

    CudaBuffer<uint64_t> keys_in(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> keys_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> ray_indices_in(static_cast<size_t>(n_rays));
    CudaBuffer<int> ray_indices_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> boundary_flags(static_cast<size_t>(n_rays));
    CudaBuffer<int> hash_group_ids(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> cluster_keys_in(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> cluster_keys_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> cluster_ray_indices_in(static_cast<size_t>(n_rays));
    CudaBuffer<int> cluster_ray_indices_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> unique_path_ids(static_cast<size_t>(n_rays));
    CudaBuffer<int> unique_count_device(1);

    check_cuda_call(cudaMemsetAsync(out_discovery_count,
                                    0,
                                    sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear discovery counts");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(out_representative_ray_index,
                                    0xFF,
                                    sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear representative indices");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(unique_count_device.get(),
                                    0,
                                    sizeof(int),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear unique counter");
    audit_cuda_memset_async();

    audit_cuda_kernel_launch("reflection_dedup_build_keys_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_build_keys({
        n_rays,
        max_bounces,
        bounce_count,
        shape_ids,
        prim_ids,
        face_offsets,
        n_meshes,
        canonical_prim_table,
        canonical_table_size,
        keys_in.get(),
        ray_indices_in.get(),
        stream
    });
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch build-keys kernel");

    size_t sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    sort_temp_size,
                                                    keys_in.get(),
                                                    keys_out.get(),
                                                    ray_indices_in.get(),
                                                    ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to size first radix sort");
    CudaBuffer<char> sort_temp(std::max<size_t>(sort_temp_size, 1));
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(sort_temp.get(),
                                                    sort_temp_size,
                                                    keys_in.get(),
                                                    keys_out.get(),
                                                    ray_indices_in.get(),
                                                    ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to run first radix sort");

    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_mark_boundaries(
        n_rays, keys_out.get(), boundary_flags.get(), stream);
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch first boundary kernel");

    size_t scan_temp_size = 0;
    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(nullptr,
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  hash_group_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to size first scan");
    CudaBuffer<char> scan_temp(std::max<size_t>(scan_temp_size, 1));
    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(scan_temp.get(),
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  hash_group_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to run first scan");
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_zero_base_ids(
        n_rays, keys_out.get(), hash_group_ids.get(), stream);
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch first id-fix kernel");

    audit_cuda_kernel_launch("reflection_dedup_sub_cluster_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_sub_cluster({
        n_rays,
        max_bounces,
        keys_out.get(),
        ray_indices_out.get(),
        hash_group_ids.get(),
        bounce_count,
        img_x,
        img_y,
        img_z,
        image_source_tolerance,
        cluster_keys_in.get(),
        cluster_ray_indices_in.get(),
        stream
    });
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch sub-cluster kernel");

    size_t cluster_sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    cluster_sort_temp_size,
                                                    cluster_keys_in.get(),
                                                    cluster_keys_out.get(),
                                                    cluster_ray_indices_in.get(),
                                                    cluster_ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to size second radix sort");
    CudaBuffer<char> cluster_sort_temp(std::max<size_t>(cluster_sort_temp_size, 1));
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(cluster_sort_temp.get(),
                                                    cluster_sort_temp_size,
                                                    cluster_keys_in.get(),
                                                    cluster_keys_out.get(),
                                                    cluster_ray_indices_in.get(),
                                                    cluster_ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to run second radix sort");

    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_mark_boundaries(
        n_rays, cluster_keys_out.get(), boundary_flags.get(), stream);
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch second boundary kernel");

    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(scan_temp.get(),
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  unique_path_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to run second scan");
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_zero_base_ids(
        n_rays, cluster_keys_out.get(), unique_path_ids.get(), stream);
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch second id-fix kernel");

    audit_cuda_kernel_launch("reflection_dedup_compact_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    shared::multipath::launch_reflection_dedup_compact({
        n_rays,
        max_bounces,
        cluster_keys_out.get(),
        cluster_ray_indices_out.get(),
        unique_path_ids.get(),
        bounce_count,
        shape_ids,
        prim_ids,
        t,
        bary_u,
        bary_v,
        hit_x,
        hit_y,
        hit_z,
        norm_x,
        norm_y,
        norm_z,
        img_x,
        img_y,
        img_z,
        unique_count_device.get(),
        out_bounce_count,
        out_shape_ids,
        out_prim_ids,
        out_t,
        out_bary_u,
        out_bary_v,
        out_hit_x,
        out_hit_y,
        out_hit_z,
        out_norm_x,
        out_norm_y,
        out_norm_z,
        out_img_x,
        out_img_y,
        out_img_z,
        out_discovery_count,
        out_representative_ray_index,
        stream
    });
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch compact kernel");

    int unique_count = 0;
    audit_cuda_memcpy_async();
    check_cuda_call(cudaMemcpyAsync(&unique_count,
                                    unique_count_device.get(),
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "reflection_dedup_gpu(): failed to copy unique count");
    audit_cuda_stream_synchronize();
    check_cuda_call(cudaStreamSynchronize(stream),
                    "reflection_dedup_gpu(): failed to finish dedup stream");
    return unique_count;
}

} // namespace rayd::torch_backend
