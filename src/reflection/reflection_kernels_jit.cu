// Copyright Xingyu Chen.
// Implements reflection support for reflection kernels Dr.Jit.

#include <src/reflection/dedup_jit.h>
#include <rayd/reflection/dedup.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <string>

#include <rayd/jit/core.h>
#include <rayd/jit/native_launch_audit.h>

namespace rayd {

namespace {

template <typename T> class CudaBuffer {
  public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) { allocate(count); }

    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
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

        const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_);
        require(error == cudaSuccess,
                std::string("reflection_dedup_gpu(): cudaMalloc failed: ") + cudaGetErrorString(error));
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

  private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

void check_cuda_call(cudaError_t error, const char* message) {
    require(error == cudaSuccess, std::string(message) + ": " + cudaGetErrorString(error));
}

// Runs the sort/scan passes the shared sequence delegates back to this
// backend. Keeping the CUB calls here keeps their template kernels
// instantiated in this translation unit, exactly as before the sequence
// orchestration moved to the shared layer.
cudaError_t run_dedup_pass(const shared::multipath::ReflectionDedupSequenceParams& params,
                           shared::multipath::ReflectionDedupDevicePass pass) {
    using shared::multipath::ReflectionDedupDevicePass;
    size_t sort_temp_bytes = params.sort_temp_bytes;
    size_t scan_temp_bytes = params.scan_temp_bytes;
    size_t cluster_sort_temp_bytes = params.cluster_sort_temp_bytes;
    switch (pass) {
    case ReflectionDedupDevicePass::kFirstSort:
        return cub::DeviceRadixSort::SortPairs(params.sort_temp, sort_temp_bytes, params.keys_in, params.keys_out,
                                               params.ray_indices_in, params.ray_indices_out, params.ray_count, 0, 64,
                                               params.stream);
    case ReflectionDedupDevicePass::kFirstScan:
        return cub::DeviceScan::InclusiveSum(params.scan_temp, scan_temp_bytes, params.boundary_flags,
                                             params.hash_group_ids, params.ray_count, params.stream);
    case ReflectionDedupDevicePass::kSecondSort:
        return cub::DeviceRadixSort::SortPairs(params.cluster_sort_temp, cluster_sort_temp_bytes,
                                               params.cluster_keys_in, params.cluster_keys_out,
                                               params.cluster_ray_indices_in, params.cluster_ray_indices_out,
                                               params.ray_count, 0, 64, params.stream);
    case ReflectionDedupDevicePass::kSecondScan:
        return cub::DeviceScan::InclusiveSum(params.scan_temp, scan_temp_bytes, params.boundary_flags,
                                             params.unique_path_ids, params.ray_count, params.stream);
    }
    return cudaErrorInvalidValue;
}

// Per-step error strings stay in this backend verbatim; the shared sequence
// only reports which step produced the failing CUDA result.
const char* sequence_step_message(shared::multipath::ReflectionDedupSequenceStep step) {
    using shared::multipath::ReflectionDedupSequenceStep;
    switch (step) {
    case ReflectionDedupSequenceStep::kBuildKeys:
        return "reflection_dedup_gpu(): failed to launch build-keys kernel";
    case ReflectionDedupSequenceStep::kFirstSort:
        return "reflection_dedup_gpu(): failed to run first radix sort";
    case ReflectionDedupSequenceStep::kFirstBoundaries:
        return "reflection_dedup_gpu(): failed to launch first boundary kernel";
    case ReflectionDedupSequenceStep::kFirstScan:
        return "reflection_dedup_gpu(): failed to run first scan";
    case ReflectionDedupSequenceStep::kFirstZeroBase:
        return "reflection_dedup_gpu(): failed to launch first id-fix kernel";
    case ReflectionDedupSequenceStep::kSubCluster:
        return "reflection_dedup_gpu(): failed to launch sub-cluster kernel";
    case ReflectionDedupSequenceStep::kSecondSort:
        return "reflection_dedup_gpu(): failed to run second radix sort";
    case ReflectionDedupSequenceStep::kSecondBoundaries:
        return "reflection_dedup_gpu(): failed to launch second boundary kernel";
    case ReflectionDedupSequenceStep::kSecondScan:
        return "reflection_dedup_gpu(): failed to run second scan";
    case ReflectionDedupSequenceStep::kSecondZeroBase:
        return "reflection_dedup_gpu(): failed to launch second id-fix kernel";
    case ReflectionDedupSequenceStep::kCompact:
        return "reflection_dedup_gpu(): failed to launch compact kernel";
    case ReflectionDedupSequenceStep::kNone:
        break;
    }
    return "reflection_dedup_gpu(): dedup sequence failed";
}

void check_sequence_status(const shared::multipath::ReflectionDedupSequenceStatus& status) {
    check_cuda_call(status.error, sequence_step_message(status.step));
}

} // namespace

int reflection_dedup_gpu(int n_rays, int max_bounces, const int* bounce_count, const int* shape_ids,
                         const int* prim_ids, const float* t, const float* bary_u, const float* bary_v,
                         const float* hit_x, const float* hit_y, const float* hit_z, const float* norm_x,
                         const float* norm_y, const float* norm_z, const float* img_x, const float* img_y,
                         const float* img_z, const int* face_offsets, int n_meshes, const int* canonical_prim_table,
                         int canonical_table_size, float image_source_tolerance, int* out_bounce_count,
                         int* out_shape_ids, int* out_prim_ids, float* out_t, float* out_bary_u, float* out_bary_v,
                         float* out_hit_x, float* out_hit_y, float* out_hit_z, float* out_norm_x, float* out_norm_y,
                         float* out_norm_z, float* out_img_x, float* out_img_y, float* out_img_z,
                         int* out_discovery_count, int* out_representative_ray_index) {
    require(n_rays >= 0, "reflection_dedup_gpu(): n_rays must be non-negative.");
    require(max_bounces > 0, "reflection_dedup_gpu(): max_bounces must be positive.");

    if (n_rays == 0) {
        return 0;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    require(stream != nullptr, "reflection_dedup_gpu(): CUDA stream is unavailable.");

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

    check_cuda_call(cudaMemsetAsync(out_discovery_count, 0, sizeof(int) * static_cast<size_t>(n_rays), stream),
                    "reflection_dedup_gpu(): failed to clear discovery counts");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(out_representative_ray_index, 0xFF, sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear representative indices");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(unique_count_device.get(), 0, sizeof(int), stream),
                    "reflection_dedup_gpu(): failed to clear unique counter");
    audit_cuda_memset_async();

    size_t sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_size, keys_in.get(), keys_out.get(),
                                                    ray_indices_in.get(), ray_indices_out.get(), n_rays, 0, 64, stream),
                    "reflection_dedup_gpu(): failed to size first radix sort");
    CudaBuffer<char> sort_temp(std::max<size_t>(sort_temp_size, 1));

    size_t scan_temp_size = 0;
    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(nullptr, scan_temp_size, boundary_flags.get(), hash_group_ids.get(),
                                                  n_rays, stream),
                    "reflection_dedup_gpu(): failed to size first scan");
    CudaBuffer<char> scan_temp(std::max<size_t>(scan_temp_size, 1));

    size_t cluster_sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr, cluster_sort_temp_size, cluster_keys_in.get(),
                                                    cluster_keys_out.get(), cluster_ray_indices_in.get(),
                                                    cluster_ray_indices_out.get(), n_rays, 0, 64, stream),
                    "reflection_dedup_gpu(): failed to size second radix sort");
    CudaBuffer<char> cluster_sort_temp(std::max<size_t>(cluster_sort_temp_size, 1));

    audit_cuda_kernel_launch("reflection_dedup_build_keys_kernel", static_cast<uint32_t>(block_count), 1, 1, block_size,
                             1, 1, static_cast<uint64_t>(n_rays));
    audit_cub_sort();
    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel", static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1, static_cast<uint64_t>(n_rays));
    audit_cub_scan();
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel", static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1, static_cast<uint64_t>(n_rays));
    audit_cuda_kernel_launch("reflection_dedup_sub_cluster_kernel", static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1, static_cast<uint64_t>(n_rays));
    audit_cub_sort();
    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel", static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1, static_cast<uint64_t>(n_rays));
    audit_cub_scan();
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel", static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1, static_cast<uint64_t>(n_rays));
    audit_cuda_kernel_launch("reflection_dedup_compact_kernel", static_cast<uint32_t>(block_count), 1, 1, block_size, 1,
                             1, static_cast<uint64_t>(n_rays));

    shared::multipath::ReflectionDedupSequenceParams sequence{};
    sequence.ray_count = n_rays;
    sequence.max_bounces = max_bounces;
    sequence.bounce_count = bounce_count;
    sequence.shape_ids = shape_ids;
    sequence.prim_ids = prim_ids;
    sequence.face_offsets = face_offsets;
    sequence.mesh_count = n_meshes;
    sequence.canonical_table = canonical_prim_table;
    sequence.canonical_table_size = canonical_table_size;
    sequence.image_source_tolerance = image_source_tolerance;
    sequence.raw_t = t;
    sequence.raw_bary_u = bary_u;
    sequence.raw_bary_v = bary_v;
    sequence.raw_hit_x = hit_x;
    sequence.raw_hit_y = hit_y;
    sequence.raw_hit_z = hit_z;
    sequence.raw_norm_x = norm_x;
    sequence.raw_norm_y = norm_y;
    sequence.raw_norm_z = norm_z;
    sequence.raw_image_x = img_x;
    sequence.raw_image_y = img_y;
    sequence.raw_image_z = img_z;
    sequence.keys_in = keys_in.get();
    sequence.keys_out = keys_out.get();
    sequence.ray_indices_in = ray_indices_in.get();
    sequence.ray_indices_out = ray_indices_out.get();
    sequence.boundary_flags = boundary_flags.get();
    sequence.hash_group_ids = hash_group_ids.get();
    sequence.cluster_keys_in = cluster_keys_in.get();
    sequence.cluster_keys_out = cluster_keys_out.get();
    sequence.cluster_ray_indices_in = cluster_ray_indices_in.get();
    sequence.cluster_ray_indices_out = cluster_ray_indices_out.get();
    sequence.unique_path_ids = unique_path_ids.get();
    sequence.sort_temp = sort_temp.get();
    sequence.sort_temp_bytes = sort_temp_size;
    sequence.scan_temp = scan_temp.get();
    sequence.scan_temp_bytes = scan_temp_size;
    sequence.cluster_sort_temp = cluster_sort_temp.get();
    sequence.cluster_sort_temp_bytes = cluster_sort_temp_size;
    sequence.out_unique_count = unique_count_device.get();
    sequence.out_bounce_count = out_bounce_count;
    sequence.out_shape_ids = out_shape_ids;
    sequence.out_prim_ids = out_prim_ids;
    sequence.out_t = out_t;
    sequence.out_bary_u = out_bary_u;
    sequence.out_bary_v = out_bary_v;
    sequence.out_hit_x = out_hit_x;
    sequence.out_hit_y = out_hit_y;
    sequence.out_hit_z = out_hit_z;
    sequence.out_norm_x = out_norm_x;
    sequence.out_norm_y = out_norm_y;
    sequence.out_norm_z = out_norm_z;
    sequence.out_image_x = out_img_x;
    sequence.out_image_y = out_img_y;
    sequence.out_image_z = out_img_z;
    sequence.out_discovery_count = out_discovery_count;
    sequence.out_representative_ray_index = out_representative_ray_index;
    sequence.run_pass = &run_dedup_pass;
    sequence.stream = stream;
    check_sequence_status(shared::multipath::launch_reflection_dedup_sequence(sequence));

    int unique_count = 0;
    audit_cuda_memcpy_async();
    check_cuda_call(cudaMemcpyAsync(&unique_count, unique_count_device.get(), sizeof(int), cudaMemcpyDeviceToHost,
                                    stream),
                    "reflection_dedup_gpu(): failed to copy unique count");
    audit_cuda_stream_synchronize();
    check_cuda_call(cudaStreamSynchronize(stream), "reflection_dedup_gpu(): failed to finish dedup stream");
    return unique_count;
}

} // namespace rayd

// Consolidated reflection EPC field kernels.
#include <src/reflection/epc_field_jit.h>
#include <rayd/contracts.h>
#include <rayd/math.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include <rayd/jit/core.h>

#include <rayd/jit/native_launch_audit.h>

namespace rayd {

namespace {

constexpr float kSmallEps = shared::SmallEpsilon;
constexpr float kPi = 3.14159265358979323846f;

using namespace shared::field;

using namespace shared::cuda_math;

static __forceinline__ __device__ bool slot_reflection_coefficients(const ReflEpcFieldParams params, int slot,
                                                                    float cos_theta, Complex& r_te, Complex& r_tm) {
    return fresnel_reflection_coefficients(params.slot_eta_r[slot], params.slot_sigma[slot], params.slot_mu_r[slot],
                                           params.slot_gain[slot], params.omega, cos_theta, r_te, r_tm, kSmallEps);
}

static __forceinline__ __device__ void store_zero_field(const ReflEpcFieldParams params, int ray_index) {
    params.out_valid[ray_index] = 0u;
    params.out_field_x_re[ray_index] = 0.f;
    params.out_field_x_im[ray_index] = 0.f;
    params.out_field_y_re[ray_index] = 0.f;
    params.out_field_y_im[ray_index] = 0.f;
    params.out_field_z_re[ray_index] = 0.f;
    params.out_field_z_im[ray_index] = 0.f;
}

// Identifier/storage layer for the shared EPC field device body. Every macro
// expands to the exact pre-dedup expression of this backend: dense reads with
// no null tests, no extra prologue exports, and unconditional output writes.
#define RAYD_REFL_EPC_MAKE3(x, y, z) make_vec3(x, y, z)
#define RAYD_REFL_EPC_EPS kSmallEps
#define RAYD_REFL_EPC_FIELD_PROLOGUE(P, RAY, BASE)
#define RAYD_REFL_EPC_LOAD_TX_POLARIZATION(P, RAY)                                                                     \
    const int tx_pol_index = (P).tx_pol_count == 1 ? 0 : (RAY);                                                        \
    const float3 tx_polarization =                                                                                     \
        make_vec3((P).tx_pol_x[tx_pol_index], (P).tx_pol_y[tx_pol_index], (P).tx_pol_z[tx_pol_index]);
#define RAYD_REFL_EPC_STORE_FIELD(P, RAY, FIELD)                                                                       \
    (P).out_valid[(RAY)] = 1u;                                                                                         \
    (P).out_field_x_re[(RAY)] = (FIELD).x.re;                                                                          \
    (P).out_field_x_im[(RAY)] = (FIELD).x.im;                                                                          \
    (P).out_field_y_re[(RAY)] = (FIELD).y.re;                                                                          \
    (P).out_field_y_im[(RAY)] = (FIELD).y.im;                                                                          \
    (P).out_field_z_re[(RAY)] = (FIELD).z.re;                                                                          \
    (P).out_field_z_im[(RAY)] = (FIELD).z.im;

#include <rayd/reflection/epc_field_device.cuh>

void check_epc_field_cuda_call(cudaError_t error, const char* message) {
    require(error == cudaSuccess, std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char* message) {
    check_epc_field_cuda_call(cudaGetLastError(), message);
}

} // namespace

void reflection_epc_field_gpu(const ReflEpcFieldParams& params) {
    require(params.n_rays >= 0, "reflection_epc_field_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0, "reflection_epc_field_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    require(stream != nullptr, "reflection_epc_field_gpu(): CUDA stream is unavailable.");

    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_field_kernel", static_cast<uint32_t>(block_count), 1, 1,
                             static_cast<uint32_t>(block_size), 1, 1, static_cast<uint64_t>(params.n_rays));
    reflection_epc_field_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error("reflection_epc_field_gpu(): failed to launch field kernel");
}

} // namespace rayd
