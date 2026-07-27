#include <rayd/torch/diffraction/accum_ad.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <rayd/torch/common/math.cuh>
#include <rayd/torch/common/native_compat.h>

namespace rayd::torch_backend {

namespace {

constexpr float kDfrEps = 1e-6f;

static __forceinline__ __device__ float read_f32_strided_or_zero(
    const float *ptr,
    int stride,
    int index) {
    return ptr != nullptr ? ptr[index * stride] : 0.f;
}

static __forceinline__ __device__ int read_i32_strided_or_default(
    const int *ptr,
    int stride,
    int index,
    int default_value) {
    return ptr != nullptr ? ptr[index * stride] : default_value;
}

static __forceinline__ __device__ bool read_u8_strided_or_false(
    const uint8_t *ptr,
    int stride,
    int index) {
    return ptr != nullptr && ptr[index * stride] != 0u;
}

static __forceinline__ __device__ float3 read_vec_strided_or_zero(
    const float *x,
    const float *y,
    const float *z,
    int stride,
    int index) {
    return make_f3(read_f32_strided_or_zero(x, stride, index),
                   read_f32_strided_or_zero(y, stride, index),
                   read_f32_strided_or_zero(z, stride, index));
}

static __forceinline__ __device__ float read_grid_or_zero(
    const float *ptr,
    int rank,
    int stride0,
    int stride1,
    int resolution0,
    int cell) {
    if (ptr == nullptr) {
        return 0.f;
    }
    if (rank == 2) {
        const int x = cell % resolution0;
        const int y = cell / resolution0;
        return ptr[y * stride0 + x * stride1];
    }
    return ptr[cell * stride0];
}

static __forceinline__ __device__ void atomic_add_strided(
    float *ptr,
    int stride,
    int index,
    float value) {
    if (ptr != nullptr) {
        atomicAdd(ptr + index * stride, value);
    }
}

static __forceinline__ __device__ void atomic_add_vec_strided(
    float *x,
    float *y,
    float *z,
    int stride,
    int index,
    float3 value) {
    atomic_add_strided(x, stride, index, value.x);
    atomic_add_strided(y, stride, index, value.y);
    atomic_add_strided(z, stride, index, value.z);
}

static __forceinline__ __device__ float material_gain_for_prim(
    const DfrDirectAccumADParams &params,
    int prim) {
    return params.material_gain != nullptr
               ? read_f32_strided_or_zero(params.material_gain, params.material_gain_stride, prim)
               : 1.f;
}

static __forceinline__ __device__ float material_gain_for_prim(
    const DfrChainAccumADParams &params,
    int prim) {
    return params.material_gain != nullptr
               ? read_f32_strided_or_zero(params.material_gain, params.material_gain_stride, prim)
               : 1.f;
}

// Nullable strided storage-access layer for the shared AD device body. Every
// macro expands to the exact pre-dedup Torch expression: reads go through the
// strided-or-default helpers above, gradient outputs carry their stride in a
// DfrGradSlot, and the dense tri gradients keep an inert stride of 1.
#define RAYD_DFR_AD_READ_F32(P, F, S, I) \
    read_f32_strided_or_zero((P).F, (P).S, (I))
#define RAYD_DFR_AD_READ_I32(P, F, S, I) \
    read_i32_strided_or_default((P).F, (P).S, (I), -1)
#define RAYD_DFR_AD_READ_I32_OR(P, F, S, I, D) \
    read_i32_strided_or_default((P).F, (P).S, (I), (D))
#define RAYD_DFR_AD_READ_VEC(P, X, Y, Z, S, I) \
    read_vec_strided_or_zero((P).X, (P).Y, (P).Z, (P).S, (I))
#define RAYD_DFR_AD_READ_VEC_GUARDED(P, X, Y, Z, S, I) \
    read_vec_strided_or_zero((P).X, (P).Y, (P).Z, (P).S, (I))
#define RAYD_DFR_AD_READ_DOT_F32(P, F, S, I) \
    read_f32_strided_or_zero((P).F, (P).S, (I))
#define RAYD_DFR_AD_READ_DOT_VEC(P, X, Y, Z, S, I) \
    read_vec_strided_or_zero((P).X, (P).Y, (P).Z, (P).S, (I))
#define RAYD_DFR_AD_MATERIAL_VALID_ENTRY(P, I) \
    read_u8_strided_or_false((P).material_valid, (P).material_valid_stride, (I))
#define RAYD_DFR_AD_MATERIAL_GAIN_TAPE(P, I) \
    read_f32_strided_or_zero((P).material_gain, (P).material_gain_stride, (I))
#define RAYD_DFR_AD_MATERIAL_GAIN_EVENT(P, I) material_gain_for_prim((P), (I))
#define RAYD_DFR_AD_MATERIAL_GAIN_OR_ONE(P, I) material_gain_for_prim((P), (I))
#define RAYD_DFR_AD_SUFFIX_FACE_PRIM(P, F, S, HAS_THIRD, SECOND, THIRD) \
    read_i32_strided_or_default((P).F, (P).S, (HAS_THIRD) ? (THIRD) : (SECOND), -1)

#include <rayd/shared/multipath/diffraction_accumulation_ad_device.cuh>

static __forceinline__ __device__ void add_chain_unit_vjp(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float grad_contribution,
    float *ptr,
    int stride,
    int index,
    const ChainTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = chain_contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index * stride, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void add_unit_vjp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution,
    float *ptr,
    int index,
    const DfrTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void add_unit_vjp_strided(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution,
    float *ptr,
    int stride,
    int index,
    const DfrTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index * stride, grad_contribution * partial);
    }
}

// The strided gradient-write layer: every unit-VJP call site expands to the
// pre-dedup Torch call, including the dense (stride-free for direct, literal-1
// stride for chain) tri gradient writes.
#define RAYD_DFR_AD_ADD_UNIT_VJP(P, PR, G, F, S, I, T) \
    add_unit_vjp_strided((P), (PR), (G), (P).F, (P).S, (I), (T))
#define RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(P, PR, G, F, I, T) \
    add_unit_vjp((P), (PR), (G), (P).F, (I), (T))
#define RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(P, PR, G, F, S, I, T) \
    add_chain_unit_vjp((P), (PR), (G), (P).F, (P).S, (I), (T))
#define RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(P, PR, G, F, I, T) \
    add_chain_unit_vjp((P), (PR), (G), (P).F, 1, (I), (T))

#include <rayd/shared/multipath/diffraction_accumulation_ad_vjp_device.cuh>

__global__ void dfr_direct_accum_jvp_kernel(DfrDirectAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }
    const float dot_contribution = direct_jvp(params, p);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kDfrEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_direct_accum_vjp_kernel(DfrDirectAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }

    float grad_contribution =
        read_grid_or_zero(params.grad_out_power,
                          params.grad_out_power_rank,
                          params.grad_out_power_stride0,
                          params.grad_out_power_stride1,
                          params.grid_resolution0,
                          p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kDfrEps) {
        grad_contribution +=
            read_grid_or_zero(params.grad_out_field_x_re,
                              params.grad_out_field_x_re_rank,
                              params.grad_out_field_x_re_stride0,
                              params.grad_out_field_x_re_stride1,
                              params.grid_resolution0,
                              p.cell) *
            0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }

    if (p.is_keller || p.is_suffix) {
        vjp_by_unit_jvps(params, p, grad_contribution);
        return;
    }

    const float grad_src_power = grad_contribution * p.common_no_src;
    atomic_add_strided(params.grad_state_src_power,
                       params.grad_state_src_power_stride,
                       p.state_idx,
                       grad_src_power);

    if (p.material_active && p.material_idx >= 0) {
        const float grad_gain =
            grad_contribution * p.contribution / fmaxf(p.material_gain, kDfrEps);
        atomic_add_strided(params.grad_material_gain,
                           params.grad_material_gain_stride,
                           p.material_idx,
                           grad_gain);
    }

    float grad_edge_length = 0.f;
    if (p.edge_length_active && p.edge_length > kDfrEps) {
        grad_edge_length = grad_contribution * p.contribution / p.edge_length;
    }
    if (p.wedge_active) {
        const float grad_wedge =
            grad_contribution * p.contribution / fmaxf(p.wedge_scale, kDfrEps);
        atomic_add_strided(params.grad_state_exterior_angle,
                           params.grad_state_exterior_angle_stride,
                           p.state_idx,
                           grad_wedge / (2.f * kPi));
    }

    const float3 source_delta = p.edge_point - p.source;
    const float3 target_delta = p.target - p.edge_point;
    const float3 d_contribution_d_edge =
        p.contribution *
        ((-2.f / p.source_dist2) * source_delta +
         (2.f / p.target_dist2) * target_delta);
    const float3 d_contribution_d_source =
        p.contribution * ((2.f / p.source_dist2) * source_delta);

    const float3 grad_edge_point = grad_contribution * d_contribution_d_edge;
    const float3 grad_source = grad_contribution * d_contribution_d_source;
    atomic_add_vec_strided(params.grad_state_src_x,
                           params.grad_state_src_y,
                           params.grad_state_src_z,
                           params.grad_state_src_stride,
                           p.state_idx,
                           grad_source);
    atomic_add_vec_strided(params.grad_state_edge_pos_x,
                           params.grad_state_edge_pos_y,
                           params.grad_state_edge_pos_z,
                           params.grad_state_edge_pos_stride,
                           p.state_idx,
                           grad_edge_point);

    const float grad_edge_t = dot3(grad_edge_point, p.edge_dir);
    atomic_add_strided(params.grad_state_edge_t_min,
                       params.grad_state_edge_t_min_stride,
                       p.state_idx,
                       (1.f - p.edge_u) * grad_edge_t - grad_edge_length);
    atomic_add_strided(params.grad_state_edge_t_max,
                       params.grad_state_edge_t_max_stride,
                       p.state_idx,
                       p.edge_u * grad_edge_t + grad_edge_length);

    const float3 grad_edge_dir = p.edge_t * grad_edge_point;
    const float3 grad_edge_dir_raw =
        (1.f / p.edge_dir_norm) *
        (grad_edge_dir - dot3(p.edge_dir, grad_edge_dir) * p.edge_dir);
    atomic_add_vec_strided(params.grad_state_edge_dir_x,
                           params.grad_state_edge_dir_y,
                           params.grad_state_edge_dir_z,
                           params.grad_state_edge_dir_stride,
                           p.state_idx,
                           grad_edge_dir_raw);
}

__global__ void dfr_chain_accum_jvp_kernel(DfrChainAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    ChainPrimal p;
    if (!load_chain_primal(params, lane, p)) {
        return;
    }
    const ChainTangent tangent = chain_read_tangent(params, p);
    const float dot_contribution = chain_contribution_jvp(params, p, tangent);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kDfrEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_chain_accum_vjp_kernel(DfrChainAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    ChainPrimal p;
    if (!load_chain_primal(params, lane, p)) {
        return;
    }
    float grad_contribution =
        read_grid_or_zero(params.grad_out_power,
                          params.grad_out_power_rank,
                          params.grad_out_power_stride0,
                          params.grad_out_power_stride1,
                          params.grid_resolution0,
                          p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kDfrEps) {
        grad_contribution +=
            read_grid_or_zero(params.grad_out_field_x_re,
                              params.grad_out_field_x_re_rank,
                              params.grad_out_field_x_re_stride0,
                              params.grad_out_field_x_re_stride1,
                              params.grid_resolution0,
                              p.cell) *
            0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }
    chain_vjp_by_unit_jvps(params, p, grad_contribution);
}

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

template <typename Params, typename Kernel>
void launch_ad_kernel(const char *name,
                      Kernel kernel,
                      const Params &params) {
    if (params.n_rays <= 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch(name,
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(params.n_rays));
    kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error("dfr_direct_accum_ad_gpu(): failed to launch kernel");
}

} // namespace

void dfr_direct_accum_jvp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_jvp_kernel",
                     dfr_direct_accum_jvp_kernel,
                     params);
}

void dfr_direct_accum_vjp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_vjp_kernel",
                     dfr_direct_accum_vjp_kernel,
                     params);
}

void dfr_chain_accum_jvp_gpu(const DfrChainAccumADParams &params) {
    launch_ad_kernel("dfr_chain_accum_jvp_kernel",
                     dfr_chain_accum_jvp_kernel,
                     params);
}

void dfr_chain_accum_vjp_gpu(const DfrChainAccumADParams &params) {
    launch_ad_kernel("dfr_chain_accum_vjp_kernel",
                     dfr_chain_accum_vjp_kernel,
                     params);
}

} // namespace rayd::torch_backend
