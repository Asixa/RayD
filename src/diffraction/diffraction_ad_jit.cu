// Copyright Xingyu Chen.
// Implements diffraction support for diffraction ad Dr.Jit.

#include <src/diffraction/accumulation_ad_jit.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <rayd/jit/native_launch_audit.h>
#include <rayd/jit/core.h>
#include <rayd/math.h>

namespace rayd {

namespace {

constexpr float kSmallEps = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;

using namespace shared::cuda_math;

// Dense storage-access layer for the shared AD device body. Every macro
// expands to the exact pre-dedup Dr.Jit expression: raw dense indexing, no
// stride multiply, and null tests only where the original code had them. The
// stride-field macro arguments exist only for the Torch instantiation and are
// intentionally unused here.
#define RAYD_DFR_AD_READ_F32(P, F, S, I) ((P).F[(I)])
#define RAYD_DFR_AD_READ_I32(P, F, S, I) ((P).F[(I)])
#define RAYD_DFR_AD_READ_I32_OR(P, F, S, I, D) \
    ((P).F != nullptr ? (P).F[(I)] : (D))
#define RAYD_DFR_AD_READ_VEC(P, X, Y, Z, S, I) \
    make_vec3((P).X[(I)], (P).Y[(I)], (P).Z[(I)])
#define RAYD_DFR_AD_READ_VEC_GUARDED(P, X, Y, Z, S, I)      \
    make_vec3((P).X != nullptr ? (P).X[(I)] : 0.f,          \
              (P).Y != nullptr ? (P).Y[(I)] : 0.f,          \
              (P).Z != nullptr ? (P).Z[(I)] : 0.f)
#define RAYD_DFR_AD_READ_DOT_F32(P, F, S, I) read_or_zero((P).F, (I))
#define RAYD_DFR_AD_READ_DOT_VEC(P, X, Y, Z, S, I) \
    read_vec_or_zero((P).X, (P).Y, (P).Z, (I))
#define RAYD_DFR_AD_MATERIAL_VALID_ENTRY(P, I) ((P).material_valid[(I)] != 0u)
#define RAYD_DFR_AD_MATERIAL_GAIN_TAPE(P, I) ((P).material_gain[(I)])
#define RAYD_DFR_AD_MATERIAL_GAIN_EVENT(P, I) ((P).material_gain[(I)])
#define RAYD_DFR_AD_MATERIAL_GAIN_OR_ONE(P, I) \
    ((P).material_gain != nullptr ? (P).material_gain[(I)] : 1.f)
#define RAYD_DFR_AD_SUFFIX_FACE_PRIM(P, F, S, HAS_THIRD, SECOND, THIRD) \
    ((HAS_THIRD) ? (P).F[(THIRD)] : (P).F[(SECOND)])

#include <rayd/diffraction/accumulation_ad_device.cuh>

static __forceinline__ __device__ void add_chain_unit_vjp(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float grad_contribution,
    float *ptr,
    int index,
    const ChainTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = chain_contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index, grad_contribution * partial);
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

// The dense gradient-write layer: every unit-VJP call site expands to the
// pre-dedup Dr.Jit call with the raw gradient pointer; the stride-field macro
// arguments exist only for the Torch instantiation.
#define RAYD_DFR_AD_ADD_UNIT_VJP(P, PR, G, F, S, I, T) \
    add_unit_vjp((P), (PR), (G), (P).F, (I), (T))
#define RAYD_DFR_AD_ADD_UNIT_VJP_DENSE(P, PR, G, F, I, T) \
    add_unit_vjp((P), (PR), (G), (P).F, (I), (T))
#define RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP(P, PR, G, F, S, I, T) \
    add_chain_unit_vjp((P), (PR), (G), (P).F, (I), (T))
#define RAYD_DFR_AD_ADD_CHAIN_UNIT_VJP_DENSE(P, PR, G, F, I, T) \
    add_chain_unit_vjp((P), (PR), (G), (P).F, (I), (T))

#include <rayd/diffraction/accumulation_ad_vjp_device.cuh>

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
        if (amp > kSmallEps) {
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
        read_or_zero(params.grad_out_power, p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kSmallEps) {
        grad_contribution +=
            read_or_zero(params.grad_out_field_x_re, p.cell) * 0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }

    if (p.is_keller || p.is_suffix) {
        vjp_by_unit_jvps(params, p, grad_contribution);
        return;
    }

    const float grad_src_power = grad_contribution * p.common_no_src;
    if (params.grad_state_src_power != nullptr) {
        atomicAdd(params.grad_state_src_power + p.state_idx, grad_src_power);
    }

    if (p.material_active &&
        p.material_idx >= 0 &&
        params.grad_material_gain != nullptr) {
        const float grad_gain =
            grad_contribution * p.contribution / fmaxf(p.material_gain, kSmallEps);
        atomicAdd(params.grad_material_gain + p.material_idx, grad_gain);
    }

    float grad_edge_length = 0.f;
    if (p.edge_length_active && p.edge_length > kSmallEps) {
        grad_edge_length = grad_contribution * p.contribution / p.edge_length;
    }
    if (p.wedge_active && params.grad_state_exterior_angle != nullptr) {
        const float grad_wedge =
            grad_contribution * p.contribution / fmaxf(p.wedge_scale, kSmallEps);
        atomicAdd(params.grad_state_exterior_angle + p.state_idx,
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
    atomic_add_vec(params.grad_state_src_x,
                   params.grad_state_src_y,
                   params.grad_state_src_z,
                   p.state_idx,
                   grad_source);
    atomic_add_vec(params.grad_state_edge_pos_x,
                   params.grad_state_edge_pos_y,
                   params.grad_state_edge_pos_z,
                   p.state_idx,
                   grad_edge_point);

    const float grad_edge_t = dot3(grad_edge_point, p.edge_dir);
    if (params.grad_state_edge_t_min != nullptr) {
        atomicAdd(params.grad_state_edge_t_min + p.state_idx,
                  (1.f - p.edge_u) * grad_edge_t - grad_edge_length);
    }
    if (params.grad_state_edge_t_max != nullptr) {
        atomicAdd(params.grad_state_edge_t_max + p.state_idx,
                  p.edge_u * grad_edge_t + grad_edge_length);
    }

    const float3 grad_edge_dir = p.edge_t * grad_edge_point;
    const float3 grad_edge_dir_raw =
        (1.f / p.edge_dir_norm) *
        (grad_edge_dir - dot3(p.edge_dir, grad_edge_dir) * p.edge_dir);
    atomic_add_vec(params.grad_state_edge_dir_x,
                   params.grad_state_edge_dir_y,
                   params.grad_state_edge_dir_z,
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
        if (amp > kSmallEps) {
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
        read_or_zero(params.grad_out_power, p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kSmallEps) {
        grad_contribution +=
            read_or_zero(params.grad_out_field_x_re, p.cell) * 0.5f / amp;
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
    require(stream != nullptr,
            "dfr_direct_accum_ad_gpu(): CUDA stream is unavailable.");
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

} // namespace rayd
