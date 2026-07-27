#include <rayd/torch/reflection/epc_field.h>
#include <rayd/shared/contracts.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include <rayd/torch/common/complex.cuh>
#include <rayd/torch/common/math.cuh>
#include <rayd/torch/common/native_compat.h>



namespace rayd::torch_backend {

namespace {

constexpr float kReflEps = shared::SmallEpsilon;

static __forceinline__ __device__ bool slot_reflection_coefficients(
    const ReflEpcFieldParams params,
    int slot,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    const float eta_r_value = params.slot_eta_r != nullptr ? params.slot_eta_r[slot] : 1.f;
    const float sigma_value = params.slot_sigma != nullptr ? params.slot_sigma[slot] : 0.f;
    const float gain = params.slot_gain != nullptr ? params.slot_gain[slot] : 1.f;
    const float mu_r_value = params.slot_mu_r != nullptr ? params.slot_mu_r[slot] : 1.f;
    return shared::field::fresnel_reflection_coefficients(
        eta_r_value,
        sigma_value,
        mu_r_value,
        gain,
        params.omega,
        cos_theta,
        r_te,
        r_tm,
        kReflEps);
}

static __forceinline__ __device__ void store_zero_field(
    const ReflEpcFieldParams params,
    int ray_index) {
    if (params.out_valid != nullptr) {
        params.out_valid[ray_index] = 0u;
    }
    if (params.out_field_x_re != nullptr) {
        params.out_field_x_re[ray_index] = 0.f;
        params.out_field_x_im[ray_index] = 0.f;
    }
    if (params.out_field_y_re != nullptr) {
        params.out_field_y_re[ray_index] = 0.f;
        params.out_field_y_im[ray_index] = 0.f;
    }
    if (params.out_field_z_re != nullptr) {
        params.out_field_z_re[ray_index] = 0.f;
        params.out_field_z_im[ray_index] = 0.f;
    }
}

__global__ void reflection_epc_forward_setup_kernel(ReflEpcForwardSetupParams params) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int slot_count = params.n_rays * params.max_bounces;
    const int total = params.n_rays > slot_count ? params.n_rays : slot_count;
    if (idx >= total) {
        return;
    }

    if (idx < params.n_rays) {
        const int base3 = idx * 3;
        const float sx = params.source_aos[base3 + 0];
        const float sy = params.source_aos[base3 + 1];
        const float sz = params.source_aos[base3 + 2];
        const float rx = params.receiver_aos[base3 + 0];
        const float ry = params.receiver_aos[base3 + 1];
        const float rz = params.receiver_aos[base3 + 2];
        const float dx = rx - sx;
        const float dy = ry - sy;
        const float dz = rz - sz;

        params.source_x[idx] = sx;
        params.source_y[idx] = sy;
        params.source_z[idx] = sz;
        params.receiver_x[idx] = rx;
        params.receiver_y[idx] = ry;
        params.receiver_z[idx] = rz;
        params.ray_dx[idx] = dx;
        params.ray_dy[idx] = dy;
        params.ray_dz[idx] = dz;
        params.ray_tmax[idx] = sqrtf(dx * dx + dy * dy + dz * dz);

        params.epc_valid[idx] = 0u;
        params.epc_bounce_count[idx] = 0;
        params.epc_path_length[idx] = __uint_as_float(0x7f800000u);
        params.first_blocked_segment[idx] = -1;
        params.first_blocked_prim[idx] = -1;
        params.first_blocked_group[idx] = -1;

        const int bary = idx * 3;
        params.tape_barycentric[bary + 0] = 0.f;
        params.tape_barycentric[bary + 1] = 0.f;
        params.tape_barycentric[bary + 2] = 0.f;
    }

    if (idx < slot_count) {
        params.point_x[idx] = 0.f;
        params.point_y[idx] = 0.f;
        params.point_z[idx] = 0.f;
        params.trace_prim_ids[idx] = -1;
        params.resolved_prim_ids[idx] = -1;
        params.surface_group_ids[idx] = -1;
        params.plane_normal_x[idx] = 0.f;
        params.plane_normal_y[idx] = 0.f;
        params.plane_normal_z[idx] = 0.f;
    }
}

// Identifier/storage layer for the shared EPC field device body. Every macro
// expands to the exact pre-dedup expression of this backend: nullable reads
// with defaults, first-prim-id prologue exports, and null-guarded output
// writes.
#define RAYD_REFL_EPC_MAKE3(x, y, z) make_f3(x, y, z)
#define RAYD_REFL_EPC_EPS kReflEps
#define RAYD_REFL_EPC_FIELD_PROLOGUE(P, RAY, BASE)                                 \
    if ((P).out_first_resolved_prim_id != nullptr) {                               \
        (P).out_first_resolved_prim_id[(RAY)] =                                    \
            (P).resolved_prim_ids != nullptr ? (P).resolved_prim_ids[(BASE)] : -1; \
    }                                                                              \
    if ((P).out_first_trace_prim_id != nullptr) {                                  \
        (P).out_first_trace_prim_id[(RAY)] =                                       \
            (P).trace_prim_ids != nullptr ? (P).trace_prim_ids[(BASE)] : -1;       \
    }
#define RAYD_REFL_EPC_LOAD_TX_POLARIZATION(P, RAY)                                 \
    float3 tx_polarization = make_f3(1.f, 0.f, 0.f);                               \
    if ((P).tx_pol_x != nullptr) {                                                 \
        const int tx_pol_index = (P).tx_pol_count == 1 ? 0 : (RAY);                \
        tx_polarization = make_f3((P).tx_pol_x[tx_pol_index],                      \
                                  (P).tx_pol_y[tx_pol_index],                      \
                                  (P).tx_pol_z[tx_pol_index]);                     \
    }
#define RAYD_REFL_EPC_STORE_FIELD(P, RAY, FIELD)                                   \
    if ((P).out_valid != nullptr) {                                                \
        (P).out_valid[(RAY)] = 1u;                                                 \
    }                                                                              \
    if ((P).out_field_x_re != nullptr) {                                           \
        (P).out_field_x_re[(RAY)] = (FIELD).x.r;                                   \
        (P).out_field_x_im[(RAY)] = (FIELD).x.i;                                   \
    }                                                                              \
    if ((P).out_field_y_re != nullptr) {                                           \
        (P).out_field_y_re[(RAY)] = (FIELD).y.r;                                   \
        (P).out_field_y_im[(RAY)] = (FIELD).y.i;                                   \
    }                                                                              \
    if ((P).out_field_z_re != nullptr) {                                           \
        (P).out_field_z_re[(RAY)] = (FIELD).z.r;                                   \
        (P).out_field_z_im[(RAY)] = (FIELD).z.i;                                   \
    }

#include <rayd/shared/multipath/reflection_epc_field_device.cuh>

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

} // namespace

void reflection_epc_forward_setup_gpu(const ReflEpcForwardSetupParams &params) {
    require(params.n_rays >= 0,
            "reflection_epc_forward_setup_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_forward_setup_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    const int slot_count = params.n_rays * params.max_bounces;
    const int total = std::max(params.n_rays, slot_count);
    const int block_size = 128;
    const int block_count = (total + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_forward_setup_kernel",
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(total));
    reflection_epc_forward_setup_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error(
        "reflection_epc_forward_setup_gpu(): failed to launch setup kernel");
}

void reflection_epc_field_gpu(const ReflEpcFieldParams &params) {
    require(params.n_rays >= 0,
            "reflection_epc_field_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_field_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());

    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_field_kernel",
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(params.n_rays));
    reflection_epc_field_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error(
        "reflection_epc_field_gpu(): failed to launch field kernel");
}

} // namespace rayd::torch_backend
