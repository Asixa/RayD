#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/shared/contracts.h>
#include <rayd/shared/field_math.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include <rayd/rayd.h>

#include <rayd/native_launch_audit.h>

namespace rayd {

namespace {

constexpr float kSmallEps = shared::SmallEpsilon;
constexpr float kPi = 3.14159265358979323846f;

using namespace shared::field;

static __forceinline__ __device__ float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float s, float3 v) {
    return make_vec3(s * v.x, s * v.y, s * v.z);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 cross(float3 a, float3 b) {
    return make_vec3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float norm3(float3 v) {
    return sqrtf(fmaxf(dot3(v, v), 0.0f));
}

static __forceinline__ __device__ float3 normalize3(float3 v) {
    const float len2 = dot3(v, v);
    if (!(len2 > 1e-12f) || !isfinite(len2)) {
        return make_vec3(0.f, 0.f, 0.f);
    }
    return rsqrtf(len2) * v;
}

static __forceinline__ __device__ bool slot_reflection_coefficients(
    const ReflEpcFieldParams params,
    int slot,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    return fresnel_reflection_coefficients(
        params.slot_eta_r[slot],
        params.slot_sigma[slot],
        params.slot_mu_r[slot],
        params.slot_gain[slot],
        params.omega,
        cos_theta,
        r_te,
        r_tm,
        kSmallEps);
}

static __forceinline__ __device__ void store_zero_field(
    const ReflEpcFieldParams params,
    int ray_index) {
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
#define RAYD_REFL_EPC_LOAD_TX_POLARIZATION(P, RAY)          \
    const int tx_pol_index =                                \
        (P).tx_pol_count == 1 ? 0 : (RAY);                  \
    const float3 tx_polarization =                          \
        make_vec3((P).tx_pol_x[tx_pol_index],               \
                  (P).tx_pol_y[tx_pol_index],               \
                  (P).tx_pol_z[tx_pol_index]);
#define RAYD_REFL_EPC_STORE_FIELD(P, RAY, FIELD)            \
    (P).out_valid[(RAY)] = 1u;                              \
    (P).out_field_x_re[(RAY)] = (FIELD).x.r;                \
    (P).out_field_x_im[(RAY)] = (FIELD).x.i;                \
    (P).out_field_y_re[(RAY)] = (FIELD).y.r;                \
    (P).out_field_y_im[(RAY)] = (FIELD).y.i;                \
    (P).out_field_z_re[(RAY)] = (FIELD).z.r;                \
    (P).out_field_z_im[(RAY)] = (FIELD).z.i;

#include <rayd/shared/multipath/reflection_epc_field_device.cuh>

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

} // namespace

void reflection_epc_field_gpu(const ReflEpcFieldParams &params) {
    require(params.n_rays >= 0,
            "reflection_epc_field_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_field_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    require(stream != nullptr,
            "reflection_epc_field_gpu(): CUDA stream is unavailable.");

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

} // namespace rayd
