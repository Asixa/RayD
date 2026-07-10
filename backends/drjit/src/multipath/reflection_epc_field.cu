#include <rayd/multipath/reflection_epc_field.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include <rayd/rayd.h>

#include <rayd/native_launch_audit.h>

namespace rayd {

namespace {

constexpr float kSmallEps = 1e-6f;
constexpr float kEpsilon0 = 8.854187817e-12f;
constexpr float kPi = 3.14159265358979323846f;

struct Complex {
    float r;
    float i;
};

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

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

static __forceinline__ __device__ Complex c_make(float r, float i = 0.f) {
    Complex z;
    z.r = r;
    z.i = i;
    return z;
}

static __forceinline__ __device__ Complex c_add(Complex a, Complex b) {
    return c_make(a.r + b.r, a.i + b.i);
}

static __forceinline__ __device__ Complex c_sub(Complex a, Complex b) {
    return c_make(a.r - b.r, a.i - b.i);
}

static __forceinline__ __device__ Complex c_mul(Complex a, Complex b) {
    return c_make(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

static __forceinline__ __device__ Complex c_scale(Complex a, float s) {
    return c_make(a.r * s, a.i * s);
}

static __forceinline__ __device__ Complex c_mul_real(Complex a, float s) {
    return c_scale(a, s);
}

static __forceinline__ __device__ Complex c_div(Complex a, Complex b) {
    const float denom = fmaxf(b.r * b.r + b.i * b.i, 1e-20f);
    return c_make((a.r * b.r + a.i * b.i) / denom,
                  (a.i * b.r - a.r * b.i) / denom);
}

static __forceinline__ __device__ float c_abs2(Complex z) {
    return z.r * z.r + z.i * z.i;
}

static __forceinline__ __device__ Complex c_sqrt(Complex z) {
    const float r = hypotf(z.r, z.i);
    if (r <= 0.f) {
        return c_make(0.f, 0.f);
    }
    const float real_mag = sqrtf(fmaxf(0.f, 0.5f * (r + z.r)));
    const float imag_mag = sqrtf(fmaxf(0.f, 0.5f * (r - z.r)));
    const float imag = copysignf(imag_mag, z.i);
    return c_make(real_mag, imag);
}

static __forceinline__ __device__ Complex c_exp_neg_i(float phase) {
    float s;
    float c;
    sincosf(phase, &s, &c);
    return c_make(c, -s);
}

static __forceinline__ __device__ Complex3 c3_zero() {
    Complex3 v;
    v.x = c_make(0.f, 0.f);
    v.y = c_make(0.f, 0.f);
    v.z = c_make(0.f, 0.f);
    return v;
}

static __forceinline__ __device__ Complex3 c3_from_real(float3 value) {
    Complex3 v;
    v.x = c_make(value.x, 0.f);
    v.y = c_make(value.y, 0.f);
    v.z = c_make(value.z, 0.f);
    return v;
}

static __forceinline__ __device__ Complex3 c3_add(Complex3 a, Complex3 b) {
    Complex3 v;
    v.x = c_add(a.x, b.x);
    v.y = c_add(a.y, b.y);
    v.z = c_add(a.z, b.z);
    return v;
}

static __forceinline__ __device__ Complex3 c3_scale_complex(float3 basis,
                                                            Complex coeff) {
    Complex3 v;
    v.x = c_mul_real(coeff, basis.x);
    v.y = c_mul_real(coeff, basis.y);
    v.z = c_mul_real(coeff, basis.z);
    return v;
}

static __forceinline__ __device__ Complex3 c3_mul_complex(Complex3 value,
                                                          Complex coeff) {
    Complex3 v;
    v.x = c_mul(value.x, coeff);
    v.y = c_mul(value.y, coeff);
    v.z = c_mul(value.z, coeff);
    return v;
}

static __forceinline__ __device__ Complex c3_dot_real(Complex3 value,
                                                      float3 basis) {
    return c_add(c_add(c_mul_real(value.x, basis.x),
                       c_mul_real(value.y, basis.y)),
                 c_mul_real(value.z, basis.z));
}

static __forceinline__ __device__ float c3_power(Complex3 value) {
    return c_abs2(value.x) + c_abs2(value.y) + c_abs2(value.z);
}

static __forceinline__ __device__ float3 fallback_axis(float3 direction) {
    return fabsf(direction.z) < 0.9f
               ? make_vec3(0.f, 0.f, 1.f)
               : make_vec3(0.f, 1.f, 0.f);
}

static __forceinline__ __device__ float3 stable_perpendicular(float3 direction,
                                                              float3 preferred) {
    const float3 dir = normalize3(direction);
    float3 projected = preferred - dot3(preferred, dir) * dir;
    if (dot3(projected, projected) > 1e-12f) {
        return normalize3(projected);
    }
    const float3 axis = fallback_axis(dir);
    projected = axis - dot3(axis, dir) * dir;
    return normalize3(projected);
}

static __forceinline__ __device__ bool finite_complex3(Complex3 value) {
    return isfinite(value.x.r) && isfinite(value.x.i) &&
           isfinite(value.y.r) && isfinite(value.y.i) &&
           isfinite(value.z.r) && isfinite(value.z.i);
}

static __forceinline__ __device__ bool slot_reflection_coefficients(
    const ReflEpcFieldParams params,
    int slot,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    const float eta_r = fmaxf(params.slot_eta_r[slot], kSmallEps);
    const float sigma = fmaxf(params.slot_sigma[slot], 0.f);
    const float gain = params.slot_gain[slot];
    const float mu_r = fmaxf(params.slot_mu_r[slot], kSmallEps);
    const float omega = fmaxf(params.omega, kSmallEps);
    const Complex eta = c_make(eta_r, -sigma / (omega * kEpsilon0));
    const Complex mu = c_make(mu_r, 0.f);
    const float cos_clamped = fminf(fmaxf(fabsf(cos_theta), kSmallEps), 1.f);
    const float sin2 = fmaxf(0.f, 1.f - cos_clamped * cos_clamped);
    const Complex a = c_sqrt(c_sub(c_mul(mu, eta), c_make(sin2, 0.f)));
    const Complex mu_cos = c_make(mu_r * cos_clamped, 0.f);
    const Complex eta_cos = c_make(eta.r * cos_clamped, eta.i * cos_clamped);
    r_te = c_scale(c_div(c_sub(mu_cos, a), c_add(mu_cos, a)), gain);
    r_tm = c_scale(c_div(c_sub(eta_cos, a), c_add(eta_cos, a)), gain);
    if (!isfinite(r_te.r) || !isfinite(r_te.i)) {
        r_te = c_make(0.f, 0.f);
    }
    if (!isfinite(r_tm.r) || !isfinite(r_tm.i)) {
        r_tm = c_make(0.f, 0.f);
    }
    return c_abs2(r_te) > 0.f || c_abs2(r_tm) > 0.f;
}

static __forceinline__ __device__ Complex3 reflect_field_vector(
    const ReflEpcFieldParams params,
    int slot,
    Complex3 field,
    float3 incident_dir) {
    const float3 incident_hat = normalize3(incident_dir);
    float3 normal_hat =
        normalize3(make_vec3(params.slot_normal_x[slot],
                             params.slot_normal_y[slot],
                             params.slot_normal_z[slot]));
    if (dot3(normal_hat, normal_hat) <= 0.f) {
        return c3_zero();
    }
    if (dot3(incident_hat, normal_hat) > 0.f) {
        normal_hat = -1.f * normal_hat;
    }

    const float dot_dn = dot3(incident_hat, normal_hat);
    const float3 reflected_dir =
        normalize3(incident_hat - 2.f * dot_dn * normal_hat);

    float3 s_hat = cross(normal_hat, incident_hat);
    if (dot3(s_hat, s_hat) <= 1e-12f) {
        s_hat = stable_perpendicular(incident_hat, normal_hat);
    } else {
        s_hat = normalize3(s_hat);
    }
    float3 p_in_hat = cross(s_hat, incident_hat);
    if (dot3(p_in_hat, p_in_hat) <= 1e-12f) {
        p_in_hat = stable_perpendicular(incident_hat, normal_hat);
    } else {
        p_in_hat = normalize3(p_in_hat);
    }
    float3 p_out_hat = cross(s_hat, reflected_dir);
    if (dot3(p_out_hat, p_out_hat) <= 1e-12f) {
        p_out_hat = stable_perpendicular(reflected_dir, normal_hat);
    } else {
        p_out_hat = normalize3(p_out_hat);
    }

    Complex r_te;
    Complex r_tm;
    const float cos_theta = fabsf(dot3(incident_hat, normal_hat));
    if (!slot_reflection_coefficients(params, slot, cos_theta, r_te, r_tm)) {
        return c3_zero();
    }

    const Complex e_s = c3_dot_real(field, s_hat);
    const Complex e_p = c3_dot_real(field, p_in_hat);
    return c3_add(c3_scale_complex(s_hat, c_mul(r_te, e_s)),
                  c3_scale_complex(p_out_hat, c_mul(r_tm, e_p)));
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

/// One ray per thread (blockIdx.x * blockDim.x + threadIdx.x, bounds-checked); evaluates the
/// complex reflected field from the ray's precomputed EPC geometry and writes the per-ray outputs.
__global__ void reflection_epc_field_kernel(ReflEpcFieldParams params) {
    const int ray_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (ray_index >= params.n_rays) {
        return;
    }

    const int base = ray_index * params.max_bounces;
    const bool epc_valid =
        params.epc_valid != nullptr && params.epc_valid[ray_index] != 0u;
    const int bounce_count =
        params.epc_bounce_count != nullptr ? params.epc_bounce_count[ray_index] : 0;
    const int clamped_bounce_count =
        min(max(bounce_count, 0), params.max_bounces);

    if (params.out_bounce_count != nullptr) {
        params.out_bounce_count[ray_index] = bounce_count;
    }
    if (params.out_path_length != nullptr) {
        params.out_path_length[ray_index] =
            params.epc_path_length != nullptr ? params.epc_path_length[ray_index]
                                              : __uint_as_float(0x7f800000u);
    }

    if (params.out_hit_x != nullptr || params.out_normal_x != nullptr ||
        params.out_resolved_prim_ids != nullptr ||
        params.out_surface_group_ids != nullptr) {
        for (int b = 0; b < params.max_bounces; ++b) {
            const int slot = base + b;
            if (params.out_hit_x != nullptr) {
                params.out_hit_x[slot] = params.hit_x[slot];
                params.out_hit_y[slot] = params.hit_y[slot];
                params.out_hit_z[slot] = params.hit_z[slot];
            }
            if (params.out_normal_x != nullptr) {
                params.out_normal_x[slot] = params.epc_normal_x[slot];
                params.out_normal_y[slot] = params.epc_normal_y[slot];
                params.out_normal_z[slot] = params.epc_normal_z[slot];
            }
            if (params.out_resolved_prim_ids != nullptr &&
                params.resolved_prim_ids != nullptr) {
                params.out_resolved_prim_ids[slot] =
                    params.resolved_prim_ids[slot];
            }
            if (params.out_surface_group_ids != nullptr &&
                params.surface_group_ids != nullptr) {
                params.out_surface_group_ids[slot] =
                    params.surface_group_ids[slot];
            }
        }
    }

    const float3 tx = make_vec3(params.ray_ox[ray_index],
                                params.ray_oy[ray_index],
                                params.ray_oz[ray_index]);
    if (params.out_tx_x != nullptr) {
        params.out_tx_x[ray_index] = tx.x;
        params.out_tx_y[ray_index] = tx.y;
        params.out_tx_z[ray_index] = tx.z;
        const float3 zero = make_vec3(0.f, 0.f, 0.f);
        float3 first = zero;
        float3 last = zero;
        if (clamped_bounce_count > 0) {
            first = make_vec3(params.hit_x[base],
                              params.hit_y[base],
                              params.hit_z[base]);
            const int last_slot = base + clamped_bounce_count - 1;
            last = make_vec3(params.hit_x[last_slot],
                             params.hit_y[last_slot],
                             params.hit_z[last_slot]);
        }
        params.out_first_hit_x[ray_index] = first.x;
        params.out_first_hit_y[ray_index] = first.y;
        params.out_first_hit_z[ray_index] = first.z;
        params.out_last_hit_x[ray_index] = last.x;
        params.out_last_hit_y[ray_index] = last.y;
        params.out_last_hit_z[ray_index] = last.z;
    }

    if (!epc_valid || params.max_bounces <= 0) {
        store_zero_field(params, ray_index);
        return;
    }

    float3 previous = tx;
    const float3 first_hit = make_vec3(params.hit_x[base],
                                       params.hit_y[base],
                                       params.hit_z[base]);
    const float3 first_dir = normalize3(first_hit - previous);
    if (dot3(first_dir, first_dir) <= 0.f) {
        store_zero_field(params, ray_index);
        return;
    }

    const int tx_pol_index =
        params.tx_pol_count == 1 ? 0 : ray_index;
    const float3 tx_polarization =
        make_vec3(params.tx_pol_x[tx_pol_index],
                  params.tx_pol_y[tx_pol_index],
                  params.tx_pol_z[tx_pol_index]);
    float3 transverse_polarization =
        tx_polarization - dot3(tx_polarization, first_dir) * first_dir;
    if (dot3(transverse_polarization, transverse_polarization) <= 1e-12f) {
        transverse_polarization = stable_perpendicular(first_dir, tx_polarization);
    } else {
        transverse_polarization = normalize3(transverse_polarization);
    }
    Complex3 field = c3_from_real(transverse_polarization);

    for (int b = 0; b < params.max_bounces; ++b) {
        const int slot = base + b;
        const float3 hit = make_vec3(params.hit_x[slot],
                                     params.hit_y[slot],
                                     params.hit_z[slot]);
        const float3 incident_dir = normalize3(hit - previous);
        if (dot3(incident_dir, incident_dir) <= 0.f) {
            store_zero_field(params, ray_index);
            return;
        }
        field = reflect_field_vector(params, slot, field, incident_dir);
        if (!finite_complex3(field)) {
            store_zero_field(params, ray_index);
            return;
        }
        previous = hit;
    }

    const int rx_id = params.rx_count == 1 ? 0 : ray_index;
    const float3 rx = make_vec3(params.rx_x[rx_id],
                                params.rx_y[rx_id],
                                params.rx_z[rx_id]);
    const float final_segment_length = norm3(rx - previous);
    const float path_length =
        params.epc_path_length != nullptr ? params.epc_path_length[ray_index]
                                          : final_segment_length;
    if (!(path_length > kSmallEps) || !isfinite(path_length)) {
        store_zero_field(params, ray_index);
        return;
    }

    const float wave_k = 2.f * kPi / fmaxf(params.wavelength, kSmallEps);
    const Complex phase = c_exp_neg_i(wave_k * path_length);
    const float amplitude =
        params.wavelength / (4.f * kPi * fmaxf(path_length, kSmallEps));
    field = c3_mul_complex(field, c_scale(phase, amplitude));
    const float power = c3_power(field);
    if (!finite_complex3(field) || !isfinite(power)) {
        store_zero_field(params, ray_index);
        return;
    }

    params.out_valid[ray_index] = 1u;
    params.out_field_x_re[ray_index] = field.x.r;
    params.out_field_x_im[ray_index] = field.x.i;
    params.out_field_y_re[ray_index] = field.y.r;
    params.out_field_y_im[ray_index] = field.y.i;
    params.out_field_z_re[ray_index] = field.z.r;
    params.out_field_z_im[ray_index] = field.z.i;
}

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
