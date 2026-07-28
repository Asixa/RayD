// Shared device body for the reflection EPC field kernel
// (reflection_epc_field_kernel in both backend translation units).
//
// This is a TEXTUAL DEVICE-BODY FRAGMENT, not a standalone header. Each backend
// includes it INSIDE its own anonymous namespace, after defining its float3
// primitive layer, its divergent helpers, and the macro table below. Every
// unqualified name in this fragment (normalize3, dot3, cross, norm3, kPi,
// slot_reflection_coefficients, store_zero_field, the Complex/Complex3 helper
// set, ReflEpcFieldParams) resolves in the includer's scope, so each backend
// keeps its own historical semantics: degenerate-input normalize3 behavior,
// nullable-vs-dense material reads inside slot_reflection_coefficients, and
// guarded-vs-unconditional output writes. Both consuming translation units
// compile under the nvcc_default numeric profile (see
// contracts/compile_policy.json).
//
// The includer MUST define, before including this fragment:
//
//   types and functions
//     ReflEpcFieldParams                  backend Params struct (identical
//                                         field names on both sides)
//     normalize3, dot3, cross, norm3      float3 primitives
//     kPi                                 float pi constant
//     slot_reflection_coefficients(params, slot, cos_theta, r_te, r_tm)
//     store_zero_field(params, ray_index)
//     c3_zero, c3_add, c3_dot_real, c3_scale_complex, c_mul, c3_from_real,
//     c3_mul_complex, c_scale, c3_power, finite_complex3, Complex, Complex3
//
//   macros (P = params, RAY = ray index, BASE = ray * max_bounces)
//     RAYD_REFL_EPC_MAKE3(x, y, z)        float3 constructor spelling
//     RAYD_REFL_EPC_EPS                   small-epsilon constant name
//     RAYD_REFL_EPC_FIELD_PROLOGUE(P, RAY, BASE)
//                                         extra per-ray exports at kernel top
//     RAYD_REFL_EPC_LOAD_TX_POLARIZATION(P, RAY)
//                                         declares float3 tx_polarization
//     RAYD_REFL_EPC_STORE_FIELD(P, RAY, FIELD)
//                                         success-path output writes

#pragma once

static __forceinline__ __device__ float3 fallback_axis(float3 direction) {
    return fabsf(direction.z) < 0.9f
               ? RAYD_REFL_EPC_MAKE3(0.f, 0.f, 1.f)
               : RAYD_REFL_EPC_MAKE3(0.f, 1.f, 0.f);
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

static __forceinline__ __device__ Complex3 reflect_field_vector(
    const ReflEpcFieldParams params,
    int slot,
    Complex3 field,
    float3 incident_dir) {
    const float3 incident_hat = normalize3(incident_dir);
    float3 normal_hat =
        normalize3(RAYD_REFL_EPC_MAKE3(params.slot_normal_x[slot],
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

/// One ray per thread (blockIdx.x * blockDim.x + threadIdx.x, bounds-checked); evaluates the
/// complex reflected field from the ray's precomputed EPC geometry and writes the per-ray outputs.
__global__ void reflection_epc_field_kernel(ReflEpcFieldParams params) {
    const int ray_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (ray_index >= params.n_rays) {
        return;
    }

    const int base = ray_index * params.max_bounces;
    RAYD_REFL_EPC_FIELD_PROLOGUE(params, ray_index, base)
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

    const float3 tx = RAYD_REFL_EPC_MAKE3(params.ray_ox[ray_index],
                                          params.ray_oy[ray_index],
                                          params.ray_oz[ray_index]);
    if (params.out_tx_x != nullptr) {
        params.out_tx_x[ray_index] = tx.x;
        params.out_tx_y[ray_index] = tx.y;
        params.out_tx_z[ray_index] = tx.z;
        const float3 zero = RAYD_REFL_EPC_MAKE3(0.f, 0.f, 0.f);
        float3 first = zero;
        float3 last = zero;
        if (clamped_bounce_count > 0) {
            first = RAYD_REFL_EPC_MAKE3(params.hit_x[base],
                                        params.hit_y[base],
                                        params.hit_z[base]);
            const int last_slot = base + clamped_bounce_count - 1;
            last = RAYD_REFL_EPC_MAKE3(params.hit_x[last_slot],
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
    const float3 first_hit = RAYD_REFL_EPC_MAKE3(params.hit_x[base],
                                                 params.hit_y[base],
                                                 params.hit_z[base]);
    const float3 first_dir = normalize3(first_hit - previous);
    if (dot3(first_dir, first_dir) <= 0.f) {
        store_zero_field(params, ray_index);
        return;
    }

    RAYD_REFL_EPC_LOAD_TX_POLARIZATION(params, ray_index)
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
        const float3 hit = RAYD_REFL_EPC_MAKE3(params.hit_x[slot],
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
    const float3 rx = RAYD_REFL_EPC_MAKE3(params.rx_x[rx_id],
                                          params.rx_y[rx_id],
                                          params.rx_z[rx_id]);
    const float final_segment_length = norm3(rx - previous);
    const float path_length =
        params.epc_path_length != nullptr ? params.epc_path_length[ray_index]
                                          : final_segment_length;
    if (!(path_length > RAYD_REFL_EPC_EPS) || !isfinite(path_length)) {
        store_zero_field(params, ray_index);
        return;
    }

    const float wave_k = 2.f * kPi / fmaxf(params.wavelength, RAYD_REFL_EPC_EPS);
    const Complex phase = shared::field::propagation_phase(wave_k, path_length);
    const float amplitude = shared::field::free_space_amplitude(
        params.wavelength, path_length, RAYD_REFL_EPC_EPS);
    field = c3_mul_complex(field, c_scale(phase, amplitude));
    const float power = c3_power(field);
    if (!finite_complex3(field) || !isfinite(power)) {
        store_zero_field(params, ray_index);
        return;
    }

    RAYD_REFL_EPC_STORE_FIELD(params, ray_index, field)
}
