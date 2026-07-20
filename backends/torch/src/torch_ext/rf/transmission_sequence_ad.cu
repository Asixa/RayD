#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <rayd/shared/rf/field_transport.cuh>
#include <rayd/shared/rf/layer_stack.cuh>
#include <rayd/torch/common/tensor_check.h>
#include <rayd/torch/rf/field_transport_ad.cuh>
#include <rayd/torch/rf/transmission.h>

#include <optional>
#include <utility>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxAdDepth = 8;
namespace field = rayd::shared::utd;
namespace em = rayd::shared::rf::em;
namespace transport = rayd::shared::rf::field_transport;
namespace ad = rayd::torch::rf::field_transport_ad;

using ad::DualC;
using ad::adj_dot;
using ad::fold_output_cotangents;
using ad::write_output_tangents;

__device__ __forceinline__ field::float3a load3f(
    const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::float3a load_sequence3f(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ ad::DualF3 load_dual3f(
    const float* values, const float* tangents, int64_t index) {
    return {
        load3f(values, index),
        tangents != nullptr ? load3f(tangents, index) : field::f3_zero()};
}

__device__ __forceinline__ ad::DualF3 load_dual_sequence3f(
    const float* values,
    const float* tangents,
    int64_t index,
    int64_t bounce,
    int64_t depth) {
    return {
        load_sequence3f(values, index, bounce, depth),
        tangents != nullptr
            ? load_sequence3f(tangents, index, bounce, depth)
            : field::f3_zero()};
}

struct TransmissionChain {
    transport::WallFrame frames[kMaxAdDepth];
    field::Complex3 value_in[kMaxAdDepth];  // field entering each wall
    field::Complex e_s[kMaxAdDepth];
    field::Complex e_p[kMaxAdDepth];
    field::Complex t_te[kMaxAdDepth];
    field::Complex t_tm[kMaxAdDepth];
    float wall_thickness[kMaxAdDepth];
    int wall_material[kMaxAdDepth];  // -1 for skipped slots
    field::float3a direction;
    field::float3a rx_axis;
    field::Complex3 value_chain;
    field::Complex propagation;
    field::Complex propagation_dfreq;
    field::Complex propagation_dcarrier;
    field::Complex propagation_dtotal;  // amplitude spread over the raw length
    float total_length;
    float carrier_length;
    float amplitude_scale;
    bool path_valid;
};

__device__ void transmission_chain_eval(
    int64_t index,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_normals,
    const int* interaction_material_id,
    const bool* interaction_valid,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const em::LayerView& layers_base,
    int64_t material_count,
    float frequency_hz,
    TransmissionChain& chain) {
    const field::float3a source_value = load3f(source, index);
    const field::float3a target_value = load3f(target, index);
    const field::float3a offset = field::f3_sub(target_value, source_value);
    chain.total_length = field::safe_length(offset);
    chain.direction = field::safe_normalize(
        offset, field::make_f3(0.0f, 0.0f, 1.0f));
    // F1: unnormalized transverse projection of the transmit polarization.
    const field::float3a tx_axis = field::project_to_wedge_plane(
        load3f(tx_polarization, index), chain.direction);
    field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
    float carrier_length = chain.total_length;
    chain.path_valid = true;
    for (int64_t wall = 0; wall < depth; ++wall) {
        chain.wall_material[wall] = -1;
        const int64_t scalar = index * depth + wall;
        if (!interaction_valid[scalar])
            continue;
        const int material = interaction_material_id[scalar];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            chain.path_valid = false;
            break;
        }
        const transport::WallFrame frame = transport::wall_frame(
            chain.direction,
            load_sequence3f(interaction_normals, index, wall, depth));
        em::LayerView layers = layers_base;
        layers.material = material;
        const em::StackRT te = em::stack_rt(
            frame.cos_theta, layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(
            frame.cos_theta, layers, frequency_hz, em::kPolTM);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_axis);
        chain.frames[wall] = frame;
        chain.value_in[wall] = value;
        chain.e_s[wall] = e_s;
        chain.e_p[wall] = e_p;
        chain.t_te[wall] = te.t;
        chain.t_tm[wall] = tm.t;
        chain.wall_material[wall] = material;
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(te.t, e_s)),
            field::cplx_scale_real(frame.p_axis, field::cplx_mul(tm.t, e_p)));
        float wall_thickness = 0.0f;
        const int first = layers_base.layer_offset[material];
        const int layers_in_wall = layers_base.layer_count[material];
        for (int layer = 0; layer < layers_in_wall; ++layer)
            wall_thickness += fmaxf(
                layers_base.layer_thickness_m[first + layer], 0.0f);
        chain.wall_thickness[wall] = wall_thickness;
        carrier_length -= wall_thickness * frame.cos_theta;
    }
    const float wave_number =
        2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float amplitude = 1.0f /
                            (2.0f * wave_number *
                             fmaxf(chain.total_length, field::UTD_EPS));
    const field::Complex propagation = field::cplx_mul_real(
        field::cplx_exp_phase(
            transport::precise_neg_kd(wave_number, carrier_length)),
        amplitude);
    chain.value_chain = value;
    // F1: receiver scalar = p_rx . E via the unnormalized transverse of p_rx.
    chain.rx_axis = field::project_to_wedge_plane(
        load3f(rx_polarization, index), chain.direction);
    chain.propagation = propagation;
    chain.carrier_length = carrier_length;
    // dP/df = P * (-1/k - j*carrier) * (2*pi/c); the amplitude spreads over
    // the full straight length (geometry, handled via dP/dtotal), the phase
    // runs over the carrier length (thickness and cos_theta dependent,
    // handled via dP/dcarrier = -j*k*P).
    const field::Complex dlog = field::cplx(-1.0f / wave_number, -carrier_length);
    chain.propagation_dfreq = field::cplx_mul_real(
        field::cplx_mul(propagation, dlog),
        2.0f * field::UTD_PI / transport::kSpeedOfLight);
    chain.propagation_dcarrier = field::cplx(
        propagation.im * wave_number, -propagation.re * wave_number);
    const float length_gate =
        chain.total_length >= field::UTD_EPS
            ? 1.0f / fmaxf(chain.total_length, field::UTD_EPS)
            : 0.0f;
    chain.propagation_dtotal = field::cplx_mul_real(propagation, -length_gate);
    chain.amplitude_scale = sqrtf(fmaxf(tx_power[index], 0.0f));
}

struct ZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct BasisSeed {
    int slot;
    int param;  // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0)
                seed.d_thickness = 1.0f;
            else if (param == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
    }
};

struct TangentSeed {
    const float* t_thickness;
    const float* t_eps;
    const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {
            t_thickness != nullptr ? t_thickness[query] : 0.0f,
            t_eps != nullptr ? t_eps[query] : 0.0f,
            t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

__global__ void transmission_sequence_backward_kernel(
    int64_t count,
    int64_t depth,
    const bool* path_valid,
    const float* source,
    const float* target,
    const float* interaction_normals,
    const int* interaction_material_id,
    const bool* interaction_valid,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    const float* grad_path_length,
    const float* grad_delay,
    float* grad_layer_thickness,
    float* grad_layer_eps_r,
    float* grad_layer_sigma_e,
    float* grad_frequency,
    float* grad_source,
    float* grad_target,
    float* grad_normals) {
    const em::LayerView layers_base{
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        0,
    };
    const bool need_geometry = grad_source != nullptr;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (!path_valid[index])
            continue;
        TransmissionChain chain;
        transmission_chain_eval(
            index, depth, source, target, interaction_normals,
            interaction_material_id, interaction_valid, tx_power,
            tx_polarization, rx_polarization, layers_base, material_count,
            frequency_hz, chain);
        if (!chain.path_valid)
            continue;  // forward zeroed the outputs; every gradient is zero

        const field::Complex3 value_final = field::c3_scale(
            chain.value_chain, chain.propagation);
        const field::Complex scalar = transport::complex3_dot_real(
            value_final, chain.rx_axis);
        const field::Complex path_field_value = field::cplx_mul_real(
            scalar, chain.amplitude_scale);
        field::Complex g_scalar = field::cplx_zero();
        field::Complex3 g_value = fold_output_cotangents(
            grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
            index, chain.rx_axis, path_field_value, chain.amplitude_scale,
            g_scalar);

        field::Complex g_propagation = field::cplx_zero();
        field::Complex3 g_chain = field::c3_zero();
        field::adj_cplx_mul(
            chain.value_chain.x, chain.propagation, g_value.x,
            g_chain.x, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.y, chain.propagation, g_value.y,
            g_chain.y, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.z, chain.propagation, g_value.z,
            g_chain.z, g_propagation);
        float g_freq = adj_dot(g_propagation, chain.propagation_dfreq);
        const float g_carrier = adj_dot(
            g_propagation, chain.propagation_dcarrier);

        // Geometry cotangents (plan 07 AD-2). The straight length L feeds the
        // amplitude spread, the carrier start (carrier = L - sum_w d_w*cos_w)
        // and the path_length_m / delay_s outputs; the shared ray direction
        // feeds the tx/rx bases and every wall frame.
        field::float3a g_direction = field::f3_zero();
        float g_total_length = 0.0f;
        if (need_geometry) {
            g_total_length = adj_dot(g_propagation, chain.propagation_dtotal) +
                             g_carrier;
            if (grad_path_length != nullptr)
                g_total_length += grad_path_length[index];
            if (grad_delay != nullptr)
                g_total_length += grad_delay[index] / transport::kSpeedOfLight;
            const field::float3a g_rx_axis = field::make_f3(
                field::cplx_adj_dot(g_scalar, value_final.x),
                field::cplx_adj_dot(g_scalar, value_final.y),
                field::cplx_adj_dot(g_scalar, value_final.z));
            field::float3a g_pol_dump = field::f3_zero();
            ad::adj_transverse_project(
                chain.direction, load3f(rx_polarization, index), g_rx_axis,
                g_direction, g_pol_dump);
        }

        for (int64_t wall = depth - 1; wall >= 0; --wall) {
            const int material = chain.wall_material[wall];
            if (material < 0)
                continue;
            const transport::WallFrame& frame = chain.frames[wall];
            field::float3a g_s_axis = field::f3_zero();
            field::float3a g_p_axis = field::f3_zero();
            field::Complex gs = field::cplx_zero();
            field::Complex gp = field::cplx_zero();
            field::adj_cplx_scale_real(
                frame.s_axis,
                field::cplx_mul(chain.t_te[wall], chain.e_s[wall]),
                g_chain, g_s_axis, gs);
            field::adj_cplx_scale_real(
                frame.p_axis,
                field::cplx_mul(chain.t_tm[wall], chain.e_p[wall]),
                g_chain, g_p_axis, gp);
            field::Complex g_t_te = field::cplx_zero();
            field::Complex g_t_tm = field::cplx_zero();
            field::Complex g_e_s = field::cplx_zero();
            field::Complex g_e_p = field::cplx_zero();
            field::adj_cplx_mul(
                chain.t_te[wall], chain.e_s[wall], gs, g_t_te, g_e_s);
            field::adj_cplx_mul(
                chain.t_tm[wall], chain.e_p[wall], gp, g_t_tm, g_e_p);
            field::Complex3 g_value_in = field::c3_zero();
            field::adj_cplx_dot_real(
                chain.value_in[wall], frame.s_axis, g_e_s, g_value_in, g_s_axis);
            field::adj_cplx_dot_real(
                chain.value_in[wall], frame.p_axis, g_e_p, g_value_in, g_p_axis);
            g_chain = g_value_in;

            em::LayerView layers = layers_base;
            layers.material = material;
            const int first = layer_offset[material];
            const int layers_in_wall = layer_count[material];
            for (int layer = 0; layer < layers_in_wall; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    float* destination = param == 0 ? grad_layer_thickness
                                         : param == 1 ? grad_layer_eps_r
                                                      : grad_layer_sigma_e;
                    if (destination == nullptr)
                        continue;
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT te = ad::stack_rt_dual(
                        frame.cos_theta, layers, frequency_hz, 0.0f, 0.0f,
                        em::kPolTE, seed);
                    const ad::DualStackRT tm = ad::stack_rt_dual(
                        frame.cos_theta, layers, frequency_hz, 0.0f, 0.0f,
                        em::kPolTM, seed);
                    float grad = adj_dot(g_t_te, te.t.d) + adj_dot(g_t_tm, tm.t.d);
                    if (param == 0 && layer_thickness_m[slot] >= 0.0f) {
                        // Carrier phase runs over L - sum_w d_w * cos(theta_w).
                        grad += g_carrier * (-frame.cos_theta);
                    }
                    atomicAdd(destination + slot, grad);
                }
            }
            if (grad_frequency != nullptr) {
                const ZeroSeed zero_seed;
                const ad::DualStackRT te = ad::stack_rt_dual(
                    frame.cos_theta, layers, frequency_hz, 0.0f, 1.0f,
                    em::kPolTE, zero_seed);
                const ad::DualStackRT tm = ad::stack_rt_dual(
                    frame.cos_theta, layers, frequency_hz, 0.0f, 1.0f,
                    em::kPolTM, zero_seed);
                g_freq += adj_dot(g_t_te, te.t.d) + adj_dot(g_t_tm, tm.t.d);
            }
            if (!need_geometry)
                continue;

            // Geometry enters this wall through cos_theta (Fresnel stack and
            // carrier chord) and through the s/p frame.
            const ZeroSeed zero_seed;
            const ad::DualStackRT te_cos = ad::stack_rt_dual(
                frame.cos_theta, layers, frequency_hz, 1.0f, 0.0f,
                em::kPolTE, zero_seed);
            const ad::DualStackRT tm_cos = ad::stack_rt_dual(
                frame.cos_theta, layers, frequency_hz, 1.0f, 0.0f,
                em::kPolTM, zero_seed);
            float g_cos_theta = adj_dot(g_t_te, te_cos.t.d) +
                                adj_dot(g_t_tm, tm_cos.t.d);
            g_cos_theta += g_carrier * (-chain.wall_thickness[wall]);
            field::float3a g_normal_raw = field::f3_zero();
            ad::adj_wall_frame(
                chain.direction,
                load_sequence3f(interaction_normals, index, wall, depth),
                g_s_axis, g_p_axis, g_cos_theta, g_direction, g_normal_raw);
            const int64_t normal_base = (index * depth + wall) * 3;
            grad_normals[normal_base] = g_normal_raw.x;
            grad_normals[normal_base + 1] = g_normal_raw.y;
            grad_normals[normal_base + 2] = g_normal_raw.z;
        }
        if (grad_frequency != nullptr)
            atomicAdd(grad_frequency, g_freq);
        if (!need_geometry)
            continue;

        // tx_axis cotangent (value_0 = tx_axis * (1 + 0j)), then the shared
        // straight offset: direction = safe_normalize(target - source, e_z)
        // and L = safe_length(target - source).
        const field::float3a g_tx_axis = field::make_f3(
            g_chain.x.re, g_chain.y.re, g_chain.z.re);
        field::float3a g_pol_dump = field::f3_zero();
        ad::adj_transverse_project(
            chain.direction, load3f(tx_polarization, index), g_tx_axis,
            g_direction, g_pol_dump);
        const field::float3a offset = field::f3_sub(
            load3f(target, index), load3f(source, index));
        field::float3a g_offset = field::f3_zero();
        field::float3a g_ez_dump = field::f3_zero();
        field::adj_safe_normalize(
            offset, field::make_f3(0.0f, 0.0f, 1.0f), g_direction, g_offset,
            g_ez_dump);
        ad::adj_safe_length(offset, g_total_length, g_offset);
        const int64_t base = index * 3;
        grad_target[base] = g_offset.x;
        grad_target[base + 1] = g_offset.y;
        grad_target[base + 2] = g_offset.z;
        grad_source[base] = -g_offset.x;
        grad_source[base + 1] = -g_offset.y;
        grad_source[base + 2] = -g_offset.z;
    }
}

__global__ void transmission_sequence_jvp_kernel(
    int64_t count,
    int64_t depth,
    const bool* path_valid,
    const float* source,
    const float* target,
    const float* interaction_normals,
    const int* interaction_material_id,
    const bool* interaction_valid,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    const float* tangent_layer_thickness,
    const float* tangent_layer_eps_r,
    const float* tangent_layer_sigma_e,
    float tangent_frequency,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_normals,
    c10::complex<float>* t_field_vector,
    c10::complex<float>* t_coefficient,
    c10::complex<float>* t_path_field,
    float* t_path_gain,
    float* t_path_length,
    float* t_delay) {
    const em::LayerView layers_base{
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        0,
    };
    const TangentSeed tangent_seed{
        tangent_layer_thickness, tangent_layer_eps_r, tangent_layer_sigma_e};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        if (!path_valid[index]) {
            t_field_vector[base] = c10::complex<float>(0.0F, 0.0F);
            t_field_vector[base + 1] = c10::complex<float>(0.0F, 0.0F);
            t_field_vector[base + 2] = c10::complex<float>(0.0F, 0.0F);
            t_coefficient[index] = c10::complex<float>(0.0F, 0.0F);
            t_path_field[index] = c10::complex<float>(0.0F, 0.0F);
            t_path_gain[index] = 0.0F;
            t_path_length[index] = 0.0F;
            t_delay[index] = 0.0F;
            continue;
        }
        TransmissionChain chain;
        transmission_chain_eval(
            index, depth, source, target, interaction_normals,
            interaction_material_id, interaction_valid, tx_power,
            tx_polarization, rx_polarization, layers_base, material_count,
            frequency_hz, chain);
        if (!chain.path_valid) {
            const int64_t base = index * 3;
            const c10::complex<float> zero(0.0f, 0.0f);
            t_field_vector[base] = zero;
            t_field_vector[base + 1] = zero;
            t_field_vector[base + 2] = zero;
            t_coefficient[index] = zero;
            t_path_field[index] = zero;
            t_path_gain[index] = 0.0f;
            t_path_length[index] = 0.0f;
            t_delay[index] = 0.0f;
            continue;
        }

        // Straight-ray geometry duals shared by every wall: the offset feeds
        // the direction (tx/rx bases and wall frames) and the raw length
        // (amplitude spread, carrier start, path_length_m / delay_s).
        const ad::DualF3 e_z = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
        const ad::DualF3 offset = ad::df3_sub(
            load_dual3f(target, tangent_target, index),
            load_dual3f(source, tangent_source, index));
        const ad::DualF total_length = ad::dual_safe_length(offset);
        const ad::DualF3 direction = ad::dual_safe_normalize(offset, e_z);
        const ad::DualF3 tx_axis = ad::dual_transverse_project(
            direction, ad::df3_const(load3f(tx_polarization, index)));
        const ad::DualF3 rx_axis = ad::dual_transverse_project(
            direction, ad::df3_const(load3f(rx_polarization, index)));
        field::Complex3 d_value = field::cplx_scale_real(
            tx_axis.d, field::cplx(1.0f, 0.0f));
        float d_carrier = total_length.d;
        for (int64_t wall = 0; wall < depth; ++wall) {
            const int material = chain.wall_material[wall];
            if (material < 0)
                continue;
            const ad::DualWallFrame frame = ad::dual_wall_frame(
                direction,
                load_dual_sequence3f(
                    interaction_normals, tangent_normals, index, wall, depth));
            em::LayerView layers = layers_base;
            layers.material = material;
            const ad::DualStackRT te = ad::stack_rt_dual(
                frame.cos_theta.v, layers, frequency_hz, frame.cos_theta.d,
                tangent_frequency, em::kPolTE, tangent_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(
                frame.cos_theta.v, layers, frequency_hz, frame.cos_theta.d,
                tangent_frequency, em::kPolTM, tangent_seed);
            const field::Complex e_s = chain.e_s[wall];
            const field::Complex e_p = chain.e_p[wall];
            const field::Complex3 value_in = chain.value_in[wall];
            const field::Complex d_e_s = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.s_axis.v),
                transport::complex3_dot_real(value_in, frame.s_axis.d));
            const field::Complex d_e_p = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.p_axis.v),
                transport::complex3_dot_real(value_in, frame.p_axis.d));
            const field::Complex w_te = field::cplx_mul(te.t.v, e_s);
            const field::Complex w_tm = field::cplx_mul(tm.t.v, e_p);
            const field::Complex d_w_te = field::cplx_add(
                field::cplx_mul(te.t.d, e_s), field::cplx_mul(te.t.v, d_e_s));
            const field::Complex d_w_tm = field::cplx_add(
                field::cplx_mul(tm.t.d, e_p), field::cplx_mul(tm.t.v, d_e_p));
            d_value = field::c3_add(
                field::c3_add(
                    field::cplx_scale_real(frame.s_axis.d, w_te),
                    field::cplx_scale_real(frame.s_axis.v, d_w_te)),
                field::c3_add(
                    field::cplx_scale_real(frame.p_axis.d, w_tm),
                    field::cplx_scale_real(frame.p_axis.v, d_w_tm)));
            // Carrier chord: d(wall_thickness * cos_theta) with the clamped
            // per-layer thickness gates of the primal accumulation.
            float d_wall_thickness = 0.0f;
            if (tangent_layer_thickness != nullptr) {
                const int first = layer_offset[material];
                const int layers_in_wall = layer_count[material];
                for (int layer = 0; layer < layers_in_wall; ++layer) {
                    const int slot = first + layer;
                    if (layer_thickness_m[slot] >= 0.0f)
                        d_wall_thickness += tangent_layer_thickness[slot];
                }
            }
            d_carrier -= d_wall_thickness * frame.cos_theta.v +
                         chain.wall_thickness[wall] * frame.cos_theta.d;
        }
        const field::Complex d_propagation = field::cplx_add(
            field::cplx_add(
                field::cplx_mul_real(chain.propagation_dfreq, tangent_frequency),
                field::cplx_mul_real(chain.propagation_dcarrier, d_carrier)),
            field::cplx_mul_real(chain.propagation_dtotal, total_length.d));
        const field::Complex3 value_final = field::c3_scale(
            chain.value_chain, chain.propagation);
        const field::Complex3 d_final = field::c3_add(
            field::c3_scale(d_value, chain.propagation),
            field::c3_scale(chain.value_chain, d_propagation));
        write_output_tangents(
            index, value_final, d_final, chain.rx_axis, rx_axis.d,
            chain.amplitude_scale, total_length.d,
            t_field_vector, t_coefficient, t_path_field, t_path_gain,
            t_path_length, t_delay);
    }
}

void check_tensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t rank) {
    rayd::torch_backend::require_cuda(tensor, name);
    rayd::torch_backend::require_dtype(tensor, dtype, name);
    rayd::torch_backend::require_rank(tensor, rank, name);
    rayd::torch_backend::require_contiguous(tensor, name);
}

void check_vec3_table(const at::Tensor& tensor, const char* name) {
    check_tensor(tensor, name, at::kFloat, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

void check_flat_tensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType dtype) {
    check_tensor(tensor, name, dtype, 1);
}

std::pair<int64_t, int64_t> check_transmission_primal(
    const rayd::torch::TransmissionSequenceRequest& request) {
    const auto& source = request.source;
    const auto& path_valid = request.path_valid;
    const auto& target = request.target;
    const auto& interaction_positions = request.interaction_positions;
    const auto& interaction_normals = request.interaction_normals;
    const auto& interaction_material_id = request.interaction_material_id;
    const auto& interaction_valid = request.interaction_valid;
    const auto& tx_power = request.tx_power;
    const auto& tx_polarization = request.tx_polarization;
    const auto& rx_polarization = request.rx_polarization;
    const auto& layer_offset = request.layer_offset;
    const auto& layer_count = request.layer_count;
    const auto& layer_thickness_m = request.layer_thickness_m;
    const auto& layer_eps_r = request.layer_eps_r;
    const auto& layer_sigma_e = request.layer_sigma_e;
    const auto& layer_mu_r = request.layer_mu_r;

    check_vec3_table(source, "source");
    check_flat_tensor(path_valid, "path_valid", at::kBool);
    check_vec3_table(target, "target");
    check_tensor(
        interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(
        interaction_normals, "interaction_normals", at::kFloat, 3);
    check_tensor(
        interaction_material_id, "interaction_material_id", at::kInt, 2);
    check_tensor(interaction_valid, "interaction_valid", at::kBool, 2);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);

    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(
        depth > 0 && depth <= kMaxAdDepth &&
            interaction_positions.size(2) == 3,
        "interaction_positions must have shape (N, D, 3) with 0 < D <= ",
        kMaxAdDepth);
    TORCH_CHECK(
        interaction_positions.size(0) == count &&
            interaction_normals.sizes() == interaction_positions.sizes(),
        "interaction tensors must match source rows");
    TORCH_CHECK(
        interaction_material_id.size(0) == count &&
            interaction_material_id.size(1) == depth &&
            interaction_valid.size(0) == count &&
            interaction_valid.size(1) == depth,
        "transmission event tensors must have shape (N, D)");
    TORCH_CHECK(
        path_valid.size(0) == count,
        "path_valid must match source rows");
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count &&
            rx_polarization.size(0) == count,
        "transmission endpoint tensors must match source rows");

    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(
        layer_count.size(0) == material_count,
        "layer_count must match layer_offset rows");
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(
            tensor.size(0) == layer_total,
            "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor : {
             path_valid,
             target,
             interaction_positions,
             interaction_normals,
             interaction_material_id,
             interaction_valid,
             tx_power,
             tx_polarization,
             rx_polarization,
             layer_offset,
             layer_count,
             layer_thickness_m,
             layer_eps_r,
             layer_sigma_e,
             layer_mu_r})
        TORCH_CHECK(
            tensor.get_device() == source.get_device(),
            "transmission tensors must share one CUDA device");
    TORCH_CHECK(
        request.frequency_hz > 0.0,
        "frequency_hz must be positive");
    return {depth, material_count};
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_filled(
    at::IntArrayRef sizes,
    const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

const at::Tensor* optional_tensor(
    const std::optional<at::Tensor>& value,
    at::Tensor& storage,
    const char* name,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (!value.has_value())
        return nullptr;
    storage = value->contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* tensor_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

} // namespace

rayd::torch::TransmissionSequenceBackwardResult

rayd::torch::field_transmission_sequence_backward(
    const rayd::torch::TransmissionSequenceBackwardRequest& request) {
    const auto [depth, material_count] =
        check_transmission_primal(request.primal);
    const auto& primal = request.primal;
    const int64_t count = primal.source.size(0);
    const int64_t layer_total = primal.layer_thickness_m.size(0);

    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    const at::Tensor* gfv = optional_tensor(
        request.grad_field_vector, gfv_storage, "grad_field_vector",
        at::kComplexFloat, {count, 3}, primal.source);
    const at::Tensor* gc = optional_tensor(
        request.grad_coefficient, gc_storage, "grad_coefficient",
        at::kComplexFloat, {count}, primal.source);
    const at::Tensor* gpf = optional_tensor(
        request.grad_path_field, gpf_storage, "grad_path_field",
        at::kComplexFloat, {count}, primal.source);
    const at::Tensor* gpg = optional_tensor(
        request.grad_path_gain, gpg_storage, "grad_path_gain",
        at::kFloat, {count}, primal.source);
    const at::Tensor* gpl = optional_tensor(
        request.grad_path_length_m, gpl_storage, "grad_path_length_m",
        at::kFloat, {count}, primal.source);
    const at::Tensor* gd = optional_tensor(
        request.grad_delay_s, gd_storage, "grad_delay_s",
        at::kFloat, {count}, primal.source);

    auto layer_grad = [&](bool needed) {
        return needed
            ? zero_filled({layer_total}, primal.source.options())
            : at::Tensor();
    };
    at::Tensor grad_thickness =
        layer_grad(request.need_grad_layer_thickness_m);
    at::Tensor grad_eps = layer_grad(request.need_grad_layer_eps_r);
    at::Tensor grad_sigma = layer_grad(request.need_grad_layer_sigma_e);
    at::Tensor grad_frequency = request.need_grad_frequency
        ? zero_filled({1}, primal.source.options())
        : at::Tensor();
    at::Tensor grad_source;
    at::Tensor grad_target;
    at::Tensor grad_normals;
    if (request.need_grad_geometry) {
        grad_source = zero_filled({count, 3}, primal.source.options());
        grad_target = zero_filled({count, 3}, primal.source.options());
        grad_normals =
            zero_filled({count, depth, 3}, primal.source.options());
    }
    const bool any_grad_in =
        gfv != nullptr || gc != nullptr || gpf != nullptr ||
        gpg != nullptr || gpl != nullptr || gd != nullptr;
    const bool any_grad_out =
        request.need_grad_layer_thickness_m ||
        request.need_grad_layer_eps_r ||
        request.need_grad_layer_sigma_e ||
        request.need_grad_frequency ||
        request.need_grad_geometry;
    if (count > 0 && any_grad_in && any_grad_out) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            primal.source.get_device()).stream();
        transmission_sequence_backward_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                primal.path_valid.data_ptr<bool>(),
                primal.source.data_ptr<float>(),
                primal.target.data_ptr<float>(),
                primal.interaction_normals.data_ptr<float>(),
                primal.interaction_material_id.data_ptr<int>(),
                primal.interaction_valid.data_ptr<bool>(),
                primal.tx_power.data_ptr<float>(),
                primal.tx_polarization.data_ptr<float>(),
                primal.rx_polarization.data_ptr<float>(),
                primal.layer_offset.data_ptr<int>(),
                primal.layer_count.data_ptr<int>(),
                primal.layer_thickness_m.data_ptr<float>(),
                primal.layer_eps_r.data_ptr<float>(),
                primal.layer_sigma_e.data_ptr<float>(),
                primal.layer_mu_r.data_ptr<float>(),
                material_count,
                static_cast<float>(primal.frequency_hz),
                gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                tensor_ptr<float>(gpg),
                tensor_ptr<float>(gpl),
                tensor_ptr<float>(gd),
                request.need_grad_layer_thickness_m
                    ? grad_thickness.data_ptr<float>()
                    : nullptr,
                request.need_grad_layer_eps_r
                    ? grad_eps.data_ptr<float>()
                    : nullptr,
                request.need_grad_layer_sigma_e
                    ? grad_sigma.data_ptr<float>()
                    : nullptr,
                request.need_grad_frequency
                    ? grad_frequency.data_ptr<float>()
                    : nullptr,
                request.need_grad_geometry
                    ? grad_source.data_ptr<float>()
                    : nullptr,
                request.need_grad_geometry
                    ? grad_target.data_ptr<float>()
                    : nullptr,
                request.need_grad_geometry
                    ? grad_normals.data_ptr<float>()
                    : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        request.need_grad_layer_thickness_m
            ? std::optional<at::Tensor>(grad_thickness)
            : std::nullopt,
        request.need_grad_layer_eps_r
            ? std::optional<at::Tensor>(grad_eps)
            : std::nullopt,
        request.need_grad_layer_sigma_e
            ? std::optional<at::Tensor>(grad_sigma)
            : std::nullopt,
        request.need_grad_frequency
            ? std::optional<at::Tensor>(grad_frequency)
            : std::nullopt,
        request.need_grad_geometry
            ? std::optional<at::Tensor>(grad_source)
            : std::nullopt,
        request.need_grad_geometry
            ? std::optional<at::Tensor>(grad_target)
            : std::nullopt,
        std::nullopt,
        request.need_grad_geometry
            ? std::optional<at::Tensor>(grad_normals)
            : std::nullopt};
}

rayd::torch::TransmissionSequenceJvpResult
rayd::torch::field_transmission_sequence_jvp(
    const rayd::torch::TransmissionSequenceJvpRequest& request) {
    const auto [depth, material_count] =
        check_transmission_primal(request.primal);
    const auto& primal = request.primal;
    const int64_t count = primal.source.size(0);
    const int64_t layer_total = primal.layer_thickness_m.size(0);
    at::Tensor tt_storage;
    at::Tensor te_storage;
    at::Tensor ts_storage;
    at::Tensor tsrc_storage;
    at::Tensor ttgt_storage;
    at::Tensor tpos_storage;
    at::Tensor tnrm_storage;
    const at::Tensor* t_thickness = optional_tensor(
        request.tangent_layer_thickness_m, tt_storage,
        "tangent_layer_thickness_m", at::kFloat, {layer_total},
        primal.source);
    const at::Tensor* t_eps = optional_tensor(
        request.tangent_layer_eps_r, te_storage,
        "tangent_layer_eps_r", at::kFloat, {layer_total},
        primal.source);
    const at::Tensor* t_sigma = optional_tensor(
        request.tangent_layer_sigma_e, ts_storage,
        "tangent_layer_sigma_e", at::kFloat, {layer_total},
        primal.source);
    const at::Tensor* t_source = optional_tensor(
        request.tangent_source, tsrc_storage, "tangent_source",
        at::kFloat, {count, 3}, primal.source);
    const at::Tensor* t_target = optional_tensor(
        request.tangent_target, ttgt_storage, "tangent_target",
        at::kFloat, {count, 3}, primal.source);
    (void)optional_tensor(
        request.tangent_interaction_positions, tpos_storage,
        "tangent_interaction_positions", at::kFloat,
        {count, depth, 3}, primal.source);
    const at::Tensor* t_normals = optional_tensor(
        request.tangent_interaction_normals, tnrm_storage,
        "tangent_interaction_normals", at::kFloat,
        {count, depth, 3}, primal.source);

    auto complex_options = primal.source.options().dtype(at::kComplexFloat);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, primal.source.options());
    auto t_path_length = at::empty({count}, primal.source.options());
    auto t_delay = at::empty({count}, primal.source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            primal.source.get_device()).stream();
        transmission_sequence_jvp_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                primal.path_valid.data_ptr<bool>(),
                primal.source.data_ptr<float>(),
                primal.target.data_ptr<float>(),
                primal.interaction_normals.data_ptr<float>(),
                primal.interaction_material_id.data_ptr<int>(),
                primal.interaction_valid.data_ptr<bool>(),
                primal.tx_power.data_ptr<float>(),
                primal.tx_polarization.data_ptr<float>(),
                primal.rx_polarization.data_ptr<float>(),
                primal.layer_offset.data_ptr<int>(),
                primal.layer_count.data_ptr<int>(),
                primal.layer_thickness_m.data_ptr<float>(),
                primal.layer_eps_r.data_ptr<float>(),
                primal.layer_sigma_e.data_ptr<float>(),
                primal.layer_mu_r.data_ptr<float>(),
                material_count,
                static_cast<float>(primal.frequency_hz),
                tensor_ptr<float>(t_thickness),
                tensor_ptr<float>(t_eps),
                tensor_ptr<float>(t_sigma),
                static_cast<float>(request.tangent_frequency),
                tensor_ptr<float>(t_source),
                tensor_ptr<float>(t_target),
                tensor_ptr<float>(t_normals),
                t_field_vector.data_ptr<c10::complex<float>>(),
                t_coefficient.data_ptr<c10::complex<float>>(),
                t_path_field.data_ptr<c10::complex<float>>(),
                t_path_gain.data_ptr<float>(),
                t_path_length.data_ptr<float>(),
                t_delay.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        t_field_vector,
        t_coefficient,
        t_path_field,
        t_path_gain,
        t_path_length,
        t_delay};
}
