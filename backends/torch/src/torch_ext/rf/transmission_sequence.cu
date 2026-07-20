#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <rayd/shared/rf/field_transport.cuh>
#include <rayd/shared/rf/layer_stack.cuh>
#include <rayd/torch/common/tensor_check.h>
#include <rayd/torch/rf/transmission.h>

#include <utility>

namespace {

constexpr int kBlockSize = 256;
namespace field = rayd::shared::utd;
namespace em = rayd::shared::rf::em;
namespace transport = rayd::shared::rf::field_transport;

__device__ __forceinline__ field::float3a load3(
    const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}
__device__ __forceinline__ field::float3a load_sequence3(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ c10::complex<float> to_complex(
    field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

// Endpoint-connection specular transmission (contract section 4).
//
// The field travels along the straight source->target ray. Each wall applies
// the layer-stack Jones operator diag(t_TE, t_TM) in the wall's s/p basis;
// t_stack is interface-to-interface, so it already carries the interior
// k_z*d phase and absorption. The exterior free-space carrier phase runs
// over (L - sum_w d_w/cos(theta_w)) and each wall adds the lateral chord
// phase exp(-j*k_par*d_w*tan(theta_w)) with k_par = k0*sin(theta_w). Both are
// pure k0 phases, so they collapse to one carrier over the effective length
//   L_eff = L - sum_w d_w/cos(theta_w) + sum_w d_w*sin^2(theta_w)/cos(theta_w)
//         = L - sum_w d_w*cos(theta_w),
// which is what this kernel accumulates. Amplitude spreading uses the FULL
// straight length (1/(2*k*L), matching free_space_complex3), and
// path_length_m/delay_s report the full straight length.
//
// Vacuum-wall identity: for a single vacuum layer t = exp(-j*k0*cos(theta)*d)
// and L_eff = L - d*cos(theta), so t * exp(-j*k0*L_eff) = exp(-j*k0*L) and
// the output equals the free-space field with no wall (unit tested).
__global__ void transmission_sequence_kernel(
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
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* path_length,
    float* delay,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        if (!path_valid[index]) {
            field_vector[base] = c10::complex<float>(0.0F, 0.0F);
            field_vector[base + 1] = c10::complex<float>(0.0F, 0.0F);
            field_vector[base + 2] = c10::complex<float>(0.0F, 0.0F);
            coefficient[index] = c10::complex<float>(0.0F, 0.0F);
            path_field[index] = c10::complex<float>(0.0F, 0.0F);
            path_gain[index] = 0.0F;
            path_length[index] = 0.0F;
            delay[index] = 0.0F;
            direction_out[base] = 0.0F;
            direction_out[base + 1] = 0.0F;
            direction_out[base + 2] = 0.0F;
            continue;
        }
        const field::float3a source_value = load3(source, index);
        const field::float3a target_value = load3(target, index);
        const field::float3a offset = field::f3_sub(target_value, source_value);
        const float total_length = field::safe_length(offset);
        const field::float3a direction = field::safe_normalize(
            offset, field::make_f3(0.0f, 0.0f, 1.0f));
        // F1: unnormalized transverse projection of the transmit polarization.
        const field::float3a tx_axis = field::project_to_wedge_plane(
            load3(tx_polarization, index), direction);
        field::Complex3 value = field::cplx_scale_real(
            tx_axis, field::cplx(1.0f, 0.0f));
        float carrier_length = total_length;
        bool chain_valid = true;
        for (int64_t wall = 0; wall < depth; ++wall) {
            const int64_t scalar = index * depth + wall;
            if (!interaction_valid[scalar])
                continue;
            const int material = interaction_material_id[scalar];
            if (material < 0 || static_cast<int64_t>(material) >= material_count) {
                chain_valid = false;
                break;
            }
            // s/p basis of the wall; outgoing direction equals incident
            // direction, so the incident basis is also the exit basis. The
            // alternate axis keeps the basis deterministic at normal
            // incidence (same construction as reflect_complex3).
            const transport::WallFrame frame = transport::wall_frame(
                direction,
                load_sequence3(interaction_normals, index, wall, depth));
            em::LayerView layers{
                layer_offset,
                layer_count,
                layer_thickness_m,
                layer_eps_r,
                layer_sigma_e,
                layer_mu_r,
                material,
            };
            const em::StackRT te = em::stack_rt(
                frame.cos_theta, layers, frequency_hz, em::kPolTE);
            const em::StackRT tm = em::stack_rt(
                frame.cos_theta, layers, frequency_hz, em::kPolTM);
            const field::Complex e_s = transport::complex3_dot_real(
                value, frame.s_axis);
            const field::Complex e_p = transport::complex3_dot_real(
                value, frame.p_axis);
            value = field::c3_add(
                field::cplx_scale_real(frame.s_axis, field::cplx_mul(te.t, e_s)),
                field::cplx_scale_real(frame.p_axis, field::cplx_mul(tm.t, e_p)));
            float wall_thickness = 0.0f;
            const int first = layer_offset[material];
            const int layers_in_wall = layer_count[material];
            for (int layer = 0; layer < layers_in_wall; ++layer)
                wall_thickness += fmaxf(layer_thickness_m[first + layer], 0.0f);
            carrier_length -= wall_thickness * frame.cos_theta;
        }
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float amplitude = 1.0f /
                                (2.0f * wave_number *
                                 fmaxf(total_length, field::UTD_EPS));
        const field::Complex propagation = field::cplx_mul_real(
            field::cplx_exp_phase(
                transport::precise_neg_kd(wave_number, carrier_length)),
            amplitude);
        value = field::c3_scale(value, propagation);
        if (!chain_valid)
            value = field::c3_zero();
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar_field = transport::project_receiver(
            value, direction, load3(rx_polarization, index));
        coefficient[index] = to_complex(scalar_field);
        const field::Complex received = field::cplx_mul_real(
            scalar_field, sqrtf(fmaxf(tx_power[index], 0.0f)));
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        path_length[index] = total_length;
        delay[index] = total_length / transport::kSpeedOfLight;
        direction_out[base] = direction.x;
        direction_out[base + 1] = direction.y;
        direction_out[base + 2] = direction.z;
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
    check_vec3_table(request.source, "source");
    check_flat_tensor(request.path_valid, "path_valid", at::kBool);
    check_vec3_table(request.target, "target");
    check_tensor(
        request.interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(
        request.interaction_normals, "interaction_normals", at::kFloat, 3);
    check_tensor(
        request.interaction_material_id,
        "interaction_material_id",
        at::kInt,
        2);
    check_tensor(
        request.interaction_valid, "interaction_valid", at::kBool, 2);
    check_flat_tensor(request.tx_power, "tx_power", at::kFloat);
    check_vec3_table(request.tx_polarization, "tx_polarization");
    check_vec3_table(request.rx_polarization, "rx_polarization");
    check_flat_tensor(request.layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(request.layer_count, "layer_count", at::kInt);
    check_flat_tensor(
        request.layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(request.layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(request.layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(request.layer_mu_r, "layer_mu_r", at::kFloat);

    const int64_t count = request.source.size(0);
    const int64_t depth = request.interaction_positions.size(1);
    TORCH_CHECK(
        depth > 0 && request.interaction_positions.size(2) == 3,
        "interaction_positions must have shape (N, D, 3) with D > 0");
    TORCH_CHECK(
        request.interaction_positions.size(0) == count,
        "interaction_positions must match source rows");
    TORCH_CHECK(
        request.interaction_normals.sizes() ==
            request.interaction_positions.sizes(),
        "interaction_normals must match interaction_positions");
    TORCH_CHECK(
        request.interaction_material_id.size(0) == count &&
            request.interaction_material_id.size(1) == depth,
        "interaction_material_id must have shape (N, D)");
    TORCH_CHECK(
        request.interaction_valid.size(0) == count &&
            request.interaction_valid.size(1) == depth,
        "interaction_valid must have shape (N, D)");
    TORCH_CHECK(
        request.path_valid.size(0) == count,
        "path_valid must match source rows");
    TORCH_CHECK(
        request.target.size(0) == count &&
            request.tx_power.size(0) == count &&
            request.tx_polarization.size(0) == count &&
            request.rx_polarization.size(0) == count,
        "transmission endpoint tensors must match source rows");
    const int64_t material_count = request.layer_offset.size(0);
    const int64_t layer_total = request.layer_thickness_m.size(0);
    TORCH_CHECK(
        request.layer_count.size(0) == material_count,
        "layer_count must match layer_offset rows");
    for (const auto& tensor : {
             request.layer_eps_r,
             request.layer_sigma_e,
             request.layer_mu_r})
        TORCH_CHECK(
            tensor.size(0) == layer_total,
            "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor : {
             request.path_valid,
             request.target,
             request.interaction_positions,
             request.interaction_normals,
             request.interaction_material_id,
             request.interaction_valid,
             request.tx_power,
             request.tx_polarization,
             request.rx_polarization,
             request.layer_offset,
             request.layer_count,
             request.layer_thickness_m,
             request.layer_eps_r,
             request.layer_sigma_e,
             request.layer_mu_r})
        TORCH_CHECK(
            tensor.get_device() == request.source.get_device(),
            "transmission tensors must share one CUDA device");
    TORCH_CHECK(request.frequency_hz > 0.0, "frequency_hz must be positive");
    return {depth, material_count};
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

} // namespace

rayd::torch::TransmissionSequenceResult
rayd::torch::field_transmission_sequence(
    const rayd::torch::TransmissionSequenceRequest& request) {
    const auto [depth, material_count] = check_transmission_primal(request);
    const int64_t count = request.source.size(0);
    auto complex_options = request.source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, request.source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(request.source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            request.source.get_device()).stream();
        transmission_sequence_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                request.path_valid.data_ptr<bool>(),
                request.source.data_ptr<float>(),
                request.target.data_ptr<float>(),
                request.interaction_normals.data_ptr<float>(),
                request.interaction_material_id.data_ptr<int>(),
                request.interaction_valid.data_ptr<bool>(),
                request.tx_power.data_ptr<float>(),
                request.tx_polarization.data_ptr<float>(),
                request.rx_polarization.data_ptr<float>(),
                request.layer_offset.data_ptr<int>(),
                request.layer_count.data_ptr<int>(),
                request.layer_thickness_m.data_ptr<float>(),
                request.layer_eps_r.data_ptr<float>(),
                request.layer_sigma_e.data_ptr<float>(),
                request.layer_mu_r.data_ptr<float>(),
                material_count,
                static_cast<float>(request.frequency_hz),
                field_vector.data_ptr<c10::complex<float>>(),
                coefficient.data_ptr<c10::complex<float>>(),
                path_field.data_ptr<c10::complex<float>>(),
                path_gain.data_ptr<float>(),
                path_length.data_ptr<float>(),
                delay.data_ptr<float>(),
                direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        field_vector,
        coefficient,
        path_field,
        path_gain,
        path_length,
        delay,
        direction};
}
