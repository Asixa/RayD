#pragma once

#include <ATen/ATen.h>

#include <array>
#include <optional>

namespace rayd::torch {

struct LayerStackRequest {
    at::Tensor cos_theta;
    at::Tensor material_id;
    at::Tensor layer_offset;
    at::Tensor layer_count;
    at::Tensor layer_thickness_m;
    at::Tensor layer_eps_r;
    at::Tensor layer_sigma_e;
    at::Tensor layer_mu_r;
    double frequency_hz = 0.0;
};

struct LayerStackResult {
    at::Tensor r_te_real;
    at::Tensor r_te_imag;
    at::Tensor r_tm_real;
    at::Tensor r_tm_imag;
    at::Tensor t_te_real;
    at::Tensor t_te_imag;
    at::Tensor t_tm_real;
    at::Tensor t_tm_imag;
    at::Tensor cap_r_te;
    at::Tensor cap_r_tm;
    at::Tensor cap_t_te;
    at::Tensor cap_t_tm;
};

LayerStackResult em_layer_stack_eval(const LayerStackRequest &request);

struct LayerStackBackwardRequest {
    LayerStackRequest primal;
    std::array<std::optional<at::Tensor>, 12> grad_outputs;
    bool need_cos_theta = false;
    bool need_layers = false;
    bool need_frequency = false;
};

struct LayerStackBackwardResult {
    at::Tensor grad_cos_theta;
    at::Tensor grad_layer_thickness_m;
    at::Tensor grad_layer_eps_r;
    at::Tensor grad_layer_sigma_e;
    at::Tensor grad_frequency;
};

LayerStackBackwardResult em_layer_stack_backward(
    const LayerStackBackwardRequest &request);

struct LayerStackJvpRequest {
    LayerStackRequest primal;
    std::optional<at::Tensor> tangent_cos_theta;
    std::optional<at::Tensor> tangent_layer_thickness_m;
    std::optional<at::Tensor> tangent_layer_eps_r;
    std::optional<at::Tensor> tangent_layer_sigma_e;
    double tangent_frequency = 0.0;
};

LayerStackResult em_layer_stack_jvp(const LayerStackJvpRequest &request);

struct TransmissionSequenceRequest {
    at::Tensor path_valid;
    at::Tensor source;
    at::Tensor target;
    at::Tensor interaction_positions;
    at::Tensor interaction_normals;
    at::Tensor interaction_material_id;
    at::Tensor interaction_valid;
    at::Tensor tx_power;
    at::Tensor tx_polarization;
    at::Tensor rx_polarization;
    at::Tensor layer_offset;
    at::Tensor layer_count;
    at::Tensor layer_thickness_m;
    at::Tensor layer_eps_r;
    at::Tensor layer_sigma_e;
    at::Tensor layer_mu_r;
    double frequency_hz = 0.0;
};

struct TransmissionSequenceResult {
    at::Tensor field_vector;
    at::Tensor coefficient;
    at::Tensor path_field;
    at::Tensor path_gain;
    at::Tensor path_length_m;
    at::Tensor delay_s;
    at::Tensor direction;
};

TransmissionSequenceResult field_transmission_sequence(
    const TransmissionSequenceRequest &request);

struct TransmissionSequenceBackwardRequest {
    TransmissionSequenceRequest primal;
    std::optional<at::Tensor> grad_field_vector;
    std::optional<at::Tensor> grad_coefficient;
    std::optional<at::Tensor> grad_path_field;
    std::optional<at::Tensor> grad_path_gain;
    std::optional<at::Tensor> grad_path_length_m;
    std::optional<at::Tensor> grad_delay_s;
    bool need_grad_layer_thickness_m = false;
    bool need_grad_layer_eps_r = false;
    bool need_grad_layer_sigma_e = false;
    bool need_grad_frequency = false;
    bool need_grad_geometry = false;
};

struct TransmissionSequenceBackwardResult {
    std::optional<at::Tensor> grad_layer_thickness_m;
    std::optional<at::Tensor> grad_layer_eps_r;
    std::optional<at::Tensor> grad_layer_sigma_e;
    std::optional<at::Tensor> grad_frequency;
    std::optional<at::Tensor> grad_source;
    std::optional<at::Tensor> grad_target;
    std::optional<at::Tensor> grad_interaction_positions;
    std::optional<at::Tensor> grad_interaction_normals;
};

TransmissionSequenceBackwardResult field_transmission_sequence_backward(
    const TransmissionSequenceBackwardRequest &request);

struct TransmissionSequenceJvpRequest {
    TransmissionSequenceRequest primal;
    std::optional<at::Tensor> tangent_layer_thickness_m;
    std::optional<at::Tensor> tangent_layer_eps_r;
    std::optional<at::Tensor> tangent_layer_sigma_e;
    double tangent_frequency = 0.0;
    std::optional<at::Tensor> tangent_source;
    std::optional<at::Tensor> tangent_target;
    std::optional<at::Tensor> tangent_interaction_positions;
    std::optional<at::Tensor> tangent_interaction_normals;
};

struct TransmissionSequenceJvpResult {
    at::Tensor field_vector;
    at::Tensor coefficient;
    at::Tensor path_field;
    at::Tensor path_gain;
    at::Tensor path_length_m;
    at::Tensor delay_s;
};

TransmissionSequenceJvpResult field_transmission_sequence_jvp(
    const TransmissionSequenceJvpRequest &request);

} // namespace rayd::torch
