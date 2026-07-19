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

} // namespace rayd::torch
