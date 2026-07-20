#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace rayd::torch {

struct DiffractionWedgeRequest {
    at::Tensor valid;
    at::Tensor source;
    at::Tensor target;
    at::Tensor edge_position;
    at::Tensor edge_direction;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    at::Tensor edge_n0;
    at::Tensor edge_n1;
    at::Tensor exterior_angle;
    at::Tensor face0_valid;
    at::Tensor face0_eps_r;
    at::Tensor face0_sigma_e;
    at::Tensor face0_mu_r;
    at::Tensor face0_gain;
    at::Tensor face1_valid;
    at::Tensor face1_eps_r;
    at::Tensor face1_sigma_e;
    at::Tensor face1_mu_r;
    at::Tensor face1_gain;
    at::Tensor tx_power;
    double frequency_hz = 0.0;
    std::optional<at::Tensor> vertex_v0;
    std::optional<at::Tensor> vertex_v1;
    std::optional<at::Tensor> vertex_opp0;
    std::optional<at::Tensor> vertex_opp1;
    std::optional<at::Tensor> edge_boundary;
    double isb_boundary_taper_width = 0.0;
};

struct DiffractionWedgeResult {
    at::Tensor field_vector;
    at::Tensor direction;
};

DiffractionWedgeResult field_diffraction_wedge(
    const DiffractionWedgeRequest &request);

struct DiffractionWedgeBackwardRequest {
    DiffractionWedgeRequest primal;
    std::optional<at::Tensor> grad_field_vector;
    std::optional<at::Tensor> grad_direction;
    bool need_grad_material = false;
    bool need_grad_frequency = false;
    bool need_grad_geometry = false;
    bool need_grad_vertices = false;
};

struct DiffractionWedgeBackwardResult {
    std::optional<at::Tensor> grad_source;
    std::optional<at::Tensor> grad_target;
    std::optional<at::Tensor> grad_face0_eps_r;
    std::optional<at::Tensor> grad_face0_sigma_e;
    std::optional<at::Tensor> grad_face0_gain;
    std::optional<at::Tensor> grad_face1_eps_r;
    std::optional<at::Tensor> grad_face1_sigma_e;
    std::optional<at::Tensor> grad_face1_gain;
    std::optional<at::Tensor> grad_frequency;
    std::optional<at::Tensor> grad_vertex_v0;
    std::optional<at::Tensor> grad_vertex_v1;
    std::optional<at::Tensor> grad_vertex_opp0;
    std::optional<at::Tensor> grad_vertex_opp1;
};

DiffractionWedgeBackwardResult field_diffraction_wedge_backward(
    const DiffractionWedgeBackwardRequest &request);

struct DiffractionWedgeJvpRequest {
    DiffractionWedgeRequest primal;
    std::optional<at::Tensor> tangent_source;
    std::optional<at::Tensor> tangent_target;
    std::optional<at::Tensor> tangent_face0_eps_r;
    std::optional<at::Tensor> tangent_face0_sigma_e;
    std::optional<at::Tensor> tangent_face0_gain;
    std::optional<at::Tensor> tangent_face1_eps_r;
    std::optional<at::Tensor> tangent_face1_sigma_e;
    std::optional<at::Tensor> tangent_face1_gain;
    double tangent_frequency = 0.0;
    std::optional<at::Tensor> tangent_vertex_v0;
    std::optional<at::Tensor> tangent_vertex_v1;
    std::optional<at::Tensor> tangent_vertex_opp0;
    std::optional<at::Tensor> tangent_vertex_opp1;
};

struct DiffractionWedgeJvpResult {
    at::Tensor tangent_field_vector;
    at::Tensor tangent_direction;
};

DiffractionWedgeJvpResult field_diffraction_wedge_jvp(
    const DiffractionWedgeJvpRequest &request);

} // namespace rayd::torch
