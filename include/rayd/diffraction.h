#pragma once

#include <rayd/scene.h>

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>

namespace rayd::torch {

struct DiffractionState {
    at::Tensor edge_index;
    at::Tensor edge_pos;
    at::Tensor edge_dir;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    at::Tensor n0;
    at::Tensor n1;
    at::Tensor prim0;
    at::Tensor prim1;
    at::Tensor exterior_angle;
    at::Tensor src;
    at::Tensor src_power;
    std::optional<at::Tensor> wi;
    std::optional<at::Tensor> d0;
};

enum class DiffractionPathLayout : std::uint8_t {
    Compact = 0,
    SourceLane = 1,
};

struct RecursiveDiffractionState {
    std::optional<at::Tensor> active;
    at::Tensor edge_index;
    at::Tensor edge_pos;
    at::Tensor edge_dir;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    at::Tensor n0;
    at::Tensor n1;
    at::Tensor prim0;
    at::Tensor prim1;
    at::Tensor exterior_angle;
    std::int64_t state_limit = 0;
};

struct DiffractionPathConfig {
    at::Tensor tx_pos;
    at::Tensor tx_pol;
    at::Tensor rx_pos;
    at::Tensor active;
    DiffractionState state;
    MaterialPayload material;
    std::int64_t state_limit = 0;
    std::int64_t capacity = 0;
    double wavelength = 0.0;
    double isb_taper_width_scale = 0.0;
    DiffractionPathLayout layout = DiffractionPathLayout::Compact;
};

struct DiffractionPathResult {
    at::Tensor count;
    at::Tensor valid;
    at::Tensor tx_id;
    at::Tensor rx_id;
    at::Tensor order;
    at::Tensor edge0;
    at::Tensor edge1;
    at::Tensor edge2;
    at::Tensor delay;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor p0;
    at::Tensor p1;
    at::Tensor p2;
};

DiffractionPathResult diffraction_paths_order1_forward(
    const SceneResource &scene,
    const DiffractionPathConfig &config);

struct DiffractionAccumulationConfig {
    std::optional<at::Tensor> active;
    DiffractionState state;
    MaterialPayload material;
    std::int64_t state_limit = 0;
    Grid2D grid;
    double wavelength = 0.0;
    std::int64_t direct_samples = 0;
    std::int64_t keller_samples = 0;
    std::int64_t suffix_samples = 0;
    std::int64_t seed = 0;
    std::int64_t max_order = 1;
    std::optional<RecursiveDiffractionState> recursive_state;
    bool export_tape = false;
    std::optional<at::Tensor> sample_state_index;
    std::optional<at::Tensor> sample_edge_weight;
};

struct DiffractionAccumulationResult {
    at::Tensor power;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor direct_count;
    at::Tensor keller_count;
    at::Tensor suffix_count;
    at::Tensor visibility_rejects;
    at::Tensor edge_visibility_rejects;
    at::Tensor utd_rejects;
    at::Tensor edge_uses;
    at::Tensor tape_active;
    at::Tensor tape_state_idx;
    at::Tensor tape_cell;
    at::Tensor tape_material_idx;
    at::Tensor tape_edge_u;
};

DiffractionAccumulationResult diffraction_accumulation_forward(
    const SceneResource &scene,
    const DiffractionAccumulationConfig &config);

struct CoherentDiffractionConfig {
    std::optional<at::Tensor> active;
    DiffractionState state;
    MaterialPayload material;
    std::int64_t state_limit = 0;
    Grid2D grid;
    double wavelength = 0.0;
    bool select_diffraction_point = false;
    bool prefilter_visibility = false;
};

struct CoherentDiffractionResult {
    at::Tensor direct_x_re;
    at::Tensor direct_x_im;
    at::Tensor direct_y_re;
    at::Tensor direct_y_im;
    at::Tensor direct_z_re;
    at::Tensor direct_z_im;
    at::Tensor multi_x_re;
    at::Tensor multi_x_im;
    at::Tensor multi_y_re;
    at::Tensor multi_y_im;
    at::Tensor multi_z_re;
    at::Tensor multi_z_im;
    at::Tensor direct_count;
    at::Tensor multi_count;
    at::Tensor visibility_reject_count;
    at::Tensor utd_reject_count;
};

CoherentDiffractionResult diffraction_coherent_accumulation_forward(
    const SceneResource &scene,
    const CoherentDiffractionConfig &config);

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
