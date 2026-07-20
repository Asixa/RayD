#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace rayd::torch {

struct ScatteringTableEvalRequest {
    at::Tensor wi;
    at::Tensor wo;
    at::Tensor f_te;
    at::Tensor f_tm;
};

struct ScatteringTableEvalResult {
    at::Tensor f_te;
    at::Tensor f_tm;
};

ScatteringTableEvalResult scattering_table_eval(
    const ScatteringTableEvalRequest& request);

struct ScatteringTableEvalBackwardRequest {
    ScatteringTableEvalRequest primal;
    std::optional<at::Tensor> grad_f_te;
    std::optional<at::Tensor> grad_f_tm;
    bool need_grad_directions = false;
    bool need_grad_tables = false;
};

struct ScatteringTableEvalBackwardResult {
    std::optional<at::Tensor> grad_wi;
    std::optional<at::Tensor> grad_wo;
    std::optional<at::Tensor> grad_f_te;
    std::optional<at::Tensor> grad_f_tm;
};

ScatteringTableEvalBackwardResult scattering_table_eval_backward(
    const ScatteringTableEvalBackwardRequest& request);

struct ScatteringTableEvalJvpRequest {
    ScatteringTableEvalRequest primal;
    std::optional<at::Tensor> tangent_wi;
    std::optional<at::Tensor> tangent_wo;
    std::optional<at::Tensor> tangent_f_te;
    std::optional<at::Tensor> tangent_f_tm;
};

struct ScatteringTableEvalJvpResult {
    at::Tensor tangent_f_te;
    at::Tensor tangent_f_tm;
};

ScatteringTableEvalJvpResult scattering_table_eval_jvp(
    const ScatteringTableEvalJvpRequest& request);

struct ScatteringTableSampleRequest {
    at::Tensor wi;
    at::Tensor uniforms;
    at::Tensor marginal_cdf;
    at::Tensor conditional_cdf;
    at::Tensor sample_density;
};

struct ScatteringTableSampleResult {
    at::Tensor wo;
    at::Tensor pdf_forward;
    at::Tensor pdf_reverse;
};

ScatteringTableSampleResult scattering_table_sample(
    const ScatteringTableSampleRequest& request);

struct ScatteringTablePdfRequest {
    at::Tensor wi;
    at::Tensor wo;
    at::Tensor sample_density;
    bool reverse = false;
};

struct ScatteringTablePdfResult {
    at::Tensor pdf;
};

ScatteringTablePdfResult scattering_table_pdf(
    const ScatteringTablePdfRequest& request);

struct ScatteringEnsembleEvalRequest {
    at::Tensor wo_rows;
    at::Tensor r2_rows;
    at::Tensor cos_o_rows;
    at::Tensor n_o;
    at::Tensor t1r;
    at::Tensor t2r;
    at::Tensor wi_local;
    at::Tensor cos_i;
    at::Tensor r1;
    at::Tensor a_te2;
    at::Tensor a_tm2;
    at::Tensor weights;
    at::Tensor material_id;
    at::Tensor backup_axis;
    at::Tensor rx_pol;
    at::Tensor rc_idx;
    at::Tensor sc_idx;
    at::Tensor f_te_flat;
    at::Tensor f_tm_flat;
    at::Tensor table_offset;
    at::Tensor table_dims;
    at::Tensor material_slot;
    double coefficient = 0.0;
    double threshold = 0.0;
};

struct ScatteringEnsembleEvalResult {
    at::Tensor gain;
    at::Tensor amplitude;
    at::Tensor length;
    at::Tensor keep;
};

ScatteringEnsembleEvalResult scattering_ensemble_eval(
    const ScatteringEnsembleEvalRequest& request);

struct ScatteringEnsembleEvalBackwardRequest {
    ScatteringEnsembleEvalRequest primal;
    std::optional<at::Tensor> grad_gain;
    std::optional<at::Tensor> grad_amplitude;
    std::optional<at::Tensor> grad_length;
    bool need_grad_rows = false;
    bool need_grad_samples = false;
    bool need_grad_tables = false;
    bool need_grad_coefficient = false;
};

struct ScatteringEnsembleEvalBackwardResult {
    std::optional<at::Tensor> grad_wo_rows;
    std::optional<at::Tensor> grad_r2_rows;
    std::optional<at::Tensor> grad_cos_o_rows;
    std::optional<at::Tensor> grad_n_o;
    std::optional<at::Tensor> grad_t1r;
    std::optional<at::Tensor> grad_t2r;
    std::optional<at::Tensor> grad_wi_local;
    std::optional<at::Tensor> grad_cos_i;
    std::optional<at::Tensor> grad_r1;
    std::optional<at::Tensor> grad_a_te2;
    std::optional<at::Tensor> grad_a_tm2;
    std::optional<at::Tensor> grad_weights;
    std::optional<at::Tensor> grad_f_te;
    std::optional<at::Tensor> grad_f_tm;
    std::optional<at::Tensor> grad_coefficient;
};

ScatteringEnsembleEvalBackwardResult scattering_ensemble_eval_backward(
    const ScatteringEnsembleEvalBackwardRequest& request);

struct ScatteringEnsembleEvalJvpRequest {
    ScatteringEnsembleEvalRequest primal;
    std::optional<at::Tensor> tangent_wo_rows;
    std::optional<at::Tensor> tangent_r2_rows;
    std::optional<at::Tensor> tangent_cos_o_rows;
    std::optional<at::Tensor> tangent_n_o;
    std::optional<at::Tensor> tangent_t1r;
    std::optional<at::Tensor> tangent_t2r;
    std::optional<at::Tensor> tangent_wi_local;
    std::optional<at::Tensor> tangent_cos_i;
    std::optional<at::Tensor> tangent_r1;
    std::optional<at::Tensor> tangent_a_te2;
    std::optional<at::Tensor> tangent_a_tm2;
    std::optional<at::Tensor> tangent_weights;
    std::optional<at::Tensor> tangent_f_te_flat;
    std::optional<at::Tensor> tangent_f_tm_flat;
    double tangent_coefficient = 0.0;
};

struct ScatteringEnsembleEvalJvpResult {
    at::Tensor tangent_gain;
    at::Tensor tangent_amplitude;
    at::Tensor tangent_length;
};

ScatteringEnsembleEvalJvpResult scattering_ensemble_eval_jvp(
    const ScatteringEnsembleEvalJvpRequest& request);

struct ScatteringPatchIntegralEvalRequest {
    at::Tensor patch_tris;
    at::Tensor patch_uvs;
    at::Tensor rows;
    at::Tensor d_i;
    at::Tensor d_o;
    at::Tensor n_rows;
    at::Tensor r_te;
    at::Tensor r_tm;
    at::Tensor pol_t;
    at::Tensor pol_r;
    at::Tensor r1_rows;
    at::Tensor r2_rows;
    at::Tensor centroids;
    at::Tensor heights;
    at::Tensor quad_a;
    at::Tensor quad_b;
    at::Tensor quad_w;
    double k0 = 0.0;
};

struct ScatteringPatchIntegralEvalResult {
    at::Tensor total;
    at::Tensor integral;
    at::Tensor row_value;
};

ScatteringPatchIntegralEvalResult scattering_patch_integral_eval(
    const ScatteringPatchIntegralEvalRequest& request);

struct ScatteringPatchIntegralEvalBackwardRequest {
    ScatteringPatchIntegralEvalRequest primal;
    at::Tensor grad_total;
    bool need_grad_heights = false;
    bool need_grad_jones = false;
    bool need_grad_geometry = false;
    bool need_grad_k0 = false;
};

struct ScatteringPatchIntegralEvalBackwardResult {
    std::optional<at::Tensor> grad_heights;
    std::optional<at::Tensor> grad_r_te;
    std::optional<at::Tensor> grad_r_tm;
    std::optional<at::Tensor> grad_d_i;
    std::optional<at::Tensor> grad_d_o;
    std::optional<at::Tensor> grad_r1_rows;
    std::optional<at::Tensor> grad_r2_rows;
    std::optional<at::Tensor> grad_centroids;
    std::optional<at::Tensor> grad_k0;
};

ScatteringPatchIntegralEvalBackwardResult scattering_patch_integral_eval_backward(
    const ScatteringPatchIntegralEvalBackwardRequest& request);

struct ScatteringPatchIntegralEvalJvpRequest {
    ScatteringPatchIntegralEvalRequest primal;
    std::optional<at::Tensor> tangent_heights;
    std::optional<at::Tensor> tangent_r_te;
    std::optional<at::Tensor> tangent_r_tm;
    std::optional<at::Tensor> tangent_d_i;
    std::optional<at::Tensor> tangent_d_o;
    std::optional<at::Tensor> tangent_r1_rows;
    std::optional<at::Tensor> tangent_r2_rows;
    std::optional<at::Tensor> tangent_centroids;
    double tangent_k0 = 0.0;
};

struct ScatteringPatchIntegralEvalJvpResult {
    at::Tensor tangent_total;
};

ScatteringPatchIntegralEvalJvpResult scattering_patch_integral_eval_jvp(
    const ScatteringPatchIntegralEvalJvpRequest& request);

struct ScatteringChainEnsembleEvalRequest {
    at::Tensor tx_pol;
    at::Tensor rx_pol;
    at::Tensor source;
    at::Tensor vertex;
    at::Tensor target;
    at::Tensor c1_positions;
    at::Tensor c1_normals;
    at::Tensor c1_eps_r;
    at::Tensor c1_sigma_e;
    at::Tensor c1_mu_r;
    at::Tensor c1_gain;
    at::Tensor c1_thickness;
    at::Tensor c1_depth;
    at::Tensor c2_positions;
    at::Tensor c2_normals;
    at::Tensor c2_eps_r;
    at::Tensor c2_sigma_e;
    at::Tensor c2_mu_r;
    at::Tensor c2_gain;
    at::Tensor c2_thickness;
    at::Tensor c2_depth;
    at::Tensor n_o;
    at::Tensor t1r;
    at::Tensor t2r;
    at::Tensor backup_axis;
    at::Tensor wi_local;
    at::Tensor cos_i;
    at::Tensor cos_o;
    at::Tensor d_i;
    at::Tensor d_o;
    at::Tensor l1;
    at::Tensor l2;
    at::Tensor weights;
    at::Tensor material_id;
    at::Tensor f_te_flat;
    at::Tensor f_tm_flat;
    at::Tensor table_offset;
    at::Tensor table_dims;
    at::Tensor material_slot;
    double coefficient = 0.0;
    double threshold = 0.0;
    double frequency_hz = 0.0;
};

struct ScatteringChainEnsembleEvalResult {
    at::Tensor gain;
    at::Tensor amplitude;
    at::Tensor length;
    at::Tensor keep;
};

ScatteringChainEnsembleEvalResult scattering_chain_ensemble_eval(
    const ScatteringChainEnsembleEvalRequest& request);

struct ScatteringChainEnsembleEvalBackwardRequest {
    ScatteringChainEnsembleEvalRequest primal;
    std::optional<at::Tensor> grad_gain;
    std::optional<at::Tensor> grad_amplitude;
    std::optional<at::Tensor> grad_length;
    bool need_grad_chain1 = false;
    bool need_grad_chain2 = false;
    bool need_grad_tables = false;
    bool need_grad_geometry = false;
    bool need_grad_coefficient = false;
    bool need_grad_frequency = false;
};

struct ScatteringChainEnsembleEvalBackwardResult {
    std::optional<at::Tensor> grad_c1_eps_r;
    std::optional<at::Tensor> grad_c1_sigma_e;
    std::optional<at::Tensor> grad_c1_gain;
    std::optional<at::Tensor> grad_c1_thickness;
    std::optional<at::Tensor> grad_c2_eps_r;
    std::optional<at::Tensor> grad_c2_sigma_e;
    std::optional<at::Tensor> grad_c2_gain;
    std::optional<at::Tensor> grad_c2_thickness;
    std::optional<at::Tensor> grad_f_te;
    std::optional<at::Tensor> grad_f_tm;
    std::optional<at::Tensor> grad_coefficient;
    std::optional<at::Tensor> grad_frequency;
};

ScatteringChainEnsembleEvalBackwardResult scattering_chain_ensemble_eval_backward(
    const ScatteringChainEnsembleEvalBackwardRequest& request);

struct ScatteringChainEnsembleEvalJvpRequest {
    ScatteringChainEnsembleEvalRequest primal;
    std::optional<at::Tensor> tangent_c1_eps_r;
    std::optional<at::Tensor> tangent_c1_sigma_e;
    std::optional<at::Tensor> tangent_c1_gain;
    std::optional<at::Tensor> tangent_c1_thickness;
    std::optional<at::Tensor> tangent_c2_eps_r;
    std::optional<at::Tensor> tangent_c2_sigma_e;
    std::optional<at::Tensor> tangent_c2_gain;
    std::optional<at::Tensor> tangent_c2_thickness;
    std::optional<at::Tensor> tangent_f_te_flat;
    std::optional<at::Tensor> tangent_f_tm_flat;
    std::optional<at::Tensor> tangent_c1_positions;
    std::optional<at::Tensor> tangent_c1_normals;
    std::optional<at::Tensor> tangent_c2_positions;
    std::optional<at::Tensor> tangent_c2_normals;
    std::optional<at::Tensor> tangent_d_i;
    std::optional<at::Tensor> tangent_d_o;
    std::optional<at::Tensor> tangent_vertex_normal;
    std::optional<at::Tensor> tangent_l1;
    std::optional<at::Tensor> tangent_l2;
    std::optional<at::Tensor> tangent_cos_i;
    std::optional<at::Tensor> tangent_cos_o;
    double tangent_coefficient = 0.0;
    double tangent_frequency = 0.0;
};

struct ScatteringChainEnsembleEvalJvpResult {
    at::Tensor tangent_gain;
    at::Tensor tangent_amplitude;
    at::Tensor tangent_length;
};

ScatteringChainEnsembleEvalJvpResult scattering_chain_ensemble_eval_jvp(
    const ScatteringChainEnsembleEvalJvpRequest& request);

struct ScatteringChainRealizationEvalRequest {
    at::Tensor patch_tris;
    at::Tensor patch_uvs;
    at::Tensor rows;
    at::Tensor d_i;
    at::Tensor d_o;
    at::Tensor n_rows;
    at::Tensor source;
    at::Tensor vertex;
    at::Tensor target;
    at::Tensor c1_positions;
    at::Tensor c1_normals;
    at::Tensor c1_eps_r;
    at::Tensor c1_sigma_e;
    at::Tensor c1_mu_r;
    at::Tensor c1_gain;
    at::Tensor c1_thickness;
    at::Tensor c1_depth;
    at::Tensor c2_positions;
    at::Tensor c2_normals;
    at::Tensor c2_eps_r;
    at::Tensor c2_sigma_e;
    at::Tensor c2_mu_r;
    at::Tensor c2_gain;
    at::Tensor c2_thickness;
    at::Tensor c2_depth;
    at::Tensor tx_pol;
    at::Tensor rx_pol;
    at::Tensor l1;
    at::Tensor l2;
    at::Tensor sp1;
    at::Tensor sp2;
    at::Tensor centroids;
    at::Tensor heights;
    at::Tensor cos_spec;
    at::Tensor material_id;
    at::Tensor layer_offset;
    at::Tensor layer_count;
    at::Tensor layer_thickness_m;
    at::Tensor layer_eps_r;
    at::Tensor layer_sigma_e;
    at::Tensor layer_mu_r;
    at::Tensor quad_a;
    at::Tensor quad_b;
    at::Tensor quad_w;
    double k0 = 0.0;
    double frequency_hz = 0.0;
};

struct ScatteringChainRealizationEvalResult {
    at::Tensor total;
    at::Tensor path_field;
    at::Tensor path_gain;
    at::Tensor integral;
    at::Tensor row_value;
};

ScatteringChainRealizationEvalResult scattering_chain_realization_eval(
    const ScatteringChainRealizationEvalRequest& request);

struct ScatteringChainRealizationEvalBackwardRequest {
    ScatteringChainRealizationEvalRequest primal;
    at::Tensor grad_total;
    std::optional<at::Tensor> grad_path_field;
    std::optional<at::Tensor> grad_path_gain;
    bool need_grad_heights = false;
    bool need_grad_layers = false;
    bool need_grad_chain1 = false;
    bool need_grad_chain2 = false;
    bool need_grad_geometry = false;
    bool need_grad_k0 = false;
    bool need_grad_frequency = false;
};

struct ScatteringChainRealizationEvalBackwardResult {
    std::optional<at::Tensor> grad_heights;
    std::optional<at::Tensor> grad_layer_thickness;
    std::optional<at::Tensor> grad_layer_eps_r;
    std::optional<at::Tensor> grad_layer_sigma_e;
    std::optional<at::Tensor> grad_c1_eps_r;
    std::optional<at::Tensor> grad_c1_sigma_e;
    std::optional<at::Tensor> grad_c1_gain;
    std::optional<at::Tensor> grad_c1_thickness;
    std::optional<at::Tensor> grad_c2_eps_r;
    std::optional<at::Tensor> grad_c2_sigma_e;
    std::optional<at::Tensor> grad_c2_gain;
    std::optional<at::Tensor> grad_c2_thickness;
    std::optional<at::Tensor> grad_d_i;
    std::optional<at::Tensor> grad_d_o;
    std::optional<at::Tensor> grad_c1_positions;
    std::optional<at::Tensor> grad_c1_normals;
    std::optional<at::Tensor> grad_c2_positions;
    std::optional<at::Tensor> grad_c2_normals;
    std::optional<at::Tensor> grad_l1;
    std::optional<at::Tensor> grad_l2;
    std::optional<at::Tensor> grad_sp1;
    std::optional<at::Tensor> grad_sp2;
    std::optional<at::Tensor> grad_centroids;
    std::optional<at::Tensor> grad_k0;
    std::optional<at::Tensor> grad_frequency;
};

ScatteringChainRealizationEvalBackwardResult scattering_chain_realization_eval_backward(
    const ScatteringChainRealizationEvalBackwardRequest& request);

struct ScatteringChainRealizationEvalJvpRequest {
    ScatteringChainRealizationEvalRequest primal;
    std::optional<at::Tensor> tangent_heights;
    std::optional<at::Tensor> tangent_layer_thickness;
    std::optional<at::Tensor> tangent_layer_eps_r;
    std::optional<at::Tensor> tangent_layer_sigma_e;
    std::optional<at::Tensor> tangent_c1_eps_r;
    std::optional<at::Tensor> tangent_c1_sigma_e;
    std::optional<at::Tensor> tangent_c1_gain;
    std::optional<at::Tensor> tangent_c1_thickness;
    std::optional<at::Tensor> tangent_c2_eps_r;
    std::optional<at::Tensor> tangent_c2_sigma_e;
    std::optional<at::Tensor> tangent_c2_gain;
    std::optional<at::Tensor> tangent_c2_thickness;
    std::optional<at::Tensor> tangent_d_i;
    std::optional<at::Tensor> tangent_d_o;
    std::optional<at::Tensor> tangent_c1_positions;
    std::optional<at::Tensor> tangent_c1_normals;
    std::optional<at::Tensor> tangent_c2_positions;
    std::optional<at::Tensor> tangent_c2_normals;
    std::optional<at::Tensor> tangent_l1;
    std::optional<at::Tensor> tangent_l2;
    std::optional<at::Tensor> tangent_sp1;
    std::optional<at::Tensor> tangent_sp2;
    std::optional<at::Tensor> tangent_centroids;
    double tangent_k0 = 0.0;
    double tangent_frequency = 0.0;
};

struct ScatteringChainRealizationEvalJvpResult {
    at::Tensor tangent_total;
    at::Tensor tangent_path_field;
    at::Tensor tangent_path_gain;
};

ScatteringChainRealizationEvalJvpResult scattering_chain_realization_eval_jvp(
    const ScatteringChainRealizationEvalJvpRequest& request);

}  // namespace rayd::torch
