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

}  // namespace rayd::torch
