#include <rayd/integration/torch.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using namespace rayd::torch;

static_assert(
    kIntegrationHeaderIdentity ==
    std::string_view(
        "rayd.torch.integration"));

static_assert(std::is_same_v<
              decltype(&scattering_table_eval),
              ScatteringTableEvalResult (*)(const ScatteringTableEvalRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_table_eval_backward),
              ScatteringTableEvalBackwardResult (*)(
                  const ScatteringTableEvalBackwardRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_table_eval_jvp),
              ScatteringTableEvalJvpResult (*)(
                  const ScatteringTableEvalJvpRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_table_sample),
              ScatteringTableSampleResult (*)(
                  const ScatteringTableSampleRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_table_pdf),
              ScatteringTablePdfResult (*)(const ScatteringTablePdfRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_ensemble_eval),
              ScatteringEnsembleEvalResult (*)(
                  const ScatteringEnsembleEvalRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_ensemble_eval_backward),
              ScatteringEnsembleEvalBackwardResult (*)(
                  const ScatteringEnsembleEvalBackwardRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_ensemble_eval_jvp),
              ScatteringEnsembleEvalJvpResult (*)(
                  const ScatteringEnsembleEvalJvpRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_patch_integral_eval),
              ScatteringPatchIntegralEvalResult (*)(
                  const ScatteringPatchIntegralEvalRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_patch_integral_eval_backward),
              ScatteringPatchIntegralEvalBackwardResult (*)(
                  const ScatteringPatchIntegralEvalBackwardRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_patch_integral_eval_jvp),
              ScatteringPatchIntegralEvalJvpResult (*)(
                  const ScatteringPatchIntegralEvalJvpRequest&)>);

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

void require_close(float actual, float expected, const std::string& message) {
    require(
        std::isfinite(actual) && std::fabs(actual - expected) <= 2.0e-5F,
        message);
}

void require_finite(const at::Tensor& tensor, const std::string& message) {
    require(at::isfinite(tensor).all().item<bool>(), message);
}

void require_exact_zero(const at::Tensor& tensor, const std::string& message) {
    require(at::count_nonzero(tensor).item<int64_t>() == 0, message);
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& message) {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::exception&) {
        return;
    }
    fail(message);
}

at::TensorOptions cuda_options(c10::ScalarType dtype) {
    return at::TensorOptions().dtype(dtype).device(at::Device(at::kCUDA, 0));
}

ScatteringTableEvalRequest empty_table_request() {
    const auto options = cuda_options(at::kFloat);
    return {
        at::empty({0}, cuda_options(at::kBool)),
        at::empty({0, 3}, options),
        at::empty({0, 3}, options),
        at::zeros({2, 2, 2, 2}, options),
        at::zeros({2, 2, 2, 2}, options),
    };
}

ScatteringEnsembleEvalRequest empty_ensemble_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto ints = cuda_options(at::kInt);
    const auto longs = cuda_options(at::kLong);
    const auto vec3 = at::empty({0, 3}, floats);
    const auto f32 = at::empty({0}, floats);
    return {
        at::empty({0}, cuda_options(at::kBool)),
        vec3,
        f32,
        f32,
        vec3,
        vec3,
        vec3,
        vec3,
        f32,
        f32,
        f32,
        f32,
        f32,
        at::empty({0}, ints),
        vec3,
        vec3,
        at::empty({0}, longs),
        at::empty({0}, longs),
        f32,
        f32,
        at::empty({0}, longs),
        at::empty({0, 4}, ints),
        at::empty({0}, ints),
        1.0,
        0.0,
    };
}

ScatteringPatchIntegralEvalRequest empty_patch_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto longs = cuda_options(at::kLong);
    const auto complex = cuda_options(at::kComplexFloat);
    return {
        at::empty({0}, cuda_options(at::kBool)),
        at::empty({0, 3, 3}, floats),
        at::empty({0, 3, 2}, floats),
        at::empty({0}, longs),
        at::empty({0, 3}, floats),
        at::empty({0, 3}, floats),
        at::empty({0, 3}, floats),
        at::empty({0}, complex),
        at::empty({0}, complex),
        at::zeros({3}, floats),
        at::zeros({3}, floats),
        at::empty({0}, floats),
        at::empty({0}, floats),
        at::empty({0, 3}, floats),
        at::zeros({2, 2}, floats),
        at::zeros({256}, floats),
        at::zeros({256}, floats),
        at::zeros({256}, floats),
        1.0,
    };
}

ScatteringTableEvalRequest nonempty_table_request() {
    const auto floats = cuda_options(at::kFloat);
    return {
        at::ones({1}, cuda_options(at::kBool)),
        at::tensor({0.8660254F, 0.0F, 0.5F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.8660254F, 0.5F}, floats).reshape({1, 3}),
        at::full({2, 2, 2, 2}, 2.0F, floats),
        at::full({2, 2, 2, 2}, 3.0F, floats),
    };
}

ScatteringEnsembleEvalRequest nonempty_ensemble_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto ints = cuda_options(at::kInt);
    const auto longs = cuda_options(at::kLong);
    return {
        at::ones({1}, cuda_options(at::kBool)),
        at::tensor({1.0F, 0.0F, 0.0F}, floats).reshape({1, 3}),
        at::ones({1}, floats),
        at::full({1}, 0.5F, floats),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3}),
        at::tensor({1.0F, 0.0F, 0.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3}),
        at::ones({1}, floats),
        at::ones({1}, floats),
        at::ones({1}, floats),
        at::zeros({1}, floats),
        at::ones({1}, floats),
        at::zeros({1}, ints),
        at::tensor({1.0F, 0.0F, 0.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        at::zeros({1}, longs),
        at::zeros({1}, longs),
        at::full({16}, 2.0F, floats),
        at::full({16}, 3.0F, floats),
        at::zeros({1}, longs),
        at::tensor({2, 2, 2, 2}, ints).reshape({1, 4}),
        at::zeros({1}, ints),
        1.0,
        0.0,
    };
}

ScatteringPatchIntegralEvalRequest nonempty_patch_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto longs = cuda_options(at::kLong);
    const auto complex = cuda_options(at::kComplexFloat);
    return {
        at::ones({1}, cuda_options(at::kBool)),
        at::tensor(
            {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
            floats).reshape({1, 3, 3}),
        at::tensor({0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F}, floats)
            .reshape({1, 3, 2}),
        at::zeros({1}, longs),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3}),
        at::ones({1}, complex),
        at::ones({1}, complex),
        at::tensor({1.0F, 0.0F, 0.0F}, floats),
        at::tensor({1.0F, 0.0F, 0.0F}, floats),
        at::ones({1}, floats),
        at::ones({1}, floats),
        at::tensor({1.0F / 3.0F, 1.0F / 3.0F, 0.0F}, floats).reshape({1, 3}),
        at::zeros({2, 2}, floats),
        at::zeros({256}, floats),
        at::zeros({256}, floats),
        at::zeros({256}, floats),
        1.0,
    };
}

ScatteringTableSampleRequest nonempty_table_sample_request() {
    const auto primal = nonempty_table_request();
    const auto floats = cuda_options(at::kFloat);
    auto marginal = at::empty({2, 2, 2}, floats);
    marginal.select(2, 0).fill_(0.5F);
    marginal.select(2, 1).fill_(1.0F);
    auto conditional = at::empty({2, 2, 2, 2}, floats);
    conditional.select(3, 0).fill_(0.5F);
    conditional.select(3, 1).fill_(1.0F);
    return {
        primal.valid,
        primal.wi,
        at::tensor({0.25F, 0.25F}, floats).reshape({1, 2}),
        marginal,
        conditional,
        at::full({2, 2, 2, 2}, 0.25F, floats),
    };
}

ScatteringTablePdfRequest nonempty_table_pdf_request() {
    const auto primal = nonempty_table_request();
    return {
        primal.valid,
        primal.wi,
        primal.wo,
        at::full({2, 2, 2, 2}, 0.25F, cuda_options(at::kFloat)),
        false,
    };
}

LayerStackRequest nonempty_layer_stack_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto ints = cuda_options(at::kInt);
    return {
        at::tensor({0.55F}, floats),
        at::tensor({0}, ints),
        at::tensor({0}, ints),
        at::tensor({1}, ints),
        at::tensor({0.12F}, floats),
        at::tensor({4.0F}, floats),
        at::tensor({0.025F}, floats),
        at::tensor({1.0F}, floats),
        3.5e9,
    };
}

TransmissionSequenceRequest nonempty_transmission_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto ints = cuda_options(at::kInt);
    const auto bools = cuda_options(at::kBool);
    return {
        at::ones({1}, bools),
        at::tensor({0.0F, 0.0F, 2.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, -2.0F}, floats).reshape({1, 3}),
        at::zeros({1, 1, 3}, floats),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 1, 3}),
        at::zeros({1, 1}, ints),
        at::ones({1, 1}, bools),
        at::ones({1}, floats),
        at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        at::zeros({1}, ints),
        at::ones({1}, ints),
        at::tensor({0.1F}, floats),
        at::tensor({4.0F}, floats),
        at::tensor({0.05F}, floats),
        at::ones({1}, floats),
        3.5e9,
    };
}

DiffractionWedgeRequest nonempty_wedge_request() {
    const auto floats = cuda_options(at::kFloat);
    const auto masks = cuda_options(at::kBool);
    DiffractionWedgeRequest request;
    request.valid = at::ones({1}, masks);
    request.source = at::tensor({-1.2F, -0.8F, 0.2F}, floats).reshape({1, 3});
    request.target = at::tensor({1.0F, 1.1F, -0.1F}, floats).reshape({1, 3});
    request.edge_position = at::zeros({1, 3}, floats);
    request.edge_direction =
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3});
    request.edge_t_min = at::tensor({-1.0F}, floats);
    request.edge_t_max = at::tensor({1.0F}, floats);
    request.edge_n0 = at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3});
    request.edge_n1 = at::tensor({-1.0F, 0.0F, 0.0F}, floats).reshape({1, 3});
    request.exterior_angle = at::tensor({4.71238898F}, floats);
    request.face0_valid = at::ones({1}, masks);
    request.face0_eps_r = at::tensor({4.0F}, floats);
    request.face0_sigma_e = at::tensor({0.01F}, floats);
    request.face0_mu_r = at::tensor({1.0F}, floats);
    request.face0_gain = at::tensor({1.0F}, floats);
    request.face1_valid = at::ones({1}, masks);
    request.face1_eps_r = at::tensor({3.0F}, floats);
    request.face1_sigma_e = at::tensor({0.02F}, floats);
    request.face1_mu_r = at::tensor({1.0F}, floats);
    request.face1_gain = at::tensor({0.9F}, floats);
    request.tx_power = at::tensor({2.0F}, floats);
    request.frequency_hz = 3.5e9;
    request.vertex_v0 = at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3});
    request.vertex_v1 = at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 3});
    request.vertex_opp0 = at::tensor({1.0F, 0.0F, 0.0F}, floats).reshape({1, 3});
    request.vertex_opp1 = at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3});
    request.edge_boundary = at::zeros({1}, masks);
    return request;
}

void test_table_empty_contracts() {
    const auto primal = empty_table_request();
    const auto result = scattering_table_eval(primal);
    require(result.f_te.sizes() == at::IntArrayRef({0}),
            "empty table TE shape differs");
    require(result.f_tm.sizes() == at::IntArrayRef({0}),
            "empty table TM shape differs");

    ScatteringTableEvalBackwardRequest backward;
    backward.primal = primal;
    backward.need_grad_directions = true;
    backward.need_grad_tables = true;
    const auto gradients = scattering_table_eval_backward(backward);
    require(
        gradients.grad_wi.has_value() &&
            gradients.grad_wi->sizes() == at::IntArrayRef({0, 3}) &&
            gradients.grad_wo.has_value() &&
            gradients.grad_f_te.has_value() &&
            gradients.grad_f_te->sizes() == primal.f_te.sizes() &&
            gradients.grad_f_tm.has_value(),
        "empty table backward schema differs");

    ScatteringTableEvalJvpRequest jvp;
    jvp.primal = primal;
    const auto tangent = scattering_table_eval_jvp(jvp);
    require(tangent.tangent_f_te.sizes() == at::IntArrayRef({0}) &&
                tangent.tangent_f_tm.sizes() == at::IntArrayRef({0}),
            "empty table JVP schema differs");

    const auto floats = cuda_options(at::kFloat);
    ScatteringTableSampleRequest sample{
        primal.valid,
        primal.wi,
        at::empty({0, 2}, floats),
        at::zeros({2, 2, 2}, floats),
        at::zeros({2, 2, 2, 2}, floats),
        at::zeros({2, 2, 2, 2}, floats),
    };
    const auto sampled = scattering_table_sample(sample);
    require(sampled.wo.sizes() == at::IntArrayRef({0, 3}) &&
                sampled.pdf_forward.sizes() == at::IntArrayRef({0}) &&
                sampled.pdf_reverse.sizes() == at::IntArrayRef({0}),
            "empty table sample schema differs");

    const auto pdf = scattering_table_pdf(
        {primal.valid, primal.wi, primal.wo, sample.sample_density, false});
    require(pdf.pdf.sizes() == at::IntArrayRef({0}),
            "empty table PDF schema differs");
}

void test_ensemble_empty_contract() {
    const auto request = empty_ensemble_request();
    const auto result = scattering_ensemble_eval(request);
    require(result.gain.sizes() == at::IntArrayRef({0}) &&
                result.amplitude.sizes() == at::IntArrayRef({0}) &&
                result.length.sizes() == at::IntArrayRef({0}) &&
                result.keep.sizes() == at::IntArrayRef({0}) &&
                result.keep.scalar_type() == at::kBool,
            "empty ensemble schema differs");

    ScatteringEnsembleEvalBackwardRequest backward;
    backward.primal = request;
    backward.need_grad_rows = true;
    backward.need_grad_samples = true;
    backward.need_grad_tables = true;
    backward.need_grad_coefficient = true;
    const auto gradients = scattering_ensemble_eval_backward(backward);
    require(
        gradients.grad_wo_rows.has_value() &&
            gradients.grad_wo_rows->sizes() == at::IntArrayRef({0, 3}) &&
            gradients.grad_n_o.has_value() &&
            gradients.grad_n_o->sizes() == at::IntArrayRef({0, 3}) &&
            gradients.grad_f_te.has_value() &&
            gradients.grad_f_te->sizes() == at::IntArrayRef({0}) &&
            gradients.grad_coefficient.has_value() &&
            gradients.grad_coefficient->sizes() == at::IntArrayRef({1}),
        "empty ensemble backward schema differs");

    ScatteringEnsembleEvalJvpRequest jvp;
    jvp.primal = request;
    const auto tangent = scattering_ensemble_eval_jvp(jvp);
    require(tangent.tangent_gain.sizes() == at::IntArrayRef({0}) &&
                tangent.tangent_amplitude.sizes() == at::IntArrayRef({0}) &&
                tangent.tangent_length.sizes() == at::IntArrayRef({0}),
            "empty ensemble JVP schema differs");
}

void test_patch_empty_contract() {
    const auto request = empty_patch_request();
    const auto result = scattering_patch_integral_eval(request);
    require(result.total.dim() == 0 &&
                result.total.scalar_type() == at::kComplexFloat &&
                result.integral.sizes() == at::IntArrayRef({0}) &&
                result.row_value.sizes() == at::IntArrayRef({0}),
            "empty patch-integral schema differs");
    at::cuda::getCurrentCUDAStream(0).synchronize();
    require(at::count_nonzero(result.total).item<std::int64_t>() == 0,
            "empty patch-integral total must be zero");

    ScatteringPatchIntegralEvalBackwardRequest backward;
    backward.primal = request;
    backward.grad_total = at::zeros({}, cuda_options(at::kComplexFloat));
    backward.need_grad_heights = true;
    backward.need_grad_jones = true;
    backward.need_grad_geometry = true;
    backward.need_grad_k0 = true;
    const auto gradients = scattering_patch_integral_eval_backward(backward);
    require(
        gradients.grad_heights.has_value() &&
            gradients.grad_heights->sizes() == request.heights.sizes() &&
            gradients.grad_r_te.has_value() &&
            gradients.grad_r_te->sizes() == at::IntArrayRef({0}) &&
            gradients.grad_d_i.has_value() &&
            gradients.grad_d_i->sizes() == at::IntArrayRef({0, 3}) &&
            gradients.grad_k0.has_value() &&
            gradients.grad_k0->sizes() == at::IntArrayRef({1}),
        "empty patch-integral backward schema differs");

    ScatteringPatchIntegralEvalJvpRequest jvp;
    jvp.primal = request;
    const auto tangent = scattering_patch_integral_eval_jvp(jvp);
    at::cuda::getCurrentCUDAStream(0).synchronize();
    require(tangent.tangent_total.dim() == 0 &&
                tangent.tangent_total.scalar_type() == at::kComplexFloat &&
                at::count_nonzero(tangent.tangent_total).item<std::int64_t>() == 0,
            "empty patch-integral JVP schema differs");
}

void test_table_nonempty_primal_ad_sample_pdf() {
    const auto primal = nonempty_table_request();
    const auto forward = scattering_table_eval(primal);

    ScatteringTableEvalBackwardRequest backward;
    backward.primal = primal;
    backward.grad_f_te = at::ones({1}, primal.wi.options());
    backward.grad_f_tm = at::zeros({1}, primal.wi.options());
    backward.need_grad_directions = true;
    backward.need_grad_tables = true;
    const auto gradients = scattering_table_eval_backward(backward);

    ScatteringTableEvalJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_f_te = at::ones_like(primal.f_te);
    const auto tangent = scattering_table_eval_jvp(jvp);

    auto marginal = at::empty({2, 2, 2}, primal.wi.options());
    marginal.select(2, 0).fill_(0.5F);
    marginal.select(2, 1).fill_(1.0F);
    auto conditional = at::empty({2, 2, 2, 2}, primal.wi.options());
    conditional.select(3, 0).fill_(0.5F);
    conditional.select(3, 1).fill_(1.0F);
    const auto density = at::full({2, 2, 2, 2}, 0.25F, primal.wi.options());
    const auto uniforms = at::tensor({0.25F, 0.25F}, primal.wi.options())
                              .reshape({1, 2});
    const auto sampled = scattering_table_sample(
        {primal.valid, primal.wi, uniforms, marginal, conditional, density});
    const auto pdf = scattering_table_pdf(
        {primal.valid, primal.wi, primal.wo, density, false});

    c10::cuda::getCurrentCUDAStream(0).synchronize();
    require_close(forward.f_te.item<float>(), 2.0F, "nonempty table TE differs");
    require_close(forward.f_tm.item<float>(), 3.0F, "nonempty table TM differs");
    require(
        gradients.grad_wi.has_value() && gradients.grad_wo.has_value() &&
            gradients.grad_f_te.has_value() && gradients.grad_f_tm.has_value(),
        "nonempty table backward omitted requested fields");
    require_finite(*gradients.grad_wi, "nonempty table grad_wi is not finite");
    require_finite(*gradients.grad_wo, "nonempty table grad_wo is not finite");
    require_finite(*gradients.grad_f_te, "nonempty table grad_f_te is not finite");
    require_finite(*gradients.grad_f_tm, "nonempty table grad_f_tm is not finite");
    require_close(
        gradients.grad_f_te->sum().item<float>(), 1.0F,
        "nonempty table gradient weights do not sum to one");
    require_close(
        tangent.tangent_f_te.item<float>(), 1.0F,
        "nonempty table JVP TE differs");
    require_close(
        tangent.tangent_f_tm.item<float>(), 0.0F,
        "nonempty table JVP TM differs");
    require_finite(sampled.wo, "nonempty table sample direction is not finite");
    require_close(
        sampled.pdf_forward.item<float>(), 0.25F,
        "nonempty table sample forward PDF differs");
    require_close(
        sampled.pdf_reverse.item<float>(), 0.25F,
        "nonempty table sample reverse PDF differs");
    require_close(pdf.pdf.item<float>(), 0.25F, "nonempty table PDF differs");
}

void test_ensemble_nonempty_primal_ad() {
    const auto primal = nonempty_ensemble_request();
    const auto forward = scattering_ensemble_eval(primal);

    ScatteringEnsembleEvalBackwardRequest backward;
    backward.primal = primal;
    backward.grad_gain = at::ones({1}, primal.wo_rows.options());
    backward.need_grad_rows = true;
    backward.need_grad_samples = true;
    backward.need_grad_tables = true;
    backward.need_grad_coefficient = true;
    const auto gradients = scattering_ensemble_eval_backward(backward);

    ScatteringEnsembleEvalJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_coefficient = 1.0;
    const auto tangent = scattering_ensemble_eval_jvp(jvp);

    c10::cuda::getCurrentCUDAStream(0).synchronize();
    require_close(forward.gain.item<float>(), 1.0F, "nonempty ensemble gain differs");
    require_close(
        forward.amplitude.item<float>(), 1.0F,
        "nonempty ensemble amplitude differs");
    require_close(forward.length.item<float>(), 2.0F, "nonempty ensemble length differs");
    require(forward.keep.item<bool>(), "nonempty ensemble keep decision differs");
    require(
        gradients.grad_wo_rows.has_value() && gradients.grad_n_o.has_value() &&
            gradients.grad_f_te.has_value() &&
            gradients.grad_coefficient.has_value(),
        "nonempty ensemble backward omitted requested fields");
    require_finite(*gradients.grad_wo_rows, "nonempty ensemble row gradient is not finite");
    require_finite(*gradients.grad_n_o, "nonempty ensemble sample gradient is not finite");
    require_finite(*gradients.grad_f_te, "nonempty ensemble table gradient is not finite");
    require_close(
        gradients.grad_coefficient->item<float>(), 1.0F,
        "nonempty ensemble coefficient gradient differs");
    require_close(
        tangent.tangent_gain.item<float>(), 1.0F,
        "nonempty ensemble gain JVP differs");
    require_close(
        tangent.tangent_amplitude.item<float>(), 0.5F,
        "nonempty ensemble amplitude JVP differs");
    require_close(
        tangent.tangent_length.item<float>(), 0.0F,
        "nonempty ensemble length JVP differs");
}

void test_patch_nonempty_primal_ad() {
    const auto primal = nonempty_patch_request();
    const auto forward = scattering_patch_integral_eval(primal);

    ScatteringPatchIntegralEvalBackwardRequest backward;
    backward.primal = primal;
    backward.grad_total = at::ones({}, cuda_options(at::kComplexFloat));
    backward.need_grad_heights = true;
    backward.need_grad_jones = true;
    backward.need_grad_geometry = true;
    backward.need_grad_k0 = true;
    const auto gradients = scattering_patch_integral_eval_backward(backward);

    ScatteringPatchIntegralEvalJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_heights = at::ones_like(primal.heights);
    jvp.tangent_k0 = 1.0;
    const auto tangent = scattering_patch_integral_eval_jvp(jvp);

    c10::cuda::getCurrentCUDAStream(0).synchronize();
    require_close(at::abs(forward.total).item<float>(), 0.0F, "nonempty patch total differs");
    require_close(
        at::abs(forward.integral).item<float>(), 0.0F,
        "nonempty patch integral differs");
    require_close(
        at::abs(forward.row_value).item<float>(), 0.0F,
        "nonempty patch row value differs");
    require(
        gradients.grad_heights.has_value() && gradients.grad_r_te.has_value() &&
            gradients.grad_d_i.has_value() && gradients.grad_k0.has_value(),
        "nonempty patch backward omitted requested fields");
    require_finite(*gradients.grad_heights, "nonempty patch height gradient is not finite");
    require_finite(*gradients.grad_r_te, "nonempty patch Jones gradient is not finite");
    require_finite(*gradients.grad_d_i, "nonempty patch geometry gradient is not finite");
    require_finite(*gradients.grad_k0, "nonempty patch k0 gradient is not finite");
    require_close(
        at::abs(tangent.tangent_total).item<float>(), 0.0F,
        "nonempty patch JVP differs");
}

void test_nondefault_stream_dependency() {
    auto request = nonempty_table_request();
    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    ScatteringTableEvalResult result;
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        request.f_te.fill_(4.0F);
        result = scattering_table_eval(request);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
            "scattering table eval changed the caller's non-default stream");
    }
    stream.synchronize();
    require_close(
        result.f_te.item<float>(), 4.0F,
        "scattering table eval ignored a non-default-stream dependency");
}

void test_invalid_rows_short_circuit_poison() {
    auto table = nonempty_table_request();
    table.valid.zero_();
    table.wi.fill_(std::numeric_limits<float>::quiet_NaN());
    table.wo.fill_(std::numeric_limits<float>::quiet_NaN());
    const auto table_out = scattering_table_eval(table);
    require_exact_zero(table_out.f_te, "invalid table TE must be exactly zero");
    require_exact_zero(table_out.f_tm, "invalid table TM must be exactly zero");
    ScatteringTableEvalBackwardRequest table_backward;
    table_backward.primal = table;
    table_backward.grad_f_te = at::ones({1}, table.wi.options());
    table_backward.grad_f_tm = at::ones({1}, table.wi.options());
    table_backward.need_grad_directions = true;
    table_backward.need_grad_tables = true;
    const auto table_grad = scattering_table_eval_backward(table_backward);
    require_exact_zero(*table_grad.grad_wi, "invalid table wi gradient must be zero");
    require_exact_zero(*table_grad.grad_wo, "invalid table wo gradient must be zero");
    require_exact_zero(*table_grad.grad_f_te, "invalid table storage gradient must be zero");
    require_exact_zero(*table_grad.grad_f_tm, "invalid table storage gradient must be zero");
    ScatteringTableEvalJvpRequest table_jvp;
    table_jvp.primal = table;
    table_jvp.tangent_wi = at::full_like(table.wi, 1.0F);
    table_jvp.tangent_wo = at::full_like(table.wo, 1.0F);
    const auto table_tangent = scattering_table_eval_jvp(table_jvp);
    require_exact_zero(table_tangent.tangent_f_te, "invalid table TE JVP must be zero");
    require_exact_zero(table_tangent.tangent_f_tm, "invalid table TM JVP must be zero");

    const auto marginal = at::ones({2, 2, 2}, table.wi.options());
    const auto density = at::ones({2, 2, 2, 2}, table.wi.options());
    const auto uniforms = at::full({1, 2}, std::numeric_limits<float>::quiet_NaN(), table.wi.options());
    const auto sampled = scattering_table_sample(
        {table.valid, table.wi, uniforms, marginal, density, density});
    require_exact_zero(sampled.wo, "invalid table sample direction must be zero");
    require_exact_zero(sampled.pdf_forward, "invalid table sample PDF must be zero");
    require_exact_zero(sampled.pdf_reverse, "invalid reverse table sample PDF must be zero");
    const auto pdf = scattering_table_pdf(
        {table.valid, table.wi, table.wo, density, false});
    require_exact_zero(pdf.pdf, "invalid table PDF must be zero");

    auto ensemble = nonempty_ensemble_request();
    ensemble.valid.zero_();
    ensemble.material_id.fill_(std::numeric_limits<int>::max());
    ensemble.wo_rows.fill_(std::numeric_limits<float>::quiet_NaN());
    const auto ensemble_out = scattering_ensemble_eval(ensemble);
    require_exact_zero(ensemble_out.gain, "invalid ensemble gain must be zero");
    require_exact_zero(ensemble_out.amplitude, "invalid ensemble amplitude must be zero");
    require_exact_zero(ensemble_out.length, "invalid ensemble length must be zero");
    require_exact_zero(ensemble_out.keep, "invalid ensemble keep must be false");
    ScatteringEnsembleEvalBackwardRequest ensemble_backward;
    ensemble_backward.primal = ensemble;
    ensemble_backward.grad_gain = at::ones({1}, ensemble.wo_rows.options());
    ensemble_backward.need_grad_rows = true;
    ensemble_backward.need_grad_samples = true;
    ensemble_backward.need_grad_tables = true;
    ensemble_backward.need_grad_coefficient = true;
    const auto ensemble_grad = scattering_ensemble_eval_backward(ensemble_backward);
    require_exact_zero(*ensemble_grad.grad_wo_rows, "invalid ensemble row gradient must be zero");
    require_exact_zero(*ensemble_grad.grad_n_o, "invalid ensemble sample gradient must be zero");
    require_exact_zero(*ensemble_grad.grad_f_te, "invalid ensemble table gradient must be zero");
    require_exact_zero(*ensemble_grad.grad_coefficient, "invalid ensemble coefficient gradient must be zero");
    ScatteringEnsembleEvalJvpRequest ensemble_jvp;
    ensemble_jvp.primal = ensemble;
    ensemble_jvp.tangent_coefficient = 1.0;
    const auto ensemble_tangent = scattering_ensemble_eval_jvp(ensemble_jvp);
    require_exact_zero(ensemble_tangent.tangent_gain, "invalid ensemble gain JVP must be zero");
    require_exact_zero(ensemble_tangent.tangent_amplitude, "invalid ensemble amplitude JVP must be zero");
    require_exact_zero(ensemble_tangent.tangent_length, "invalid ensemble length JVP must be zero");

    auto patch = nonempty_patch_request();
    patch.valid.zero_();
    patch.rows.fill_(std::numeric_limits<int64_t>::max());
    patch.d_i.fill_(std::numeric_limits<float>::quiet_NaN());
    const auto patch_out = scattering_patch_integral_eval(patch);
    require_exact_zero(patch_out.total, "invalid patch total must be zero");
    require_exact_zero(patch_out.integral, "invalid patch integral must be zero");
    require_exact_zero(patch_out.row_value, "invalid patch row value must be zero");
    ScatteringPatchIntegralEvalBackwardRequest patch_backward;
    patch_backward.primal = patch;
    patch_backward.grad_total = at::ones({}, cuda_options(at::kComplexFloat));
    patch_backward.need_grad_heights = true;
    patch_backward.need_grad_jones = true;
    patch_backward.need_grad_geometry = true;
    patch_backward.need_grad_k0 = true;
    const auto patch_grad = scattering_patch_integral_eval_backward(patch_backward);
    require_exact_zero(*patch_grad.grad_heights, "invalid patch height gradient must be zero");
    require_exact_zero(*patch_grad.grad_r_te, "invalid patch Jones gradient must be zero");
    require_exact_zero(*patch_grad.grad_d_i, "invalid patch geometry gradient must be zero");
    require_exact_zero(*patch_grad.grad_k0, "invalid patch k0 gradient must be zero");
    ScatteringPatchIntegralEvalJvpRequest patch_jvp;
    patch_jvp.primal = patch;
    patch_jvp.tangent_k0 = 1.0;
    const auto patch_tangent = scattering_patch_integral_eval_jvp(patch_jvp);
    require_exact_zero(patch_tangent.tangent_total, "invalid patch JVP must be zero");
}

void test_invalid_contracts_fail_loudly() {
    auto bad_table_valid_dtype = nonempty_table_request();
    bad_table_valid_dtype.valid = at::ones({1}, bad_table_valid_dtype.wi.options());
    require_throws(
        [&] { (void)scattering_table_eval(bad_table_valid_dtype); },
        "wrong table valid dtype must fail loudly");
    auto bad_ensemble_valid_shape = nonempty_ensemble_request();
    bad_ensemble_valid_shape.valid = at::ones(
        {2}, bad_ensemble_valid_shape.valid.options());
    require_throws(
        [&] { (void)scattering_ensemble_eval(bad_ensemble_valid_shape); },
        "wrong ensemble valid shape must fail loudly");
    auto bad_patch_valid_shape = nonempty_patch_request();
    bad_patch_valid_shape.valid = at::ones(
        {2}, bad_patch_valid_shape.valid.options());
    require_throws(
        [&] { (void)scattering_patch_integral_eval(bad_patch_valid_shape); },
        "wrong patch valid shape must fail loudly");

    auto cpu = empty_table_request();
    cpu.wi = cpu.wi.cpu();
    require_throws(
        [&] { (void)scattering_table_eval(cpu); },
        "CPU table input must fail loudly");

    auto dtype = empty_table_request();
    dtype.wi = dtype.wi.to(at::kDouble);
    require_throws(
        [&] { (void)scattering_table_eval(dtype); },
        "wrong table dtype must fail loudly");

    auto shape = empty_table_request();
    shape.wi = at::empty({0, 2}, shape.wi.options());
    require_throws(
        [&] { (void)scattering_table_eval(shape); },
        "wrong table shape must fail loudly");

    ScatteringTableEvalBackwardRequest bad_cotangent;
    bad_cotangent.primal = nonempty_table_request();
    bad_cotangent.grad_f_te = at::ones({2}, bad_cotangent.primal.wi.options());
    bad_cotangent.need_grad_directions = true;
    require_throws(
        [&] { (void)scattering_table_eval_backward(bad_cotangent); },
        "wrong optional table cotangent shape must fail loudly");

    ScatteringTableEvalJvpRequest bad_tangent;
    bad_tangent.primal = nonempty_table_request();
    bad_tangent.tangent_wi = at::ones(
        {1, 3}, bad_tangent.primal.wi.options().dtype(at::kDouble));
    require_throws(
        [&] { (void)scattering_table_eval_jvp(bad_tangent); },
        "wrong optional table tangent dtype must fail loudly");

    auto ensemble = nonempty_ensemble_request();
    auto noncontiguous_rows = at::ones({1, 3, 2}, ensemble.wo_rows.options())
                                  .select(2, 0);
    require(!noncontiguous_rows.is_contiguous(), "ensemble fixture must be non-contiguous");
    ensemble.wo_rows = noncontiguous_rows;
    require_throws(
        [&] { (void)scattering_ensemble_eval(ensemble); },
        "non-contiguous ensemble input must fail loudly");

    ScatteringEnsembleEvalBackwardRequest ensemble_backward;
    ensemble_backward.primal = nonempty_ensemble_request();
    ensemble_backward.grad_gain = at::ones(
        {1}, ensemble_backward.primal.wo_rows.options().dtype(at::kDouble));
    ensemble_backward.need_grad_rows = true;
    require_throws(
        [&] { (void)scattering_ensemble_eval_backward(ensemble_backward); },
        "wrong optional ensemble cotangent dtype must fail loudly");

    ScatteringEnsembleEvalJvpRequest ensemble_jvp;
    ensemble_jvp.primal = nonempty_ensemble_request();
    ensemble_jvp.tangent_r2_rows = at::ones(
        {2}, ensemble_jvp.primal.r2_rows.options());
    require_throws(
        [&] { (void)scattering_ensemble_eval_jvp(ensemble_jvp); },
        "wrong optional ensemble tangent shape must fail loudly");

    auto patch = nonempty_patch_request();
    auto noncontiguous_tris = at::ones({1, 3, 3, 2}, patch.patch_tris.options())
                                  .select(3, 0);
    require(!noncontiguous_tris.is_contiguous(), "patch fixture must be non-contiguous");
    patch.patch_tris = noncontiguous_tris;
    require_throws(
        [&] { (void)scattering_patch_integral_eval(patch); },
        "non-contiguous patch input must fail loudly");

    ScatteringPatchIntegralEvalBackwardRequest patch_backward;
    patch_backward.primal = nonempty_patch_request();
    patch_backward.grad_total = at::ones({}, cuda_options(at::kFloat));
    patch_backward.need_grad_heights = true;
    require_throws(
        [&] { (void)scattering_patch_integral_eval_backward(patch_backward); },
        "wrong patch cotangent dtype must fail loudly");

    ScatteringPatchIntegralEvalJvpRequest patch_jvp;
    patch_jvp.primal = nonempty_patch_request();
    patch_jvp.tangent_heights = at::ones(
        patch_jvp.primal.heights.sizes(),
        patch_jvp.primal.heights.options().dtype(at::kDouble));
    require_throws(
        [&] { (void)scattering_patch_integral_eval_jvp(patch_jvp); },
        "wrong optional patch tangent dtype must fail loudly");
}

// Every guarded entry point resolves exactly one CUDA device from its primal
// request, so a request whose tensors straddle two devices must fail loudly
// instead of launching against foreign pointers. Each fixture is exercised on
// device 0 first, so a rejection below is a rejection of the device split and
// not of an unrelated contract.
void test_cross_device_inputs_fail_loudly() {
    if (at::cuda::device_count() < 2) {
        std::cout << "single CUDA device; cross-device rejection skipped\n";
        return;
    }
    const auto second = at::Device(at::kCUDA, 1);

    const auto table = nonempty_table_request();
    (void)scattering_table_eval(table);
    auto bad_table_direction_device = table;
    bad_table_direction_device.wo = bad_table_direction_device.wo.to(second);
    require_throws(
        [&] { (void)scattering_table_eval(bad_table_direction_device); },
        "cross-device table direction must fail loudly");
    auto bad_table_payload_device = table;
    bad_table_payload_device.f_tm = bad_table_payload_device.f_tm.to(second);
    require_throws(
        [&] { (void)scattering_table_eval(bad_table_payload_device); },
        "cross-device table payload must fail loudly");

    ScatteringTableEvalBackwardRequest table_backward;
    table_backward.primal = table;
    table_backward.grad_f_te = at::ones({1}, table.wi.options().device(second));
    table_backward.need_grad_directions = true;
    require_throws(
        [&] { (void)scattering_table_eval_backward(table_backward); },
        "cross-device table cotangent must fail loudly");

    ScatteringTableEvalJvpRequest table_jvp;
    table_jvp.primal = table;
    table_jvp.tangent_wi = at::ones({1, 3}, table.wi.options().device(second));
    require_throws(
        [&] { (void)scattering_table_eval_jvp(table_jvp); },
        "cross-device table tangent must fail loudly");

    const auto sample = nonempty_table_sample_request();
    (void)scattering_table_sample(sample);
    auto bad_sample_device = sample;
    bad_sample_device.sample_density = bad_sample_device.sample_density.to(second);
    require_throws(
        [&] { (void)scattering_table_sample(bad_sample_device); },
        "cross-device table sample density must fail loudly");

    const auto pdf = nonempty_table_pdf_request();
    (void)scattering_table_pdf(pdf);
    auto bad_pdf_device = pdf;
    bad_pdf_device.sample_density = bad_pdf_device.sample_density.to(second);
    require_throws(
        [&] { (void)scattering_table_pdf(bad_pdf_device); },
        "cross-device table PDF density must fail loudly");

    const auto ensemble = nonempty_ensemble_request();
    (void)scattering_ensemble_eval(ensemble);
    auto bad_ensemble_device = ensemble;
    bad_ensemble_device.n_o = bad_ensemble_device.n_o.to(second);
    require_throws(
        [&] { (void)scattering_ensemble_eval(bad_ensemble_device); },
        "cross-device ensemble input must fail loudly");

    ScatteringEnsembleEvalBackwardRequest ensemble_backward;
    ensemble_backward.primal = ensemble;
    ensemble_backward.grad_gain = at::ones({1}, ensemble.wo_rows.options().device(second));
    ensemble_backward.need_grad_rows = true;
    require_throws(
        [&] { (void)scattering_ensemble_eval_backward(ensemble_backward); },
        "cross-device ensemble cotangent must fail loudly");

    ScatteringEnsembleEvalJvpRequest ensemble_jvp;
    ensemble_jvp.primal = ensemble;
    ensemble_jvp.tangent_r2_rows = at::ones({1}, ensemble.r2_rows.options().device(second));
    require_throws(
        [&] { (void)scattering_ensemble_eval_jvp(ensemble_jvp); },
        "cross-device ensemble tangent must fail loudly");

    const auto patch = nonempty_patch_request();
    (void)scattering_patch_integral_eval(patch);
    auto bad_patch_device = patch;
    bad_patch_device.heights = bad_patch_device.heights.to(second);
    require_throws(
        [&] { (void)scattering_patch_integral_eval(bad_patch_device); },
        "cross-device patch input must fail loudly");

    ScatteringPatchIntegralEvalBackwardRequest patch_backward;
    patch_backward.primal = patch;
    patch_backward.grad_total = at::ones({}, cuda_options(at::kComplexFloat).device(second));
    patch_backward.need_grad_heights = true;
    require_throws(
        [&] { (void)scattering_patch_integral_eval_backward(patch_backward); },
        "cross-device patch cotangent must fail loudly");

    ScatteringPatchIntegralEvalJvpRequest patch_jvp;
    patch_jvp.primal = patch;
    patch_jvp.tangent_heights = at::ones(
        patch.heights.sizes(), patch.heights.options().device(second));
    require_throws(
        [&] { (void)scattering_patch_integral_eval_jvp(patch_jvp); },
        "cross-device patch tangent must fail loudly");

    const auto transmission = nonempty_transmission_request();
    (void)field_transmission_sequence(transmission);
    auto bad_transmission_device = transmission;
    bad_transmission_device.layer_eps_r =
        bad_transmission_device.layer_eps_r.to(second);
    require_throws(
        [&] { (void)field_transmission_sequence(bad_transmission_device); },
        "cross-device transmission input must fail loudly");

    TransmissionSequenceJvpRequest transmission_jvp;
    transmission_jvp.primal = transmission;
    transmission_jvp.tangent_source =
        at::ones({1, 3}, transmission.source.options().device(second));
    require_throws(
        [&] { (void)field_transmission_sequence_jvp(transmission_jvp); },
        "cross-device transmission tangent must fail loudly");

    const auto layer_stack = nonempty_layer_stack_request();
    (void)em_layer_stack_eval(layer_stack);
    auto bad_layer_stack_device = layer_stack;
    bad_layer_stack_device.layer_eps_r =
        bad_layer_stack_device.layer_eps_r.to(second);
    require_throws(
        [&] { (void)em_layer_stack_eval(bad_layer_stack_device); },
        "cross-device layer-stack input must fail loudly");

    LayerStackBackwardRequest layer_stack_backward;
    layer_stack_backward.primal = layer_stack;
    layer_stack_backward.grad_outputs[0] =
        at::ones({1}, layer_stack.cos_theta.options().device(second));
    layer_stack_backward.need_cos_theta = true;
    require_throws(
        [&] { (void)em_layer_stack_backward(layer_stack_backward); },
        "cross-device layer-stack cotangent must fail loudly");

    const auto wedge = nonempty_wedge_request();
    (void)field_diffraction_wedge(wedge);
    auto bad_wedge_device = wedge;
    bad_wedge_device.target = bad_wedge_device.target.to(second);
    require_throws(
        [&] { (void)field_diffraction_wedge(bad_wedge_device); },
        "cross-device wedge input must fail loudly");
    auto bad_wedge_vertex_device = wedge;
    bad_wedge_vertex_device.vertex_opp1 =
        bad_wedge_vertex_device.vertex_opp1->to(second);
    require_throws(
        [&] { (void)field_diffraction_wedge(bad_wedge_vertex_device); },
        "cross-device wedge vertex must fail loudly");

    DiffractionWedgeJvpRequest wedge_jvp;
    wedge_jvp.primal = wedge;
    wedge_jvp.tangent_source =
        at::ones({1, 3}, wedge.source.options().device(second));
    require_throws(
        [&] { (void)field_diffraction_wedge_jvp(wedge_jvp); },
        "cross-device wedge tangent must fail loudly");
}

}  // namespace

int main() {
    try {
        if (!at::cuda::is_available()) {
            std::cout << "CUDA unavailable; typed scattering compile surface passed\n";
            return 0;
        }
        std::cout << "[RUN] test_table_empty_contracts\n";
        test_table_empty_contracts();
        std::cout << "[RUN] test_ensemble_empty_contract\n";
        test_ensemble_empty_contract();
        std::cout << "[RUN] test_patch_empty_contract\n";
        test_patch_empty_contract();
        std::cout << "[RUN] test_table_nonempty_primal_ad_sample_pdf\n";
        test_table_nonempty_primal_ad_sample_pdf();
        std::cout << "[RUN] test_ensemble_nonempty_primal_ad\n";
        test_ensemble_nonempty_primal_ad();
        std::cout << "[RUN] test_patch_nonempty_primal_ad\n";
        test_patch_nonempty_primal_ad();
        std::cout << "[RUN] test_nondefault_stream_dependency\n";
        test_nondefault_stream_dependency();
        std::cout << "[RUN] test_invalid_rows_short_circuit_poison\n";
        test_invalid_rows_short_circuit_poison();
        std::cout << "[RUN] test_invalid_contracts_fail_loudly\n";
        test_invalid_contracts_fail_loudly();
        std::cout << "[RUN] test_cross_device_inputs_fail_loudly\n";
        test_cross_device_inputs_fail_loudly();
        std::cout << "rayd::torch scattering direct contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rayd::torch scattering direct contract failure: "
                  << error.what() << '\n';
        return 1;
    }
}
