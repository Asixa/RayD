#include <rayd/torch/rf/scattering.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using namespace rayd::torch;

static_assert(std::is_same_v<
              decltype(&scattering_chain_ensemble_eval),
              ScatteringChainEnsembleEvalResult (*)(
                  const ScatteringChainEnsembleEvalRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_chain_ensemble_eval_backward),
              ScatteringChainEnsembleEvalBackwardResult (*)(
                  const ScatteringChainEnsembleEvalBackwardRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_chain_ensemble_eval_jvp),
              ScatteringChainEnsembleEvalJvpResult (*)(
                  const ScatteringChainEnsembleEvalJvpRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_chain_realization_eval),
              ScatteringChainRealizationEvalResult (*)(
                  const ScatteringChainRealizationEvalRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_chain_realization_eval_backward),
              ScatteringChainRealizationEvalBackwardResult (*)(
                  const ScatteringChainRealizationEvalBackwardRequest&)>);
static_assert(std::is_same_v<
              decltype(&scattering_chain_realization_eval_jvp),
              ScatteringChainRealizationEvalJvpResult (*)(
                  const ScatteringChainRealizationEvalJvpRequest&)>);

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
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

void require_finite(const at::Tensor& tensor, const std::string& message) {
    require(at::isfinite(tensor).all().item<bool>(), message);
}

void require_exact_zero(const at::Tensor& tensor, const std::string& message) {
    require(at::count_nonzero(tensor).item<int64_t>() == 0, message);
}

at::TensorOptions cuda_options(c10::ScalarType dtype) {
    return at::TensorOptions().dtype(dtype).device(at::Device(at::kCUDA, 0));
}

at::Tensor repeated_vec3(
    std::initializer_list<float> values,
    int64_t rows) {
    return at::tensor(values, cuda_options(at::kFloat))
        .reshape({1, 3})
        .repeat({rows, 1})
        .contiguous();
}

ScatteringChainEnsembleEvalRequest ensemble_request(int64_t rows = 1) {
    const auto f32 = cuda_options(at::kFloat);
    const auto i32 = cuda_options(at::kInt);
    const auto i64 = cuda_options(at::kLong);
    const auto leg_vec = at::zeros({rows, 8, 3}, f32);
    const auto leg_one = at::ones({rows, 8}, f32);
    const auto leg_zero = at::zeros({rows, 8}, f32);
    const auto depth = at::zeros({rows}, i32);
    const auto scalar_one = at::ones({rows}, f32);
    return {
        at::ones({rows}, cuda_options(at::kBool)),
        repeated_vec3({1.0F, 0.0F, 0.0F}, rows),
        repeated_vec3({1.0F, 0.0F, 0.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 0.0F}, rows),
        repeated_vec3({1.0F, 0.0F, 1.0F}, rows),
        leg_vec.clone(), leg_vec.clone(),
        4.0F * leg_one, 0.02F * leg_one, leg_one,
        leg_one, 0.1F * leg_one, depth.clone(),
        leg_vec.clone(), leg_vec.clone(),
        4.0F * leg_one, 0.02F * leg_one, leg_one,
        leg_one, 0.1F * leg_one, depth.clone(),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        repeated_vec3({1.0F, 0.0F, 0.0F}, rows),
        repeated_vec3({0.0F, 1.0F, 0.0F}, rows),
        repeated_vec3({1.0F, 0.0F, 0.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        scalar_one.clone(), scalar_one.clone(),
        repeated_vec3({0.0F, 0.0F, -1.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        scalar_one.clone(), scalar_one.clone(), scalar_one.clone(),
        at::zeros({rows}, i32),
        at::ones({16}, f32), at::ones({16}, f32),
        at::zeros({1}, i64),
        at::tensor({2, 2, 2, 2}, i32).reshape({1, 4}),
        at::zeros({1}, i32),
        3.0e-4, -1.0, 3.0e9,
    };
}

ScatteringChainRealizationEvalRequest realization_request(int64_t rows = 1) {
    const auto f32 = cuda_options(at::kFloat);
    const auto i32 = cuda_options(at::kInt);
    const auto i64 = cuda_options(at::kLong);
    const auto leg_vec = at::zeros({rows, 8, 3}, f32);
    const auto leg_one = at::ones({rows, 8}, f32);
    const auto depth = at::zeros({rows}, i32);
    const auto scalar_one = at::ones({rows}, f32);
    const auto patch_tris = at::tensor(
        {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
        f32).reshape({1, 3, 3});
    const auto patch_uvs = at::tensor(
        {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F},
        f32).reshape({1, 3, 2});
    return {
        at::ones({rows}, cuda_options(at::kBool)),
        patch_tris, patch_uvs, at::zeros({rows}, i64),
        repeated_vec3({0.0F, 0.0F, -1.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        repeated_vec3({1.0F / 3.0F, 1.0F / 3.0F, 0.0F}, rows),
        repeated_vec3({0.0F, 0.0F, 1.0F}, rows),
        leg_vec.clone(), leg_vec.clone(),
        4.0F * leg_one, 0.02F * leg_one, leg_one,
        leg_one, 0.1F * leg_one, depth.clone(),
        leg_vec.clone(), leg_vec.clone(),
        4.0F * leg_one, 0.02F * leg_one, leg_one,
        leg_one, 0.1F * leg_one, depth.clone(),
        repeated_vec3({1.0F, 0.0F, 0.0F}, rows),
        repeated_vec3({1.0F, 0.0F, 0.0F}, rows),
        scalar_one.clone(), scalar_one.clone(),
        scalar_one.clone(), scalar_one.clone(),
        repeated_vec3({1.0F / 3.0F, 1.0F / 3.0F, 0.0F}, rows),
        at::zeros({4, 4}, f32),
        0.8F * scalar_one,
        at::zeros({rows}, i32),
        at::zeros({1}, i32), at::ones({1}, i32),
        at::tensor({0.1F}, f32), at::tensor({4.0F}, f32),
        at::tensor({0.02F}, f32), at::tensor({1.0F}, f32),
        at::zeros({256}, f32), at::zeros({256}, f32),
        at::full({256}, 1.0F / 256.0F, f32),
        62.8753506586, 3.0e9,
    };
}

void set_full_depth(ScatteringChainEnsembleEvalRequest& request) {
    const auto z = at::arange(1, 9, request.tx_pol.options()) / 10.0F;
    const auto zero = at::zeros_like(z);
    request.c1_positions =
        at::stack({zero, zero, 1.0F - z}, 1).unsqueeze(0).contiguous();
    request.c2_positions =
        at::stack({zero, zero, z}, 1).unsqueeze(0).contiguous();
    request.c1_normals =
        repeated_vec3({0.0F, 0.0F, 1.0F}, 8).reshape({1, 8, 3});
    request.c2_normals = request.c1_normals.clone();
    request.c1_depth.fill_(8);
    request.c2_depth.fill_(8);
}

void set_full_depth(ScatteringChainRealizationEvalRequest& request) {
    const auto z = at::arange(1, 9, request.patch_tris.options()) / 10.0F;
    const auto zero = at::zeros_like(z);
    request.c1_positions =
        at::stack({zero, zero, 1.0F - z}, 1).unsqueeze(0).contiguous();
    request.c2_positions =
        at::stack({zero, zero, z}, 1).unsqueeze(0).contiguous();
    request.c1_normals =
        repeated_vec3({0.0F, 0.0F, 1.0F}, 8).reshape({1, 8, 3});
    request.c2_normals = request.c1_normals.clone();
    request.c1_depth.fill_(8);
    request.c2_depth.fill_(8);
}

void test_nonempty_six_typed_operations() {
    auto ensemble = ensemble_request();
    const auto ensemble_forward = scattering_chain_ensemble_eval(ensemble);
    require(ensemble_forward.gain.sizes() == at::IntArrayRef({1}),
            "ensemble gain shape differs");
    require(ensemble_forward.keep.scalar_type() == at::kBool,
            "ensemble keep dtype differs");
    require_finite(ensemble_forward.gain, "ensemble primal is not finite");
    require_finite(ensemble_forward.amplitude, "ensemble amplitude is not finite");
    require_finite(ensemble_forward.length, "ensemble length is not finite");

    ScatteringChainEnsembleEvalBackwardRequest ensemble_backward;
    ensemble_backward.primal = ensemble;
    ensemble_backward.grad_gain = at::ones({1}, ensemble.tx_pol.options());
    ensemble_backward.need_grad_chain1 = true;
    ensemble_backward.need_grad_chain2 = true;
    ensemble_backward.need_grad_tables = true;
    ensemble_backward.need_grad_coefficient = true;
    ensemble_backward.need_grad_frequency = true;
    const auto ensemble_gradient =
        scattering_chain_ensemble_eval_backward(ensemble_backward);
    require(
        ensemble_gradient.grad_c1_eps_r && ensemble_gradient.grad_c1_sigma_e &&
            ensemble_gradient.grad_c1_gain &&
            ensemble_gradient.grad_c1_thickness &&
            ensemble_gradient.grad_c2_eps_r &&
            ensemble_gradient.grad_c2_sigma_e &&
            ensemble_gradient.grad_c2_gain &&
            ensemble_gradient.grad_c2_thickness &&
            ensemble_gradient.grad_f_te && ensemble_gradient.grad_f_tm &&
            ensemble_gradient.grad_coefficient &&
            ensemble_gradient.grad_frequency,
        "ensemble backward omitted a requested field");
    require_finite(*ensemble_gradient.grad_c1_eps_r,
                   "ensemble chain gradient is not finite");

    ScatteringChainEnsembleEvalJvpRequest ensemble_jvp;
    ensemble_jvp.primal = ensemble;
    ensemble_jvp.tangent_c1_eps_r = at::zeros_like(ensemble.c1_eps_r);
    ensemble_jvp.tangent_d_i = at::zeros_like(ensemble.d_i);
    ensemble_jvp.tangent_frequency = 1.0;
    const auto ensemble_tangent = scattering_chain_ensemble_eval_jvp(ensemble_jvp);
    require_finite(ensemble_tangent.tangent_gain,
                   "ensemble gain tangent is not finite");
    require_finite(ensemble_tangent.tangent_amplitude,
                   "ensemble amplitude tangent is not finite");
    require_finite(ensemble_tangent.tangent_length,
                   "ensemble length tangent is not finite");

    auto realization = realization_request();
    const auto realization_forward = scattering_chain_realization_eval(realization);
    require(realization_forward.total.dim() == 0,
            "realization total must be scalar");
    require(realization_forward.path_field.scalar_type() == at::kComplexFloat,
            "realization path field dtype differs");
    require_finite(realization_forward.total, "realization total is not finite");
    require_finite(realization_forward.path_field,
                   "realization path field is not finite");
    require_finite(realization_forward.path_gain,
                   "realization path gain is not finite");

    ScatteringChainRealizationEvalBackwardRequest realization_backward;
    realization_backward.primal = realization;
    realization_backward.grad_total =
        at::ones({}, realization.patch_tris.options().dtype(at::kComplexFloat));
    realization_backward.need_grad_heights = true;
    realization_backward.need_grad_layers = true;
    realization_backward.need_grad_chain1 = true;
    realization_backward.need_grad_chain2 = true;
    realization_backward.need_grad_geometry = true;
    realization_backward.need_grad_k0 = true;
    realization_backward.need_grad_frequency = true;
    const auto realization_gradient =
        scattering_chain_realization_eval_backward(realization_backward);
    require(
        realization_gradient.grad_d_i && realization_gradient.grad_d_o &&
            realization_gradient.grad_c1_positions &&
            realization_gradient.grad_c1_normals &&
            realization_gradient.grad_c2_positions &&
            realization_gradient.grad_c2_normals &&
            realization_gradient.grad_l1 && realization_gradient.grad_l2 &&
            realization_gradient.grad_sp1 && realization_gradient.grad_sp2 &&
            realization_gradient.grad_centroids,
        "realization backward omitted one of 11 geometry gradients");
    require(
        realization_gradient.grad_heights &&
            realization_gradient.grad_layer_thickness &&
            realization_gradient.grad_layer_eps_r &&
            realization_gradient.grad_layer_sigma_e &&
            realization_gradient.grad_c1_eps_r &&
            realization_gradient.grad_c1_sigma_e &&
            realization_gradient.grad_c1_gain &&
            realization_gradient.grad_c1_thickness &&
            realization_gradient.grad_c2_eps_r &&
            realization_gradient.grad_c2_sigma_e &&
            realization_gradient.grad_c2_gain &&
            realization_gradient.grad_c2_thickness &&
            realization_gradient.grad_k0 && realization_gradient.grad_frequency,
        "realization backward omitted another requested field");
    require_finite(*realization_gradient.grad_d_i,
                   "realization geometry gradient is not finite");

    ScatteringChainRealizationEvalJvpRequest realization_jvp;
    realization_jvp.primal = realization;
    realization_jvp.tangent_heights = at::zeros_like(realization.heights);
    realization_jvp.tangent_d_i = at::zeros_like(realization.d_i);
    realization_jvp.tangent_frequency = 1.0;
    const auto realization_tangent =
        scattering_chain_realization_eval_jvp(realization_jvp);
    require_finite(realization_tangent.tangent_total,
                   "realization total tangent is not finite");
    require_finite(realization_tangent.tangent_path_field,
                   "realization field tangent is not finite");
    require_finite(realization_tangent.tangent_path_gain,
                   "realization gain tangent is not finite");
}

void test_ad_capability_and_optional_gating() {
    ScatteringChainEnsembleEvalBackwardRequest unsupported;
    unsupported.primal = ensemble_request();
    unsupported.need_grad_geometry = true;
    require_throws(
        [&] { (void)scattering_chain_ensemble_eval_backward(unsupported); },
        "ensemble geometry VJP must fail loudly");

    ScatteringChainEnsembleEvalBackwardRequest ensemble_none;
    ensemble_none.primal = ensemble_request();
    ensemble_none.grad_gain = at::ones({1}, ensemble_none.primal.tx_pol.options());
    const auto ensemble_gated =
        scattering_chain_ensemble_eval_backward(ensemble_none);
    require(
        !ensemble_gated.grad_c1_eps_r && !ensemble_gated.grad_c2_eps_r &&
            !ensemble_gated.grad_f_te && !ensemble_gated.grad_coefficient &&
            !ensemble_gated.grad_frequency,
        "ensemble backward returned an unrequested field");

    ScatteringChainRealizationEvalBackwardRequest realization_none;
    realization_none.primal = realization_request();
    realization_none.grad_total = at::ones(
        {}, realization_none.primal.patch_tris.options().dtype(at::kComplexFloat));
    const auto realization_gated =
        scattering_chain_realization_eval_backward(realization_none);
    require(
        !realization_gated.grad_heights &&
            !realization_gated.grad_layer_thickness &&
            !realization_gated.grad_c1_eps_r &&
            !realization_gated.grad_c2_eps_r && !realization_gated.grad_d_i &&
            !realization_gated.grad_centroids && !realization_gated.grad_k0 &&
            !realization_gated.grad_frequency,
        "realization backward returned an unrequested field");
}

void test_max_depth_primal_and_ad() {
    auto ensemble = ensemble_request();
    set_full_depth(ensemble);
    const auto ensemble_forward = scattering_chain_ensemble_eval(ensemble);
    require(ensemble_forward.gain.sizes() == at::IntArrayRef({1}),
            "max-depth ensemble gain shape differs");
    require_finite(ensemble_forward.gain,
                   "max-depth ensemble primal is not finite");

    ScatteringChainEnsembleEvalBackwardRequest ensemble_backward;
    ensemble_backward.primal = ensemble;
    ensemble_backward.grad_gain = at::ones({1}, ensemble.tx_pol.options());
    ensemble_backward.need_grad_chain1 = true;
    const auto ensemble_gradient =
        scattering_chain_ensemble_eval_backward(ensemble_backward);
    require(ensemble_gradient.grad_c1_eps_r.has_value(),
            "max-depth ensemble backward omitted requested chain gradient");
    require_finite(*ensemble_gradient.grad_c1_eps_r,
                   "max-depth ensemble gradient is not finite");

    ScatteringChainEnsembleEvalJvpRequest ensemble_jvp;
    ensemble_jvp.primal = ensemble;
    ensemble_jvp.tangent_c1_eps_r = at::zeros_like(ensemble.c1_eps_r);
    const auto ensemble_tangent = scattering_chain_ensemble_eval_jvp(ensemble_jvp);
    require_finite(ensemble_tangent.tangent_gain,
                   "max-depth ensemble tangent is not finite");

    auto realization = realization_request();
    set_full_depth(realization);
    const auto realization_forward = scattering_chain_realization_eval(realization);
    require(realization_forward.path_field.sizes() == at::IntArrayRef({1}),
            "max-depth realization field shape differs");
    require_finite(realization_forward.total,
                   "max-depth realization primal is not finite");
    require_finite(realization_forward.path_field,
                   "max-depth realization field is not finite");

    ScatteringChainRealizationEvalBackwardRequest realization_backward;
    realization_backward.primal = realization;
    realization_backward.grad_total = at::ones(
        {}, realization.patch_tris.options().dtype(at::kComplexFloat));
    realization_backward.need_grad_chain1 = true;
    const auto realization_gradient =
        scattering_chain_realization_eval_backward(realization_backward);
    require(realization_gradient.grad_c1_eps_r.has_value(),
            "max-depth realization backward omitted requested chain gradient");
    require_finite(*realization_gradient.grad_c1_eps_r,
                   "max-depth realization gradient is not finite");

    ScatteringChainRealizationEvalJvpRequest realization_jvp;
    realization_jvp.primal = realization;
    realization_jvp.tangent_c1_eps_r = at::zeros_like(realization.c1_eps_r);
    const auto realization_tangent =
        scattering_chain_realization_eval_jvp(realization_jvp);
    require_finite(realization_tangent.tangent_total,
                   "max-depth realization tangent is not finite");
}

void test_zero_rows() {
    auto ensemble = ensemble_request(0);
    const auto forward = scattering_chain_ensemble_eval(ensemble);
    require(forward.gain.numel() == 0 && forward.keep.numel() == 0,
            "zero-row ensemble outputs must be empty");
    ScatteringChainEnsembleEvalBackwardRequest backward;
    backward.primal = ensemble;
    backward.need_grad_chain1 = true;
    const auto gradient = scattering_chain_ensemble_eval_backward(backward);
    require(gradient.grad_c1_eps_r && gradient.grad_c1_eps_r->numel() == 0,
            "zero-row ensemble requested gradient must be empty and defined");
    ScatteringChainEnsembleEvalJvpRequest jvp;
    jvp.primal = ensemble;
    const auto tangent = scattering_chain_ensemble_eval_jvp(jvp);
    require(tangent.tangent_gain.numel() == 0,
            "zero-row ensemble tangent must be empty");

    auto realization = realization_request(0);
    const auto realization_forward = scattering_chain_realization_eval(realization);
    require(realization_forward.path_field.numel() == 0,
            "zero-row realization per-row output must be empty");
    require(at::abs(realization_forward.total).item<float>() == 0.0F,
            "zero-row realization total must be zero");
    ScatteringChainRealizationEvalBackwardRequest realization_backward;
    realization_backward.primal = realization;
    realization_backward.grad_total = at::ones(
        {}, realization.patch_tris.options().dtype(at::kComplexFloat));
    realization_backward.need_grad_geometry = true;
    const auto realization_gradient =
        scattering_chain_realization_eval_backward(realization_backward);
    require(realization_gradient.grad_d_i &&
                realization_gradient.grad_d_i->numel() == 0,
            "zero-row realization geometry gradient must be empty and defined");
    ScatteringChainRealizationEvalJvpRequest realization_jvp;
    realization_jvp.primal = realization;
    const auto realization_tangent =
        scattering_chain_realization_eval_jvp(realization_jvp);
    require(at::abs(realization_tangent.tangent_total).item<float>() == 0.0F,
            "zero-row realization tangent total must be zero");
}

void test_dmax_and_optional_contracts_fail_loudly() {
    auto bad_ensemble_valid_dtype = ensemble_request();
    bad_ensemble_valid_dtype.valid = at::ones(
        {1}, bad_ensemble_valid_dtype.tx_pol.options());
    require_throws(
        [&] { (void)scattering_chain_ensemble_eval(bad_ensemble_valid_dtype); },
        "wrong chain ensemble valid dtype must fail loudly");
    auto bad_realization_valid_shape = realization_request();
    bad_realization_valid_shape.valid = at::ones(
        {2}, bad_realization_valid_shape.valid.options());
    require_throws(
        [&] { (void)scattering_chain_realization_eval(bad_realization_valid_shape); },
        "wrong chain realization valid shape must fail loudly");

    auto ensemble = ensemble_request();
    ensemble.c1_positions = at::zeros({1, 9, 3}, ensemble.tx_pol.options());
    require_throws(
        [&] { (void)scattering_chain_ensemble_eval(ensemble); },
        "ensemble padded Dmax=9 must fail loudly");

    auto realization = realization_request();
    realization.c2_positions =
        at::zeros({1, 9, 3}, realization.patch_tris.options());
    require_throws(
        [&] { (void)scattering_chain_realization_eval(realization); },
        "realization padded Dmax=9 must fail loudly");

    ScatteringChainEnsembleEvalJvpRequest bad_tangent;
    bad_tangent.primal = ensemble_request();
    bad_tangent.tangent_c1_eps_r =
        at::zeros({1, 9}, bad_tangent.primal.tx_pol.options());
    require_throws(
        [&] { (void)scattering_chain_ensemble_eval_jvp(bad_tangent); },
        "wrong ensemble optional tangent shape must fail loudly");

    ScatteringChainEnsembleEvalBackwardRequest bad_ensemble_backward_primal;
    bad_ensemble_backward_primal.primal = ensemble_request();
    bad_ensemble_backward_primal.primal.c1_eps_r = at::zeros(
        {1, 7}, bad_ensemble_backward_primal.primal.tx_pol.options());
    require_throws(
        [&] {
            (void)scattering_chain_ensemble_eval_backward(
                bad_ensemble_backward_primal);
        },
        "ensemble backward must validate the complete primal leg contract");

    ScatteringChainEnsembleEvalJvpRequest bad_ensemble_jvp_primal;
    bad_ensemble_jvp_primal.primal = ensemble_request();
    bad_ensemble_jvp_primal.primal.c1_sigma_e = at::zeros(
        {1, 7}, bad_ensemble_jvp_primal.primal.tx_pol.options());
    require_throws(
        [&] {
            (void)scattering_chain_ensemble_eval_jvp(
                bad_ensemble_jvp_primal);
        },
        "ensemble JVP must validate the complete primal leg contract");

    ScatteringChainRealizationEvalBackwardRequest bad_cotangent;
    bad_cotangent.primal = realization_request();
    bad_cotangent.grad_total =
        at::ones({}, bad_cotangent.primal.patch_tris.options());
    require_throws(
        [&] { (void)scattering_chain_realization_eval_backward(bad_cotangent); },
        "wrong realization cotangent dtype must fail loudly");

    ScatteringChainRealizationEvalBackwardRequest bad_realization_leg;
    bad_realization_leg.primal = realization_request();
    bad_realization_leg.primal.c1_normals =
        bad_realization_leg.primal.c1_normals.to(at::kDouble);
    bad_realization_leg.grad_total = at::ones(
        {}, bad_realization_leg.primal.patch_tris.options().dtype(
                at::kComplexFloat));
    require_throws(
        [&] {
            (void)scattering_chain_realization_eval_backward(
                bad_realization_leg);
        },
        "realization backward must validate the complete primal leg contract");

    ScatteringChainRealizationEvalJvpRequest bad_realization_layers;
    bad_realization_layers.primal = realization_request();
    bad_realization_layers.primal.layer_eps_r = at::ones(
        {2}, bad_realization_layers.primal.patch_tris.options());
    require_throws(
        [&] {
            (void)scattering_chain_realization_eval_jvp(
                bad_realization_layers);
        },
        "realization JVP must validate the complete primal layer contract");

    auto bad_realization_quad_device = realization_request();
    bad_realization_quad_device.quad_w =
        bad_realization_quad_device.quad_w.cpu();
    require_throws(
        [&] {
            (void)scattering_chain_realization_eval(
                bad_realization_quad_device);
        },
        "realization primal must reject a CPU quadrature tensor");
}

void test_nondefault_stream_dependency() {
    const auto reference = scattering_chain_ensemble_eval(ensemble_request());
    at::cuda::getDefaultCUDAStream().synchronize();

    auto request = ensemble_request();
    request.weights.zero_();
    const auto producer = c10::cuda::getStreamFromPool(false, 0);
    const auto consumer = c10::cuda::getStreamFromPool(false, 0);
    require(producer.stream() != consumer.stream(),
            "stream fixtures must differ");
    {
        c10::cuda::CUDAStreamGuard guard(producer);
        request.weights.fill_(1.0F);
    }
    cudaEvent_t ready = nullptr;
    cudaEvent_t complete = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, producer.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(consumer.stream(), ready, 0));

    ScatteringChainEnsembleEvalResult result;
    {
        c10::cuda::CUDAStreamGuard guard(consumer);
        result = scattering_chain_ensemble_eval(request);
        require(c10::cuda::getCurrentCUDAStream(0).stream() == consumer.stream(),
                "chain ensemble changed the caller's active stream");
        C10_CUDA_CHECK(cudaEventRecord(complete, consumer.stream()));
    }
    C10_CUDA_CHECK(cudaEventSynchronize(complete));
    C10_CUDA_CHECK(cudaEventDestroy(complete));
    C10_CUDA_CHECK(cudaEventDestroy(ready));
    require(at::allclose(result.gain, reference.gain, 1.0e-5, 1.0e-8),
            "chain ensemble ignored a non-default-stream dependency");
}

void test_sparse_invalid_rows_short_circuit_poison() {
    auto ensemble = ensemble_request(2);
    ensemble.valid.select(0, 1).fill_(false);
    ensemble.source.select(0, 1).fill_(std::numeric_limits<float>::quiet_NaN());
    ensemble.material_id.select(0, 1).fill_(std::numeric_limits<int>::max());
    const auto ensemble_forward = scattering_chain_ensemble_eval(ensemble);
    require_finite(ensemble_forward.gain.select(0, 0), "valid sparse ensemble row must be finite");
    require_exact_zero(ensemble_forward.gain.select(0, 1), "invalid sparse ensemble gain must be zero");
    require_exact_zero(ensemble_forward.amplitude.select(0, 1), "invalid sparse ensemble amplitude must be zero");
    require_exact_zero(ensemble_forward.length.select(0, 1), "invalid sparse ensemble length must be zero");
    require_exact_zero(ensemble_forward.keep.select(0, 1), "invalid sparse ensemble keep must be false");
    ScatteringChainEnsembleEvalBackwardRequest ensemble_backward;
    ensemble_backward.primal = ensemble;
    ensemble_backward.grad_gain = at::ones({2}, ensemble.tx_pol.options());
    ensemble_backward.need_grad_chain1 = true;
    ensemble_backward.need_grad_chain2 = true;
    ensemble_backward.need_grad_tables = true;
    ensemble_backward.need_grad_coefficient = true;
    ensemble_backward.need_grad_frequency = true;
    const auto ensemble_grad = scattering_chain_ensemble_eval_backward(ensemble_backward);
    for (const auto* tensor : {
             &*ensemble_grad.grad_c1_eps_r, &*ensemble_grad.grad_c2_eps_r,
             &*ensemble_grad.grad_f_te, &*ensemble_grad.grad_f_tm,
             &*ensemble_grad.grad_coefficient, &*ensemble_grad.grad_frequency})
        require_finite(*tensor, "invalid sparse ensemble row contaminated a shared gradient");
    require_exact_zero(
        ensemble_grad.grad_c1_eps_r->select(0, 1),
        "invalid sparse ensemble chain gradient must be zero");
    ScatteringChainEnsembleEvalJvpRequest ensemble_jvp;
    ensemble_jvp.primal = ensemble;
    ensemble_jvp.tangent_coefficient = 1.0;
    const auto ensemble_tangent = scattering_chain_ensemble_eval_jvp(ensemble_jvp);
    require_exact_zero(ensemble_tangent.tangent_gain.select(0, 1), "invalid sparse ensemble gain JVP must be zero");
    require_exact_zero(ensemble_tangent.tangent_amplitude.select(0, 1), "invalid sparse ensemble amplitude JVP must be zero");
    require_exact_zero(ensemble_tangent.tangent_length.select(0, 1), "invalid sparse ensemble length JVP must be zero");

    auto realization = realization_request(2);
    realization.valid.select(0, 1).fill_(false);
    realization.rows.select(0, 1).fill_(std::numeric_limits<int64_t>::max());
    realization.d_i.select(0, 1).fill_(std::numeric_limits<float>::quiet_NaN());
    realization.material_id.select(0, 1).fill_(std::numeric_limits<int>::max());
    const auto realization_forward = scattering_chain_realization_eval(realization);
    require_finite(realization_forward.total, "invalid sparse realization row contaminated total");
    require_exact_zero(realization_forward.path_field.select(0, 1), "invalid sparse realization field must be zero");
    require_exact_zero(realization_forward.path_gain.select(0, 1), "invalid sparse realization gain must be zero");
    require_exact_zero(realization_forward.integral.select(0, 1), "invalid sparse realization integral must be zero");
    require_exact_zero(realization_forward.row_value.select(0, 1), "invalid sparse realization value must be zero");
    ScatteringChainRealizationEvalBackwardRequest realization_backward;
    realization_backward.primal = realization;
    realization_backward.grad_total = at::ones({}, realization.heights.options().dtype(at::kComplexFloat));
    realization_backward.grad_path_gain = at::ones({2}, realization.heights.options());
    realization_backward.need_grad_heights = true;
    realization_backward.need_grad_layers = true;
    realization_backward.need_grad_chain1 = true;
    realization_backward.need_grad_chain2 = true;
    realization_backward.need_grad_geometry = true;
    realization_backward.need_grad_k0 = true;
    realization_backward.need_grad_frequency = true;
    const auto realization_grad = scattering_chain_realization_eval_backward(realization_backward);
    for (const auto* tensor : {
             &*realization_grad.grad_heights, &*realization_grad.grad_layer_eps_r,
             &*realization_grad.grad_c1_eps_r, &*realization_grad.grad_c2_eps_r,
             &*realization_grad.grad_d_i, &*realization_grad.grad_k0,
             &*realization_grad.grad_frequency})
        require_finite(*tensor, "invalid sparse realization row contaminated a gradient");
    require_exact_zero(realization_grad.grad_d_i->select(0, 1), "invalid sparse realization geometry gradient must be zero");
    require_exact_zero(realization_grad.grad_l1->select(0, 1), "invalid sparse realization length gradient must be zero");
    ScatteringChainRealizationEvalJvpRequest realization_jvp;
    realization_jvp.primal = realization;
    realization_jvp.tangent_k0 = 1.0;
    const auto realization_tangent = scattering_chain_realization_eval_jvp(realization_jvp);
    require_finite(realization_tangent.tangent_total, "invalid sparse realization row contaminated JVP total");
    require_exact_zero(realization_tangent.tangent_path_field.select(0, 1), "invalid sparse realization field JVP must be zero");
    require_exact_zero(realization_tangent.tangent_path_gain.select(0, 1), "invalid sparse realization gain JVP must be zero");
}

}  // namespace

int main() {
    try {
        if (!at::cuda::is_available()) {
            std::cout << "CUDA unavailable; typed chain scattering compile surface passed\n";
            return 0;
        }
        std::cout << "[RUN] test_nonempty_six_typed_operations\n";
        test_nonempty_six_typed_operations();
        std::cout << "[RUN] test_ad_capability_and_optional_gating\n";
        test_ad_capability_and_optional_gating();
        std::cout << "[RUN] test_max_depth_primal_and_ad\n";
        test_max_depth_primal_and_ad();
        std::cout << "[RUN] test_zero_rows\n";
        test_zero_rows();
        std::cout << "[RUN] test_dmax_and_optional_contracts_fail_loudly\n";
        test_dmax_and_optional_contracts_fail_loudly();
        std::cout << "[RUN] test_nondefault_stream_dependency\n";
        test_nondefault_stream_dependency();
        std::cout << "[RUN] test_sparse_invalid_rows_short_circuit_poison\n";
        test_sparse_invalid_rows_short_circuit_poison();
        std::cout << "rayd::torch chain scattering direct contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rayd::torch chain scattering direct contract failure: "
                  << error.what() << '\n';
        return 1;
    }
}
