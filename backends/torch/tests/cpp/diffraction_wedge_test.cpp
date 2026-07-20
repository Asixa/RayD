#include <rayd/torch/integration.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

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

at::TensorOptions float_options(int device = 0) {
    return at::TensorOptions()
        .dtype(at::kFloat)
        .device(at::Device(at::kCUDA, device));
}

at::TensorOptions bool_options(int device = 0) {
    return at::TensorOptions()
        .dtype(at::kBool)
        .device(at::Device(at::kCUDA, device));
}

rayd::torch::DiffractionWedgeRequest wedge_request(
    bool with_vertices = true,
    int device = 0) {
    const auto floats = float_options(device);
    const auto masks = bool_options(device);
    rayd::torch::DiffractionWedgeRequest request;
    request.valid = at::ones({1}, masks);
    request.source = at::tensor(
        {-1.2F, -0.8F, 0.2F}, floats).reshape({1, 3});
    request.target = at::tensor(
        {1.0F, 1.1F, -0.1F}, floats).reshape({1, 3});
    request.edge_position = at::zeros({1, 3}, floats);
    request.edge_direction = at::tensor(
        {0.0F, 0.0F, 1.0F}, floats).reshape({1, 3});
    request.edge_t_min = at::tensor({-1.0F}, floats);
    request.edge_t_max = at::tensor({1.0F}, floats);
    request.edge_n0 = at::tensor(
        {0.0F, 1.0F, 0.0F}, floats).reshape({1, 3});
    request.edge_n1 = at::tensor(
        {-1.0F, 0.0F, 0.0F}, floats).reshape({1, 3});
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
    if (with_vertices) {
        request.vertex_v0 = at::tensor(
            {0.0F, 0.0F, -1.0F}, floats).reshape({1, 3});
        request.vertex_v1 = at::tensor(
            {0.0F, 0.0F, 1.0F}, floats).reshape({1, 3});
        request.vertex_opp0 = at::tensor(
            {1.0F, 0.0F, 0.0F}, floats).reshape({1, 3});
        request.vertex_opp1 = at::tensor(
            {0.0F, 1.0F, 0.0F}, floats).reshape({1, 3});
        request.edge_boundary = at::zeros({1}, masks);
    }
    return request;
}

rayd::torch::DiffractionWedgeRequest empty_wedge_request() {
    auto request = wedge_request(true);
#define EMPTY_ROWS(name) request.name = request.name.narrow(0, 0, 0)
    EMPTY_ROWS(valid);
    EMPTY_ROWS(source);
    EMPTY_ROWS(target);
    EMPTY_ROWS(edge_position);
    EMPTY_ROWS(edge_direction);
    EMPTY_ROWS(edge_t_min);
    EMPTY_ROWS(edge_t_max);
    EMPTY_ROWS(edge_n0);
    EMPTY_ROWS(edge_n1);
    EMPTY_ROWS(exterior_angle);
    EMPTY_ROWS(face0_valid);
    EMPTY_ROWS(face0_eps_r);
    EMPTY_ROWS(face0_sigma_e);
    EMPTY_ROWS(face0_mu_r);
    EMPTY_ROWS(face0_gain);
    EMPTY_ROWS(face1_valid);
    EMPTY_ROWS(face1_eps_r);
    EMPTY_ROWS(face1_sigma_e);
    EMPTY_ROWS(face1_mu_r);
    EMPTY_ROWS(face1_gain);
    EMPTY_ROWS(tx_power);
    request.vertex_v0 = request.vertex_v0->narrow(0, 0, 0);
    request.vertex_v1 = request.vertex_v1->narrow(0, 0, 0);
    request.vertex_opp0 = request.vertex_opp0->narrow(0, 0, 0);
    request.vertex_opp1 = request.vertex_opp1->narrow(0, 0, 0);
    request.edge_boundary = request.edge_boundary->narrow(0, 0, 0);
#undef EMPTY_ROWS
    return request;
}

void require_result_schema(
    const rayd::torch::DiffractionWedgeResult& result,
    int64_t rows,
    int device = 0) {
    require(
        result.field_vector.sizes() == at::IntArrayRef({rows, 3}) &&
            result.field_vector.scalar_type() == at::kComplexFloat,
        "wedge field-vector schema differs");
    require(
        result.direction.sizes() == at::IntArrayRef({rows, 3}) &&
            result.direction.scalar_type() == at::kFloat,
        "wedge direction schema differs");
    for (const auto& tensor : {result.field_vector, result.direction})
        require(
            tensor.is_cuda() && tensor.get_device() == device &&
                tensor.is_contiguous(),
            "wedge output must be contiguous CUDA storage");
}

void require_jvp_schema(
    const rayd::torch::DiffractionWedgeJvpResult& result,
    int64_t rows,
    int device = 0) {
    require(
        result.tangent_field_vector.sizes() == at::IntArrayRef({rows, 3}) &&
            result.tangent_field_vector.scalar_type() == at::kComplexFloat,
        "wedge JVP field-vector schema differs");
    require(
        result.tangent_direction.sizes() == at::IntArrayRef({rows, 3}) &&
            result.tangent_direction.scalar_type() == at::kFloat,
        "wedge JVP direction schema differs");
    for (const auto& tensor : {
             result.tangent_field_vector, result.tangent_direction})
        require(
            tensor.is_cuda() && tensor.get_device() == device &&
                tensor.is_contiguous(),
            "wedge JVP output must be contiguous CUDA storage");
}

std::array<const std::optional<at::Tensor>*, 13> gradient_slots(
    const rayd::torch::DiffractionWedgeBackwardResult& result) {
    return {
        &result.grad_source,
        &result.grad_target,
        &result.grad_face0_eps_r,
        &result.grad_face0_sigma_e,
        &result.grad_face0_gain,
        &result.grad_face1_eps_r,
        &result.grad_face1_sigma_e,
        &result.grad_face1_gain,
        &result.grad_frequency,
        &result.grad_vertex_v0,
        &result.grad_vertex_v1,
        &result.grad_vertex_opp0,
        &result.grad_vertex_opp1};
}

void require_all_gradients_present(
    const rayd::torch::DiffractionWedgeBackwardResult& result) {
    for (const auto* slot : gradient_slots(result))
        require(slot->has_value(), "requested wedge gradient must be present");
}

void require_all_gradients_absent(
    const rayd::torch::DiffractionWedgeBackwardResult& result) {
    for (const auto* slot : gradient_slots(result))
        require(!slot->has_value(), "disabled wedge gradient must be absent");
}

void require_all_gradients_zero(
    const rayd::torch::DiffractionWedgeBackwardResult& result) {
    require_all_gradients_present(result);
    for (const auto* slot : gradient_slots(result))
        require(
            at::count_nonzero(**slot).item<int64_t>() == 0,
            "zero-cotangent wedge gradient must be exactly zero");
}

void enable_all_gradients(
    rayd::torch::DiffractionWedgeBackwardRequest& request) {
    request.need_grad_material = true;
    request.need_grad_frequency = true;
    request.need_grad_geometry = true;
    request.need_grad_vertices = true;
}

void test_empty_schema_and_optional_bundle() {
    const auto empty = empty_wedge_request();
    require_result_schema(rayd::torch::field_diffraction_wedge(empty), 0);

    rayd::torch::DiffractionWedgeJvpRequest empty_jvp;
    empty_jvp.primal = empty;
    require_jvp_schema(rayd::torch::field_diffraction_wedge_jvp(empty_jvp), 0);

    rayd::torch::DiffractionWedgeBackwardRequest empty_backward;
    empty_backward.primal = empty;
    enable_all_gradients(empty_backward);
    const auto empty_gradients =
        rayd::torch::field_diffraction_wedge_backward(empty_backward);
    require_all_gradients_zero(empty_gradients);
    require(
        empty_gradients.grad_frequency->sizes() == at::IntArrayRef({1}),
        "empty wedge frequency gradient schema differs");

    auto partial = wedge_request(true);
    partial.edge_boundary.reset();
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(partial); },
        "partial five-tensor wedge bundle must fail");

    rayd::torch::DiffractionWedgeBackwardRequest vertex_need;
    vertex_need.primal = wedge_request(false);
    vertex_need.need_grad_vertices = true;
    require_throws(
        [&] {
            (void)rayd::torch::field_diffraction_wedge_backward(vertex_need);
        },
        "vertex gradient need without winner bundle must fail");

    rayd::torch::DiffractionWedgeJvpRequest vertex_tangent;
    vertex_tangent.primal = wedge_request(false);
    vertex_tangent.tangent_vertex_v0 = at::ones(
        {1, 3}, vertex_tangent.primal.source.options());
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge_jvp(vertex_tangent); },
        "vertex tangent without winner bundle must fail");
}

void test_primal_ad_duality_need_and_zero() {
    const auto primal = wedge_request(true);
    const auto floats = primal.source.options();
    const auto forward = rayd::torch::field_diffraction_wedge(primal);
    require_result_schema(forward, 1);
    require(
        at::isfinite(at::view_as_real(forward.field_vector)).all().item<bool>() &&
            at::isfinite(forward.direction).all().item<bool>(),
        "wedge primal outputs must be finite");

    rayd::torch::DiffractionWedgeJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_source = at::tensor(
        {0.01F, -0.02F, 0.03F}, floats).reshape({1, 3});
    jvp.tangent_target = at::tensor(
        {-0.02F, 0.01F, 0.015F}, floats).reshape({1, 3});
    jvp.tangent_face0_eps_r = at::tensor({0.2F}, floats);
    jvp.tangent_face0_sigma_e = at::tensor({0.001F}, floats);
    jvp.tangent_face0_gain = at::tensor({0.03F}, floats);
    jvp.tangent_face1_eps_r = at::tensor({-0.15F}, floats);
    jvp.tangent_face1_sigma_e = at::tensor({0.002F}, floats);
    jvp.tangent_face1_gain = at::tensor({-0.02F}, floats);
    jvp.tangent_frequency = 1.0e6;
    jvp.tangent_vertex_v0 = at::tensor(
        {0.002F, -0.001F, 0.003F}, floats).reshape({1, 3});
    jvp.tangent_vertex_v1 = at::tensor(
        {-0.001F, 0.002F, -0.002F}, floats).reshape({1, 3});
    jvp.tangent_vertex_opp0 = at::tensor(
        {0.003F, 0.001F, -0.001F}, floats).reshape({1, 3});
    jvp.tangent_vertex_opp1 = at::tensor(
        {-0.002F, 0.003F, 0.001F}, floats).reshape({1, 3});
    const auto tangents = rayd::torch::field_diffraction_wedge_jvp(jvp);
    require_jvp_schema(tangents, 1);

    rayd::torch::DiffractionWedgeBackwardRequest backward;
    backward.primal = primal;
    backward.grad_field_vector = at::ones(
        {1, 3}, floats.dtype(at::kComplexFloat));
    backward.grad_direction = at::full({1, 3}, 0.25F, floats);
    enable_all_gradients(backward);
    const auto gradients =
        rayd::torch::field_diffraction_wedge_backward(backward);
    require_all_gradients_present(gradients);
    at::cuda::getCurrentCUDAStream(0).synchronize();

    const double lhs =
        at::view_as_real(tangents.tangent_field_vector)
            .select(-1, 0).sum().item<double>() +
        0.25 * tangents.tangent_direction.sum().item<double>();
    const double rhs =
        ((*gradients.grad_source) * (*jvp.tangent_source)).sum().item<double>() +
        ((*gradients.grad_target) * (*jvp.tangent_target)).sum().item<double>() +
        ((*gradients.grad_face0_eps_r) * (*jvp.tangent_face0_eps_r))
            .sum().item<double>() +
        ((*gradients.grad_face0_sigma_e) * (*jvp.tangent_face0_sigma_e))
            .sum().item<double>() +
        ((*gradients.grad_face0_gain) * (*jvp.tangent_face0_gain))
            .sum().item<double>() +
        ((*gradients.grad_face1_eps_r) * (*jvp.tangent_face1_eps_r))
            .sum().item<double>() +
        ((*gradients.grad_face1_sigma_e) * (*jvp.tangent_face1_sigma_e))
            .sum().item<double>() +
        ((*gradients.grad_face1_gain) * (*jvp.tangent_face1_gain))
            .sum().item<double>() +
        gradients.grad_frequency->item<double>() * jvp.tangent_frequency +
        ((*gradients.grad_vertex_v0) * (*jvp.tangent_vertex_v0))
            .sum().item<double>() +
        ((*gradients.grad_vertex_v1) * (*jvp.tangent_vertex_v1))
            .sum().item<double>() +
        ((*gradients.grad_vertex_opp0) * (*jvp.tangent_vertex_opp0))
            .sum().item<double>() +
        ((*gradients.grad_vertex_opp1) * (*jvp.tangent_vertex_opp1))
            .sum().item<double>();
    const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
    require(
        std::isfinite(lhs) && std::isfinite(rhs) &&
            std::fabs(lhs - rhs) <= 5.0e-3 * scale,
        "wedge JVP/VJP dot-product duality differs");

    rayd::torch::DiffractionWedgeBackwardRequest disabled;
    disabled.primal = primal;
    require_all_gradients_absent(
        rayd::torch::field_diffraction_wedge_backward(disabled));

    rayd::torch::DiffractionWedgeBackwardRequest zero;
    zero.primal = primal;
    enable_all_gradients(zero);
    require_all_gradients_zero(
        rayd::torch::field_diffraction_wedge_backward(zero));
}

void test_negative_contracts() {
    auto bad_valid_dtype = wedge_request(true);
    bad_valid_dtype.valid = at::ones({1}, bad_valid_dtype.source.options());
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(bad_valid_dtype); },
        "wrong wedge valid dtype must fail");
    auto bad_valid_shape = wedge_request(true);
    bad_valid_shape.valid = at::ones({2}, bad_valid_shape.valid.options());
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(bad_valid_shape); },
        "wrong wedge valid shape must fail");

    auto bad_shape = wedge_request(true);
    bad_shape.target = at::zeros({1, 2}, bad_shape.target.options());
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(bad_shape); },
        "wrong wedge shape must fail");

    auto bad_dtype = wedge_request(true);
    bad_dtype.face0_eps_r = bad_dtype.face0_eps_r.to(at::kDouble);
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(bad_dtype); },
        "wrong wedge dtype must fail");

    auto bad_device = wedge_request(true);
    bad_device.source = bad_device.source.cpu();
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(bad_device); },
        "CPU wedge input must fail");

    auto noncontiguous = wedge_request(true);
    noncontiguous.source = at::ones(
        {1, 6}, noncontiguous.source.options()).slice(1, 0, 6, 2);
    require(!noncontiguous.source.is_contiguous(), "fixture must be noncontiguous");
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(noncontiguous); },
        "noncontiguous wedge input must fail");

    auto noncontiguous_optional = wedge_request(true);
    noncontiguous_optional.vertex_v0 = at::ones(
        {1, 6}, noncontiguous_optional.source.options()).slice(1, 0, 6, 2);
    require_throws(
        [&] {
            (void)rayd::torch::field_diffraction_wedge(noncontiguous_optional);
        },
        "noncontiguous wedge optional input must fail");

    auto zero_frequency = wedge_request(true);
    zero_frequency.frequency_hz = 0.0;
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(zero_frequency); },
        "zero wedge frequency must fail");
    auto negative_frequency = wedge_request(true);
    negative_frequency.frequency_hz = -1.0;
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge(negative_frequency); },
        "negative wedge frequency must fail");

    auto infinite_empty = empty_wedge_request();
    infinite_empty.frequency_hz = std::numeric_limits<double>::infinity();
    require_result_schema(
        rayd::torch::field_diffraction_wedge(infinite_empty), 0);

    rayd::torch::DiffractionWedgeBackwardRequest bad_grad;
    bad_grad.primal = wedge_request(true);
    bad_grad.grad_direction = at::ones(
        {1, 6}, bad_grad.primal.source.options()).slice(1, 0, 6, 2);
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge_backward(bad_grad); },
        "noncontiguous wedge cotangent must fail");

    rayd::torch::DiffractionWedgeJvpRequest bad_tangent;
    bad_tangent.primal = wedge_request(true);
    bad_tangent.tangent_face0_eps_r = at::ones(
        {1}, bad_tangent.primal.source.options().dtype(at::kDouble));
    require_throws(
        [&] { (void)rayd::torch::field_diffraction_wedge_jvp(bad_tangent); },
        "wrong wedge tangent dtype must fail");

    if (at::cuda::device_count() > 1) {
        auto mixed_device = wedge_request(true, 0);
        mixed_device.target = mixed_device.target.to(at::Device(at::kCUDA, 1));
        require_throws(
            [&] { (void)rayd::torch::field_diffraction_wedge(mixed_device); },
            "mixed-device wedge input must fail");
    }
}

void test_nondefault_stream_dependency() {
    const auto reference_primal = wedge_request(true);
    rayd::torch::DiffractionWedgeJvpRequest reference_jvp;
    reference_jvp.primal = reference_primal;
    reference_jvp.tangent_source = at::ones(
        {1, 3}, reference_primal.source.options());
    rayd::torch::DiffractionWedgeBackwardRequest reference_backward;
    reference_backward.primal = reference_primal;
    reference_backward.grad_field_vector = at::ones(
        {1, 3}, reference_primal.source.options().dtype(at::kComplexFloat));
    reference_backward.need_grad_material = true;
    const auto reference =
        rayd::torch::field_diffraction_wedge(reference_primal);
    const auto reference_tangent =
        rayd::torch::field_diffraction_wedge_jvp(reference_jvp);
    const auto reference_gradient =
        rayd::torch::field_diffraction_wedge_backward(reference_backward);
    at::cuda::getDefaultCUDAStream().synchronize();

    auto primal = wedge_request(true);
    primal.face0_eps_r.zero_();
    const auto producer = c10::cuda::getStreamFromPool(false, 0);
    const auto consumer = c10::cuda::getStreamFromPool(false, 0);
    require(producer.stream() != consumer.stream(), "stream fixtures must differ");
    {
        c10::cuda::CUDAStreamGuard guard(producer);
        primal.face0_eps_r.fill_(4.0F);
    }
    cudaEvent_t ready = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, producer.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(consumer.stream(), ready, 0));

    rayd::torch::DiffractionWedgeResult result;
    rayd::torch::DiffractionWedgeJvpResult tangent;
    rayd::torch::DiffractionWedgeBackwardResult gradient;
    {
        c10::cuda::CUDAStreamGuard guard(consumer);
        result = rayd::torch::field_diffraction_wedge(primal);
        rayd::torch::DiffractionWedgeJvpRequest jvp;
        jvp.primal = primal;
        jvp.tangent_source = at::ones({1, 3}, primal.source.options());
        tangent = rayd::torch::field_diffraction_wedge_jvp(jvp);
        rayd::torch::DiffractionWedgeBackwardRequest backward;
        backward.primal = primal;
        backward.grad_field_vector = at::ones(
            {1, 3}, primal.source.options().dtype(at::kComplexFloat));
        backward.need_grad_material = true;
        gradient = rayd::torch::field_diffraction_wedge_backward(backward);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == consumer.stream(),
            "wedge entries changed the caller's active CUDA stream");
    }
    consumer.synchronize();
    C10_CUDA_CHECK(cudaEventDestroy(ready));

    require(
        at::allclose(result.field_vector, reference.field_vector, 1.0e-5, 1.0e-8) &&
            at::allclose(result.direction, reference.direction, 1.0e-6, 1.0e-8),
        "wedge primal ignored stream dependency");
    require(
        at::allclose(
            tangent.tangent_field_vector,
            reference_tangent.tangent_field_vector,
            1.0e-5,
            1.0e-8) &&
            at::allclose(
                tangent.tangent_direction,
                reference_tangent.tangent_direction,
                1.0e-5,
                1.0e-8),
        "wedge JVP ignored stream dependency");
    require(
        gradient.grad_face0_eps_r.has_value() &&
            reference_gradient.grad_face0_eps_r.has_value() &&
            at::allclose(
                *gradient.grad_face0_eps_r,
                *reference_gradient.grad_face0_eps_r,
                1.0e-5,
                1.0e-8),
        "wedge backward ignored stream dependency");
}

void test_invalid_row_short_circuits_poison() {
    auto primal = wedge_request(true);
    primal.valid.zero_();
    primal.source.fill_(std::numeric_limits<float>::quiet_NaN());
    primal.edge_direction.fill_(std::numeric_limits<float>::quiet_NaN());
    primal.face0_eps_r.fill_(std::numeric_limits<float>::quiet_NaN());
    primal.vertex_v0->fill_(std::numeric_limits<float>::quiet_NaN());
    const auto forward = rayd::torch::field_diffraction_wedge(primal);
    require(
        at::count_nonzero(forward.field_vector).item<int64_t>() == 0 &&
            at::count_nonzero(forward.direction).item<int64_t>() == 0,
        "invalid poisoned wedge primal outputs must be exactly zero");

    rayd::torch::DiffractionWedgeJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_frequency = 1.0;
    const auto tangent = rayd::torch::field_diffraction_wedge_jvp(jvp);
    require(
        at::count_nonzero(tangent.tangent_field_vector).item<int64_t>() == 0 &&
            at::count_nonzero(tangent.tangent_direction).item<int64_t>() == 0,
        "invalid poisoned wedge JVP outputs must be exactly zero");

    rayd::torch::DiffractionWedgeBackwardRequest backward;
    backward.primal = primal;
    backward.grad_field_vector = at::ones(
        {1, 3}, primal.source.options().dtype(at::kComplexFloat));
    backward.grad_direction = at::ones({1, 3}, primal.source.options());
    enable_all_gradients(backward);
    require_all_gradients_zero(
        rayd::torch::field_diffraction_wedge_backward(backward));
}

} // namespace

int main() {
    try {
        require(at::cuda::is_available(), "CUDA is required for wedge tests");
        std::cout << "[RUN] test_empty_schema_and_optional_bundle\n";
        test_empty_schema_and_optional_bundle();
        std::cout << "[RUN] test_primal_ad_duality_need_and_zero\n";
        test_primal_ad_duality_need_and_zero();
        std::cout << "[RUN] test_negative_contracts\n";
        test_negative_contracts();
        std::cout << "[RUN] test_nondefault_stream_dependency\n";
        test_nondefault_stream_dependency();
        std::cout << "[RUN] test_invalid_row_short_circuits_poison\n";
        test_invalid_row_short_circuits_poison();
        std::cout << "rayd::torch pure-wedge direct contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rayd::torch pure-wedge direct contract failure: "
                  << error.what() << '\n';
        return 1;
    }
}
