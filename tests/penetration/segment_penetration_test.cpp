#include <rayd/integration/torch.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

at::Tensor channel_deterministic_normalize_vec3_oracle(
    const at::Tensor &values,
    float epsilon);
at::Tensor channel_segment_restart_epsilon_oracle(
    const at::Tensor &positions,
    float scene_diagonal,
    bool use_l2_norm);
at::Tensor channel_segment_restart_point_oracle(
    const at::Tensor &positions,
    const at::Tensor &directions,
    const at::Tensor &epsilon);

namespace {

static_assert(rayd::torch::kIntegrationApiVersion == 7);

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition)
        fail(message);
}

template <typename Fn>
void require_throws(Fn &&function, const std::string &message) {
    try {
        std::forward<Fn>(function)();
    } catch (const std::exception &) {
        return;
    }
    fail(message);
}

struct Fixture {
    at::Tensor vertices;
    at::Tensor faces;
    at::Tensor uv;
    at::Tensor face_uv;
    at::Tensor transform_left;
    at::Tensor transform_right;
};

Fixture three_parallel_walls() {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    const auto integers = floats.dtype(at::kInt);
    return {
        at::tensor(
            {
                -8.0f, -8.0f, 0.0f, 8.0f, -8.0f, 0.0f, 0.0f, 8.0f, 0.0f,
                -8.0f, -8.0f, 1.0f, 8.0f, -8.0f, 1.0f, 0.0f, 8.0f, 1.0f,
                -8.0f, -8.0f, 2.0f, 8.0f, -8.0f, 2.0f, 0.0f, 8.0f, 2.0f,
            },
            floats)
            .reshape({9, 3}),
        at::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8}, integers).reshape({3, 3}),
        at::empty({0, 2}, floats),
        at::empty({0, 3}, integers),
        at::empty({0, 4}, floats),
        at::empty({0, 4}, floats),
    };
}

Fixture non_axis_walls() {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    const auto integers = floats.dtype(at::kInt);
    return {
        at::tensor(
            {
                35.125f, -32.875f, 18.17625f,
                48.625f, -32.125f, 23.00625f,
                39.375f, -19.875f, 16.88875f,
                34.385f, -32.435f, 20.17625f,
                47.885f, -31.685f, 25.00625f,
                38.635f, -19.435f, 18.88875f,
            },
            floats)
            .reshape({6, 3}),
        at::tensor({0, 1, 2, 3, 4, 5}, integers).reshape({2, 3}),
        at::empty({0, 2}, floats),
        at::empty({0, 3}, integers),
        at::empty({0, 4}, floats),
        at::empty({0, 4}, floats),
    };
}

rayd::torch::MeshInput mesh_input(const Fixture &fixture) {
    return {
        fixture.vertices,
        fixture.faces,
        fixture.uv,
        fixture.face_uv,
        fixture.transform_left,
        fixture.transform_right,
        false,
        true,
        false,
    };
}

at::Tensor failure_state() {
    return at::zeros(
        {1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
}

rayd::torch::SegmentPenetrationRequest request(
    const rayd::torch::SceneResource &scene,
    at::Tensor origins,
    at::Tensor targets,
    std::optional<at::Tensor> active,
    bool input_active_any,
    int64_t capacity,
    rayd::torch::SegmentPenetrationPolicy policy,
    at::Tensor failure,
    int failure_bit = 4,
    double scene_diagonal = 0.0) {
    return {
        scene,
        std::move(origins),
        std::move(targets),
        std::move(active),
        input_active_any,
        capacity,
        policy,
        scene_diagonal,
        std::move(failure),
        failure_bit,
    };
}

void require_result_contract(
    const rayd::torch::SegmentPenetrationResult &result,
    int64_t segments,
    int64_t capacity) {
    require(result.valid.sizes().equals({segments, capacity}), "valid shape");
    require(result.num_hits.sizes().equals({segments}), "num_hits shape");
    require(result.reached_target.sizes().equals({segments}), "reached_target shape");
    require(result.overflow.sizes().equals({segments}), "overflow shape");
    require(result.distance.sizes().equals({segments}), "distance shape");
    require(result.direction.sizes().equals({segments, 3}), "direction shape");
    require(result.t.sizes().equals({segments, capacity}), "t shape");
    for (const at::Tensor *tensor : {
             &result.position,
             &result.normal,
             &result.geometric_normal}) {
        require(tensor->sizes().equals({segments, capacity, 3}), "vector payload shape");
    }
    require(
        result.global_primitive_id.sizes().equals({segments, capacity}),
        "primitive shape");
    for (const at::Tensor *tensor : {
             &result.valid,
             &result.num_hits,
             &result.reached_target,
             &result.overflow,
             &result.distance,
             &result.direction,
             &result.t,
             &result.position,
             &result.normal,
             &result.geometric_normal,
             &result.global_primitive_id}) {
        require(tensor->is_cuda() && tensor->is_contiguous(), "result residency/contiguity");
    }
}

void require_result_exact(
    const rayd::torch::SegmentPenetrationResult &a,
    const rayd::torch::SegmentPenetrationResult &b) {
    for (const auto &pair : {
             std::pair<const at::Tensor *, const at::Tensor *>{&a.valid, &b.valid},
             {&a.num_hits, &b.num_hits},
             {&a.reached_target, &b.reached_target},
             {&a.overflow, &b.overflow},
             {&a.distance, &b.distance},
             {&a.direction, &b.direction},
             {&a.t, &b.t},
             {&a.position, &b.position},
             {&a.normal, &b.normal},
             {&a.geometric_normal, &b.geometric_normal},
             {&a.global_primitive_id, &b.global_primitive_id}}) {
        require(at::equal(*pair.first, *pair.second), "plain/tape primal mismatch");
    }
}

void require_inert(
    const rayd::torch::SegmentPenetrationResult &result,
    bool allow_overflow_diagnostic) {
    require(!result.valid.any().item<bool>(), "failed valid must be inert");
    require(at::equal(result.num_hits, at::zeros_like(result.num_hits)), "failed count");
    require(!result.reached_target.any().item<bool>(), "failed reached_target");
    if (!allow_overflow_diagnostic)
        require(!result.overflow.any().item<bool>(), "unexpected overflow diagnostic");
    require(at::equal(result.distance, at::zeros_like(result.distance)), "failed distance");
    require(at::equal(result.direction, at::zeros_like(result.direction)), "failed direction");
    require(at::equal(result.t, at::full_like(result.t, -1.0f)), "failed t");
    require(at::equal(result.position, at::zeros_like(result.position)), "failed position");
    require(at::equal(result.normal, at::zeros_like(result.normal)), "failed normal");
    require(
        at::equal(result.geometric_normal, at::zeros_like(result.geometric_normal)),
        "failed geometric normal");
    require(
        at::equal(
            result.global_primitive_id,
            at::full_like(result.global_primitive_id, -1)),
        "failed primitive");
}

void test_forward_contracts(const rayd::torch::SceneResource &scene) {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    const auto booleans = floats.dtype(at::kBool);
    auto origins = at::tensor({0.0f, 0.0f, -1.0f, 7.0f, 7.0f, -1.0f}, floats)
                       .reshape({2, 3});
    auto targets = at::tensor({0.0f, 0.0f, 1.5f, 7.0f, 7.0f, 1.5f}, floats)
                       .reshape({2, 3});
    auto failure = failure_state();
    auto full = request(
        scene,
        origins,
        targets,
        std::nullopt,
        true,
        2,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure);
    const auto plain = rayd::torch::segment_penetration_forward(full);
    const auto tape = rayd::torch::segment_penetration_forward_tape(full);
    require_result_contract(plain, 2, 2);
    require_result_exact(plain, tape.result);
    require(failure.item<int>() == 0, "successful failure state changed");
    require(plain.num_hits[0].item<int>() == 2, "exact-D hit count");
    require(plain.reached_target[0].item<bool>(), "exact-D clear tail");
    require(plain.num_hits[1].item<int>() == 0, "clear count");
    require(plain.reached_target[1].item<bool>(), "clear reached_target");
    require(
        at::equal(plain.global_primitive_id[0], at::tensor({0, 1}, plain.global_primitive_id.options())),
        "ordered global primitive ids");
    require(
        tape.tape_barycentric.sizes().equals({2, 2, 2}) &&
            tape.tape_restart_epsilon.sizes().equals({2, 2}),
        "tape shape");

    rayd::torch::RayBatch first_ray{
        origins.slice(0, 0, 1),
        plain.direction.slice(0, 0, 1),
        plain.distance.slice(0, 0, 1),
        std::nullopt,
    };
    const auto direct_hit = rayd::torch::intersect_forward(scene, first_ray, 7);
    require(at::equal(plain.t[0][0], direct_hit.t[0]), "first t vs intersect");
    require(at::equal(plain.position[0][0], direct_hit.p[0]), "first position vs intersect");
    require(
        at::equal(plain.geometric_normal[0][0], direct_hit.geo_n[0]),
        "geometric normal vs intersect");

    auto empty_origins = at::empty({0, 3}, floats);
    auto empty_targets = at::empty({0, 3}, floats);
    auto empty_active = at::empty({0}, booleans);
    auto empty = request(
        scene,
        empty_origins,
        empty_targets,
        empty_active,
        false,
        0,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    require_result_contract(rayd::torch::segment_penetration_forward(empty), 0, 0);

    auto degenerate_points = at::tensor({1.0f, 1.0f, 0.5f}, floats).reshape({1, 3});
    auto degenerate = request(
        scene,
        degenerate_points,
        degenerate_points.clone(),
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    const auto degenerate_result = rayd::torch::segment_penetration_forward(degenerate);
    require(degenerate_result.reached_target[0].item<bool>(), "active degenerate reached_target");

    auto zero_capacity_clear = request(
        scene,
        origins.slice(0, 1, 2),
        targets.slice(0, 1, 2),
        std::nullopt,
        true,
        0,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    const auto zero_capacity_clear_result =
        rayd::torch::segment_penetration_forward(zero_capacity_clear);
    require(
        zero_capacity_clear_result.reached_target[0].item<bool>(),
        "D=0 clear segment did not reach target");

    auto zero_capacity_failure = failure_state();
    auto zero_capacity_hit = request(
        scene,
        origins.slice(0, 0, 1),
        targets.slice(0, 0, 1),
        std::nullopt,
        true,
        0,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        zero_capacity_failure,
        32);
    const auto zero_capacity_hit_result =
        rayd::torch::segment_penetration_forward(zero_capacity_hit);
    require(zero_capacity_failure.item<int>() == 32, "D=0 first hit did not overflow");
    require_inert(zero_capacity_hit_result, true);
}

void test_policy_differences(const rayd::torch::SceneResource &scene) {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto origin = at::tensor({2.0f, 2.0f, -1.0f}, floats).reshape({1, 3});
    auto target = at::tensor({2.0f, 2.0f, 0.5f}, floats).reshape({1, 3});
    auto enumerated = request(
        scene,
        origin,
        target,
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    auto monte_carlo = request(
        scene,
        origin,
        target,
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset,
        failure_state());
    const auto enum_tape = rayd::torch::segment_penetration_forward_tape(enumerated);
    const auto mc_tape = rayd::torch::segment_penetration_forward_tape(monte_carlo);
    const float expected_l2 = std::sqrt(8.0f) * 1.0e-6f;
    const float expected_linf = 2.0e-6f;
    require(enum_tape.tape_restart_epsilon[0][0].item<float>() == expected_l2, "L2 epsilon");
    require(mc_tape.tape_restart_epsilon[0][0].item<float>() == expected_linf, "Linf epsilon");
    require(
        at::equal(mc_tape.result.normal[0][0], mc_tape.result.geometric_normal[0][0]),
        "Monte Carlo shading normal contract");

    auto endpoint_origin = at::tensor({0.0f, 0.0f, -0.5f}, floats).reshape({1, 3});
    auto endpoint_target = at::tensor({0.0f, 0.0f, 1.0e-6f}, floats).reshape({1, 3});
    auto inclusive = request(
        scene,
        endpoint_origin,
        endpoint_target,
        std::nullopt,
        true,
        2,
        rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset,
        failure_state());
    const auto inclusive_result = rayd::torch::segment_penetration_forward(inclusive);
    require(inclusive_result.num_hits[0].item<int>() >= 1, "inclusive inset endpoint hit");

    auto strict = request(
        scene,
        endpoint_origin,
        at::tensor({0.0f, 0.0f, 0.0f}, floats).reshape({1, 3}),
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    const auto strict_result = rayd::torch::segment_penetration_forward(strict);
    require(
        strict_result.num_hits[0].item<int>() == 0 &&
            strict_result.reached_target[0].item<bool>(),
        "strict full-distance endpoint rejected");

    auto zero_inset = request(
        scene,
        at::tensor({0.0f, 0.0f, 0.0f}, floats).reshape({1, 3}),
        at::tensor({0.0f, 0.0f, 0.0000005f}, floats).reshape({1, 3}),
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset,
        failure_state());
    const auto zero_inset_result = rayd::torch::segment_penetration_forward(zero_inset);
    require(zero_inset_result.reached_target[0].item<bool>(), "zero inset reached_target");
}

void test_non_axis_exact_policy_parity() {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    Fixture fixture = non_axis_walls();
    auto scene = rayd::torch::create_scene({mesh_input(fixture)});
    auto origin = at::tensor({42.867f, -28.599f, 15.35887f}, floats).reshape({1, 3});
    auto target = at::tensor({39.537f, -26.619f, 24.35887f}, floats).reshape({1, 3});
    const auto delta = target - origin;
    constexpr float scene_diagonal = 0.0f;

    auto enumerated = request(
        scene,
        origin,
        target,
        std::nullopt,
        true,
        2,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state(),
        4,
        scene_diagonal);
    auto monte_carlo = request(
        scene,
        origin,
        target,
        std::nullopt,
        true,
        2,
        rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset,
        failure_state(),
        4,
        scene_diagonal);
    const auto enumerated_tape = rayd::torch::segment_penetration_forward_tape(enumerated);
    const auto monte_carlo_tape = rayd::torch::segment_penetration_forward_tape(monte_carlo);
    require(enumerated_tape.result.num_hits[0].item<int>() == 2, "non-axis enumerated hits");
    require(monte_carlo_tape.result.num_hits[0].item<int>() == 2, "non-axis Monte Carlo hits");

    const auto expected_enumerated_direction =
        channel_deterministic_normalize_vec3_oracle(delta, 1.0e-9f);
    const auto expected_monte_carlo_direction =
        channel_deterministic_normalize_vec3_oracle(delta, 1.0e-6f);
    require(
        at::equal(enumerated_tape.result.direction, expected_enumerated_direction),
        "non-axis enumerated direction bits");
    require(
        at::equal(monte_carlo_tape.result.direction, expected_monte_carlo_direction),
        "non-axis Monte Carlo direction bits");

    rayd::torch::RayBatch direct_ray{
        origin,
        enumerated_tape.result.direction,
        enumerated_tape.result.distance,
        std::nullopt,
    };
    const auto direct_hit = rayd::torch::intersect_forward(scene, direct_ray, 7);
    require(
        at::equal(enumerated_tape.result.t[0][0], direct_hit.t[0]),
        "non-axis hit t vs typed intersect");
    require(
        at::equal(enumerated_tape.result.position[0][0], direct_hit.p[0]),
        "non-axis hit position vs typed intersect");
    require(
        at::equal(monte_carlo_tape.result.t[0][0], direct_hit.t[0]),
        "non-axis Monte Carlo hit t vs typed intersect");
    require(
        at::equal(monte_carlo_tape.result.position[0][0], direct_hit.p[0]),
        "non-axis Monte Carlo hit position vs typed intersect");
    require(
        at::equal(enumerated_tape.result.geometric_normal[0][0], direct_hit.geo_n[0]),
        "non-axis geometric normal bits vs typed intersect");
    require(
        at::equal(monte_carlo_tape.result.geometric_normal[0][0], direct_hit.geo_n[0]),
        "non-axis Monte Carlo geometric normal bits vs typed intersect");
    require(
        at::equal(monte_carlo_tape.result.normal[0][0], direct_hit.n[0]),
        "non-axis Monte Carlo shading normal bits vs typed intersect");

    const auto expected_enumerated_normal =
        channel_deterministic_normalize_vec3_oracle(direct_hit.geo_n, 1.0e-9f);
    require(
        at::equal(enumerated_tape.result.normal[0][0], expected_enumerated_normal[0]),
        "non-axis enumerated second-normalization bits");
    const auto expected_l2_restart = channel_segment_restart_epsilon_oracle(
        enumerated_tape.result.position.select(1, 0), scene_diagonal, true);
    const auto expected_linf_restart = channel_segment_restart_epsilon_oracle(
        monte_carlo_tape.result.position.select(1, 0), scene_diagonal, false);
    require(
        at::equal(enumerated_tape.tape_restart_epsilon.select(1, 0), expected_l2_restart),
        "non-axis L2 restart epsilon bits");
    require(
        at::equal(monte_carlo_tape.tape_restart_epsilon.select(1, 0), expected_linf_restart),
        "non-axis Linf restart epsilon bits");
    require(
        !at::equal(expected_l2_restart, expected_linf_restart),
        "non-axis fixture does not distinguish restart norms");

    const auto enumerated_restart_origin = channel_segment_restart_point_oracle(
        enumerated_tape.result.position.select(1, 0),
        enumerated_tape.result.direction,
        enumerated_tape.tape_restart_epsilon.select(1, 0));
    rayd::torch::RayBatch enumerated_second_ray{
        enumerated_restart_origin,
        enumerated_tape.result.direction,
        enumerated_tape.result.distance,
        std::nullopt,
    };
    const auto enumerated_second =
        rayd::torch::intersect_forward(scene, enumerated_second_ray, 7);
    require(
        at::equal(enumerated_tape.result.t[0][1], enumerated_second.t[0]),
        "non-axis post-restart enumerated t bits");
    require(
        at::equal(enumerated_tape.result.position[0][1], enumerated_second.p[0]),
        "non-axis post-restart enumerated position bits");
    require(
        at::equal(enumerated_tape.result.geometric_normal[0][1], enumerated_second.geo_n[0]),
        "non-axis post-restart enumerated geometric normal bits");

    const auto monte_carlo_restart_origin = channel_segment_restart_point_oracle(
        monte_carlo_tape.result.position.select(1, 0),
        monte_carlo_tape.result.direction,
        monte_carlo_tape.tape_restart_epsilon.select(1, 0));
    rayd::torch::RayBatch monte_carlo_second_ray{
        monte_carlo_restart_origin,
        monte_carlo_tape.result.direction,
        monte_carlo_tape.result.distance,
        std::nullopt,
    };
    const auto monte_carlo_second =
        rayd::torch::intersect_forward(scene, monte_carlo_second_ray, 7);
    require(
        at::equal(monte_carlo_tape.result.t[0][1], monte_carlo_second.t[0]),
        "non-axis post-restart Monte Carlo t bits");
    require(
        at::equal(monte_carlo_tape.result.position[0][1], monte_carlo_second.p[0]),
        "non-axis post-restart Monte Carlo position bits");
    require(
        at::equal(monte_carlo_tape.result.normal[0][1], monte_carlo_second.n[0]),
        "non-axis post-restart Monte Carlo shading normal bits");
}

void test_failure_transaction(const rayd::torch::SceneResource &scene) {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    const auto booleans = floats.dtype(at::kBool);
    auto origins = at::tensor({7.0f, 7.0f, -1.0f, 0.0f, 0.0f, -1.0f}, floats)
                       .reshape({2, 3});
    auto targets = at::tensor({7.0f, 7.0f, 3.0f, 0.0f, 0.0f, 3.0f}, floats)
                       .reshape({2, 3});
    auto failure = failure_state();
    auto overflow = request(
        scene,
        origins,
        targets,
        std::nullopt,
        true,
        2,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure,
        8);
    const auto failed = rayd::torch::segment_penetration_forward_tape(overflow);
    require(failure.item<int>() == 8, "overflow failure bit");
    require(!failed.result.overflow[0].item<bool>() && failed.result.overflow[1].item<bool>(), "mixed overflow diagnostic");
    require_inert(failed.result, true);
    require(
        at::equal(failed.tape_primitive_id, at::full_like(failed.tape_primitive_id, -1)),
        "failed primitive tape");
    require(
        at::equal(failed.tape_barycentric, at::zeros_like(failed.tape_barycentric)),
        "failed barycentric tape");

    auto false_mask = at::zeros({1}, booleans);
    auto all_inactive = request(
        scene,
        at::full({1, 3}, std::numeric_limits<float>::quiet_NaN(), floats),
        at::full({1, 3}, std::numeric_limits<float>::infinity(), floats),
        false_mask,
        false,
        0,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    const auto inactive_tape = rayd::torch::segment_penetration_forward_tape(all_inactive);
    require_inert(inactive_tape.result, false);
    rayd::torch::SegmentPenetrationJvpRequest inactive_jvp{
        all_inactive,
        inactive_tape,
        std::nullopt,
        at::ones_like(all_inactive.origins),
        at::ones_like(all_inactive.targets),
    };
    const auto inactive_tangents = rayd::torch::segment_penetration_jvp(inactive_jvp);
    require(
        at::equal(
            inactive_tangents.tangent_direction,
            at::zeros_like(inactive_tangents.tangent_direction)),
        "input-inactive JVP was nonzero");
    rayd::torch::SegmentPenetrationBackwardRequest inactive_backward{
        all_inactive,
        inactive_tape,
        at::ones_like(inactive_tape.result.distance),
        at::ones_like(inactive_tape.result.direction),
        at::ones_like(inactive_tape.result.t),
        at::ones_like(inactive_tape.result.position),
        at::ones_like(inactive_tape.result.normal),
        at::ones_like(inactive_tape.result.geometric_normal),
        true,
        true,
        true,
    };
    const auto inactive_gradients =
        rayd::torch::segment_penetration_backward(inactive_backward);
    require(
        at::equal(
            inactive_gradients.grad_origins,
            at::zeros_like(inactive_gradients.grad_origins)),
        "input-inactive VJP was nonzero");

    const auto check_nonfinite = [&](at::Tensor test_origin,
                                     at::Tensor test_target,
                                     const char *label) {
        auto nonfinite_failure = failure_state();
        auto nonfinite = request(
            scene,
            std::move(test_origin),
            std::move(test_target),
            std::nullopt,
            true,
            1,
            rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
            nonfinite_failure,
            64);
        const auto failed_nonfinite =
            rayd::torch::segment_penetration_forward_tape(nonfinite);
        require(
            nonfinite_failure.item<int>() == 64,
            std::string("nonfinite input did not fail device transaction: ") + label);
        require_inert(failed_nonfinite.result, false);
        require(
            at::equal(
                failed_nonfinite.tape_primitive_id,
                at::full_like(failed_nonfinite.tape_primitive_id, -1)) &&
                at::equal(
                    failed_nonfinite.tape_barycentric,
                    at::zeros_like(failed_nonfinite.tape_barycentric)) &&
                at::equal(
                    failed_nonfinite.tape_restart_epsilon,
                    at::zeros_like(failed_nonfinite.tape_restart_epsilon)) &&
                at::equal(
                    failed_nonfinite.tape_restart_branch,
                    at::zeros_like(failed_nonfinite.tape_restart_branch)) &&
                at::equal(
                    failed_nonfinite.tape_restart_tie_mask,
                    at::zeros_like(failed_nonfinite.tape_restart_tie_mask)) &&
                at::equal(
                    failed_nonfinite.tape_direction_denominator_branch,
                    at::zeros_like(failed_nonfinite.tape_direction_denominator_branch)),
            std::string("nonfinite tape was not inert: ") + label);
    };
    const auto finite_origin = at::tensor({0.0f, 0.0f, -1.0f}, floats).reshape({1, 3});
    const auto finite_target = at::tensor({0.0f, 0.0f, 3.0f}, floats).reshape({1, 3});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float positive_infinity = std::numeric_limits<float>::infinity();
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    check_nonfinite(
        at::tensor({nan, 0.0f, -1.0f}, floats).reshape({1, 3}),
        finite_target,
        "origin NaN");
    check_nonfinite(
        finite_origin,
        at::tensor({nan, 0.0f, 3.0f}, floats).reshape({1, 3}),
        "target NaN");
    check_nonfinite(
        at::tensor({positive_infinity, 0.0f, -1.0f}, floats).reshape({1, 3}),
        finite_target,
        "origin +Inf");
    check_nonfinite(
        at::tensor({negative_infinity, 0.0f, -1.0f}, floats).reshape({1, 3}),
        finite_target,
        "origin -Inf");
    check_nonfinite(
        finite_origin,
        at::tensor({positive_infinity, 0.0f, 3.0f}, floats).reshape({1, 3}),
        "target +Inf");
    check_nonfinite(
        finite_origin,
        at::tensor({negative_infinity, 0.0f, 3.0f}, floats).reshape({1, 3}),
        "target -Inf");
    check_nonfinite(
        at::tensor({-3.0e38f, 0.0f, 0.0f}, floats).reshape({1, 3}),
        at::tensor({3.0e38f, 0.0f, 0.0f}, floats).reshape({1, 3}),
        "finite subtraction overflow");
    check_nonfinite(
        at::zeros({1, 3}, floats),
        at::tensor({1.0e20f, 1.0e20f, 0.0f}, floats).reshape({1, 3}),
        "finite squared-norm overflow");

    auto contradiction_failure = failure_state();
    auto contradiction = request(
        scene,
        origins.slice(0, 0, 1),
        targets.slice(0, 0, 1),
        at::ones({1}, booleans),
        false,
        0,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        contradiction_failure,
        16);
    const auto contradiction_result = rayd::torch::segment_penetration_forward(contradiction);
    require(contradiction_failure.item<int>() == 16, "mask contradiction failure bit");
    require_inert(contradiction_result, false);

    auto poisoned_failure = at::ones(
        {1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    auto poisoned = request(
        scene,
        at::full({1, 3}, std::numeric_limits<float>::quiet_NaN(), floats),
        at::full({1, 3}, std::numeric_limits<float>::quiet_NaN(), floats),
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        poisoned_failure);
    require_inert(rayd::torch::segment_penetration_forward(poisoned), false);
}

void test_ad_and_stream(const rayd::torch::SceneResource &scene, const Fixture &fixture) {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto origins = at::tensor({0.25f, 0.25f, -1.0f}, floats).reshape({1, 3});
    auto targets = at::tensor({0.35f, 0.20f, 1.5f}, floats).reshape({1, 3});
    auto primal = request(
        scene,
        origins,
        targets,
        std::nullopt,
        true,
        2,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    rayd::torch::SegmentPenetrationTapeResult tape = [&]() {
        c10::cuda::CUDAStreamGuard guard(stream);
        auto value = rayd::torch::segment_penetration_forward_tape(primal);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
            "penetration changed current stream");
        return value;
    }();
    stream.synchronize();

    auto tangent_vertices = at::full_like(fixture.vertices, 0.001f);
    auto tangent_origins = at::tensor({0.01f, -0.02f, 0.03f}, floats).reshape({1, 3});
    auto tangent_targets = at::tensor({-0.03f, 0.01f, 0.02f}, floats).reshape({1, 3});
    rayd::torch::SegmentPenetrationJvpRequest jvp_request{
        primal,
        tape,
        tangent_vertices,
        tangent_origins,
        tangent_targets,
    };
    const auto tangents = rayd::torch::segment_penetration_jvp(jvp_request);
    auto grad_distance = at::full_like(tape.result.distance, 0.3f);
    auto grad_direction = at::full_like(tape.result.direction, -0.2f);
    auto grad_t = at::full_like(tape.result.t, 0.4f);
    auto grad_position = at::full_like(tape.result.position, 0.1f);
    auto grad_normal = at::full_like(tape.result.normal, -0.15f);
    auto grad_geo = at::full_like(tape.result.geometric_normal, 0.05f);
    rayd::torch::SegmentPenetrationBackwardRequest backward_request{
        primal,
        tape,
        grad_distance,
        grad_direction,
        grad_t,
        grad_position,
        grad_normal,
        grad_geo,
        true,
        true,
        true,
    };
    const auto gradients = rayd::torch::segment_penetration_backward(backward_request);
    const auto lhs =
        (tangents.tangent_distance * grad_distance).sum() +
        (tangents.tangent_direction * grad_direction).sum() +
        (tangents.tangent_t * grad_t).sum() +
        (tangents.tangent_position * grad_position).sum() +
        (tangents.tangent_normal * grad_normal).sum() +
        (tangents.tangent_geometric_normal * grad_geo).sum();
    const auto rhs =
        (gradients.grad_vertices * tangent_vertices).sum() +
        (gradients.grad_origins * tangent_origins).sum() +
        (gradients.grad_targets * tangent_targets).sum();
    require(
        at::allclose(lhs, rhs, 2.0e-4, 2.0e-5),
        "fixed-winner JVP/VJP duality");

    rayd::torch::SegmentPenetrationJvpRequest zero_jvp{primal, tape};
    const auto zero = rayd::torch::segment_penetration_jvp(zero_jvp);
    require(at::equal(zero.tangent_t, at::zeros_like(zero.tangent_t)), "optional zero JVP");
}

void test_validation(const rayd::torch::SceneResource &scene) {
    const auto floats = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto origin = at::zeros({1, 3}, floats);
    auto target = at::ones({1, 3}, floats);
    auto bad_bit = request(
        scene,
        origin,
        target,
        std::nullopt,
        true,
        1,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state(),
        3);
    require_throws(
        [&]() { (void)rayd::torch::segment_penetration_forward(bad_bit); },
        "multi-bit failure_bit accepted");
    auto missing_mask = request(
        scene,
        origin,
        target,
        std::nullopt,
        false,
        1,
        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
        failure_state());
    require_throws(
        [&]() { (void)rayd::torch::segment_penetration_forward(missing_mask); },
        "all-inactive request without mask accepted");
    auto invalid_policy = request(
        scene,
        origin,
        target,
        std::nullopt,
        true,
        1,
        static_cast<rayd::torch::SegmentPenetrationPolicy>(77),
        failure_state());
    require_throws(
        [&]() { (void)rayd::torch::segment_penetration_forward(invalid_policy); },
        "invalid policy accepted");
}

} // namespace

int main() {
    try {
        if (!at::cuda::is_available())
            return 0;
        Fixture fixture = three_parallel_walls();
        auto scene = rayd::torch::create_scene({mesh_input(fixture)});
        test_forward_contracts(scene);
        test_policy_differences(scene);
        test_non_axis_exact_policy_parity();
        test_failure_transaction(scene);
        test_ad_and_stream(scene, fixture);
        test_validation(scene);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "segment penetration direct test failed: " << error.what() << '\n';
        return 1;
    }
}
