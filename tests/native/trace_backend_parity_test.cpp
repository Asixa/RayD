// Copyright Xingyu Chen.
// Checks that every scene operation gives the same results on the OptiX and the pure-CUDA trace backends.

#include <rayd/integration.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// CTest reads this exit code as a skipped test (SKIP_RETURN_CODE).
constexpr int kSkipReturnCode = 77;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

void require_equal(const at::Tensor& optix, const at::Tensor& cuda, const std::string& name) {
    require(optix.sizes() == cuda.sizes(), name + ": shape differs");
    require(optix.scalar_type() == cuda.scalar_type(), name + ": dtype differs");
    if (!at::equal(optix, cuda)) {
        const int64_t mismatches = optix.ne(cuda).sum().item<int64_t>();
        fail(name + ": " + std::to_string(mismatches) + " of " + std::to_string(optix.numel()) + " entries differ");
    }
}

// Floating-point outputs come from two different triangle tests (OptiX hardware and the CUDA watertight test), so
// they agree to float32 rounding of the hit distance rather than bit for bit.
void require_close(const at::Tensor& optix, const at::Tensor& cuda, double rtol, double atol, const std::string& name) {
    require(optix.sizes() == cuda.sizes(), name + ": shape differs");
    require(optix.scalar_type() == cuda.scalar_type(), name + ": dtype differs");
    if (!at::allclose(optix, cuda, rtol, atol)) {
        const double worst = (optix - cuda).abs().max().item<double>();
        fail(name + ": max abs difference " + std::to_string(worst));
    }
}

at::TensorOptions floats() {
    return at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
}

at::TensorOptions ints() {
    return at::TensorOptions().device(at::kCUDA).dtype(at::kInt);
}

at::TensorOptions bools() {
    return at::TensorOptions().device(at::kCUDA).dtype(at::kBool);
}

at::Tensor uniform(std::mt19937& rng, int64_t rows, int64_t cols, float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(static_cast<size_t>(rows * cols));
    for (float& value : values)
        value = dist(rng);
    return at::tensor(values, at::TensorOptions().dtype(at::kFloat)).reshape({rows, cols}).to(floats());
}

rayd::torch::MeshInput mesh_input(at::Tensor vertices, at::Tensor faces) {
    return {
        std::move(vertices),
        std::move(faces),
        at::empty({0, 2}, floats()),
        at::empty({0, 3}, ints()),
        at::empty({0, 4}, floats()),
        at::empty({0, 4}, floats()),
        false,
        true,
        false,
    };
}

// Two meshes so that shape ids and per-mesh face offsets take part: scattered random triangles inside a cube, and
// five tilted walls crossing it.
std::vector<rayd::torch::MeshInput> scene_meshes(std::mt19937& rng, int scattered_triangles) {
    // Triangle size shrinks with the count so that a segment crosses a similar number of them at every scale.
    const float extent = 0.6f * std::sqrt(2048.0f / static_cast<float>(scattered_triangles));
    const at::Tensor centers = uniform(rng, scattered_triangles, 3, -4.0f, 4.0f);
    const at::Tensor offsets = uniform(rng, scattered_triangles * 3, 3, -extent, extent);
    const at::Tensor scattered_vertices = centers.repeat_interleave(3, 0) + offsets;
    const at::Tensor scattered_faces =
        at::arange(scattered_triangles * 3, at::TensorOptions().dtype(at::kInt)).reshape({-1, 3}).to(ints());

    std::vector<float> wall_vertices;
    std::vector<int> wall_faces;
    for (int wall = 0; wall < 5; ++wall) {
        const float x = -3.0f + 1.5f * static_cast<float>(wall);
        const float tilt = 0.15f * static_cast<float>(wall - 2);
        const int base = static_cast<int>(wall_vertices.size() / 3);
        for (const auto& corner : {std::pair<float, float>{-5.0f, -5.0f}, {5.0f, -5.0f}, {5.0f, 5.0f}, {-5.0f, 5.0f}}) {
            wall_vertices.push_back(x + tilt * corner.first);
            wall_vertices.push_back(corner.first);
            wall_vertices.push_back(corner.second);
        }
        for (int index : {0, 1, 2, 0, 2, 3})
            wall_faces.push_back(base + index);
    }
    std::vector<rayd::torch::MeshInput> meshes;
    meshes.push_back(mesh_input(scattered_vertices.contiguous(), scattered_faces.contiguous()));
    meshes.push_back(
        mesh_input(at::tensor(wall_vertices, at::TensorOptions().dtype(at::kFloat)).reshape({-1, 3}).to(floats()),
                   at::tensor(wall_faces, at::TensorOptions().dtype(at::kInt)).reshape({-1, 3}).to(ints())));
    return meshes;
}

struct ScenePair {
    rayd::torch::SceneResource optix;
    rayd::torch::SceneResource cuda;
};

ScenePair make_scenes(int scattered_triangles, unsigned seed) {
    std::mt19937 optix_rng(seed);
    std::mt19937 cuda_rng(seed);
    return {
        rayd::torch::create_scene(scene_meshes(optix_rng, scattered_triangles),
                                  {rayd::torch::SceneTraceBackend::Optix}),
        rayd::torch::create_scene(scene_meshes(cuda_rng, scattered_triangles), {rayd::torch::SceneTraceBackend::Cuda}),
    };
}

void test_backend_selection_and_probe(bool optix_available) {
    std::mt19937 rng(7);
    auto automatic = rayd::torch::create_scene(scene_meshes(rng, 8));
    const auto expected =
        optix_available ? rayd::torch::SceneTraceBackend::Optix : rayd::torch::SceneTraceBackend::Cuda;
    require(rayd::torch::scene_trace_backend(automatic) == expected, "Auto must resolve to OptiX exactly when usable");
    auto cuda = rayd::torch::create_scene(scene_meshes(rng, 8), {rayd::torch::SceneTraceBackend::Cuda});
    require(rayd::torch::scene_trace_backend(cuda) == rayd::torch::SceneTraceBackend::Cuda,
            "a forced CUDA scene must report the CUDA backend");
    if (optix_available) {
        auto optix = rayd::torch::create_scene(scene_meshes(rng, 8), {rayd::torch::SceneTraceBackend::Optix});
        require(rayd::torch::scene_trace_backend(optix) == rayd::torch::SceneTraceBackend::Optix,
                "a forced OptiX scene must report the OptiX backend");
    } else {
        bool threw = false;
        try {
            (void)rayd::torch::create_scene(scene_meshes(rng, 8), {rayd::torch::SceneTraceBackend::Optix});
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, "a forced OptiX scene must fail loudly when OptiX is unusable");
    }
    bool threw = false;
    try {
        (void)rayd::torch::optix_availability(-1);
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "an invalid device index must fail loudly");
}

at::Tensor random_unit_rows(std::mt19937& rng, int64_t rows) {
    const at::Tensor raw = uniform(rng, rows, 3, -1.0f, 1.0f);
    return (raw / raw.norm(2, 1, true).clamp_min(1.0e-3f)).contiguous();
}

rayd::torch::AxialEdgeVisibilityRequest axial_request(unsigned seed, int64_t states) {
    std::mt19937 rng(seed);
    rayd::torch::AxialEdgeVisibilityRequest request;
    request.tx = at::tensor({-4.7f, 0.3f, 0.2f}, at::TensorOptions().dtype(at::kFloat)).to(floats());
    request.edge_position = uniform(rng, states, 3, -4.5f, 4.5f);
    request.edge_direction = random_unit_rows(rng, states);
    request.edge_t_min = uniform(rng, states, 1, -1.0f, 0.0f).reshape({states}).contiguous();
    request.edge_t_max = (request.edge_t_min + uniform(rng, states, 1, 0.0f, 2.0f).reshape({states})).contiguous();
    return request;
}

void test_axial_edge_visibility_parity(const ScenePair& scenes) {
    constexpr int64_t states = 8192;
    auto request = axial_request(11, states);
    const at::Tensor optix = rayd::torch::axial_edge_visibility_forward(scenes.optix, request).any_visible;
    const at::Tensor cuda = rayd::torch::axial_edge_visibility_forward(scenes.cuda, request).any_visible;
    require_equal(optix, cuda, "axial-edge any_visible");
    const int64_t visible = optix.sum().item<int64_t>();
    require(visible > states / 10 && visible < states - states / 10,
            "axial-edge parity fixture must mix visible and blocked states");

    // Masked, non-finite, and empty requests take the same early-outs on both backends.
    request.active = (at::arange(states, ints()).remainder(3) != 0).contiguous();
    request.edge_position.index_put_({at::indexing::Slice(0, 64, 7), 1}, std::numeric_limits<float>::quiet_NaN());
    request.edge_t_max.index_put_({at::indexing::Slice(1, 64, 5)}, std::numeric_limits<float>::infinity());
    require_equal(rayd::torch::axial_edge_visibility_forward(scenes.optix, request).any_visible,
                  rayd::torch::axial_edge_visibility_forward(scenes.cuda, request).any_visible,
                  "axial-edge masked and non-finite any_visible");
    auto empty = axial_request(12, 0);
    require(rayd::torch::axial_edge_visibility_forward(scenes.cuda, empty).any_visible.numel() == 0,
            "empty CUDA axial-edge request");
}

rayd::torch::SegmentPenetrationRequest penetration_request(const rayd::torch::SceneResource& scene,
                                                           const at::Tensor& origins, const at::Tensor& targets,
                                                           const std::optional<at::Tensor>& active,
                                                           std::int64_t capacity,
                                                           rayd::torch::SegmentPenetrationPolicy policy) {
    return {
        scene, origins, targets, active, true, capacity, policy, 17.3, at::zeros({1}, ints()), 4,
    };
}

void require_penetration_result_parity(const rayd::torch::SegmentPenetrationResult& optix,
                                       const rayd::torch::SegmentPenetrationResult& cuda, const std::string& name) {
    require_equal(optix.valid, cuda.valid, name + " valid");
    require_equal(optix.num_hits, cuda.num_hits, name + " num_hits");
    require_equal(optix.reached_target, cuda.reached_target, name + " reached_target");
    require_equal(optix.overflow, cuda.overflow, name + " overflow");
    require_equal(optix.global_primitive_id, cuda.global_primitive_id, name + " global_primitive_id");
    require_equal(optix.distance, cuda.distance, name + " distance");
    require_equal(optix.direction, cuda.direction, name + " direction");
    require_close(optix.t, cuda.t, 1.0e-5, 1.0e-5, name + " t");
    require_close(optix.position, cuda.position, 1.0e-5, 1.0e-5, name + " position");
    // Both normals come from the winning triangle alone, but the CUDA kernel and the OptiX module are compiled
    // separately and may contract the cross product into FMAs differently.
    require_close(optix.normal, cuda.normal, 0.0, 1.0e-6, name + " normal");
    require_close(optix.geometric_normal, cuda.geometric_normal, 0.0, 1.0e-6, name + " geometric_normal");
}

void test_segment_penetration_parity(const ScenePair& scenes) {
    constexpr int64_t segments = 8192;
    std::mt19937 rng(23);
    const at::Tensor origins = uniform(rng, segments, 3, -4.8f, 4.8f);
    const at::Tensor targets = uniform(rng, segments, 3, -4.8f, 4.8f);
    const at::Tensor active = (at::arange(segments, ints()).remainder(17) != 0).contiguous();
    for (const auto policy : {rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
                              rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset}) {
        const std::string name =
            policy == rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance ? "enumerated" : "monte-carlo";
        constexpr std::int64_t capacity = 24;
        auto optix_request = penetration_request(scenes.optix, origins, targets, active, capacity, policy);
        auto cuda_request = penetration_request(scenes.cuda, origins, targets, active, capacity, policy);
        const auto optix = rayd::torch::segment_penetration_forward_tape(optix_request);
        const auto cuda = rayd::torch::segment_penetration_forward_tape(cuda_request);
        require_equal(optix_request.capacity_failure_state, cuda_request.capacity_failure_state,
                      name + " failure state");
        require(optix_request.capacity_failure_state.item<int>() == 0, name + " fixture must not overflow");
        require_penetration_result_parity(optix.result, cuda.result, name);
        require(optix.result.num_hits.sum().item<int64_t>() > segments, name + " fixture must cross many surfaces");
        require_equal(optix.tape_primitive_id, cuda.tape_primitive_id, name + " tape_primitive_id");
        require_close(optix.tape_barycentric, cuda.tape_barycentric, 1.0e-4, 1.0e-4, name + " tape_barycentric");
        require_close(optix.tape_restart_epsilon, cuda.tape_restart_epsilon, 1.0e-5, 1.0e-9,
                      name + " tape_restart_epsilon");
        require_equal(optix.tape_restart_branch, cuda.tape_restart_branch, name + " tape_restart_branch");
        require_equal(optix.tape_restart_tie_mask, cuda.tape_restart_tie_mask, name + " tape_restart_tie_mask");
        require_equal(optix.tape_direction_denominator_branch, cuda.tape_direction_denominator_branch,
                      name + " tape_direction_denominator_branch");

        const auto plain = rayd::torch::segment_penetration_forward(cuda_request);
        require_penetration_result_parity(cuda.result, plain, name + " CUDA plain versus tape");

        // Fixed-winner derivatives consume each backend's own tape.
        std::mt19937 grad_rng(29);
        const at::Tensor grad_t = uniform(grad_rng, segments, capacity, -1.0f, 1.0f);
        const at::Tensor grad_position =
            uniform(grad_rng, segments * capacity, 3, -1.0f, 1.0f).reshape({segments, capacity, 3});
        const at::Tensor grad_normal =
            uniform(grad_rng, segments * capacity, 3, -1.0f, 1.0f).reshape({segments, capacity, 3});
        const at::Tensor grad_distance = uniform(grad_rng, segments, 1, -1.0f, 1.0f).reshape({segments});
        auto backward = [&](const rayd::torch::SegmentPenetrationRequest& request,
                            const rayd::torch::SegmentPenetrationTapeResult& tape) {
            return rayd::torch::segment_penetration_backward({request, tape, grad_distance, std::nullopt, grad_t,
                                                              grad_position, grad_normal, std::nullopt, true, true,
                                                              true});
        };
        const auto optix_vjp = backward(optix_request, optix);
        const auto cuda_vjp = backward(cuda_request, cuda);
        require_close(optix_vjp.grad_vertices, cuda_vjp.grad_vertices, 1.0e-3, 1.0e-3, name + " grad_vertices");
        require_close(optix_vjp.grad_origins, cuda_vjp.grad_origins, 1.0e-3, 1.0e-3, name + " grad_origins");
        require_close(optix_vjp.grad_targets, cuda_vjp.grad_targets, 1.0e-3, 1.0e-3, name + " grad_targets");

        const at::Tensor tangent_origins = uniform(grad_rng, segments, 3, -1.0f, 1.0f);
        const at::Tensor tangent_targets = uniform(grad_rng, segments, 3, -1.0f, 1.0f);
        auto jvp = [&](const rayd::torch::SegmentPenetrationRequest& request,
                       const rayd::torch::SegmentPenetrationTapeResult& tape) {
            return rayd::torch::segment_penetration_jvp(
                {request, tape, std::nullopt, tangent_origins, tangent_targets});
        };
        const auto optix_jvp = jvp(optix_request, optix);
        const auto cuda_jvp = jvp(cuda_request, cuda);
        require_close(optix_jvp.tangent_distance, cuda_jvp.tangent_distance, 1.0e-4, 1.0e-4,
                      name + " tangent_distance");
        require_close(optix_jvp.tangent_t, cuda_jvp.tangent_t, 1.0e-3, 1.0e-3, name + " tangent_t");
        require_close(optix_jvp.tangent_position, cuda_jvp.tangent_position, 1.0e-3, 1.0e-3,
                      name + " tangent_position");
    }

    // A capacity too small for some segments fails the shared transaction identically and leaves the same overflow
    // diagnostic.
    auto optix_small = penetration_request(scenes.optix, origins, targets, active, 1,
                                           rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance);
    auto cuda_small = penetration_request(scenes.cuda, origins, targets, active, 1,
                                          rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance);
    const auto optix_overflow = rayd::torch::segment_penetration_forward(optix_small);
    const auto cuda_overflow = rayd::torch::segment_penetration_forward(cuda_small);
    require(optix_small.capacity_failure_state.item<int>() == 4, "small capacity must fail the transaction");
    require_equal(optix_small.capacity_failure_state, cuda_small.capacity_failure_state, "overflow failure state");
    require_penetration_result_parity(optix_overflow, cuda_overflow, "overflow");

    // DropRow drops the same rows on both backends and leaves the rest of the batch valid.
    for (const auto policy : {rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance,
                              rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset}) {
        auto optix_drop = penetration_request(scenes.optix, origins, targets, active, 1, policy);
        auto cuda_drop = penetration_request(scenes.cuda, origins, targets, active, 1, policy);
        optix_drop.overflow_policy = rayd::torch::SegmentPenetrationOverflowPolicy::DropRow;
        cuda_drop.overflow_policy = rayd::torch::SegmentPenetrationOverflowPolicy::DropRow;
        optix_drop.dropped_row_count = at::zeros({1}, ints());
        cuda_drop.dropped_row_count = at::zeros({1}, ints());
        const auto optix_dropped = rayd::torch::segment_penetration_forward(optix_drop);
        const auto cuda_dropped = rayd::torch::segment_penetration_forward(cuda_drop);
        require(optix_drop.capacity_failure_state.item<int>() == 0 && cuda_drop.capacity_failure_state.item<int>() == 0,
                "DropRow must not fail the transaction");
        require(optix_drop.dropped_row_count->item<int>() == optix_dropped.overflow.sum().item<int>() &&
                    optix_drop.dropped_row_count->item<int>() > 0,
                "DropRow count must equal the overflow rows");
        require_equal(*optix_drop.dropped_row_count, *cuda_drop.dropped_row_count, "DropRow count");
        require(optix_dropped.valid.any().item<bool>(), "DropRow fixture must keep valid rows");
        require_penetration_result_parity(optix_dropped, cuda_dropped, "DropRow");
    }

    // input_active_any=false submits no trace on either backend.
    auto inactive = penetration_request(scenes.cuda, origins, targets, at::zeros({segments}, bools()), 4,
                                        rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance);
    inactive.input_active_any = false;
    const auto none = rayd::torch::segment_penetration_forward(inactive);
    require(inactive.capacity_failure_state.item<int>() == 0 && none.num_hits.sum().item<int64_t>() == 0,
            "all-inactive CUDA request must stay inert");
}

void test_intersection_parity(const ScenePair& scenes) {
    constexpr int64_t rays = 8192;
    std::mt19937 rng(31);
    const rayd::torch::RayBatch batch = {uniform(rng, rays, 3, -4.8f, 4.8f), random_unit_rows(rng, rays), std::nullopt,
                                         std::nullopt};
    const auto optix = rayd::torch::intersect_forward(scenes.optix, batch, 7);
    const auto cuda = rayd::torch::intersect_forward(scenes.cuda, batch, 7);
    require_equal(optix.global_prim_id, cuda.global_prim_id, "intersect global_prim_id");
    require_equal(optix.shape_id, cuda.shape_id, "intersect shape_id");
    const at::Tensor hit = optix.global_prim_id >= 0;
    require_close(optix.t.masked_select(hit), cuda.t.masked_select(hit), 1.0e-5, 1.0e-5, "intersect t");
}

// A closed box whose twelve convex edges give real wedges; the two diagonal edges of each square face are flat and
// are dropped.
struct WedgeFixture {
    rayd::torch::DiffractionState state;
    at::Tensor active;
    int64_t state_limit;
    int64_t face_count;
};

std::vector<rayd::torch::MeshInput> box_meshes() {
    const std::vector<float> vertices = {
        -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, 1.0f,  1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 1.0f,
    };
    const std::vector<int> faces = {
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    };
    std::vector<rayd::torch::MeshInput> meshes;
    meshes.push_back(
        mesh_input(at::tensor(vertices, at::TensorOptions().dtype(at::kFloat)).reshape({-1, 3}).to(floats()),
                   at::tensor(faces, at::TensorOptions().dtype(at::kInt)).reshape({-1, 3}).to(ints())));
    return meshes;
}

WedgeFixture wedge_fixture(const rayd::torch::SceneResource& scene, const at::Tensor& source) {
    const auto records = rayd::torch::scene_edge_records(scene);
    const at::Tensor normals =
        at::stack({records.tri_fn_x, records.tri_fn_y, records.tri_fn_z}, 1).to(at::kFloat).contiguous();
    const at::Tensor unit_normals = normals / normals.norm(2, 1, true).clamp_min(1.0e-12f);
    const at::Tensor face0 = records.edge_face0_global.to(at::kLong);
    const at::Tensor face1 = records.edge_face1_global.to(at::kLong);
    const at::Tensor has_two_faces = face1 >= 0;
    const at::Tensor n0_all = unit_normals.index_select(0, face0);
    const at::Tensor n1_all = unit_normals.index_select(0, face1.clamp_min(0));
    const at::Tensor keep = at::logical_and(has_two_faces, (n0_all * n1_all).sum(1).abs() < 0.5f);
    const at::Tensor rows = keep.nonzero().reshape({-1});
    const int64_t count = rows.numel();
    const at::Tensor v0 = records.global_vertices.index_select(0, records.edge_v0.index_select(0, rows).to(at::kLong));
    const at::Tensor v1 = records.global_vertices.index_select(0, records.edge_v1.index_select(0, rows).to(at::kLong));
    const at::Tensor span = v1 - v0;
    const at::Tensor length = span.norm(2, 1);
    rayd::torch::DiffractionState state = {
        rows.to(at::kInt).contiguous(),
        v0.contiguous(),
        (span / length.unsqueeze(1)).contiguous(),
        at::zeros({count}, floats()),
        length.contiguous(),
        n0_all.index_select(0, rows).contiguous(),
        n1_all.index_select(0, rows).contiguous(),
        face0.index_select(0, rows).to(at::kInt).contiguous(),
        face1.index_select(0, rows).to(at::kInt).contiguous(),
        at::full({count}, static_cast<float>(1.5 * 3.14159265358979323846), floats()),
        source.reshape({1, 3}).expand({count, 3}).contiguous(),
        at::ones({count}, floats()),
        std::nullopt,
        std::nullopt,
    };
    return {std::move(state), at::ones({count}, bools()), count, records.tri_fn_x.numel()};
}

// Material columns are views with stride 2, so a backend that ignored the strides would read the wrong faces.
rayd::torch::MaterialPayload strided_material(int64_t faces) {
    std::mt19937 rng(53);
    const at::Tensor eta = uniform(rng, faces, 2, 2.0f, 6.0f);
    const at::Tensor sigma = uniform(rng, faces, 2, 0.0f, 0.2f);
    const at::Tensor mu = uniform(rng, faces, 2, 1.0f, 1.5f);
    const at::Tensor gain = uniform(rng, faces, 2, 0.3f, 1.0f);
    at::Tensor valid = at::ones({faces, 2}, bools());
    valid.index_put_({3, 0}, false);
    return {eta.select(1, 0), sigma.select(1, 0), mu.select(1, 0), gain.select(1, 0), valid.select(1, 0)};
}

// Receivers on a shell around the box plus receivers just off the incident shadow boundary of every edge midpoint,
// where a non-zero ISB taper changes the field.
at::Tensor wedge_receivers(const WedgeFixture& fixture, const at::Tensor& source) {
    std::mt19937 rng(59);
    const at::Tensor shell = random_unit_rows(rng, 48) * 3.5f;
    const at::Tensor midpoint =
        fixture.state.edge_pos + fixture.state.edge_dir * (0.5f * fixture.state.edge_t_max).unsqueeze(1);
    const at::Tensor away = midpoint - source.reshape({1, 3});
    const at::Tensor away_unit = away / away.norm(2, 1, true);
    const at::Tensor offset = uniform(rng, fixture.state_limit, 3, -0.05f, 0.05f);
    const at::Tensor boundary = midpoint + away_unit * 2.0f + offset;
    return at::cat({shell, boundary}, 0).contiguous();
}

void require_field_close(const at::Tensor& optix, const at::Tensor& cuda, const std::string& name) {
    // OptiX modules compile with fast math and the CUDA kernels without it, so amplitudes agree to a few 1e-4
    // relative; scale the absolute tolerance by the largest entry.
    const double scale = std::max(optix.abs().max().item<double>(), 1.0e-30);
    require_close(optix, cuda, 2.0e-3, 2.0e-3 * scale, name);
}

void require_paths_parity(const rayd::torch::DiffractionPathResult& optix,
                          const rayd::torch::DiffractionPathResult& cuda, rayd::torch::DiffractionPathLayout layout,
                          int64_t rx_count, const std::string& name) {
    require_equal(optix.count, cuda.count, name + " count");
    std::vector<at::Tensor> optix_columns = {optix.valid.to(at::kFloat), optix.tx_id.to(at::kFloat),
                                             optix.rx_id.to(at::kFloat), optix.order.to(at::kFloat),
                                             optix.edge0.to(at::kFloat), optix.edge1.to(at::kFloat),
                                             optix.edge2.to(at::kFloat)};
    at::Tensor optix_order = at::arange(optix.valid.numel(), optix.valid.options().dtype(at::kLong));
    at::Tensor cuda_order = optix_order;
    if (layout == rayd::torch::DiffractionPathLayout::Compact) {
        // Compact rows are appended in atomic order, which differs between the backends; compare them by key.
        auto key = [&](const rayd::torch::DiffractionPathResult& result) {
            // edge0 is a scene edge id, not a state index, so it gets its own wide digit.
            const at::Tensor raw = (result.tx_id.to(at::kLong) * rx_count + result.rx_id.to(at::kLong)) * (1ll << 32) +
                                   result.edge0.to(at::kLong);
            return at::where(result.valid, raw, at::full_like(raw, std::numeric_limits<int64_t>::max()));
        };
        optix_order = std::get<1>(key(optix).sort());
        cuda_order = std::get<1>(key(cuda).sort());
    }
    auto pick = [](const at::Tensor& tensor, const at::Tensor& order) { return tensor.index_select(0, order); };
    for (const auto& entry : {std::pair<const at::Tensor*, const at::Tensor*>{&optix.valid, &cuda.valid},
                              {&optix.tx_id, &cuda.tx_id},
                              {&optix.rx_id, &cuda.rx_id},
                              {&optix.order, &cuda.order},
                              {&optix.edge0, &cuda.edge0},
                              {&optix.edge1, &cuda.edge1},
                              {&optix.edge2, &cuda.edge2}}) {
        require_equal(pick(*entry.first, optix_order), pick(*entry.second, cuda_order), name + " path ids");
    }
    require_close(pick(optix.delay, optix_order), pick(cuda.delay, cuda_order), 1.0e-5, 1.0e-12, name + " delay");
    require_close(pick(optix.p0, optix_order), pick(cuda.p0, cuda_order), 1.0e-5, 1.0e-5, name + " p0");
    require_equal(pick(optix.p1, optix_order), pick(cuda.p1, cuda_order), name + " p1");
    require_equal(pick(optix.p2, optix_order), pick(cuda.p2, cuda_order), name + " p2");
    const at::Tensor optix_field = at::stack({optix.field_x_re, optix.field_x_im, optix.field_y_re, optix.field_y_im,
                                              optix.field_z_re, optix.field_z_im},
                                             1);
    const at::Tensor cuda_field = at::stack({cuda.field_x_re, cuda.field_x_im, cuda.field_y_re, cuda.field_y_im,
                                             cuda.field_z_re, cuda.field_z_im},
                                            1);
    require_field_close(pick(optix_field, optix_order), pick(cuda_field, cuda_order), name + " field");
}

at::Tensor path_field(const rayd::torch::DiffractionPathResult& result) {
    return at::stack({result.field_x_re, result.field_x_im, result.field_y_re, result.field_y_im, result.field_z_re,
                      result.field_z_im},
                     1);
}

void test_diffraction_parity() {
    auto optix_scene = rayd::torch::create_scene(box_meshes(), {rayd::torch::SceneTraceBackend::Optix});
    auto cuda_scene = rayd::torch::create_scene(box_meshes(), {rayd::torch::SceneTraceBackend::Cuda});
    const at::Tensor source = at::tensor({3.0f, 0.4f, 2.2f}, at::TensorOptions().dtype(at::kFloat)).to(floats());
    const WedgeFixture fixture = wedge_fixture(optix_scene, source);
    require(fixture.state_limit == 12, "the box fixture must expose its twelve convex edges");
    const auto material = strided_material(fixture.face_count);
    const at::Tensor receivers = wedge_receivers(fixture, source);
    const int64_t rx_count = receivers.size(0);
    const at::Tensor tx_pos = at::stack({source, source + 0.25f}, 0).contiguous();

    const std::vector<std::pair<std::string, at::Tensor>> polarizations = {
        {"x", at::tensor({1.0f, 0.0f, 0.0f}, at::TensorOptions().dtype(at::kFloat)).reshape({1, 3}).to(floats())},
        {"z", at::tensor({0.0f, 0.0f, 1.0f}, at::TensorOptions().dtype(at::kFloat)).reshape({1, 3}).to(floats())},
        {"per-transmitter", at::tensor({0.0f, 1.0f, 0.0f, 0.3f, 0.8f, -0.52f}, at::TensorOptions().dtype(at::kFloat))
                                .reshape({2, 3})
                                .to(floats())},
    };
    auto run = [&](const rayd::torch::SceneResource& scene, const at::Tensor& tx_pol, double isb,
                   rayd::torch::DiffractionPathLayout layout) {
        rayd::torch::DiffractionPathConfig config;
        config.tx_pos = tx_pos;
        config.tx_pol = tx_pol;
        config.rx_pos = receivers;
        config.active = fixture.active;
        config.state = fixture.state;
        config.material = material;
        config.state_limit = fixture.state_limit;
        config.capacity = 2 * rx_count * fixture.state_limit;
        config.wavelength = 0.125;
        config.isb_taper_width_scale = isb;
        config.layout = layout;
        return rayd::torch::diffraction_paths_order1_forward(scene, config);
    };
    std::optional<at::Tensor> x_field;
    std::optional<at::Tensor> hard_field;
    for (const auto& polarization : polarizations) {
        for (const double isb : {0.0, 0.35}) {
            for (const auto layout :
                 {rayd::torch::DiffractionPathLayout::Compact, rayd::torch::DiffractionPathLayout::SourceLane}) {
                const std::string name =
                    "diffraction paths pol=" + polarization.first + " isb=" + std::to_string(isb) +
                    (layout == rayd::torch::DiffractionPathLayout::Compact ? " compact" : " source-lane");
                const auto optix = run(optix_scene, polarization.second, isb, layout);
                const auto cuda = run(cuda_scene, polarization.second, isb, layout);
                require(optix.count.item<int>() > 2 * fixture.state_limit, name + ": fixture exports too few paths");
                require_paths_parity(optix, cuda, layout, rx_count, name);
                if (layout == rayd::torch::DiffractionPathLayout::SourceLane) {
                    if (polarization.first == "x" && isb == 0.0)
                        x_field = path_field(cuda);
                    if (polarization.first == "z" && isb == 0.0)
                        hard_field = path_field(cuda);
                    if (polarization.first == "z" && isb != 0.0) {
                        const double change = (path_field(cuda) - *hard_field).abs().max().item<double>();
                        require(change > 0.0, "the fixture must place receivers where the ISB taper acts");
                    }
                }
            }
        }
    }
    // The CUDA backend must not collapse every polarization to +x.
    require((*x_field - *hard_field).abs().max().item<double>() > 1.0e-3 * x_field->abs().max().item<double>(),
            "z polarization must change the CUDA diffraction field");

    rayd::torch::Grid2D grid = {2, -1.5, -4.0, 4.0, -4.0, 4.0, 16, 16, 0.25};
    auto accumulate = [&](const rayd::torch::SceneResource& scene) {
        rayd::torch::DiffractionAccumulationConfig config;
        config.active = fixture.active;
        config.state = fixture.state;
        config.material = material;
        config.state_limit = fixture.state_limit;
        config.grid = grid;
        config.wavelength = 0.125;
        config.direct_samples = 4096;
        config.keller_samples = 1024;
        config.seed = 17;
        return rayd::torch::diffraction_accumulation_forward(scene, config);
    };
    const auto optix_accumulated = accumulate(optix_scene);
    const auto cuda_accumulated = accumulate(cuda_scene);
    for (const auto& counter : {std::pair<const at::Tensor*, const at::Tensor*>{&optix_accumulated.direct_count,
                                                                                &cuda_accumulated.direct_count},
                                {&optix_accumulated.keller_count, &cuda_accumulated.keller_count},
                                {&optix_accumulated.visibility_rejects, &cuda_accumulated.visibility_rejects},
                                {&optix_accumulated.edge_visibility_rejects, &cuda_accumulated.edge_visibility_rejects},
                                {&optix_accumulated.utd_rejects, &cuda_accumulated.utd_rejects},
                                {&optix_accumulated.edge_uses, &cuda_accumulated.edge_uses}}) {
        require_equal(*counter.first, *counter.second, "diffraction accumulation counters");
    }
    require(optix_accumulated.direct_count.item<int>() > 0, "accumulation fixture must reach the grid");
    require_field_close(optix_accumulated.power, cuda_accumulated.power, "diffraction accumulation power");
    require_field_close(at::stack({optix_accumulated.field_x_re, optix_accumulated.field_x_im,
                                   optix_accumulated.field_y_re, optix_accumulated.field_y_im,
                                   optix_accumulated.field_z_re, optix_accumulated.field_z_im}),
                        at::stack({cuda_accumulated.field_x_re, cuda_accumulated.field_x_im,
                                   cuda_accumulated.field_y_re, cuda_accumulated.field_y_im,
                                   cuda_accumulated.field_z_re, cuda_accumulated.field_z_im}),
                        "diffraction accumulation field");

    for (const bool select_point : {false, true}) {
        auto coherent = [&](const rayd::torch::SceneResource& scene) {
            rayd::torch::CoherentDiffractionConfig config;
            config.active = fixture.active;
            config.state = fixture.state;
            config.material = material;
            config.state_limit = fixture.state_limit;
            config.grid = grid;
            config.wavelength = 0.125;
            config.select_diffraction_point = select_point;
            config.prefilter_visibility = select_point;
            return rayd::torch::diffraction_coherent_accumulation_forward(scene, config);
        };
        const auto optix_coherent = coherent(optix_scene);
        const auto cuda_coherent = coherent(cuda_scene);
        const std::string name = std::string("coherent diffraction select=") + (select_point ? "1" : "0");
        require_equal(optix_coherent.direct_count, cuda_coherent.direct_count, name + " direct_count");
        require_equal(optix_coherent.multi_count, cuda_coherent.multi_count, name + " multi_count");
        require_equal(optix_coherent.visibility_reject_count, cuda_coherent.visibility_reject_count,
                      name + " visibility_reject_count");
        require_equal(optix_coherent.utd_reject_count, cuda_coherent.utd_reject_count, name + " utd_reject_count");
        require(optix_coherent.direct_count.sum().item<int64_t>() > 0, name + ": fixture must reach the grid");
        require_field_close(at::stack(
                                {optix_coherent.direct_x_re, optix_coherent.direct_x_im, optix_coherent.direct_y_re,
                                 optix_coherent.direct_y_im, optix_coherent.direct_z_re, optix_coherent.direct_z_im,
                                 optix_coherent.multi_x_re, optix_coherent.multi_x_im, optix_coherent.multi_y_re,
                                 optix_coherent.multi_y_im, optix_coherent.multi_z_re, optix_coherent.multi_z_im}),
                            at::stack({cuda_coherent.direct_x_re, cuda_coherent.direct_x_im, cuda_coherent.direct_y_re,
                                       cuda_coherent.direct_y_im, cuda_coherent.direct_z_re, cuda_coherent.direct_z_im,
                                       cuda_coherent.multi_x_re, cuda_coherent.multi_x_im, cuda_coherent.multi_y_re,
                                       cuda_coherent.multi_y_im, cuda_coherent.multi_z_re, cuda_coherent.multi_z_im}),
                            name + " field");
    }
}

double median_ms(const std::function<void()>& work, int repetitions) {
    std::vector<double> samples;
    for (int index = 0; index < repetitions; ++index) {
        at::cuda::getCurrentCUDAStream().synchronize();
        const auto start = std::chrono::steady_clock::now();
        work();
        at::cuda::getCurrentCUDAStream().synchronize();
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Opt-in timing (RAYD_TRACE_BACKEND_BENCH=1): one line per operation and backend, median of repeated calls.
void benchmark(int scattered_triangles) {
    const ScenePair scenes = make_scenes(scattered_triangles, 3);
    constexpr int64_t count = 1 << 18;
    auto axial = axial_request(41, count);
    std::mt19937 rng(43);
    const at::Tensor origins = uniform(rng, count, 3, -4.8f, 4.8f);
    const at::Tensor targets = uniform(rng, count, 3, -4.8f, 4.8f);
    for (const auto* entry : {&scenes.optix, &scenes.cuda}) {
        const char* backend = entry == &scenes.optix ? "optix" : "cuda";
        const double axial_ms = median_ms([&] { (void)rayd::torch::axial_edge_visibility_forward(*entry, axial); }, 21);
        auto request = penetration_request(*entry, origins, targets, std::nullopt, 64,
                                           rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance);
        const double penetration_ms = median_ms([&] { (void)rayd::torch::segment_penetration_forward(request); }, 21);
        const double tape_ms = median_ms([&] { (void)rayd::torch::segment_penetration_forward_tape(request); }, 21);
        require(request.capacity_failure_state.item<int>() == 0, "benchmark segments must not overflow");
        std::cout << "[BENCH] triangles=" << scattered_triangles + 10 << " lanes=" << count << " backend=" << backend
                  << " axial_edge_visibility_ms=" << axial_ms << " segment_penetration_forward_ms=" << penetration_ms
                  << " segment_penetration_forward_tape_ms=" << tape_ms << std::endl;
    }
}

} // namespace

int main() {
    try {
        if (!at::cuda::is_available() || at::cuda::device_count() == 0) {
            std::cout << "[SKIP] CUDA is unavailable" << std::endl;
            return kSkipReturnCode;
        }
        const auto probe = rayd::torch::optix_availability(0);
        std::cout << "[RUN] test_backend_selection_and_probe (OptiX "
                  << (probe.available ? "available" : "unavailable: " + probe.reason) << ")" << std::endl;
        test_backend_selection_and_probe(probe.available);
        if (!probe.available) {
            std::cout << "[SKIP] OptiX is unavailable, so the OptiX/CUDA parity checks cannot run" << std::endl;
            return kSkipReturnCode;
        }
        const ScenePair scenes = make_scenes(2048, 5);
        std::cout << "[RUN] test_axial_edge_visibility_parity" << std::endl;
        test_axial_edge_visibility_parity(scenes);
        std::cout << "[RUN] test_segment_penetration_parity" << std::endl;
        test_segment_penetration_parity(scenes);
        std::cout << "[RUN] test_intersection_parity" << std::endl;
        test_intersection_parity(scenes);
        std::cout << "[RUN] test_diffraction_parity" << std::endl;
        test_diffraction_parity();
        const char* bench = std::getenv("RAYD_TRACE_BACKEND_BENCH");
        if (bench != nullptr && std::string(bench) == "1") {
            for (int triangles : {2048, 65536})
                benchmark(triangles);
        }
        std::cout << "rayd::torch trace backend parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rayd::torch trace backend parity failure: " << error.what() << '\n';
        return 1;
    }
}
