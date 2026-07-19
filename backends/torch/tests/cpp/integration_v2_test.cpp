#include <rayd/torch/integration.h>
#include <rayd/torch/integration_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

static_assert(rayd::torch::kIntegrationApiVersion == 2);
static_assert(!rayd::torch::kIntegrationHeaderIdentity.empty());

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition)
        fail(message);
}

template <typename Fn>
void require_throws(Fn &&fn, const std::string &message) {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::exception &) {
        return;
    }
    fail(message);
}

void require_tensor_exact(
    const at::Tensor &actual,
    const at::Tensor &expected,
    const std::string &name) {
    require(actual.defined() == expected.defined(), name + ": defined state differs");
    if (!actual.defined())
        return;
    require(actual.sizes() == expected.sizes(), name + ": shape differs");
    require(actual.strides() == expected.strides(), name + ": stride differs");
    require(actual.scalar_type() == expected.scalar_type(), name + ": dtype differs");
    require(actual.device() == expected.device(), name + ": device differs");
    require(at::equal(actual, expected), name + ": values differ");
}

template <std::size_t Size>
void require_tensor_arrays_exact(
    const std::array<at::Tensor, Size> &actual,
    const std::array<at::Tensor, Size> &expected,
    const std::string &name) {
    for (std::size_t index = 0; index < Size; ++index)
        require_tensor_exact(actual[index], expected[index], name + " " + std::to_string(index));
}

const at::Tensor *optional_tensor_ptr(const std::optional<at::Tensor> &tensor) {
    return tensor.has_value() && tensor->defined() ? &*tensor : nullptr;
}

struct MeshFixture {
    at::Tensor vertices;
    at::Tensor faces;
    at::Tensor uv;
    at::Tensor face_uv;
    at::Tensor to_world_left;
    at::Tensor to_world_right;
};

MeshFixture make_triangle(int device_index = 0) {
    const auto float_options = at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(at::Device(at::kCUDA, device_index));
    const auto int_options = at::TensorOptions()
                                 .dtype(at::kInt)
                                 .device(at::Device(at::kCUDA, device_index));
    return {
        at::tensor(
            {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
            float_options)
            .reshape({3, 3}),
        at::tensor({0, 1, 2}, int_options).reshape({1, 3}),
        at::empty({0, 2}, float_options),
        at::empty({0, 3}, int_options),
        at::empty({0, 4}, float_options),
        at::empty({0, 4}, float_options),
    };
}

rayd::torch::MeshInput mesh_input(const MeshFixture &mesh) {
    return {
        mesh.vertices,
        mesh.faces,
        mesh.uv,
        mesh.face_uv,
        mesh.to_world_left,
        mesh.to_world_right,
        false,
        true,
        false,
    };
}

struct LegacyScene {
    explicit LegacyScene(const MeshFixture &mesh) {
        const std::int64_t flags = 2;
        handle = rayd_torch_native_scene_create(
            &mesh.vertices,
            &mesh.faces,
            &mesh.uv,
            &mesh.face_uv,
            &mesh.to_world_left,
            &mesh.to_world_right,
            &flags,
            1);
        require(handle != 0, "legacy scene creation returned a null handle");
    }

    LegacyScene(const LegacyScene &) = delete;
    LegacyScene &operator=(const LegacyScene &) = delete;

    ~LegacyScene() {
        if (handle != 0)
            rayd_torch_native_scene_destroy(handle);
    }

    std::int64_t handle = 0;
};

rayd::torch::RayBatch one_hit_ray(int device_index = 0) {
    const auto float_options = at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(at::Device(at::kCUDA, device_index));
    return {
        at::tensor({0.25F, 0.25F, -1.0F}, float_options).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, 1.0F}, float_options).reshape({1, 3}),
        std::nullopt,
        std::nullopt,
    };
}

rayd::torch::RayBatch empty_rays_with_present_empty_mask(int device_index = 0) {
    const auto float_options = at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(at::Device(at::kCUDA, device_index));
    const auto bool_options = at::TensorOptions()
                                  .dtype(at::kBool)
                                  .device(at::Device(at::kCUDA, device_index));
    return {
        at::empty({0, 3}, float_options),
        at::empty({0, 3}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, bool_options),
    };
}

struct EmptyDiffractionFixture {
    rayd::torch::DiffractionState state;
    rayd::torch::MaterialPayload material;
    rayd::torch::Grid2D grid;
    at::Tensor active;
    at::Tensor tx_pos;
    at::Tensor tx_pol;
    at::Tensor rx_pos;
    at::Tensor sample_state_index;
    at::Tensor sample_edge_weight;
};

EmptyDiffractionFixture make_empty_diffraction_fixture(int device_index = 0) {
    const auto float_options = at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(at::Device(at::kCUDA, device_index));
    const auto int_options = at::TensorOptions()
                                 .dtype(at::kInt)
                                 .device(at::Device(at::kCUDA, device_index));
    const auto bool_options = at::TensorOptions()
                                  .dtype(at::kBool)
                                  .device(at::Device(at::kCUDA, device_index));
    rayd::torch::DiffractionState state = {
        at::empty({0}, int_options),
        at::empty({0, 3}, float_options),
        at::empty({0, 3}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0, 3}, float_options),
        at::empty({0, 3}, float_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, float_options),
        at::empty({0, 3}, float_options),
        at::empty({0}, float_options),
        std::nullopt,
        std::nullopt,
    };
    rayd::torch::MaterialPayload material = {
        at::ones({1}, float_options),
        at::zeros({1}, float_options),
        at::ones({1}, float_options),
        at::ones({1}, float_options),
        at::ones({1}, bool_options),
    };
    rayd::torch::Grid2D grid = {
        2,
        0.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        2,
        2,
        1.0,
    };
    return {
        std::move(state),
        std::move(material),
        grid,
        at::empty({0}, bool_options),
        at::tensor({0.0F, 0.0F, -1.0F}, float_options).reshape({1, 3}),
        at::tensor({1.0F, 0.0F, 0.0F}, float_options).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, 1.0F}, float_options).reshape({1, 3}),
        at::empty({0}, int_options),
        at::empty({0}, float_options),
    };
}

std::array<at::Tensor, 10> legacy_intersect(
    const LegacyScene &scene,
    const rayd::torch::RayBatch &rays,
    std::int64_t flags = 0) {
    std::array<at::Tensor, 10> values;
    const at::Tensor *active =
        rays.active.has_value() && rays.active->defined() ? &*rays.active : nullptr;
    const at::Tensor ray_tmax =
        rays.ray_tmax.has_value() && rays.ray_tmax->defined()
        ? *rays.ray_tmax
        : at::Tensor();
    const std::int64_t count = rayd_torch_native_intersect_forward(
        scene.handle,
        &rays.ray_o,
        &rays.ray_d,
        &ray_tmax,
        active,
        flags,
        values.data(),
        static_cast<std::int64_t>(values.size()));
    require(count == static_cast<std::int64_t>(values.size()), "legacy intersect output count differs");
    return values;
}

void test_scene_and_intersection_lockstep() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy(mesh);

    require(scene.valid(), "typed scene is not valid");
    require(scene.device_index() == 0, "typed scene reports the wrong device");

    const auto typed_edges = rayd::torch::scene_edge_records(scene);
    std::array<at::Tensor, 12> legacy_edges;
    const std::int64_t edge_count = rayd_torch_native_scene_edge_records(
        legacy.handle,
        legacy_edges.data(),
        static_cast<std::int64_t>(legacy_edges.size()));
    require(edge_count == static_cast<std::int64_t>(legacy_edges.size()), "legacy edge output count differs");
    const std::array<at::Tensor, 12> typed_edge_values = {
        typed_edges.global_vertices,
        typed_edges.global_faces,
        typed_edges.tri_fn_x,
        typed_edges.tri_fn_y,
        typed_edges.tri_fn_z,
        typed_edges.edge_v0,
        typed_edges.edge_v1,
        typed_edges.edge_face0_global,
        typed_edges.edge_face1_global,
        typed_edges.edge_shape_id,
        typed_edges.edge_local_id,
        typed_edges.edge_opposite,
    };
    for (std::size_t index = 0; index < typed_edge_values.size(); ++index)
        require_tensor_exact(typed_edge_values[index], legacy_edges[index], "scene edge output " + std::to_string(index));

    const auto rays = one_hit_ray();
    const auto typed = rayd::torch::intersect_forward(scene, rays, 7);
    const auto legacy_values = legacy_intersect(legacy, rays, 7);
    const std::array<at::Tensor, 10> typed_values = {
        typed.t,
        typed.p,
        typed.n,
        typed.geo_n,
        typed.uv,
        typed.barycentric,
        typed.shape_id,
        typed.prim_id,
        typed.local_prim_id,
        typed.global_prim_id,
    };
    for (std::size_t index = 0; index < typed_values.size(); ++index)
        require_tensor_exact(typed_values[index], legacy_values[index], "intersect output " + std::to_string(index));

    rayd::torch::IntersectBackwardRequest backward_request;
    backward_request.rays = rays;
    backward_request.tape_prim_id = typed.global_prim_id;
    backward_request.tape_barycentric = typed.barycentric;
    backward_request.grad_t = at::ones_like(typed.t);
    backward_request.need_grad_vertices = true;
    backward_request.need_grad_ray_o = true;
    backward_request.need_grad_ray_d = true;
    backward_request.need_grad_ray_tmax = true;
    const auto typed_backward = rayd::torch::intersect_backward(scene, backward_request);
    const at::Tensor legacy_ray_tmax =
        rays.ray_tmax.has_value() && rays.ray_tmax->defined()
        ? *rays.ray_tmax
        : at::Tensor();
    std::array<at::Tensor, 4> legacy_backward;
    const std::int64_t backward_count = rayd_torch_native_intersect_backward(
        legacy.handle,
        &rays.ray_o,
        &rays.ray_d,
        &legacy_ray_tmax,
        nullptr,
        &backward_request.tape_prim_id,
        &backward_request.tape_barycentric,
        optional_tensor_ptr(backward_request.grad_t),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        backward_request.need_grad_vertices,
        backward_request.need_grad_ray_o,
        backward_request.need_grad_ray_d,
        backward_request.need_grad_ray_tmax,
        legacy_backward.data(),
        static_cast<std::int64_t>(legacy_backward.size()));
    require(backward_count == static_cast<std::int64_t>(legacy_backward.size()), "legacy intersect backward output count differs");
    const std::array<at::Tensor, 4> typed_backward_values = {
        typed_backward.grad_vertices,
        typed_backward.grad_ray_o,
        typed_backward.grad_ray_d,
        typed_backward.grad_ray_tmax,
    };
    require_tensor_arrays_exact(typed_backward_values, legacy_backward, "intersect backward output");

    rayd::torch::IntersectJvpRequest jvp_request;
    jvp_request.ray_o = rays.ray_o;
    jvp_request.ray_d = rays.ray_d;
    jvp_request.tape_prim_id = typed.global_prim_id;
    jvp_request.tape_barycentric = typed.barycentric;
    jvp_request.tangent_vertices = at::zeros_like(mesh.vertices);
    jvp_request.tangent_ray_o = at::ones_like(rays.ray_o);
    jvp_request.tangent_ray_d = at::zeros_like(rays.ray_d);
    jvp_request.flags = 7;
    const auto typed_jvp = rayd::torch::intersect_jvp(scene, jvp_request);
    std::array<at::Tensor, 6> legacy_jvp;
    const std::int64_t jvp_count = rayd_torch_native_intersect_jvp(
        legacy.handle,
        &jvp_request.ray_o,
        &jvp_request.ray_d,
        nullptr,
        &jvp_request.tape_prim_id,
        &jvp_request.tape_barycentric,
        optional_tensor_ptr(jvp_request.tangent_vertices),
        optional_tensor_ptr(jvp_request.tangent_ray_o),
        optional_tensor_ptr(jvp_request.tangent_ray_d),
        jvp_request.flags,
        legacy_jvp.data(),
        static_cast<std::int64_t>(legacy_jvp.size()));
    require(jvp_count == static_cast<std::int64_t>(legacy_jvp.size()), "legacy intersect JVP output count differs");
    const std::array<at::Tensor, 6> typed_jvp_values = {
        typed_jvp.tangent_t,
        typed_jvp.tangent_p,
        typed_jvp.tangent_n,
        typed_jvp.tangent_geo_n,
        typed_jvp.tangent_uv,
        typed_jvp.tangent_barycentric,
    };
    require_tensor_arrays_exact(typed_jvp_values, legacy_jvp, "intersect JVP output");
}

void test_empty_and_stream_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy(mesh);
    const auto rays = empty_rays_with_present_empty_mask();

    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        const auto typed = rayd::torch::intersect_forward(scene, rays, 7);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
            "typed intersection changed the caller's active CUDA stream");
        const auto legacy_values = legacy_intersect(legacy, rays, 7);
        const std::array<at::Tensor, 10> typed_values = {
            typed.t,
            typed.p,
            typed.n,
            typed.geo_n,
            typed.uv,
            typed.barycentric,
            typed.shape_id,
            typed.prim_id,
            typed.local_prim_id,
            typed.global_prim_id,
        };
        for (std::size_t index = 0; index < typed_values.size(); ++index) {
            require(typed_values[index].defined(), "empty typed output must remain defined");
            require_tensor_exact(typed_values[index], legacy_values[index], "empty intersect output " + std::to_string(index));
        }
    }
}

void test_visibility_trace_and_face_normal_lockstep() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy(mesh);
    const auto float_options = mesh.vertices.options();

    rayd::torch::VisibilityRequest visibility_request = {
        at::tensor({0.25F, 0.25F, -1.0F}, float_options).reshape({1, 3}),
        at::tensor({0.25F, 0.25F, 1.0F}, float_options).reshape({1, 3}),
        std::nullopt,
    };
    const auto typed_visibility = rayd::torch::visibility_forward(scene, visibility_request);
    std::array<at::Tensor, 3> legacy_visibility;
    rayd_torch_native_visibility_forward(
        legacy.handle,
        &visibility_request.start,
        &visibility_request.end,
        nullptr,
        &legacy_visibility[0],
        &legacy_visibility[1],
        &legacy_visibility[2]);
    const std::array<at::Tensor, 3> typed_visibility_values = {
        typed_visibility.visible,
        typed_visibility.blocker_prim,
        typed_visibility.tape_t,
    };
    require_tensor_arrays_exact(
        typed_visibility_values, legacy_visibility, "visibility output");

    const auto rays = one_hit_ray();
    const at::Tensor legacy_ray_tmax =
        rays.ray_tmax.has_value() && rays.ray_tmax->defined()
        ? *rays.ray_tmax
        : at::Tensor();
    const rayd::torch::ReflectionTraceRequest trace_request = {rays, 1};
    const auto typed_trace = rayd::torch::trace_reflections_forward(scene, trace_request);
    std::array<at::Tensor, 3> legacy_trace;
    const std::int64_t trace_count = rayd_torch_native_trace_reflections_forward(
        legacy.handle,
        &rays.ray_o,
        &rays.ray_d,
        &legacy_ray_tmax,
        nullptr,
        trace_request.max_bounces,
        legacy_trace.data(),
        static_cast<std::int64_t>(legacy_trace.size()));
    require(trace_count == static_cast<std::int64_t>(legacy_trace.size()), "legacy reflection trace output count differs");
    const std::array<at::Tensor, 3> typed_trace_values = {
        typed_trace.valid,
        typed_trace.t,
        typed_trace.prim_ids,
    };
    require_tensor_arrays_exact(typed_trace_values, legacy_trace, "reflection trace output");

    const auto typed_tape = rayd::torch::trace_reflections_forward_tape(scene, trace_request);
    std::array<at::Tensor, 9> legacy_tape;
    const std::int64_t tape_count = rayd_torch_native_trace_reflections_forward_tape(
        legacy.handle,
        &rays.ray_o,
        &rays.ray_d,
        &legacy_ray_tmax,
        nullptr,
        trace_request.max_bounces,
        legacy_tape.data(),
        static_cast<std::int64_t>(legacy_tape.size()));
    require(tape_count == static_cast<std::int64_t>(legacy_tape.size()), "legacy reflection tape output count differs");
    const std::array<at::Tensor, 9> typed_tape_values = {
        typed_tape.valid,
        typed_tape.t,
        typed_tape.image_sources,
        typed_tape.prim_ids,
        typed_tape.tape_prim_id,
        typed_tape.tape_barycentric,
        typed_tape.tape_hit_points,
        typed_tape.tape_normals,
        typed_tape.active_ctx,
    };
    require_tensor_arrays_exact(typed_tape_values, legacy_tape, "reflection tape output");
    require(
        typed_tape.prim_ids.unsafeGetTensorImpl() == typed_tape.tape_prim_id.unsafeGetTensorImpl(),
        "typed reflection tape must preserve prim-id tensor aliasing");

    rayd::torch::ReflectionTraceBackwardRequest trace_backward;
    trace_backward.rays = rays;
    trace_backward.tape_prim_id = typed_tape.tape_prim_id;
    trace_backward.tape_barycentric = typed_tape.tape_barycentric;
    trace_backward.tape_hit_points = typed_tape.tape_hit_points;
    trace_backward.tape_normals = typed_tape.tape_normals;
    trace_backward.image_sources = typed_tape.image_sources;
    trace_backward.grad_t = at::ones_like(typed_tape.t);
    trace_backward.grad_image_sources = at::zeros_like(typed_tape.image_sources);
    const auto typed_trace_backward =
        rayd::torch::trace_reflections_backward(scene, trace_backward);
    std::array<at::Tensor, 4> legacy_trace_backward;
    const std::int64_t trace_backward_count =
        rayd_torch_native_trace_reflections_backward(
            legacy.handle,
            &rays.ray_o,
            &rays.ray_d,
            &legacy_ray_tmax,
            nullptr,
            &trace_backward.tape_prim_id,
            &trace_backward.tape_barycentric,
            &trace_backward.tape_hit_points,
            &trace_backward.tape_normals,
            &trace_backward.image_sources,
            optional_tensor_ptr(trace_backward.grad_t),
            optional_tensor_ptr(trace_backward.grad_image_sources),
            legacy_trace_backward.data(),
            static_cast<std::int64_t>(legacy_trace_backward.size()));
    require(
        trace_backward_count == static_cast<std::int64_t>(legacy_trace_backward.size()),
        "legacy reflection trace backward output count differs");
    const std::array<at::Tensor, 4> typed_trace_backward_values = {
        typed_trace_backward.grad_vertices,
        typed_trace_backward.grad_ray_o,
        typed_trace_backward.grad_ray_d,
        typed_trace_backward.grad_ray_tmax,
    };
    require_tensor_arrays_exact(
        typed_trace_backward_values,
        legacy_trace_backward,
        "reflection trace backward output");

    rayd::torch::ReflectionTraceJvpRequest trace_jvp;
    trace_jvp.ray_o = rays.ray_o;
    trace_jvp.ray_d = rays.ray_d;
    trace_jvp.tape_prim_id = typed_tape.tape_prim_id;
    trace_jvp.tape_barycentric = typed_tape.tape_barycentric;
    trace_jvp.tape_hit_points = typed_tape.tape_hit_points;
    trace_jvp.tape_normals = typed_tape.tape_normals;
    trace_jvp.tangent_vertices = at::zeros_like(mesh.vertices);
    trace_jvp.tangent_ray_o = at::ones_like(rays.ray_o);
    trace_jvp.tangent_ray_d = at::zeros_like(rays.ray_d);
    trace_jvp.image_sources = typed_tape.image_sources;
    const auto typed_trace_jvp = rayd::torch::trace_reflections_jvp(scene, trace_jvp);
    std::array<at::Tensor, 2> legacy_trace_jvp;
    const std::int64_t trace_jvp_count = rayd_torch_native_trace_reflections_jvp(
        legacy.handle,
        &trace_jvp.ray_o,
        &trace_jvp.ray_d,
        nullptr,
        &trace_jvp.tape_prim_id,
        &trace_jvp.tape_barycentric,
        &trace_jvp.tape_hit_points,
        &trace_jvp.tape_normals,
        optional_tensor_ptr(trace_jvp.tangent_vertices),
        optional_tensor_ptr(trace_jvp.tangent_ray_o),
        optional_tensor_ptr(trace_jvp.tangent_ray_d),
        &trace_jvp.image_sources,
        legacy_trace_jvp.data(),
        static_cast<std::int64_t>(legacy_trace_jvp.size()));
    require(trace_jvp_count == static_cast<std::int64_t>(legacy_trace_jvp.size()), "legacy reflection trace JVP output count differs");
    const std::array<at::Tensor, 2> typed_trace_jvp_values = {
        typed_trace_jvp.tangent_t,
        typed_trace_jvp.tangent_image_sources,
    };
    require_tensor_arrays_exact(
        typed_trace_jvp_values, legacy_trace_jvp, "reflection trace JVP output");

    const auto edges = rayd::torch::scene_edge_records(scene);
    const at::Tensor grad_face_normals =
        at::ones({edges.global_faces.size(0), 3}, float_options);
    const auto typed_face_backward =
        rayd::torch::scene_face_normals_backward(scene, grad_face_normals);
    std::array<at::Tensor, 1> legacy_face_backward;
    const std::int64_t backward_count = rayd_torch_native_scene_face_normals_backward(
        legacy.handle,
        &grad_face_normals,
        legacy_face_backward.data(),
        1);
    require(backward_count == 1, "legacy face-normal backward output count differs");
    require_tensor_exact(
        typed_face_backward, legacy_face_backward[0], "face-normal backward output");

    const at::Tensor tangent_vertices = at::ones_like(edges.global_vertices);
    const auto typed_face_jvp = rayd::torch::scene_face_normals_jvp(scene, tangent_vertices);
    std::array<at::Tensor, 1> legacy_face_jvp;
    const std::int64_t jvp_count = rayd_torch_native_scene_face_normals_jvp(
        legacy.handle,
        &tangent_vertices,
        legacy_face_jvp.data(),
        1);
    require(jvp_count == 1, "legacy face-normal JVP output count differs");
    require_tensor_exact(typed_face_jvp, legacy_face_jvp[0], "face-normal JVP output");
}

void test_reflection_accumulation_and_epc_lockstep() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy(mesh);
    const auto float_options = mesh.vertices.options();
    const auto int_options = mesh.faces.options();
    const auto bool_options = at::TensorOptions().dtype(at::kBool).device(mesh.vertices.device());

    const rayd::torch::MaterialPayload material = {
        at::ones({1}, float_options),
        at::zeros({1}, float_options),
        at::ones({1}, float_options),
        at::ones({1}, float_options),
        at::ones({1}, bool_options),
    };
    const rayd::torch::Grid2D grid = {
        2,
        0.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        2,
        2,
        1.0,
    };
    rayd::torch::ReflectionAccumulationConfig accumulation = {
        empty_rays_with_present_empty_mask(),
        at::empty({0, 3}, float_options),
        at::empty({0, 3}, float_options),
        material,
        0,
        grid,
        0.1,
        0.0,
        false,
        false,
        0,
        1,
        0,
        0,
        0,
        0,
        false,
    };
    const auto typed_accumulation =
        rayd::torch::reflection_accumulation_forward(scene, accumulation);
    const std::array<at::Tensor, 18> typed_accumulation_values = {
        typed_accumulation.power,
        typed_accumulation.field_x_re,
        typed_accumulation.field_x_im,
        typed_accumulation.field_y_re,
        typed_accumulation.field_y_im,
        typed_accumulation.field_z_re,
        typed_accumulation.field_z_im,
        typed_accumulation.reflection_count,
        typed_accumulation.wedge_count,
        typed_accumulation.wedge_ray_index,
        typed_accumulation.wedge_hit,
        typed_accumulation.wedge_normal,
        typed_accumulation.wedge_prim_id,
        typed_accumulation.wedge_direction,
        typed_accumulation.wedge_source,
        typed_accumulation.wedge_source_power,
        typed_accumulation.wedge_initial_direction,
        typed_accumulation.wedge_bounce_depth,
    };
    std::array<at::Tensor, 18> legacy_accumulation;
    const at::Tensor legacy_accumulation_ray_tmax =
        accumulation.rays.ray_tmax.has_value() && accumulation.rays.ray_tmax->defined()
        ? *accumulation.rays.ray_tmax
        : at::Tensor();
    const std::int64_t accumulation_count =
        rayd_torch_native_reflection_accumulation_forward(
            legacy.handle,
            &accumulation.rays.ray_o,
            &accumulation.rays.ray_d,
            &legacy_accumulation_ray_tmax,
            optional_tensor_ptr(accumulation.rays.active),
            &accumulation.tx,
            &accumulation.tx_pol,
            &accumulation.material.eta_r,
            &accumulation.material.sigma,
            &accumulation.material.mu_r,
            &accumulation.material.gain,
            &accumulation.material.valid,
            accumulation.max_bounces,
            accumulation.grid.axis,
            accumulation.grid.position,
            accumulation.grid.coord0_min,
            accumulation.grid.coord0_max,
            accumulation.grid.coord1_min,
            accumulation.grid.coord1_max,
            accumulation.grid.resolution0,
            accumulation.grid.resolution1,
            accumulation.wavelength,
            accumulation.solid_angle_per_ray,
            accumulation.collect_wedges,
            accumulation.collect_wedge_prefixes,
            accumulation.wedge_capacity,
            accumulation.wedge_sample_stride,
            accumulation.accumulation_strategy,
            accumulation.compact_min_samples,
            accumulation.staged_min_samples_per_cell,
            accumulation.procedural_sample_count,
            accumulation.include_los,
            legacy_accumulation.data(),
            static_cast<std::int64_t>(legacy_accumulation.size()));
    require(
        accumulation_count == static_cast<std::int64_t>(legacy_accumulation.size()),
        "legacy reflection accumulation output count differs");
    require_tensor_arrays_exact(
        typed_accumulation_values,
        legacy_accumulation,
        "reflection accumulation output");

    rayd::torch::ReflectionEpcRequest epc = {
        at::empty({0, 3}, float_options),
        at::empty({0, 3}, float_options),
        at::empty({0}, bool_options),
        at::empty({0, 1}, int_options),
        at::empty({0, 1, 3}, float_options),
        at::empty({0, 1, 3}, float_options),
        at::empty({0}, int_options),
        at::zeros({1}, int_options),
        at::empty({0}, int_options),
        1,
        0,
        0.0,
    };
    const auto typed_epc = rayd::torch::reflection_epc_paths_forward(scene, epc);
    const std::array<at::Tensor, 6> typed_epc_values = {
        typed_epc.valid,
        typed_epc.path_length,
        typed_epc.resolved_prim_ids,
        typed_epc.surface_group_ids,
        typed_epc.hit_positions,
        typed_epc.normals,
    };
    std::array<at::Tensor, 6> legacy_epc;
    const std::int64_t epc_count = rayd_torch_native_reflection_epc_paths_forward(
        legacy.handle,
        &epc.source,
        &epc.receiver,
        optional_tensor_ptr(epc.active),
        &epc.expected_prim_ids,
        &epc.direct_plane_points,
        &epc.direct_plane_normals,
        &epc.surface_group_id,
        &epc.surface_group_size,
        &epc.surface_group_members,
        epc.max_bounces,
        epc.visibility_ignore_mode,
        epc.plane_tolerance,
        legacy_epc.data(),
        static_cast<std::int64_t>(legacy_epc.size()));
    require(epc_count == static_cast<std::int64_t>(legacy_epc.size()), "legacy reflection EPC output count differs");
    require_tensor_arrays_exact(typed_epc_values, legacy_epc, "reflection EPC output");

    const at::Tensor bounce_count = at::empty({0}, int_options);
    rayd::torch::ReflectionEpcBackwardRequest epc_backward;
    epc_backward.source = epc.source;
    epc_backward.receiver = epc.receiver;
    epc_backward.sequence = epc.expected_prim_ids;
    epc_backward.plane_points = epc.direct_plane_points;
    epc_backward.plane_normals = epc.direct_plane_normals;
    epc_backward.valid = typed_epc.valid;
    epc_backward.bounce_count = bounce_count;
    epc_backward.need_grad_vertices = true;
    epc_backward.need_grad_source = true;
    epc_backward.need_grad_receiver = true;
    const auto typed_epc_backward =
        rayd::torch::reflection_epc_paths_backward(scene, epc_backward);
    std::array<at::Tensor, 3> legacy_epc_backward;
    const std::int64_t epc_backward_count =
        rayd_torch_native_reflection_epc_paths_backward(
            legacy.handle,
            &epc_backward.source,
            &epc_backward.receiver,
            &epc_backward.sequence,
            &epc_backward.plane_points,
            &epc_backward.plane_normals,
            &epc_backward.valid,
            &epc_backward.bounce_count,
            nullptr,
            nullptr,
            nullptr,
            epc_backward.need_grad_vertices,
            epc_backward.need_grad_source,
            epc_backward.need_grad_receiver,
            legacy_epc_backward.data(),
            static_cast<std::int64_t>(legacy_epc_backward.size()));
    require(
        epc_backward_count == static_cast<std::int64_t>(legacy_epc_backward.size()),
        "legacy reflection EPC backward output count differs");
    const std::array<at::Tensor, 3> typed_epc_backward_values = {
        typed_epc_backward.grad_vertices,
        typed_epc_backward.grad_source,
        typed_epc_backward.grad_receiver,
    };
    require_tensor_arrays_exact(
        typed_epc_backward_values,
        legacy_epc_backward,
        "reflection EPC backward output");

    rayd::torch::ReflectionEpcJvpRequest epc_jvp;
    epc_jvp.source = epc.source;
    epc_jvp.receiver = epc.receiver;
    epc_jvp.sequence = epc.expected_prim_ids;
    epc_jvp.plane_points = epc.direct_plane_points;
    epc_jvp.plane_normals = epc.direct_plane_normals;
    epc_jvp.valid = typed_epc.valid;
    epc_jvp.bounce_count = bounce_count;
    epc_jvp.tangent_vertices = at::zeros_like(mesh.vertices);
    epc_jvp.tangent_source = at::empty_like(epc.source);
    epc_jvp.tangent_receiver = at::empty_like(epc.receiver);
    const auto typed_epc_jvp = rayd::torch::reflection_epc_paths_jvp(scene, epc_jvp);
    std::array<at::Tensor, 3> legacy_epc_jvp;
    const std::int64_t epc_jvp_count = rayd_torch_native_reflection_epc_paths_jvp(
        legacy.handle,
        &epc_jvp.source,
        &epc_jvp.receiver,
        &epc_jvp.sequence,
        &epc_jvp.plane_points,
        &epc_jvp.plane_normals,
        &epc_jvp.valid,
        &epc_jvp.bounce_count,
        optional_tensor_ptr(epc_jvp.tangent_vertices),
        optional_tensor_ptr(epc_jvp.tangent_source),
        optional_tensor_ptr(epc_jvp.tangent_receiver),
        legacy_epc_jvp.data(),
        static_cast<std::int64_t>(legacy_epc_jvp.size()));
    require(epc_jvp_count == static_cast<std::int64_t>(legacy_epc_jvp.size()), "legacy reflection EPC JVP output count differs");
    const std::array<at::Tensor, 3> typed_epc_jvp_values = {
        typed_epc_jvp.tangent_points,
        typed_epc_jvp.tangent_normals,
        typed_epc_jvp.tangent_path_length,
    };
    require_tensor_arrays_exact(
        typed_epc_jvp_values, legacy_epc_jvp, "reflection EPC JVP output");
}

void test_error_and_lifecycle_contracts() {
    require_throws(
        [] { (void)rayd::torch::create_scene({}); },
        "empty typed scene construction must fail");

    MeshFixture bad_mesh = make_triangle();
    bad_mesh.vertices = bad_mesh.vertices.to(at::kInt);
    require_throws(
        [&] { (void)rayd::torch::create_scene({mesh_input(bad_mesh)}); },
        "invalid mesh dtype must fail scene construction");

    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    auto moved = std::move(scene);
    require(!scene.valid(), "moved-from scene must be invalid");
    require(moved.valid(), "moved-to scene must retain ownership");
    require_throws(
        [&] { (void)rayd::torch::scene_edge_records(scene); },
        "operation on moved-from scene must fail");

    auto invalid_rays = one_hit_ray();
    invalid_rays.ray_o = invalid_rays.ray_o.to(at::kDouble);
    require_throws(
        [&] { (void)rayd::torch::intersect_forward(moved, invalid_rays, 0); },
        "invalid ray dtype must fail");
    auto invalid_shape = one_hit_ray();
    invalid_shape.ray_o = at::empty({1, 2}, mesh.vertices.options());
    require_throws(
        [&] { (void)rayd::torch::intersect_forward(moved, invalid_shape, 0); },
        "invalid ray shape must fail");
    const auto valid = rayd::torch::intersect_forward(moved, one_hit_ray(), 0);
    require(valid.t.numel() == 1, "scene must remain usable after a rejected operation");

    for (int iteration = 0; iteration < 4; ++iteration) {
        auto temporary = rayd::torch::create_scene({mesh_input(mesh)});
        require(temporary.valid(), "repeated scene construction failed");
    }

    std::vector<rayd::torch::SceneResource> simultaneous;
    simultaneous.reserve(3);
    simultaneous.push_back(rayd::torch::create_scene({mesh_input(mesh)}));
    simultaneous.push_back(rayd::torch::create_scene({mesh_input(mesh)}));
    simultaneous.push_back(rayd::torch::create_scene({mesh_input(mesh)}));
    simultaneous.erase(simultaneous.begin() + 1);
    require(simultaneous.size() == 2, "non-creation-order scene teardown failed");
    require(simultaneous[0].valid() && simultaneous[1].valid(), "independent scene ownership was corrupted");
    require(
        rayd::torch::intersect_forward(simultaneous[0], one_hit_ray(), 0).t.numel() == 1 &&
            rayd::torch::intersect_forward(simultaneous[1], one_hit_ray(), 0).t.numel() == 1,
        "surviving scenes failed after independent teardown");

    require_throws(
        [&] {
            auto exception_scene = rayd::torch::create_scene({mesh_input(mesh)});
            auto invalid = one_hit_ray();
            invalid.ray_d = invalid.ray_d.to(at::kDouble);
            (void)rayd::torch::intersect_forward(exception_scene, invalid, 0);
        },
        "scene exception teardown path did not reject an invalid operation");

    if (at::cuda::device_count() > 1) {
        const auto rays_on_second_device = one_hit_ray(1);
        require_throws(
            [&] { (void)rayd::torch::intersect_forward(moved, rays_on_second_device, 0); },
            "cross-device operation must fail");
    }
}

void test_diffraction_paths_lockstep() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy_scene(mesh);
    auto fixture = make_empty_diffraction_fixture();

    rayd::torch::DiffractionPathConfig config = {
        fixture.tx_pos,
        fixture.tx_pol,
        fixture.rx_pos,
        fixture.active,
        fixture.state,
        fixture.material,
        0,
        0,
        0.1,
        0.0,
    };
    const auto typed = rayd::torch::diffraction_paths_order1_forward(scene, config);
    const std::array<at::Tensor, 18> typed_values = {
        typed.count,
        typed.valid,
        typed.tx_id,
        typed.rx_id,
        typed.order,
        typed.edge0,
        typed.edge1,
        typed.edge2,
        typed.delay,
        typed.field_x_re,
        typed.field_x_im,
        typed.field_y_re,
        typed.field_y_im,
        typed.field_z_re,
        typed.field_z_im,
        typed.p0,
        typed.p1,
        typed.p2,
    };
    std::array<at::Tensor, 18> legacy_values;
    const std::int64_t count = rayd_torch_native_diffraction_paths_order1_forward(
        legacy_scene.handle,
        &config.tx_pos,
        &config.tx_pol,
        &config.rx_pos,
        optional_tensor_ptr(config.active),
        &config.state.edge_index,
        &config.state.edge_pos,
        &config.state.edge_dir,
        &config.state.edge_t_min,
        &config.state.edge_t_max,
        &config.state.n0,
        &config.state.n1,
        &config.state.prim0,
        &config.state.prim1,
        &config.state.exterior_angle,
        &config.state.src,
        &config.state.src_power,
        &config.material.eta_r,
        &config.material.sigma,
        &config.material.mu_r,
        &config.material.gain,
        &config.material.valid,
        config.state_limit,
        config.capacity,
        config.wavelength,
        config.isb_taper_width_scale,
        legacy_values.data(),
        static_cast<std::int64_t>(legacy_values.size()));
    require(count == static_cast<std::int64_t>(legacy_values.size()), "legacy diffraction path output count differs");
    require_tensor_arrays_exact(typed_values, legacy_values, "diffraction path output");
}

void test_diffraction_accumulation_lockstep() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy_scene(mesh);
    auto fixture = make_empty_diffraction_fixture();

    rayd::torch::DiffractionAccumulationConfig config = {
        fixture.active,
        fixture.state,
        fixture.material,
        0,
        fixture.grid,
        0.1,
        0,
        0,
        0,
        17,
        1,
        std::nullopt,
        false,
        fixture.sample_state_index,
        fixture.sample_edge_weight,
    };
    const auto typed = rayd::torch::diffraction_accumulation_forward(scene, config);
    const std::array<at::Tensor, 19> typed_values = {
        typed.power,
        typed.field_x_re,
        typed.field_x_im,
        typed.field_y_re,
        typed.field_y_im,
        typed.field_z_re,
        typed.field_z_im,
        typed.direct_count,
        typed.keller_count,
        typed.suffix_count,
        typed.visibility_rejects,
        typed.edge_visibility_rejects,
        typed.utd_rejects,
        typed.edge_uses,
        typed.tape_active,
        typed.tape_state_idx,
        typed.tape_cell,
        typed.tape_material_idx,
        typed.tape_edge_u,
    };
    std::array<at::Tensor, 19> legacy_values;
    const std::int64_t count = rayd_torch_native_diffraction_accumulation_forward(
        legacy_scene.handle,
        optional_tensor_ptr(config.active),
        &config.state.edge_index,
        &config.state.edge_pos,
        &config.state.edge_dir,
        &config.state.edge_t_min,
        &config.state.edge_t_max,
        &config.state.n0,
        &config.state.n1,
        &config.state.prim0,
        &config.state.prim1,
        &config.state.exterior_angle,
        &config.state.src,
        &config.state.src_power,
        optional_tensor_ptr(config.state.wi),
        optional_tensor_ptr(config.state.d0),
        &config.material.eta_r,
        &config.material.sigma,
        &config.material.mu_r,
        &config.material.gain,
        &config.material.valid,
        config.state_limit,
        config.grid.axis,
        config.grid.position,
        config.grid.coord0_min,
        config.grid.coord0_max,
        config.grid.coord1_min,
        config.grid.coord1_max,
        config.grid.resolution0,
        config.grid.resolution1,
        config.grid.cell_area,
        config.wavelength,
        config.direct_samples,
        config.keller_samples,
        config.suffix_samples,
        config.seed,
        config.max_order,
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        config.export_tape ? 1 : 0,
        optional_tensor_ptr(config.sample_state_index),
        optional_tensor_ptr(config.sample_edge_weight),
        legacy_values.data(),
        static_cast<std::int64_t>(legacy_values.size()));
    require(count == static_cast<std::int64_t>(legacy_values.size()), "legacy diffraction accumulation output count differs");
    require_tensor_arrays_exact(typed_values, legacy_values, "diffraction accumulation output");
    for (std::size_t index = 14; index < typed_values.size(); ++index) {
        require(typed_values[index].defined(), "disabled diffraction tape output must remain defined");
        require(typed_values[index].numel() == 0, "disabled diffraction tape output must be empty");
    }
}

void test_coherent_diffraction_lockstep() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    LegacyScene legacy_scene(mesh);
    auto fixture = make_empty_diffraction_fixture();

    rayd::torch::CoherentDiffractionConfig config = {
        fixture.active,
        fixture.state,
        fixture.material,
        0,
        fixture.grid,
        0.1,
        true,
        true,
    };
    const auto typed = rayd::torch::diffraction_coherent_accumulation_forward(scene, config);
    const std::array<at::Tensor, 16> typed_values = {
        typed.direct_x_re,
        typed.direct_x_im,
        typed.direct_y_re,
        typed.direct_y_im,
        typed.direct_z_re,
        typed.direct_z_im,
        typed.multi_x_re,
        typed.multi_x_im,
        typed.multi_y_re,
        typed.multi_y_im,
        typed.multi_z_re,
        typed.multi_z_im,
        typed.direct_count,
        typed.multi_count,
        typed.visibility_reject_count,
        typed.utd_reject_count,
    };
    std::array<at::Tensor, 16> legacy_values;
    const std::int64_t count = rayd_torch_native_diffraction_coherent_accumulation_forward(
        legacy_scene.handle,
        optional_tensor_ptr(config.active),
        &config.state.edge_index,
        &config.state.edge_pos,
        &config.state.edge_dir,
        &config.state.edge_t_min,
        &config.state.edge_t_max,
        &config.state.n0,
        &config.state.n1,
        &config.state.prim0,
        &config.state.prim1,
        &config.state.exterior_angle,
        &config.state.src,
        &config.state.src_power,
        optional_tensor_ptr(config.state.wi),
        optional_tensor_ptr(config.state.d0),
        &config.material.eta_r,
        &config.material.sigma,
        &config.material.mu_r,
        &config.material.gain,
        &config.material.valid,
        config.state_limit,
        config.grid.axis,
        config.grid.position,
        config.grid.coord0_min,
        config.grid.coord0_max,
        config.grid.coord1_min,
        config.grid.coord1_max,
        config.grid.resolution0,
        config.grid.resolution1,
        config.grid.cell_area,
        config.wavelength,
        config.select_diffraction_point,
        config.prefilter_visibility,
        legacy_values.data(),
        static_cast<std::int64_t>(legacy_values.size()));
    require(count == static_cast<std::int64_t>(legacy_values.size()), "legacy coherent diffraction output count differs");
    require_tensor_arrays_exact(typed_values, legacy_values, "coherent diffraction output");
}

std::array<at::Tensor, 12> layer_stack_values(
    const rayd::torch::LayerStackResult &result) {
    return {
        result.r_te_real, result.r_te_imag, result.r_tm_real, result.r_tm_imag,
        result.t_te_real, result.t_te_imag, result.t_tm_real, result.t_tm_imag,
        result.cap_r_te, result.cap_r_tm, result.cap_t_te, result.cap_t_tm};
}

void test_layer_stack_empty_and_contracts() {
    const auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    const auto int_options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    rayd::torch::LayerStackRequest primal = {
        at::empty({0}, float_options),
        at::empty({0}, int_options),
        at::zeros({1}, int_options),
        at::zeros({1}, int_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        3.5e9,
    };

    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard guard(stream);
    const auto forward = rayd::torch::em_layer_stack_eval(primal);
    for (const auto &tensor : layer_stack_values(forward)) {
        require(tensor.defined(), "layer-stack empty forward output must be defined");
        require(tensor.numel() == 0, "layer-stack empty forward output must be empty");
        require(tensor.scalar_type() == at::kFloat, "layer-stack output dtype differs");
        require(tensor.device() == primal.cos_theta.device(), "layer-stack output device differs");
    }
    require(
        c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
        "layer-stack forward changed the caller's active CUDA stream");

    rayd::torch::LayerStackJvpRequest jvp;
    jvp.primal = primal;
    const auto tangent = rayd::torch::em_layer_stack_jvp(jvp);
    for (const auto &tensor : layer_stack_values(tangent))
        require(tensor.defined() && tensor.numel() == 0, "layer-stack empty JVP schema differs");

    rayd::torch::LayerStackBackwardRequest backward;
    backward.primal = primal;
    backward.need_cos_theta = true;
    backward.need_layers = true;
    backward.need_frequency = true;
    const auto gradients = rayd::torch::em_layer_stack_backward(backward);
    require(gradients.grad_cos_theta.numel() == 0, "empty cos-theta gradient schema differs");
    require(gradients.grad_layer_thickness_m.numel() == 0, "empty thickness gradient schema differs");
    require(gradients.grad_layer_eps_r.numel() == 0, "empty eps gradient schema differs");
    require(gradients.grad_layer_sigma_e.numel() == 0, "empty sigma gradient schema differs");
    require(
        gradients.grad_frequency.dim() == 1 && gradients.grad_frequency.size(0) == 1,
        "frequency gradient schema differs");

    auto invalid = primal;
    invalid.cos_theta = invalid.cos_theta.to(at::kDouble);
    require_throws(
        [&] { (void)rayd::torch::em_layer_stack_eval(invalid); },
        "layer-stack invalid dtype must fail loudly");
}

void test_layer_stack_nonempty_ad_and_stream() {
    const auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    const auto int_options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard guard(stream);

    rayd::torch::LayerStackRequest primal = {
        at::tensor({0.65F}, float_options),
        at::tensor({0}, int_options),
        at::tensor({0}, int_options),
        at::tensor({0}, int_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        3.5e9,
    };

    const auto identity = rayd::torch::em_layer_stack_eval(primal);
    stream.synchronize();
    const std::array<float, 12> identity_expected = {
        0.0F, 0.0F, 0.0F, 0.0F,
        1.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 1.0F,
    };
    const auto identity_values = layer_stack_values(identity);
    for (std::size_t index = 0; index < identity_values.size(); ++index) {
        const float value = identity_values[index].item<float>();
        require(
            std::fabs(value - identity_expected[index]) <= 1.0e-6F,
            "zero-layer identity output differs at field " + std::to_string(index));
    }

    primal.layer_count = at::tensor({1}, int_options);
    primal.layer_thickness_m = at::tensor({0.02F}, float_options);
    primal.layer_eps_r = at::tensor({4.0F}, float_options);
    primal.layer_sigma_e = at::tensor({0.01F}, float_options);
    primal.layer_mu_r = at::tensor({1.0F}, float_options);

    rayd::torch::LayerStackJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_cos_theta = at::tensor({0.1F}, float_options);
    jvp.tangent_layer_thickness_m = at::tensor({0.002F}, float_options);
    jvp.tangent_layer_eps_r = at::tensor({0.2F}, float_options);
    jvp.tangent_layer_sigma_e = at::tensor({0.001F}, float_options);
    jvp.tangent_frequency = 1.0e6;
    const auto tangent = rayd::torch::em_layer_stack_jvp(jvp);

    rayd::torch::LayerStackBackwardRequest backward;
    backward.primal = primal;
    for (std::size_t index = 0; index < backward.grad_outputs.size(); ++index)
        backward.grad_outputs[index] =
            at::full({1}, 0.125F * static_cast<float>(index + 1), float_options);
    backward.need_cos_theta = true;
    backward.need_layers = true;
    backward.need_frequency = true;
    const auto gradients = rayd::torch::em_layer_stack_backward(backward);
    stream.synchronize();

    double lhs = 0.0;
    const auto tangent_values = layer_stack_values(tangent);
    for (std::size_t index = 0; index < tangent_values.size(); ++index) {
        const double value = tangent_values[index].item<float>();
        require(std::isfinite(value), "nonempty layer-stack JVP must be finite");
        lhs += value * (0.125 * static_cast<double>(index + 1));
    }
    double rhs =
        static_cast<double>(gradients.grad_cos_theta.item<float>()) * 0.1 +
        static_cast<double>(gradients.grad_layer_thickness_m.item<float>()) * 0.002 +
        static_cast<double>(gradients.grad_layer_eps_r.item<float>()) * 0.2 +
        static_cast<double>(gradients.grad_layer_sigma_e.item<float>()) * 0.001 +
        static_cast<double>(gradients.grad_frequency.item<float>()) * 1.0e6;
    require(std::isfinite(rhs), "nonempty layer-stack VJP must be finite");
    const double duality_scale = std::fabs(lhs) > std::fabs(rhs) ? std::fabs(lhs) : std::fabs(rhs);
    require(
        std::fabs(lhs - rhs) <= 5.0e-4 * (duality_scale > 1.0 ? duality_scale : 1.0),
        "layer-stack JVP/VJP dot-product duality differs");
    require(
        c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
        "nonempty layer-stack entries changed the caller's active CUDA stream");

    auto disabled = backward;
    disabled.need_cos_theta = false;
    disabled.need_layers = false;
    disabled.need_frequency = false;
    const auto disabled_gradients = rayd::torch::em_layer_stack_backward(disabled);
    stream.synchronize();
    require(
        disabled_gradients.grad_cos_theta.item<float>() == 0.0F,
        "disabled cos gradient must be zero");
    require(
        disabled_gradients.grad_layer_thickness_m.item<float>() == 0.0F,
        "disabled thickness gradient must be zero");
    require(
        disabled_gradients.grad_layer_eps_r.item<float>() == 0.0F,
        "disabled eps gradient must be zero");
    require(
        disabled_gradients.grad_layer_sigma_e.item<float>() == 0.0F,
        "disabled sigma gradient must be zero");
    require(
        disabled_gradients.grad_frequency.item<float>() == 0.0F,
        "disabled frequency gradient must be zero");

    auto invalid_jvp = jvp;
    invalid_jvp.tangent_cos_theta = at::tensor(
        {0.1}, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA));
    require_throws(
        [&] { (void)rayd::torch::em_layer_stack_jvp(invalid_jvp); },
        "layer-stack invalid optional tangent dtype must fail loudly");
}

rayd::torch::LayerStackRequest lossy_layer_stack_request(bool active_layer) {
    const auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    const auto int_options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    return {
        at::tensor({0.55F}, float_options),
        at::tensor({0}, int_options),
        at::tensor({0}, int_options),
        at::tensor({active_layer ? 1 : 0}, int_options),
        at::tensor({0.12F}, float_options),
        at::tensor({4.0F}, float_options),
        at::tensor({0.025F}, float_options),
        at::tensor({1.0F}, float_options),
        3.5e9,
    };
}

void require_layer_stack_scalar(
    const at::Tensor &tensor,
    float expected,
    const std::string &name) {
    const float value = tensor.item<float>();
    require(
        std::isfinite(value) && std::fabs(value - expected) <= 3.0e-5F,
        name + " differs from the frozen complex128 baseline");
}

void test_layer_stack_lossy_primal_and_negative_contracts() {
    const auto request = lossy_layer_stack_request(true);
    const auto result = rayd::torch::em_layer_stack_eval(request);
    at::cuda::getCurrentCUDAStream(0).synchronize();

    // Independent complex128 transfer-matrix baseline for this one-layer row.
    const std::array<float, 12> expected = {
        -0.35482694F, -0.14114616F, -0.05475237F, -0.02465952F,
        -0.55237210F, 0.22183056F, -0.69762940F, 0.21103882F,
        0.14582440F, 0.00360591F, 0.35432373F, 0.53122417F,
    };
    const auto values = layer_stack_values(result);
    for (std::size_t index = 0; index < values.size(); ++index)
        require_layer_stack_scalar(
            values[index], expected[index], "lossy layer-stack field " + std::to_string(index));
    const float rta_te = result.cap_r_te.item<float>() + result.cap_t_te.item<float>();
    const float rta_tm = result.cap_r_tm.item<float>() + result.cap_t_tm.item<float>();
    require(rta_te >= 0.0F && rta_te <= 1.0F + 1.0e-5F, "lossy TE R+T is outside [0,1]");
    require(rta_tm >= 0.0F && rta_tm <= 1.0F + 1.0e-5F, "lossy TM R+T is outside [0,1]");

    auto bad_shape = request;
    bad_shape.material_id = at::tensor({0, 0}, request.material_id.options());
    require_throws(
        [&] { (void)rayd::torch::em_layer_stack_eval(bad_shape); },
        "layer-stack row-shape mismatch must fail loudly");

    auto bad_contiguous = request;
    bad_contiguous.cos_theta =
        at::ones({2, 2}, request.cos_theta.options()).select(1, 0);
    bad_contiguous.material_id =
        at::zeros({2}, request.material_id.options());
    require(!bad_contiguous.cos_theta.is_contiguous(), "test fixture must be non-contiguous");
    require_throws(
        [&] { (void)rayd::torch::em_layer_stack_eval(bad_contiguous); },
        "layer-stack non-contiguous input must fail loudly");

    rayd::torch::LayerStackBackwardRequest backward;
    backward.primal = request;
    backward.grad_outputs[0] =
        at::ones({1}, request.cos_theta.options().dtype(at::kDouble));
    require_throws(
        [&] { (void)rayd::torch::em_layer_stack_backward(backward); },
        "layer-stack backward cotangent dtype mismatch must fail loudly");
    backward.grad_outputs[0] = at::ones({2}, request.cos_theta.options());
    require_throws(
        [&] { (void)rayd::torch::em_layer_stack_backward(backward); },
        "layer-stack backward cotangent shape mismatch must fail loudly");

    if (at::cuda::device_count() > 1) {
        auto bad_device = request;
        bad_device.layer_eps_r = bad_device.layer_eps_r.to(at::Device(at::kCUDA, 1));
        require_throws(
            [&] { (void)rayd::torch::em_layer_stack_eval(bad_device); },
            "layer-stack cross-device input must fail loudly");
    }
}

void test_layer_stack_nondefault_stream_dependency() {
    auto request = lossy_layer_stack_request(false);
    at::cuda::getDefaultCUDAStream().synchronize();
    const auto producer = c10::cuda::getStreamFromPool(false, 0);
    const auto consumer = c10::cuda::getStreamFromPool(false, 0);
    require(producer.stream() != consumer.stream(), "stream-pool fixtures must differ");

    auto scratch = at::empty(
        {64 * 1024 * 1024}, request.cos_theta.options().dtype(at::kByte));
    cudaEvent_t ready = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    for (int iteration = 0; iteration < 32; ++iteration) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            scratch.data_ptr(), iteration, static_cast<std::size_t>(scratch.numel()),
            producer.stream()));
    }
    {
        c10::cuda::CUDAStreamGuard producer_guard(producer);
        request.layer_count.fill_(1);
    }
    C10_CUDA_CHECK(cudaEventRecord(ready, producer.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(consumer.stream(), ready, 0));

    rayd::torch::LayerStackResult result;
    {
        c10::cuda::CUDAStreamGuard consumer_guard(consumer);
        result = rayd::torch::em_layer_stack_eval(request);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == consumer.stream(),
            "layer-stack changed the caller's non-default stream");
    }
    consumer.synchronize();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    C10_CUDA_CHECK(cudaEventDestroy(ready));

    // A wrong default-stream launch races ahead of the delayed layer-count
    // update and returns the transparent identity row instead of this value.
    require_layer_stack_scalar(result.r_te_real, -0.35482694F, "stream-affinity r_te_real");
    require_layer_stack_scalar(result.t_te_real, -0.55237210F, "stream-affinity t_te_real");
}

} // namespace

int main() {
    try {
        require(at::cuda::is_available(), "CUDA is required for the typed integration tests");
        std::cout << "[RUN] test_scene_and_intersection_lockstep" << std::endl;
        test_scene_and_intersection_lockstep();
        std::cout << "[RUN] test_empty_and_stream_contracts" << std::endl;
        test_empty_and_stream_contracts();
        std::cout << "[RUN] test_visibility_trace_and_face_normal_lockstep" << std::endl;
        test_visibility_trace_and_face_normal_lockstep();
        std::cout << "[RUN] test_reflection_accumulation_and_epc_lockstep" << std::endl;
        test_reflection_accumulation_and_epc_lockstep();
        std::cout << "[RUN] test_error_and_lifecycle_contracts" << std::endl;
        test_error_and_lifecycle_contracts();
        std::cout << "[RUN] test_diffraction_paths_lockstep" << std::endl;
        test_diffraction_paths_lockstep();
        std::cout << "[RUN] test_diffraction_accumulation_lockstep" << std::endl;
        test_diffraction_accumulation_lockstep();
        std::cout << "[RUN] test_coherent_diffraction_lockstep" << std::endl;
        test_coherent_diffraction_lockstep();
        std::cout << "[RUN] test_layer_stack_empty_and_contracts" << std::endl;
        test_layer_stack_empty_and_contracts();
        std::cout << "[RUN] test_layer_stack_nonempty_ad_and_stream" << std::endl;
        test_layer_stack_nonempty_ad_and_stream();
        std::cout << "[RUN] test_layer_stack_lossy_primal_and_negative_contracts" << std::endl;
        test_layer_stack_lossy_primal_and_negative_contracts();
        std::cout << "[RUN] test_layer_stack_nondefault_stream_dependency" << std::endl;
        test_layer_stack_nondefault_stream_dependency();
        std::cout << "rayd::torch integration v2 direct contracts passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "rayd::torch integration v2 direct contract failure: " << error.what() << '\n';
        return 1;
    }
}
