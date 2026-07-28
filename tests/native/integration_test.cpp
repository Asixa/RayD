#include <rayd/integration/torch.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

static_assert(rayd::torch::kIntegrationApiVersion == 7);
static_assert(!rayd::torch::kIntegrationHeaderIdentity.empty());
static_assert(rayd::torch::kDiffractionTxAxialEdgeFractionBits[0] == 0x3ca3d70au);
static_assert(rayd::torch::kDiffractionTxAxialEdgeFractionBits[1] == 0x3eaaaaabu);
static_assert(rayd::torch::kDiffractionTxAxialEdgeFractionBits[2] == 0x3f2aaaabu);
static_assert(rayd::torch::kDiffractionTxAxialEdgeFractionBits[3] == 0x3f7ae148u);

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

void require_tensor_contract(
    const at::Tensor &tensor,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    const at::Device &device,
    const std::string &name) {
    require(tensor.defined(), name + ": tensor is undefined");
    require(tensor.sizes().equals(sizes), name + ": shape differs");
    require(tensor.scalar_type() == dtype, name + ": dtype differs");
    require(tensor.device() == device, name + ": device differs");
}

void require_finite(const at::Tensor &tensor, const std::string &name) {
    require(
        at::isfinite(tensor).all().item<bool>(),
        name + ": contains a non-finite value");
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

EmptyDiffractionFixture make_one_diffraction_fixture(int device_index = 0) {
    const auto float_options = at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(at::Device(at::kCUDA, device_index));
    const auto int_options = at::TensorOptions()
                                 .dtype(at::kInt)
                                 .device(at::Device(at::kCUDA, device_index));
    const auto bool_options = at::TensorOptions()
                                  .dtype(at::kBool)
                                  .device(at::Device(at::kCUDA, device_index));
    const at::Tensor tx_pos =
        at::tensor({0.0F, -1.0F, 0.25F}, float_options).reshape({1, 3});
    const at::Tensor zeros = at::zeros({1, 3}, float_options);
    rayd::torch::DiffractionState state = {
        at::tensor({0}, int_options),
        at::tensor({0.0F, 0.0F, 0.0F}, float_options).reshape({1, 3}),
        at::tensor({1.0F, 0.0F, 0.0F}, float_options).reshape({1, 3}),
        at::tensor({-1.0F}, float_options),
        at::tensor({1.0F}, float_options),
        at::tensor({0.0F, 0.0F, 1.0F}, float_options).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, -1.0F}, float_options).reshape({1, 3}),
        at::tensor({0}, int_options),
        at::tensor({0}, int_options),
        at::tensor({static_cast<float>(3.14159265358979323846)}, float_options),
        tx_pos,
        at::ones({1}, float_options),
        zeros,
        zeros,
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
        4,
        4,
        0.25,
    };
    return {
        std::move(state),
        std::move(material),
        grid,
        at::ones({1}, bool_options),
        tx_pos,
        at::tensor({1.0F, 0.0F, 0.0F}, float_options).reshape({1, 3}),
        at::tensor({0.0F, 1.0F, 0.25F}, float_options).reshape({1, 3}),
        at::tensor({0}, int_options),
        at::ones({1}, float_options),
    };
}

void test_scene_and_intersection_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    const auto device = mesh.vertices.device();
    const auto floats = mesh.vertices.options();
    const auto ints = mesh.faces.options();

    require(scene.valid(), "typed scene is not valid");
    require(scene.device_index() == 0, "typed scene reports the wrong device");

    const auto edges = rayd::torch::scene_edge_records(scene);
    require_tensor_exact(edges.global_vertices, mesh.vertices, "scene global vertices");
    require_tensor_exact(edges.global_faces, mesh.faces, "scene global faces");
    require_tensor_exact(edges.tri_fn_x, at::zeros({1}, floats), "scene face normal x");
    require_tensor_exact(edges.tri_fn_y, at::zeros({1}, floats), "scene face normal y");
    require_tensor_exact(edges.tri_fn_z, at::ones({1}, floats), "scene face normal z");
    for (const auto &entry : {
             edges.edge_v0,
             edges.edge_v1,
             edges.edge_face0_global,
             edges.edge_face1_global,
             edges.edge_shape_id,
             edges.edge_local_id,
             edges.edge_opposite})
        require_tensor_contract(entry, {3}, at::kInt, device, "scene edge record");
    require(
        at::all(edges.edge_shape_id == 0).item<bool>(),
        "single-mesh edge records must use shape id zero");

    auto rays = one_hit_ray();
    rays.ray_tmax = at::full({1}, 2.0F, floats);
    rays.active = at::ones({1}, floats.dtype(at::kBool));
    const auto hit = rayd::torch::intersect_forward(scene, rays, 7);
    require_tensor_exact(hit.t, at::ones({1}, floats), "intersect t");
    require_tensor_exact(
        hit.p,
        at::tensor({0.25F, 0.25F, 0.0F}, floats).reshape({1, 3}),
        "intersect point");
    require_tensor_contract(hit.n, {1, 3}, at::kFloat, device, "intersect normal");
    require_tensor_contract(hit.geo_n, {1, 3}, at::kFloat, device, "intersect geometric normal");
    require_tensor_contract(hit.uv, {1, 2}, at::kFloat, device, "intersect uv");
    require_tensor_contract(hit.barycentric, {1, 3}, at::kFloat, device, "intersect barycentric");
    for (const auto &entry : {
             hit.shape_id, hit.prim_id, hit.local_prim_id, hit.global_prim_id}) {
        require_tensor_contract(entry, {1}, at::kInt, device, "intersect id");
        require(entry.item<int>() == 0, "single-triangle intersect id must be zero");
    }

    rayd::torch::IntersectBackwardRequest backward;
    backward.rays = rays;
    backward.tape_prim_id = hit.global_prim_id;
    backward.tape_barycentric = hit.barycentric;
    backward.grad_t = at::ones_like(hit.t);
    backward.need_grad_vertices = true;
    backward.need_grad_ray_o = true;
    backward.need_grad_ray_d = true;
    backward.need_grad_ray_tmax = true;
    const auto gradients = rayd::torch::intersect_backward(scene, backward);
    require_tensor_contract(
        gradients.grad_vertices, {3, 3}, at::kFloat, device, "intersect vertex gradient");
    require_tensor_exact(
        gradients.grad_ray_o,
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        "intersect ray-origin gradient");
    require_tensor_exact(
        gradients.grad_ray_d,
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        "intersect ray-direction gradient");
    require_tensor_exact(
        gradients.grad_ray_tmax,
        at::zeros({1}, floats),
        "intersect ray-tmax gradient");
    require(
        at::allclose(
            gradients.grad_vertices.sum(0),
            at::tensor({0.0F, 0.0F, 1.0F}, floats),
            0.0,
            0.0),
        "intersect vertex gradients must translate the hit plane exactly");

    rayd::torch::IntersectJvpRequest jvp;
    jvp.ray_o = rays.ray_o;
    jvp.ray_d = rays.ray_d;
    jvp.active = rays.active;
    jvp.tape_prim_id = hit.global_prim_id;
    jvp.tape_barycentric = hit.barycentric;
    jvp.tangent_vertices = at::zeros_like(mesh.vertices);
    jvp.tangent_ray_o = at::ones_like(rays.ray_o);
    jvp.tangent_ray_d = at::zeros_like(rays.ray_d);
    jvp.flags = 7;
    const auto tangents = rayd::torch::intersect_jvp(scene, jvp);
    require_tensor_exact(
        tangents.tangent_t, at::full({1}, -1.0F, floats), "intersect tangent t");
    require_tensor_exact(
        tangents.tangent_p,
        at::tensor({1.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        "intersect tangent point");
    require_tensor_contract(
        tangents.tangent_n, {1, 3}, at::kFloat, device, "intersect tangent normal");
    require_tensor_contract(
        tangents.tangent_geo_n, {1, 3}, at::kFloat, device, "intersect tangent geo-normal");
    require_tensor_contract(
        tangents.tangent_uv, {1, 2}, at::kFloat, device, "intersect tangent uv");
    require_tensor_contract(
        tangents.tangent_barycentric,
        {1, 3},
        at::kFloat,
        device,
        "intersect tangent barycentric");
    require_finite(tangents.tangent_n, "intersect tangent normal");
    require_finite(tangents.tangent_geo_n, "intersect tangent geo-normal");
}

void test_empty_and_stream_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    const auto rays = empty_rays_with_present_empty_mask();

    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        const auto typed = rayd::torch::intersect_forward(scene, rays, 7);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
            "typed intersection changed the caller's active CUDA stream");
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
            require(typed_values[index].size(0) == 0, "empty typed output must keep zero rows");
        }
    }
}

void test_visibility_trace_and_face_normal_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    const auto device = mesh.vertices.device();
    const auto floats = mesh.vertices.options();
    const auto bools = floats.dtype(at::kBool);

    rayd::torch::VisibilityRequest visibility_request = {
        at::tensor({0.25F, 0.25F, -1.0F}, floats).reshape({1, 3}),
        at::tensor({0.25F, 0.25F, 1.0F}, floats).reshape({1, 3}),
        at::ones({1}, bools),
    };
    const auto visibility =
        rayd::torch::visibility_forward(scene, visibility_request);
    require_tensor_exact(
        visibility.visible, at::zeros({1}, bools), "blocked segment visibility");
    require_tensor_exact(
        visibility.blocker_prim,
        at::zeros({1}, mesh.faces.options()),
        "visibility blocker primitive");
    require_tensor_exact(
        visibility.tape_t,
        at::full({1}, std::numeric_limits<float>::infinity(), floats),
        "visibility tape sentinel");

    auto rays = one_hit_ray();
    rays.ray_tmax = at::full({1}, 2.0F, floats);
    rays.active = at::ones({1}, bools);
    const rayd::torch::ReflectionTraceRequest trace_request = {rays, 1};
    const auto trace =
        rayd::torch::trace_reflections_forward(scene, trace_request);
    require_tensor_exact(
        trace.valid, at::ones({1, 1}, bools), "reflection trace validity");
    require_tensor_exact(
        trace.t, at::ones({1, 1}, floats), "reflection trace distance");
    require_tensor_exact(
        trace.prim_ids,
        at::zeros({1, 1}, mesh.faces.options()),
        "reflection trace primitive");

    const auto tape =
        rayd::torch::trace_reflections_forward_tape(scene, trace_request);
    require_tensor_exact(tape.valid, trace.valid, "reflection tape validity");
    require_tensor_exact(tape.t, trace.t, "reflection tape distance");
    require_tensor_exact(tape.prim_ids, trace.prim_ids, "reflection tape primitive");
    require(
        tape.prim_ids.unsafeGetTensorImpl() == tape.tape_prim_id.unsafeGetTensorImpl(),
        "reflection tape must preserve prim-id tensor aliasing");
    require_tensor_contract(
        tape.image_sources, {1, 1, 3}, at::kFloat, device, "reflection image sources");
    require_tensor_contract(
        tape.tape_barycentric, {1, 1, 3}, at::kFloat, device, "reflection barycentric tape");
    require_tensor_contract(
        tape.tape_hit_points, {1, 1, 3}, at::kFloat, device, "reflection hit-point tape");
    require_tensor_contract(
        tape.tape_normals, {1, 1, 3}, at::kFloat, device, "reflection normal tape");
    require_tensor_exact(tape.active_ctx, *rays.active, "reflection active tape");
    require_finite(tape.image_sources, "reflection image sources");
    require_finite(tape.tape_hit_points, "reflection hit-point tape");
    require_finite(tape.tape_normals, "reflection normal tape");

    rayd::torch::ReflectionTraceBackwardRequest backward;
    backward.rays = rays;
    backward.tape_prim_id = tape.tape_prim_id;
    backward.tape_barycentric = tape.tape_barycentric;
    backward.tape_hit_points = tape.tape_hit_points;
    backward.tape_normals = tape.tape_normals;
    backward.image_sources = tape.image_sources;
    backward.grad_t = at::ones_like(tape.t);
    backward.grad_image_sources = at::zeros_like(tape.image_sources);
    const auto gradients =
        rayd::torch::trace_reflections_backward(scene, backward);
    require_tensor_contract(
        gradients.grad_vertices, {3, 3}, at::kFloat, device, "reflection vertex gradient");
    require_tensor_exact(
        gradients.grad_ray_o,
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        "reflection ray-origin gradient");
    require_tensor_exact(
        gradients.grad_ray_d,
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        "reflection ray-direction gradient");
    require_tensor_exact(
        gradients.grad_ray_tmax,
        at::zeros({1}, floats),
        "reflection ray-tmax gradient");

    rayd::torch::ReflectionTraceJvpRequest jvp;
    jvp.ray_o = rays.ray_o;
    jvp.ray_d = rays.ray_d;
    jvp.active = rays.active;
    jvp.tape_prim_id = tape.tape_prim_id;
    jvp.tape_barycentric = tape.tape_barycentric;
    jvp.tape_hit_points = tape.tape_hit_points;
    jvp.tape_normals = tape.tape_normals;
    jvp.tangent_vertices = at::zeros_like(mesh.vertices);
    jvp.tangent_ray_o = at::ones_like(rays.ray_o);
    jvp.tangent_ray_d = at::zeros_like(rays.ray_d);
    jvp.image_sources = tape.image_sources;
    const auto tangents = rayd::torch::trace_reflections_jvp(scene, jvp);
    require_tensor_exact(
        tangents.tangent_t,
        at::full({1, 1}, -1.0F, floats),
        "reflection trace tangent distance");
    require_tensor_contract(
        tangents.tangent_image_sources,
        {1, 1, 3},
        at::kFloat,
        device,
        "reflection image-source tangent");
    require_finite(
        tangents.tangent_image_sources, "reflection image-source tangent");

    const auto edges = rayd::torch::scene_edge_records(scene);
    const at::Tensor grad_face_normals = at::ones({1, 3}, floats);
    const auto face_gradients =
        rayd::torch::scene_face_normals_backward(scene, grad_face_normals);
    require_tensor_contract(
        face_gradients, {3, 3}, at::kFloat, device, "face-normal vertex gradient");
    require_finite(face_gradients, "face-normal vertex gradient");
    require(
        at::allclose(
            face_gradients.sum(0), at::zeros({3}, floats), 0.0, 1.0e-6),
        "face-normal gradients must be translation invariant");

    const at::Tensor rigid_translation = at::ones_like(edges.global_vertices);
    const auto face_tangent =
        rayd::torch::scene_face_normals_jvp(scene, rigid_translation);
    require_tensor_exact(
        face_tangent,
        at::zeros({1, 3}, floats),
        "face-normal rigid-translation tangent");
}

void test_axial_edge_visibility_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    const auto floats = mesh.vertices.options();
    const auto bools = floats.dtype(at::kBool);

    rayd::torch::AxialEdgeVisibilityRequest request = {
        at::tensor({0.25F, 0.25F, -1.0F}, floats),
        at::tensor(
            {
                -2.0F, 0.25F, 1.0F,
                0.25F, 0.25F, 1.0F,
                1.0F, 0.25F, 1.0F,
            },
            floats).reshape({3, 3}),
        at::tensor(
            {
                0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F,
                1.0F, 0.0F, 0.0F,
            },
            floats).reshape({3, 3}),
        at::zeros({3}, floats),
        at::ones({3}, floats),
        std::nullopt,
        {},
    };

    const auto result =
        rayd::torch::axial_edge_visibility_forward(scene, request);
    require_tensor_exact(
        result.any_visible,
        at::tensor({1, 0, 1}, mesh.faces.options()).to(at::kBool),
        "axial-edge visibility all/partial mask");
    require(result.any_visible.is_contiguous(), "axial-edge output must be contiguous");

    auto active_request = request;
    active_request.active =
        at::tensor({1, 1, 0}, mesh.faces.options()).to(at::kBool);
    require_tensor_exact(
        rayd::torch::axial_edge_visibility_forward(scene, active_request).any_visible,
        at::tensor({1, 0, 0}, mesh.faces.options()).to(at::kBool),
        "axial-edge active mask");

    rayd::torch::AxialEdgeVisibilityRequest empty_request = {
        request.tx,
        at::empty({0, 3}, floats),
        at::empty({0, 3}, floats),
        at::empty({0}, floats),
        at::empty({0}, floats),
        at::empty({0}, bools),
        {},
    };
    const auto empty =
        rayd::torch::axial_edge_visibility_forward(scene, empty_request);
    require_tensor_contract(
        empty.any_visible,
        {0},
        at::kBool,
        mesh.vertices.device(),
        "empty axial-edge visibility");
    require(empty.any_visible.is_contiguous(), "empty axial-edge output must be contiguous");

    auto nonfinite_request = request;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    nonfinite_request.edge_position = at::tensor(
        {
            nan, 0.25F, 1.0F,
            -2.0F, 0.25F, 1.0F,
            -2.0F, 0.25F, 1.0F,
        },
        floats).reshape({3, 3});
    nonfinite_request.edge_direction = at::tensor(
        {
            0.0F, 0.0F, 0.0F,
            inf, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F,
        },
        floats).reshape({3, 3});
    nonfinite_request.edge_t_max = at::tensor({1.0F, 1.0F, inf}, floats);
    require_tensor_exact(
        rayd::torch::axial_edge_visibility_forward(scene, nonfinite_request).any_visible,
        at::zeros({3}, bools),
        "axial-edge nonfinite lanes");

    auto nonfinite_tx = request;
    nonfinite_tx.tx = at::tensor({nan, 0.25F, -1.0F}, floats);
    require_tensor_exact(
        rayd::torch::axial_edge_visibility_forward(scene, nonfinite_tx).any_visible,
        at::zeros({3}, bools),
        "axial-edge nonfinite transmitter");

    const std::array<bool, 4> fraction_visibility = {false, true, true, true};
    for (std::size_t fraction_index = 0;
         fraction_index < fraction_visibility.size();
         ++fraction_index) {
        auto fraction_request = request;
        fraction_request.edge_position = request.edge_position.narrow(0, 2, 1);
        fraction_request.edge_direction = request.edge_direction.narrow(0, 2, 1);
        fraction_request.edge_t_min = request.edge_t_min.narrow(0, 2, 1);
        fraction_request.edge_t_max = request.edge_t_max.narrow(0, 2, 1);
        fraction_request.config.sample_fraction_bits.fill(
            rayd::torch::kDiffractionTxAxialEdgeFractionBits[fraction_index]);
        require_tensor_exact(
            rayd::torch::axial_edge_visibility_forward(scene, fraction_request)
                .any_visible,
            fraction_visibility[fraction_index]
                ? at::ones({1}, bools)
                : at::zeros({1}, bools),
            "axial-edge fraction boundary");
    }

    auto bad_tx_view = request;
    bad_tx_view.tx = at::zeros({3, 2}, floats).select(1, 0);
    require(!bad_tx_view.tx.is_contiguous(), "tx noncontiguous fixture is contiguous");
    require_throws(
        [&] { (void)rayd::torch::axial_edge_visibility_forward(scene, bad_tx_view); },
        "axial-edge noncontiguous tx must fail loudly");

    auto bad_edge_view = request;
    bad_edge_view.edge_position = at::zeros({3, 3, 2}, floats).select(2, 0);
    require(
        !bad_edge_view.edge_position.is_contiguous(),
        "edge noncontiguous fixture is contiguous");
    require_throws(
        [&] { (void)rayd::torch::axial_edge_visibility_forward(scene, bad_edge_view); },
        "axial-edge noncontiguous edge tensor must fail loudly");

    auto bad_scalar_view = request;
    bad_scalar_view.edge_t_min = at::zeros({3, 2}, floats).select(1, 0);
    require(
        !bad_scalar_view.edge_t_min.is_contiguous(),
        "scalar noncontiguous fixture is contiguous");
    require_throws(
        [&] { (void)rayd::torch::axial_edge_visibility_forward(scene, bad_scalar_view); },
        "axial-edge noncontiguous scalar tensor must fail loudly");

    auto bad_active_view = request;
    bad_active_view.active = at::zeros({3, 2}, bools).select(1, 0);
    require(
        !bad_active_view.active->is_contiguous(),
        "active noncontiguous fixture is contiguous");
    require_throws(
        [&] { (void)rayd::torch::axial_edge_visibility_forward(scene, bad_active_view); },
        "axial-edge noncontiguous active tensor must fail loudly");

    auto bad_fraction = request;
    bad_fraction.config.sample_fraction_bits[0] = 0x7f800000u;
    require_throws(
        [&] { (void)rayd::torch::axial_edge_visibility_forward(scene, bad_fraction); },
        "axial-edge nonfinite fraction must fail loudly");

    auto cpu_tx = request;
    cpu_tx.tx = cpu_tx.tx.cpu();
    require_throws(
        [&] { (void)rayd::torch::axial_edge_visibility_forward(scene, cpu_tx); },
        "axial-edge CPU tx must fail loudly");

    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        const auto streamed =
            rayd::torch::axial_edge_visibility_forward(scene, request);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
            "axial-edge visibility changed the caller's CUDA stream");
        stream.synchronize();
        require_tensor_exact(
            streamed.any_visible,
            at::tensor({1, 0, 1}, mesh.faces.options()).to(at::kBool),
            "axial-edge current-stream result");
    }
}

float test_float_from_bits(std::uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::array<float, 3> exact_axial_sample_host(
    const std::array<float, 3> &position,
    const std::array<float, 3> &direction,
    float t_min,
    float t_max,
    float fraction) {
    volatile float span = t_max - t_min;
    volatile float scaled_fraction = fraction * span;
    volatile float t = t_min + scaled_fraction;
    std::array<float, 3> point{};
    for (std::size_t component = 0; component < point.size(); ++component) {
        volatile float scaled_direction = t * direction[component];
        volatile float value = position[component] + scaled_direction;
        point[component] = value;
    }
    return point;
}

void test_axial_edge_visibility_legacy_parity() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    constexpr std::size_t random_count = 257;
    constexpr std::size_t boundary_count = 3;
    constexpr std::size_t state_count = boundary_count + random_count;

    std::vector<std::array<float, 3>> positions(state_count);
    std::vector<std::array<float, 3>> directions(state_count);
    std::vector<float> t_min(state_count);
    std::vector<float> t_max(state_count);
    positions[0] = {-2.0F, 0.25F, 1.0F};
    positions[1] = {0.25F, 0.25F, 1.0F};
    positions[2] = {1.0F, 0.25F, 1.0F};
    directions[0] = {0.0F, 0.0F, 0.0F};
    directions[1] = {0.0F, 0.0F, 0.0F};
    directions[2] = {1.0F, 0.0F, 0.0F};
    t_min[0] = t_min[1] = t_min[2] = 0.0F;
    t_max[0] = t_max[1] = t_max[2] = 1.0F;

    std::mt19937 generator(0x29A81u);
    std::uniform_real_distribution<float> position_distribution(-2.0F, 2.0F);
    std::uniform_real_distribution<float> direction_distribution(-1.0F, 1.0F);
    std::uniform_real_distribution<float> minimum_distribution(-0.5F, 0.5F);
    std::uniform_real_distribution<float> span_distribution(0.01F, 2.0F);
    for (std::size_t row = boundary_count; row < state_count; ++row) {
        positions[row] = {
            position_distribution(generator),
            position_distribution(generator),
            position_distribution(generator) + 1.0F,
        };
        directions[row] = {
            direction_distribution(generator),
            direction_distribution(generator),
            direction_distribution(generator),
        };
        t_min[row] = minimum_distribution(generator);
        t_max[row] = t_min[row] + span_distribution(generator);
    }

    std::vector<float> position_aos;
    std::vector<float> direction_aos;
    position_aos.reserve(state_count * 3);
    direction_aos.reserve(state_count * 3);
    for (std::size_t row = 0; row < state_count; ++row) {
        position_aos.insert(position_aos.end(), positions[row].begin(), positions[row].end());
        direction_aos.insert(direction_aos.end(), directions[row].begin(), directions[row].end());
    }

    const auto cpu_floats = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const auto device = mesh.vertices.device();
    const auto to_device_matrix = [&](std::vector<float> &values) {
        return at::from_blob(
                   values.data(),
                   {static_cast<int64_t>(state_count), 3},
                   cpu_floats)
            .clone()
            .to(device);
    };
    const auto to_device_vector = [&](std::vector<float> &values) {
        return at::from_blob(
                   values.data(),
                   {static_cast<int64_t>(state_count)},
                   cpu_floats)
            .clone()
            .to(device);
    };
    const at::Tensor tx =
        at::tensor({0.25F, 0.25F, -1.0F}, cpu_floats).to(device);
    rayd::torch::AxialEdgeVisibilityRequest request = {
        tx,
        to_device_matrix(position_aos),
        to_device_matrix(direction_aos),
        to_device_vector(t_min),
        to_device_vector(t_max),
        std::nullopt,
        {},
    };
    const at::Tensor single_launch =
        rayd::torch::axial_edge_visibility_forward(scene, request).any_visible;

    std::vector<float> starts_aos;
    starts_aos.reserve(state_count * 3);
    for (std::size_t row = 0; row < state_count; ++row)
        starts_aos.insert(starts_aos.end(), {0.25F, 0.25F, -1.0F});
    const at::Tensor starts = to_device_matrix(starts_aos);
    at::Tensor four_launches = at::zeros(
        {static_cast<int64_t>(state_count)}, mesh.vertices.options().dtype(at::kBool));
    for (const std::uint32_t fraction_bits :
         rayd::torch::kDiffractionTxAxialEdgeFractionBits) {
        const float fraction = test_float_from_bits(fraction_bits);
        std::vector<float> ends_aos;
        ends_aos.reserve(state_count * 3);
        for (std::size_t row = 0; row < state_count; ++row) {
            const auto point = exact_axial_sample_host(
                positions[row], directions[row], t_min[row], t_max[row], fraction);
            ends_aos.insert(ends_aos.end(), point.begin(), point.end());
        }
        const auto visibility = rayd::torch::visibility_forward(
            scene, {starts, to_device_matrix(ends_aos), std::nullopt});
        four_launches.bitwise_or_(visibility.visible);
    }
    require_tensor_exact(
        single_launch,
        four_launches,
        "single axial launch versus four segment launches boundary/random parity");
}

void test_reflection_accumulation_and_epc_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    const auto device = mesh.vertices.device();
    const auto floats = mesh.vertices.options();
    const auto ints = mesh.faces.options();
    const auto bools = floats.dtype(at::kBool);

    const rayd::torch::MaterialPayload material = {
        at::full({1}, 4.0F, floats),
        at::zeros({1}, floats),
        at::ones({1}, floats),
        at::ones({1}, floats),
        at::ones({1}, bools),
    };
    const rayd::torch::Grid2D grid = {
        2,
        0.5,
        -1.0,
        1.0,
        -1.0,
        1.0,
        4,
        4,
        0.25,
    };
    rayd::torch::RayBatch rays = {
        at::tensor({0.25F, 0.25F, 1.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        at::full({1}, 2.0F, floats),
        at::ones({1}, bools),
    };
    rayd::torch::ReflectionAccumulationConfig accumulation = {
        rays,
        rays.ray_o,
        at::tensor({1.0F, 0.0F, 0.0F}, floats).reshape({1, 3}),
        material,
        1,
        grid,
        1.0,
        1.0,
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
    const auto accumulated =
        rayd::torch::reflection_accumulation_forward(scene, accumulation);
    for (const auto &field : {
             accumulated.power,
             accumulated.field_x_re,
             accumulated.field_x_im,
             accumulated.field_y_re,
             accumulated.field_y_im,
             accumulated.field_z_re,
             accumulated.field_z_im}) {
        require_tensor_contract(
            field, {4, 4}, at::kFloat, device, "reflection accumulation grid");
        require_finite(field, "reflection accumulation grid");
    }
    require(
        at::all(accumulated.power >= 0).item<bool>(),
        "reflection accumulated power must be non-negative");
    require_tensor_contract(
        accumulated.reflection_count,
        {1},
        at::kInt,
        device,
        "reflection accumulation count");
    const int reflection_count = accumulated.reflection_count.item<int>();
    require(
        reflection_count > 0,
        "nonempty reflection ray produced no reflected-grid hit; count=" +
            std::to_string(reflection_count));
    require_tensor_exact(
        accumulated.wedge_count,
        at::zeros({1}, ints),
        "disabled wedge count");
    for (const auto &wedge : {
             accumulated.wedge_ray_index,
             accumulated.wedge_hit,
             accumulated.wedge_normal,
             accumulated.wedge_prim_id,
             accumulated.wedge_direction,
             accumulated.wedge_source,
             accumulated.wedge_source_power,
             accumulated.wedge_initial_direction,
             accumulated.wedge_bounce_depth})
        require(wedge.defined() && wedge.numel() == 0, "disabled wedge output must be defined-empty");

    rayd::torch::ReflectionEpcRequest epc = {
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        at::ones({1}, bools),
        at::zeros({1, 1}, ints),
        at::zeros({1, 1, 3}, floats),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 1, 3}),
        at::zeros({1}, ints),
        at::ones({1}, ints),
        at::zeros({1}, ints),
        1,
        0,
        1.0e-5,
    };
    const auto path = rayd::torch::reflection_epc_paths_forward(scene, epc);
    require_tensor_exact(path.valid, at::ones({1}, bools), "reflection EPC validity");
    require_tensor_exact(
        path.path_length, at::full({1}, 2.0F, floats), "reflection EPC path length");
    require_tensor_exact(
        path.resolved_prim_ids, at::zeros({1, 1}, ints), "reflection EPC primitive");
    require_tensor_exact(
        path.surface_group_ids, at::zeros({1, 1}, ints), "reflection EPC group");
    require_tensor_exact(
        path.hit_positions, at::zeros({1, 1, 3}, floats), "reflection EPC hit");
    require_tensor_exact(
        path.normals,
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 1, 3}),
        "reflection EPC normal");

    rayd::torch::ReflectionEpcBackwardRequest backward;
    backward.source = epc.source;
    backward.receiver = epc.receiver;
    backward.sequence = epc.expected_prim_ids;
    backward.plane_points = epc.direct_plane_points;
    backward.plane_normals = epc.direct_plane_normals;
    backward.valid = path.valid;
    backward.bounce_count = at::ones({1}, ints);
    backward.grad_path_length = at::ones({1}, floats);
    backward.need_grad_vertices = true;
    backward.need_grad_source = true;
    backward.need_grad_receiver = true;
    const auto gradients =
        rayd::torch::reflection_epc_paths_backward(scene, backward);
    require_tensor_contract(
        gradients.grad_vertices, {3, 3}, at::kFloat, device, "reflection EPC vertex gradient");
    require_tensor_exact(
        gradients.grad_source,
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        "reflection EPC source gradient");
    require_tensor_exact(
        gradients.grad_receiver,
        at::tensor({0.0F, 0.0F, -1.0F}, floats).reshape({1, 3}),
        "reflection EPC receiver gradient");
    require(
        at::allclose(
            gradients.grad_vertices.sum(0),
            at::tensor({0.0F, 0.0F, 2.0F}, floats),
            0.0,
            1.0e-6),
        "reflection EPC plane gradient must balance endpoint gradients");

    rayd::torch::ReflectionEpcJvpRequest jvp;
    jvp.source = epc.source;
    jvp.receiver = epc.receiver;
    jvp.sequence = epc.expected_prim_ids;
    jvp.plane_points = epc.direct_plane_points;
    jvp.plane_normals = epc.direct_plane_normals;
    jvp.valid = path.valid;
    jvp.bounce_count = at::ones({1}, ints);
    jvp.tangent_vertices = at::ones_like(mesh.vertices);
    jvp.tangent_source = at::ones_like(epc.source);
    jvp.tangent_receiver = at::ones_like(epc.receiver);
    const auto tangents = rayd::torch::reflection_epc_paths_jvp(scene, jvp);
    require_tensor_exact(
        tangents.tangent_points,
        at::ones({1, 1, 3}, floats),
        "reflection EPC rigid-translation point tangent");
    require_tensor_exact(
        tangents.tangent_normals,
        at::zeros({1, 1, 3}, floats),
        "reflection EPC rigid-translation normal tangent");
    require_tensor_exact(
        tangents.tangent_path_length,
        at::zeros({1}, floats),
        "reflection EPC rigid-translation length tangent");
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

void test_diffraction_paths_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    auto fixture = make_one_diffraction_fixture();
    const auto device = mesh.vertices.device();
    const auto floats = mesh.vertices.options();
    const auto ints = mesh.faces.options();
    const auto bools = floats.dtype(at::kBool);

    rayd::torch::DiffractionPathConfig config = {
        fixture.tx_pos,
        fixture.tx_pol,
        fixture.rx_pos,
        fixture.active,
        fixture.state,
        fixture.material,
        1,
        8,
        1.0,
        0.0,
    };
    const auto paths =
        rayd::torch::diffraction_paths_order1_forward(scene, config);
    require_tensor_contract(paths.count, {1}, at::kInt, device, "diffraction path count");
    require_tensor_contract(paths.valid, {8}, at::kBool, device, "diffraction path valid");
    for (const auto &entry : {
             paths.tx_id,
             paths.rx_id,
             paths.order,
             paths.edge0,
             paths.edge1,
             paths.edge2})
        require_tensor_contract(entry, {8}, at::kInt, device, "diffraction path index");
    for (const auto &entry : {
             paths.delay,
             paths.field_x_re,
             paths.field_x_im,
             paths.field_y_re,
             paths.field_y_im,
             paths.field_z_re,
             paths.field_z_im}) {
        require_tensor_contract(entry, {8}, at::kFloat, device, "diffraction path scalar");
        require_finite(entry, "diffraction path scalar");
    }
    for (const auto &entry : {paths.p0, paths.p1, paths.p2}) {
        require_tensor_contract(entry, {8, 3}, at::kFloat, device, "diffraction path point");
        require_finite(entry, "diffraction path point");
    }
    const int count = paths.count.item<int>();
    require(count >= 0 && count <= 8, "diffraction path count is outside capacity");
    require(
        paths.valid.sum().item<int64_t>() == count,
        "diffraction valid rows must equal the exported count");
    if (count > 0) {
        const auto rows = at::arange(count, ints.dtype(at::kLong));
        require(
            at::all(paths.tx_id.index_select(0, rows) == 0).item<bool>() &&
                at::all(paths.rx_id.index_select(0, rows) == 0).item<bool>() &&
                at::all(paths.order.index_select(0, rows) == 1).item<bool>() &&
                at::all(paths.edge0.index_select(0, rows) == 0).item<bool>(),
            "single-state order-1 path identity differs");
        require(
            at::all(paths.delay.index_select(0, rows) >= 0).item<bool>(),
            "diffraction path delay must be non-negative");
    }

    auto source_lane = config;
    source_lane.active = at::zeros({2}, bools);
    source_lane.active.select(0, 1).fill_(true);
    source_lane.state.edge_index = fixture.state.edge_index.repeat({2});
    source_lane.state.edge_pos = fixture.state.edge_pos.repeat({2, 1});
    source_lane.state.edge_dir = fixture.state.edge_dir.repeat({2, 1});
    source_lane.state.edge_t_min = fixture.state.edge_t_min.repeat({2});
    source_lane.state.edge_t_max = fixture.state.edge_t_max.repeat({2});
    source_lane.state.n0 = fixture.state.n0.repeat({2, 1});
    source_lane.state.n1 = fixture.state.n1.repeat({2, 1});
    source_lane.state.prim0 = fixture.state.prim0.repeat({2});
    source_lane.state.prim1 = fixture.state.prim1.repeat({2});
    source_lane.state.exterior_angle = fixture.state.exterior_angle.repeat({2});
    source_lane.state.src = fixture.state.src.repeat({2, 1});
    source_lane.state.src_power = fixture.state.src_power.repeat({2});
    source_lane.state_limit = 2;
    source_lane.capacity = 2;
    source_lane.layout = rayd::torch::DiffractionPathLayout::SourceLane;
    const auto lane_paths =
        rayd::torch::diffraction_paths_order1_forward(scene, source_lane);
    require(lane_paths.count.item<int>() == 1, "source-lane count must track valid paths");
    require(
        !lane_paths.valid.select(0, 0).item<bool>() &&
            lane_paths.valid.select(0, 1).item<bool>(),
        "source-lane export must preserve the pair/state lane index");
    require(
        lane_paths.tx_id.select(0, 1).item<int>() == 0 &&
            lane_paths.rx_id.select(0, 1).item<int>() == 0 &&
            lane_paths.edge0.select(0, 1).item<int>() == 0,
        "source-lane path identity differs");
    require(
        lane_paths.tx_id.select(0, 0).item<int>() == -1 &&
            lane_paths.rx_id.select(0, 0).item<int>() == -1 &&
            lane_paths.order.select(0, 0).item<int>() == 0 &&
            lane_paths.edge0.select(0, 0).item<int>() == -1 &&
            lane_paths.edge1.select(0, 0).item<int>() == -1 &&
            lane_paths.edge2.select(0, 0).item<int>() == -1,
        "source-lane inactive identity must remain canonical");
    for (const auto &entry : {
             lane_paths.delay,
             lane_paths.field_x_re,
             lane_paths.field_x_im,
             lane_paths.field_y_re,
             lane_paths.field_y_im,
             lane_paths.field_z_re,
             lane_paths.field_z_im}) {
        require(
            entry.select(0, 0).item<float>() == 0.0f,
            "source-lane inactive scalar payload must remain exactly zero");
    }
    for (const auto &entry : {lane_paths.p0, lane_paths.p1, lane_paths.p2}) {
        require(
            at::equal(entry.select(0, 0), at::zeros_like(entry.select(0, 0))),
            "source-lane inactive point payload must remain exactly zero");
    }

    auto compact_lane = source_lane;
    compact_lane.layout = rayd::torch::DiffractionPathLayout::Compact;
    const auto compact_paths =
        rayd::torch::diffraction_paths_order1_forward(scene, compact_lane);
    require(
        compact_paths.count.item<int>() == 1 && compact_paths.valid.select(0, 0).item<bool>(),
        "compact parity fixture must export one row");
    for (const auto &pair : {
             std::pair<at::Tensor, at::Tensor>{compact_paths.tx_id, lane_paths.tx_id},
             {compact_paths.rx_id, lane_paths.rx_id},
             {compact_paths.order, lane_paths.order},
             {compact_paths.edge0, lane_paths.edge0},
             {compact_paths.edge1, lane_paths.edge1},
             {compact_paths.edge2, lane_paths.edge2},
             {compact_paths.delay, lane_paths.delay},
             {compact_paths.field_x_re, lane_paths.field_x_re},
             {compact_paths.field_x_im, lane_paths.field_x_im},
             {compact_paths.field_y_re, lane_paths.field_y_re},
             {compact_paths.field_y_im, lane_paths.field_y_im},
             {compact_paths.field_z_re, lane_paths.field_z_re},
             {compact_paths.field_z_im, lane_paths.field_z_im},
             {compact_paths.p0, lane_paths.p0},
             {compact_paths.p1, lane_paths.p1},
             {compact_paths.p2, lane_paths.p2}}) {
        require(
            at::equal(pair.first.select(0, 0), pair.second.select(0, 1)),
            "compact and source-lane payloads must be bit-identical");
    }

    auto multi_lane = source_lane;
    multi_lane.tx_pos = fixture.tx_pos.repeat({2, 1});
    multi_lane.tx_pol = fixture.tx_pol.repeat({2, 1});
    multi_lane.rx_pos = fixture.rx_pos.repeat({2, 1});
    multi_lane.capacity = 8;
    const auto multi_paths =
        rayd::torch::diffraction_paths_order1_forward(scene, multi_lane);
    require(multi_paths.count.item<int>() == 4, "multi-pair source-lane count differs");
    const auto pair_lanes = multi_paths.valid.reshape({4, 2});
    require(
        !pair_lanes.select(1, 0).any().item<bool>() &&
            pair_lanes.select(1, 1).all().item<bool>(),
        "source-lane state must be the fastest-varying coordinate");
    const auto odd_rows = at::tensor({1, 3, 5, 7}, ints.dtype(at::kLong));
    require(
        at::equal(
            multi_paths.tx_id.index_select(0, odd_rows),
            at::tensor({0, 0, 1, 1}, ints)) &&
            at::equal(
                multi_paths.rx_id.index_select(0, odd_rows),
                at::tensor({0, 1, 0, 1}, ints)),
        "source-lane transmitter/receiver identity formula differs");

    auto invalid_layout = config;
    invalid_layout.layout = static_cast<rayd::torch::DiffractionPathLayout>(77);
    require_throws(
        [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, invalid_layout); },
        "invalid diffraction path layout must fail loudly");

    auto missing_active = config;
    missing_active.active = at::Tensor();
    require_throws(
        [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, missing_active); },
        "missing diffraction path active tensor must fail loudly");

    auto wrong_shape = config;
    wrong_shape.active = at::ones({2}, bools);
    require_throws(
        [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, wrong_shape); },
        "diffraction path active shape must equal state_limit");

    auto wrong_dtype = config;
    wrong_dtype.active = at::ones({1}, ints);
    require_throws(
        [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, wrong_dtype); },
        "diffraction path active dtype must be bool");

    auto noncontiguous_active = config;
    noncontiguous_active.active = at::zeros({2, 2}, bools).select(1, 0);
    require(
        !noncontiguous_active.active.is_contiguous(),
        "diffraction path noncontiguous active fixture is contiguous");
    require_throws(
        [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, noncontiguous_active); },
        "diffraction path active tensor must be contiguous");

    auto cpu_active = config;
    cpu_active.active = at::ones({1}, at::TensorOptions().dtype(at::kBool));
    require_throws(
        [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, cpu_active); },
        "diffraction path active tensor must be CUDA resident");

    const int device_count = at::cuda::device_count();
    if (device_count > 1) {
        auto other_device_active = config;
        other_device_active.active = at::ones(
            {1},
            at::TensorOptions().dtype(at::kBool).device(at::Device(at::kCUDA, 1)));
        require_throws(
            [&] { (void)rayd::torch::diffraction_paths_order1_forward(scene, other_device_active); },
            "cross-device diffraction path active tensor must fail loudly");
    }

    auto poison = config;
    poison.layout = rayd::torch::DiffractionPathLayout::SourceLane;
    poison.active = at::zeros({1}, bools);
    poison.rx_pos = at::full_like(poison.rx_pos, std::numeric_limits<float>::quiet_NaN());
    poison.state.edge_index = at::full_like(poison.state.edge_index, std::numeric_limits<int>::max());
    poison.state.edge_pos = at::full_like(poison.state.edge_pos, std::numeric_limits<float>::quiet_NaN());
    poison.state.edge_dir = at::full_like(poison.state.edge_dir, std::numeric_limits<float>::quiet_NaN());
    poison.state.edge_t_min = at::full_like(poison.state.edge_t_min, std::numeric_limits<float>::quiet_NaN());
    poison.state.edge_t_max = at::full_like(poison.state.edge_t_max, std::numeric_limits<float>::quiet_NaN());
    poison.state.n0 = at::full_like(poison.state.n0, std::numeric_limits<float>::quiet_NaN());
    poison.state.n1 = at::full_like(poison.state.n1, std::numeric_limits<float>::quiet_NaN());
    poison.state.prim0 = at::full_like(poison.state.prim0, std::numeric_limits<int>::max());
    poison.state.prim1 = at::full_like(poison.state.prim1, std::numeric_limits<int>::max());
    poison.state.exterior_angle = at::full_like(poison.state.exterior_angle, std::numeric_limits<float>::quiet_NaN());
    poison.state.src = at::full_like(poison.state.src, std::numeric_limits<float>::quiet_NaN());
    poison.state.src_power = at::full_like(poison.state.src_power, std::numeric_limits<float>::quiet_NaN());
    poison.material.eta_r = at::full_like(poison.material.eta_r, std::numeric_limits<float>::quiet_NaN());
    poison.material.sigma = at::full_like(poison.material.sigma, std::numeric_limits<float>::quiet_NaN());
    poison.material.mu_r = at::full_like(poison.material.mu_r, std::numeric_limits<float>::quiet_NaN());
    poison.material.gain = at::full_like(poison.material.gain, std::numeric_limits<float>::quiet_NaN());
    const auto poisoned = rayd::torch::diffraction_paths_order1_forward(scene, poison);
    require(poisoned.count.item<int>() == 0, "inactive poisoned diffraction row was exported");
    require(!poisoned.valid.any().item<bool>(), "inactive poisoned diffraction row became valid");
    for (const auto &entry : {
             poisoned.delay,
             poisoned.field_x_re,
             poisoned.field_x_im,
             poisoned.field_y_re,
             poisoned.field_y_im,
             poisoned.field_z_re,
             poisoned.field_z_im,
             poisoned.p0,
             poisoned.p1,
             poisoned.p2}) {
        require(
            at::equal(entry, at::zeros_like(entry)),
            "inactive poisoned diffraction payload must remain exactly zero");
    }

    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        const auto streamed = rayd::torch::diffraction_paths_order1_forward(scene, source_lane);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
            "diffraction path export changed the caller's CUDA stream");
        stream.synchronize();
        require(streamed.count.item<int>() >= 0, "streamed diffraction path export failed");
    }

    auto empty_fixture = make_empty_diffraction_fixture();
    rayd::torch::DiffractionPathConfig empty_config = {
        empty_fixture.tx_pos,
        empty_fixture.tx_pol,
        empty_fixture.rx_pos,
        empty_fixture.active,
        empty_fixture.state,
        empty_fixture.material,
        0,
        0,
        1.0,
        0.0,
    };
    empty_config.layout = rayd::torch::DiffractionPathLayout::SourceLane;
    const auto empty = rayd::torch::diffraction_paths_order1_forward(scene, empty_config);
    require_tensor_contract(empty.count, {1}, at::kInt, device, "empty diffraction path count");
    require(empty.count.item<int>() == 0, "empty diffraction path count must be zero");
    require_tensor_contract(empty.valid, {0}, at::kBool, device, "empty diffraction path valid");
}

void test_diffraction_accumulation_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    auto fixture = make_one_diffraction_fixture();
    const auto device = mesh.vertices.device();
    const auto floats = mesh.vertices.options();
    const auto ints = mesh.faces.options();

    rayd::torch::DiffractionAccumulationConfig config = {
        fixture.active,
        fixture.state,
        fixture.material,
        1,
        fixture.grid,
        1.0,
        1,
        0,
        0,
        17,
        1,
        std::nullopt,
        false,
        fixture.sample_state_index,
        fixture.sample_edge_weight,
    };
    const auto accumulated =
        rayd::torch::diffraction_accumulation_forward(scene, config);
    for (const auto &entry : {
             accumulated.power,
             accumulated.field_x_re,
             accumulated.field_x_im,
             accumulated.field_y_re,
             accumulated.field_y_im,
             accumulated.field_z_re,
             accumulated.field_z_im}) {
        require_tensor_contract(
            entry, {4, 4}, at::kFloat, device, "diffraction accumulation grid");
        require_finite(entry, "diffraction accumulation grid");
    }
    require(
        at::all(accumulated.power >= 0).item<bool>(),
        "diffraction accumulated power must be non-negative");
    for (const auto &counter : {
             accumulated.direct_count,
             accumulated.keller_count,
             accumulated.suffix_count,
             accumulated.visibility_rejects,
             accumulated.edge_visibility_rejects,
             accumulated.utd_rejects,
             accumulated.edge_uses}) {
        require_tensor_contract(
            counter, {1}, at::kInt, device, "diffraction accumulation counter");
        require(
            counter.item<int>() >= 0,
            "diffraction accumulation counter must be non-negative");
    }
    require(
        accumulated.direct_count.item<int>() +
                accumulated.visibility_rejects.item<int>() +
                accumulated.edge_visibility_rejects.item<int>() +
                accumulated.utd_rejects.item<int>() >
            0,
        "nonempty direct diffraction samples produced no accounting outcome");
    for (const auto &tape : {
             accumulated.tape_active,
             accumulated.tape_state_idx,
             accumulated.tape_cell,
             accumulated.tape_material_idx,
             accumulated.tape_edge_u})
        require(tape.defined() && tape.numel() == 0, "disabled diffraction tape must be defined-empty");
}

void test_coherent_diffraction_typed_contracts() {
    MeshFixture mesh = make_triangle();
    auto scene = rayd::torch::create_scene({mesh_input(mesh)});
    auto fixture = make_one_diffraction_fixture();
    const auto device = mesh.vertices.device();
    const auto floats = mesh.vertices.options();

    rayd::torch::CoherentDiffractionConfig config = {
        fixture.active,
        fixture.state,
        fixture.material,
        1,
        fixture.grid,
        1.0,
        true,
        true,
    };
    const auto coherent =
        rayd::torch::diffraction_coherent_accumulation_forward(scene, config);
    for (const auto &entry : {
             coherent.direct_x_re,
             coherent.direct_x_im,
             coherent.direct_y_re,
             coherent.direct_y_im,
             coherent.direct_z_re,
             coherent.direct_z_im,
             coherent.multi_x_re,
             coherent.multi_x_im,
             coherent.multi_y_re,
             coherent.multi_y_im,
             coherent.multi_z_re,
             coherent.multi_z_im}) {
        require_tensor_contract(
            entry, {4, 4}, at::kFloat, device, "coherent diffraction field");
        require_finite(entry, "coherent diffraction field");
    }
    for (const auto &counter : {
             coherent.direct_count,
             coherent.multi_count,
             coherent.visibility_reject_count,
             coherent.utd_reject_count}) {
        require_tensor_contract(
            counter, {4, 4}, at::kInt, device, "coherent diffraction counter");
        require(
            at::all(counter >= 0).item<bool>(),
            "coherent diffraction counters must be non-negative");
    }
    require(
        coherent.direct_count.sum().item<int64_t>() +
                coherent.visibility_reject_count.sum().item<int64_t>() +
                coherent.utd_reject_count.sum().item<int64_t>() >
            0,
        "nonempty coherent diffraction state produced no accounting outcome");
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

rayd::torch::TransmissionSequenceRequest transmission_request(
    bool active,
    float eps_r = 4.0F,
    float sigma_e = 0.05F,
    float thickness_m = 0.1F) {
    const auto floats = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    const auto ints = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    const auto bools = at::TensorOptions().dtype(at::kBool).device(at::kCUDA);
    return {
        at::ones({1}, bools),
        at::tensor({0.0F, 0.0F, 2.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 0.0F, -2.0F}, floats).reshape({1, 3}),
        at::zeros({1, 1, 3}, floats),
        at::tensor({0.0F, 0.0F, 1.0F}, floats).reshape({1, 1, 3}),
        at::zeros({1, 1}, ints),
        active ? at::ones({1, 1}, bools) : at::zeros({1, 1}, bools),
        at::ones({1}, floats),
        at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        at::tensor({0.0F, 1.0F, 0.0F}, floats).reshape({1, 3}),
        at::zeros({1}, ints),
        at::ones({1}, ints),
        at::tensor({thickness_m}, floats),
        at::tensor({eps_r}, floats),
        at::tensor({sigma_e}, floats),
        at::ones({1}, floats),
        3.5e9,
    };
}

void require_transmission_schema(
    const rayd::torch::TransmissionSequenceResult& result,
    int64_t rows,
    int expected_device = 0) {
    require(
        result.field_vector.sizes() == at::IntArrayRef({rows, 3}) &&
            result.field_vector.scalar_type() == at::kComplexFloat,
        "transmission field-vector schema differs");
    for (const auto& tensor : {
             result.coefficient,
             result.path_field})
        require(
            tensor.sizes() == at::IntArrayRef({rows}) &&
                tensor.scalar_type() == at::kComplexFloat,
            "transmission complex scalar schema differs");
    for (const auto& tensor : {
             result.path_gain,
             result.path_length_m,
             result.delay_s})
        require(
            tensor.sizes() == at::IntArrayRef({rows}) &&
                tensor.scalar_type() == at::kFloat,
            "transmission real scalar schema differs");
    require(
        result.direction.sizes() == at::IntArrayRef({rows, 3}) &&
            result.direction.scalar_type() == at::kFloat,
        "transmission direction schema differs");
    for (const auto& tensor : {
             result.field_vector,
             result.coefficient,
             result.path_field,
             result.path_gain,
             result.path_length_m,
             result.delay_s,
             result.direction})
        require(
            tensor.is_cuda() && tensor.get_device() == expected_device &&
                tensor.is_contiguous(),
            "transmission output must be contiguous CUDA storage");
}

void require_transmission_jvp_schema(
    const rayd::torch::TransmissionSequenceJvpResult& result,
    int64_t rows,
    int expected_device = 0) {
    require(
        result.field_vector.sizes() == at::IntArrayRef({rows, 3}) &&
            result.field_vector.scalar_type() == at::kComplexFloat,
        "transmission JVP field-vector schema differs");
    for (const auto& tensor : {result.coefficient, result.path_field})
        require(
            tensor.sizes() == at::IntArrayRef({rows}) &&
                tensor.scalar_type() == at::kComplexFloat,
            "transmission JVP complex scalar schema differs");
    for (const auto& tensor : {
             result.path_gain,
             result.path_length_m,
             result.delay_s})
        require(
            tensor.sizes() == at::IntArrayRef({rows}) &&
                tensor.scalar_type() == at::kFloat,
            "transmission JVP real scalar schema differs");
    for (const auto& tensor : {
             result.field_vector,
             result.coefficient,
             result.path_field,
             result.path_gain,
             result.path_length_m,
             result.delay_s})
        require(
            tensor.is_cuda() && tensor.get_device() == expected_device &&
                tensor.is_contiguous(),
            "transmission JVP output must be contiguous CUDA storage");
}

void test_transmission_sequence_primal_and_depth_contracts() {
    auto empty = transmission_request(false);
    empty.path_valid = empty.path_valid.narrow(0, 0, 0);
    empty.source = empty.source.narrow(0, 0, 0);
    empty.target = empty.target.narrow(0, 0, 0);
    empty.interaction_positions = empty.interaction_positions.narrow(0, 0, 0);
    empty.interaction_normals = empty.interaction_normals.narrow(0, 0, 0);
    empty.interaction_material_id = empty.interaction_material_id.narrow(0, 0, 0);
    empty.interaction_valid = empty.interaction_valid.narrow(0, 0, 0);
    empty.tx_power = empty.tx_power.narrow(0, 0, 0);
    empty.tx_polarization = empty.tx_polarization.narrow(0, 0, 0);
    empty.rx_polarization = empty.rx_polarization.narrow(0, 0, 0);
    const auto empty_result = rayd::torch::field_transmission_sequence(empty);
    require_transmission_schema(empty_result, 0);
    rayd::torch::TransmissionSequenceJvpRequest empty_jvp;
    empty_jvp.primal = empty;
    require_transmission_jvp_schema(
        rayd::torch::field_transmission_sequence_jvp(empty_jvp), 0);
    rayd::torch::TransmissionSequenceBackwardRequest empty_backward;
    empty_backward.primal = empty;
    const auto empty_absent =
        rayd::torch::field_transmission_sequence_backward(empty_backward);
    require(
        !empty_absent.grad_layer_thickness_m.has_value() &&
            !empty_absent.grad_layer_eps_r.has_value() &&
            !empty_absent.grad_layer_sigma_e.has_value() &&
            !empty_absent.grad_frequency.has_value() &&
            !empty_absent.grad_source.has_value() &&
            !empty_absent.grad_target.has_value() &&
            !empty_absent.grad_interaction_positions.has_value() &&
            !empty_absent.grad_interaction_normals.has_value(),
        "empty transmission backward disabled outputs must remain absent");
    empty_backward.need_grad_layer_thickness_m = true;
    empty_backward.need_grad_layer_eps_r = true;
    empty_backward.need_grad_layer_sigma_e = true;
    empty_backward.need_grad_frequency = true;
    empty_backward.need_grad_geometry = true;
    const auto empty_defined =
        rayd::torch::field_transmission_sequence_backward(empty_backward);
    require(
        empty_defined.grad_layer_thickness_m.has_value() &&
            empty_defined.grad_layer_eps_r.has_value() &&
            empty_defined.grad_layer_sigma_e.has_value() &&
            empty_defined.grad_frequency.has_value() &&
            empty_defined.grad_source.has_value() &&
            empty_defined.grad_target.has_value() &&
            !empty_defined.grad_interaction_positions.has_value() &&
            empty_defined.grad_interaction_normals.has_value() &&
            empty_defined.grad_layer_thickness_m->sizes() ==
                at::IntArrayRef({empty.layer_thickness_m.size(0)}) &&
            empty_defined.grad_layer_eps_r->sizes() ==
                at::IntArrayRef({empty.layer_eps_r.size(0)}) &&
            empty_defined.grad_layer_sigma_e->sizes() ==
                at::IntArrayRef({empty.layer_sigma_e.size(0)}) &&
            empty_defined.grad_frequency->sizes() == at::IntArrayRef({1}) &&
            empty_defined.grad_source->sizes() == at::IntArrayRef({0, 3}) &&
            empty_defined.grad_target->sizes() == at::IntArrayRef({0, 3}) &&
            empty_defined.grad_interaction_normals->sizes() ==
                at::IntArrayRef({0, 1, 3}),
        "empty transmission backward requested-output schema differs");
    for (const auto& tensor : {
             *empty_defined.grad_layer_thickness_m,
             *empty_defined.grad_layer_eps_r,
             *empty_defined.grad_layer_sigma_e,
             *empty_defined.grad_frequency,
             *empty_defined.grad_source,
             *empty_defined.grad_target,
             *empty_defined.grad_interaction_normals})
        require(
            tensor.is_cuda() &&
                tensor.get_device() == empty.source.get_device() &&
                tensor.scalar_type() == at::kFloat && tensor.is_contiguous() &&
                at::count_nonzero(tensor).item<int64_t>() == 0,
            "empty transmission requested gradient must be a CUDA zero tensor");

    const auto inactive = rayd::torch::field_transmission_sequence(
        transmission_request(false));
    const auto lossy = rayd::torch::field_transmission_sequence(
        transmission_request(true));
    at::cuda::getCurrentCUDAStream(0).synchronize();
    require_transmission_schema(lossy, 1);
    require(
        std::fabs(lossy.path_length_m.item<float>() - 4.0F) <= 1.0e-6F,
        "transmission full straight path length differs");
    require(
        std::fabs(
            lossy.delay_s.item<float>() -
            4.0F / 299792458.0F) <=
            1.0e-12F,
        "transmission delay differs");
    require(
        at::allclose(
            lossy.direction,
            at::tensor(
                {0.0F, 0.0F, -1.0F}, lossy.direction.options()).reshape({1, 3}),
            0.0,
            0.0),
        "transmission direction differs");
    require(
        lossy.path_gain.item<float>() < inactive.path_gain.item<float>(),
        "lossy transmission must attenuate the inactive-wall field");

    const auto vacuum = rayd::torch::field_transmission_sequence(
        transmission_request(true, 1.0F, 0.0F, 0.3F));
    at::cuda::getCurrentCUDAStream(0).synchronize();
    require(
        at::allclose(vacuum.field_vector, inactive.field_vector, 1.0e-4, 1.0e-9),
        "vacuum wall must reproduce the no-wall field");

    auto depth_nine = transmission_request(false);
    depth_nine.interaction_positions = at::zeros(
        {1, 9, 3}, depth_nine.source.options());
    depth_nine.interaction_normals = at::zeros(
        {1, 9, 3}, depth_nine.source.options());
    depth_nine.interaction_material_id = at::zeros(
        {1, 9}, depth_nine.layer_offset.options());
    depth_nine.interaction_valid = at::zeros(
        {1, 9}, depth_nine.interaction_valid.options());
    require_transmission_schema(
        rayd::torch::field_transmission_sequence(depth_nine), 1);
    rayd::torch::TransmissionSequenceJvpRequest too_deep_jvp;
    too_deep_jvp.primal = depth_nine;
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence_jvp(too_deep_jvp); },
        "transmission JVP must reject D > 8 while primal remains valid");
}

void test_transmission_sequence_ad_duality_and_optional_schema() {
    auto primal = transmission_request(true);
    const auto floats = primal.source.options();
    rayd::torch::TransmissionSequenceJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_layer_thickness_m = at::tensor({0.002F}, floats);
    jvp.tangent_layer_eps_r = at::tensor({0.2F}, floats);
    jvp.tangent_layer_sigma_e = at::tensor({0.001F}, floats);
    jvp.tangent_frequency = 1.0e6;
    jvp.tangent_source =
        at::tensor({0.01F, -0.02F, 0.03F}, floats).reshape({1, 3});
    jvp.tangent_target =
        at::tensor({-0.01F, 0.01F, -0.02F}, floats).reshape({1, 3});
    jvp.tangent_interaction_positions = at::ones({1, 1, 3}, floats);
    jvp.tangent_interaction_normals =
        at::tensor({0.02F, 0.01F, 0.0F}, floats).reshape({1, 1, 3});
    const auto tangents = rayd::torch::field_transmission_sequence_jvp(jvp);
    require_transmission_jvp_schema(tangents, 1);

    rayd::torch::TransmissionSequenceBackwardRequest backward;
    backward.primal = primal;
    const auto complex_options = floats.dtype(at::kComplexFloat);
    backward.grad_field_vector = at::ones({1, 3}, complex_options);
    backward.grad_coefficient = at::ones({1}, complex_options);
    backward.grad_path_field = at::ones({1}, complex_options);
    backward.grad_path_gain = at::ones({1}, floats);
    backward.grad_path_length_m = at::ones({1}, floats);
    backward.grad_delay_s = at::ones({1}, floats);
    backward.need_grad_layer_thickness_m = true;
    backward.need_grad_layer_eps_r = true;
    backward.need_grad_layer_sigma_e = true;
    backward.need_grad_frequency = true;
    backward.need_grad_geometry = true;
    const auto gradients =
        rayd::torch::field_transmission_sequence_backward(backward);
    at::cuda::getCurrentCUDAStream(0).synchronize();

    const double lhs =
        at::view_as_real(tangents.field_vector).select(-1, 0).sum().item<double>() +
        at::view_as_real(tangents.coefficient).select(-1, 0).sum().item<double>() +
        at::view_as_real(tangents.path_field).select(-1, 0).sum().item<double>() +
        tangents.path_gain.sum().item<double>() +
        tangents.path_length_m.sum().item<double>() +
        tangents.delay_s.sum().item<double>();
    double rhs =
        ((*gradients.grad_layer_thickness_m) *
         (*jvp.tangent_layer_thickness_m)).sum().item<double>() +
        ((*gradients.grad_layer_eps_r) *
         (*jvp.tangent_layer_eps_r)).sum().item<double>() +
        ((*gradients.grad_layer_sigma_e) *
         (*jvp.tangent_layer_sigma_e)).sum().item<double>() +
        gradients.grad_frequency->item<double>() * jvp.tangent_frequency +
        ((*gradients.grad_source) * (*jvp.tangent_source)).sum().item<double>() +
        ((*gradients.grad_target) * (*jvp.tangent_target)).sum().item<double>() +
        ((*gradients.grad_interaction_normals) *
         (*jvp.tangent_interaction_normals)).sum().item<double>();
    const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
    require(
        std::isfinite(lhs) && std::isfinite(rhs) &&
            std::fabs(lhs - rhs) <= 2.0e-3 * scale,
        "transmission JVP/VJP dot-product duality differs");
    require(
        !gradients.grad_interaction_positions.has_value(),
        "transmission crossing-position gradient must remain absent");

    rayd::torch::TransmissionSequenceBackwardRequest disabled;
    disabled.primal = primal;
    const auto disabled_result =
        rayd::torch::field_transmission_sequence_backward(disabled);
    require(
        !disabled_result.grad_layer_thickness_m.has_value() &&
            !disabled_result.grad_layer_eps_r.has_value() &&
            !disabled_result.grad_layer_sigma_e.has_value() &&
            !disabled_result.grad_frequency.has_value() &&
            !disabled_result.grad_source.has_value() &&
            !disabled_result.grad_target.has_value() &&
            !disabled_result.grad_interaction_positions.has_value() &&
            !disabled_result.grad_interaction_normals.has_value(),
        "disabled transmission backward outputs must remain absent");

    rayd::torch::TransmissionSequenceBackwardRequest zero_cotangents;
    zero_cotangents.primal = primal;
    zero_cotangents.need_grad_layer_thickness_m = true;
    zero_cotangents.need_grad_layer_eps_r = true;
    zero_cotangents.need_grad_layer_sigma_e = true;
    zero_cotangents.need_grad_frequency = true;
    zero_cotangents.need_grad_geometry = true;
    const auto zero_gradients =
        rayd::torch::field_transmission_sequence_backward(zero_cotangents);
    require(
        zero_gradients.grad_layer_thickness_m.has_value() &&
            zero_gradients.grad_layer_eps_r.has_value() &&
            zero_gradients.grad_layer_sigma_e.has_value() &&
            zero_gradients.grad_frequency.has_value() &&
            zero_gradients.grad_source.has_value() &&
            zero_gradients.grad_target.has_value() &&
            !zero_gradients.grad_interaction_positions.has_value() &&
            zero_gradients.grad_interaction_normals.has_value(),
        "requested transmission gradients must remain defined without cotangents");
    for (const auto& tensor : {
             *zero_gradients.grad_layer_thickness_m,
             *zero_gradients.grad_layer_eps_r,
             *zero_gradients.grad_layer_sigma_e,
             *zero_gradients.grad_frequency,
             *zero_gradients.grad_source,
             *zero_gradients.grad_target,
             *zero_gradients.grad_interaction_normals})
        require(
            at::count_nonzero(tensor).item<int64_t>() == 0,
            "transmission gradient without cotangents must be zero");
}

void test_transmission_sequence_negative_and_stream_contracts() {
    auto primal = transmission_request(true);
    auto bad_path_valid_dtype = primal;
    bad_path_valid_dtype.path_valid = at::ones({1}, primal.source.options());
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_path_valid_dtype); },
        "transmission path_valid dtype mismatch must fail loudly");
    auto bad_path_valid_shape = primal;
    bad_path_valid_shape.path_valid = at::ones({2}, primal.path_valid.options());
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_path_valid_shape); },
        "transmission path_valid shape mismatch must fail loudly");
    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard guard(stream);
    const auto result = rayd::torch::field_transmission_sequence(primal);
    rayd::torch::TransmissionSequenceJvpRequest jvp;
    jvp.primal = primal;
    const auto tangent = rayd::torch::field_transmission_sequence_jvp(jvp);
    rayd::torch::TransmissionSequenceBackwardRequest backward;
    backward.primal = primal;
    backward.grad_path_gain = at::ones({1}, primal.source.options());
    backward.need_grad_geometry = true;
    const auto gradients =
        rayd::torch::field_transmission_sequence_backward(backward);
    require(
        c10::cuda::getCurrentCUDAStream(0).stream() == stream.stream(),
        "transmission entries changed the caller's active CUDA stream");
    stream.synchronize();
    require(result.path_gain.item<float>() >= 0.0F, "stream primal is invalid");
    require(tangent.path_gain.item<float>() == 0.0F, "zero JVP must be zero");
    require(gradients.grad_source.has_value(), "requested geometry gradient missing");

    auto bad_dtype = primal;
    bad_dtype.source = bad_dtype.source.to(at::kDouble);
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_dtype); },
        "transmission wrong dtype must fail loudly");
    auto cpu_primal = primal;
    cpu_primal.source = cpu_primal.source.cpu();
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(cpu_primal); },
        "transmission CPU primal must fail loudly");
    auto noncontiguous_primal = primal;
    noncontiguous_primal.source =
        at::zeros({1, 3, 2}, primal.source.options()).select(2, 0);
    require(
        !noncontiguous_primal.source.is_contiguous(),
        "noncontiguous transmission fixture unexpectedly became contiguous");
    require_throws(
        [&] {
            (void)rayd::torch::field_transmission_sequence(
                noncontiguous_primal);
        },
        "transmission noncontiguous primal must fail loudly");
    auto bad_rows = primal;
    bad_rows.target = at::zeros({2, 3}, primal.target.options());
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_rows); },
        "transmission row mismatch must fail loudly");
    auto bad_csr = primal;
    bad_csr.layer_count = at::ones({2}, primal.layer_count.options());
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_csr); },
        "transmission CSR material-count mismatch must fail loudly");
    auto bad_depth = primal;
    bad_depth.interaction_positions = at::empty(
        {1, 0, 3}, primal.source.options());
    bad_depth.interaction_normals = at::empty(
        {1, 0, 3}, primal.source.options());
    bad_depth.interaction_material_id = at::empty(
        {1, 0}, primal.layer_offset.options());
    bad_depth.interaction_valid = at::empty(
        {1, 0}, primal.interaction_valid.options());
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_depth); },
        "transmission D=0 must fail loudly");
    auto bad_frequency = primal;
    bad_frequency.frequency_hz = 0.0;
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence(bad_frequency); },
        "transmission non-positive frequency must fail loudly");
    auto bad_material = primal;
    bad_material.interaction_material_id =
        bad_material.interaction_material_id.clone();
    bad_material.interaction_material_id.fill_(1);
    const auto invalid_material =
        rayd::torch::field_transmission_sequence(bad_material);
    require(
        invalid_material.path_gain.item<float>() == 0.0F,
        "transmission out-of-range material id must invalidate the path");
    auto inactive_invalid = transmission_request(false);
    const auto inactive_reference =
        rayd::torch::field_transmission_sequence(inactive_invalid);
    inactive_invalid.interaction_material_id =
        inactive_invalid.interaction_material_id.clone();
    inactive_invalid.interaction_material_id.fill_(1);
    const auto inactive_with_invalid_material =
        rayd::torch::field_transmission_sequence(inactive_invalid);
    require(
        at::allclose(
            inactive_with_invalid_material.coefficient,
            inactive_reference.coefficient,
            0.0,
            0.0),
        "inactive transmission slot must ignore its material id");
    auto bad_jvp = jvp;
    bad_jvp.tangent_source = at::ones({2, 3}, primal.source.options());
    require_throws(
        [&] { (void)rayd::torch::field_transmission_sequence_jvp(bad_jvp); },
        "transmission tangent shape mismatch must fail loudly");
    auto noncontiguous_jvp = jvp;
    noncontiguous_jvp.tangent_source =
        at::ones({1, 3, 2}, primal.source.options()).select(2, 0);
    require(
        !noncontiguous_jvp.tangent_source->is_contiguous(),
        "noncontiguous tangent fixture unexpectedly became contiguous");
    require_transmission_jvp_schema(
        rayd::torch::field_transmission_sequence_jvp(noncontiguous_jvp), 1);
    auto bad_backward = backward;
    bad_backward.grad_path_gain = at::ones(
        {1}, primal.source.options().dtype(at::kDouble));
    require_throws(
        [&] {
            (void)rayd::torch::field_transmission_sequence_backward(
                bad_backward);
        },
        "transmission cotangent dtype mismatch must fail loudly");
    auto noncontiguous_backward = backward;
    noncontiguous_backward.grad_field_vector = at::ones(
        {1, 3, 2},
        primal.source.options().dtype(at::kComplexFloat)).select(2, 0);
    noncontiguous_backward.need_grad_layer_thickness_m = true;
    require(
        !noncontiguous_backward.grad_field_vector->is_contiguous(),
        "noncontiguous cotangent fixture unexpectedly became contiguous");
    require(
        rayd::torch::field_transmission_sequence_backward(
            noncontiguous_backward).grad_layer_thickness_m.has_value(),
        "noncontiguous transmission cotangent must be accepted");
    auto too_deep = primal;
    too_deep.interaction_positions = at::zeros(
        {1, 9, 3}, primal.interaction_positions.options());
    too_deep.interaction_normals = at::zeros(
        {1, 9, 3}, primal.interaction_normals.options());
    too_deep.interaction_material_id = at::zeros(
        {1, 9}, primal.interaction_material_id.options());
    too_deep.interaction_valid = at::zeros(
        {1, 9}, primal.interaction_valid.options());
    auto too_deep_backward = backward;
    too_deep_backward.primal = too_deep;
    require_throws(
        [&] {
            (void)rayd::torch::field_transmission_sequence_backward(
                too_deep_backward);
        },
        "transmission backward must reject D > 8");
    if (at::cuda::device_count() > 1) {
        auto bad_device = primal;
        bad_device.layer_eps_r = bad_device.layer_eps_r.to(
            at::Device(at::kCUDA, 1));
        require_throws(
            [&] {
                (void)rayd::torch::field_transmission_sequence(bad_device);
            },
            "transmission cross-device primal must fail loudly");
        auto bad_tangent_device = jvp;
        bad_tangent_device.tangent_source = at::ones(
            {1, 3},
            primal.source.options().device(at::Device(at::kCUDA, 1)));
        require_throws(
            [&] {
                (void)rayd::torch::field_transmission_sequence_jvp(
                    bad_tangent_device);
            },
            "transmission cross-device tangent must fail loudly");
        auto bad_cotangent_device = backward;
        bad_cotangent_device.grad_path_gain = at::ones(
            {1},
            primal.source.options().device(at::Device(at::kCUDA, 1)));
        require_throws(
            [&] {
                (void)rayd::torch::field_transmission_sequence_backward(
                    bad_cotangent_device);
            },
            "transmission cross-device cotangent must fail loudly");
    }
}

void test_transmission_sequence_nondefault_stream_dependency() {
    auto request = transmission_request(false);
    const auto reference_request = transmission_request(true);
    rayd::torch::TransmissionSequenceJvpRequest reference_jvp;
    reference_jvp.primal = reference_request;
    reference_jvp.tangent_layer_thickness_m =
        at::ones({1}, reference_request.layer_thickness_m.options());
    rayd::torch::TransmissionSequenceBackwardRequest reference_backward;
    reference_backward.primal = reference_request;
    reference_backward.grad_path_gain =
        at::ones({1}, reference_request.tx_power.options());
    reference_backward.need_grad_layer_thickness_m = true;
    const auto reference =
        rayd::torch::field_transmission_sequence(reference_request);
    const auto reference_tangent =
        rayd::torch::field_transmission_sequence_jvp(reference_jvp);
    const auto reference_gradient =
        rayd::torch::field_transmission_sequence_backward(reference_backward);
    at::cuda::getDefaultCUDAStream().synchronize();

    const auto producer = c10::cuda::getStreamFromPool(false, 0);
    const auto consumer = c10::cuda::getStreamFromPool(false, 0);
    require(producer.stream() != consumer.stream(), "stream-pool fixtures must differ");
    auto scratch = at::empty(
        {64 * 1024 * 1024}, request.source.options().dtype(at::kByte));
    cudaEvent_t ready = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    for (int iteration = 0; iteration < 32; ++iteration) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            scratch.data_ptr(), iteration, static_cast<std::size_t>(scratch.numel()),
            producer.stream()));
    }
    {
        c10::cuda::CUDAStreamGuard producer_guard(producer);
        request.interaction_valid.fill_(true);
    }
    C10_CUDA_CHECK(cudaEventRecord(ready, producer.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(consumer.stream(), ready, 0));

    rayd::torch::TransmissionSequenceResult result;
    rayd::torch::TransmissionSequenceJvpResult tangent;
    rayd::torch::TransmissionSequenceBackwardResult gradient;
    {
        c10::cuda::CUDAStreamGuard consumer_guard(consumer);
        result = rayd::torch::field_transmission_sequence(request);
        rayd::torch::TransmissionSequenceJvpRequest jvp;
        jvp.primal = request;
        jvp.tangent_layer_thickness_m =
            at::ones({1}, request.layer_thickness_m.options());
        tangent = rayd::torch::field_transmission_sequence_jvp(jvp);
        rayd::torch::TransmissionSequenceBackwardRequest backward;
        backward.primal = request;
        backward.grad_path_gain = at::ones({1}, request.tx_power.options());
        backward.need_grad_layer_thickness_m = true;
        gradient = rayd::torch::field_transmission_sequence_backward(backward);
        require(
            c10::cuda::getCurrentCUDAStream(0).stream() == consumer.stream(),
            "transmission entries changed the caller's non-default stream");
    }
    consumer.synchronize();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    C10_CUDA_CHECK(cudaEventDestroy(ready));

    require(
        at::allclose(result.path_gain, reference.path_gain, 1.0e-6, 1.0e-12),
        "transmission primal ignored producer-consumer stream dependency");
    require(
        at::allclose(
            tangent.path_gain, reference_tangent.path_gain, 1.0e-6, 1.0e-12),
        "transmission JVP ignored producer-consumer stream dependency");
    require(
        gradient.grad_layer_thickness_m.has_value() &&
            at::allclose(
                *gradient.grad_layer_thickness_m,
                *reference_gradient.grad_layer_thickness_m,
                1.0e-6,
                1.0e-12),
        "transmission backward ignored producer-consumer stream dependency");
}

void test_transmission_path_valid_short_circuits_poison() {
    auto primal = transmission_request(true);
    primal.path_valid.zero_();
    primal.source.fill_(std::numeric_limits<float>::quiet_NaN());
    primal.interaction_material_id.fill_(std::numeric_limits<int>::max());
    primal.layer_offset.fill_(std::numeric_limits<int>::max());
    const auto forward = rayd::torch::field_transmission_sequence(primal);
    at::cuda::getCurrentCUDAStream().synchronize();
    for (const auto& tensor : {
             forward.field_vector, forward.coefficient, forward.path_field,
             forward.path_gain, forward.path_length_m, forward.delay_s,
             forward.direction})
        require(
            at::count_nonzero(tensor).item<int64_t>() == 0,
            "invalid poisoned transmission primal output must be exactly zero");

    rayd::torch::TransmissionSequenceJvpRequest jvp;
    jvp.primal = primal;
    jvp.tangent_frequency = 1.0;
    const auto tangent = rayd::torch::field_transmission_sequence_jvp(jvp);
    at::cuda::getCurrentCUDAStream().synchronize();
    for (const auto& tensor : {
             tangent.field_vector, tangent.coefficient, tangent.path_field,
             tangent.path_gain, tangent.path_length_m, tangent.delay_s})
        require(
            at::count_nonzero(tensor).item<int64_t>() == 0,
            "invalid poisoned transmission JVP output must be exactly zero");

    rayd::torch::TransmissionSequenceBackwardRequest backward;
    backward.primal = primal;
    backward.grad_field_vector = at::ones(
        {1, 3}, primal.source.options().dtype(at::kComplexFloat));
    backward.grad_path_gain = at::ones({1}, primal.source.options());
    backward.need_grad_layer_thickness_m = true;
    backward.need_grad_layer_eps_r = true;
    backward.need_grad_layer_sigma_e = true;
    backward.need_grad_frequency = true;
    backward.need_grad_geometry = true;
    const auto gradients =
        rayd::torch::field_transmission_sequence_backward(backward);
    at::cuda::getCurrentCUDAStream().synchronize();
    require(
        !gradients.grad_interaction_positions.has_value(),
        "transmission interaction-position VJP remains intentionally unsupported");
    for (const auto* tensor : {
             &*gradients.grad_layer_thickness_m, &*gradients.grad_layer_eps_r,
             &*gradients.grad_layer_sigma_e, &*gradients.grad_frequency,
             &*gradients.grad_source, &*gradients.grad_target,
             &*gradients.grad_interaction_normals})
        require(
            at::count_nonzero(*tensor).item<int64_t>() == 0,
            "invalid poisoned transmission gradient must be exactly zero");
}

} // namespace

int main() {
    try {
        require(at::cuda::is_available(), "CUDA is required for the typed integration tests");
        std::cout << "[RUN] test_scene_and_intersection_typed_contracts" << std::endl;
        test_scene_and_intersection_typed_contracts();
        std::cout << "[RUN] test_empty_and_stream_contracts" << std::endl;
        test_empty_and_stream_contracts();
        std::cout << "[RUN] test_visibility_trace_and_face_normal_typed_contracts" << std::endl;
        test_visibility_trace_and_face_normal_typed_contracts();
        std::cout << "[RUN] test_axial_edge_visibility_typed_contracts" << std::endl;
        test_axial_edge_visibility_typed_contracts();
        std::cout << "[RUN] test_axial_edge_visibility_legacy_parity" << std::endl;
        test_axial_edge_visibility_legacy_parity();
        std::cout << "[RUN] test_reflection_accumulation_and_epc_typed_contracts" << std::endl;
        test_reflection_accumulation_and_epc_typed_contracts();
        std::cout << "[RUN] test_error_and_lifecycle_contracts" << std::endl;
        test_error_and_lifecycle_contracts();
        std::cout << "[RUN] test_diffraction_paths_typed_contracts" << std::endl;
        test_diffraction_paths_typed_contracts();
        std::cout << "[RUN] test_diffraction_accumulation_typed_contracts" << std::endl;
        test_diffraction_accumulation_typed_contracts();
        std::cout << "[RUN] test_coherent_diffraction_typed_contracts" << std::endl;
        test_coherent_diffraction_typed_contracts();
        std::cout << "[RUN] test_layer_stack_empty_and_contracts" << std::endl;
        test_layer_stack_empty_and_contracts();
        std::cout << "[RUN] test_layer_stack_nonempty_ad_and_stream" << std::endl;
        test_layer_stack_nonempty_ad_and_stream();
        std::cout << "[RUN] test_layer_stack_lossy_primal_and_negative_contracts" << std::endl;
        test_layer_stack_lossy_primal_and_negative_contracts();
        std::cout << "[RUN] test_layer_stack_nondefault_stream_dependency" << std::endl;
        test_layer_stack_nondefault_stream_dependency();
        std::cout << "[RUN] test_transmission_sequence_primal_and_depth_contracts" << std::endl;
        test_transmission_sequence_primal_and_depth_contracts();
        std::cout << "[RUN] test_transmission_sequence_ad_duality_and_optional_schema" << std::endl;
        test_transmission_sequence_ad_duality_and_optional_schema();
        std::cout << "[RUN] test_transmission_sequence_negative_and_stream_contracts" << std::endl;
        test_transmission_sequence_negative_and_stream_contracts();
        std::cout << "[RUN] test_transmission_sequence_nondefault_stream_dependency" << std::endl;
        test_transmission_sequence_nondefault_stream_dependency();
        std::cout << "[RUN] test_transmission_path_valid_short_circuits_poison" << std::endl;
        test_transmission_path_valid_short_circuits_poison();
        std::cout << "rayd::torch integration direct contracts passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "rayd::torch integration direct contract failure: " << error.what() << '\n';
        return 1;
    }
}
