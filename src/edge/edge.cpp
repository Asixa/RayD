#include <rayd/torch/edge/kernels.h>
#include <rayd/torch/scene/cache.h>
#include <rayd/torch/bindings/tensor_contract.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace rayd::torch_backend {

// Device contract for this file: every public edge op makes the scene device
// current before dispatching, so the launchers in edge_forward.cu,
// edge_topk.cu, and edge_backward.cu resolve their stream on the scene device
// and add no guard of their own.

namespace {

const at::Tensor *optional_tensor(py::object obj, at::Tensor &storage) {
    if (obj.is_none())
        return nullptr;
    storage = obj.cast<at::Tensor>();
    if (!storage.defined() || storage.numel() == 0)
        return nullptr;
    return &storage;
}

void require_ray_tmax(const at::Tensor &ray_tmax, int64_t ray_count) {
    require_scalar_f(ray_tmax, "ray_tmax");
    if (ray_tmax.numel() != 0 && ray_tmax.size(0) != ray_count)
        throw std::runtime_error("ray_tmax must be empty or match the ray batch size.");
}

} // namespace

py::tuple nearest_edge_forward_op(int64_t scene_handle, at::Tensor point) {
    require_vec3f(point, "point");
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeForwardOutputs out = scene.edge_backend == EdgeBackend::Cuda
        ? edge_forward_bvh_cuda(scene, point)
        : edge_forward_cuda(scene, point);
    return py::make_tuple(
        out.distance,
        out.edge_point,
        out.edge_t,
        out.shape_id,
        out.edge_id,
        out.global_edge_id,
        out.tape_edge_id,
        out.tape_s,
        out.tape_d);
}

py::tuple nearest_edge_forward_noad_op(int64_t scene_handle, at::Tensor point) {
    require_vec3f(point, "point");
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeForwardPublicOutputs out = scene.edge_backend == EdgeBackend::Cuda
        ? edge_forward_noad_bvh_cuda(scene, point)
        : edge_forward_noad_cuda(scene, point);
    return py::make_tuple(
        out.distance,
        out.edge_point,
        out.edge_t,
        out.shape_id,
        out.edge_id,
        out.global_edge_id);
}

py::tuple nearest_edges_topk_forward_op(
    int64_t scene_handle,
    at::Tensor point,
    int64_t k,
    at::Tensor active) {
    require_vec3f(point, "point");
    require_mask(active, "active");
    if (k < 1 || k > 16)
        throw std::runtime_error("k must be in [1, 16].");
    if (active.numel() != 0 && active.size(0) != point.size(0))
        throw std::runtime_error("active must be empty or match the point batch size.");
    SceneCache &scene = get_scene(scene_handle);
    if (point.get_device() != scene.device_index ||
        (active.numel() != 0 && active.get_device() != scene.device_index))
        throw std::runtime_error("point and active must be on the scene CUDA device.");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    if (point.size(0) != 0 && scene.edge_v0.numel() != 0)
        ensure_custom_edge_bvh(scene);
    EdgeTopKForwardOutputs out = edge_topk_forward_cuda(scene, point, k, active);
    return py::make_tuple(
        out.is_valid,
        out.distances,
        out.points,
        out.edge_t,
        out.edge_points,
        out.shape_ids,
        out.edge_ids,
        out.global_edge_ids,
        out.is_boundary,
        out.tape_edge_id,
        out.tape_s,
        out.tape_d);
}

py::tuple nearest_edge_backward_op(
    int64_t scene_handle,
    at::Tensor point,
    at::Tensor tape_edge_id,
    at::Tensor tape_s,
    at::Tensor tape_d,
    at::Tensor grad_distance,
    at::Tensor grad_edge_point,
        at::Tensor grad_edge_t) {
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeBackwardOutputs out = edge_backward_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        grad_distance,
        grad_edge_point,
        grad_edge_t);
    return py::make_tuple(out.grad_vertices, out.grad_point);
}

py::tuple nearest_edge_backward_optional_op(
    int64_t scene_handle,
    at::Tensor point,
    at::Tensor tape_edge_id,
    at::Tensor tape_s,
    at::Tensor tape_d,
    py::object grad_distance_obj,
    py::object grad_edge_point_obj,
    py::object grad_edge_t_obj,
    py::object grad_edge_t_alias_obj) {
    at::Tensor grad_distance_storage;
    at::Tensor grad_edge_point_storage;
    at::Tensor grad_edge_t_storage;
    at::Tensor grad_edge_t_alias_storage;
    const at::Tensor *grad_distance = optional_tensor(grad_distance_obj, grad_distance_storage);
    const at::Tensor *grad_edge_point = optional_tensor(grad_edge_point_obj, grad_edge_point_storage);
    const at::Tensor *grad_edge_t = optional_tensor(grad_edge_t_obj, grad_edge_t_storage);
    const at::Tensor *grad_edge_t_alias = optional_tensor(grad_edge_t_alias_obj, grad_edge_t_alias_storage);
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeBackwardOutputs out = edge_backward_optional_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        grad_distance,
        grad_edge_point,
        grad_edge_t,
        grad_edge_t_alias);
    return py::make_tuple(out.grad_vertices, out.grad_point);
}

py::tuple nearest_edge_ray_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    require_mask(active, "active");
    if (ray_d.size(0) != ray_o.size(0) ||
        (active.numel() != 0 && active.size(0) != ray_o.size(0)))
        throw std::runtime_error("ray_d, ray_tmax, and active must match ray_o batch size.");
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeRayForwardOutputs out = scene.edge_backend == EdgeBackend::Cuda
        ? edge_ray_forward_bvh_cuda(scene, ray_o, ray_d, ray_tmax, active)
        : edge_ray_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);
    return py::make_tuple(
        out.distance,
        out.ray_t,
        out.point,
        out.edge_t,
        out.edge_point,
        out.shape_id,
        out.edge_id,
        out.global_edge_id,
        out.tape_edge_id);
}

py::tuple nearest_edge_ray_backward_optional_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor tape_edge_id,
    at::Tensor ray_t,
    at::Tensor edge_t,
    py::object grad_distance_obj,
    py::object grad_ray_t_obj,
    py::object grad_point_obj,
    py::object grad_edge_t_obj,
    py::object grad_edge_point_obj) {
    at::Tensor grad_distance_storage;
    at::Tensor grad_ray_t_storage;
    at::Tensor grad_point_storage;
    at::Tensor grad_edge_t_storage;
    at::Tensor grad_edge_point_storage;
    const at::Tensor *grad_distance =
        optional_tensor(grad_distance_obj, grad_distance_storage);
    const at::Tensor *grad_ray_t =
        optional_tensor(grad_ray_t_obj, grad_ray_t_storage);
    const at::Tensor *grad_point =
        optional_tensor(grad_point_obj, grad_point_storage);
    const at::Tensor *grad_edge_t =
        optional_tensor(grad_edge_t_obj, grad_edge_t_storage);
    const at::Tensor *grad_edge_point =
        optional_tensor(grad_edge_point_obj, grad_edge_point_storage);
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeRayBackwardOutputs out = edge_ray_backward_optional_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        ray_o,
        ray_d,
        ray_tmax,
        tape_edge_id,
        ray_t,
        edge_t,
        grad_distance,
        grad_ray_t,
        grad_point,
        grad_edge_t,
        grad_edge_point);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d);
}

py::tuple nearest_edge_ray_jvp_optional_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor tape_edge_id,
    at::Tensor ray_t,
    at::Tensor edge_t,
    py::object tangent_vertices_obj,
    py::object tangent_ray_o_obj,
    py::object tangent_ray_d_obj) {
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor *tangent_vertices =
        optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor *tangent_ray_o =
        optional_tensor(tangent_ray_o_obj, tangent_ray_o_storage);
    const at::Tensor *tangent_ray_d =
        optional_tensor(tangent_ray_d_obj, tangent_ray_d_storage);
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor tangent_vertices_global;
    if (tangent_vertices != nullptr &&
        tangent_vertices->sizes() != scene.global_vertices.sizes()) {
        if (tangent_vertices->dim() != 2 || tangent_vertices->size(1) != 3 ||
            tangent_vertices->size(0) > scene.global_vertices.size(0)) {
            throw std::runtime_error(
                "tangent_vertices must fit the scene-global vertex buffer.");
        }
        tangent_vertices_global = at::zeros_like(scene.global_vertices);
        tangent_vertices_global.narrow(0, 0, tangent_vertices->size(0))
            .copy_(*tangent_vertices);
        tangent_vertices = &tangent_vertices_global;
    }
    EdgeRayJvpOutputs out = edge_ray_jvp_optional_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        ray_o,
        ray_d,
        ray_tmax,
        tape_edge_id,
        ray_t,
        edge_t,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d);
    return py::make_tuple(
        out.tangent_distance,
        out.tangent_ray_t,
        out.tangent_point,
        out.tangent_edge_t,
        out.tangent_edge_point);
}

py::tuple nearest_edge_jvp_op(
    int64_t scene_handle,
    at::Tensor point,
    at::Tensor tape_edge_id,
    at::Tensor tape_s,
    at::Tensor tape_d,
    at::Tensor tangent_vertices,
        at::Tensor tangent_point) {
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeJvpOutputs out = edge_jvp_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        tangent_vertices,
        tangent_point);
    return py::make_tuple(
        out.tangent_distance,
        out.tangent_edge_point,
        out.tangent_edge_t,
        out.tangent_tape_s,
        out.tangent_tape_d);
}

py::tuple nearest_edge_jvp_optional_op(
    int64_t scene_handle,
    at::Tensor point,
    at::Tensor tape_edge_id,
    at::Tensor tape_s,
    at::Tensor tape_d,
    py::object tangent_vertices_obj,
    py::object tangent_point_obj) {
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_point_storage;
    const at::Tensor *tangent_vertices = optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor *tangent_point = optional_tensor(tangent_point_obj, tangent_point_storage);
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeJvpOutputs out = edge_jvp_optional_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        tangent_vertices,
        tangent_point);
    return py::make_tuple(
        out.tangent_distance,
        out.tangent_edge_point,
        out.tangent_edge_t,
        out.tangent_tape_s,
        out.tangent_tape_d);
}

} // namespace rayd::torch_backend
