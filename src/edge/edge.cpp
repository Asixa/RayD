// Copyright Xingyu Chen.
// Implements edge support for edge.

#include <src/edge/kernels.h>
#include <src/scene/cache.h>
#include <src/bindings/tensor_contract.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

// Device contract for this file: every public edge op makes the scene device
// current before dispatching, so the launchers in edge_forward.cu,
// edge_topk.cu, and edge_backward.cu resolve their stream on the scene device
// and add no guard of their own.

namespace {

const at::Tensor* optional_tensor(py::object obj, at::Tensor& storage) {
    if (obj.is_none())
        return nullptr;
    storage = obj.cast<at::Tensor>();
    if (!storage.defined() || storage.numel() == 0)
        return nullptr;
    return &storage;
}

void require_ray_tmax(const at::Tensor& ray_tmax, int64_t ray_count) {
    require_scalar_f(ray_tmax, "ray_tmax");
    if (ray_tmax.numel() != 0 && ray_tmax.size(0) != ray_count)
        throw std::runtime_error("ray_tmax must be empty or match the ray batch size.");
}

// Edge kernels read query tensors through the scene device's pointers, so a tensor on another GPU is an illegal
// address that poisons the context; reject it before launch.
void require_scene_device(const SceneCache& scene, const at::Tensor* tensor, const char* name) {
    if (tensor != nullptr && tensor->defined() && tensor->get_device() != scene.device_index)
        throw std::runtime_error(std::string(name) + " is on " + tensor->device().str() +
                                 " but the scene is on cuda:" + std::to_string(scene.device_index) +
                                 "; every query tensor must be on the same CUDA device as the scene.");
}

void require_scene_device(const SceneCache& scene, const at::Tensor& tensor, const char* name) {
    require_scene_device(scene, &tensor, name);
}

// Derives the lazily built edge structures the selected backend reads; a query with no rows or a scene without edges
// reads only the edge SoA.
void prepare_edge_query(SceneCache& scene, int64_t query_count, bool custom_bvh) {
    if (query_count == 0 || scene.edge_v0.numel() == 0)
        ensure_edge_soa(scene);
    else if (custom_bvh)
        ensure_custom_edge_bvh(scene);
    else
        ensure_optix_edge_accel(scene);
}

} // namespace

std::vector<at::Tensor> nearest_edge_forward_op(int64_t scene_handle, at::Tensor point) {
    require_vec3f(point, "point");
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, point, "point");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    prepare_edge_query(scene, point.size(0), scene.edge_backend == EdgeBackend::Cuda);
    EdgeForwardOutputs out =
        scene.edge_backend == EdgeBackend::Cuda ? edge_forward_bvh_cuda(scene, point) : edge_forward_cuda(scene, point);
    return {out.distance,       out.edge_point,   out.edge_t, out.shape_id, out.edge_id,
            out.global_edge_id, out.tape_edge_id, out.tape_s, out.tape_d};
}

std::vector<at::Tensor> nearest_edge_forward_noad_op(int64_t scene_handle, at::Tensor point) {
    require_vec3f(point, "point");
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, point, "point");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    prepare_edge_query(scene, point.size(0), scene.edge_backend == EdgeBackend::Cuda);
    EdgeForwardPublicOutputs out = scene.edge_backend == EdgeBackend::Cuda ? edge_forward_noad_bvh_cuda(scene, point)
                                                                           : edge_forward_noad_cuda(scene, point);
    return {out.distance, out.edge_point, out.edge_t, out.shape_id, out.edge_id, out.global_edge_id};
}

std::vector<at::Tensor> nearest_edges_topk_forward_op(int64_t scene_handle, at::Tensor point, int64_t k,
                                                      at::Tensor active) {
    require_vec3f(point, "point");
    require_mask(active, "active");
    if (k < 1 || k > 16)
        throw std::runtime_error("k must be in [1, 16].");
    if (active.numel() != 0 && active.size(0) != point.size(0))
        throw std::runtime_error("active must be empty or match the point batch size.");
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    if (point.get_device() != scene.device_index || (active.numel() != 0 && active.get_device() != scene.device_index))
        throw std::runtime_error("point and active must be on the scene CUDA device.");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    prepare_edge_query(scene, point.size(0), true);
    EdgeTopKForwardOutputs out = edge_topk_forward_cuda(scene, point, k, active);
    return {out.is_valid, out.distances,       out.points,      out.edge_t,       out.edge_points, out.shape_ids,
            out.edge_ids, out.global_edge_ids, out.is_boundary, out.tape_edge_id, out.tape_s,      out.tape_d};
}

py::tuple nearest_edge_backward_op(int64_t scene_handle, at::Tensor point, at::Tensor tape_edge_id, at::Tensor tape_s,
                                   at::Tensor tape_d, at::Tensor grad_distance, at::Tensor grad_edge_point,
                                   at::Tensor grad_edge_t) {
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, point, "point");
    require_scene_device(scene, tape_edge_id, "tape_edge_id");
    require_scene_device(scene, tape_s, "tape_s");
    require_scene_device(scene, tape_d, "tape_d");
    require_scene_device(scene, grad_distance, "grad_distance");
    require_scene_device(scene, grad_edge_point, "grad_edge_point");
    require_scene_device(scene, grad_edge_t, "grad_edge_t");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeBackwardOutputs out =
        edge_backward_cuda(scene.global_vertices, scene.edge_v0, scene.edge_v1, point, tape_edge_id, tape_s, tape_d,
                           grad_distance, grad_edge_point, grad_edge_t);
    return py::make_tuple(out.grad_vertices, out.grad_point);
}

py::tuple nearest_edge_backward_optional_op(int64_t scene_handle, at::Tensor point, at::Tensor tape_edge_id,
                                            at::Tensor tape_s, at::Tensor tape_d, py::object grad_distance_obj,
                                            py::object grad_edge_point_obj, py::object grad_edge_t_obj,
                                            py::object grad_edge_t_alias_obj) {
    at::Tensor grad_distance_storage;
    at::Tensor grad_edge_point_storage;
    at::Tensor grad_edge_t_storage;
    at::Tensor grad_edge_t_alias_storage;
    const at::Tensor* grad_distance = optional_tensor(grad_distance_obj, grad_distance_storage);
    const at::Tensor* grad_edge_point = optional_tensor(grad_edge_point_obj, grad_edge_point_storage);
    const at::Tensor* grad_edge_t = optional_tensor(grad_edge_t_obj, grad_edge_t_storage);
    const at::Tensor* grad_edge_t_alias = optional_tensor(grad_edge_t_alias_obj, grad_edge_t_alias_storage);
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, point, "point");
    require_scene_device(scene, tape_edge_id, "tape_edge_id");
    require_scene_device(scene, tape_s, "tape_s");
    require_scene_device(scene, tape_d, "tape_d");
    require_scene_device(scene, grad_distance, "grad_distance");
    require_scene_device(scene, grad_edge_point, "grad_edge_point");
    require_scene_device(scene, grad_edge_t, "grad_edge_t");
    require_scene_device(scene, grad_edge_t_alias, "grad_edge_t");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeBackwardOutputs out =
        edge_backward_optional_cuda(scene.global_vertices, scene.edge_v0, scene.edge_v1, point, tape_edge_id, tape_s,
                                    tape_d, grad_distance, grad_edge_point, grad_edge_t, grad_edge_t_alias);
    return py::make_tuple(out.grad_vertices, out.grad_point);
}

std::vector<at::Tensor> nearest_edge_ray_forward_op(int64_t scene_handle, at::Tensor ray_o, at::Tensor ray_d,
                                                    at::Tensor ray_tmax, at::Tensor active) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    require_mask(active, "active");
    if (ray_d.size(0) != ray_o.size(0) || (active.numel() != 0 && active.size(0) != ray_o.size(0)))
        throw std::runtime_error("ray_d, ray_tmax, and active must match ray_o batch size.");
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, ray_tmax, "ray_tmax");
    require_scene_device(scene, active, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    prepare_edge_query(scene, ray_o.size(0), scene.edge_backend == EdgeBackend::Cuda);
    EdgeRayForwardOutputs out = scene.edge_backend == EdgeBackend::Cuda
                                    ? edge_ray_forward_bvh_cuda(scene, ray_o, ray_d, ray_tmax, active)
                                    : edge_ray_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);
    return {out.distance, out.ray_t,   out.point,          out.edge_t,      out.edge_point,
            out.shape_id, out.edge_id, out.global_edge_id, out.tape_edge_id};
}

py::tuple nearest_edge_ray_backward_optional_op(int64_t scene_handle, at::Tensor ray_o, at::Tensor ray_d,
                                                at::Tensor ray_tmax, at::Tensor tape_edge_id, at::Tensor ray_t,
                                                at::Tensor edge_t, py::object grad_distance_obj,
                                                py::object grad_ray_t_obj, py::object grad_point_obj,
                                                py::object grad_edge_t_obj, py::object grad_edge_point_obj) {
    at::Tensor grad_distance_storage;
    at::Tensor grad_ray_t_storage;
    at::Tensor grad_point_storage;
    at::Tensor grad_edge_t_storage;
    at::Tensor grad_edge_point_storage;
    const at::Tensor* grad_distance = optional_tensor(grad_distance_obj, grad_distance_storage);
    const at::Tensor* grad_ray_t = optional_tensor(grad_ray_t_obj, grad_ray_t_storage);
    const at::Tensor* grad_point = optional_tensor(grad_point_obj, grad_point_storage);
    const at::Tensor* grad_edge_t = optional_tensor(grad_edge_t_obj, grad_edge_t_storage);
    const at::Tensor* grad_edge_point = optional_tensor(grad_edge_point_obj, grad_edge_point_storage);
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, ray_tmax, "ray_tmax");
    require_scene_device(scene, tape_edge_id, "tape_edge_id");
    require_scene_device(scene, ray_t, "ray_t");
    require_scene_device(scene, edge_t, "edge_t");
    require_scene_device(scene, grad_distance, "grad_distance");
    require_scene_device(scene, grad_ray_t, "grad_ray_t");
    require_scene_device(scene, grad_point, "grad_point");
    require_scene_device(scene, grad_edge_t, "grad_edge_t");
    require_scene_device(scene, grad_edge_point, "grad_edge_point");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeRayBackwardOutputs out =
        edge_ray_backward_optional_cuda(scene.global_vertices, scene.edge_v0, scene.edge_v1, ray_o, ray_d, ray_tmax,
                                        tape_edge_id, ray_t, edge_t, grad_distance, grad_ray_t, grad_point, grad_edge_t,
                                        grad_edge_point);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d);
}

py::tuple nearest_edge_ray_jvp_optional_op(int64_t scene_handle, at::Tensor ray_o, at::Tensor ray_d,
                                           at::Tensor ray_tmax, at::Tensor tape_edge_id, at::Tensor ray_t,
                                           at::Tensor edge_t, py::object tangent_vertices_obj,
                                           py::object tangent_ray_o_obj, py::object tangent_ray_d_obj) {
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor* tangent_vertices = optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor* tangent_ray_o = optional_tensor(tangent_ray_o_obj, tangent_ray_o_storage);
    const at::Tensor* tangent_ray_d = optional_tensor(tangent_ray_d_obj, tangent_ray_d_storage);
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, ray_tmax, "ray_tmax");
    require_scene_device(scene, tape_edge_id, "tape_edge_id");
    require_scene_device(scene, ray_t, "ray_t");
    require_scene_device(scene, edge_t, "edge_t");
    require_scene_device(scene, tangent_vertices, "tangent_vertices");
    require_scene_device(scene, tangent_ray_o, "tangent_ray_o");
    require_scene_device(scene, tangent_ray_d, "tangent_ray_d");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor tangent_vertices_global;
    if (tangent_vertices != nullptr && tangent_vertices->sizes() != scene.global_vertices.sizes()) {
        if (tangent_vertices->dim() != 2 || tangent_vertices->size(1) != 3 ||
            tangent_vertices->size(0) > scene.global_vertices.size(0)) {
            throw std::runtime_error("tangent_vertices must fit the scene-global vertex buffer.");
        }
        tangent_vertices_global = at::zeros_like(scene.global_vertices);
        tangent_vertices_global.narrow(0, 0, tangent_vertices->size(0)).copy_(*tangent_vertices);
        tangent_vertices = &tangent_vertices_global;
    }
    EdgeRayJvpOutputs out =
        edge_ray_jvp_optional_cuda(scene.global_vertices, scene.edge_v0, scene.edge_v1, ray_o, ray_d, ray_tmax,
                                   tape_edge_id, ray_t, edge_t, tangent_vertices, tangent_ray_o, tangent_ray_d);
    return py::make_tuple(out.tangent_distance, out.tangent_ray_t, out.tangent_point, out.tangent_edge_t,
                          out.tangent_edge_point);
}

py::tuple nearest_edge_jvp_op(int64_t scene_handle, at::Tensor point, at::Tensor tape_edge_id, at::Tensor tape_s,
                              at::Tensor tape_d, at::Tensor tangent_vertices, at::Tensor tangent_point) {
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, point, "point");
    require_scene_device(scene, tape_edge_id, "tape_edge_id");
    require_scene_device(scene, tape_s, "tape_s");
    require_scene_device(scene, tape_d, "tape_d");
    require_scene_device(scene, tangent_vertices, "tangent_vertices");
    require_scene_device(scene, tangent_point, "tangent_point");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeJvpOutputs out = edge_jvp_cuda(scene.global_vertices, scene.edge_v0, scene.edge_v1, point, tape_edge_id, tape_s,
                                       tape_d, tangent_vertices, tangent_point);
    return py::make_tuple(out.tangent_distance, out.tangent_edge_point, out.tangent_edge_t, out.tangent_tape_s,
                          out.tangent_tape_d);
}

py::tuple nearest_edge_jvp_optional_op(int64_t scene_handle, at::Tensor point, at::Tensor tape_edge_id,
                                       at::Tensor tape_s, at::Tensor tape_d, py::object tangent_vertices_obj,
                                       py::object tangent_point_obj) {
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_point_storage;
    const at::Tensor* tangent_vertices = optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor* tangent_point = optional_tensor(tangent_point_obj, tangent_point_storage);
    ScenePin scene_pin = get_scene(scene_handle);
    SceneCache& scene = *scene_pin;
    require_scene_device(scene, point, "point");
    require_scene_device(scene, tape_edge_id, "tape_edge_id");
    require_scene_device(scene, tape_s, "tape_s");
    require_scene_device(scene, tape_d, "tape_d");
    require_scene_device(scene, tangent_vertices, "tangent_vertices");
    require_scene_device(scene, tangent_point, "tangent_point");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    EdgeJvpOutputs out = edge_jvp_optional_cuda(scene.global_vertices, scene.edge_v0, scene.edge_v1, point,
                                                tape_edge_id, tape_s, tape_d, tangent_vertices, tangent_point);
    return py::make_tuple(out.tangent_distance, out.tangent_edge_point, out.tangent_edge_t, out.tangent_tape_s,
                          out.tangent_tape_d);
}

} // namespace rayd::torch_backend
