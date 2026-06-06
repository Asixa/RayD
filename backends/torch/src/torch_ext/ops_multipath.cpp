#include <raydtorch/geometry_kernels.h>
#include <raydtorch/multipath_kernels.h>
#include <raydtorch/scene_cache.h>
#include <raydtorch/tensor_check.h>

#include <torch/extension.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

void require_same_batch(const at::Tensor &a, const at::Tensor &b, const char *name) {
    if (a.size(0) != b.size(0))
        throw std::runtime_error(std::string(name) + " tensors must have the same batch size.");
}

at::Tensor first_bounce_column(const at::Tensor &value, int64_t ray_count) {
    if (value.dim() == 1)
        return value.reshape({ray_count}).contiguous();
    return value.slice(1, 0, 1).reshape({ray_count}).contiguous();
}

} // namespace

py::tuple visibility_forward_op(
    int64_t scene_handle,
    at::Tensor start,
    at::Tensor end,
    at::Tensor active) {
    require_vec3f(start, "start");
    require_vec3f(end, "end");
    require_mask(active, "active");
    require_same_batch(start, end, "visibility");
    if (active.size(0) != start.size(0))
        throw std::runtime_error("active must match the visibility batch size.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("visibility_forward: first milestone supports exactly one mesh.");
    VisibilityForwardOutputs out = visibility_forward_cuda(scene, start, end, active);
    return py::make_tuple(out.visible, out.tape_prim_id, out.tape_t);
}

py::tuple trace_reflections_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t max_bounces) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    require_same_batch(ray_o, ray_d, "trace_reflections");
    if (ray_tmax.size(0) != ray_o.size(0) || active.size(0) != ray_o.size(0))
        throw std::runtime_error("ray_tmax and active must match the ray batch size.");
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("trace_reflections_forward: first milestone supports exactly one mesh.");
    IntersectForwardOutputs hit = intersect_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);

    const int64_t ray_count = ray_o.size(0);
    auto fopts = ray_o.options();
    auto iopts = hit.global_prim_id.options();
    at::Tensor valid = at::zeros({ray_count, max_bounces}, active.options());
    at::Tensor t = at::full(
        {ray_count, max_bounces},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor image_sources = at::zeros({ray_count, max_bounces, 3}, fopts);
    at::Tensor prim_ids = at::full({ray_count, max_bounces}, -1, iopts);

    at::Tensor first_valid = hit.global_prim_id.ge(0).reshape({ray_count, 1});
    valid.slice(1, 0, 1).copy_(first_valid);
    t.slice(1, 0, 1).copy_(hit.t.reshape({ray_count, 1}));
    prim_ids.slice(1, 0, 1).copy_(hit.global_prim_id.reshape({ray_count, 1}));

    at::Tensor offset = at::sum((ray_o - hit.p) * hit.geo_n, {1}, true);
    at::Tensor image = (ray_o - 2.0 * offset * hit.geo_n).reshape({ray_count, 1, 3});
    image_sources.slice(1, 0, 1).copy_(
        at::where(first_valid.reshape({ray_count, 1, 1}), image, at::zeros_like(image)));

    return py::make_tuple(
        valid,
        t,
        image_sources,
        prim_ids,
        hit.tape_prim_id,
        hit.tape_barycentric,
        hit.tape_t);
}

py::tuple trace_reflections_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t) {
    SceneCache &scene = get_scene(scene_handle);
    const MeshRecord &mesh = scene.meshes[0];
    ReflectionBackwardOutputs out = reflection_backward_cuda(
        mesh.vertices,
        mesh.faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        first_bounce_column(grad_t, ray_o.size(0)));
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple trace_reflections_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tangent_vertices,
    at::Tensor tangent_ray_o,
    at::Tensor tangent_ray_d,
    at::Tensor image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    const MeshRecord &mesh = scene.meshes[0];
    ReflectionJvpOutputs out = reflection_jvp_cuda(
        mesh.vertices,
        mesh.faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices.contiguous(),
        tangent_ray_o.contiguous(),
        tangent_ray_d.contiguous(),
        image_sources);
    return py::make_tuple(out.tangent_t, out.tangent_image_sources);
}

void bind_multipath_ops(py::module_ &m) {
    m.def("visibility_forward", &visibility_forward_op);
    m.def("trace_reflections_forward", &trace_reflections_forward_op);
    m.def("trace_reflections_backward", &trace_reflections_backward_op);
    m.def("trace_reflections_jvp", &trace_reflections_jvp_op);
}

} // namespace raydtorch
