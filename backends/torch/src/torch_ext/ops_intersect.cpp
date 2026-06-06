#include <raydtorch/geometry_kernels.h>
#include <raydtorch/scene_cache.h>
#include <raydtorch/tensor_check.h>

#include <torch/extension.h>

namespace raydtorch {

py::tuple intersect_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("intersect_forward: first milestone supports exactly one mesh.");
    const MeshRecord &mesh = scene.meshes[0];
    IntersectForwardOutputs out = intersect_forward_cuda(mesh.vertices, mesh.faces, ray_o, ray_d, ray_tmax, active);
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_t);
}

void bind_intersect_ops(py::module_ &m) {
    m.def("intersect_forward", &intersect_forward_op);
}

} // namespace raydtorch
