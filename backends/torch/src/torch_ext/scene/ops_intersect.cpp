#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/scene/cache.h>
#include <raydtorch/common/tensor_check.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <stdexcept>

namespace raydtorch {

namespace {

class IntersectTSumFunction : public torch::autograd::Function<IntersectTSumFunction> {
  public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int64_t scene_handle,
        torch::autograd::Variable vertices,
        torch::autograd::Variable ray_o,
        torch::autograd::Variable ray_d,
        torch::autograd::Variable ray_tmax,
        torch::autograd::Variable active,
        int64_t flags) {
        SceneCache &scene = get_scene(scene_handle);
        IntersectForwardOutputs out =
            intersect_forward_ad_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
        ctx->saved_data["scene_handle"] = scene_handle;
        ctx->saved_data["ray_count"] = ray_o.size(0);
        ctx->saved_data["need_grad_vertices"] = vertices.requires_grad();
        ctx->saved_data["need_grad_ray_o"] = ray_o.requires_grad();
        ctx->saved_data["need_grad_ray_d"] = ray_d.requires_grad();
        ctx->saved_data["need_grad_ray_tmax"] = ray_tmax.requires_grad();
        ctx->save_for_backward({ray_o, ray_d, active, out.tape_prim_id, out.tape_barycentric});
        return {out.t.sum()};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor &ray_o = saved[0];
        const at::Tensor &ray_d = saved[1];
        const at::Tensor &active = saved[2];
        const at::Tensor &tape_prim_id = saved[3];
        const at::Tensor &tape_barycentric = saved[4];
        const at::Tensor &grad_loss = grad_outputs[0];
        const int64_t scene_handle = ctx->saved_data["scene_handle"].toInt();
        const int64_t ray_count = ctx->saved_data["ray_count"].toInt();
        const bool need_grad_vertices = ctx->saved_data["need_grad_vertices"].toBool();
        const bool need_grad_ray_o = ctx->saved_data["need_grad_ray_o"].toBool();
        const bool need_grad_ray_d = ctx->saved_data["need_grad_ray_d"].toBool();
        const bool need_grad_ray_tmax = ctx->saved_data["need_grad_ray_tmax"].toBool();
        if (!grad_loss.defined() ||
            !(need_grad_vertices || need_grad_ray_o || need_grad_ray_d || need_grad_ray_tmax)) {
            return {
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
            };
        }

        SceneCache &scene = get_scene(scene_handle);
        at::Tensor grad_t = grad_loss.reshape({}).expand({ray_count});
        IntersectBackwardOutputs out = intersect_backward_t_cuda(
            scene.global_vertices,
            scene.global_faces,
            ray_o,
            ray_d,
            active,
            tape_prim_id,
            tape_barycentric,
            grad_t,
            grad_t.stride(0),
            need_grad_vertices,
            need_grad_ray_o,
            need_grad_ray_d,
            need_grad_ray_tmax);
        return {
            at::Tensor(),
            need_grad_vertices ? out.grad_vertices : at::Tensor(),
            need_grad_ray_o ? out.grad_ray_o : at::Tensor(),
            need_grad_ray_d ? out.grad_ray_d : at::Tensor(),
            need_grad_ray_tmax ? out.grad_ray_tmax : at::Tensor(),
            at::Tensor(),
            at::Tensor(),
        };
    }
};

} // namespace

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
    IntersectForwardOutputs out = intersect_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);
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

py::tuple intersection_public_tuple(const IntersectForwardOutputs &out) {
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
        out.global_prim_id);
}

py::tuple intersect_forward_flags_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t flags) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out =
        intersect_forward_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
    return intersection_public_tuple(out);
}

py::tuple intersect_forward_ad_flags_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t flags) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out =
        intersect_forward_ad_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
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

at::Tensor intersect_t_sum_ad_op(
    int64_t scene_handle,
    at::Tensor vertices,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t flags) {
    require_vec3f(vertices, "vertices");
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    return IntersectTSumFunction::apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        flags)[0];
}

py::tuple intersect_t_sum_vjp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t flags,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs forward =
        intersect_forward_ad_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
    at::Tensor loss = forward.t.sum();
    IntersectBackwardOutputs backward = intersect_backward_t_sum_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        forward.tape_prim_id,
        forward.tape_barycentric,
        need_grad_vertices,
        need_grad_ray_o,
        need_grad_ray_d,
        need_grad_ray_tmax);
    return py::make_tuple(
        loss,
        backward.grad_vertices,
        backward.grad_ray_o,
        backward.grad_ray_d,
        backward.grad_ray_tmax);
}

py::tuple intersect_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t,
    at::Tensor grad_p,
    at::Tensor grad_n,
    at::Tensor grad_geo_n,
    at::Tensor grad_uv,
    at::Tensor grad_barycentric) {
    SceneCache &scene = get_scene(scene_handle);
    IntersectBackwardOutputs out = intersect_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t.contiguous(),
        grad_p.contiguous(),
        grad_n.contiguous(),
        grad_geo_n.contiguous(),
        grad_uv.contiguous(),
        grad_barycentric.contiguous());
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple intersect_backward_t_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_mask(active, "active");
    require_contiguous(tape_prim_id, "tape_prim_id");
    require_dtype(tape_prim_id, at::kInt, "tape_prim_id");
    require_rank(tape_prim_id, 1, "tape_prim_id");
    require_contiguous(tape_barycentric, "tape_barycentric");
    require_dtype(tape_barycentric, at::kFloat, "tape_barycentric");
    require_rank(tape_barycentric, 2, "tape_barycentric");
    require_cuda(grad_t, "grad_t");
    require_dtype(grad_t, at::kFloat, "grad_t");
    require_rank(grad_t, 1, "grad_t");
    if (grad_t.size(0) != ray_d.size(0)) {
        throw std::runtime_error("grad_t has the wrong length.");
    }
    SceneCache &scene = get_scene(scene_handle);
    IntersectBackwardOutputs out = intersect_backward_t_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t,
        grad_t.stride(0),
        need_grad_vertices,
        need_grad_ray_o,
        need_grad_ray_d,
        need_grad_ray_tmax);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple intersect_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tangent_vertices,
    at::Tensor tangent_ray_o,
    at::Tensor tangent_ray_d) {
    SceneCache &scene = get_scene(scene_handle);
    IntersectJvpOutputs out = intersect_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices.contiguous(),
        tangent_ray_o.contiguous(),
        tangent_ray_d.contiguous());
    return py::make_tuple(
        out.tangent_t,
        out.tangent_p,
        out.tangent_n,
        out.tangent_geo_n,
        out.tangent_uv,
        out.tangent_barycentric);
}

void bind_intersect_ops(py::module_ &m) {
    m.def("intersect_forward", &intersect_forward_op);
    m.def("intersect_forward_flags", &intersect_forward_flags_op);
    m.def("intersect_forward_ad_flags", &intersect_forward_ad_flags_op);
    m.def("intersect_t_sum_ad", &intersect_t_sum_ad_op);
    m.def("intersect_t_sum_vjp", &intersect_t_sum_vjp_op);
    m.def("intersect_backward", &intersect_backward_op);
    m.def("intersect_backward_t", &intersect_backward_t_op);
    m.def("intersect_jvp", &intersect_jvp_op);
}

} // namespace raydtorch
