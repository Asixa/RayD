// Copyright Xingyu Chen.
// Declares internal scene support for geometry kernels.

#pragma once

#include <ATen/ATen.h>

#include <src/scene/cache.h>

namespace rayd::torch_backend {

struct IntersectForwardOutputs {
    at::Tensor t;
    at::Tensor p;
    at::Tensor n;
    at::Tensor geo_n;
    at::Tensor uv;
    at::Tensor barycentric;
    at::Tensor shape_id;
    at::Tensor prim_id;
    at::Tensor local_prim_id;
    at::Tensor global_prim_id;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    at::Tensor tape_t;
};

IntersectForwardOutputs intersect_forward_cuda(const SceneCache& scene, const at::Tensor& ray_o,
                                               const at::Tensor& ray_d, const at::Tensor& ray_tmax,
                                               const at::Tensor& active);

IntersectForwardOutputs intersect_forward_flags_cuda(const SceneCache& scene, const at::Tensor& ray_o,
                                                     const at::Tensor& ray_d, const at::Tensor& ray_tmax,
                                                     const at::Tensor& active, int64_t flags);

IntersectForwardOutputs intersect_forward_ad_flags_cuda(const SceneCache& scene, const at::Tensor& ray_o,
                                                        const at::Tensor& ray_d, const at::Tensor& ray_tmax,
                                                        const at::Tensor& active, int64_t flags);

IntersectForwardOutputs intersect_forward_tape_cuda(const SceneCache& scene, const at::Tensor& ray_o,
                                                    const at::Tensor& ray_d, const at::Tensor& ray_tmax,
                                                    const at::Tensor& active);

// The t-only AD trace that additionally keeps the OptiX barycentric uv of each hit, so that the remaining intersection
// fields can later be completed without tracing again. `t` and `tape_prim_id` are bitwise those of
// intersect_forward_tape_cuda; the uv lands in tape_barycentric.
IntersectForwardOutputs intersect_forward_staged_cuda(const SceneCache& scene, const at::Tensor& ray_o,
                                                      const at::Tensor& ray_d, const at::Tensor& ray_tmax,
                                                      const at::Tensor& active);

// Runs only the intersection post-pass of intersect_forward_ad_flags_cuda over a staged hit record, producing the
// same non-t fields bitwise. The shape and local primitive ids the trace would have reported are looked up from the
// global primitive id. The returned `t` is the staged one.
IntersectForwardOutputs intersect_complete_staged_cuda(const SceneCache& scene, const at::Tensor& ray_o,
                                                       const at::Tensor& ray_d, const at::Tensor& active,
                                                       const at::Tensor& t, const at::Tensor& global_prim_id,
                                                       const at::Tensor& bary_uv, int64_t flags);

at::Tensor intersect_forward_t_only_cuda(const SceneCache& scene, const at::Tensor& ray_o, const at::Tensor& ray_d,
                                         const at::Tensor& ray_tmax, const at::Tensor& active);

struct IntersectBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
    at::Tensor grad_ray_tmax;
};

IntersectBackwardOutputs intersect_backward_cuda(const at::Tensor& vertices, const at::Tensor& faces,
                                                 const at::Tensor& ray_o, const at::Tensor& ray_d,
                                                 const at::Tensor& ray_tmax, const at::Tensor& active,
                                                 const at::Tensor& tape_prim_id, const at::Tensor& tape_barycentric,
                                                 const at::Tensor& grad_t, const at::Tensor& grad_p,
                                                 const at::Tensor& grad_n, const at::Tensor& grad_geo_n,
                                                 const at::Tensor& grad_uv, const at::Tensor& grad_barycentric,
                                                 const at::Tensor& triangle_uv);

IntersectBackwardOutputs intersect_backward_optional_cuda(
    const at::Tensor& vertices, const at::Tensor& faces, const at::Tensor& ray_o, const at::Tensor& ray_d,
    const at::Tensor& ray_tmax, const at::Tensor& active, const at::Tensor& tape_prim_id,
    const at::Tensor& tape_barycentric, const at::Tensor* grad_t, const at::Tensor* grad_p, const at::Tensor* grad_n,
    const at::Tensor* grad_geo_n, const at::Tensor* grad_uv, const at::Tensor* grad_barycentric,
    bool need_grad_vertices, bool need_grad_ray_o, bool need_grad_ray_d, bool need_grad_ray_tmax,
    const at::Tensor& triangle_uv);

IntersectBackwardOutputs intersect_backward_t_cuda(const at::Tensor& vertices, const at::Tensor& faces,
                                                   const at::Tensor& ray_o, const at::Tensor& ray_d,
                                                   const at::Tensor& active, const at::Tensor& tape_prim_id,
                                                   const at::Tensor& tape_barycentric, const at::Tensor& grad_t,
                                                   int64_t grad_t_stride, bool need_grad_vertices, bool need_grad_ray_o,
                                                   bool need_grad_ray_d, bool need_grad_ray_tmax);

struct IntersectJvpOutputs {
    at::Tensor tangent_t;
    at::Tensor tangent_p;
    at::Tensor tangent_n;
    at::Tensor tangent_geo_n;
    at::Tensor tangent_uv;
    at::Tensor tangent_barycentric;
};

IntersectJvpOutputs intersect_jvp_cuda(const at::Tensor& vertices, const at::Tensor& faces, const at::Tensor& ray_o,
                                       const at::Tensor& ray_d, const at::Tensor& active,
                                       const at::Tensor& tape_prim_id, const at::Tensor& tape_barycentric,
                                       const at::Tensor& tangent_vertices, const at::Tensor& tangent_ray_o,
                                       const at::Tensor& tangent_ray_d, const at::Tensor& triangle_uv);

IntersectJvpOutputs intersect_jvp_optional_cuda(const at::Tensor& vertices, const at::Tensor& faces,
                                                const at::Tensor& ray_o, const at::Tensor& ray_d,
                                                const at::Tensor& active, const at::Tensor& tape_prim_id,
                                                const at::Tensor& tape_barycentric, const at::Tensor* tangent_vertices,
                                                const at::Tensor* tangent_ray_o, const at::Tensor* tangent_ray_d,
                                                int64_t flags, const at::Tensor& triangle_uv);

} // namespace rayd::torch_backend
