from __future__ import annotations

import torch

from . import _C
from .types import Intersection, NearestPointEdge


def _native_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if torch._C._functorch.is_functorch_wrapped_tensor(value) or torch._C._functorch.is_gradtrackingtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    return value


class _IntersectFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.intersect_forward(int(scene_handle), ray_o, ray_d, ray_tmax, active)
        return outputs[:10]

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, ray_o, ray_d, ray_tmax, active = inputs
        t, _p, _n, _geo_n, _uv, barycentric, shape_id, prim_id, local_prim_id, global_prim_id = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        ctx.scene_handle = int(scene_handle)
        ctx.save_for_backward(ray_o, ray_d, ray_tmax, active, global_prim_id, barycentric, t)
        ctx.save_for_forward(vertices, ray_o, ray_d, active, global_prim_id, barycentric, t)
        ctx.mark_non_differentiable(shape_id, prim_id, local_prim_id, global_prim_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        grad_t = grad_outputs[0].contiguous() if grad_outputs[0] is not None else torch.zeros_like(tape_t)
        grad_p = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(ray_o)
        grad_n = grad_outputs[2].contiguous() if grad_outputs[2] is not None else torch.zeros_like(ray_o)
        grad_geo_n = grad_outputs[3].contiguous() if grad_outputs[3] is not None else torch.zeros_like(ray_o)
        grad_uv = (
            grad_outputs[4].contiguous()
            if grad_outputs[4] is not None
            else torch.zeros((ray_o.shape[0], 2), device=ray_o.device, dtype=ray_o.dtype)
        )
        grad_barycentric = (
            grad_outputs[5].contiguous() if grad_outputs[5] is not None else torch.zeros_like(tape_barycentric)
        )
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = _C.intersect_backward(
            ctx.scene_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tape_prim_id,
            tape_barycentric,
            grad_t,
            grad_p,
            grad_n,
            grad_geo_n,
            grad_uv,
            grad_barycentric,
        )
        return None, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, None

    @staticmethod
    def jvp(ctx, grad_scene_handle, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, grad_active):
        vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, _tape_t = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_ray_o is None:
            grad_ray_o = torch.zeros_like(ray_o)
        if grad_ray_d is None:
            grad_ray_d = torch.zeros_like(ray_d)
        with torch._C._DisableFuncTorch():
            values = _C.intersect_jvp(
                ctx.scene_handle,
                _native_tensor(ray_o),
                _native_tensor(ray_d),
                _native_tensor(active),
                _native_tensor(tape_prim_id),
                _native_tensor(tape_barycentric),
                _native_tensor(grad_vertices),
                _native_tensor(grad_ray_o),
                _native_tensor(grad_ray_d),
            )
        tangent_t, tangent_p, tangent_n, tangent_geo_n, tangent_uv, tangent_barycentric = values
        return tangent_t, tangent_p, tangent_n, tangent_geo_n, tangent_uv, tangent_barycentric, None, None, None, None


def intersect(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor,
) -> Intersection:
    values = _IntersectFunction.apply(scene_handle, vertices, ray_o, ray_d, ray_tmax, active)
    return Intersection(*values)


class _NearestEdgeFunction(torch.autograd.Function):
    @staticmethod
    def forward(scene_handle: int, vertices: torch.Tensor, point: torch.Tensor):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.nearest_edge_forward(int(scene_handle), point)
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, point = inputs
        distance, edge_point, edge_t, shape_id, edge_id, global_edge_id, tape_edge_id, tape_s, tape_d = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        point = torch.autograd.forward_ad.unpack_dual(point).primal
        ctx.scene_handle = int(scene_handle)
        ctx.save_for_backward(point, tape_edge_id, tape_s, tape_d, distance)
        ctx.save_for_forward(vertices, point, tape_edge_id, tape_s, tape_d)
        ctx.mark_non_differentiable(shape_id, edge_id, global_edge_id, tape_edge_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        point, tape_edge_id, tape_s, tape_d, distance = ctx.saved_tensors
        grad_distance = grad_outputs[0].contiguous() if grad_outputs[0] is not None else torch.zeros_like(distance)
        grad_edge_point = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(point)
        grad_edge_t = grad_outputs[2].contiguous() if grad_outputs[2] is not None else torch.zeros_like(distance)
        grad_vertices, grad_point = _C.nearest_edge_backward(
            ctx.scene_handle,
            point,
            tape_edge_id,
            tape_s,
            tape_d,
            grad_distance,
            grad_edge_point,
            grad_edge_t,
        )
        return None, grad_vertices, grad_point

    @staticmethod
    def jvp(ctx, grad_scene_handle, grad_vertices, grad_point):
        vertices, point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_point is None:
            grad_point = torch.zeros_like(point)
        with torch._C._DisableFuncTorch():
            tangent_distance, tangent_edge_point, tangent_edge_t = _C.nearest_edge_jvp(
                ctx.scene_handle,
                _native_tensor(point),
                _native_tensor(tape_edge_id),
                _native_tensor(tape_s),
                _native_tensor(tape_d),
                _native_tensor(grad_vertices),
                _native_tensor(grad_point),
            )
        return tangent_distance, tangent_edge_point, tangent_edge_t, None, None, None, None, None, None


def nearest_edge(scene_handle: int, vertices: torch.Tensor, point: torch.Tensor) -> NearestPointEdge:
    values = _NearestEdgeFunction.apply(scene_handle, vertices, point)
    return NearestPointEdge(*values[:6])


class NativeOpUnavailable(RuntimeError):
    pass
