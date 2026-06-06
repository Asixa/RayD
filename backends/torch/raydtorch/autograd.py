from __future__ import annotations

import torch

from . import _C
from .types import Intersection


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


class NativeOpUnavailable(RuntimeError):
    pass
