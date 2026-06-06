from __future__ import annotations

import torch

from . import _C
from .types import Intersection


class _IntersectFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scene_handle: int,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.intersect_forward(int(scene_handle), ray_o, ray_d, ray_tmax, active)
        (
            t,
            p,
            n,
            geo_n,
            uv,
            barycentric,
            shape_id,
            prim_id,
            local_prim_id,
            global_prim_id,
            tape_prim_id,
            tape_barycentric,
            tape_t,
        ) = outputs
        ctx.scene_handle = int(scene_handle)
        ctx.save_for_backward(ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_t)
        ctx.mark_non_differentiable(shape_id, prim_id, local_prim_id, global_prim_id)
        return t, p, n, geo_n, uv, barycentric, shape_id, prim_id, local_prim_id, global_prim_id

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("intersect backward is implemented in Task 7.")


def intersect(
    scene_handle: int,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor,
) -> Intersection:
    values = _IntersectFunction.apply(scene_handle, ray_o, ray_d, ray_tmax, active)
    return Intersection(*values)


class NativeOpUnavailable(RuntimeError):
    pass
