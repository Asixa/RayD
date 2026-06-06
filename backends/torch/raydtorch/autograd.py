from __future__ import annotations

import torch

from . import _C
from .types import DfrDirectAccum, Intersection, NearestPointEdge, ReflEpcField, ReflectionChain


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


def visible(scene_handle: int, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    values = _C.visibility_forward(int(scene_handle), start, end, active)
    return values[0]


class _TraceReflectionsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
        max_bounces: int,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.trace_reflections_forward(
            int(scene_handle),
            ray_o,
            ray_d,
            ray_tmax,
            active,
            int(max_bounces),
        )
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, ray_o, ray_d, ray_tmax, active, max_bounces = inputs
        valid, t, image_sources, prim_ids, tape_prim_id, tape_barycentric, tape_t = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        ctx.scene_handle = int(scene_handle)
        ctx.max_bounces = int(max_bounces)
        ctx.save_for_backward(ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_t)
        ctx.save_for_forward(vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, image_sources)
        ctx.mark_non_differentiable(valid, prim_ids, tape_prim_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        grad_t = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(tape_t)
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = _C.trace_reflections_backward(
            ctx.scene_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tape_prim_id,
            tape_barycentric,
            grad_t,
        )
        return None, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, None, None

    @staticmethod
    def jvp(
        ctx,
        grad_scene_handle,
        grad_vertices,
        grad_ray_o,
        grad_ray_d,
        grad_ray_tmax,
        grad_active,
        grad_max_bounces,
    ):
        vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, image_sources = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_ray_o is None:
            grad_ray_o = torch.zeros_like(ray_o)
        if grad_ray_d is None:
            grad_ray_d = torch.zeros_like(ray_d)
        with torch._C._DisableFuncTorch():
            tangent_t, tangent_image_sources = _C.trace_reflections_jvp(
                ctx.scene_handle,
                _native_tensor(ray_o),
                _native_tensor(ray_d),
                _native_tensor(active),
                _native_tensor(tape_prim_id),
                _native_tensor(tape_barycentric),
                _native_tensor(grad_vertices),
                _native_tensor(grad_ray_o),
                _native_tensor(grad_ray_d),
                _native_tensor(image_sources),
            )
        return None, tangent_t, tangent_image_sources, None, None, None, None


def trace_reflections(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor,
    max_bounces: int,
) -> ReflectionChain:
    values = _TraceReflectionsFunction.apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(max_bounces),
    )
    return ReflectionChain(*values[:4])


class _TraceReflEpcFieldFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        source: torch.Tensor,
        receiver: torch.Tensor,
        active: torch.Tensor,
        max_bounces: int,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        return _C.trace_refl_epc_field_forward(
            int(scene_handle),
            source,
            receiver,
            active,
            int(max_bounces),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, source, receiver, active, max_bounces = inputs
        field_real, field_imag, path_length, valid, resolved_prim_ids, tape_prim_id, tape_barycentric, tape_t = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        source = torch.autograd.forward_ad.unpack_dual(source).primal
        receiver = torch.autograd.forward_ad.unpack_dual(receiver).primal
        ctx.scene_handle = int(scene_handle)
        ctx.max_bounces = int(max_bounces)
        ctx.save_for_backward(source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
        ctx.save_for_forward(vertices, source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
        ctx.mark_non_differentiable(valid, resolved_prim_ids, tape_prim_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        grad_field_real = grad_outputs[0].contiguous() if grad_outputs[0] is not None else torch.zeros_like(tape_t)
        grad_field_imag = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(tape_t)
        grad_path_length = grad_outputs[2].contiguous() if grad_outputs[2] is not None else torch.zeros_like(tape_t)
        grad_vertices, grad_source, grad_receiver = _C.trace_refl_epc_field_backward(
            ctx.scene_handle,
            source,
            receiver,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_t,
            grad_field_real,
            grad_field_imag,
            grad_path_length,
        )
        return None, grad_vertices, grad_source, grad_receiver, None, None

    @staticmethod
    def jvp(ctx, grad_scene_handle, grad_vertices, grad_source, grad_receiver, grad_active, grad_max_bounces):
        vertices, source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_source is None:
            grad_source = torch.zeros_like(source)
        if grad_receiver is None:
            grad_receiver = torch.zeros_like(receiver)
        with torch._C._DisableFuncTorch():
            tangent_field_real, tangent_field_imag, tangent_path_length = _C.trace_refl_epc_field_jvp(
                ctx.scene_handle,
                _native_tensor(source),
                _native_tensor(receiver),
                _native_tensor(active),
                _native_tensor(tape_prim_id),
                _native_tensor(tape_barycentric),
                _native_tensor(tape_t),
                _native_tensor(grad_vertices),
                _native_tensor(grad_source),
                _native_tensor(grad_receiver),
            )
        return tangent_field_real, tangent_field_imag, tangent_path_length, None, None, None, None, None


def trace_refl_epc_field(
    scene_handle: int,
    vertices: torch.Tensor,
    source: torch.Tensor,
    receiver: torch.Tensor,
    active: torch.Tensor,
    max_bounces: int,
) -> ReflEpcField:
    values = _TraceReflEpcFieldFunction.apply(
        scene_handle,
        vertices,
        source,
        receiver,
        active,
        int(max_bounces),
    )
    return ReflEpcField(*values[:5])


class _AccumDfrDirectFunction(torch.autograd.Function):
    @staticmethod
    def forward(edge_pos: torch.Tensor, edge_dir: torch.Tensor, src: torch.Tensor):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        return _C.accum_dfr_direct_forward(edge_pos, edge_dir, src)

    @staticmethod
    def setup_context(ctx, inputs, output):
        edge_pos, edge_dir, src = inputs
        edge_pos = torch.autograd.forward_ad.unpack_dual(edge_pos).primal
        edge_dir = torch.autograd.forward_ad.unpack_dual(edge_dir).primal
        src = torch.autograd.forward_ad.unpack_dual(src).primal
        ctx.save_for_backward(edge_pos, edge_dir, src)
        ctx.save_for_forward(edge_pos, edge_dir, src)

    @staticmethod
    def backward(ctx, *grad_outputs):
        edge_pos, edge_dir, src = ctx.saved_tensors
        grad_power = grad_outputs[0].contiguous() if grad_outputs[0] is not None else torch.zeros((edge_pos.shape[0],), device=edge_pos.device, dtype=edge_pos.dtype)
        grad_field_x_re = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(grad_power)
        grad_field_x_im = grad_outputs[2].contiguous() if grad_outputs[2] is not None else torch.zeros_like(grad_power)
        grad_edge_pos, grad_edge_dir, grad_src = _C.accum_dfr_direct_backward(
            edge_pos,
            edge_dir,
            src,
            grad_power,
            grad_field_x_re,
            grad_field_x_im,
        )
        return grad_edge_pos, grad_edge_dir, grad_src

    @staticmethod
    def jvp(ctx, grad_edge_pos, grad_edge_dir, grad_src):
        edge_pos, edge_dir, src = ctx.saved_tensors
        if grad_edge_pos is None:
            grad_edge_pos = torch.zeros_like(edge_pos)
        if grad_edge_dir is None:
            grad_edge_dir = torch.zeros_like(edge_dir)
        if grad_src is None:
            grad_src = torch.zeros_like(src)
        with torch._C._DisableFuncTorch():
            return _C.accum_dfr_direct_jvp(
                _native_tensor(edge_pos),
                _native_tensor(edge_dir),
                _native_tensor(src),
                _native_tensor(grad_edge_pos),
                _native_tensor(grad_edge_dir),
                _native_tensor(grad_src),
            )


def accum_dfr_direct(edge_pos: torch.Tensor, edge_dir: torch.Tensor, src: torch.Tensor) -> DfrDirectAccum:
    values = _AccumDfrDirectFunction.apply(edge_pos, edge_dir, src)
    return DfrDirectAccum(*values)


class NativeOpUnavailable(RuntimeError):
    pass
