from __future__ import annotations

import torch

from . import _NATIVE_AVAILABLE
from . import _legacy
from .types import (
    DfrAccum,
    DfrCoherentAccum,
    DfrGrid,
    DfrMaterial,
    DfrPaths,
    DfrStates,
    Intersection,
    NearestPointEdge,
    NearestEdgesTopK,
    NearestRayEdge,
    ReflEpcField,
    ReflectionChain,
)


def _require_native_dispatcher() -> None:
    """Fail loudly when the `torch.ops.rayd_torch` dispatcher never loaded.

    Keyed on `_NATIVE_AVAILABLE`, not on the `_C` metadata module: the
    dispatcher can be loaded while `_C` is absent (RAYD_TORCH_LEGACY_LIBRARY)
    and `_C` says nothing about dispatcher availability.
    """
    if not _NATIVE_AVAILABLE:
        raise RuntimeError(
            "RayD Torch native dispatcher is unavailable: the _legacy_ops "
            "library did not load, so torch.ops.rayd_torch has no operators."
        ) from _legacy.LOAD_ERROR


def _native_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if torch._C._functorch.is_functorch_wrapped_tensor(value) or torch._C._functorch.is_gradtrackingtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    return value


def _native_tangent_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    value = _native_tensor(value)
    if value is None:
        return None
    try:
        value.data_ptr()
    except RuntimeError:
        return None
    return value


def _needs_reverse_or_forward_ad(*values: torch.Tensor | None) -> bool:
    if torch.autograd.forward_ad._current_level < 0:
        return any(value is not None and value.requires_grad for value in values)
    for value in values:
        if value is None:
            continue
        unpacked = torch.autograd.forward_ad.unpack_dual(value)
        if unpacked.primal.requires_grad or unpacked.tangent is not None:
            return True
    return False


def _needs_forward_ad(*values: torch.Tensor | None) -> bool:
    if torch.autograd.forward_ad._current_level < 0:
        return False
    for value in values:
        if value is None:
            continue
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


_RAY_FLAG_ALL = 0x01 | 0x02 | 0x04


def _has_grad(value: torch.Tensor | None) -> bool:
    return value is not None and value.numel() != 0


def _active_ctx_tensor(active: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if active is not None:
        return active
    return torch.empty((0,), device=like.device, dtype=torch.bool)


def _save_optional_tensor(value: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if value is not None:
        return value
    return torch.empty((0, 3), device=like.device, dtype=like.dtype)


def _named_autograd_function(function_cls, name: str):
    """Restore the public identity of a factory-built autograd Function.

    The class statement inside a factory carries the factory-local name, so the
    autograd node produced by ``FunctionMeta`` would otherwise report that name.
    """
    function_cls.__name__ = name
    function_cls.__qualname__ = name
    # `_backward_cls` is a private torch attribute; if a torch release drops
    # it, the backward node keeps its cosmetic default name instead of the
    # import failing.
    backward_cls = getattr(function_cls, "_backward_cls", None)
    if backward_cls is not None:
        backward_cls.__name__ = f"{name}Backward"
        backward_cls.__qualname__ = f"{name}Backward"
    return function_cls


def _make_intersect_function(fused: bool):
    """Build the ray-triangle intersection autograd Function.

    ``fused=True`` takes a single scene-global vertex tensor at apply index 1;
    ``fused=False`` takes one vertex tensor per mesh as trailing varargs.
    """
    vertex_at = 1 if fused else None
    ray_o_at = 2 if fused else 1
    ray_d_at = ray_o_at + 1
    ray_tmax_at = ray_o_at + 2
    active_at = ray_o_at + 3
    flags_at = ray_o_at + 4
    mesh_at = 6

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(*args):
            _require_native_dispatcher()
            outputs = torch.ops.rayd_torch.intersect_forward_ad_flags(
                args[0], args[ray_o_at], args[ray_d_at], args[ray_tmax_at], args[active_at], int(args[flags_at])
            )
            return tuple(outputs[:12]) + (outputs[13],)

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)
            (
                t,
                _p,
                _n,
                _geo_n,
                _uv,
                _barycentric,
                shape_id,
                prim_id,
                local_prim_id,
                global_prim_id,
                tape_prim_id,
                tape_barycentric,
                active,
            ) = output
            ray_o = torch.autograd.forward_ad.unpack_dual(inputs[ray_o_at]).primal
            ray_d = torch.autograd.forward_ad.unpack_dual(inputs[ray_d_at]).primal
            ray_tmax = torch.autograd.forward_ad.unpack_dual(inputs[ray_tmax_at]).primal
            ctx.scene = inputs[0]
            ctx.flags = int(inputs[flags_at])
            ctx.save_for_backward(ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, t)
            if fused:
                vertices = torch.autograd.forward_ad.unpack_dual(inputs[vertex_at]).primal
                ctx.save_for_forward(vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, t)
            else:
                ctx.mesh_count = len(inputs) - mesh_at
                ctx.save_for_forward(ray_o, ray_d, active, tape_prim_id, tape_barycentric, t)
            ctx.mark_non_differentiable(shape_id, prim_id, local_prim_id, global_prim_id, tape_prim_id, tape_barycentric, active)

        @staticmethod
        def backward(ctx, *grad_outputs):
            ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
            grad_t = grad_outputs[0]
            if fused:
                need_grad_vertices = bool(ctx.needs_input_grad[vertex_at])
            else:
                needs_mesh_grad = tuple(bool(value) for value in ctx.needs_input_grad[mesh_at : mesh_at + ctx.mesh_count])
                need_grad_vertices = any(needs_mesh_grad)
            need_grad_ray_o = bool(ctx.needs_input_grad[ray_o_at])
            need_grad_ray_d = bool(ctx.needs_input_grad[ray_d_at])
            need_grad_ray_tmax = bool(ctx.needs_input_grad[ray_tmax_at])
            only_t_grad = all(not _has_grad(value) for value in grad_outputs[1:6])
            if ctx.flags == 0 or only_t_grad:
                if not (
                    _has_grad(grad_t)
                    and (need_grad_vertices or need_grad_ray_o or need_grad_ray_d or need_grad_ray_tmax)
                ):
                    return (None,) * (7 if fused else 6 + ctx.mesh_count)
                grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = torch.ops.rayd_torch.intersect_backward_t(
                    ctx.scene,
                    ray_o,
                    ray_d,
                    active,
                    tape_prim_id,
                    tape_barycentric,
                    grad_t,
                    need_grad_vertices,
                    need_grad_ray_o,
                    need_grad_ray_d,
                    need_grad_ray_tmax,
                )
            else:
                grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = torch.ops.rayd_torch.intersect_backward_optional(
                    ctx.scene,
                    ray_o,
                    ray_d,
                    ray_tmax,
                    active,
                    tape_prim_id,
                    tape_barycentric,
                    grad_t,
                    grad_outputs[1],
                    grad_outputs[2],
                    grad_outputs[3],
                    grad_outputs[4],
                    grad_outputs[5],
                    need_grad_vertices,
                    need_grad_ray_o,
                    need_grad_ray_d,
                    need_grad_ray_tmax,
                )
            if fused:
                return (
                    None,
                    grad_vertices if need_grad_vertices else None,
                    grad_ray_o if need_grad_ray_o else None,
                    grad_ray_d if need_grad_ray_d else None,
                    grad_ray_tmax if need_grad_ray_tmax else None,
                    None,
                    None,
                )
            if need_grad_vertices:
                mesh_grad_tuple = torch.ops.rayd_torch.split_scene_vertex_grad(ctx.scene, grad_vertices)
                mesh_grads = tuple(mesh_grad_tuple[i] if needs_mesh_grad[i] else None for i in range(ctx.mesh_count))
            else:
                mesh_grads = (None,) * ctx.mesh_count
            return (
                None,
                grad_ray_o if need_grad_ray_o else None,
                grad_ray_d if need_grad_ray_d else None,
                grad_ray_tmax if need_grad_ray_tmax else None,
                None,
                None,
                *mesh_grads,
            )

        @staticmethod
        def jvp(ctx, *tangents):
            if fused:
                vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, _tape_t = ctx.saved_tensors
            else:
                ray_o, ray_d, active, tape_prim_id, tape_barycentric, _tape_t = ctx.saved_tensors
                native_mesh_tangents = tuple(_native_tangent_or_none(value) for value in tangents[mesh_at:])
                with torch._C._DisableFuncTorch():
                    packed_tangents = torch.ops.rayd_torch.pack_scene_vertex_tangents(ctx.scene, list(native_mesh_tangents))
            with torch._C._DisableFuncTorch():
                values = torch.ops.rayd_torch.intersect_jvp_optional(
                    ctx.scene,
                    _native_tensor(ray_o),
                    _native_tensor(ray_d),
                    _native_tensor(active),
                    _native_tensor(tape_prim_id),
                    _native_tensor(tape_barycentric),
                    _native_tangent_or_none(tangents[vertex_at]) if fused else _native_tensor(packed_tangents),
                    _native_tangent_or_none(tangents[ray_o_at]),
                    _native_tangent_or_none(tangents[ray_d_at]),
                    ctx.flags,
                )
            tangent_t, tangent_p, tangent_n, tangent_geo_n, tangent_uv, tangent_barycentric = values
            return (
                tangent_t,
                tangent_p,
                tangent_n,
                tangent_geo_n,
                tangent_uv,
                tangent_barycentric,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    return _named_autograd_function(
        _Fn, "_IntersectFunction" if fused else "_IntersectMeshesFunction"
    )


_IntersectFunction = _make_intersect_function(True)
_IntersectMeshesFunction = _make_intersect_function(False)


def intersect(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    flags: int = _RAY_FLAG_ALL,
    mesh_vertices: tuple[torch.Tensor, ...] | None = None,
) -> Intersection:
    tracked_vertices = (vertices,) if mesh_vertices is None else tuple(mesh_vertices)
    if len(tracked_vertices) > 1:
        values = _IntersectMeshesFunction.apply(
            scene_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            int(flags),
            *tracked_vertices,
        )
        return Intersection(*values[:10])
    if not _needs_forward_ad(vertices, ray_o, ray_d, ray_tmax):
        values = torch.ops.rayd_torch.intersect_ad_flags(scene_handle, vertices, ray_o, ray_d, ray_tmax, active, int(flags))
        return Intersection(*values)
    values = _IntersectFunction.apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(flags),
    )
    return Intersection(*values[:10])


def _contiguous_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    """Make an upstream gradient or tangent readable by the SDF dispatcher.

    The native operation reads these buffers with contiguous arithmetic, while
    autograd hands out an expanded stride-0 gradient for the common
    `output.sum()` objective.
    """
    if value is None:
        return None
    return value.contiguous()


class _SdfIntersectFunction(torch.autograd.Function):
    """ADR-0037 SDF ray intersection under the frozen-winner IFT.

    The six differentiable inputs come first so `ctx.needs_input_grad` lines up
    with the native gradient request; the four trailing march parameters are
    host scalars and carry no derivative. Backward and JVP consume the frozen
    tape and never re-march.
    """

    @staticmethod
    def forward(values, position, rotation, scale, origins, directions, tmax, max_steps, relaxation, eps_hit):
        _require_native_dispatcher()
        return tuple(torch.ops.rayd_torch.sdf_intersect_forward(
            values,
            position,
            rotation,
            scale,
            origins,
            directions,
            tmax,
            max_steps,
            relaxation,
            eps_hit,
        ))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        _t, hit_mask, _position, _normal, steps, tape_t, tape_base = output
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:6]
        ) + (tape_t, hit_mask, tape_base)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(hit_mask, steps, tape_t, tape_base)

    @staticmethod
    def backward(ctx, *grad_outputs):
        values, position, rotation, scale, origins, directions, tape_t, tape_hit, tape_base = ctx.saved_tensors
        grads = torch.ops.rayd_torch.sdf_intersect_backward(
            values,
            position,
            rotation,
            scale,
            origins,
            directions,
            tape_t,
            tape_hit,
            tape_base,
            _contiguous_or_none(grad_outputs[0]),
            _contiguous_or_none(grad_outputs[2]),
            _contiguous_or_none(grad_outputs[3]),
            *(bool(flag) for flag in ctx.needs_input_grad[:6]),
        )
        return (*grads, None, None, None, None)

    @staticmethod
    def jvp(ctx, *tangents):
        values, position, rotation, scale, origins, directions, tape_t, tape_hit, tape_base = ctx.saved_tensors
        with torch._C._DisableFuncTorch():
            tangent_t, tangent_position, tangent_normal = torch.ops.rayd_torch.sdf_intersect_jvp(
                _native_tensor(values),
                _native_tensor(position),
                _native_tensor(rotation),
                _native_tensor(scale),
                _native_tensor(origins),
                _native_tensor(directions),
                _native_tensor(tape_t),
                _native_tensor(tape_hit),
                _native_tensor(tape_base),
                *(_contiguous_or_none(_native_tangent_or_none(value)) for value in tangents[:6]),
            )
        return tangent_t, None, tangent_position, tangent_normal, None, None, None


def _make_nearest_edge_function(fused: bool):
    """Build the point nearest-edge autograd Function.

    ``fused=True`` takes a single scene-global vertex tensor at apply index 1;
    ``fused=False`` takes one vertex tensor per mesh as trailing varargs.
    """
    vertex_at = 1 if fused else None
    point_at = 2 if fused else 1
    mesh_at = 2

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(*args):
            _require_native_dispatcher()
            return tuple(torch.ops.rayd_torch.nearest_edge_forward(args[0], args[point_at]))

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)
            distance, edge_point, edge_t, shape_id, edge_id, global_edge_id, tape_edge_id, tape_s, tape_d = output
            point = torch.autograd.forward_ad.unpack_dual(inputs[point_at]).primal
            ctx.scene = inputs[0]
            ctx.save_for_backward(point, tape_edge_id, tape_s, tape_d, distance)
            if fused:
                vertices = torch.autograd.forward_ad.unpack_dual(inputs[vertex_at]).primal
                ctx.save_for_forward(vertices, point, tape_edge_id, tape_s, tape_d)
            else:
                ctx.mesh_count = len(inputs) - mesh_at
                ctx.save_for_forward(point, tape_edge_id, tape_s, tape_d)
            ctx.mark_non_differentiable(shape_id, edge_id, global_edge_id, tape_edge_id)

        @staticmethod
        def backward(ctx, *grad_outputs):
            point, tape_edge_id, tape_s, tape_d, distance = ctx.saved_tensors
            if not fused:
                needs_mesh_grad = tuple(bool(value) for value in ctx.needs_input_grad[mesh_at : mesh_at + ctx.mesh_count])
                need_grad_vertices = any(needs_mesh_grad)
                need_grad_point = bool(ctx.needs_input_grad[point_at])
            grad_vertices, grad_point = torch.ops.rayd_torch.nearest_edge_backward_optional(
                ctx.scene,
                point,
                tape_edge_id,
                tape_s,
                tape_d,
                grad_outputs[0],
                grad_outputs[1],
                grad_outputs[2],
                grad_outputs[7] if len(grad_outputs) > 7 else None,
            )
            if fused:
                return None, grad_vertices, grad_point
            if need_grad_vertices:
                mesh_grad_tuple = torch.ops.rayd_torch.split_scene_vertex_grad(ctx.scene, grad_vertices)
                mesh_grads = tuple(mesh_grad_tuple[i] if needs_mesh_grad[i] else None for i in range(ctx.mesh_count))
            else:
                mesh_grads = (None,) * ctx.mesh_count
            return (None, grad_point if need_grad_point else None, *mesh_grads)

        @staticmethod
        def jvp(ctx, *tangents):
            if fused:
                vertices, point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
            else:
                point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
                native_mesh_tangents = tuple(_native_tangent_or_none(value) for value in tangents[mesh_at:])
                with torch._C._DisableFuncTorch():
                    packed_tangents = torch.ops.rayd_torch.pack_scene_vertex_tangents(ctx.scene, list(native_mesh_tangents))
            with torch._C._DisableFuncTorch():
                tangent_distance, tangent_edge_point, tangent_edge_t, tangent_tape_s, tangent_tape_d = torch.ops.rayd_torch.nearest_edge_jvp_optional(
                    ctx.scene,
                    _native_tensor(point),
                    _native_tensor(tape_edge_id),
                    _native_tensor(tape_s),
                    _native_tensor(tape_d),
                    _native_tangent_or_none(tangents[vertex_at]) if fused else _native_tensor(packed_tangents),
                    _native_tangent_or_none(tangents[point_at]),
                )
            return (
                tangent_distance,
                tangent_edge_point,
                tangent_edge_t,
                None,
                None,
                None,
                None,
                tangent_tape_s,
                tangent_tape_d,
            )

    return _named_autograd_function(
        _Fn, "_NearestEdgeFunction" if fused else "_NearestEdgeMeshesFunction"
    )


_NearestEdgeFunction = _make_nearest_edge_function(True)
_NearestEdgeMeshesFunction = _make_nearest_edge_function(False)


def _needs_nearest_edge_ad(point: torch.Tensor, *vertices: torch.Tensor) -> bool:
    for value in (*vertices, point):
        if value.requires_grad:
            return True
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


def nearest_edge(
    scene_handle: int,
    vertices: torch.Tensor,
    point: torch.Tensor,
    mesh_vertices: tuple[torch.Tensor, ...] | None = None,
) -> NearestPointEdge:
    _require_native_dispatcher()
    tracked_vertices = (vertices,) if mesh_vertices is None else tuple(mesh_vertices)
    if not _needs_nearest_edge_ad(point, *tracked_vertices):
        values = torch.ops.rayd_torch.nearest_edge_forward_noad(scene_handle, point)
        return NearestPointEdge(*values)
    if len(tracked_vertices) > 1:
        values = _NearestEdgeMeshesFunction.apply(scene_handle, point, *tracked_vertices)
        return NearestPointEdge(*values[:6])
    values = _NearestEdgeFunction.apply(scene_handle, vertices, point)
    return NearestPointEdge(*values[:6])


def _flatten_topk_point(point: torch.Tensor, k: int) -> torch.Tensor:
    return point.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)


def _reshape_topk_optional(value: torch.Tensor | None, trailing: int | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if trailing is None:
        return value.reshape(-1)
    return value.reshape(-1, trailing)


def _make_nearest_edges_topk_function(fused: bool):
    """Build the top-k nearest-edge autograd Function.

    ``fused=True`` takes a single scene-global vertex tensor at apply index 1;
    ``fused=False`` takes one vertex tensor per mesh as trailing varargs.
    """
    vertex_at = 1 if fused else None
    point_at = 2 if fused else 1
    k_at = 3 if fused else 2
    active_at = 4 if fused else 3
    mesh_at = 4

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(*args):
            _require_native_dispatcher()
            return tuple(
                torch.ops.rayd_torch.nearest_edges_topk_forward(
                    args[0], args[point_at], int(args[k_at]), args[active_at]
                )
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)
            tape_edge_id, tape_s, tape_d = output[9:12]
            point = torch.autograd.forward_ad.unpack_dual(inputs[point_at]).primal
            ctx.scene = inputs[0]
            ctx.k = int(inputs[k_at])
            ctx.save_for_backward(point, tape_edge_id, tape_s, tape_d)
            if fused:
                vertices = torch.autograd.forward_ad.unpack_dual(inputs[vertex_at]).primal
                ctx.save_for_forward(vertices, point, tape_edge_id, tape_s, tape_d)
            else:
                ctx.mesh_count = len(inputs) - mesh_at
                ctx.save_for_forward(point, tape_edge_id, tape_s, tape_d)
            ctx.mark_non_differentiable(output[0], output[5], output[6], output[7], output[8], tape_edge_id)

        @staticmethod
        def backward(ctx, *grad_outputs):
            point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
            k = ctx.k
            if not fused:
                needs_mesh_grad = tuple(bool(value) for value in ctx.needs_input_grad[mesh_at : mesh_at + ctx.mesh_count])
                need_grad_vertices = any(needs_mesh_grad)
                need_grad_point = bool(ctx.needs_input_grad[point_at])
            grad_vertices, grad_point_slots = torch.ops.rayd_torch.nearest_edge_backward_optional(
                ctx.scene,
                _flatten_topk_point(point, k),
                tape_edge_id.reshape(-1),
                tape_s.reshape(-1),
                tape_d.reshape(-1, 3),
                _reshape_topk_optional(grad_outputs[1]),
                _reshape_topk_optional(grad_outputs[4], 3),
                _reshape_topk_optional(grad_outputs[3]),
                _reshape_topk_optional(grad_outputs[10]),
            )
            grad_point = grad_point_slots.reshape(point.shape[0], k, 3).sum(dim=1)
            if _has_grad(grad_outputs[2]):
                grad_point = grad_point + grad_outputs[2].sum(dim=1)
            if fused:
                return None, grad_vertices, grad_point, None, None
            if need_grad_vertices:
                split_grad = torch.ops.rayd_torch.split_scene_vertex_grad(ctx.scene, grad_vertices)
                mesh_grads = tuple(split_grad[i] if needs_mesh_grad[i] else None for i in range(ctx.mesh_count))
            else:
                mesh_grads = (None,) * ctx.mesh_count
            return (None, grad_point if need_grad_point else None, None, None, *mesh_grads)

        @staticmethod
        def jvp(ctx, *tangents):
            if fused:
                vertices, point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
            else:
                point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
                native_mesh_tangents = tuple(_native_tangent_or_none(value) for value in tangents[mesh_at:])
            k = ctx.k
            grad_point = tangents[point_at]
            with torch._C._DisableFuncTorch():
                if not fused:
                    packed_tangents = torch.ops.rayd_torch.pack_scene_vertex_tangents(
                        ctx.scene, list(native_mesh_tangents)
                    )
                values = torch.ops.rayd_torch.nearest_edge_jvp_optional(
                    ctx.scene,
                    _native_tensor(_flatten_topk_point(point, k)),
                    _native_tensor(tape_edge_id.reshape(-1)),
                    _native_tensor(tape_s.reshape(-1)),
                    _native_tensor(tape_d.reshape(-1, 3)),
                    _native_tangent_or_none(tangents[vertex_at]) if fused else _native_tensor(packed_tangents),
                    _native_tangent_or_none(
                        None if grad_point is None else _flatten_topk_point(grad_point, k)
                    ),
                )
            tangent_distance, tangent_edge_point, tangent_edge_t, tangent_tape_s, tangent_tape_d = values
            if grad_point is None:
                tangent_points = torch.zeros_like(point).unsqueeze(1).expand(-1, k, -1)
            else:
                tangent_points = grad_point.unsqueeze(1).expand(-1, k, -1)
            tangent_points = tangent_points * (tape_edge_id >= 0).unsqueeze(-1)
            return (
                None,
                tangent_distance.reshape(point.shape[0], k),
                tangent_points,
                tangent_edge_t.reshape(point.shape[0], k),
                tangent_edge_point.reshape(point.shape[0], k, 3),
                None,
                None,
                None,
                None,
                None,
                tangent_tape_s.reshape(point.shape[0], k),
                tangent_tape_d.reshape(point.shape[0], k, 3),
            )

    return _named_autograd_function(
        _Fn, "_NearestEdgesTopKFunction" if fused else "_NearestEdgesTopKMeshesFunction"
    )


_NearestEdgesTopKFunction = _make_nearest_edges_topk_function(True)
_NearestEdgesTopKMeshesFunction = _make_nearest_edges_topk_function(False)


def nearest_edges(
    scene_handle: int,
    vertices: torch.Tensor,
    point: torch.Tensor,
    k: int,
    active: torch.Tensor | None,
    mesh_vertices: tuple[torch.Tensor, ...] | None = None,
) -> NearestEdgesTopK:
    _require_native_dispatcher()
    if not 1 <= int(k) <= 16:
        raise ValueError("k must be in [1, 16].")
    active_arg = _active_ctx_tensor(active, point)
    tracked_vertices = (vertices,) if mesh_vertices is None else tuple(mesh_vertices)
    if not _needs_nearest_edge_ad(point, *tracked_vertices):
        values = torch.ops.rayd_torch.nearest_edges_topk_forward(
            scene_handle, point, int(k), active_arg
        )
    elif len(tracked_vertices) > 1:
        values = _NearestEdgesTopKMeshesFunction.apply(
            scene_handle, point, int(k), active_arg, *tracked_vertices
        )
    else:
        values = _NearestEdgesTopKFunction.apply(
            scene_handle, vertices, point, int(k), active_arg
        )
    return NearestEdgesTopK(int(point.shape[0]), int(k), *values[:9])


class _NearestEdgeRayFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
    ):
        _require_native_dispatcher()
        return tuple(torch.ops.rayd_torch.nearest_edge_ray_forward(scene_handle, ray_o, ray_d, ray_tmax, active))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        scene_handle, vertices, ray_o, ray_d, ray_tmax, _active = inputs
        distance, ray_t, point, edge_t, edge_point, shape_id, edge_id, global_edge_id, tape_edge_id = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        ctx.scene = scene_handle
        ctx.save_for_backward(vertices, ray_o, ray_d, ray_tmax, tape_edge_id, ray_t, edge_t)
        ctx.save_for_forward(vertices, ray_o, ray_d, ray_tmax, tape_edge_id, ray_t, edge_t)
        ctx.mark_non_differentiable(shape_id, edge_id, global_edge_id, tape_edge_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        vertices, ray_o, ray_d, ray_tmax, tape_edge_id, ray_t, edge_t = ctx.saved_tensors
        grad_vertices, grad_ray_o, grad_ray_d = torch.ops.rayd_torch.nearest_edge_ray_backward_optional(
            ctx.scene,
            ray_o,
            ray_d,
            ray_tmax,
            tape_edge_id,
            ray_t,
            edge_t,
            grad_outputs[0],
            grad_outputs[1],
            grad_outputs[2],
            grad_outputs[3],
            grad_outputs[4],
        )
        if not ctx.needs_input_grad[1]:
            grad_vertices = None
        elif grad_vertices.shape != vertices.shape:
            grad_vertices = grad_vertices[: vertices.shape[0]]
        return None, grad_vertices, grad_ray_o, grad_ray_d, None, None

    @staticmethod
    def jvp(
        ctx,
        grad_scene_handle,
        grad_vertices,
        grad_ray_o,
        grad_ray_d,
        grad_ray_tmax,
        grad_active,
    ):
        vertices, ray_o, ray_d, ray_tmax, tape_edge_id, ray_t, edge_t = ctx.saved_tensors
        with torch._C._DisableFuncTorch():
            values = torch.ops.rayd_torch.nearest_edge_ray_jvp_optional(
                ctx.scene,
                _native_tensor(ray_o),
                _native_tensor(ray_d),
                _native_tensor(ray_tmax),
                _native_tensor(tape_edge_id),
                _native_tensor(ray_t),
                _native_tensor(edge_t),
                _native_tangent_or_none(grad_vertices),
                _native_tangent_or_none(grad_ray_o),
                _native_tangent_or_none(grad_ray_d),
            )
        return (*values, None, None, None, None)


def nearest_edge_ray(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
) -> NearestRayEdge:
    values = _NearestEdgeRayFunction.apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        _active_ctx_tensor(active, ray_o),
    )
    return NearestRayEdge(*values[:8])


def visible(scene_handle: int, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor | None) -> torch.Tensor:
    _require_native_dispatcher()
    values = torch.ops.rayd_torch.visibility_forward(scene_handle, start, end, active)
    return values[0]


def _needs_trace_reflection_ad(*values: torch.Tensor) -> bool:
    for value in values:
        if value.requires_grad:
            return True
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


def _make_trace_reflections_function(fused: bool):
    """Build the specular reflection-chain autograd Function.

    ``fused=True`` takes a single scene-global vertex tensor at apply index 1;
    ``fused=False`` takes one vertex tensor per mesh as trailing varargs.
    """
    vertex_at = 1 if fused else None
    ray_o_at = 2 if fused else 1
    ray_d_at = ray_o_at + 1
    ray_tmax_at = ray_o_at + 2
    active_at = ray_o_at + 3
    max_bounces_at = ray_o_at + 4
    mesh_at = 6

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(*args):
            _require_native_dispatcher()
            return tuple(torch.ops.rayd_torch.trace_reflections_forward(
                args[0],
                args[ray_o_at],
                args[ray_d_at],
                args[ray_tmax_at],
                args[active_at],
                int(args[max_bounces_at]),
            ))

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)
            valid, t, image_sources, prim_ids, tape_prim_id, tape_barycentric, tape_hits, tape_normals, active = output
            ray_o = torch.autograd.forward_ad.unpack_dual(inputs[ray_o_at]).primal
            ray_d = torch.autograd.forward_ad.unpack_dual(inputs[ray_d_at]).primal
            ray_tmax = torch.autograd.forward_ad.unpack_dual(inputs[ray_tmax_at]).primal
            ctx.scene = inputs[0]
            ctx.max_bounces = int(inputs[max_bounces_at])
            ctx.save_for_backward(
                ray_o,
                ray_d,
                ray_tmax,
                active,
                tape_prim_id,
                tape_barycentric,
                tape_hits,
                tape_normals,
                image_sources,
            )
            if fused:
                vertices = torch.autograd.forward_ad.unpack_dual(inputs[vertex_at]).primal
                ctx.save_for_forward(
                    vertices,
                    ray_o,
                    ray_d,
                    active,
                    tape_prim_id,
                    tape_barycentric,
                    tape_hits,
                    tape_normals,
                    image_sources,
                )
            else:
                ctx.mesh_count = len(inputs) - mesh_at
                ctx.save_for_forward(
                    ray_o,
                    ray_d,
                    active,
                    tape_prim_id,
                    tape_barycentric,
                    tape_hits,
                    tape_normals,
                    image_sources,
                )
            ctx.mark_non_differentiable(
                valid,
                prim_ids,
                tape_prim_id,
                tape_barycentric,
                tape_hits,
                tape_normals,
                active,
            )

        @staticmethod
        def backward(ctx, *grad_outputs):
            (
                ray_o,
                ray_d,
                ray_tmax,
                active,
                tape_prim_id,
                tape_barycentric,
                tape_hits,
                tape_normals,
                image_sources,
            ) = ctx.saved_tensors
            if not fused:
                needs_mesh_grad = tuple(bool(value) for value in ctx.needs_input_grad[mesh_at : mesh_at + ctx.mesh_count])
                need_grad_vertices = any(needs_mesh_grad)
                need_grad_ray_o = bool(ctx.needs_input_grad[ray_o_at])
                need_grad_ray_d = bool(ctx.needs_input_grad[ray_d_at])
                need_grad_ray_tmax = bool(ctx.needs_input_grad[ray_tmax_at])
            grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = torch.ops.rayd_torch.trace_reflections_backward_optional(
                ctx.scene,
                ray_o,
                ray_d,
                ray_tmax,
                active,
                tape_prim_id,
                tape_barycentric,
                tape_hits,
                tape_normals,
                image_sources,
                grad_outputs[1],
                grad_outputs[2],
            )
            if fused:
                return None, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, None, None
            if need_grad_vertices:
                mesh_grad_tuple = torch.ops.rayd_torch.split_scene_vertex_grad(ctx.scene, grad_vertices)
                mesh_grads = tuple(mesh_grad_tuple[i] if needs_mesh_grad[i] else None for i in range(ctx.mesh_count))
            else:
                mesh_grads = (None,) * ctx.mesh_count
            return (
                None,
                grad_ray_o if need_grad_ray_o else None,
                grad_ray_d if need_grad_ray_d else None,
                grad_ray_tmax if need_grad_ray_tmax else None,
                None,
                None,
                *mesh_grads,
            )

        @staticmethod
        def jvp(ctx, *tangents):
            if fused:
                (
                    vertices,
                    ray_o,
                    ray_d,
                    active,
                    tape_prim_id,
                    tape_barycentric,
                    tape_hits,
                    tape_normals,
                    image_sources,
                ) = ctx.saved_tensors
            else:
                (
                    ray_o,
                    ray_d,
                    active,
                    tape_prim_id,
                    tape_barycentric,
                    tape_hits,
                    tape_normals,
                    image_sources,
                ) = ctx.saved_tensors
                native_mesh_tangents = tuple(_native_tangent_or_none(value) for value in tangents[mesh_at:])
                with torch._C._DisableFuncTorch():
                    packed_tangents = torch.ops.rayd_torch.pack_scene_vertex_tangents(ctx.scene, list(native_mesh_tangents))
            with torch._C._DisableFuncTorch():
                tangent_t, tangent_image_sources = torch.ops.rayd_torch.trace_reflections_jvp_optional(
                    ctx.scene,
                    _native_tensor(ray_o),
                    _native_tensor(ray_d),
                    _native_tensor(active),
                    _native_tensor(tape_prim_id),
                    _native_tensor(tape_barycentric),
                    _native_tensor(tape_hits),
                    _native_tensor(tape_normals),
                    _native_tangent_or_none(tangents[vertex_at]) if fused else _native_tensor(packed_tangents),
                    _native_tangent_or_none(tangents[ray_o_at]),
                    _native_tangent_or_none(tangents[ray_d_at]),
                    _native_tensor(image_sources),
                )
            return None, tangent_t, tangent_image_sources, None, None, None, None, None, None

    return _named_autograd_function(
        _Fn, "_TraceReflectionsFunction" if fused else "_TraceReflectionsMeshesFunction"
    )


_TraceReflectionsFunction = _make_trace_reflections_function(True)
_TraceReflectionsMeshesFunction = _make_trace_reflections_function(False)


def trace_reflections(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    max_bounces: int,
    mesh_vertices: tuple[torch.Tensor, ...] | None = None,
) -> ReflectionChain:
    _require_native_dispatcher()
    tracked_vertices = (vertices,) if mesh_vertices is None else tuple(mesh_vertices)
    if not _needs_trace_reflection_ad(*tracked_vertices, ray_o, ray_d, ray_tmax):
        def load(full: bool):
            if full:
                valid, t, image_sources, prim_ids = torch.ops.rayd_torch.trace_reflections_forward_noad(
                    scene_handle,
                    ray_o,
                    ray_d,
                    ray_tmax,
                    active,
                    int(max_bounces),
                )
                return valid, t, image_sources, prim_ids
            valid, t, prim_ids = torch.ops.rayd_torch.trace_reflections_forward_reduced(
                scene_handle,
                ray_o,
                ray_d,
                ray_tmax,
                active,
                int(max_bounces),
            )
            return valid, t, None, prim_ids

        return ReflectionChain(loader=load)
    if len(tracked_vertices) > 1:
        values = _TraceReflectionsMeshesFunction.apply(
            scene_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            int(max_bounces),
            *tracked_vertices,
        )
        return ReflectionChain(*values[:4])
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


def _make_trace_refl_epc_field_function(fused: bool):
    """Build the reflection equivalent-path-correction field autograd Function.

    ``fused=True`` takes a single scene-global vertex tensor at apply index 1;
    ``fused=False`` takes one vertex tensor per mesh as trailing varargs.
    """
    vertex_at = 1 if fused else None
    source_at = 2 if fused else 1
    receiver_at = source_at + 1
    active_at = source_at + 2
    max_bounces_at = source_at + 3
    mesh_at = 5

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(*args):
            _require_native_dispatcher()
            return tuple(torch.ops.rayd_torch.trace_refl_epc_field_forward(
                args[0],
                args[source_at],
                args[receiver_at],
                args[active_at],
                int(args[max_bounces_at]),
            ))

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)
            field_real, field_imag, path_length, valid, resolved_prim_ids, tape_prim_id, tape_barycentric, active = output
            tape_t = path_length
            source = torch.autograd.forward_ad.unpack_dual(inputs[source_at]).primal
            receiver = torch.autograd.forward_ad.unpack_dual(inputs[receiver_at]).primal
            ctx.scene = inputs[0]
            ctx.max_bounces = int(inputs[max_bounces_at])
            ctx.save_for_backward(source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
            if fused:
                vertices = torch.autograd.forward_ad.unpack_dual(inputs[vertex_at]).primal
                ctx.save_for_forward(vertices, source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
            else:
                ctx.mesh_count = len(inputs) - mesh_at
                ctx.save_for_forward(source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
            ctx.mark_non_differentiable(valid, resolved_prim_ids, tape_prim_id, tape_barycentric, active)

        @staticmethod
        def backward(ctx, *grad_outputs):
            source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
            if fused:
                need_grad_vertices = bool(ctx.needs_input_grad[vertex_at])
            else:
                needs_mesh_grad = tuple(bool(value) for value in ctx.needs_input_grad[mesh_at : mesh_at + ctx.mesh_count])
                need_grad_vertices = any(needs_mesh_grad)
            need_grad_source = bool(ctx.needs_input_grad[source_at])
            need_grad_receiver = bool(ctx.needs_input_grad[receiver_at])
            grad_vertices, grad_source, grad_receiver = torch.ops.rayd_torch.trace_refl_epc_field_backward(
                ctx.scene,
                source,
                receiver,
                active,
                tape_prim_id,
                tape_barycentric,
                tape_t,
                _native_tensor(grad_outputs[0]),
                _native_tensor(grad_outputs[1]),
                _native_tensor(grad_outputs[2]),
                need_grad_vertices,
                need_grad_source,
                need_grad_receiver,
            )
            if fused:
                return (
                    None,
                    grad_vertices if need_grad_vertices else None,
                    grad_source if need_grad_source else None,
                    grad_receiver if need_grad_receiver else None,
                    None,
                    None,
                )
            if need_grad_vertices:
                mesh_grad_tuple = torch.ops.rayd_torch.split_scene_vertex_grad(ctx.scene, grad_vertices)
                mesh_grads = tuple(mesh_grad_tuple[i] if needs_mesh_grad[i] else None for i in range(ctx.mesh_count))
            else:
                mesh_grads = (None,) * ctx.mesh_count
            return (
                None,
                grad_source if need_grad_source else None,
                grad_receiver if need_grad_receiver else None,
                None,
                None,
                *mesh_grads,
            )

        @staticmethod
        def jvp(ctx, *tangents):
            if fused:
                _vertices, source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
            else:
                source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
                native_mesh_tangents = tuple(_native_tangent_or_none(value) for value in tangents[mesh_at:])
                with torch._C._DisableFuncTorch():
                    packed_tangents = torch.ops.rayd_torch.pack_scene_vertex_tangents(ctx.scene, list(native_mesh_tangents))
            with torch._C._DisableFuncTorch():
                tangent_field_real, tangent_field_imag, tangent_path_length = torch.ops.rayd_torch.trace_refl_epc_field_jvp(
                    ctx.scene,
                    _native_tensor(source),
                    _native_tensor(receiver),
                    _native_tensor(active),
                    _native_tensor(tape_prim_id),
                    _native_tensor(tape_barycentric),
                    _native_tensor(tape_t),
                    _native_tangent_or_none(tangents[vertex_at]) if fused else _native_tensor(packed_tangents),
                    _native_tangent_or_none(tangents[source_at]),
                    _native_tangent_or_none(tangents[receiver_at]),
                )
            return tangent_field_real, tangent_field_imag, tangent_path_length, None, None, None, None, None

    return _named_autograd_function(
        _Fn, "_TraceReflEpcFieldFunction" if fused else "_TraceReflEpcFieldMeshesFunction"
    )


_TraceReflEpcFieldFunction = _make_trace_refl_epc_field_function(True)
_TraceReflEpcFieldMeshesFunction = _make_trace_refl_epc_field_function(False)


def trace_refl_epc_field(
    scene_handle: int,
    vertices: torch.Tensor,
    source: torch.Tensor,
    receiver: torch.Tensor,
    active: torch.Tensor | None,
    max_bounces: int,
    mesh_vertices: tuple[torch.Tensor, ...] | None = None,
) -> ReflEpcField:
    tracked_vertices = (vertices,) if mesh_vertices is None else tuple(mesh_vertices)
    if len(tracked_vertices) > 1:
        values = _TraceReflEpcFieldMeshesFunction.apply(
            scene_handle,
            source,
            receiver,
            active,
            int(max_bounces),
            *tracked_vertices,
        )
        return ReflEpcField(*values[:5])
    values = _TraceReflEpcFieldFunction.apply(
        scene_handle,
        vertices,
        source,
        receiver,
        active,
        int(max_bounces),
    )
    return ReflEpcField(*values[:5])


def trace_dfr_paths_order1_native(
    scene_handle: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    states: DfrStates,
    material: DfrMaterial,
    *,
    active: torch.Tensor,
    max_paths: int,
    wavelength: float,
    tx_polarization: torch.Tensor | None = None,
) -> DfrPaths:
    _require_native_dispatcher()
    state_limit = min(states.state_count, int(max_paths))
    capacity = int(tx_positions.shape[0]) * int(rx_positions.shape[0]) * state_limit
    if tx_polarization is None:
        tx_polarization = torch.zeros_like(tx_positions)
        tx_polarization[:, 0] = 1.0
    values = torch.ops.rayd_torch.diffraction_paths_order1_forward(
        scene_handle,
        tx_positions,
        tx_polarization,
        rx_positions,
        active,
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        state_limit,
        capacity,
        float(wavelength),
    )
    return DfrPaths(capacity, *values)


class _DfrDirectAccumFunction(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        _require_native_dispatcher()
        return tuple(torch.ops.rayd_torch.diffraction_accumulation_forward(
            *args[:21],
            int(args[21]),
            *args[22:],
            1,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
        ))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.scene = inputs[0]
        ctx.state_count = int(inputs[21])
        ctx.grid_axis = int(inputs[22])
        ctx.grid_position = float(inputs[23])
        ctx.grid_coord0_min = float(inputs[24])
        ctx.grid_coord0_max = float(inputs[25])
        ctx.grid_coord1_min = float(inputs[26])
        ctx.grid_coord1_max = float(inputs[27])
        ctx.grid_resolution0 = int(inputs[28])
        ctx.grid_resolution1 = int(inputs[29])
        ctx.grid_cell_area = float(inputs[30])
        ctx.wavelength = float(inputs[31])
        ctx.direct_samples = int(inputs[32])
        ctx.keller_samples = int(inputs[33])
        ctx.suffix_samples = int(inputs[34])
        ctx.seed = int(inputs[35])
        ctx.has_state_wi = inputs[14] is not None
        saved = (
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[9],
            inputs[10],
            inputs[11],
            inputs[12],
            inputs[13],
            _save_optional_tensor(inputs[14], inputs[3]),
            inputs[19],
            inputs[20],
            output[14],
            output[15],
            output[16],
            output[17],
            output[18],
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[7:19])

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            state_wi,
            material_gain,
            material_valid,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u,
        ) = ctx.saved_tensors
        state_wi_arg = state_wi if ctx.has_state_wi else None
        grad_power = _native_tensor(grad_outputs[0])
        grad_field_x_re = _native_tensor(grad_outputs[1])
        (
            grad_state_edge_pos,
            grad_state_edge_dir,
            grad_state_edge_t_min,
            grad_state_edge_t_max,
            grad_state_src,
            grad_state_wi,
            grad_state_src_power,
            grad_state_exterior_angle,
            grad_material_gain,
        ) = torch.ops.rayd_torch.diffraction_accumulation_direct_backward(
            ctx.scene,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            state_wi_arg,
            material_gain,
            material_valid,
            ctx.state_count,
            ctx.grid_axis,
            ctx.grid_position,
            ctx.grid_coord0_min,
            ctx.grid_coord0_max,
            ctx.grid_coord1_min,
            ctx.grid_coord1_max,
            ctx.grid_resolution0,
            ctx.grid_resolution1,
            ctx.grid_cell_area,
            ctx.wavelength,
            ctx.direct_samples,
            ctx.keller_samples,
            ctx.suffix_samples,
            ctx.seed,
            grad_power,
            grad_field_x_re,
        )
        grads = [None] * 36
        grads[3] = grad_state_edge_pos
        grads[4] = grad_state_edge_dir
        grads[5] = grad_state_edge_t_min
        grads[6] = grad_state_edge_t_max
        grads[11] = grad_state_exterior_angle
        grads[12] = grad_state_src
        grads[13] = grad_state_src_power
        grads[14] = grad_state_wi if ctx.has_state_wi else None
        grads[19] = grad_material_gain
        return tuple(grads)

    @staticmethod
    def jvp(ctx, *grad_inputs):
        (
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            state_wi,
            material_gain,
            material_valid,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u,
        ) = ctx.saved_tensors
        state_wi_arg = state_wi if ctx.has_state_wi else None

        def tangent_at(index: int) -> torch.Tensor | None:
            return grad_inputs[index] if index < len(grad_inputs) else None

        with torch._C._DisableFuncTorch():
            dot_power, dot_field_x_re, zero = torch.ops.rayd_torch.diffraction_accumulation_direct_jvp(
                ctx.scene,
                _native_tensor(tape_active),
                _native_tensor(tape_state_idx),
                _native_tensor(tape_cell),
                _native_tensor(tape_material_idx),
                _native_tensor(tape_edge_u),
                _native_tensor(state_edge_pos),
                _native_tensor(state_edge_dir),
                _native_tensor(state_edge_t_min),
                _native_tensor(state_edge_t_max),
                _native_tensor(state_prim0),
                _native_tensor(state_prim1),
                _native_tensor(state_exterior_angle),
                _native_tensor(state_src),
                _native_tensor(state_src_power),
                _native_tensor(state_wi_arg),
                _native_tensor(material_gain),
                _native_tensor(material_valid),
                ctx.state_count,
                ctx.grid_axis,
                ctx.grid_position,
                ctx.grid_coord0_min,
                ctx.grid_coord0_max,
                ctx.grid_coord1_min,
                ctx.grid_coord1_max,
                ctx.grid_resolution0,
                ctx.grid_resolution1,
                ctx.grid_cell_area,
                ctx.wavelength,
                ctx.direct_samples,
                ctx.keller_samples,
                ctx.suffix_samples,
                ctx.seed,
                _native_tangent_or_none(tangent_at(3)),
                _native_tangent_or_none(tangent_at(4)),
                _native_tangent_or_none(tangent_at(5)),
                _native_tangent_or_none(tangent_at(6)),
                _native_tangent_or_none(tangent_at(11)),
                _native_tangent_or_none(tangent_at(12)),
                _native_tangent_or_none(tangent_at(13)),
                _native_tangent_or_none(tangent_at(14)) if ctx.has_state_wi else None,
                _native_tangent_or_none(tangent_at(19)),
            )
        return (
            dot_power,
            dot_field_x_re,
            zero,
            zero,
            zero,
            zero,
            zero,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def accum_dfr_direct_native(
    scene_handle: int,
    states: DfrStates,
    grid: DfrGrid,
    material: DfrMaterial,
    *,
    active: torch.Tensor | None,
    wavelength: float,
    direct_samples: int,
    keller_samples: int = 0,
    suffix_samples: int = 0,
    seed: int = 0,
) -> DfrAccum:
    active_arg = active
    if not _needs_reverse_or_forward_ad(
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        material.gain,
    ):
        state_limit = states.state_count
        values = torch.ops.rayd_torch.diffraction_accumulation_forward(
            scene_handle,
            active_arg,
            states.edge_index,
            states.edge_pos,
            states.edge_dir,
            states.edge_t_min,
            states.edge_t_max,
            states.n0,
            states.n1,
            states.prim0,
            states.prim1,
            states.exterior_angle,
            states.src,
            states.src_power,
            states.wi,
            states.d0,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            state_limit,
            int(grid.axis),
            float(grid.position),
            float(grid.coord0_min),
            float(grid.coord0_max),
            float(grid.coord1_min),
            float(grid.coord1_max),
            int(grid.resolution0),
            int(grid.resolution1),
            grid.resolved_cell_area(),
            float(wavelength),
            int(direct_samples),
            int(keller_samples),
            int(suffix_samples),
            int(seed),
            1,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
        )
        grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
        return DfrAccum(grid_cell_count, *values[:14])
    state_limit = states.state_count
    values = _DfrDirectAccumFunction.apply(
        scene_handle,
        active_arg,
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        state_limit,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(suffix_samples),
        int(seed),
    )
    grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
    return DfrAccum(grid_cell_count, *values[:14])


class _DfrChainAccumFunction(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        _require_native_dispatcher()
        return tuple(torch.ops.rayd_torch.diffraction_accumulation_forward(
            *args[:21],
            int(args[21]),
            *args[23:38],
            int(args[22]),
            *args[38:49],
            int(args[49]),
        ))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.scene = inputs[0]
        ctx.state_count = int(inputs[21])
        ctx.recursive_state_count = int(inputs[22])
        ctx.grid_axis = int(inputs[23])
        ctx.grid_position = float(inputs[24])
        ctx.grid_coord0_min = float(inputs[25])
        ctx.grid_coord0_max = float(inputs[26])
        ctx.grid_coord1_min = float(inputs[27])
        ctx.grid_coord1_max = float(inputs[28])
        ctx.grid_resolution0 = int(inputs[29])
        ctx.grid_resolution1 = int(inputs[30])
        ctx.grid_cell_area = float(inputs[31])
        ctx.wavelength = float(inputs[32])
        ctx.direct_samples = int(inputs[33])
        ctx.keller_samples = int(inputs[34])
        ctx.suffix_samples = int(inputs[35])
        ctx.seed = int(inputs[36])
        ctx.max_order = int(inputs[37])
        saved = (
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[9],
            inputs[10],
            inputs[11],
            inputs[12],
            inputs[13],
            inputs[19],
            inputs[20],
            inputs[39],
            inputs[40],
            inputs[41],
            inputs[42],
            inputs[43],
            inputs[46],
            inputs[47],
            inputs[48],
            output[14],
            output[16],
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[7:19])

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            recursive_state_edge_index,
            recursive_state_edge_pos,
            recursive_state_edge_dir,
            recursive_state_edge_t_min,
            recursive_state_edge_t_max,
            recursive_state_prim0,
            recursive_state_prim1,
            recursive_state_exterior_angle,
            tape_active,
            tape_cell,
        ) = ctx.saved_tensors
        grad_power = _native_tensor(grad_outputs[0])
        grad_field_x_re = _native_tensor(grad_outputs[1])
        (
            grad_state_edge_pos,
            grad_state_edge_dir,
            grad_state_edge_t_min,
            grad_state_edge_t_max,
            grad_state_src,
            grad_state_src_power,
            grad_state_exterior_angle,
            grad_recursive_state_edge_pos,
            grad_recursive_state_edge_dir,
            grad_recursive_state_edge_t_min,
            grad_recursive_state_edge_t_max,
            grad_recursive_state_exterior_angle,
            grad_material_gain,
        ) = torch.ops.rayd_torch.diffraction_accumulation_chain_backward(
            ctx.scene,
            tape_active,
            tape_cell,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            recursive_state_edge_index,
            recursive_state_edge_pos,
            recursive_state_edge_dir,
            recursive_state_edge_t_min,
            recursive_state_edge_t_max,
            recursive_state_prim0,
            recursive_state_prim1,
            recursive_state_exterior_angle,
            material_gain,
            material_valid,
            ctx.state_count,
            ctx.recursive_state_count,
            ctx.grid_axis,
            ctx.grid_position,
            ctx.grid_coord0_min,
            ctx.grid_coord0_max,
            ctx.grid_coord1_min,
            ctx.grid_coord1_max,
            ctx.grid_resolution0,
            ctx.grid_resolution1,
            ctx.grid_cell_area,
            ctx.wavelength,
            ctx.direct_samples,
            ctx.keller_samples,
            ctx.suffix_samples,
            ctx.seed,
            ctx.max_order,
            grad_power,
            grad_field_x_re,
        )
        grads = [None] * 50
        grads[3] = grad_state_edge_pos
        grads[4] = grad_state_edge_dir
        grads[5] = grad_state_edge_t_min
        grads[6] = grad_state_edge_t_max
        grads[11] = grad_state_exterior_angle
        grads[12] = grad_state_src
        grads[13] = grad_state_src_power
        grads[19] = grad_material_gain
        grads[40] = grad_recursive_state_edge_pos
        grads[41] = grad_recursive_state_edge_dir
        grads[42] = grad_recursive_state_edge_t_min
        grads[43] = grad_recursive_state_edge_t_max
        grads[48] = grad_recursive_state_exterior_angle
        return tuple(grads)

    @staticmethod
    def jvp(ctx, *grad_inputs):
        (
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            recursive_state_edge_index,
            recursive_state_edge_pos,
            recursive_state_edge_dir,
            recursive_state_edge_t_min,
            recursive_state_edge_t_max,
            recursive_state_prim0,
            recursive_state_prim1,
            recursive_state_exterior_angle,
            tape_active,
            tape_cell,
        ) = ctx.saved_tensors

        def tangent_at(index: int) -> torch.Tensor | None:
            return grad_inputs[index] if index < len(grad_inputs) else None

        with torch._C._DisableFuncTorch():
            dot_power, dot_field_x_re, zero = torch.ops.rayd_torch.diffraction_accumulation_chain_jvp(
                ctx.scene,
                _native_tensor(tape_active),
                _native_tensor(tape_cell),
                _native_tensor(state_edge_index),
                _native_tensor(state_edge_pos),
                _native_tensor(state_edge_dir),
                _native_tensor(state_edge_t_min),
                _native_tensor(state_edge_t_max),
                _native_tensor(state_prim0),
                _native_tensor(state_prim1),
                _native_tensor(state_exterior_angle),
                _native_tensor(state_src),
                _native_tensor(state_src_power),
                _native_tensor(recursive_state_edge_index),
                _native_tensor(recursive_state_edge_pos),
                _native_tensor(recursive_state_edge_dir),
                _native_tensor(recursive_state_edge_t_min),
                _native_tensor(recursive_state_edge_t_max),
                _native_tensor(recursive_state_prim0),
                _native_tensor(recursive_state_prim1),
                _native_tensor(recursive_state_exterior_angle),
                _native_tensor(material_gain),
                _native_tensor(material_valid),
                ctx.state_count,
                ctx.recursive_state_count,
                ctx.grid_axis,
                ctx.grid_position,
                ctx.grid_coord0_min,
                ctx.grid_coord0_max,
                ctx.grid_coord1_min,
                ctx.grid_coord1_max,
                ctx.grid_resolution0,
                ctx.grid_resolution1,
                ctx.grid_cell_area,
                ctx.wavelength,
                ctx.direct_samples,
                ctx.keller_samples,
                ctx.suffix_samples,
                ctx.seed,
                ctx.max_order,
                _native_tangent_or_none(tangent_at(3)),
                _native_tangent_or_none(tangent_at(4)),
                _native_tangent_or_none(tangent_at(5)),
                _native_tangent_or_none(tangent_at(6)),
                _native_tangent_or_none(tangent_at(11)),
                _native_tangent_or_none(tangent_at(12)),
                _native_tangent_or_none(tangent_at(13)),
                _native_tangent_or_none(tangent_at(40)),
                _native_tangent_or_none(tangent_at(41)),
                _native_tangent_or_none(tangent_at(42)),
                _native_tangent_or_none(tangent_at(43)),
                _native_tangent_or_none(tangent_at(48)),
                _native_tangent_or_none(tangent_at(19)),
            )
        return (
            dot_power,
            dot_field_x_re,
            zero,
            zero,
            zero,
            zero,
            zero,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def accum_dfr_chain_native(
    scene_handle: int,
    initial_states: DfrStates,
    recursive_states: DfrStates,
    grid: DfrGrid,
    material: DfrMaterial,
    *,
    active: torch.Tensor | None,
    recursive_active: torch.Tensor | None,
    wavelength: float,
    direct_samples: int,
    keller_samples: int = 0,
    suffix_samples: int = 0,
    seed: int = 0,
    max_order: int = 2,
) -> DfrAccum:
    _require_native_dispatcher()
    active_arg = active
    recursive_active_arg = recursive_active
    if not _needs_reverse_or_forward_ad(
        initial_states.edge_pos,
        initial_states.edge_dir,
        initial_states.edge_t_min,
        initial_states.edge_t_max,
        initial_states.exterior_angle,
        initial_states.src,
        initial_states.src_power,
        recursive_states.edge_pos,
        recursive_states.edge_dir,
        recursive_states.edge_t_min,
        recursive_states.edge_t_max,
        recursive_states.exterior_angle,
        material.gain,
    ):
        values = torch.ops.rayd_torch.diffraction_accumulation_forward(
            scene_handle,
            active_arg,
            initial_states.edge_index,
            initial_states.edge_pos,
            initial_states.edge_dir,
            initial_states.edge_t_min,
            initial_states.edge_t_max,
            initial_states.n0,
            initial_states.n1,
            initial_states.prim0,
            initial_states.prim1,
            initial_states.exterior_angle,
            initial_states.src,
            initial_states.src_power,
            initial_states.wi,
            initial_states.d0,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            initial_states.state_count,
            int(grid.axis),
            float(grid.position),
            float(grid.coord0_min),
            float(grid.coord0_max),
            float(grid.coord1_min),
            float(grid.coord1_max),
            int(grid.resolution0),
            int(grid.resolution1),
            grid.resolved_cell_area(),
            float(wavelength),
            int(direct_samples),
            int(keller_samples),
            int(suffix_samples),
            int(seed),
            int(max_order),
            recursive_states.state_count,
            recursive_active_arg,
            recursive_states.edge_index,
            recursive_states.edge_pos,
            recursive_states.edge_dir,
            recursive_states.edge_t_min,
            recursive_states.edge_t_max,
            recursive_states.n0,
            recursive_states.n1,
            recursive_states.prim0,
            recursive_states.prim1,
            recursive_states.exterior_angle,
            0,
        )
        grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
        return DfrAccum(grid_cell_count, *values[:14])
    values = _DfrChainAccumFunction.apply(
        scene_handle,
        active_arg,
        initial_states.edge_index,
        initial_states.edge_pos,
        initial_states.edge_dir,
        initial_states.edge_t_min,
        initial_states.edge_t_max,
        initial_states.n0,
        initial_states.n1,
        initial_states.prim0,
        initial_states.prim1,
        initial_states.exterior_angle,
        initial_states.src,
        initial_states.src_power,
        initial_states.wi,
        initial_states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        initial_states.state_count,
        recursive_states.state_count,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(suffix_samples),
        int(seed),
        int(max_order),
        recursive_active_arg,
        recursive_states.edge_index,
        recursive_states.edge_pos,
        recursive_states.edge_dir,
        recursive_states.edge_t_min,
        recursive_states.edge_t_max,
        recursive_states.n0,
        recursive_states.n1,
        recursive_states.prim0,
        recursive_states.prim1,
        recursive_states.exterior_angle,
        1,
    )
    grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
    return DfrAccum(grid_cell_count, *values[:14])


def accum_dfr_coherent_direct_native(
    scene_handle: int,
    states: DfrStates,
    grid: DfrGrid,
    material: DfrMaterial,
    *,
    active: torch.Tensor | None,
    wavelength: float,
    select_diffraction_point: bool = True,
    prefilter_visibility: bool = True,
) -> DfrCoherentAccum:
    _require_native_dispatcher()
    active_arg = active
    state_limit = states.state_count
    values = torch.ops.rayd_torch.diffraction_coherent_accumulation_forward(
        scene_handle,
        active_arg,
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        state_limit,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        float(wavelength),
        bool(select_diffraction_point),
        bool(prefilter_visibility),
    )
    grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
    return DfrCoherentAccum(grid_cell_count, *values)


class NativeOpUnavailable(RuntimeError):
    pass
