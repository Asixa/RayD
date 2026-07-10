"""torch.compile support for the static intersect VJP path.

The handle-based ops (`intersect_forward_tape_h` / `intersect_backward_t_h`)
take a plain int64 scene handle so compiled graphs never carry ScriptObjects.
Fake (meta) implementations let FakeTensor tracing propagate shapes, and
`torch.library.register_autograd` installs the backward so AOTAutograd can
capture forward and backward into compiled graphs. Eager execution never
routes through these ops; `Scene.intersect` selects them only under
`torch.compiler.is_compiling()` for the vertices-only-gradient case.
"""
from __future__ import annotations

import torch


def _fake_intersect_forward_tape_h(scene_handle, vertices, ray_o, ray_d, ray_tmax):
    n = ray_o.shape[0]
    t = ray_o.new_empty((n,))
    tape_prim_id = ray_o.new_empty((n,), dtype=torch.int32)
    return t, tape_prim_id


def _fake_intersect_backward_t_h(scene_handle, vertices, ray_o, ray_d, tape_prim_id, grad_t):
    return vertices.new_empty(vertices.shape)


def _setup_context(ctx, inputs, output):
    scene_handle, vertices, ray_o, ray_d, _ray_tmax = inputs
    _t, tape_prim_id = output
    ctx.scene_handle = scene_handle
    ctx.save_for_backward(vertices, ray_o, ray_d, tape_prim_id)


def _backward(ctx, grad_t, _grad_tape_prim_id):
    vertices, ray_o, ray_d, tape_prim_id = ctx.saved_tensors
    if grad_t is None:
        grad_vertices = torch.zeros_like(vertices)
    else:
        grad_vertices = torch.ops.raydn.intersect_backward_t_h(
            ctx.scene_handle,
            vertices,
            ray_o,
            ray_d,
            tape_prim_id,
            grad_t,
        )
    return None, grad_vertices, None, None, None


def register() -> None:
    torch.library.register_fake("raydn::intersect_forward_tape_h")(_fake_intersect_forward_tape_h)
    torch.library.register_fake("raydn::intersect_backward_t_h")(_fake_intersect_backward_t_h)
    torch.library.register_autograd(
        "raydn::intersect_forward_tape_h",
        _backward,
        setup_context=_setup_context,
    )
