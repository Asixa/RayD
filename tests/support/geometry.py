# Copyright Xingyu Chen.
# Builds deterministic test geometry and exact tensor bit views.

from __future__ import annotations

import torch

import rayd.torch as rt


def grid_mesh(device: torch.device, cells: int = 8, span: float = 2.0):
    axis = torch.linspace(-0.5 * span, 0.5 * span, cells + 1, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    flat_x = x.reshape(-1)
    vertices = torch.stack((flat_x, y.reshape(-1), torch.zeros_like(flat_x)), dim=1)
    index = torch.arange((cells + 1) * (cells + 1), dtype=torch.int32).reshape(cells + 1, cells + 1)
    a = index[:-1, :-1].reshape(-1)
    b = index[:-1, 1:].reshape(-1)
    c = index[1:, :-1].reshape(-1)
    d = index[1:, 1:].reshape(-1)
    faces = torch.cat((torch.stack((a, b, c), dim=1), torch.stack((b, d, c), dim=1)))
    return vertices.contiguous().to(device), faces.contiguous().to(device)


def box_mesh(device: torch.device, half: float = 1.0):
    """Closed axis-aligned cube centred at the origin, every face wound outward."""
    corners = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    vertices = torch.tensor(corners, dtype=torch.float32) * half
    faces = torch.tensor(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4]]
        + [[1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=torch.int32,
    )
    return vertices.to(device), faces.to(device)


def tensor_bits(tensor: torch.Tensor) -> torch.Tensor:
    host = tensor.detach().contiguous().cpu()
    if host.dtype == torch.float32:
        return host.view(torch.int32)
    if host.dtype == torch.float64:
        return host.view(torch.int64)
    return host


def build_scene(device: torch.device, **kwargs) -> rt.Scene:
    vertices, faces = grid_mesh(device)
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene
