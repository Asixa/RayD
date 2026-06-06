from __future__ import annotations

from dataclasses import dataclass
import torch


def _empty_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


@dataclass
class Mesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    uv: torch.Tensor | None = None
    face_uv: torch.Tensor | None = None
    use_face_normals: bool = False
    edges_enabled: bool = True
    to_world_left: torch.Tensor | None = None
    to_world_right: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Mesh.vertices must have shape (V, 3).")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Mesh.faces must have shape (F, 3).")
        if self.uv is None:
            self.uv = _empty_tensor((0, 2), torch.float32, self.vertices.device)
        if self.face_uv is None:
            self.face_uv = _empty_tensor((0, 3), torch.int32, self.vertices.device)
        if self.to_world_left is None:
            self.to_world_left = torch.eye(4, dtype=torch.float32, device=self.vertices.device)
        if self.to_world_right is None:
            self.to_world_right = torch.eye(4, dtype=torch.float32, device=self.vertices.device)
