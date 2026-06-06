from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Ray:
    o: torch.Tensor
    d: torch.Tensor
    tmax: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.o.ndim != 2 or self.o.shape[1] != 3:
            raise ValueError("Ray.o must have shape (N, 3).")
        if self.d.ndim != 2 or self.d.shape[1] != 3:
            raise ValueError("Ray.d must have shape (N, 3).")
        if self.o.shape[0] != self.d.shape[0]:
            raise ValueError("Ray.o and Ray.d must have the same batch size.")
        if self.tmax is None:
            object.__setattr__(
                self,
                "tmax",
                torch.full((self.o.shape[0],), float("inf"), device=self.o.device, dtype=self.o.dtype),
            )
        elif self.tmax.ndim != 1 or self.tmax.shape[0] != self.o.shape[0]:
            raise ValueError("Ray.tmax must have shape (N,).")


@dataclass(frozen=True)
class Intersection:
    t: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor
    geo_n: torch.Tensor
    uv: torch.Tensor
    barycentric: torch.Tensor
    shape_id: torch.Tensor
    prim_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor

    def is_valid(self) -> torch.Tensor:
        return self.shape_id >= 0


@dataclass(frozen=True)
class NearestPointEdge:
    distance: torch.Tensor
    edge_point: torch.Tensor
    edge_t: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


@dataclass(frozen=True)
class NearestRayEdge:
    distance: torch.Tensor
    ray_t: torch.Tensor
    edge_point: torch.Tensor
    edge_t: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


@dataclass(frozen=True)
class ReflectionChain:
    valid: torch.Tensor
    t: torch.Tensor
    image_sources: torch.Tensor
    prim_ids: torch.Tensor


@dataclass(frozen=True)
class SceneGlobalGeometry:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normal: torch.Tensor
    shape_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor
