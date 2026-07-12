from dataclasses import dataclass
import torch
from .types import Ray

@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fov_x: float
    @property
    def aspect(self) -> float: ...
    def sample_to_world(self, sample: torch.Tensor, depth: float = ...) -> torch.Tensor: ...
    def world_to_sample(self, point: torch.Tensor) -> torch.Tensor: ...
    def sample_ray(self, sample: torch.Tensor) -> Ray: ...
