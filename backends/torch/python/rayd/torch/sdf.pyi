from dataclasses import dataclass
import torch

from .types import SdfIntersection

DEFAULT_MAX_STEPS: int
DEFAULT_RELAXATION: float

@dataclass(frozen=True)
class SdfGrid:
    values: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor

def sdf_intersect(
    grid: SdfGrid,
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    tmax: float = ...,
    max_steps: int = ...,
    relaxation: float = ...,
    eps_hit: float | None = ...,
) -> SdfIntersection: ...
