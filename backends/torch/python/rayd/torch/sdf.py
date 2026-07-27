"""Standalone differentiable SDF ray intersection (ADR-0037).

`SdfGrid` is a thin holder for the caller's field and its oriented box, and
`sdf_intersect` sphere-traces a ray batch against it. The primitive owns no
acceleration structure, never touches OptiX, and never joins a `Scene`.

Validation here is structural only, matching ADR-0037 section 8: reading a
resident tensor's values to check them would cost a stream synchronization, so
`scale_i > 0` and the finiteness of `values` are the device path's job and
surface as misses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .autograd import (
    _SdfIntersectFunction,
    _needs_reverse_or_forward_ad,
    _require_native_dispatcher,
)
from .types import SdfIntersection


# ADR-0037 section 7 caller defaults. `eps_hit=None` sends the non-positive
# device-derivation sentinel, which is why the operation never reads `scale`
# back to the host to size its hit tolerance.
DEFAULT_MAX_STEPS = 64
DEFAULT_RELAXATION = 0.9
_EPS_HIT_DEVICE_DERIVED = -1.0


def _require_resident_float32(value: torch.Tensor, name: str) -> None:
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be a CUDA tensor (got device {value.device}).")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32 (got {value.dtype}).")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous; call .contiguous() first.")


def _require_vec(value: torch.Tensor, length: int, name: str) -> None:
    _require_resident_float32(value, name)
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},) (got {tuple(value.shape)}).")


def _require_ray_batch(value: torch.Tensor, name: str) -> None:
    _require_resident_float32(value, name)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) (got {tuple(value.shape)}).")


@dataclass(frozen=True)
class SdfGrid:
    """Caller-owned dense signed distance field placed by an oriented box.

    `values` holds vertex-centred float32 samples of shape `[Nx, Ny, Nz]` with
    `N_i >= 2`, in world-metric distance and negative inside. `position` is the
    world centre of the box, `rotation` a scalar-first quaternion, and `scale`
    the full side lengths, so the local frame spans `[-scale/2, +scale/2]`.
    All four tensors stay caller-owned; RayD never bakes, rescales, or copies
    them.
    """

    values: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        _require_resident_float32(self.values, "SdfGrid.values")
        if self.values.ndim != 3:
            raise ValueError(
                f"SdfGrid.values must have shape (Nx, Ny, Nz) (got {tuple(self.values.shape)})."
            )
        if min(self.values.shape) < 2:
            raise ValueError(
                "SdfGrid.values must have at least 2 samples on every axis "
                f"(got {tuple(self.values.shape)})."
            )
        _require_vec(self.position, 3, "SdfGrid.position")
        _require_vec(self.rotation, 4, "SdfGrid.rotation")
        _require_vec(self.scale, 3, "SdfGrid.scale")
        for name in ("position", "rotation", "scale"):
            if getattr(self, name).device != self.values.device:
                raise ValueError(
                    f"SdfGrid.{name} must be on the same CUDA device as SdfGrid.values "
                    f"({getattr(self, name).device} != {self.values.device})."
                )


def sdf_intersect(
    grid: SdfGrid,
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    tmax: float = float("inf"),
    max_steps: int = DEFAULT_MAX_STEPS,
    relaxation: float = DEFAULT_RELAXATION,
    eps_hit: float | None = None,
) -> SdfIntersection:
    """Sphere-trace a ray batch against `grid` (ADR-0037).

    `origins` and `directions` are float32 CUDA `[N, 3]` on the grid's device;
    directions are normalized inside the operation, so `t` is a metric
    distance. The traced interval is the ray's overlap with the box clipped to
    `tmax`, and a ray whose origin is inside the box is a supported case.
    `relaxation` is the sphere-trace step factor in `(0, 1]`, and `eps_hit`
    defaults to a resolution-derived tolerance computed on the device.

    Gradients and tangents flow to `grid.values`, `grid.position`,
    `grid.rotation`, `grid.scale`, `origins` and `directions` through the
    frozen-winner implicit function theorem. Missed lanes are bitwise inert.
    """
    _require_native_dispatcher()
    _require_ray_batch(origins, "origins")
    _require_ray_batch(directions, "directions")
    if origins.shape[0] != directions.shape[0]:
        raise ValueError(
            "origins and directions must have the same ray count "
            f"({origins.shape[0]} != {directions.shape[0]})."
        )
    if origins.device != grid.values.device:
        raise ValueError(
            f"origins and directions must be on the grid's device "
            f"({origins.device} != {grid.values.device})."
        )
    if not tmax > 0.0:
        raise ValueError(f"tmax must be positive (got {tmax}).")
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1 (got {max_steps}).")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError(f"relaxation must lie in (0, 1] (got {relaxation}).")
    if eps_hit is not None and not eps_hit > 0.0:
        raise ValueError(
            f"eps_hit must be positive, or None to derive it on the device (got {eps_hit})."
        )

    request = (
        grid.values,
        grid.position,
        grid.rotation,
        grid.scale,
        origins,
        directions,
        float(tmax),
        int(max_steps),
        float(relaxation),
        _EPS_HIT_DEVICE_DERIVED if eps_hit is None else float(eps_hit),
    )
    if _needs_reverse_or_forward_ad(*request[:6]):
        outputs = _SdfIntersectFunction.apply(*request)
    else:
        outputs = torch.ops.rayd_torch.sdf_intersect_forward(*request)
    return SdfIntersection(*outputs[:5])
