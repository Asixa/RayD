# Copyright Xingyu Chen.
# Implements shared Python support for sdf.

"""Provides the standalone differentiable SDF intersection API."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .multipath import _SdfIntersectFunction, _needs_reverse_or_forward_ad, _require_native_dispatcher
from .geometry import Ray, ReflectionChain, SdfIntersection


# ADR-0037 section 7 caller defaults. `eps_hit=None` sends the non-positive
# device-derivation sentinel, which is why the operation never reads `scale`
# back to the host to size its hit tolerance.
DEFAULT_MAX_STEPS = 64
DEFAULT_RELAXATION = 0.9
_EPS_HIT_DEVICE_DERIVED = -1.0
_RAY_EPSILON = 1.0e-3


@dataclass(frozen=True, slots=True)
class SdfTraceOptions:
    """Controls bounded sphere tracing inside an SDF grid's oriented bounding box."""

    max_steps: int = DEFAULT_MAX_STEPS
    relaxation: float = DEFAULT_RELAXATION
    eps_hit: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("SdfTraceOptions.max_steps must be at least 1.")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("SdfTraceOptions.relaxation must lie in (0, 1].")
        if self.eps_hit is not None and not self.eps_hit > 0.0:
            raise ValueError("SdfTraceOptions.eps_hit must be positive or None.")


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


def _require_active(active: torch.Tensor | None, count: int, device: torch.device) -> torch.Tensor:
    if active is None:
        return torch.ones((count,), dtype=torch.bool, device=device)
    if active.device != device or active.dtype != torch.bool or active.shape != (count,) or not active.is_contiguous():
        raise ValueError("active must be a contiguous CUDA bool tensor with shape (N,) on the ray device.")
    return active


@dataclass(frozen=True)
class SdfGrid:
    """Provides the standalone differentiable SDF intersection API."""

    values: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        _require_resident_float32(self.values, "SdfGrid.values")
        if self.values.ndim != 3:
            raise ValueError(f"SdfGrid.values must have shape (Nx, Ny, Nz) (got {tuple(self.values.shape)}).")
        if min(self.values.shape) < 2:
            raise ValueError(
                f"SdfGrid.values must have at least 2 samples on every axis (got {tuple(self.values.shape)})."
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

    def _query_bias(self, eps_hit: float | None) -> torch.Tensor:
        if eps_hit is None:
            shape = self.values.shape
            resolved = (
                torch.minimum(
                    self.scale[0] / float(shape[0] - 1),
                    torch.minimum(self.scale[1] / float(shape[1] - 1), self.scale[2] / float(shape[2] - 1)),
                )
                * 1.0e-3
            )
        else:
            if not eps_hit > 0.0:
                raise ValueError(f"eps_hit must be positive, or None to derive it on the device (got {eps_hit}).")
            resolved = self.scale.new_tensor(float(eps_hit))
        return torch.maximum(2.0 * resolved, self.scale.new_tensor(_RAY_EPSILON))

    def intersect(
        self,
        ray: Ray,
        *,
        active: torch.Tensor | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        relaxation: float = DEFAULT_RELAXATION,
        eps_hit: float | None = None,
    ) -> SdfIntersection:
        """Trace a ray batch against this grid, honoring per-lane ``Ray.tmax`` and ``active``."""
        if not isinstance(ray, Ray):
            raise TypeError("SdfGrid.intersect() expects rayd.torch.Ray.")
        lane_active = _require_active(active, ray.o.shape[0], ray.o.device)
        hit = sdf_intersect(self, ray.o, ray.d, max_steps=max_steps, relaxation=relaxation, eps_hit=eps_hit)
        valid = hit.hit_mask & lane_active
        if ray.tmax.numel() != 0:
            valid = valid & (hit.t < ray.tmax)
        inf = torch.full_like(hit.t, float("inf"))
        zero3 = torch.zeros_like(hit.position)
        return SdfIntersection(
            torch.where(valid, hit.t, inf),
            valid,
            torch.where(valid[:, None], hit.position, zero3),
            torch.where(valid[:, None], hit.normal, zero3),
            torch.where(lane_active, hit.steps, torch.zeros_like(hit.steps)),
        )

    def visible(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        active: torch.Tensor | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        relaxation: float = DEFAULT_RELAXATION,
        eps_hit: float | None = None,
    ) -> torch.Tensor:
        """Return segment LOS; only SDF intersections can block the segment."""
        _require_ray_batch(start, "start")
        _require_ray_batch(end, "end")
        if start.shape != end.shape or start.device != self.values.device or end.device != self.values.device:
            raise ValueError("start and end must have equal shape and be on the SDF grid's CUDA device.")
        lane_active = _require_active(active, start.shape[0], start.device)
        delta = end - start
        length = torch.linalg.vector_norm(delta, dim=1)
        bias = self._query_bias(eps_hit)
        short = length <= 2.0 * bias
        direction = delta / torch.clamp_min(length, 1.0e-12)[:, None]
        ray = Ray(
            (start + direction * bias).contiguous(),
            direction.contiguous(),
            torch.clamp_min(length - 2.0 * bias, 0.0).contiguous(),
        )
        hit = self.intersect(
            ray, active=lane_active & ~short, max_steps=max_steps, relaxation=relaxation, eps_hit=eps_hit
        )
        return lane_active & (short | ~hit.hit_mask)

    def trace_reflections(
        self,
        ray: Ray,
        max_bounces: int,
        active: torch.Tensor | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        relaxation: float = DEFAULT_RELAXATION,
        eps_hit: float | None = None,
    ) -> ReflectionChain:
        """Trace specular SDF reflections without adding any diffraction path."""
        if not isinstance(ray, Ray):
            raise TypeError("SdfGrid.trace_reflections() expects rayd.torch.Ray.")
        if max_bounces < 0:
            raise ValueError("max_bounces must be non-negative.")
        count = ray.o.shape[0]
        lane_active = _require_active(active, count, ray.o.device)
        if max_bounces == 0:
            return ReflectionChain(
                torch.empty((count, 0), dtype=torch.bool, device=ray.o.device),
                torch.empty((count, 0), dtype=ray.o.dtype, device=ray.o.device),
                torch.empty((count, 0, 3), dtype=ray.o.dtype, device=ray.o.device),
                torch.empty((count, 0), dtype=torch.int32, device=ray.o.device),
            )

        direction = ray.d / torch.clamp_min(torch.linalg.vector_norm(ray.d, dim=1), 1.0e-12)[:, None]
        current_ray = Ray(ray.o, direction.contiguous(), ray.tmax)
        current_image_source = ray.o
        bias = self._query_bias(eps_hit)
        valid_slots: list[torch.Tensor] = []
        t_slots: list[torch.Tensor] = []
        image_slots: list[torch.Tensor] = []
        id_slots: list[torch.Tensor] = []
        for _bounce in range(max_bounces):
            hit = self.intersect(
                current_ray, active=lane_active, max_steps=max_steps, relaxation=relaxation, eps_hit=eps_hit
            )
            bounce_hit = lane_active & hit.hit_mask
            normal = torch.where((torch.sum(current_ray.d * hit.normal, dim=1) > 0.0)[:, None], -hit.normal, hit.normal)
            plane_distance = torch.sum((current_image_source - hit.position) * normal, dim=1)
            image_source = current_image_source - 2.0 * plane_distance[:, None] * normal
            reflected = current_ray.d - 2.0 * torch.sum(current_ray.d * normal, dim=1)[:, None] * normal

            valid_slots.append(bounce_hit)
            t_slots.append(torch.where(bounce_hit, hit.t, torch.full_like(hit.t, float("inf"))))
            image_slots.append(torch.where(bounce_hit[:, None], image_source, torch.zeros_like(image_source)))
            id_slots.append(
                torch.where(
                    bounce_hit,
                    torch.zeros((count,), dtype=torch.int32, device=ray.o.device),
                    torch.full((count,), -1, dtype=torch.int32, device=ray.o.device),
                )
            )

            next_origin = hit.position + bias * reflected
            current_ray = Ray(
                torch.where(bounce_hit[:, None], next_origin, current_ray.o).contiguous(),
                torch.where(bounce_hit[:, None], reflected, current_ray.d).contiguous(),
            )
            current_image_source = torch.where(bounce_hit[:, None], image_source, current_image_source)
            lane_active = bounce_hit

        return ReflectionChain(
            torch.stack(valid_slots, dim=1),
            torch.stack(t_slots, dim=1),
            torch.stack(image_slots, dim=1),
            torch.stack(id_slots, dim=1),
        )


@dataclass(frozen=True)
class SdfGridBatch:
    """A caller-owned packed group of shape-compatible dense SDF grids.

    The leading dimension is the grid owner dimension. Untracked queries use
    one native CUDA launch; AD queries retain the existing per-grid frozen-tape
    implementation.
    """

    values: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        _require_resident_float32(self.values, "SdfGridBatch.values")
        if self.values.ndim != 4 or self.values.shape[0] < 2 or min(self.values.shape[1:]) < 2:
            raise ValueError(
                "SdfGridBatch.values must have shape (G, Nx, Ny, Nz) with G >= 2 and every spatial axis >= 2."
            )
        count = int(self.values.shape[0])
        for value, width, name in (
            (self.position, 3, "position"),
            (self.rotation, 4, "rotation"),
            (self.scale, 3, "scale"),
        ):
            _require_resident_float32(value, f"SdfGridBatch.{name}")
            if value.shape != (count, width):
                raise ValueError(f"SdfGridBatch.{name} must have shape ({count}, {width}).")
            if value.device != self.values.device:
                raise ValueError(f"SdfGridBatch.{name} must be on the values device.")

    @property
    def grid_count(self) -> int:
        return int(self.values.shape[0])

    def grid(self, index: int) -> SdfGrid:
        if index < 0 or index >= self.grid_count:
            raise IndexError("SdfGridBatch grid index is out of range.")
        return SdfGrid(self.values[index], self.position[index], self.rotation[index], self.scale[index])

    def intersect(
        self,
        ray: Ray,
        *,
        active: torch.Tensor | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        relaxation: float = DEFAULT_RELAXATION,
        eps_hit: float | None = None,
    ) -> tuple[SdfIntersection, ...]:
        _require_native_dispatcher()
        if not isinstance(ray, Ray):
            raise TypeError("SdfGridBatch.intersect() expects rayd.torch.Ray.")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1 (got {max_steps}).")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(f"relaxation must lie in (0, 1] (got {relaxation}).")
        if eps_hit is not None and not eps_hit > 0.0:
            raise ValueError(f"eps_hit must be positive, or None to derive it on the device (got {eps_hit}).")
        lane_active = _require_active(active, ray.o.shape[0], ray.o.device)
        if _needs_reverse_or_forward_ad(self.values, self.position, self.rotation, self.scale, ray.o, ray.d):
            return tuple(
                self.grid(index).intersect(
                    ray, active=lane_active, max_steps=max_steps, relaxation=relaxation, eps_hit=eps_hit
                )
                for index in range(self.grid_count)
            )
        if ray.o.device != self.values.device:
            raise ValueError("ray must be on the SDF batch values device.")
        values = torch.ops.rayd_torch.sdf_batch_intersect_forward(
            self.values,
            self.position,
            self.rotation,
            self.scale,
            ray.o,
            ray.d,
            float("inf"),
            int(max_steps),
            float(relaxation),
            _EPS_HIT_DEVICE_DERIVED if eps_hit is None else float(eps_hit),
        )
        results = []
        for index in range(self.grid_count):
            valid = values[1][index] & lane_active
            if ray.tmax.numel() != 0:
                valid = valid & (values[0][index] < ray.tmax)
            zero3 = torch.zeros_like(values[2][index])
            results.append(
                SdfIntersection(
                    torch.where(valid, values[0][index], torch.full_like(values[0][index], float("inf"))),
                    valid,
                    torch.where(valid[:, None], values[2][index], zero3),
                    torch.where(valid[:, None], values[3][index], zero3),
                    torch.where(lane_active, values[4][index], torch.zeros_like(values[4][index])),
                )
            )
        return tuple(results)


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
    """Provides the standalone differentiable SDF intersection API."""
    _require_native_dispatcher()
    _require_ray_batch(origins, "origins")
    _require_ray_batch(directions, "directions")
    if origins.shape[0] != directions.shape[0]:
        raise ValueError(
            f"origins and directions must have the same ray count ({origins.shape[0]} != {directions.shape[0]})."
        )
    if origins.device != grid.values.device:
        raise ValueError(
            f"origins and directions must be on the grid's device ({origins.device} != {grid.values.device})."
        )
    if not tmax > 0.0:
        raise ValueError(f"tmax must be positive (got {tmax}).")
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1 (got {max_steps}).")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError(f"relaxation must lie in (0, 1] (got {relaxation}).")
    if eps_hit is not None and not eps_hit > 0.0:
        raise ValueError(f"eps_hit must be positive, or None to derive it on the device (got {eps_hit}).")

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
