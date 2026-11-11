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
    """Sphere-tracing options for SDF grid queries.

    Options record for ``SdfGrid.intersect()``, ``visible()``, ``trace_reflections()``, ``SdfGridBatch.intersect()``,
    and ``MixedScene.add_sdf()``. Invalid values raise ``ValueError``.

    Attributes:
        max_steps: Maximum sphere-tracing steps; must be at least 1.
        relaxation: Step relaxation factor in (0, 1].
        eps_hit: Hit threshold in world units; ``None`` selects 1e-3 of the smallest voxel size.
    """

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


def _trace_options(options: SdfTraceOptions | None, operation: str) -> SdfTraceOptions:
    if options is None:
        return SdfTraceOptions()
    if not isinstance(options, SdfTraceOptions):
        raise TypeError(f"{operation} expects rayd.torch.SdfTraceOptions, got {type(options).__name__}.")
    return options


def _require_active(active: torch.Tensor | None, count: int, device: torch.device) -> torch.Tensor:
    if active is None:
        return torch.ones((count,), dtype=torch.bool, device=device)
    if active.device != device or active.dtype != torch.bool or active.shape != (count,) or not active.is_contiguous():
        raise ValueError("active must be a contiguous CUDA bool tensor with shape (N,) on the ray device.")
    return active


@dataclass(frozen=True)
class SdfGrid:
    """Signed distance field sampled on a regular grid inside an oriented box.

    Standalone geometry for ray queries; it can also be added to a ``MixedScene``. Every tensor is a contiguous CUDA
    ``float32`` tensor on one device; wrong devices or dtypes raise ``TypeError`` and wrong shapes ``ValueError``.
    Queries are differentiable (VJP and JVP) with respect to the values, the placement, and the ray, with the hit fixed
    by the forward pass.

    Attributes:
        values: Vertex-centered signed distances in world units, negative inside, ``(Nx, Ny, Nz)`` with at least 2
            samples per axis.
        position: World-space center of the box, ``(3,)``.
        rotation: Box rotation as a unit quaternion in (w, x, y, z) order, ``(4,)``.
        scale: Full side lengths of the box, ``(3,)``.
    """

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
        self, ray: Ray, options: SdfTraceOptions | None = None, active: torch.Tensor | None = None
    ) -> SdfIntersection:
        """Sphere-trace each ray against the zero level set and return an ``SdfIntersection``.

        Tracing is limited to the oriented box and ``ray.tmax``. ``options`` is an ``SdfTraceOptions`` (``None`` uses
        the defaults) and ``active`` an optional ``bool`` ``(N,)`` mask. The result is differentiable with respect to
        the ray, the values, and the placement. A wrong argument type raises ``TypeError``.
        """
        if not isinstance(ray, Ray):
            raise TypeError("SdfGrid.intersect() expects rayd.torch.Ray.")
        options = _trace_options(options, "SdfGrid.intersect()")
        lane_active = _require_active(active, ray.o.shape[0], ray.o.device)
        hit = sdf_intersect(
            self, ray.o, ray.d, max_steps=options.max_steps, relaxation=options.relaxation, eps_hit=options.eps_hit
        )
        valid = hit.valid & lane_active
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
        options: SdfTraceOptions | None = None,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a ``bool`` ``(N,)`` mask that is True where the SDF surface does not block the segment.

        ``start`` and ``end`` are ``float32`` ``(N, 3)``. The result is discrete and has no derivative.
        """
        options = _trace_options(options, "SdfGrid.visible()")
        _require_ray_batch(start, "start")
        _require_ray_batch(end, "end")
        if start.shape != end.shape or start.device != self.values.device or end.device != self.values.device:
            raise ValueError("start and end must have equal shape and be on the SDF grid's CUDA device.")
        lane_active = _require_active(active, start.shape[0], start.device)
        delta = end - start
        length = torch.linalg.vector_norm(delta, dim=1)
        bias = self._query_bias(options.eps_hit)
        short = length <= 2.0 * bias
        direction = delta / torch.clamp_min(length, 1.0e-12)[:, None]
        ray = Ray(
            (start + direction * bias).contiguous(),
            direction.contiguous(),
            torch.clamp_min(length - 2.0 * bias, 0.0).contiguous(),
        )
        hit = self.intersect(ray, options, lane_active & ~short)
        return lane_active & (short | ~hit.valid)

    def trace_reflections(
        self, ray: Ray, max_bounces: int, options: SdfTraceOptions | None = None, active: torch.Tensor | None = None
    ) -> ReflectionChain:
        """Trace specular reflections off the SDF surface for up to ``max_bounces`` bounces.

        Returns a ``ReflectionChain``. ``max_bounces`` must be non-negative.
        """
        if not isinstance(ray, Ray):
            raise TypeError("SdfGrid.trace_reflections() expects rayd.torch.Ray.")
        options = _trace_options(options, "SdfGrid.trace_reflections()")
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
        bias = self._query_bias(options.eps_hit)
        valid_slots: list[torch.Tensor] = []
        t_slots: list[torch.Tensor] = []
        image_slots: list[torch.Tensor] = []
        id_slots: list[torch.Tensor] = []
        for _bounce in range(max_bounces):
            hit = self.intersect(current_ray, options, lane_active)
            bounce_hit = lane_active & hit.valid
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
    """Several SDF grids of the same sample shape, packed along a leading grid axis.

    Untracked queries run one native launch for every grid; differentiable queries trace each grid on its own. Wrong
    devices or dtypes raise ``TypeError`` and wrong shapes ``ValueError``.

    Attributes:
        values: Signed distances, ``float32`` ``(G, Nx, Ny, Nz)`` with ``G >= 2`` and at least 2 samples per axis.
        position: Box centers, ``float32`` ``(G, 3)``.
        rotation: Box rotations as (w, x, y, z) quaternions, ``float32`` ``(G, 4)``.
        scale: Box side lengths, ``float32`` ``(G, 3)``.
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
        """Number of grids ``G``."""
        return int(self.values.shape[0])

    def grid(self, index: int) -> SdfGrid:
        """Return grid ``index`` as an ``SdfGrid`` view. An index out of range raises ``IndexError``."""
        if index < 0 or index >= self.grid_count:
            raise IndexError("SdfGridBatch grid index is out of range.")
        return SdfGrid(self.values[index], self.position[index], self.rotation[index], self.scale[index])

    def intersect(
        self, ray: Ray, options: SdfTraceOptions | None = None, active: torch.Tensor | None = None
    ) -> tuple[SdfIntersection, ...]:
        """Intersect each ray with every grid and return one ``SdfIntersection`` per grid.

        Arguments and differentiability are those of ``SdfGrid.intersect()``.
        """
        _require_native_dispatcher()
        if not isinstance(ray, Ray):
            raise TypeError("SdfGridBatch.intersect() expects rayd.torch.Ray.")
        options = _trace_options(options, "SdfGridBatch.intersect()")
        lane_active = _require_active(active, ray.o.shape[0], ray.o.device)
        if _needs_reverse_or_forward_ad(self.values, self.position, self.rotation, self.scale, ray.o, ray.d):
            return tuple(self.grid(index).intersect(ray, options, lane_active) for index in range(self.grid_count))
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
            int(options.max_steps),
            float(options.relaxation),
            _EPS_HIT_DEVICE_DERIVED if options.eps_hit is None else float(options.eps_hit),
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
    """Sphere-trace a ray batch against one ``SdfGrid``; the low-level form of ``SdfGrid.intersect()``.

    Args:
        grid: The ``SdfGrid``.
        origins: Ray origins, ``float32`` ``(N, 3)`` on the grid's device.
        directions: Ray directions, ``float32`` ``(N, 3)``.
        tmax: Maximum ray distance for every ray; must be positive.
        max_steps: Maximum sphere-tracing steps; must be at least 1.
        relaxation: Step relaxation factor in (0, 1].
        eps_hit: Hit threshold in world units; ``None`` selects 1e-3 of the smallest voxel size.

    Returns:
        ``SdfIntersection``, differentiable (VJP and JVP) with respect to the rays, the values, and the placement.

    Invalid values raise ``ValueError`` and wrong tensor types ``TypeError``.
    """
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
