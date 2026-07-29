# Copyright Xingyu Chen.
# Implements the Torch surfel query, visibility, reflection, and transmission paths.

"""Differentiable standalone surfels backed by the Torch scene accelerator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import Ray, RayFlags, ReflectionChain, _CONTRACT_VALUES
from .scene import Mesh, Scene


_RAY_EPSILON = float(_CONTRACT_VALUES["ray_epsilon"])
_SHADOW_EPSILON = float(_CONTRACT_VALUES["shadow_epsilon"])


def _require_float_cuda(value: torch.Tensor, name: str, rank: int, last_dim: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cuda" or value.dtype != torch.float32:
        raise TypeError(f"{name} must be a CUDA torch.float32 tensor.")
    if value.ndim != rank or (last_dim is not None and value.shape[-1] != last_dim):
        suffix = f" with last dimension {last_dim}" if last_dim is not None else ""
        raise ValueError(f"{name} must have rank {rank}{suffix}.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _require_active(active: torch.Tensor | None, count: int, device: torch.device) -> torch.Tensor:
    if active is None:
        return torch.ones((count,), dtype=torch.bool, device=device)
    if active.device != device or active.dtype != torch.bool or active.shape != (count,) or not active.is_contiguous():
        raise ValueError("active must be a contiguous CUDA bool tensor with shape (N,) on the ray device.")
    return active


@dataclass(frozen=True, slots=True)
class SurfelTraceOptions:
    """Controls the Gaussian acceptance domain and bounded candidate work."""

    alpha_min: float = 1.0 / 255.0
    cutoff: float = float("inf")
    alpha_cap: float = 0.99
    max_candidate_hits: int = 8
    face_forward: bool = True
    opacity_aware_proxy_bounds: bool = False
    transmittance_min: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_min < 1.0:
            raise ValueError("SurfelTraceOptions.alpha_min must lie in (0, 1).")
        if not self.cutoff > 0.0:
            raise ValueError("SurfelTraceOptions.cutoff must be positive.")
        if not 0.0 < self.alpha_cap <= 1.0:
            raise ValueError("SurfelTraceOptions.alpha_cap must lie in (0, 1].")
        if self.max_candidate_hits < 1:
            raise ValueError("SurfelTraceOptions.max_candidate_hits must be positive.")
        if not 0.0 <= self.transmittance_min < 1.0:
            raise ValueError("SurfelTraceOptions.transmittance_min must lie in [0, 1).")


@dataclass(frozen=True, slots=True)
class SurfelCloud:
    """Stores one center, two Gaussian tangent axes, opacity, and scalar value per surfel."""

    center: torch.Tensor
    tangent_u: torch.Tensor
    tangent_v: torch.Tensor
    opacity: torch.Tensor | None = None
    value: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_float_cuda(self.center, "SurfelCloud.center", 2, 3)
        _require_float_cuda(self.tangent_u, "SurfelCloud.tangent_u", 2, 3)
        _require_float_cuda(self.tangent_v, "SurfelCloud.tangent_v", 2, 3)
        count = self.center.shape[0]
        if count < 1 or self.tangent_u.shape != self.center.shape or self.tangent_v.shape != self.center.shape:
            raise ValueError("surfel centers and tangent axes must have the same non-empty shape (M, 3).")
        opacity = self.opacity
        value = self.value
        if opacity is None:
            opacity = torch.ones((count,), dtype=self.center.dtype, device=self.center.device)
            object.__setattr__(self, "opacity", opacity)
        if value is None:
            value = torch.ones((count,), dtype=self.center.dtype, device=self.center.device)
            object.__setattr__(self, "value", value)
        _require_float_cuda(opacity, "SurfelCloud.opacity", 1)
        _require_float_cuda(value, "SurfelCloud.value", 1)
        if opacity.shape != (count,) or value.shape != (count,):
            raise ValueError("SurfelCloud.opacity and value must have shape (M,).")
        tensors = (self.tangent_u, self.tangent_v, opacity, value)
        if any(tensor.device != self.center.device for tensor in tensors):
            raise ValueError("all SurfelCloud tensors must be on the same CUDA device.")

    @property
    def surfel_count(self) -> int:
        return int(self.center.shape[0])


@dataclass(frozen=True, slots=True)
class SurfelIntersection:
    t: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor
    local_uv: torch.Tensor
    gaussian_weight: torch.Tensor
    opacity: torch.Tensor
    alpha: torch.Tensor
    value: torch.Tensor
    surfel_id: torch.Tensor
    triangle_id: torch.Tensor

    def is_valid(self) -> torch.Tensor:
        return self.surfel_id >= 0


@dataclass(frozen=True, slots=True)
class SurfelComposite:
    intensity: torch.Tensor
    alpha: torch.Tensor
    transmittance: torch.Tensor
    depth: torch.Tensor
    candidate_count: torch.Tensor
    candidate_buffer_full: torch.Tensor

    def is_valid(self) -> torch.Tensor:
        return self.alpha > 0.0


class SurfelScene:
    """Accelerates surfel LOS/reflection and evaluates differentiable Gaussian hits."""

    def __init__(self, cloud: SurfelCloud, options: SurfelTraceOptions | None = None) -> None:
        if not isinstance(cloud, SurfelCloud):
            raise TypeError("SurfelScene() expects rayd.torch.SurfelCloud.")
        self.cloud = cloud
        self.options = SurfelTraceOptions() if options is None else options
        if not isinstance(self.options, SurfelTraceOptions):
            raise TypeError("options must be rayd.torch.SurfelTraceOptions.")
        self._proxy: Scene | None = None
        self._build_count = 0

    @property
    def surfel_count(self) -> int:
        return self.cloud.surfel_count

    @property
    def triangle_count(self) -> int:
        return 2 * self.surfel_count

    @property
    def build_count(self) -> int:
        return self._build_count

    def is_ready(self) -> bool:
        return self._proxy is not None and self._proxy.is_ready()

    def build(self) -> None:
        """Build a detached quad proxy; accepted hits are always recomputed from the cloud."""
        cloud = self.cloud
        options = self.options
        with torch.no_grad():
            if options.opacity_aware_proxy_bounds:
                safe_opacity = torch.maximum(cloud.opacity.detach(), cloud.opacity.new_tensor(options.alpha_min))
            else:
                safe_opacity = torch.ones_like(cloud.opacity)
            radius = torch.sqrt(torch.clamp_min(2.0 * torch.log(safe_opacity / options.alpha_min), 0.0))
            if math.isfinite(options.cutoff):
                radius = torch.clamp_max(radius, options.cutoff)
            radius = radius * 1.0001 + 1.0e-6
            scaled_u = radius[:, None] * cloud.tangent_u.detach()
            scaled_v = radius[:, None] * cloud.tangent_v.detach()
            center = cloud.center.detach()
            vertices = (
                torch.stack(
                    (
                        center - scaled_u - scaled_v,
                        center + scaled_u - scaled_v,
                        center + scaled_u + scaled_v,
                        center - scaled_u + scaled_v,
                    ),
                    dim=1,
                )
                .reshape(-1, 3)
                .contiguous()
            )
            base = (torch.arange(self.surfel_count, device=center.device, dtype=torch.int32) * 4)[:, None]
            local_faces = torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int32, device=center.device)
            faces = (base[:, None, :] + local_faces[None, :, :]).reshape(-1, 3).contiguous()
        proxy = Scene()
        proxy.add_mesh(Mesh(vertices, faces, edges_enabled=False))
        proxy.build()
        self._proxy = proxy
        self._build_count += 1

    def _require_ready(self) -> Scene:
        if self._proxy is None or not self._proxy.is_ready():
            raise RuntimeError("SurfelScene is not ready. Call build() before querying.")
        return self._proxy

    @staticmethod
    def _ray_tmax(ray: Ray) -> torch.Tensor:
        if ray.tmax.numel() == 0:
            return torch.full((ray.o.shape[0],), float("inf"), dtype=ray.o.dtype, device=ray.o.device)
        return ray.tmax

    def _analytic_candidate(
        self, ray: Ray, surfel_id: torch.Tensor, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        safe_id = torch.clamp_min(surfel_id, 0).to(torch.int64)
        center = self.cloud.center[safe_id]
        tangent_u = self.cloud.tangent_u[safe_id]
        tangent_v = self.cloud.tangent_v[safe_id]
        opacity = self.cloud.opacity[safe_id]
        value = self.cloud.value[safe_id]

        raw_normal = torch.linalg.cross(tangent_u, tangent_v)
        normal_len_sq = torch.sum(raw_normal * raw_normal, dim=1)
        normal_valid = normal_len_sq > 1.0e-16
        normal = (
            raw_normal / torch.sqrt(torch.where(normal_valid, normal_len_sq, torch.ones_like(normal_len_sq)))[:, None]
        )
        if self.options.face_forward:
            normal = torch.where((torch.sum(normal * ray.d, dim=1) > 0.0)[:, None], -normal, normal)
        denominator = torch.sum(ray.d * normal, dim=1)
        plane_valid = torch.abs(denominator) > 1.0e-8
        safe_denominator = torch.where(plane_valid, denominator, torch.ones_like(denominator))
        plane_t = torch.sum((center - ray.o) * normal, dim=1) / safe_denominator
        valid = (
            candidate
            & normal_valid
            & plane_valid
            & torch.isfinite(plane_t)
            & (plane_t > _RAY_EPSILON)
            & (plane_t < self._ray_tmax(ray))
        )
        point = ray.o + torch.where(valid, plane_t, torch.zeros_like(plane_t))[:, None] * ray.d
        delta = point - center
        uu = torch.sum(tangent_u * tangent_u, dim=1)
        uv = torch.sum(tangent_u * tangent_v, dim=1)
        vv = torch.sum(tangent_v * tangent_v, dim=1)
        du = torch.sum(delta * tangent_u, dim=1)
        dv = torch.sum(delta * tangent_v, dim=1)
        basis_det = uu * vv - uv * uv
        basis_valid = torch.abs(basis_det) > 1.0e-16
        safe_basis_det = torch.where(basis_valid, basis_det, torch.ones_like(basis_det))
        local_u = (du * vv - dv * uv) / safe_basis_det
        local_v = (dv * uu - du * uv) / safe_basis_det
        gaussian = torch.exp(-0.5 * (local_u * local_u + local_v * local_v))
        alpha_uncapped = opacity * gaussian
        slack = 1.0e-6 * max(1.0, self.options.alpha_min)
        valid = valid & basis_valid & (alpha_uncapped + slack >= self.options.alpha_min)
        alpha = torch.where(valid, torch.clamp(alpha_uncapped, 0.0, self.options.alpha_cap), torch.zeros_like(plane_t))
        return plane_t, point, normal, torch.stack((local_u, local_v), dim=1), gaussian, opacity, alpha, value, valid

    def intersect(self, ray: Ray, active: torch.Tensor | None = None) -> SurfelIntersection:
        """Return the nearest accepted Gaussian surfel hit."""
        if not isinstance(ray, Ray):
            raise TypeError("SurfelScene.intersect() expects rayd.torch.Ray.")
        proxy = self._require_ready()
        count = ray.o.shape[0]
        lane_active = _require_active(active, count, ray.o.device)
        result_t = torch.full((count,), float("inf"), dtype=ray.o.dtype, device=ray.o.device)
        result_vec3 = torch.zeros((count, 3), dtype=ray.o.dtype, device=ray.o.device)
        result_uv = torch.zeros((count, 2), dtype=ray.o.dtype, device=ray.o.device)
        result_scalar = torch.zeros((count,), dtype=ray.o.dtype, device=ray.o.device)
        result_id = torch.full((count,), -1, dtype=torch.int32, device=ray.o.device)
        result = (
            result_t,
            result_vec3,
            result_vec3,
            result_uv,
            result_scalar,
            result_scalar,
            result_scalar,
            result_scalar,
            result_id,
            result_id,
        )

        search_origin = ray.o.detach()
        search_direction = ray.d.detach()
        remaining = self._ray_tmax(ray).detach()
        search_active = lane_active.detach()
        for _candidate_index in range(self.options.max_candidate_hits):
            proxy_ray = Ray(search_origin.contiguous(), search_direction.contiguous(), remaining.contiguous())
            proxy_hit = proxy.intersect(proxy_ray, search_active, RayFlags.All)
            proxy_valid = search_active & proxy_hit.is_valid()
            triangle_id = proxy_hit.global_prim_id.to(torch.int32)
            surfel_id = torch.div(torch.clamp_min(triangle_id, 0), 2, rounding_mode="floor").to(torch.int32)
            analytic = self._analytic_candidate(ray, surfel_id, proxy_valid)
            take = analytic[-1] & (result[8] < 0)
            result = tuple(
                torch.where(take[:, None], new, old) if new.ndim == 2 else torch.where(take, new, old)
                for old, new in zip(result[:8], analytic[:8])
            ) + (torch.where(take, surfel_id, result[8]), torch.where(take, triangle_id, result[9]))

            advance = proxy_hit.t.detach() + _RAY_EPSILON
            search_origin = torch.where(
                proxy_valid[:, None], search_origin + advance[:, None] * search_direction, search_origin
            )
            remaining = torch.where(proxy_valid, torch.clamp_min(remaining - advance, 0.0), remaining)
            search_active = proxy_valid & ~take.detach()

        valid = result[8] >= 0
        zero3 = torch.zeros_like(result[1])
        zero2 = torch.zeros_like(result[3])
        zero = torch.zeros_like(result[0])
        return SurfelIntersection(
            torch.where(valid, result[0], torch.full_like(result[0], float("inf"))),
            torch.where(valid[:, None], result[1], zero3),
            torch.where(valid[:, None], result[2], zero3),
            torch.where(valid[:, None], result[3], zero2),
            torch.where(valid, result[4], zero),
            torch.where(valid, result[5], zero),
            torch.where(valid, result[6], zero),
            torch.where(valid, result[7], zero),
            result[8],
            result[9],
        )

    def visible(self, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """Return LOS blocked only by accepted surfels."""
        _require_float_cuda(start, "start", 2, 3)
        _require_float_cuda(end, "end", 2, 3)
        if start.shape != end.shape or start.device != self.cloud.center.device or end.device != start.device:
            raise ValueError("start and end must have equal shape and be on the surfel CUDA device.")
        lane_active = _require_active(active, start.shape[0], start.device)
        delta = end - start
        length_sq = torch.sum(delta * delta, dim=1)
        valid_segment = length_sq > (2.0 * _SHADOW_EPSILON) ** 2
        length = torch.sqrt(torch.where(valid_segment, length_sq, torch.ones_like(length_sq)))
        direction = delta / length[:, None]
        ray = Ray(
            (start + _SHADOW_EPSILON * direction).contiguous(),
            direction.contiguous(),
            torch.clamp_min(length - 2.0 * _SHADOW_EPSILON, 0.0).contiguous(),
        )
        hit = self.intersect(ray, lane_active & valid_segment)
        return lane_active & valid_segment & ~hit.is_valid()

    def trace_reflections(self, ray: Ray, max_bounces: int, active: torch.Tensor | None = None) -> ReflectionChain:
        """Trace specular surfel reflections without creating diffraction paths."""
        if not isinstance(ray, Ray):
            raise TypeError("SurfelScene.trace_reflections() expects rayd.torch.Ray.")
        if max_bounces < 0:
            raise ValueError("max_bounces must be non-negative.")
        count = ray.o.shape[0]
        lane_active = _require_active(active, count, ray.o.device)
        direction = ray.d / torch.clamp_min(torch.linalg.vector_norm(ray.d, dim=1), 1.0e-12)[:, None]
        current_ray = Ray(ray.o, direction.contiguous(), ray.tmax)
        current_image_source = ray.o
        valid_slots: list[torch.Tensor] = []
        t_slots: list[torch.Tensor] = []
        image_slots: list[torch.Tensor] = []
        id_slots: list[torch.Tensor] = []
        for _bounce in range(max_bounces):
            hit = self.intersect(current_ray, lane_active)
            bounce_hit = lane_active & hit.is_valid()
            normal = torch.where((torch.sum(current_ray.d * hit.n, dim=1) > 0.0)[:, None], -hit.n, hit.n)
            distance = torch.sum((current_image_source - hit.p) * normal, dim=1)
            image_source = current_image_source - 2.0 * distance[:, None] * normal
            reflected = current_ray.d - 2.0 * torch.sum(current_ray.d * normal, dim=1)[:, None] * normal
            valid_slots.append(bounce_hit)
            t_slots.append(torch.where(bounce_hit, hit.t, torch.full_like(hit.t, float("inf"))))
            image_slots.append(torch.where(bounce_hit[:, None], image_source, torch.zeros_like(image_source)))
            id_slots.append(torch.where(bounce_hit, hit.surfel_id, torch.full_like(hit.surfel_id, -1)))
            current_ray = Ray(
                torch.where(bounce_hit[:, None], hit.p + _RAY_EPSILON * reflected, current_ray.o).contiguous(),
                torch.where(bounce_hit[:, None], reflected, current_ray.d).contiguous(),
            )
            current_image_source = torch.where(bounce_hit[:, None], image_source, current_image_source)
            lane_active = bounce_hit
        if max_bounces == 0:
            return ReflectionChain(
                torch.empty((count, 0), dtype=torch.bool, device=ray.o.device),
                torch.empty((count, 0), dtype=ray.o.dtype, device=ray.o.device),
                torch.empty((count, 0, 3), dtype=ray.o.dtype, device=ray.o.device),
                torch.empty((count, 0), dtype=torch.int32, device=ray.o.device),
            )
        return ReflectionChain(
            torch.stack(valid_slots, dim=1),
            torch.stack(t_slots, dim=1),
            torch.stack(image_slots, dim=1),
            torch.stack(id_slots, dim=1),
        )

    def composite_alpha(self, ray: Ray, active: torch.Tensor | None = None) -> SurfelComposite:
        """Front-to-back scalar alpha composition; its final transmittance is the surfel transmission."""
        if not isinstance(ray, Ray):
            raise TypeError("SurfelScene.composite_alpha() expects rayd.torch.Ray.")
        self._require_ready()
        count = ray.o.shape[0]
        lane_active = _require_active(active, count, ray.o.device)
        capacity = self.options.max_candidate_hits
        slot_t = [torch.full((count,), float("inf"), dtype=ray.o.dtype, device=ray.o.device) for _ in range(capacity)]
        slot_alpha = [torch.zeros((count,), dtype=ray.o.dtype, device=ray.o.device) for _ in range(capacity)]
        slot_value = [torch.zeros((count,), dtype=ray.o.dtype, device=ray.o.device) for _ in range(capacity)]
        slot_id = [torch.full((count,), -1, dtype=torch.int32, device=ray.o.device) for _ in range(capacity)]
        candidate_count = torch.zeros((count,), dtype=torch.int32, device=ray.o.device)
        for surfel in range(self.surfel_count):
            candidate_id = torch.full((count,), surfel, dtype=torch.int32, device=ray.o.device)
            analytic = self._analytic_candidate(ray, candidate_id, lane_active)
            candidate_t, candidate_alpha, candidate_value, candidate_valid = (
                analytic[0],
                analytic[6],
                analytic[7],
                analytic[8],
            )
            candidate_count = candidate_count + candidate_valid.to(torch.int32)
            for slot in range(capacity):
                old_t, old_alpha, old_value, old_id = slot_t[slot], slot_alpha[slot], slot_value[slot], slot_id[slot]
                before = (candidate_t < old_t - 1.0e-6) | (
                    (torch.abs(candidate_t - old_t) <= 1.0e-6) & ((old_id < 0) | (candidate_id < old_id))
                )
                take = candidate_valid & before
                slot_t[slot] = torch.where(take, candidate_t, old_t)
                slot_alpha[slot] = torch.where(take, candidate_alpha, old_alpha)
                slot_value[slot] = torch.where(take, candidate_value, old_value)
                slot_id[slot] = torch.where(take, candidate_id, old_id)
                candidate_t = torch.where(take, old_t, candidate_t)
                candidate_alpha = torch.where(take, old_alpha, candidate_alpha)
                candidate_value = torch.where(take, old_value, candidate_value)
                candidate_id = torch.where(take, old_id, candidate_id)
                candidate_valid = torch.where(take, old_id >= 0, candidate_valid)

        intensity = torch.zeros((count,), dtype=ray.o.dtype, device=ray.o.device)
        accumulated_alpha = torch.zeros_like(intensity)
        transmittance = torch.ones_like(intensity)
        depth_numerator = torch.zeros_like(intensity)
        compose_active = lane_active
        for slot in range(capacity):
            hit = compose_active & (slot_id[slot] >= 0)
            contribution = torch.where(hit, transmittance * slot_alpha[slot], torch.zeros_like(transmittance))
            intensity = intensity + contribution * slot_value[slot]
            accumulated_alpha = accumulated_alpha + contribution
            depth_numerator = depth_numerator + contribution * torch.where(
                hit, slot_t[slot], torch.zeros_like(slot_t[slot])
            )
            transmittance = transmittance * torch.where(hit, 1.0 - slot_alpha[slot], torch.ones_like(transmittance))
            compose_active = compose_active & (transmittance > self.options.transmittance_min)
        depth = torch.where(
            accumulated_alpha > 0.0,
            depth_numerator / torch.clamp_min(accumulated_alpha, 1.0e-12),
            torch.full_like(accumulated_alpha, float("inf")),
        )
        return SurfelComposite(
            intensity,
            accumulated_alpha,
            torch.where(lane_active, transmittance, torch.ones_like(transmittance)),
            depth,
            torch.minimum(candidate_count, torch.full_like(candidate_count, capacity)),
            lane_active & (candidate_count >= capacity),
        )

    def transmittance(self, ray: Ray, active: torch.Tensor | None = None) -> torch.Tensor:
        """Return the surfel-only transmitted fraction along each ray."""
        return self.composite_alpha(ray, active).transmittance
