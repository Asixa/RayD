# Copyright Xingyu Chen.
# Implements unified differentiable mesh, SDF, and surfel ray queries for Torch.

"""GPU-resident mixed geometry composition without diffraction integration."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .geometry import (
    Intersection,
    Ray,
    RayFlags,
    ReflectionChain,
    SdfIntersection,
    _CONTRACT_VALUES,
    _require_float_cuda_tensor,
)
from .scene import Mesh, Scene
from .sdf import SdfGrid, SdfGridBatch, SdfTraceOptions, _require_active
from .surfel import SurfelCloud, SurfelScene, SurfelTraceOptions


_RAY_EPSILON = float(_CONTRACT_VALUES["ray_epsilon"])


def _invalid_intersection(ray: Ray) -> Intersection:
    count = ray.o.shape[0]
    zero3 = torch.zeros((count, 3), dtype=ray.o.dtype, device=ray.o.device)
    zero2 = torch.zeros((count, 2), dtype=ray.o.dtype, device=ray.o.device)
    invalid_id = torch.full((count,), -1, dtype=torch.int32, device=ray.o.device)
    return Intersection(
        torch.full((count,), float("inf"), dtype=ray.o.dtype, device=ray.o.device),
        zero3,
        zero3,
        zero3,
        zero2,
        zero3,
        invalid_id,
        invalid_id,
        invalid_id,
        invalid_id,
    )


def _select_intersection(current: Intersection, candidate: Intersection, take: torch.Tensor) -> Intersection:
    return Intersection(
        torch.where(take, candidate.t, current.t),
        torch.where(take[:, None], candidate.p, current.p),
        torch.where(take[:, None], candidate.n, current.n),
        torch.where(take[:, None], candidate.geo_n, current.geo_n),
        torch.where(take[:, None], candidate.uv, current.uv),
        torch.where(take[:, None], candidate.barycentric, current.barycentric),
        torch.where(take, candidate.shape_id, current.shape_id),
        torch.where(take, candidate.prim_id, current.prim_id),
        torch.where(take, candidate.local_prim_id, current.local_prim_id),
        torch.where(take, candidate.global_prim_id, current.global_prim_id),
    )


def _mask_intersection_fields(candidate: Intersection, flags: RayFlags) -> Intersection:
    if flags == RayFlags.All:
        return candidate
    count = candidate.t.shape[0]
    zero3 = torch.zeros((count, 3), dtype=candidate.t.dtype, device=candidate.t.device)
    zero2 = torch.zeros((count, 2), dtype=candidate.t.dtype, device=candidate.t.device)
    return Intersection(
        candidate.t,
        candidate.p if flags & RayFlags.Geometric else zero3,
        candidate.n if flags & RayFlags.ShadingN else zero3,
        candidate.geo_n if flags & RayFlags.Geometric else zero3,
        candidate.uv if flags & RayFlags.UV else zero2,
        candidate.barycentric if flags & RayFlags.Geometric else zero3,
        candidate.shape_id,
        candidate.prim_id,
        candidate.local_prim_id,
        candidate.global_prim_id,
    )


class _MixedIntersection:
    __slots__ = ("_full", "_load_full", "_winner", "t")

    def __init__(self, candidate_t: list[torch.Tensor], load_full: list[Callable[[], Intersection]]) -> None:
        self._load_full = load_full
        self.t, self._winner = torch.min(torch.stack(tuple(candidate_t)), dim=0)
        self._full: Intersection | None = None

    def _ensure_full(self) -> Intersection:
        if self._full is None:
            result = self._load_full[0]()
            for index, load_candidate in enumerate(self._load_full[1:], start=1):
                candidate = load_candidate()
                result = _select_intersection(result, candidate, self._winner == index)
            self._full = result
        return self._full

    def is_valid(self) -> torch.Tensor:
        return torch.isfinite(self.t)

    @property
    def p(self) -> torch.Tensor:
        return self._ensure_full().p

    @property
    def n(self) -> torch.Tensor:
        return self._ensure_full().n

    @property
    def geo_n(self) -> torch.Tensor:
        return self._ensure_full().geo_n

    @property
    def uv(self) -> torch.Tensor:
        return self._ensure_full().uv

    @property
    def barycentric(self) -> torch.Tensor:
        return self._ensure_full().barycentric

    @property
    def shape_id(self) -> torch.Tensor:
        return self._ensure_full().shape_id

    @property
    def prim_id(self) -> torch.Tensor:
        return self._ensure_full().prim_id

    @property
    def local_prim_id(self) -> torch.Tensor:
        return self._ensure_full().local_prim_id

    @property
    def global_prim_id(self) -> torch.Tensor:
        return self._ensure_full().global_prim_id


class MixedScene:
    """Owns mesh, bounded SDF, and surfel geometry under one query surface.

    Queries stay on the current CUDA stream. The closest accepted hit is selected
    on the device and derivatives flow only through that fixed winner. A mesh-only
    instance forwards directly to ``Scene``. Diffraction is intentionally absent.
    """

    def __init__(self, trace_backend: str = "auto", edge_bvh_backend: str = "auto") -> None:
        self._mesh_scene = Scene(trace_backend=trace_backend, edge_bvh_backend=edge_bvh_backend)
        self._mesh_count = 0
        self._mesh_face_count = 0
        self._sdfs: list[tuple[SdfGrid, SdfTraceOptions]] = []
        self._sdf_batches: dict[int, tuple[SdfGridBatch, SdfTraceOptions]] = {}
        self._surfels: list[SurfelScene] = []
        self._surfel_prefix: list[int] = []
        self._device: torch.device | None = None
        self._ready = False

    def _accept_device(self, device: torch.device, context: str) -> None:
        if self._device is None:
            self._device = device
        elif device != self._device:
            raise ValueError(f"{context} must use the mixed scene's CUDA device ({device} != {self._device}).")

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("MixedScene.add_mesh() expects rayd.torch.Mesh.")
        self._accept_device(mesh.vertices.device, "mesh")
        mesh_id = self._mesh_scene.add_mesh(mesh, dynamic)
        self._mesh_count += 1
        self._mesh_face_count += int(mesh.faces.shape[0])
        self._ready = False
        return mesh_id

    def add_sdf(self, grid: SdfGrid, options: SdfTraceOptions | None = None) -> int:
        if not isinstance(grid, SdfGrid):
            raise TypeError("MixedScene.add_sdf() expects rayd.torch.SdfGrid.")
        trace_options = SdfTraceOptions() if options is None else options
        if not isinstance(trace_options, SdfTraceOptions):
            raise TypeError("options must be rayd.torch.SdfTraceOptions.")
        self._accept_device(grid.values.device, "SDF grid")
        self._sdfs.append((grid, trace_options))
        self._ready = False
        return len(self._sdfs) - 1

    def add_sdf_batch(self, batch: SdfGridBatch, options: SdfTraceOptions | None = None) -> int:
        """Insert a packed compatible SDF group and return its first logical grid ID."""
        if not isinstance(batch, SdfGridBatch):
            raise TypeError("MixedScene.add_sdf_batch() expects rayd.torch.SdfGridBatch.")
        trace_options = SdfTraceOptions() if options is None else options
        if not isinstance(trace_options, SdfTraceOptions):
            raise TypeError("options must be rayd.torch.SdfTraceOptions.")
        self._accept_device(batch.values.device, "SDF grid batch")
        first = len(self._sdfs)
        self._sdf_batches[first] = (batch, trace_options)
        self._sdfs.extend((batch.grid(index), trace_options) for index in range(batch.grid_count))
        self._ready = False
        return first

    def add_surfel(self, surfel: SurfelCloud | SurfelScene, options: SurfelTraceOptions | None = None) -> int:
        if isinstance(surfel, SurfelCloud):
            scene = SurfelScene(surfel, options)
        elif isinstance(surfel, SurfelScene):
            if options is not None:
                raise ValueError("options must be omitted when adding an existing SurfelScene.")
            scene = surfel
        else:
            raise TypeError("MixedScene.add_surfel() expects rayd.torch.SurfelCloud or SurfelScene.")
        self._accept_device(scene.cloud.center.device, "surfel scene")
        self._surfels.append(scene)
        self._ready = False
        return len(self._surfels) - 1

    def build(self) -> None:
        if self._mesh_count:
            self._mesh_scene.build()
        prefix = 0
        self._surfel_prefix = []
        for scene in self._surfels:
            if not scene.is_ready():
                scene.build()
            self._surfel_prefix.append(prefix)
            prefix += scene.surfel_count
        self._ready = True

    def is_ready(self) -> bool:
        return (
            self._ready
            and (self._mesh_count == 0 or self._mesh_scene.is_ready())
            and all(scene.is_ready() for scene in self._surfels)
        )

    @property
    def num_meshes(self) -> int:
        return self._mesh_count

    @property
    def num_sdfs(self) -> int:
        return len(self._sdfs)

    @property
    def num_surfel_scenes(self) -> int:
        return len(self._surfels)

    def _require_ready(self, ray: Ray | None = None) -> None:
        if not self.is_ready():
            raise RuntimeError("MixedScene is not ready. Call build() before querying.")
        if ray is not None and self._device is not None and ray.o.device != self._device:
            raise ValueError("ray must be on the mixed scene's CUDA device.")

    def _sdf_candidate(
        self, grid: SdfGrid, options: SdfTraceOptions, ray: Ray, active: torch.Tensor, index: int, flags: RayFlags
    ) -> tuple[torch.Tensor, Callable[[], Intersection]]:
        hit = grid.intersect(
            ray, active=active, max_steps=options.max_steps, relaxation=options.relaxation, eps_hit=options.eps_hit
        )
        return self._sdf_candidate_from_hit(hit, ray, index, flags)

    def _sdf_candidate_from_hit(
        self, hit: SdfIntersection, ray: Ray, index: int, flags: RayFlags
    ) -> tuple[torch.Tensor, Callable[[], Intersection]]:
        def load_full() -> Intersection:
            count = ray.o.shape[0]
            shape_id = torch.full((count,), self._mesh_count + index, dtype=torch.int32, device=ray.o.device)
            prim_id = torch.zeros((count,), dtype=torch.int32, device=ray.o.device)
            global_id = torch.full((count,), self._mesh_face_count + index, dtype=torch.int32, device=ray.o.device)
            zero2 = torch.zeros((count, 2), dtype=ray.o.dtype, device=ray.o.device)
            zero3 = torch.zeros((count, 3), dtype=ray.o.dtype, device=ray.o.device)
            geometric = bool(flags & RayFlags.Geometric)
            shading_n = bool(flags & RayFlags.ShadingN)
            return Intersection(
                hit.t,
                hit.position if geometric else zero3,
                hit.normal if shading_n else zero3,
                hit.normal if geometric else zero3,
                zero2,
                zero3,
                shape_id,
                prim_id,
                prim_id,
                global_id,
            )

        return hit.t, load_full

    def _surfel_candidate(
        self, scene: SurfelScene, ray: Ray, active: torch.Tensor, index: int, flags: RayFlags
    ) -> tuple[torch.Tensor, Callable[[], Intersection]]:
        hit = scene.intersect(ray, active)

        def load_full() -> Intersection:
            count = ray.o.shape[0]
            shape_id = torch.full(
                (count,), self._mesh_count + len(self._sdfs) + index, dtype=torch.int32, device=ray.o.device
            )
            global_id = self._mesh_face_count + len(self._sdfs) + self._surfel_prefix[index] + hit.surfel_id
            zero3 = torch.zeros((count, 3), dtype=ray.o.dtype, device=ray.o.device)
            geometric = bool(flags & RayFlags.Geometric)
            shading_n = bool(flags & RayFlags.ShadingN)
            uv = bool(flags & RayFlags.UV)
            return Intersection(
                hit.t,
                hit.p if geometric else zero3,
                hit.n if shading_n else zero3,
                hit.n if geometric else zero3,
                hit.local_uv if uv else torch.zeros((count, 2), dtype=ray.o.dtype, device=ray.o.device),
                zero3,
                shape_id,
                hit.surfel_id,
                hit.surfel_id,
                global_id.to(torch.int32),
            )

        return hit.t, load_full

    def intersect(self, ray: Ray, active: torch.Tensor | None = None, flags: RayFlags = RayFlags.All) -> Intersection:
        if not isinstance(ray, Ray):
            raise TypeError("MixedScene.intersect() expects rayd.torch.Ray.")
        self._require_ready(ray)
        if self._mesh_count and not self._sdfs and not self._surfels:
            return self._mesh_scene.intersect(ray, active, flags)
        lane_active = _require_active(active, ray.o.shape[0], ray.o.device)
        if self._mesh_count:
            first = self._mesh_scene.intersect(ray, lane_active, flags)
            candidate_t = [first.t]
            if int(flags) == 0:

                def load_first() -> Intersection:
                    full = self._mesh_scene.intersect(ray, lane_active, RayFlags.Geometric)
                    return _mask_intersection_fields(full, flags)

            else:

                def load_first() -> Intersection:
                    return _mask_intersection_fields(first, flags)

        else:
            invalid_t = torch.full((ray.o.shape[0],), float("inf"), dtype=ray.o.dtype, device=ray.o.device)
            candidate_t = [invalid_t]

            def load_first() -> Intersection:
                return _invalid_intersection(ray)

        load_full = [load_first]
        index = 0
        while index < len(self._sdfs):
            batch_entry = self._sdf_batches.get(index)
            if batch_entry is None:
                grid, options = self._sdfs[index]
                candidate, load_candidate = self._sdf_candidate(grid, options, ray, lane_active, index, flags)
                candidate_t.append(candidate)
                load_full.append(load_candidate)
                index += 1
                continue
            batch, options = batch_entry
            hits = batch.intersect(
                ray,
                active=lane_active,
                max_steps=options.max_steps,
                relaxation=options.relaxation,
                eps_hit=options.eps_hit,
            )
            for local_index, hit in enumerate(hits):
                candidate, load_candidate = self._sdf_candidate_from_hit(hit, ray, index + local_index, flags)
                candidate_t.append(candidate)
                load_full.append(load_candidate)
            index += batch.grid_count
        for index, scene in enumerate(self._surfels):
            candidate, load_candidate = self._surfel_candidate(scene, ray, lane_active, index, flags)
            candidate_t.append(candidate)
            load_full.append(load_candidate)
        return _MixedIntersection(candidate_t, load_full)  # type: ignore[return-value]

    def visible(self, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        _require_float_cuda_tensor(start, "start", 3)
        _require_float_cuda_tensor(end, "end", 3)
        if end.shape != start.shape:
            raise ValueError("start and end must have equal shape (N, 3).")
        self._require_ready()
        if self._device is not None and start.device != self._device:
            raise ValueError("start and end must be on the mixed scene's CUDA device.")
        lane_active = _require_active(active, start.shape[0], start.device)
        result = lane_active
        if self._mesh_count:
            result = result & self._mesh_scene.visible(start, end, lane_active)
        for grid, options in self._sdfs:
            result = result & grid.visible(
                start,
                end,
                lane_active,
                max_steps=options.max_steps,
                relaxation=options.relaxation,
                eps_hit=options.eps_hit,
            )
        for scene in self._surfels:
            result = result & scene.visible(start, end, lane_active)
        return result

    def transmittance(self, ray: Ray, active: torch.Tensor | None = None) -> torch.Tensor:
        """Return surfel alpha transmission with mesh hits opaque and SDFs ignored."""
        if not isinstance(ray, Ray):
            raise TypeError("MixedScene.transmittance() expects rayd.torch.Ray.")
        self._require_ready(ray)
        lane_active = _require_active(active, ray.o.shape[0], ray.o.device)
        opaque = torch.zeros_like(lane_active)
        if self._mesh_count:
            opaque = opaque | self._mesh_scene.intersect(ray, lane_active, RayFlags(0)).is_valid()
        result = torch.ones((ray.o.shape[0],), dtype=ray.o.dtype, device=ray.o.device)
        for scene in self._surfels:
            result = result * scene.transmittance(ray, lane_active)
        return torch.where(lane_active, torch.where(opaque, torch.zeros_like(result), result), torch.ones_like(result))

    def _reflection_bias(self, shape_id: torch.Tensor) -> torch.Tensor:
        bias = torch.full(shape_id.shape, _RAY_EPSILON, dtype=torch.float32, device=shape_id.device)
        for index, (grid, options) in enumerate(self._sdfs):
            bias = torch.where(shape_id == self._mesh_count + index, grid._query_bias(options.eps_hit), bias)
        return bias

    def trace_reflections(self, ray: Ray, max_bounces: int, active: torch.Tensor | None = None) -> ReflectionChain:
        if not isinstance(ray, Ray):
            raise TypeError("MixedScene.trace_reflections() expects rayd.torch.Ray.")
        if max_bounces < 0:
            raise ValueError("max_bounces must be non-negative.")
        self._require_ready(ray)
        if self._mesh_count and not self._sdfs and not self._surfels:
            return self._mesh_scene.trace_reflections(ray, max_bounces, active)
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
        valid_slots: list[torch.Tensor] = []
        t_slots: list[torch.Tensor] = []
        image_slots: list[torch.Tensor] = []
        id_slots: list[torch.Tensor] = []
        for _bounce in range(max_bounces):
            hit = self.intersect(current_ray, lane_active, RayFlags.All)
            bounce_hit = lane_active & hit.is_valid()
            normal = torch.where((torch.sum(current_ray.d * hit.geo_n, dim=1) > 0.0)[:, None], -hit.geo_n, hit.geo_n)
            distance = torch.sum((current_image_source - hit.p) * normal, dim=1)
            image_source = current_image_source - 2.0 * distance[:, None] * normal
            reflected = current_ray.d - 2.0 * torch.sum(current_ray.d * normal, dim=1)[:, None] * normal
            valid_slots.append(bounce_hit)
            t_slots.append(torch.where(bounce_hit, hit.t, torch.full_like(hit.t, float("inf"))))
            image_slots.append(torch.where(bounce_hit[:, None], image_source, torch.zeros_like(image_source)))
            id_slots.append(torch.where(bounce_hit, hit.global_prim_id, torch.full_like(hit.global_prim_id, -1)))
            bias = self._reflection_bias(hit.shape_id)
            current_ray = Ray(
                torch.where(bounce_hit[:, None], hit.p + bias[:, None] * reflected, current_ray.o).contiguous(),
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
