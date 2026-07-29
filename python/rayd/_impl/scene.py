# Copyright Xingyu Chen.
# Implements shared Python support for scene.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import torch

from .multipath import _require_native_dispatcher
from .multipath import accum_dfr_chain_native as _accum_dfr_chain
from .multipath import accum_dfr_coherent_direct_native as _accum_dfr_coherent_direct
from .multipath import accum_dfr_direct_native as _accum_dfr_direct
from .multipath import intersect as _intersect
from .multipath import nearest_edge as _nearest_edge
from .multipath import nearest_edge_ray as _nearest_edge_ray
from .multipath import trace_dfr_paths_order1_native as _trace_dfr_paths
from .multipath import trace_refl_epc_field as _trace_refl_epc_field
from .multipath import trace_reflections as _trace_reflections
from .multipath import visible as _visible
from .geometry import (
    DfrGrid,
    DfrMaterial,
    DfrStates,
    Intersection,
    Ray,
    RayFlags,
    _LazyIntersection,
    _ReducedIntersection,
)

if TYPE_CHECKING:
    # Annotation-only imports. `._multi` in particular must stay unimported at
    # runtime for a single-device scene (D9), and the remaining names are the
    # result records the public signatures below name.
    from collections.abc import Callable, Iterable, Sequence
    from typing import Any

    from . import multi as _multi
    from .geometry import (
        AxialEdgeVisibility,
        DfrAccum,
        DfrCoherentAccum,
        DfrPaths,
        NearestEdgesTopK,
        NearestPointEdge,
        NearestRayEdge,
        ReflEpcField,
        ReflectionChain,
        SceneGlobalGeometry,
        SegmentChainVisibility,
        SegmentPairVisibility,
    )


# Mesh and single-device Scene intentionally share one lifecycle owner.


def _empty_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _require_tensor(value: torch.Tensor, name: str, dtype: torch.dtype, rank: int, last_dim: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be CUDA.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if last_dim is not None and value.shape[-1] != last_dim:
        raise ValueError(f"{name} last dimension must be {last_dim}.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _require_transform(value: torch.Tensor, name: str) -> None:
    _require_tensor(value, name, torch.float32, 2, 4)
    if value.shape[0] not in (0, 4):
        raise ValueError(f"{name} must be empty or have shape (4, 4).")


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
        _require_tensor(self.vertices, "vertices", torch.float32, 2, 3)
        _require_tensor(self.faces, "faces", torch.int32, 2, 3)
        if self.uv is not None:
            _require_tensor(self.uv, "uv", torch.float32, 2, 2)
        if self.face_uv is not None:
            _require_tensor(self.face_uv, "face_uv", torch.int32, 2, 3)
        if self.uv is None:
            self.uv = _empty_tensor((0, 2), torch.float32, self.vertices.device)
        if self.face_uv is None:
            self.face_uv = _empty_tensor((0, 3), torch.int32, self.vertices.device)
        if self.to_world_left is None:
            self.to_world_left = _empty_tensor((0, 4), torch.float32, self.vertices.device)
        else:
            _require_transform(self.to_world_left, "to_world_left")
        if self.to_world_right is None:
            self.to_world_right = _empty_tensor((0, 4), torch.float32, self.vertices.device)
        else:
            _require_transform(self.to_world_right, "to_world_right")


def _native_scene_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if torch._C._functorch.is_functorch_wrapped_tensor(value) or torch._C._functorch.is_gradtrackingtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    try:
        value.data_ptr()
    except RuntimeError:
        value = value.detach().clone()
    return value


def _has_reverse_or_forward_ad(*values: torch.Tensor) -> bool:
    if torch.autograd.forward_ad._current_level < 0:
        return any(value.requires_grad for value in values)
    for value in values:
        unpacked = torch.autograd.forward_ad.unpack_dual(value)
        if unpacked.primal.requires_grad or unpacked.tangent is not None:
            return True
    return False


def _has_forward_ad(*values: torch.Tensor) -> bool:
    if torch.autograd.forward_ad._current_level < 0:
        return False
    for value in values:
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


class Scene:
    def __init__(
        self,
        trace_backend: str = "auto",
        edge_bvh_backend: str = "auto",
        *,
        devices: Sequence[int | str | torch.device] | None = None,
        options: _multi.MultiDeviceOptions | None = None,
    ) -> None:
        trace_backends = {"auto": 0, "optix": 1, "cuda": 2}
        edge_backends = {"auto": 0, "optix": 1, "cuda": 2}
        if trace_backend not in trace_backends:
            raise ValueError("trace_backend must be 'auto', 'optix', or 'cuda'.")
        if edge_bvh_backend not in edge_backends:
            raise ValueError("edge_bvh_backend must be 'auto', 'optix', or 'cuda'.")
        self._trace_backend_code = trace_backends[trace_backend]
        self._edge_backend_code = edge_backends[edge_bvh_backend]
        self._meshes: list[tuple[Mesh, bool]] = []
        self._native_scene = None
        self._native_handle = 0
        self._ready = False
        self._pending_updates = False
        # `None` for every scene that did not ask for several devices or for
        # chunked execution, which is what every op checks before dispatching.
        # `rayd._impl.multi` is imported only when `devices=` is passed, and it
        # still returns `None` for a one-device scene that wants nothing from
        # the chunked executor, so the single-device path is untouched (D9).
        # Deliberately `Any` and not `_multi._ReplicatedScene | None`: the
        # orchestrator's per-op methods answer `X | None`, because the `offload`
        # hook streams chunks instead of returning them, so naming the real type
        # here would only trade this one annotation for a `# type: ignore` on
        # every `return self._multi.<op>(...)` below without adding precision.
        self._multi: Any = None
        if devices is not None or options is not None:
            from .multi import plan as _plan_multi_device

            self._multi = _plan_multi_device(
                devices, options, trace_backend=trace_backend, edge_bvh_backend=edge_bvh_backend
            )

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects rayd.torch.Mesh.")
        if self._native_scene is not None:
            self._native_scene = None
        if self._multi is not None:
            self._multi.discard()
        self._meshes.append((mesh, bool(dynamic)))
        self._ready = False
        self._pending_updates = False
        return len(self._meshes) - 1

    def _mesh_spec(self, mesh: Mesh, dynamic: bool) -> dict[str, object]:
        return {
            "vertices": _native_scene_tensor(mesh.vertices),
            "faces": _native_scene_tensor(mesh.faces),
            "uv": _native_scene_tensor(mesh.uv),
            "face_uv": _native_scene_tensor(mesh.face_uv),
            "to_world_left": _native_scene_tensor(mesh.to_world_left),
            "to_world_right": _native_scene_tensor(mesh.to_world_right),
            "use_face_normals": mesh.use_face_normals,
            "edges_enabled": mesh.edges_enabled,
            "dynamic": dynamic,
        }

    def build(self) -> None:
        if self._multi is not None:
            # One ordinary single-device Scene per device; the master replica's
            # native scene answers this object's scene-level metadata.
            try:
                self._multi.build(self._meshes)
                self._native_scene = self._multi.master_native_scene()
                self._native_handle = int(self._native_scene.handle())
            except BaseException:
                # Do not leave stale master metadata looking usable when only
                # part of a replica rebuild completed.
                self._native_scene = None
                self._native_handle = 0
                self._ready = False
                self._pending_updates = True
                raise
            self._ready = True
            self._pending_updates = False
            return
        _require_native_dispatcher()
        specs = [self._mesh_spec(mesh, dynamic) for mesh, dynamic in self._meshes]
        mesh_flags = []
        for mesh, dynamic in self._meshes:
            flags = 0
            if mesh.use_face_normals:
                flags |= 1
            if mesh.edges_enabled:
                flags |= 2
            if dynamic:
                flags |= 4
            mesh_flags.append(flags)
        if mesh_flags:
            mesh_flags[0] |= self._trace_backend_code << 8
            mesh_flags[0] |= self._edge_backend_code << 10
        with torch._C._DisableFuncTorch():
            native_scene = torch.classes.rayd_torch.Scene(
                [spec["vertices"] for spec in specs],
                [spec["faces"] for spec in specs],
                [spec["uv"] for spec in specs],
                [spec["face_uv"] for spec in specs],
                [spec["to_world_left"] for spec in specs],
                [spec["to_world_right"] for spec in specs],
                mesh_flags,
            )
        self._native_scene = native_scene
        self._native_handle = int(native_scene.handle())
        self._ready = True
        self._pending_updates = False

    def _require_native_scene(self):
        if self._multi is not None:
            self._multi.require_healthy()
        if not self._ready or self._native_scene is None:
            raise RuntimeError("Scene is not ready. Call build() before querying.")
        return self._native_scene

    def calibrate_devices(
        self,
        *,
        operation: str | None = None,
        rays: int = 1 << 20,
        max_bounces: int = 0,
        probe: Callable[[Any, torch.device], object] | None = None,
        repeats: int = 3,
        warm_up: int = 1,
        refine: bool = True,
    ) -> _multi.DeviceCalibration:
        """Measure this scene's devices and set the shard split from what it sees.

        Only a scene built with more than one device has a split to set;
        anything else raises, because silently doing nothing -- or answering a
        one-device scene with the `(1.0,)` it already had -- would look like a
        calibration. That includes a one-device `Scene(devices=[d],
        options=MultiDeviceOptions(chunk_rays=...))`, which is orchestrated
        (the chunked executor is a per-device memory story, D7) but has nothing
        to shard. Every replica first runs the same probe on its own device with resident
        inputs, so what is measured is the device rather than the interconnect,
        and the weights come out inversely proportional to the times. The
        refinement stage then times the real multi-device dispatch of the same
        probe at a ladder of remote shares down to zero and keeps the fastest,
        which is what catches an operation that moves more bytes per row than
        it spends compute on. The returned record (also kept on the scene)
        carries the per-device seconds, the ladder, and the weights they
        produced, so a caller can log exactly what it is running.

        The default probe is `rays` rays drawn from a fixed seed inside this
        scene's bounding box, put through full `intersect` -- or through
        `trace_reflections` when `max_bounces` is set. It is an op *shape*, not
        a workload: pass both `operation="..."` and `probe(scene, device)` to
        time another call. The chosen split is stored only for that exact
        operation; use `device_weights_for()` to inspect it. Calibration only
        chooses weights, and is a noisy measurement rather than a performance
        guarantee.
        """
        if self._multi is None or len(self._multi.devices) < 2:
            raise RuntimeError(
                "Scene.calibrate_devices() needs a scene with more than one device; "
                "build the Scene with devices=[...] first. There is no shard split "
                "to measure on one device."
            )
        return self._multi.calibrate(
            operation=operation,
            rows=int(rays),
            max_bounces=int(max_bounces),
            probe=probe,
            repeats=int(repeats),
            warm_up=int(warm_up),
            refine=bool(refine),
        )

    @property
    def device_weights(self) -> tuple[float, ...] | None:
        """The latest calibrated/base split, or `None` when nothing orchestrates.

        A one-device scene that engaged the chunked executor answers `(1.0,)`:
        it has a (degenerate) split, it just has no way to change it.
        Prefer `device_weights_for(operation)` when operation-local weights or
        more than one calibration are in use.
        """
        if self._multi is None:
            return None
        return self._multi.weights

    def device_weights_for(self, operation: str) -> tuple[float, ...] | None:
        """Effective split for one operation, including operation-local calibration."""
        if self._multi is None:
            return None
        if not isinstance(operation, str) or not operation:
            raise TypeError("Scene.device_weights_for() expects a non-empty operation name.")
        return self._multi.weights_for(operation)

    def _mesh_vertex_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(mesh.vertices for mesh, _dynamic in self._meshes)

    @staticmethod
    def _forward_intersection(scene, ray: Ray, active, flags: int) -> Intersection:
        if flags == 0:
            t = torch.ops.rayd_torch.intersect_forward_t(scene, ray.o, ray.d, ray.tmax, active)
            # The lazy stand-ins answer the `Intersection` field surface without
            # materializing the fields nobody asked for, so the public result
            # type of an intersection query stays `Intersection`.
            return _ReducedIntersection(scene, t)  # type: ignore[return-value]
        values = torch.ops.rayd_torch.intersect_forward_flags(scene, ray.o, ray.d, ray.tmax, active, flags)
        return Intersection(*values)

    @staticmethod
    def _lazy_intersection(scene, vertices, ray: Ray, active, flags: int) -> Intersection:
        def load_t():
            return torch.ops.rayd_torch.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active)

        def load_full():
            values = torch.ops.rayd_torch.intersect_ad_flags(scene, vertices, ray.o, ray.d, ray.tmax, active, flags)
            return Intersection(*values)

        return _LazyIntersection(load_t, load_full)  # type: ignore[return-value]

    def is_ready(self) -> bool:
        return self._ready and (self._multi is None or not self._multi.is_poisoned)

    @property
    def trace_backend(self) -> str:
        return str(self._require_native_scene().trace_backend())

    @property
    def edge_bvh_backend(self) -> str:
        return str(self._require_native_scene().edge_backend())

    @property
    def num_meshes(self) -> int:
        scene = self._require_native_scene()
        return int(scene.num_meshes())

    @property
    def version(self) -> int:
        scene = self._require_native_scene()
        return int(scene.version())

    def intersect(self, ray: Ray, active: torch.Tensor | None = None, flags: RayFlags = RayFlags.All) -> Intersection:
        if self._multi is not None:
            return self._multi.intersect(ray, active, int(flags))
        scene = self._require_native_scene()
        flags_value = int(flags)
        if len(self._meshes) == 1 and torch.autograd.forward_ad._current_level < 0:
            vertices = self._meshes[0][0].vertices
            if not (vertices.requires_grad or ray.o.requires_grad or ray.d.requires_grad or ray.tmax.requires_grad):
                return self._forward_intersection(scene, ray, active, flags_value)
            if (
                torch.compiler.is_compiling()
                and flags_value == 0
                and active is None
                and vertices.requires_grad
                and not (ray.o.requires_grad or ray.d.requires_grad or ray.tmax.requires_grad)
            ):
                t, _tape_prim = torch.ops.rayd_torch.intersect_forward_tape_h(
                    self._native_handle, vertices, ray.o, ray.d, ray.tmax
                )
                return _ReducedIntersection(scene, t)  # type: ignore[return-value]
            if flags_value != 0:
                return self._lazy_intersection(scene, vertices, ray, active, flags_value)
            t = torch.ops.rayd_torch.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active)
            return _ReducedIntersection(scene, t)  # type: ignore[return-value]
        mesh_vertices = self._mesh_vertex_tensors()
        if not _has_reverse_or_forward_ad(*mesh_vertices, ray.o, ray.d, ray.tmax):
            return self._forward_intersection(scene, ray, active, flags_value)
        if len(mesh_vertices) == 1:
            vertices = mesh_vertices[0]
            if not _has_forward_ad(vertices, ray.o, ray.d, ray.tmax):
                if flags_value != 0:
                    return self._lazy_intersection(scene, vertices, ray, active, flags_value)
                t = torch.ops.rayd_torch.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active)
                return _ReducedIntersection(scene, t)  # type: ignore[return-value]
            return _intersect(scene, vertices, ray.o, ray.d, ray.tmax, active, flags_value)
        return _intersect(
            scene, mesh_vertices[0], ray.o, ray.d, ray.tmax, active, flags_value, mesh_vertices=mesh_vertices
        )

    @overload
    def nearest_edge(self, point: torch.Tensor) -> NearestPointEdge: ...

    @overload
    def nearest_edge(self, point: Ray) -> NearestRayEdge: ...

    def nearest_edge(self, point: torch.Tensor | Ray) -> NearestPointEdge | NearestRayEdge:
        if self._multi is not None:
            return self._multi.nearest_edge(point)
        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        if isinstance(point, Ray):
            return _nearest_edge_ray(scene, mesh_vertices[0], point.o, point.d, point.tmax, None)
        return _nearest_edge(scene, mesh_vertices[0], point, mesh_vertices=mesh_vertices)

    def visible(self, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        if self._multi is not None:
            return self._multi.visible(start, end, active)
        scene = self._require_native_scene()
        return _visible(scene, start, end, active)

    def trace_reflections(self, ray: Ray, max_bounces: int, active: torch.Tensor | None = None) -> ReflectionChain:
        if self._multi is not None:
            return self._multi.trace_reflections(ray, int(max_bounces), active)
        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        return _trace_reflections(
            scene, mesh_vertices[0], ray.o, ray.d, ray.tmax, active, int(max_bounces), mesh_vertices=mesh_vertices
        )

    def trace_refl_epc_field(
        self, source: torch.Tensor, receiver: torch.Tensor, max_bounces: int, active: torch.Tensor | None = None
    ) -> ReflEpcField:
        if self._multi is not None:
            return self._multi.trace_refl_epc_field(source, receiver, int(max_bounces), active)
        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        return _trace_refl_epc_field(
            scene, mesh_vertices[0], source, receiver, active, int(max_bounces), mesh_vertices=mesh_vertices
        )

    def _default_dfr_material(self, *, like: torch.Tensor) -> DfrMaterial:
        face_count = sum(int(mesh.faces.shape[0]) for mesh, _dynamic in self._meshes)
        return DfrMaterial(*torch.ops.rayd_torch.default_dfr_material(face_count, like))

    def trace_dfr_paths(
        self,
        *,
        tx_positions: torch.Tensor,
        rx_positions: torch.Tensor,
        states: DfrStates,
        material: DfrMaterial | None = None,
        active: torch.Tensor,
        max_paths: int | None = None,
        wavelength: float = 1.0,
    ) -> DfrPaths:
        if self._multi is not None:
            self._multi.unsupported("trace_dfr_paths")
        scene = self._require_native_scene()
        if material is None:
            material = self._default_dfr_material(like=states.edge_pos)
        if max_paths is None:
            max_paths = states.state_count
        return _trace_dfr_paths(
            scene,
            tx_positions,
            rx_positions,
            states,
            material,
            active=active,
            max_paths=int(max_paths),
            wavelength=float(wavelength),
        )

    def accum_dfr_direct(
        self,
        *,
        states: DfrStates | None = None,
        grid: DfrGrid | None = None,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        wavelength: float = 1.0,
        direct_samples: int = 0,
        keller_samples: int = 0,
        suffix_samples: int = 0,
        seed: int = 0,
        lane_offset: int = 0,
        lane_count: int = -1,
    ) -> DfrAccum:
        scene = self._require_native_scene()
        if states is None:
            raise TypeError("Scene.accum_dfr_direct() requires RayD-style DfrStates and DfrGrid.")
        if grid is None:
            raise TypeError("Scene.accum_dfr_direct() requires DfrGrid when states are provided.")
        if material is None:
            material = self._default_dfr_material(like=states.edge_pos)
        if self._multi is not None:
            # Sharded over the Monte-Carlo lane space, not over a batch axis:
            # every device runs the same states over its own sample window.
            return self._multi.accum_dfr_direct(
                states=states,
                grid=grid,
                material=material,
                active=active,
                wavelength=float(wavelength),
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                suffix_samples=int(suffix_samples),
                seed=int(seed),
                lane_offset=int(lane_offset),
                lane_count=int(lane_count),
            )
        return _accum_dfr_direct(
            scene,
            states,
            grid,
            material,
            active=active,
            wavelength=float(wavelength),
            direct_samples=int(direct_samples),
            keller_samples=int(keller_samples),
            suffix_samples=int(suffix_samples),
            seed=int(seed),
            lane_offset=int(lane_offset),
            lane_count=int(lane_count),
        )

    def accum_dfr(
        self,
        initial_states: DfrStates | None = None,
        recursive_states: DfrStates | None = None,
        grid: DfrGrid | None = None,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        recursive_active: torch.Tensor | None = None,
        wavelength: float = 1.0,
        direct_samples: int = 0,
        keller_samples: int = 0,
        suffix_samples: int = 0,
        seed: int = 0,
        max_order: int = 2,
        lane_offset: int = 0,
        lane_count: int = -1,
        **kwargs: object,
    ) -> DfrAccum:
        if initial_states is None and recursive_states is None:
            return self.accum_dfr_direct(
                lane_offset=int(lane_offset),
                lane_count=int(lane_count),
                **kwargs,  # type: ignore[arg-type]
            )
        scene = self._require_native_scene()
        if initial_states is None or recursive_states is None or grid is None:
            raise TypeError("Scene.accum_dfr() requires initial_states, recursive_states, and grid.")
        if material is None:
            material = self._default_dfr_material(like=initial_states.edge_pos)
        if self._multi is not None:
            return self._multi.accum_dfr(
                initial_states=initial_states,
                recursive_states=recursive_states,
                grid=grid,
                material=material,
                active=active,
                recursive_active=recursive_active,
                wavelength=float(wavelength),
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                suffix_samples=int(suffix_samples),
                seed=int(seed),
                max_order=int(max_order),
                lane_offset=int(lane_offset),
                lane_count=int(lane_count),
            )
        return _accum_dfr_chain(
            scene,
            initial_states,
            recursive_states,
            grid,
            material,
            active=active,
            recursive_active=recursive_active,
            wavelength=float(wavelength),
            direct_samples=int(direct_samples),
            keller_samples=int(keller_samples),
            suffix_samples=int(suffix_samples),
            seed=int(seed),
            max_order=int(max_order),
            lane_offset=int(lane_offset),
            lane_count=int(lane_count),
        )

    def accum_dfr_coherent_direct(
        self,
        *,
        states: DfrStates,
        grid: DfrGrid,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        wavelength: float = 1.0,
        select_diffraction_point: bool = True,
        prefilter_visibility: bool = True,
    ) -> DfrCoherentAccum:
        if self._multi is not None:
            self._multi.unsupported("accum_dfr_coherent_direct")
        scene = self._require_native_scene()
        if material is None:
            material = self._default_dfr_material(like=states.edge_pos)
        return _accum_dfr_coherent_direct(
            scene,
            states,
            grid,
            material,
            active=active,
            wavelength=float(wavelength),
            select_diffraction_point=bool(select_diffraction_point),
            prefilter_visibility=bool(prefilter_visibility),
        )

    def update_mesh_vertices(self, mesh_id: int, positions: torch.Tensor) -> None:
        scene = self._require_native_scene()
        mesh, dynamic = self._meshes[mesh_id]
        if not dynamic:
            raise RuntimeError("Scene.update_mesh_vertices(): target mesh is not dynamic.")
        updated_vertices = positions.contiguous()
        if self._multi is not None:
            # Broadcast: every replica takes its own copy of the new positions,
            # including the master, whose native scene this object reads.
            try:
                self._multi.update_mesh_vertices(int(mesh_id), updated_vertices)
            except BaseException:
                self._ready = False
                self._pending_updates = True
                raise
            # Commit the public mesh only after every replica accepted the
            # broadcast.  A failed broadcast is recovered by build(), and that
            # rebuild must use the last caller-visible committed geometry.
            mesh.vertices = updated_vertices
            self._pending_updates = True
            return
        mesh.vertices = updated_vertices
        with torch._C._DisableFuncTorch():
            scene.update_vertices(int(mesh_id), _native_scene_tensor(mesh.vertices))
        self._pending_updates = True

    def sync(self) -> None:
        if self._multi is not None:
            self._require_native_scene()
            try:
                self._multi.sync()
            except BaseException:
                self._ready = False
                self._pending_updates = True
                raise
            self._pending_updates = False
            return
        scene = self._require_native_scene()
        with torch._C._DisableFuncTorch():
            scene.sync()
        self._pending_updates = False

    def has_pending_updates(self) -> bool:
        return bool(self._pending_updates or (self._multi is not None and self._multi.is_poisoned))

    def nearest_edges(self, point: torch.Tensor, k: int, active: torch.Tensor | None = None) -> NearestEdgesTopK:
        if self._multi is not None:
            return self._multi.nearest_edges(point, int(k), active)
        from .multipath import nearest_edges as _nearest_edges

        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        return _nearest_edges(scene, mesh_vertices[0], point, int(k), active, mesh_vertices=mesh_vertices)

    def edge_mask(self) -> torch.Tensor:
        scene = self._require_native_scene()
        return scene.edge_mask()

    def set_edge_mask(self, mask: torch.Tensor) -> None:
        scene = self._require_native_scene()
        if self._multi is not None:
            try:
                self._multi.set_edge_mask(mask)
            except BaseException:
                self._ready = False
                self._pending_updates = True
                raise
            return
        scene.set_edge_mask(mask)

    def global_geometry(self) -> SceneGlobalGeometry:
        from .geometry import SceneGlobalGeometry

        scene = self._require_native_scene()
        return SceneGlobalGeometry(*scene.global_geometry())

    def visible_pair(
        self,
        start: torch.Tensor,
        end_a: torch.Tensor,
        end_b: torch.Tensor,
        ignore_prim_ids: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> SegmentPairVisibility:
        if self._multi is not None:
            return self._multi.visible_pair(start, end_a, end_b, ignore_prim_ids, active)
        from .geometry import SegmentPairVisibility

        scene = self._require_native_scene()
        values = torch.ops.rayd_torch.visible_pair_forward(scene, start, end_a, end_b, ignore_prim_ids, active)
        return SegmentPairVisibility(int(start.shape[0]), *values)

    def visible_edge(
        self,
        source: torch.Tensor,
        edge_position: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_t_min: torch.Tensor,
        edge_t_max: torch.Tensor,
        sample_fractions: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        active: torch.Tensor | None = None,
    ) -> AxialEdgeVisibility:
        if self._multi is not None:
            return self._multi.visible_edge(
                source, edge_position, edge_direction, edge_t_min, edge_t_max, sample_fractions, active
            )
        from .geometry import AxialEdgeVisibility

        scene = self._require_native_scene()
        values = torch.ops.rayd_torch.visible_edge_forward(
            scene,
            source,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            [float(value) for value in sample_fractions],
            active,
        )
        return AxialEdgeVisibility(int(source.shape[0]), *values)

    def visible_chain(
        self,
        points: torch.Tensor,
        chain_length: torch.Tensor,
        ignore_prim_per_segment: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> SegmentChainVisibility:
        if self._multi is not None:
            return self._multi.visible_chain(points, chain_length, ignore_prim_per_segment, active)
        from .geometry import SegmentChainVisibility

        scene = self._require_native_scene()
        values = torch.ops.rayd_torch.visible_chain_forward(
            scene, points, chain_length, ignore_prim_per_segment, active
        )
        return SegmentChainVisibility(int(points.shape[0]), int(points.shape[1]) - 1, *values)
