from __future__ import annotations

import weakref

import torch

from . import _C
from .autograd import accum_dfr_direct as _legacy_accum_dfr_direct
from .autograd import accum_dfr_direct_native as _accum_dfr_direct_native
from .autograd import intersect as _intersect
from .autograd import nearest_edge as _nearest_edge
from .autograd import nearest_edge_ray as _nearest_edge_ray
from .autograd import trace_refl_epc_field as _trace_refl_epc_field
from .autograd import trace_dfr_paths_order1_native as _trace_dfr_paths_order1_native
from .autograd import trace_reflections as _trace_reflections
from .autograd import visible as _visible
from .mesh import Mesh
from .types import DfrGrid, DfrMaterial, DfrStates, Ray


def _native_scene_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if torch._C._functorch.is_functorch_wrapped_tensor(value) or torch._C._functorch.is_gradtrackingtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    try:
        value.data_ptr()
    except RuntimeError:
        value = value.detach().clone()
    return value


class Scene:
    def __init__(self) -> None:
        self._meshes: list[tuple[Mesh, bool]] = []
        self._native_handle: int | None = None
        self._finalizer: weakref.finalize | None = None
        self._ready = False
        self._pending_updates = False

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects raydtorch.Mesh.")
        if self._native_handle is not None:
            _C.destroy_scene(self._native_handle)
            self._native_handle = None
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
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        specs = [self._mesh_spec(mesh, dynamic) for mesh, dynamic in self._meshes]
        with torch._C._DisableFuncTorch():
            handle = int(_C.create_scene(specs))
        self._native_handle = handle
        self._finalizer = weakref.finalize(self, _C.destroy_scene, handle)
        self._ready = True
        self._pending_updates = False

    def _require_ready(self) -> int:
        if not self._ready or self._native_handle is None:
            raise RuntimeError("Scene is not ready. Call build() before querying.")
        return self._native_handle

    def is_ready(self) -> bool:
        return self._ready

    @property
    def num_meshes(self) -> int:
        handle = self._require_ready()
        return int(_C.scene_num_meshes(handle))

    @property
    def version(self) -> int:
        handle = self._require_ready()
        return int(_C.scene_version(handle))

    def intersect(self, ray: Ray, active=None):
        handle = self._require_ready()
        if active is None:
            active = torch.ones((ray.o.shape[0],), device=ray.o.device, dtype=torch.bool)
        vertices = self._meshes[0][0].vertices
        return _intersect(handle, vertices, ray.o, ray.d, ray.tmax, active.contiguous())

    def nearest_edge(self, point: torch.Tensor | Ray):
        handle = self._require_ready()
        vertices = self._meshes[0][0].vertices
        if isinstance(point, Ray):
            active = torch.ones((point.o.shape[0],), device=point.o.device, dtype=torch.bool)
            return _nearest_edge_ray(
                handle,
                vertices,
                point.o,
                point.d,
                point.tmax,
                active,
            )
        return _nearest_edge(handle, vertices, point.contiguous())

    def visible(self, start: torch.Tensor, end: torch.Tensor, active=None):
        handle = self._require_ready()
        start = start.contiguous()
        end = end.contiguous()
        if active is None:
            active = torch.ones((start.shape[0],), device=start.device, dtype=torch.bool)
        return _visible(handle, start, end, active.contiguous())

    def trace_reflections(self, ray: Ray, max_bounces: int, active=None):
        handle = self._require_ready()
        if active is None:
            active = torch.ones((ray.o.shape[0],), device=ray.o.device, dtype=torch.bool)
        vertices = self._meshes[0][0].vertices
        return _trace_reflections(
            handle,
            vertices,
            ray.o,
            ray.d,
            ray.tmax,
            active.contiguous(),
            int(max_bounces),
        )

    def trace_refl_epc_field(self, source: torch.Tensor, receiver: torch.Tensor, max_bounces: int, active=None):
        handle = self._require_ready()
        source = source.contiguous()
        receiver = receiver.contiguous()
        if active is None:
            active = torch.ones((source.shape[0],), device=source.device, dtype=torch.bool)
        vertices = self._meshes[0][0].vertices
        return _trace_refl_epc_field(
            handle,
            vertices,
            source,
            receiver,
            active.contiguous(),
            int(max_bounces),
        )

    def _default_dfr_material(self, *, device: torch.device, dtype: torch.dtype) -> DfrMaterial:
        face_count = int(self._meshes[0][0].faces.shape[0])
        return DfrMaterial.default(face_count, device=device, dtype=dtype)

    def trace_dfr_paths(
        self,
        *,
        tx_positions: torch.Tensor,
        rx_positions: torch.Tensor,
        states: DfrStates,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        max_paths: int | None = None,
        wavelength: float = 1.0,
    ):
        handle = self._require_ready()
        if material is None:
            material = self._default_dfr_material(device=states.edge_pos.device, dtype=states.edge_pos.dtype)
        if active is None:
            active = torch.ones((states.state_count,), device=states.edge_pos.device, dtype=torch.bool)
        if max_paths is None:
            max_paths = states.state_count
        return _trace_dfr_paths_order1_native(
            handle,
            tx_positions,
            rx_positions,
            states,
            material,
            active=active,
            max_paths=int(max_paths),
            wavelength=float(wavelength),
        )

    def accum_dfr_legacy_direct(self, *, edge_pos: torch.Tensor, edge_dir: torch.Tensor, src: torch.Tensor):
        self._require_ready()
        return _legacy_accum_dfr_direct(edge_pos.contiguous(), edge_dir.contiguous(), src.contiguous())

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
        edge_pos: torch.Tensor | None = None,
        edge_dir: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
    ):
        handle = self._require_ready()
        if states is None:
            if edge_pos is not None and edge_dir is not None and src is not None:
                return self.accum_dfr_legacy_direct(edge_pos=edge_pos, edge_dir=edge_dir, src=src)
            raise TypeError("Scene.accum_dfr_direct() requires RayD-style DfrStates and DfrGrid.")
        if grid is None:
            raise TypeError("Scene.accum_dfr_direct() requires DfrGrid when states are provided.")
        if material is None:
            material = self._default_dfr_material(device=states.edge_pos.device, dtype=states.edge_pos.dtype)
        if active is None:
            active = torch.ones((states.state_count,), device=states.edge_pos.device, dtype=torch.bool)
        return _accum_dfr_direct_native(
            handle,
            states,
            grid,
            material,
            active=active,
            wavelength=float(wavelength),
            direct_samples=int(direct_samples),
            keller_samples=int(keller_samples),
        )

    def accum_dfr(self, **kwargs):
        return self.accum_dfr_direct(**kwargs)

    def update_mesh_vertices(self, mesh_id: int, positions):
        handle = self._require_ready()
        mesh, dynamic = self._meshes[mesh_id]
        if not dynamic:
            raise RuntimeError("Scene.update_mesh_vertices(): target mesh is not dynamic.")
        mesh.vertices = positions.contiguous()
        with torch._C._DisableFuncTorch():
            _C.update_mesh_vertices(handle, int(mesh_id), _native_scene_tensor(mesh.vertices))
        self._pending_updates = True

    def sync(self) -> None:
        handle = self._require_ready()
        with torch._C._DisableFuncTorch():
            _C.sync_scene(handle)
        self._pending_updates = False

    def has_pending_updates(self) -> bool:
        return bool(self._pending_updates)
