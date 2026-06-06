from __future__ import annotations

import weakref

from . import _C
from .mesh import Mesh
from .types import Ray


class Scene:
    def __init__(self) -> None:
        self._meshes: list[tuple[Mesh, bool]] = []
        self._native_handle: int | None = None
        self._finalizer: weakref.finalize | None = None
        self._ready = False

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects raydtorch.Mesh.")
        if self._native_handle is not None:
            _C.destroy_scene(self._native_handle)
            self._native_handle = None
        self._meshes.append((mesh, bool(dynamic)))
        self._ready = False
        return len(self._meshes) - 1

    def _mesh_spec(self, mesh: Mesh, dynamic: bool) -> dict[str, object]:
        return {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "uv": mesh.uv,
            "face_uv": mesh.face_uv,
            "to_world_left": mesh.to_world_left,
            "to_world_right": mesh.to_world_right,
            "use_face_normals": mesh.use_face_normals,
            "edges_enabled": mesh.edges_enabled,
            "dynamic": dynamic,
        }

    def build(self) -> None:
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        specs = [self._mesh_spec(mesh, dynamic) for mesh, dynamic in self._meshes]
        handle = int(_C.create_scene(specs))
        self._native_handle = handle
        self._finalizer = weakref.finalize(self, _C.destroy_scene, handle)
        self._ready = True

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

    def intersect(self, ray: Ray):
        self._require_ready()
        raise RuntimeError("Scene.intersect(): native intersect op is not implemented in this milestone.")
