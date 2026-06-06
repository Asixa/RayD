from __future__ import annotations

from .mesh import Mesh


class Scene:
    def __init__(self) -> None:
        self._meshes: list[tuple[Mesh, bool]] = []
        self._native_handle: int | None = None
        self._ready = False

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects raydtorch.Mesh.")
        self._meshes.append((mesh, bool(dynamic)))
        self._ready = False
        return len(self._meshes) - 1

    def build(self) -> None:
        raise RuntimeError("RayDTorch extension is not built yet.")
