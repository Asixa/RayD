from __future__ import annotations

from typing import Any

from .types import _StructRepr


class _MeshState(_StructRepr):
    DRJIT_STRUCT = {
        "vertex_positions": object,
        "face_indices": object,
        "vertex_uv": object,
        "face_uv_indices": object,
        "to_world": object,
        "to_world_left": object,
        "to_world_right": object,
        "use_face_normals": object,
        "edges_enabled": object,
        "verbose": object,
    }

    def __init__(
        self,
        vertex_positions: Any = None,
        face_indices: Any = None,
        vertex_uv: Any = None,
        face_uv_indices: Any = None,
        to_world: Any = None,
        to_world_left: Any = None,
        to_world_right: Any = None,
        use_face_normals: bool = False,
        edges_enabled: bool = True,
        verbose: bool = False,
    ):
        self.vertex_positions = vertex_positions
        self.face_indices = face_indices
        self.vertex_uv = vertex_uv
        self.face_uv_indices = face_uv_indices
        self.to_world = to_world
        self.to_world_left = to_world_left
        self.to_world_right = to_world_right
        self.use_face_normals = use_face_normals
        self.edges_enabled = edges_enabled
        self.verbose = verbose

    def copy(self) -> "_MeshState":
        return _MeshState(
            vertex_positions=self.vertex_positions,
            face_indices=self.face_indices,
            vertex_uv=self.vertex_uv,
            face_uv_indices=self.face_uv_indices,
            to_world=self.to_world,
            to_world_left=self.to_world_left,
            to_world_right=self.to_world_right,
            use_face_normals=self.use_face_normals,
            edges_enabled=self.edges_enabled,
            verbose=self.verbose,
        )


class _CameraState(_StructRepr):
    DRJIT_STRUCT = {
        "mode": object,
        "fov_x": object,
        "fx": object,
        "fy": object,
        "cx": object,
        "cy": object,
        "near_clip": object,
        "far_clip": object,
        "width": object,
        "height": object,
        "cache": object,
        "to_world": object,
        "to_world_left": object,
        "to_world_right": object,
    }

    def __init__(
        self,
        mode: str = "fov",
        fov_x: float = 45.0,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
        near_clip: float = 1e-4,
        far_clip: float = 1e4,
        width: int = 1,
        height: int = 1,
        cache: bool = True,
        to_world: Any = None,
        to_world_left: Any = None,
        to_world_right: Any = None,
    ):
        self.mode = mode
        self.fov_x = fov_x
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.width = width
        self.height = height
        self.cache = cache
        self.to_world = to_world
        self.to_world_left = to_world_left
        self.to_world_right = to_world_right
