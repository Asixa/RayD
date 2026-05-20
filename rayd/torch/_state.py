from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any

from .types import _register_drjit_struct


@_register_drjit_struct
@dataclass
class _MeshState:
    vertex_positions: Any = None
    face_indices: Any = None
    vertex_uv: Any = None
    face_uv_indices: Any = None
    to_world: Any = None
    to_world_left: Any = None
    to_world_right: Any = None
    use_face_normals: bool = False
    edges_enabled: bool = True
    verbose: bool = False

    def copy(self) -> "_MeshState":
        return dataclasses.replace(self)


@_register_drjit_struct
@dataclass
class _CameraState:
    mode: str = "fov"
    fov_x: float = 45.0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    near_clip: float = 1e-4
    far_clip: float = 1e4
    width: int = 1
    height: int = 1
    cache: bool = True
    to_world: Any = None
    to_world_left: Any = None
    to_world_right: Any = None
