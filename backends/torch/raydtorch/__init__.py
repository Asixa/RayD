from __future__ import annotations

import torch as _torch  # noqa: F401

try:
    from . import _raydtorch as _C
except ImportError as exc:
    _C = None
    _EXTENSION_IMPORT_ERROR = exc
else:
    _EXTENSION_IMPORT_ERROR = None

from .mesh import Mesh
from .scene import Scene
from .types import (
    DfrDirectAccum,
    Intersection,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    ReflEpcField,
    ReflectionChain,
    SceneGlobalGeometry,
)

__all__ = [
    "DfrDirectAccum",
    "Intersection",
    "Mesh",
    "NearestPointEdge",
    "NearestRayEdge",
    "Ray",
    "ReflEpcField",
    "ReflectionChain",
    "Scene",
    "SceneGlobalGeometry",
]
