from __future__ import annotations

import torch as _torch  # noqa: F401

from . import _stable

try:
    from . import _C
except ImportError as exc:
    _C = None
    _EXTENSION_IMPORT_ERROR = exc
else:
    _EXTENSION_IMPORT_ERROR = None

if _C is not None:
    from . import _compile as _compile_support

    _compile_support.register()

from .camera import Camera
from .mesh import Mesh
from .scene import Scene
from .types import (
    DfrAccum,
    DfrGrid,
    DfrMaterial,
    DfrPaths,
    DfrStates,
    Intersection,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    RayFlags,
    ReflEpcField,
    ReflectionChain,
    SceneGlobalGeometry,
)


def backend_capabilities():
    return {
        "backend": "torch",
        "intersect": True,
        "nearest_edge_point": True,
        "nearest_edge_ray": True,
        "nearest_edges_topk": False,
        "visibility": True,
        "visibility_pair": False,
        "reflection_trace": True,
        "reflection_accumulation": True,
        "diffraction_direct": True,
        "diffraction_chain": True,
        "surfel": False,
        "reverse_ad": True,
        "forward_ad": True,
        "torch_compile": True,
    }

__all__ = [
    "DfrAccum",
    "DfrGrid",
    "DfrMaterial",
    "DfrPaths",
    "DfrStates",
    "Camera",
    "Intersection",
    "Mesh",
    "NearestPointEdge",
    "NearestRayEdge",
    "Ray",
    "RayFlags",
    "ReflEpcField",
    "ReflectionChain",
    "Scene",
    "SceneGlobalGeometry",
    "backend_capabilities",
]
