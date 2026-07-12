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
from ._capabilities import api_manifest, backend_capabilities
from .mesh import Mesh
from .scene import Scene
from .types import (
    DfrAccum,
    DfrCoherentAccum,
    DfrGrid,
    DfrMaterial,
    DfrPaths,
    DfrStates,
    Intersection,
    AxialEdgeVisibility,
    NearestEdgesTopK,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    RayFlags,
    ReflEpcField,
    ReflectionChain,
    SceneGlobalGeometry,
    SegmentChainVisibility,
    SegmentPairVisibility,
)
__all__ = [
    "DfrAccum",
    "DfrCoherentAccum",
    "DfrGrid",
    "DfrMaterial",
    "DfrPaths",
    "DfrStates",
    "Camera",
    "Intersection",
    "AxialEdgeVisibility",
    "Mesh",
    "NearestPointEdge",
    "NearestEdgesTopK",
    "NearestRayEdge",
    "Ray",
    "RayFlags",
    "ReflEpcField",
    "ReflectionChain",
    "Scene",
    "SceneGlobalGeometry",
    "SegmentChainVisibility",
    "SegmentPairVisibility",
    "api_manifest",
    "backend_capabilities",
]
