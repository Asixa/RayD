from __future__ import annotations

import torch as _torch  # noqa: F401

from . import _stable
from . import _legacy
from . import _compat

try:
    from . import _C as _compat_extension
except ImportError as exc:
    _compat_extension = None
    _COMPAT_IMPORT_ERROR = exc
else:
    _COMPAT_IMPORT_ERROR = None

# During migration, older combined `_C` builds may register the legacy
# dispatcher as an import side effect. New builds load `_legacy_ops` first and
# keep `_C` as metadata-only compatibility surface.
_NATIVE_AVAILABLE = _legacy.AVAILABLE or _legacy.is_registered()
_C = (_compat_extension or _compat) if _NATIVE_AVAILABLE else None
_EXTENSION_IMPORT_ERROR = None if _NATIVE_AVAILABLE else _legacy.LOAD_ERROR

if _NATIVE_AVAILABLE:
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
