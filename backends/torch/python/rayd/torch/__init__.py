from __future__ import annotations

import torch as _torch  # noqa: F401

from . import _stable
from . import _legacy

try:
    from . import _C as _extension
except ImportError:
    _extension = None

# Three native artifacts back this package. `_legacy_ops` is the primary
# dispatcher and owns `torch.ops.rayd_torch` plus `torch.classes.rayd_torch.Scene`;
# `_stable_ops` is the LibTorch Stable ABI slice loaded by `_stable`; `_C` is a
# metadata-only pybind11 module built alongside `_legacy_ops`. `_NATIVE_AVAILABLE`
# is the one dispatcher-availability signal and is what submodules gate native
# calls on. `_C` is not that signal in either direction: it is forced to None
# when the dispatcher did not load, but it is also None when the dispatcher DID
# load (e.g. via RAYD_TORCH_LEGACY_LIBRARY) while the metadata module is absent.
_NATIVE_AVAILABLE = _legacy.AVAILABLE or _legacy.is_registered()
_C = _extension if _NATIVE_AVAILABLE else None
_EXTENSION_IMPORT_ERROR = None if _NATIVE_AVAILABLE else _legacy.LOAD_ERROR

if _NATIVE_AVAILABLE:
    from . import _compile as _compile_support

    _compile_support.register()

from .camera import Camera
from ._capabilities import api_manifest, backend_capabilities
from .mesh import Mesh
from .scene import Scene
from .sdf import SdfGrid, sdf_intersect
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
    SdfIntersection,
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
    "SdfGrid",
    "SdfIntersection",
    "SegmentChainVisibility",
    "SegmentPairVisibility",
    "api_manifest",
    "backend_capabilities",
    "sdf_intersect",
]
