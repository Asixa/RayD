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

def __getattr__(name: str):
    """Resolve `MultiDeviceOptions` without importing the multi-device layer.

    `rayd.torch._multi` is the private orchestration module of the multi-GPU
    plan's Phase 2, and a single-device program must never reach it (D9). It
    holds the one public name that layer has, so that name is bound lazily
    here: importing `rayd.torch` and running single-device ops leaves
    `rayd.torch._multi` unimported, while `rayd.torch.MultiDeviceOptions`
    (and `from rayd.torch import *`, which consults `__all__` through this
    hook) imports it on first use.
    """
    if name == "MultiDeviceOptions":
        from ._multi import MultiDeviceOptions as _MultiDeviceOptions

        globals()["MultiDeviceOptions"] = _MultiDeviceOptions
        return _MultiDeviceOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "MultiDeviceOptions",
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
