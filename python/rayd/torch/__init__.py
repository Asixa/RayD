# Copyright Xingyu Chen.
# Exposes the public Torch Python package.

from __future__ import annotations

from typing import TYPE_CHECKING

from rayd._impl import runtime as _runtime

_NATIVE_AVAILABLE = _runtime.NATIVE_AVAILABLE
_C = _runtime.C
_EXTENSION_IMPORT_ERROR = _runtime.EXTENSION_IMPORT_ERROR

from rayd._impl.camera import Camera
from rayd._impl.capabilities import api_manifest, backend_capabilities
from rayd._impl.geometry import (
    AxialEdgeVisibility,
    DfrAccum,
    DfrCoherentAccum,
    DfrGrid,
    DfrMaterial,
    DfrPaths,
    DfrStates,
    Intersection,
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
from rayd._impl.scene import Mesh
from rayd._impl.scene import Scene
from rayd._impl.sdf import SdfGrid, sdf_intersect
from rayd._impl.surfel import SurfelCloud, SurfelComposite, SurfelIntersection, SurfelScene, SurfelTraceOptions

if TYPE_CHECKING:
    from rayd._impl.multi import MultiDeviceOptions as MultiDeviceOptions

if not TYPE_CHECKING:

    def __getattr__(name: str):
        if name == "MultiDeviceOptions":
            from rayd._impl.multi import MultiDeviceOptions as implementation

            globals()["MultiDeviceOptions"] = implementation
            return implementation
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
    "SurfelCloud",
    "SurfelComposite",
    "SurfelIntersection",
    "SurfelScene",
    "SurfelTraceOptions",
    "SegmentChainVisibility",
    "SegmentPairVisibility",
    "api_manifest",
    "backend_capabilities",
    "sdf_intersect",
]
