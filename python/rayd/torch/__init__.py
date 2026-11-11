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
    AccumGrid,
    AccumOptions,
    AccumResult,
    AccumStrategy,
    AxialEdgeVisibility,
    DfrAccum,
    DfrCoherentAccum,
    DfrCoherentOptions,
    DfrGrid,
    DfrMaterial,
    DfrOptions,
    DfrPathLayout,
    DfrPathOptions,
    DfrPaths,
    DfrStates,
    EdgeBVHBackend,
    Intersection,
    NearestEdgesTopK,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    RayFlags,
    ReflEpc,
    ReflEpcField,
    ReflEpcOptions,
    ReflMaterial,
    ReflectionChain,
    SceneGeometry,
    SdfIntersection,
    SegmentChainVisibility,
    SegmentPairVisibility,
    TraceBackend,
    WedgeEvents,
)
from rayd._impl.mixed import MixedScene
from rayd._impl.scene import Mesh
from rayd._impl.scene import Scene
from rayd._impl.sdf import SdfGrid, SdfGridBatch, SdfTraceOptions, sdf_intersect
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
    "AccumGrid",
    "AccumOptions",
    "AccumResult",
    "AccumStrategy",
    "DfrAccum",
    "DfrCoherentAccum",
    "DfrCoherentOptions",
    "DfrGrid",
    "DfrMaterial",
    "DfrOptions",
    "DfrPathLayout",
    "DfrPathOptions",
    "DfrPaths",
    "DfrStates",
    "Camera",
    "EdgeBVHBackend",
    "Intersection",
    "AxialEdgeVisibility",
    "Mesh",
    "MixedScene",
    "MultiDeviceOptions",
    "NearestPointEdge",
    "NearestEdgesTopK",
    "NearestRayEdge",
    "Ray",
    "RayFlags",
    "ReflEpc",
    "ReflEpcField",
    "ReflEpcOptions",
    "ReflMaterial",
    "ReflectionChain",
    "Scene",
    "SceneGeometry",
    "SdfGrid",
    "SdfGridBatch",
    "SdfIntersection",
    "SdfTraceOptions",
    "SurfelCloud",
    "SurfelComposite",
    "SurfelIntersection",
    "SurfelScene",
    "SurfelTraceOptions",
    "SegmentChainVisibility",
    "SegmentPairVisibility",
    "TraceBackend",
    "WedgeEvents",
    "api_manifest",
    "backend_capabilities",
    "sdf_intersect",
]
