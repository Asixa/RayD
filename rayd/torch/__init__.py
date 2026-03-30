from __future__ import annotations

from ._interop_patch import install_drjit_interop_patch


install_drjit_interop_patch()

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised in subprocess tests
    raise ImportError(
        "rayd.torch requires the optional 'torch' dependency. "
        "Install it with `pip install rayd[torch]` or `pip install torch`."
    ) from exc

# Public API re-exports
from .types import (
    Intersection,
    NearestPointEdge,
    NearestRayEdge,
    PrimaryEdgeSample,
    Ray,
    SceneSyncProfile,
    SceneEdgeInfo,
    SceneEdgeTopology,
    SecondaryEdgeInfo,
)
from .mesh import Mesh
from .scene import Scene
from .camera import Camera
from ._env import _native as _native

device_count = _native.device_count
current_device = _native.current_device
set_device = _native.set_device

__all__ = [
    "Camera",
    "current_device",
    "device_count",
    "Intersection",
    "Mesh",
    "NearestPointEdge",
    "NearestRayEdge",
    "PrimaryEdgeSample",
    "Ray",
    "Scene",
    "SceneSyncProfile",
    "SceneEdgeInfo",
    "SceneEdgeTopology",
    "SecondaryEdgeInfo",
    "set_device",
]
