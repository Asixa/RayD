from __future__ import annotations

# ---------- drjit 1.3.x interop monkey-patch ----------
# drjit.interop._flatten/_unflatten shadow the *desc* list parameter with the
# DRJIT_STRUCT dict, breaking recursive traversal of struct fields.
# Fix: use a separate local for the struct descriptor.
import drjit.interop as _interop  # noqa: E402

def _flatten_fixed(a, flat, desc, /):
    tp = type(a)
    desc.append(tp)
    if tp is list or tp is tuple:
        desc.append(len(a))
        for v in a:
            _flatten_fixed(v, flat, desc)
    elif tp is dict:
        desc.append(tuple(a.keys()))
        for v in a.values():
            _flatten_fixed(v, flat, desc)
    else:
        sd = getattr(tp, 'DRJIT_STRUCT', None)
        if type(sd) is dict:
            for k in sd:
                _flatten_fixed(getattr(a, k), flat, desc)
        else:
            flat.append(a)

def _unflatten_fixed(flat, desc, /):
    tp = desc.pop()
    if tp is list or tp is tuple:
        n = desc.pop()
        return tp(_unflatten_fixed(flat, desc) for _ in range(n))
    elif tp is dict:
        keys = desc.pop()
        return {k: _unflatten_fixed(flat, desc) for k in keys}
    else:
        sd = getattr(tp, 'DRJIT_STRUCT', None)
        if type(sd) is dict:
            result = tp()
            for k in sd:
                setattr(result, k, _unflatten_fixed(flat, desc))
            return result
        else:
            return flat.pop()

# Only patch if the bug is present (desc parameter gets shadowed)
if hasattr(_interop, '_flatten'):
    _interop._flatten = _flatten_fixed
    _interop._unflatten = _unflatten_fixed
# ---------- end monkey-patch ----------

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
    SceneCommitProfile,
    SecondaryEdgeInfo,
)
from .mesh import Mesh
from .scene import Scene
from .camera import Camera

__all__ = [
    "Camera",
    "Intersection",
    "Mesh",
    "NearestPointEdge",
    "NearestRayEdge",
    "PrimaryEdgeSample",
    "Ray",
    "Scene",
    "SceneCommitProfile",
    "SecondaryEdgeInfo",
]
