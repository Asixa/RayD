import os as _os
import pathlib as _pathlib
import sys as _sys

import drjit as _drjit

if _sys.platform == "win32" and hasattr(_os, "add_dll_directory"):
    _os.add_dll_directory(str(_pathlib.Path(_drjit.__file__).resolve().parent))

from ._C import *  # noqa: F401,F403


def backend_capabilities():
    return {
        "backend": "drjit",
        "intersect": True,
        "nearest_edge_point": True,
        "nearest_edge_ray": True,
        "nearest_edges_topk": True,
        "visibility": True,
        "visibility_pair": True,
        "reflection_trace": True,
        "reflection_accumulation": True,
        "diffraction_direct": True,
        "diffraction_chain": True,
        "surfel": True,
        "reverse_ad": True,
        "forward_ad": True,
        "torch_compile": False,
    }
