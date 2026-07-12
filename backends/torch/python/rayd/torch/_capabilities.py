from __future__ import annotations

from copy import deepcopy
from typing import Any

_SCHEMA_VERSION = 1
_SCHEMA_SHA256 = "5fb52cb0da6211cce30f572a5a3238743b432e527742e41bb6327dae6067cc1f"
_BACKEND = "torch"
_TYPING = "complete"

_NAMING_CONVENTIONS = {
    "options": "New configuration records use PascalCase <Operation>Options; existing domain records retain compatibility names until a major release.",
    "results": "Result records use PascalCase semantic nouns; plural names denote collections and AD does not change the public result type name.",
    "fields": "Fields use snake_case; identifiers end in _id or _ids and ambiguous identifier spaces carry explicit local_ or global_ prefixes.",
}

_CAPABILITIES = {
    "intersect": True,
    "nearest_edge_point": True,
    "nearest_edge_ray": True,
    "nearest_edges_topk": True,
    "edge_mask": True,
    "visibility": True,
    "visibility_pair": True,
    "visibility_edge": True,
    "visibility_chain": True,
    "reflection_trace": True,
    "reflection_accumulation": True,
    "diffraction_direct": True,
    "diffraction_chain": True,
    "surfel": False,
    "reverse_ad": True,
    "forward_ad": True,
    "torch_compile": True,
}

_API_CLASSIFICATION = {
    "intersect": ("core", "stable"),
    "nearest_edge_point": ("core", "stable"),
    "nearest_edge_ray": ("core", "stable"),
    "nearest_edges_topk": ("core", "provisional"),
    "edge_mask": ("core", "provisional"),
    "visibility": ("core", "stable"),
    "visibility_pair": ("multipath", "provisional"),
    "visibility_edge": ("multipath", "provisional"),
    "visibility_chain": ("multipath", "provisional"),
    "reflection_trace": ("multipath", "stable"),
    "reflection_accumulation": ("multipath", "provisional"),
    "diffraction_direct": ("multipath", "provisional"),
    "diffraction_chain": ("multipath", "experimental"),
    "surfel": ("surfel", "experimental"),
    "reverse_ad": ("core", "stable"),
    "forward_ad": ("core", "provisional"),
    "torch_compile": ("experimental", "provisional"),
}

_ALIASES = {
    "edge_bvh_backend": {
        "hybrid": {
            "canonical": "optix_drjit",
            "stability": "deprecated",
            "summary": "Compatibility alias only; unrelated to the removed HLBVH experiment.",
        },
        "mixed": {
            "canonical": "optix_drjit",
            "stability": "deprecated",
            "summary": "Legacy compatibility alias.",
        },
        "optix_ray": {
            "canonical": "optix_drjit",
            "stability": "deprecated",
            "summary": "Legacy compatibility alias.",
        },
        "ray_optix": {
            "canonical": "optix_drjit",
            "stability": "deprecated",
            "summary": "Legacy compatibility alias.",
        },
    }
}


def backend_capabilities() -> dict[str, bool | str]:
    """Return the backward-compatible flat backend capability mapping."""
    return {"backend": _BACKEND, **_CAPABILITIES}


def api_manifest() -> dict[str, Any]:
    """Return classified public API, stability, alias, and typing metadata."""
    return {
        "version": _SCHEMA_VERSION,
        "schema_sha256": _SCHEMA_SHA256,
        "backend": _BACKEND,
        "typing": _TYPING,
        "naming_conventions": deepcopy(_NAMING_CONVENTIONS),
        "capabilities": backend_capabilities(),
        "apis": {
            name: {"category": category, "stability": stability}
            for name, (category, stability) in _API_CLASSIFICATION.items()
        },
        "aliases": deepcopy(_ALIASES),
    }
