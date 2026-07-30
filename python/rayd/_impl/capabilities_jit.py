# Copyright Xingyu Chen.
# Implements shared Python support for capabilities.

from __future__ import annotations

from copy import deepcopy
from typing import Any

_SCHEMA_VERSION = 3
_SCHEMA_SHA256 = "a9f179cf24656ef33f5cda4725f1f8609337d5a8238649fba2b735ca7d86dcf9"
_BACKEND = "drjit"
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
    "surfel": True,
    "sdf_intersect": True,
    "mixed_scene": True,
    "reverse_ad": True,
    "forward_ad": True,
    "torch_compile": False,
    "multi_device_replicated": False,
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
    "sdf_intersect": ("core", "provisional"),
    "mixed_scene": ("core", "provisional"),
    "reverse_ad": ("core", "stable"),
    "forward_ad": ("core", "provisional"),
    "torch_compile": ("experimental", "provisional"),
    "multi_device_replicated": ("core", "provisional"),
}

_ALIASES = {
    "edge_bvh_backend": {
        "hybrid": {
            "canonical": "optix_drjit",
            "stability": "deprecated",
            "summary": "Compatibility alias only; unrelated to the removed HLBVH experiment.",
        },
        "mixed": {"canonical": "optix_drjit", "stability": "deprecated", "summary": "Legacy compatibility alias."},
        "optix_ray": {"canonical": "optix_drjit", "stability": "deprecated", "summary": "Legacy compatibility alias."},
        "ray_optix": {"canonical": "optix_drjit", "stability": "deprecated", "summary": "Legacy compatibility alias."},
    }
}

_DERIVATIVE_STATUSES = {
    "supported": "The backend implements and tests this derivative for the named variant and input domain.",
    "unsupported": "The derivative has meaningful semantics but is unavailable or not validated for this backend path.",
    "not_applicable": "The named input domain or variant has no derivative semantics.",
}

_SUPPORTED = {"drjit": ("supported", "supported"), "torch": ("supported", "supported")}
_UNSUPPORTED = {"drjit": ("unsupported", "unsupported"), "torch": ("unsupported", "unsupported")}
_NOT_APPLICABLE = {"drjit": ("not_applicable", "not_applicable"), "torch": ("not_applicable", "not_applicable")}
_DRJIT_ONLY = {"drjit": ("supported", "supported"), "torch": ("unsupported", "unsupported")}
_DRJIT_SUPPORTED_TORCH_NA = {"drjit": ("supported", "supported"), "torch": ("not_applicable", "not_applicable")}

_DERIVATIVE_CONTRACT = {
    "intersect": {"intersect": {"geometry": _SUPPORTED, "ray": _SUPPORTED}},
    "nearest_edge_point": {"nearest_edge": {"geometry": _SUPPORTED, "point": _SUPPORTED}},
    "nearest_edge_ray": {"nearest_edge": {"geometry": _SUPPORTED, "ray": _SUPPORTED}},
    "nearest_edges_topk": {"nearest_edges": {"geometry": _SUPPORTED, "point": _SUPPORTED}},
    "visibility": {
        "visible": {
            "domains": {"continuous_inputs": _NOT_APPLICABLE},
            "note": "Visibility is a discrete boolean result.",
        }
    },
    "visibility_pair": {
        "visible_pair": {
            "domains": {"continuous_inputs": _NOT_APPLICABLE},
            "note": "Visibility is a discrete boolean result.",
        }
    },
    "visibility_edge": {
        "visible_edge": {
            "domains": {"continuous_inputs": _NOT_APPLICABLE},
            "note": "Visibility is a discrete boolean result.",
        }
    },
    "visibility_chain": {
        "visible_chain": {
            "domains": {"continuous_inputs": _NOT_APPLICABLE},
            "note": "Visibility and blocker identifiers are discrete results.",
        }
    },
    "reflection_trace": {
        "trace_reflections": {"geometry": _SUPPORTED, "ray": _SUPPORTED},
        "trace_refl_epc": {
            "domains": {
                "geometry": _UNSUPPORTED,
                "ray": _UNSUPPORTED,
                "receiver": _UNSUPPORTED,
                "plane_geometry": _UNSUPPORTED,
            },
            "backend_notes": {
                "drjit": "The native path-discovery overload explicitly rejects AD inputs.",
                "torch": "Multi-bounce EPC receiver derivatives are not cross-backend validated.",
            },
        },
        "trace_refl_epc_field": {
            "domains": {
                "geometry": _DRJIT_ONLY,
                "source": _DRJIT_ONLY,
                "receiver": _DRJIT_ONLY,
                "material": _DRJIT_SUPPORTED_TORCH_NA,
            },
            "backend_notes": {
                "torch": "The public Torch EPC-field path is forward-only because its native derivatives do not "
                "differentiate the physical Fresnel and polarization forward computation."
            },
        },
    },
    "reflection_accumulation": {
        "accumulate_reflections": {
            "domains": {"geometry": _DRJIT_ONLY, "source_receiver": _DRJIT_ONLY, "material": _DRJIT_ONLY},
            "backend_notes": {"torch": "The Torch high-level accumulation path is forward-only."},
        }
    },
    "diffraction_direct": {
        "trace_dfr_paths": {
            "domains": {"continuous_inputs": _DRJIT_ONLY},
            "backend_notes": {"torch": "Torch diffraction path export is forward-only."},
        },
        "accum_dfr_direct": {"geometry": _SUPPORTED, "material": _SUPPORTED},
        "accum_dfr_coherent_direct": {
            "domains": {"continuous_inputs": _UNSUPPORTED},
            "note": "Coherent diffraction accumulation is forward-only.",
        },
    },
    "diffraction_chain": {"accum_dfr": {"geometry": _SUPPORTED, "material": _SUPPORTED}},
    "sdf_intersect": {"intersect": {"sdf_values": _SUPPORTED, "placement": _SUPPORTED, "ray": _SUPPORTED}},
    "mixed_scene": {
        "intersect": {"selected_geometry": _SUPPORTED, "ray": _SUPPORTED},
        "transmittance": {"surfel_fields": _SUPPORTED, "ray": _SUPPORTED},
    },
}

_TRACE = {
    "backends": {
        "optix": {
            "stability": "stable",
            "summary": "OptiX GPU ray-tracing backend for triangle intersection, edge, visibility, reflection, and diffraction queries shared by both frontends.",
        },
        "cuda": {
            "stability": "provisional",
            "summary": "Pure-CUDA scene-level triangle BVH backend for eager closest-hit and occlusion queries; no OptiX driver required. The P4 fused executor also serves the full multipath surface (visibility, reflection trace and accumulation, diffraction paths and accumulation, and EPC) with no OptiX driver.",
        },
    },
    "integration_modes": ["jit_symbolic", "eager_native"],
    "frontend_support": {
        "drjit": {"optix": ["jit_symbolic", "eager_native"], "cuda": ["eager_native"]},
        "torch": {"optix": ["eager_native"], "cuda": ["eager_native"]},
    },
}


def _backend_derivatives() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for operation, variants in _DERIVATIVE_CONTRACT.items():
        resolved_variants = {}
        for variant, metadata in variants.items():
            domains = metadata.get("domains", metadata)
            resolved = {
                "primal": True,
                "input_domains": {
                    domain: {"vjp": statuses[_BACKEND][0], "jvp": statuses[_BACKEND][1]}
                    for domain, statuses in domains.items()
                },
            }
            note = metadata.get("backend_notes", {}).get(_BACKEND, metadata.get("note"))
            if note is not None:
                resolved["note"] = note
            resolved_variants[variant] = resolved
        result[operation] = resolved_variants
    return result


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
        "derivative_statuses": deepcopy(_DERIVATIVE_STATUSES),
        "derivatives": _backend_derivatives(),
        "capabilities": backend_capabilities(),
        "apis": {
            name: {"category": category, "stability": stability}
            for name, (category, stability) in _API_CLASSIFICATION.items()
        },
        "aliases": deepcopy(_ALIASES),
        "trace": deepcopy(_TRACE),
    }
