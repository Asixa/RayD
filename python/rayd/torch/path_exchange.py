# Copyright Xingyu Chen.
# Exposes the Torch path exchange Python API.

"""Public Torch path-exchange compatibility module."""

from rayd._impl.path_exchange import (
    PathDerivative,
    PathDerivativeField,
    PathDerivativeMode,
    PathInteraction,
    PathInteractionDerivative,
    PathInteractionKind,
    PathProvenance,
    PathRecord,
    diffraction_path_record,
    reflection_path_record,
)

__all__ = [
    "PathInteractionKind",
    "PathProvenance",
    "PathDerivativeMode",
    "PathDerivativeField",
    "PathInteraction",
    "PathDerivative",
    "PathInteractionDerivative",
    "PathRecord",
    "reflection_path_record",
    "diffraction_path_record",
]

# This documented public submodule remains the serialization/introspection
# identity even though its implementation is co-located under rayd._impl.
for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
