# Copyright Xingyu Chen.
# Exposes the Dr.Jit path exchange Python API.

from rayd._impl.path_exchange_jit import *  # noqa: F401,F403

# Keep the documented public module identity stable for pickle, repr, and
# downstream introspection while the implementation lives in the private
# PEP 420 subtree.
for _public_name in (
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
):
    globals()[_public_name].__module__ = __name__
del _public_name