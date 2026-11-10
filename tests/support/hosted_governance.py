# Copyright Xingyu Chen.
# Runs every governance and packaging test that needs neither a GPU nor a built RayD backend.

"""Hosted governance suite: `python -m tests.support.hosted_governance`.

The suite is discovered, not listed: every `test_*.py` module at the root of
`tests/` and in `tests/governance/` and `tests/packaging/` runs unless it is
named in `NEEDS_NATIVE_BACKEND` or `KNOWN_FAILURES` below with the reason. A new
governance module therefore runs on the next push without a workflow edit, and
a new module that imports Torch, Dr.Jit, or a built backend fails here until it
is moved under a behaviour directory or added to the exclusions. Modules that
need an optional tool (a C++ compiler, CMake) skip themselves when it is absent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = ("tests", "tests/governance", "tests/packaging")

# Modules that import torch, drjit, or a built rayd backend, directly or in a subprocess.
NEEDS_NATIVE_BACKEND = {
    "tests.test_f3_torch_stable_abi": "probes the built rayd.torch in subprocesses",
    "tests.test_multi_device_coherent_accum": "imports torch",
    "tests.test_multi_device_diffraction_paths": "imports torch",
    "tests.test_multi_device_reflection_accum": "imports torch",
    "tests.test_namespace_isolation": "imports both built backends in subprocesses",
    "tests.packaging.test_no_drjit_import": "imports the built rayd.torch",
    "tests.packaging.test_torch_public_api_contract": "imports the built rayd.torch",
}

# Modules that also fail on an untouched checkout; listed so the hosted run stays meaningful until they are fixed.
KNOWN_FAILURES: dict[str, str] = {}


def discover() -> list[str]:
    names = []
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).glob("test_*.py")):
            names.append(".".join(path.relative_to(ROOT).with_suffix("").parts))
    return names


def main() -> int:
    modules = discover()
    excluded = {**NEEDS_NATIVE_BACKEND, **KNOWN_FAILURES}
    stale = sorted(set(excluded) - set(modules))
    if stale:
        print(f"Excluded modules that no longer exist: {', '.join(stale)}", file=sys.stderr)
        return 1
    for name in sorted(excluded):
        print(f"skipping {name}: {excluded[name]}")
    selected = [name for name in modules if name not in excluded]
    suite = unittest.defaultTestLoader.loadTestsFromNames(selected)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
