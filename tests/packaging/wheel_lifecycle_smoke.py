# Copyright Xingyu Chen.
# Validates an installed backend wheel and its removal from an isolated interpreter.

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sysconfig
from pathlib import Path


_DISTRIBUTIONS = {"drjit": "rayd-drjit", "torch": "rayd-torch"}


def _site_roots() -> tuple[Path, ...]:
    roots = {
        Path(sysconfig.get_path(name)).resolve()
        for name in ("platlib", "purelib")
        if sysconfig.get_path(name) is not None
    }
    return tuple(sorted(roots))


def _require_installed_path(path: str | None, label: str) -> Path:
    if path is None:
        raise AssertionError(f"{label} has no filesystem origin")
    resolved = Path(path).resolve()
    if not any(resolved.is_relative_to(root) for root in _site_roots()):
        raise AssertionError(f"{label} loaded outside the isolated environment: {resolved}")
    return resolved


def _probe_installed(backend: str) -> None:
    module = importlib.import_module(f"rayd.{backend}")
    module_path = _require_installed_path(module.__file__, f"rayd.{backend}")
    capabilities = module.backend_capabilities()
    if capabilities["backend"] != backend:
        raise AssertionError(capabilities)

    native = importlib.import_module(f"rayd.{backend}._C")
    native_path = _require_installed_path(native.__file__, f"rayd.{backend}._C")
    if backend == "torch":
        import torch

        if not module._NATIVE_AVAILABLE:
            extension_error = module._EXTENSION_IMPORT_ERROR
            raise AssertionError(f"RayD Torch native runtime is unavailable: {extension_error!r}") from extension_error
        if not hasattr(torch.ops.rayd_torch, "intersect_forward_t"):
            raise AssertionError("RayD Torch legacy native operators were not registered")
        if not hasattr(torch.ops.rayd_torch_stable, "intersection_valid"):
            raise AssertionError("RayD Torch Stable ABI operators were not registered")
    else:
        if not hasattr(module, "Scene"):
            raise AssertionError("RayD Dr.Jit native Scene binding is unavailable")
        for options_name in ("SdfTraceOptions", "SurfelTraceOptions", "SurfelRenderOptions"):
            getattr(module, options_name)()

    print(f"installed {backend} module: {module_path}")
    print(f"installed {backend} native extension: {native_path}")


def _probe_absent(backend: str) -> None:
    name = f"rayd.{backend}"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        raise AssertionError(f"{name} still has an import spec after uninstall")
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        print(f"confirmed absent after uninstall: {name}")
    else:
        raise AssertionError(f"{name} survived {_DISTRIBUTIONS[backend]} uninstall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("installed", "absent"))
    parser.add_argument("backend", choices=tuple(_DISTRIBUTIONS))
    args = parser.parse_args()
    if args.phase == "installed":
        _probe_installed(args.backend)
    else:
        _probe_absent(args.backend)


if __name__ == "__main__":
    main()
