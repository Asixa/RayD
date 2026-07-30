# Copyright Xingyu Chen.
# Fails single-GPU acceptance before skip-guarded tests can pass vacuously.

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


ROOT = Path(__file__).resolve().parents[2]


def _device_preflight() -> None:
    import drjit as dr
    import drjit.cuda as cuda
    import torch

    if not torch.cuda.is_available():
        raise AssertionError("torch.cuda.is_available() is false on the CUDA acceptance runner")
    count = torch.cuda.device_count()
    if count < 1:
        raise AssertionError(f"CUDA acceptance requires at least one GPU, found {count}")
    print(f"torch {torch.__version__}, {count} visible CUDA device(s)")
    for index in range(count):
        print(
            f"  cuda:{index} {torch.cuda.get_device_name(index)}, capability {torch.cuda.get_device_capability(index)}"
        )

    value = cuda.Float([1.0])
    dr.eval(value)
    if float(value[0]) != 1.0:
        raise AssertionError("Dr.Jit CUDA evaluation returned the wrong value")
    print(f"drjit {dr.__version__}, CUDA evaluation succeeded")


def _origin(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    if path is None:
        raise AssertionError(f"{module_name} has no filesystem origin")
    resolved = Path(path).resolve()
    print(f"{module_name}: {resolved}")
    return resolved


def _require_under(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise AssertionError(f"{label} resolved outside this checkout: {path} is not under {root}")


def _require_source_copy(module_name: str, source: Path) -> None:
    origin = _origin(module_name)
    if origin.read_bytes() != source.read_bytes():
        raise AssertionError(f"{module_name} does not match the current checkout source {source}")


def _require_editable_distribution(name: str, package_root: Path) -> None:
    distribution = importlib.metadata.distribution(name)
    payload = distribution.read_text("direct_url.json")
    if payload is None:
        raise AssertionError(f"{name} has no direct_url.json and may be a stale regular install")
    direct_url = json.loads(payload)
    if not direct_url.get("dir_info", {}).get("editable", False):
        raise AssertionError(f"{name} is not installed as an editable distribution: {direct_url}")
    parsed = urlparse(direct_url["url"])
    if parsed.scheme != "file":
        raise AssertionError(f"{name} direct URL is not a local checkout: {direct_url}")
    resolved = Path(url2pathname(parsed.path)).resolve()
    if resolved != package_root.resolve():
        raise AssertionError(f"{name} points at {resolved}, expected {package_root.resolve()}")


def _require_current_native(module_name: str) -> None:
    origin = _origin(module_name)
    marker_value = os.environ.get("RAYD_CI_BUILD_MARKER")
    if marker_value is None:
        raise AssertionError("RAYD_CI_BUILD_MARKER must identify the start of the current CI build")
    marker = Path(marker_value).resolve()
    if not marker.is_file():
        raise AssertionError(f"current-build marker does not exist: {marker}")
    if origin.stat().st_mtime_ns < marker.stat().st_mtime_ns:
        raise AssertionError(f"{module_name} predates the current CI build marker: {origin}")


def _require_current_torch_library(filename: str) -> None:
    import torch

    matches = [Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).stem == filename]
    if len(matches) != 1:
        raise AssertionError(f"expected one loaded Torch library named {filename}, found {matches}")
    marker = Path(os.environ["RAYD_CI_BUILD_MARKER"]).resolve()
    if matches[0].stat().st_mtime_ns < marker.stat().st_mtime_ns:
        raise AssertionError(f"loaded Torch library predates the current CI build marker: {matches[0]}")
    print(f"Torch loaded library {filename}: {matches[0]}")


def _torch_optix() -> None:
    import torch
    import rayd.torch as rt

    _require_editable_distribution("rayd-torch", ROOT / "torch")
    _require_under(_origin("rayd.torch"), ROOT / "python", "rayd.torch")
    _require_current_native("rayd.torch._C")
    _require_current_torch_library("_legacy_ops")
    _require_current_torch_library("_stable_ops")
    _require_source_copy("rayd._impl.runtime", ROOT / "python" / "rayd" / "_impl" / "runtime.py")
    if not rt._NATIVE_AVAILABLE:
        raise AssertionError(f"RayD Torch native runtime is unavailable: {rt._EXTENSION_IMPORT_ERROR}")

    vertices = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=torch.float32, device="cuda")
    faces = torch.tensor(((0, 1, 2),), dtype=torch.int32, device="cuda")
    scene = rt.Scene(trace_backend="optix", edge_bvh_backend="optix")
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    ray = rt.Ray(
        torch.tensor(((0.25, 0.25, -1.0),), dtype=torch.float32, device="cuda"),
        torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda"),
    )
    hit = scene.intersect(ray)
    torch.cuda.synchronize()
    if hit.global_prim_id.tolist() != [0]:
        raise AssertionError(f"Torch OptiX preflight missed the test triangle: {hit.global_prim_id.tolist()}")
    print(f"Torch OptiX hit t={hit.t.item():.6f}")


def _drjit_optix() -> None:
    import drjit.cuda as cuda
    import rayd.drjit as rt

    _require_editable_distribution("rayd-drjit", ROOT / "drjit")
    _require_under(_origin("rayd.drjit"), ROOT / "python", "rayd.drjit")
    _require_current_native("rayd.drjit._C")
    _require_source_copy("rayd._impl.runtime_jit", ROOT / "python" / "rayd" / "_impl" / "runtime_jit.py")
    if rt.device_count() < 1:
        raise AssertionError("RayD Dr.Jit reports no CUDA device")

    mesh = rt.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]), cuda.Array3i([0], [1], [2]))
    scene = rt.Scene()
    scene.add_mesh(mesh)
    scene.build()
    ray = rt.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
    hit = scene.intersect(ray)
    if not bool(hit.is_valid()[0]):
        raise AssertionError("Dr.Jit OptiX preflight missed the test triangle")
    print(f"Dr.Jit OptiX hit t={float(hit.t[0]):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("device", "optix"))
    args = parser.parse_args()
    if args.phase == "device":
        _device_preflight()
    else:
        _torch_optix()
        _drjit_optix()


if __name__ == "__main__":
    main()
