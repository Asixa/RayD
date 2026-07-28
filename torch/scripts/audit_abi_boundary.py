from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


TORCH_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TORCH_ROOT.parent
DEFAULT_OUTPUT = TORCH_ROOT / "abi_audit.json"


def _ops(source: str) -> list[str]:
    return sorted(set(re.findall(r'm\.def\("([A-Za-z0-9_]+)', source)))


def _display_path(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def audit() -> dict[str, object]:
    library_path = WORKSPACE_ROOT / "src/bindings/library.cpp"
    module_path = WORKSPACE_ROOT / "src/bindings/module.cpp"
    runtime_path = WORKSPACE_ROOT / "python/rayd/_impl/runtime.py"
    stable_sources = [
        WORKSPACE_ROOT / "src/camera/camera_stable.cu",
        WORKSPACE_ROOT / "src/scene/intersection_stable.cu",
    ]
    library = library_path.read_text(encoding="utf-8")
    module = module_path.read_text(encoding="utf-8")
    stable_text = "\n".join(
        path.read_text(encoding="utf-8") for path in stable_sources
    )
    legacy_ops = _ops(library)
    stable_ops = _ops(stable_text)
    module_exports = _ops(module)
    py_object_files: dict[str, int] = {}
    for path in sorted((WORKSPACE_ROOT / "src").rglob("*")):
        if path.suffix not in {".h", ".cpp", ".cu"}:
            continue
        count = path.read_text(encoding="utf-8").count("py::object")
        if count:
            py_object_files[_display_path(path)] = count

    hashed = [
        TORCH_ROOT / "CMakeLists.txt",
        library_path,
        module_path,
        runtime_path,
        *stable_sources,
    ]
    digest = hashlib.sha256()
    for path in hashed:
        digest.update(_display_path(path).encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))

    return {
        "version": 2,
        "source_sha256": digest.hexdigest(),
        "decision": {
            "_C": "minimal_metadata_compatibility_shim",
            "_stable_ops": "independent_libtorch_stable_abi_dispatcher",
            "_legacy_ops": "python_and_libtorch_abi_bound_dispatcher_and_custom_classes",
        },
        "artifacts": {
            "_C": {
                "sources": [_display_path(module_path)],
                "exports": module_exports,
                "links_native_core": False,
                "uses_torch_extension_header": "torch/extension.h" in module,
            },
            "_stable_ops": {
                "sources": [_display_path(path) for path in stable_sources],
                "operators": stable_ops,
                "operator_count": len(stable_ops),
                "links_torch_python": False,
                "uses_stable_registration": "STABLE_TORCH_LIBRARY" in stable_text,
            },
            "_legacy_ops": {
                "sources": [
                    _display_path(library_path),
                    "src/bindings/legacy_anchor.cpp",
                ],
                "operators": legacy_ops,
                "operator_count": len(legacy_ops),
                "owns_scene_custom_class": "m.class_<SceneHandle>" in library,
                "links_native_core": True,
                "links_torch_python": True,
            },
        },
        "migration": {
            "stable": stable_ops,
            "typed_native_candidates": {
                "axial_edge_visibility_forward": (
                    "dormant_same_graph_exact_optix_source_integration"
                ),
                "segment_penetration_complete_family": (
                    "dormant_same_graph_batched_optix_fixed_winner_ad"
                ),
            },
            "legacy_retained": {
                "scene_custom_class_and_stateful_queries": (
                    "Scene ownership, OptiX/CUDA acceleration state, intrusive_ptr custom classes, and "
                    "ATen tensor ABI require the matched Python/LibTorch build."
                ),
                "geometry_ad_and_multipath": (
                    "Implementations share rayd_torch_native_core ATen/autograd objects and "
                    "py::object optional adapters; port kernel-by-kernel before moving."
                ),
            },
            "retired": {
                "plan13_extern_c_integration": (
                    "All same-graph native consumers use the versioned typed "
                    "rayd::torch integration surface."
                ),
            },
        },
        "inventory": {
            "py_object_occurrences_by_file": py_object_files,
            "py_object_occurrence_count": sum(py_object_files.values()),
            "legacy_dispatcher_operator_count": len(legacy_ops),
            "stable_dispatcher_operator_count": len(stable_ops),
            "compatibility_pybind_export_count": len(module_exports),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RayD Torch ABI boundaries")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(audit(), indent=2, sort_keys=True) + "\n"
    if args.check:
        actual = args.output.read_text(encoding="utf-8")
        if actual != rendered:
            raise SystemExit(f"ABI audit is stale: {args.output}")
        print(f"ABI audit is current: {args.output}")
        return
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
