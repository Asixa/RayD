# Copyright Xingyu Chen.
# Tests concept axis layout.

"""Checks the repository concept-oriented layout."""

import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
PRODUCTION_ROOTS = (ROOT / "src", ROOT / "include", ROOT / "python")


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


class ConceptAxisLayoutTests(unittest.TestCase):
    def test_retired_backend_container_is_absent(self):
        self.assertFalse((ROOT / "backends").exists())

    def test_agent_guides_are_synchronized(self):
        self.assertEqual((ROOT / "AGENTS.md").read_bytes(), (ROOT / "CLAUDE.md").read_bytes())

    def test_no_tracked_production_implementation_under_backends(self):
        offenders = []
        for path in tracked_files():
            rel = path.relative_to(ROOT)
            if not rel.parts or rel.parts[0] != "backends":
                continue
            if any(part in {"tests", "examples", "docs", "build", "scripts"} for part in rel.parts):
                continue
            if path.suffix.lower() in PRODUCTION_SUFFIXES:
                offenders.append(rel.as_posix())
        self.assertEqual(offenders, [], "production code must be concept-owned at root")

    def test_backend_frontends_do_not_own_native_implementation_trees(self):
        for frontend in (ROOT / "drjit", ROOT / "torch"):
            with self.subTest(frontend=frontend.name):
                self.assertFalse((frontend / "src").exists())
                self.assertFalse((frontend / "include").exists())

    def test_concept_variant_filename_convention(self):
        bad_backend_suffixes = []
        for root in PRODUCTION_ROOTS:
            for path in root.rglob("*"):
                if path.is_file() and re.search(r"_(?:torch|drjit)(?=\.)", path.name):
                    bad_backend_suffixes.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(bad_backend_suffixes, [])

    def test_rf_is_not_a_production_owner(self):
        rf_directories = []
        rf_namespaces = []
        namespace_re = re.compile(r"namespace\s+(?:::)?rayd(?:::\w+)*::rf\b")
        for root in PRODUCTION_ROOTS:
            for path in root.rglob("*"):
                if path.is_dir() and path.name == "rf":
                    rf_directories.append(path.relative_to(ROOT).as_posix())
                elif path.is_file() and path.suffix.lower() in PRODUCTION_SUFFIXES:
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    if namespace_re.search(text):
                        rf_namespaces.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(rf_directories, [])
        self.assertEqual(rf_namespaces, [])

    def test_public_headers_use_flat_default_and_jit_surfaces(self):
        include_root = ROOT / "include" / "rayd"
        self.assertEqual(
            {path.name for path in include_root.iterdir() if path.is_file()},
            {
                "diffraction.h",
                "integration.h",
                "path_exchange.h",
                "penetration.h",
                "reflection.h",
                "scattering.h",
                "scene.h",
                "transmission.h",
                "visibility.h",
                "contracts.h",
                "field_transport.cuh",
                "math.h",
                "scattering_table.cuh",
            },
        )
        self.assertEqual(
            {path.name for path in include_root.iterdir() if path.is_dir()},
            {"bvh", "diffraction", "edge", "jit", "reflection", "rt", "scene", "sdf", "transmission", "visibility"},
        )

        jit_root = include_root / "jit"
        self.assertEqual([path for path in jit_root.iterdir() if path.is_dir()], [])
        self.assertEqual(
            {path.name for path in jit_root.iterdir() if path.is_file()},
            {
                "core.h",
                "cuda_trace_backend.h",
                "diffraction_accumulation.h",
                "diffraction_paths.h",
                "edge.h",
                "edge_bvh.h",
                "edge_bvh_config.h",
                "edge_optix_params.h",
                "mesh.h",
                "native_launch_audit.h",
                "optix.h",
                "optix_trace_backend.h",
                "ray.h",
                "reflection_accumulation.h",
                "reflection_epc.h",
                "reflection_trace.h",
                "scene.h",
                "scene_edge.h",
                "scene_edge_optix.h",
                "scene_optix.h",
                "surfel.h",
                "surfel_optix.h",
                "surfel_trace_params.h",
                "trace_backend.h",
                "transform.h",
                "triangle_bvh_gpu.h",
                "types.h",
                "utils.h",
                "visibility.h",
            },
        )

        self.assertFalse((include_root / "detail").exists())
        self.assertTrue((include_root / "math.h").is_file())
        self.assertFalse(any(include_root.rglob("torch.h")))
        self.assertFalse(any(include_root.rglob("drjit.h")))
        self.assertFalse((include_root / "shared").exists())

    def test_torch_backend_private_headers_are_concept_owned(self):
        expected = {
            "src/bindings/tensor_contract.h",
            "src/camera/camera.h",
            "src/camera/camera_kernels.cuh",
            "src/diffraction/accum_ad.h",
            "src/diffraction/accum_params.h",
            "src/diffraction/accum_reduce.h",
            "src/diffraction/common.h",
            "src/diffraction/paths_init.h",
            "src/diffraction/paths_params.h",
            "src/diffraction/pipeline.h",
            "src/edge/bvh.h",
            "src/edge/kernels.h",
            "src/edge/optix_params.h",
            "src/penetration/segment_penetration_kernels.h",
            "src/penetration/segment_penetration_params.h",
            "src/reflection/accum_params.h",
            "src/reflection/accum_reduce.h",
            "src/reflection/dedup.h",
            "src/reflection/epc_field.h",
            "src/reflection/epc_params.h",
            "src/reflection/kernels.h",
            "src/reflection/pipeline.h",
            "src/reflection/trace_params.h",
            "src/runtime/diagnostics.h",
            "src/runtime/native_compat.h",
            "src/runtime/optix_context.h",
            "src/runtime/optix_pipeline.h",
            "src/scene/cache.h",
            "src/scene/cache_kernels.h",
            "src/scene/geometry_kernels.h",
            "src/scene/multipath_cuda.h",
            "src/scene/optix_intersect_params.h",
            "src/scene/triangle_bvh.h",
            "src/sdf/derivatives.cuh",
            "src/sdf/kernels.h",
            "src/visibility/axial_edge_visibility_params.h",
            "src/visibility/visibility.h",
            "src/visibility/visibility_params.h",
        }
        actual = set()
        for path in (ROOT / "src").rglob("*"):
            if path.suffix not in {".h", ".cuh"}:
                continue
            if "namespace rayd::torch_backend" in path.read_text(encoding="utf-8"):
                actual.add(path.relative_to(ROOT).as_posix())
        self.assertEqual(actual, expected)
        self.assertFalse((ROOT / "include/rayd/torch").exists())

    def test_torch_generated_ptx_headers_are_concept_owned(self):
        cmake = (ROOT / "torch/CMakeLists.txt").read_text(encoding="utf-8")
        expected = {
            "rayd/scene/intersection_torch_ptx.h",
            "rayd/edge/point_ray_torch_ptx.h",
            "rayd/edge/topk_torch_ptx.h",
            "rayd/reflection/trace_torch_ptx.h",
            "rayd/reflection/epc_torch_ptx.h",
            "rayd/reflection/accumulation_torch_ptx.h",
            "rayd/visibility/segment_torch_ptx.h",
            "rayd/visibility/axial_edge_torch_ptx.h",
            "rayd/diffraction/paths_torch_ptx.h",
            "rayd/diffraction/accumulation_torch_ptx.h",
            "rayd/penetration/segment_torch_ptx.h",
        }
        actual = set(re.findall(r"generated/(rayd/[^\"]+_torch_ptx\.h)", cmake))
        self.assertEqual(actual, expected)
        self.assertNotIn("generated/rayd/torch", cmake)
        self.assertEqual(cmake.count('-I "${RAYD_ROOT_DIR}"'), 11)
        for path in (ROOT / "src").rglob("*"):
            if not path.is_file() or path.suffix not in PRODUCTION_SUFFIXES:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"#include\s*<rayd/torch/[^>]*ptx\.h>")

    def test_shared_physical_sources_have_one_root_owner(self):
        expected = {
            "src/bvh/build_shared.cu",
            "src/bvh/triangle_query_shared.cu",
            "src/edge/edge_shared.cu",
            "src/reflection/dedup_shared.cu",
            "src/scene/packing_shared.cu",
        }
        actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*_shared.*") if path.is_file()}
        self.assertEqual(actual, expected)
        for rel in expected:
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_compile_policy_preserves_logical_tu_identity(self):
        contract = json.loads((ROOT / "contracts" / "compile_policy.json").read_text(encoding="utf-8"))
        units = contract["translation_units"]
        identities = {(entry["backend"], entry["unit"]) for entry in units}
        self.assertEqual(len(units), 80)
        self.assertEqual(len(identities), 80)
        self.assertEqual({entry["backend"] for entry in units}, {"drjit", "torch"})
        for entry in units:
            with self.subTest(unit=f"{entry['backend']}/{entry['unit']}"):
                self.assertTrue((ROOT / entry["source"]).is_file(), entry["source"])

    def test_public_python_frontends_share_one_namespace_source_root(self):
        namespace = ROOT / "python" / "rayd"
        self.assertFalse((namespace / "__init__.py").exists())
        self.assertFalse((ROOT / "drjit" / "python").exists())
        self.assertFalse((ROOT / "torch" / "python").exists())
        self.assertEqual(
            {path.name for path in (namespace / "drjit").iterdir() if path.is_file()},
            {"__init__.py", "__init__.pyi", "_C.pyi", "path_exchange.py", "py.typed"},
        )
        self.assertEqual(
            {path.name for path in (namespace / "torch").iterdir() if path.is_file()},
            {"__init__.py", "path_exchange.py", "py.typed"},
        )

    def test_private_python_implementation_ownership_is_disjoint(self):
        implementation = ROOT / "python" / "rayd" / "_impl"
        self.assertFalse((implementation / "__init__.py").exists())
        files = {path.name for path in implementation.glob("*.py")}
        jit = {name for name in files if name.endswith("_jit.py")}
        torch_owned = files - jit
        self.assertTrue(jit)
        self.assertTrue(torch_owned)
        self.assertFalse(jit & torch_owned)


if __name__ == "__main__":
    unittest.main()
