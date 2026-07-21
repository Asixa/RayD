from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT / "backends" / "torch"


class TorchStableAbiBoundaryTests(unittest.TestCase):
    def test_machine_readable_audit_is_current(self):
        result = subprocess.run(
            [
                sys.executable,
                str(TORCH / "scripts" / "audit_abi_boundary.py"),
                "--check",
            ],
            cwd=TORCH,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stable_sources_use_only_stable_torch_headers(self):
        sources = list((TORCH / "src" / "stable").glob("*.cu"))
        self.assertEqual({path.name for path in sources}, {"camera.cu", "core.cu"})
        combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        for forbidden in ("at::", "c10::", "py::", "torch/extension.h", "torch/library.h"):
            self.assertNotIn(forbidden, combined)
        self.assertIn('m.def("intersection_valid(Tensor t, Tensor shape_id) -> Tensor")', combined)

    def test_c_extension_is_metadata_only_compatibility_shim(self):
        source = (TORCH / "src" / "torch_ext" / "module.cpp").read_text(encoding="utf-8")
        self.assertIn("<pybind11/pybind11.h>", source)
        self.assertNotIn("torch/extension.h", source)
        self.assertEqual(source.count("m.def("), 2)
        self.assertIn('m.def("build_info"', source)
        self.assertIn('m.def("contract_values"', source)

        cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        c_target = cmake[cmake.index("Python_add_library(_C"):]
        c_target = c_target[: c_target.index("endif()")]
        self.assertIn("src/torch_ext/module.cpp", c_target)
        self.assertNotIn("src/torch_ext/library.cpp", c_target)
        self.assertNotIn("rayd_torch_native_core", c_target)

    def test_legacy_dispatcher_is_separately_loadable(self):
        cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        start = cmake.index("rayd_torch_legacy_ops\n            SHARED")
        end = cmake.index("Python_add_library(_C", start)
        target = cmake[start:end]
        self.assertIn("src/torch_ext/library.cpp", target)
        self.assertIn("src/torch_ext/legacy_ops_anchor.cpp", target)
        self.assertIn("rayd_torch_native_core", target)
        self.assertIn("Python::Module", target)

        anchor = (TORCH / "src" / "torch_ext" / "legacy_ops_anchor.cpp").read_text(encoding="utf-8")
        self.assertIn("__declspec(dllexport)", anchor)
        self.assertIn("rayd_torch_legacy_ops_anchor", anchor)

        loader = (TORCH / "python" / "rayd" / "torch" / "_legacy.py").read_text(encoding="utf-8")
        self.assertIn('return "_legacy_ops.dll"', loader)
        self.assertIn("torch.ops.load_library", loader)
        self.assertIn("torch.classes.rayd_torch", loader)

    def test_package_loads_legacy_before_using_custom_classes(self):
        source = (TORCH / "python" / "rayd" / "torch" / "__init__.py").read_text(encoding="utf-8")
        self.assertLess(source.index("from . import _legacy"), source.index("from .scene import Scene"))
        self.assertIn("_NATIVE_AVAILABLE = _legacy.AVAILABLE or _legacy.is_registered()", source)
        self.assertIn("_C = (_compat_extension or _compat) if _NATIVE_AVAILABLE else None", source)

    def test_plan13_extern_c_integration_surface_is_retired(self):
        include = TORCH / "include" / "rayd" / "torch"
        self.assertFalse((include / "integration_v2.h").exists())
        typed = (include / "integration.h").read_text(encoding="utf-8")
        self.assertIn("namespace rayd::torch", typed)
        self.assertIn("kIntegrationApiVersion = 5", typed)
        self.assertIn('"rayd.torch.integration"', typed)
        self.assertIn("at::Tensor", typed)

        self.assertFalse(
            (TORCH / "src" / "torch_ext" / "integration_v2_internal.h").exists()
        )
        self.assertFalse(
            (TORCH / "tests" / "cpp" / "integration_v2_test.cpp").exists()
        )
        cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("integration_v2", cmake)
        self.assertIn("rayd_torch_integration_test", cmake)
        self.assertIn("NAME rayd_torch_integration", cmake)

        native_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (TORCH / "src" / "torch_ext").rglob("*")
            if path.suffix in {".h", ".cpp", ".cu"}
        )
        self.assertNotIn("rayd_torch_native_", native_sources)

    def test_audit_records_retained_abi_boundaries(self):
        audit = json.loads((TORCH / "abi_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["decision"]["_C"], "minimal_metadata_compatibility_shim")
        self.assertEqual(audit["artifacts"]["_C"]["exports"], ["build_info", "contract_values"])
        self.assertFalse(audit["artifacts"]["_C"]["links_native_core"])
        self.assertTrue(audit["artifacts"]["_legacy_ops"]["owns_scene_custom_class"])
        self.assertIn("intersection_valid", audit["migration"]["stable"])
        self.assertEqual(set(audit["migration"]["legacy_retained"]), {
            "scene_custom_class_and_stateful_queries",
            "geometry_ad_and_multipath",
        })
        self.assertNotIn("integration_h", audit["artifacts"])
        self.assertEqual(
            set(audit["migration"]["retired"]),
            {"plan13_extern_c_integration"},
        )


if __name__ == "__main__":
    unittest.main()
