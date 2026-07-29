from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT / "torch"


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
        sources = [
            ROOT / "src" / "camera" / "camera_stable.cu",
            ROOT / "src" / "scene" / "intersection_stable.cu",
        ]
        self.assertTrue(all(path.is_file() for path in sources))
        combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        for forbidden in ("at::", "c10::", "py::", "torch/extension.h", "torch/library.h"):
            self.assertNotIn(forbidden, combined)
        self.assertIn('m.def("intersection_valid(Tensor t, Tensor shape_id) -> Tensor")', combined)

    def test_c_extension_is_metadata_only_compatibility_shim(self):
        source = (ROOT / "src" / "bindings" / "module.cpp").read_text(encoding="utf-8")
        self.assertIn("<pybind11/pybind11.h>", source)
        self.assertNotIn("torch/extension.h", source)
        self.assertEqual(source.count("m.def("), 2)
        self.assertIn('m.def("build_info"', source)
        self.assertIn('m.def("contract_values"', source)

        cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        c_target = cmake[cmake.index("Python_add_library(_C"):]
        c_target = c_target[: c_target.index("endif()")]
        self.assertIn("src/bindings/module.cpp", c_target)
        self.assertNotIn("src/bindings/library.cpp", c_target)
        self.assertNotIn("rayd_torch_native_core", c_target)

    def test_legacy_dispatcher_is_separately_loadable(self):
        cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        start = cmake.index("rayd_torch_legacy_ops\n            SHARED")
        end = cmake.index("Python_add_library(_C", start)
        target = cmake[start:end]
        self.assertIn("src/bindings/library.cpp", target)
        self.assertIn("src/bindings/legacy_anchor.cpp", target)
        self.assertIn("rayd_torch_native_core", target)
        self.assertIn("Python::Module", target)

        anchor = (ROOT / "src" / "bindings" / "legacy_anchor.cpp").read_text(encoding="utf-8")
        self.assertIn("__declspec(dllexport)", anchor)
        self.assertIn("rayd_torch_legacy_ops_anchor", anchor)

        loader = (ROOT / "python" / "rayd" / "_impl" / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('_candidates("_legacy_ops", "RAYD_TORCH_LEGACY_LIBRARY")', loader)
        self.assertIn('RAYD_TORCH_LEGACY_LIBRARY', loader)
        self.assertIn("torch.ops.load_library", loader)
        self.assertIn("torch.classes.rayd_torch", loader)

    def run_probe(self, body: str) -> None:
        environment = os.environ.copy()
        # Do not let another checkout's regular ``rayd`` package shadow this
        # worktree's PEP 420 namespace.
        environment["PYTHONPATH"] = str(ROOT / "python")
        bootstrap = (
            "import importlib.machinery, sys, types\n"
            "rayd = types.ModuleType(\"rayd\")\n"
            f"rayd.__path__ = [{str(ROOT / 'python' / 'rayd')!r}]\n"
            "rayd.__package__ = \"rayd\"\n"
            "rayd.__spec__ = importlib.machinery.ModuleSpec("
            "\"rayd\", loader=None, is_package=True)\n"
            "sys.modules[\"rayd\"] = rayd\n"
            "spec = importlib.util.spec_from_file_location("
            "\"rayd.torch\", "
            f"{str(ROOT / 'python' / 'rayd' / 'torch' / '__init__.py')!r}, "
            f"submodule_search_locations=[{str(ROOT / 'python' / 'rayd' / 'torch')!r}])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules[\"rayd.torch\"] = module\n"
            "spec.loader.exec_module(module)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", bootstrap + textwrap.dedent(body)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_loads_legacy_before_using_custom_classes(self):
        # sys.modules preserves insertion order, and a module is inserted when
        # its execution starts, so this observes the real import order rather
        # than the text of __init__.py.
        self.run_probe(
            """
            import sys
            import rayd.torch
            order = list(sys.modules)
            assert "rayd._impl.runtime" in order, "canonical runtime was never imported"
            assert order.index("rayd._impl.runtime") < order.index("rayd._impl.scene"), order
            from rayd._impl import runtime
            if runtime.is_registered():
                import torch
                torch.classes.rayd_torch.Scene
            """
        )

    def test_native_metadata_shim_tracks_the_legacy_dispatcher(self):
        # `_C` doubles as the "native dispatcher available" sentinel for the
        # autograd/scene guards, so it must never be non-None while the legacy
        # dispatcher is missing, and the pure-Python stand-in must stay deleted.
        self.run_probe(
            """
            import rayd.torch as rt
            from rayd._impl import runtime
            assert not hasattr(rt, "_compat"), "pure-Python _C stand-in is back"
            assert rt._NATIVE_AVAILABLE == (runtime.AVAILABLE or runtime.is_registered())
            assert rt._C is None or rt._NATIVE_AVAILABLE, "_C is set without a dispatcher"
            assert (rt._EXTENSION_IMPORT_ERROR is None) == rt._NATIVE_AVAILABLE
            if rt._C is not None:
                assert rt._C.build_info()["backend"] == "torch"
                assert rt._C.__file__.endswith((".pyd", ".so", ".dylib"))
            """
        )

    def test_stable_accessors_have_no_legacy_dispatch_fallback(self):
        self.run_probe(
            """
            from unittest import mock
            import torch
            import rayd.torch
            from rayd._impl import runtime

            if runtime.AVAILABLE or runtime.is_registered():
                assert runtime._STABLE_AVAILABLE, runtime._STABLE_LOAD_ERROR
            if runtime._STABLE_AVAILABLE:
                assert runtime.camera_ops() is torch.ops.rayd_torch_stable
                assert runtime.core_ops() is torch.ops.rayd_torch_stable

            with mock.patch.object(runtime, "_STABLE_AVAILABLE", False):
                for accessor in (runtime.camera_ops, runtime.core_ops):
                    try:
                        accessor()
                    except RuntimeError as error:
                        assert "stable ABI operators are unavailable" in str(error)
                        assert error.__cause__ is runtime._STABLE_LOAD_ERROR
                    else:
                        raise AssertionError(f"{accessor.__name__} fell back to legacy ops")
            """
        )

        source = (ROOT / "python" / "rayd" / "_impl" / "runtime.py").read_text(encoding="utf-8")
        stable_source = source[: source.index("_LEGACY_REQUIRED")]
        self.assertIsNone(re.search(r"torch\.ops\.rayd_torch(?!_stable)", stable_source))

    def test_plan13_extern_c_integration_surface_is_retired(self):
        include = ROOT / "include" / "rayd"
        self.assertFalse((include / "integration" / "torch_v2.h").exists())
        typed = (include / "integration.h").read_text(encoding="utf-8")
        scene = (include / "scene.h").read_text(encoding="utf-8")
        self.assertIn("namespace rayd::torch", typed)
        self.assertIn("kIntegrationApiVersion = 8", typed)
        self.assertIn('"rayd.torch.integration"', typed)
        self.assertIn("at::Tensor", scene)

        self.assertFalse(
            (ROOT / "src" / "bindings" / "integration_v2_internal.h").exists()
        )
        self.assertFalse(
            (ROOT / "tests" / "native" / "integration_v2_test.cpp").exists()
        )
        cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("integration_v2", cmake)
        self.assertIn("rayd_torch_integration_test", cmake)
        self.assertIn("NAME rayd_torch_integration", cmake)

        native_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src").rglob("*")
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
