import os
import unittest
import zipfile
from email.parser import BytesParser
from pathlib import Path


class WheelLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = os.environ.get("RAYD_META_WHEEL")
        drjit = os.environ.get("RAYD_DRJIT_WHEEL")
        torch = os.environ.get("RAYD_TORCH_WHEEL")
        if not meta or not drjit or not torch:
            raise unittest.SkipTest("wheel paths are provided by the packaging job")
        cls.meta_wheel = Path(meta)
        cls.drjit_wheel = Path(drjit)
        cls.torch_wheel = Path(torch)

    @staticmethod
    def names(path):
        with zipfile.ZipFile(path) as wheel:
            return set(wheel.namelist())

    @staticmethod
    def metadata(path):
        with zipfile.ZipFile(path) as wheel:
            metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
            return BytesParser().parsebytes(wheel.read(metadata_path))

    def test_namespace_root_is_implicit(self):
        for wheel in (self.drjit_wheel, self.torch_wheel):
            self.assertNotIn("rayd/__init__.py", self.names(wheel))

    def test_backend_files_are_disjoint(self):
        drjit = {name for name in self.names(self.drjit_wheel) if name.startswith("rayd/")}
        torch = {name for name in self.names(self.torch_wheel) if name.startswith("rayd/")}
        self.assertTrue(drjit)
        self.assertTrue(torch)
        self.assertFalse(drjit & torch)
        self.assertTrue(all(name.startswith("rayd/drjit/") for name in drjit))
        self.assertTrue(all(name.startswith("rayd/torch/") for name in torch))

    def test_torch_wheel_contains_untagged_stable_abi_library(self):
        names = self.names(self.torch_wheel)
        stable = [
            name
            for name in names
            if name.startswith("rayd/torch/_stable_ops")
            and name.endswith((".dll", ".so", ".dylib"))
        ]
        self.assertEqual(len(stable), 1, stable)
        self.assertNotRegex(stable[0], r"cp3(?:10|11|12|13|14)")

    def test_torch_wheel_contains_integrity_described_source_bundle(self):
        names = self.names(self.torch_wheel)
        prefix = "rayd/torch/_source/"
        self.assertIn(f"{prefix}rayd-source.json", names)
        self.assertIn(f"{prefix}source-files.json", names)
        self.assertIn(f"{prefix}source/backends/torch/CMakeLists.txt", names)
        self.assertIn(
            f"{prefix}source/backends/torch/include/rayd/torch/integration.h",
            names,
        )
        self.assertTrue(
            any(name.startswith(f"{prefix}source/shared/include/") for name in names)
        )
        self.assertTrue(
            any(name.startswith(f"{prefix}source/shared/src/") for name in names)
        )
        forbidden = ("/.git/", "/__pycache__/", ".obj", ".pdb", ".pyc")
        self.assertFalse(any(token in name for name in names for token in forbidden))

    def test_torch_wheel_separates_legacy_dispatcher_and_compatibility_shim(self):
        names = self.names(self.torch_wheel)
        legacy = [
            name
            for name in names
            if name.startswith("rayd/torch/_legacy_ops")
            and name.endswith((".dll", ".so", ".dylib"))
        ]
        compat = [
            name
            for name in names
            if name.startswith("rayd/torch/_C")
            and name.endswith((".pyd", ".so"))
        ]
        self.assertEqual(len(legacy), 1, legacy)
        self.assertEqual(len(compat), 1, compat)

    def test_backend_wheels_include_complete_typing_metadata(self):
        drjit = self.names(self.drjit_wheel)
        torch = self.names(self.torch_wheel)
        self.assertIn("rayd/drjit/py.typed", drjit)
        self.assertIn("rayd/drjit/__init__.pyi", drjit)
        self.assertIn("rayd/drjit/_capabilities.pyi", drjit)
        self.assertIn("rayd/torch/py.typed", torch)
        for name in (
            "__init__.pyi",
            "_capabilities.pyi",
            "camera.pyi",
            "mesh.pyi",
            "scene.pyi",
            "types.pyi",
        ):
            self.assertIn(f"rayd/torch/{name}", torch)

    def test_meta_wheel_is_file_free_and_pins_both_backends(self):
        self.assertFalse(any(name.startswith("rayd/") for name in self.names(self.meta_wheel)))
        meta = self.metadata(self.meta_wheel)
        drjit = self.metadata(self.drjit_wheel)
        torch = self.metadata(self.torch_wheel)
        self.assertEqual(meta["Version"], drjit["Version"])
        self.assertEqual(meta["Version"], torch["Version"])
        self.assertEqual(
            set(meta.get_all("Requires-Dist", [])),
            {f"rayd-drjit=={meta['Version']}", f"rayd-torch=={meta['Version']}"},
        )


if __name__ == "__main__":
    unittest.main()
