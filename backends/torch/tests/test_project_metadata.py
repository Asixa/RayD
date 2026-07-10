import tomllib
import unittest
from pathlib import Path


class ProjectMetadataTests(unittest.TestCase):
    def test_project_name_is_rayd_torch(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(data["project"]["name"], "rayd-torch")

    def test_default_dependencies_require_torch_not_dr_jit(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        deps = [dep.lower() for dep in data["project"].get("dependencies", [])]
        self.assertTrue(any(dep.startswith("torch") for dep in deps))
        self.assertFalse(any(dep.startswith("dr" + "jit") for dep in deps))

    def test_transitional_wheels_cover_supported_python_and_torch_baseline(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(data["project"]["requires-python"], ">=3.10,<3.15")
        self.assertIn("torch>=2.10,<2.11", data["project"]["dependencies"])
        self.assertIn("torch==2.10.0", data["build-system"]["requires"])

    def test_public_python_source_has_no_obsolete_product_name(self):
        source_root = Path("python") / "rayd" / "torch"
        source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
        self.assertNotIn("ray" + "dn", source.lower())
        self.assertNotIn("rayd-native", source.lower())
        self.assertNotIn("_ray" + "dn", source.lower())

    def test_stable_abi_slice_avoids_unstable_torch_and_python_apis(self):
        stable_source = Path("src/stable/camera.cu").read_text(encoding="utf-8")
        cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
        for forbidden in ("at::", "c10::", "py::", "torch/extension.h", "torch/library.h"):
            self.assertNotIn(forbidden, stable_source)
        self.assertIn("STABLE_TORCH_LIBRARY(rayd_torch_stable", stable_source)
        self.assertIn("TORCH_TARGET_VERSION=0x020a000000000000", cmake)
        stable_target = cmake[cmake.index("add_library(rayd_torch_stable_ops"):cmake.index("execute_process(")]
        self.assertNotIn("TORCH_PYTHON_LIBRARY", stable_target)

    def test_stable_abi_audit_script_is_packaged_with_the_backend(self):
        script = Path("scripts/verify_stable_abi.py")
        self.assertTrue(script.is_file())
        source = script.read_text(encoding="utf-8")
        for dependency in ("torch_python", "c10.dll", "libc10.so", "python3"):
            self.assertIn(dependency, source)

    def test_cuda_fat_binary_covers_witwin_platform_matrix(self):
        cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
        expected = "70-real;75-real;80-real;86-real;89-real;90-real;100-real;101-real;120-real;120-virtual"
        self.assertIn(f'set(RAYD_TORCH_DEFAULT_CUDA_ARCHITECTURES "{expected}")', cmake)
