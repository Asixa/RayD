# Copyright Xingyu Chen.
# Tests the installation diagnostics shared by the backend runtimes and doctors without a GPU.

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python" / "rayd" / "_impl" / "diagnostics.py"


def _load():
    spec = importlib.util.spec_from_file_location("rayd_diagnostics_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnostics = _load()

RELEASE_RECORD = {
    "backend": "torch",
    "cuda_sass": [70, 75, 80, 86, 89, 90, 100, 120],
    "cuda_ptx": [120],
    "optix_ptx": 70,
    "optix_abi": 93,
    "cuda_toolkit": "12.8.93",
    "framework": "torch",
    "framework_version": "2.10.0+cu128",
    "framework_cuda": "12.8",
}


class TorchMismatchTests(unittest.TestCase):
    def test_cpu_only_torch_is_named_with_the_cu128_command(self):
        error = OSError("libc10_cuda.so: cannot open shared object file: No such file or directory")
        message = diagnostics.torch_load_failure(
            "_legacy_ops.so", Path("/site/rayd/torch/_legacy_ops.so"), error, "2.10.0+cpu", None, RELEASE_RECORD
        )
        self.assertIn("could not load its native library _legacy_ops.so from /site/rayd/torch/_legacy_ops.so", message)
        self.assertIn("OSError: libc10_cuda.so", message)
        self.assertIn("Installed torch: 2.10.0+cpu, CPU-only (torch.version.cuda is None).", message)
        self.assertIn("compiled against torch 2.10.0+cu128 (CUDA 12.8)", message)
        self.assertIn("is a CPU-only build", message)
        self.assertIn(
            'python -m pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128', message
        )
        self.assertIn("python -m rayd.torch.doctor", message)

    def test_other_torch_release_is_named(self):
        message = diagnostics.torch_load_failure(
            "_legacy_ops.so", None, OSError("undefined symbol: _ZN3c10"), "2.11.0+cu128", "12.8", RELEASE_RECORD
        )
        self.assertIn("differs from torch 2.10.0+cu128", message)
        self.assertIn('"torch==2.10.0+cu128"', message)

    def test_matching_torch_points_at_a_damaged_library(self):
        message = diagnostics.torch_load_failure(
            "_legacy_ops.so",
            Path("/x/_legacy_ops.so"),
            OSError("invalid ELF header"),
            "2.10.0+cu128",
            "12.8",
            RELEASE_RECORD,
        )
        self.assertIn("the library itself is damaged", message)
        self.assertIn("--force-reinstall --no-deps rayd-torch", message)

    def test_missing_library_without_a_record_describes_an_unbuilt_tree(self):
        missing = FileNotFoundError("RayD library _legacy_ops.so was not found beside rayd.torch._C")
        message = diagnostics.torch_load_failure("_legacy_ops.so", None, missing, "2.10.0+cu128", "12.8", None)
        self.assertIn("carries no record of the torch it was compiled against", message)
        self.assertIn("source tree that was never built", message)

    def test_install_command_without_a_record_forces_the_cuda_build(self):
        self.assertEqual(
            diagnostics.torch_install_command(None),
            'python -m pip install --force-reinstall "torch==2.10.*" '
            "--index-url https://download.pytorch.org/whl/cu128",
        )

    def test_cuda_major_mismatch_is_named(self):
        self.assertIn("CUDA 13.0", diagnostics.torch_mismatch(RELEASE_RECORD, "2.10.0+cu130", "13.0"))
        self.assertIsNone(diagnostics.torch_mismatch(RELEASE_RECORD, "2.10.0+cu126", "12.6"))

    def test_drjit_version_mismatch_names_the_pin(self):
        record = {"framework_version": "1.3.1"}
        message = diagnostics.drjit_load_failure(ImportError("undefined symbol"), "1.2.0", record)
        self.assertIn("compiled against drjit 1.3.1", message)
        self.assertIn('python -m pip install "drjit==1.3.1"', message)


class OptixRequirementTests(unittest.TestCase):
    def test_release_abi_names_the_r555_floor(self):
        self.assertEqual(
            diagnostics.optix_driver_requirement(93), "OptiX 8.1 (ABI 93), which needs driver R555 or newer"
        )
        self.assertIn("OptiX 9.1 (ABI 118)", diagnostics.optix_driver_requirement(118))

    def test_driver_branch_parses_linux_and_windows_versions(self):
        self.assertEqual(diagnostics.DriverInfo(driver_version="595.99.02").branch, 595)
        self.assertEqual(diagnostics.DriverInfo(driver_version="572.16").branch, 572)
        self.assertIsNone(diagnostics.DriverInfo().branch)

    def test_old_driver_fails_the_version_check(self):
        report = diagnostics.Report("rayd-torch")
        driver = diagnostics.DriverInfo(driver_version="550.54.14", gpus=[diagnostics.Gpu(0, "GPU", 86)])
        diagnostics.add_driver(report, driver, RELEASE_RECORD)
        self.assertEqual(report.failures, 1)
        self.assertIn("[FAIL] driver version: R550; OptiX 8.1 (ABI 93), which needs driver R555", report.render())
        self.assertIn("update the NVIDIA driver to R555 or newer", report.render())

    def test_missing_optix_library_carries_the_docker_hint(self):
        report = diagnostics.Report("rayd-torch")
        with mock.patch.object(diagnostics, "_optix_library_candidates", return_value=["/nonexistent/libnvoptix.so.1"]):
            probe = diagnostics.add_optix(report, RELEASE_RECORD, disabled=False)
        self.assertIsNone(probe.library)
        self.assertEqual(report.failures, 1)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility", report.render())


class EditableStalenessTests(unittest.TestCase):
    def _tree(self, root: Path):
        source = root / "checkout" / "python" / "rayd"
        installed = root / "site" / "rayd" / "_impl"
        (source / "torch").mkdir(parents=True)
        (source / "_impl").mkdir()
        installed.mkdir(parents=True)
        (source / "torch" / "__init__.py").write_text("")
        for name, text in {
            "runtime.py": "a",
            "scene.py": "new",
            "runtime_jit.py": "jit",
            "device_support.py": "ds",
        }.items():
            (source / "_impl" / name).write_text(text)
        for name, text in {"runtime.py": "a", "scene.py": "old", "device_support.py": "ds"}.items():
            (installed / name).write_text(text)
        return source, installed

    def _spec(self, origin: Path):
        return types.SimpleNamespace(origin=str(origin))

    def test_torch_reports_only_its_own_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source, installed = self._tree(Path(directory))
            spec = self._spec(source / "torch" / "__init__.py")
            with mock.patch.object(diagnostics.importlib.util, "find_spec", return_value=spec):
                checkout, stale = diagnostics.stale_editable_sources("torch", str(installed / "runtime.py"))
                self.assertEqual(stale, ["scene.py"])
                self.assertEqual(checkout, Path(directory).resolve() / "checkout")
                with self.assertRaises(ImportError) as raised:
                    diagnostics.require_current_editable_install("torch", str(installed / "runtime.py"))
        message = str(raised.exception)
        self.assertIn("no longer match the checkout: scene.py", message)
        self.assertIn("scripts/build_local.sh torch", message)
        self.assertIn("pip install --no-build-isolation -e", message)

    def test_drjit_compares_its_renamed_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            source, installed = self._tree(Path(directory))
            (source / "drjit").mkdir()
            (source / "drjit" / "__init__.py").write_text("")
            (installed / "runtime_jit.py").write_text("jit")
            spec = self._spec(source / "drjit" / "__init__.py")
            with mock.patch.object(diagnostics.importlib.util, "find_spec", return_value=spec):
                _, stale = diagnostics.stale_editable_sources("drjit", str(installed / "runtime_jit.py"))
                self.assertEqual(stale, ["device_support.py"])
                (installed / "device_support_jit.py").write_text("ds")
                self.assertIsNone(diagnostics.stale_editable_sources("drjit", str(installed / "runtime_jit.py")))

    def test_wheel_and_source_tree_installs_are_never_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self._tree(Path(directory))
            spec = self._spec(source / "torch" / "__init__.py")
            with mock.patch.object(diagnostics.importlib.util, "find_spec", return_value=spec):
                self.assertIsNone(diagnostics.stale_editable_sources("torch", str(source / "_impl" / "runtime.py")))


class ReportTests(unittest.TestCase):
    def test_skipped_checks_do_not_fail_the_report(self):
        report = diagnostics.Report("rayd-torch")
        report.check("a", True, "fine")
        report.check("b", None, "not run")
        self.assertEqual(report.failures, 0)
        self.assertIn("Result: all checks passed.", report.render())
        report.check("c", False, "broken", "do this")
        self.assertIn("         fix: do this", report.render())
        self.assertIn("Result: 1 check(s) failed.", report.render())


if __name__ == "__main__":
    unittest.main()
