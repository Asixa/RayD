# Copyright Xingyu Chen.
# Prevents GPU and wheel lifecycle workflows from degrading into vacuous validation.

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SINGLE_GPU = WORKFLOWS / "single_gpu.yml"
MULTI_GPU = WORKFLOWS / "multi_gpu.yml"
RELEASE = WORKFLOWS / "pypi.yml"
HOSTED = WORKFLOWS / "ci.yml"
PREFLIGHT = ROOT / "tests" / "support" / "single_gpu_ci_preflight.py"
WHEEL_SMOKE = ROOT / "tests" / "packaging" / "wheel_lifecycle_smoke.py"
FIELD_PARITY = ROOT / "tests" / "parity" / "test_share2_ad.py"


class GpuCiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.single_gpu = SINGLE_GPU.read_text(encoding="utf-8")
        cls.multi_gpu = MULTI_GPU.read_text(encoding="utf-8")
        cls.release = RELEASE.read_text(encoding="utf-8")
        cls.hosted = HOSTED.read_text(encoding="utf-8")
        cls.preflight = PREFLIGHT.read_text(encoding="utf-8")
        cls.wheel_smoke = WHEEL_SMOKE.read_text(encoding="utf-8")

    def test_single_gpu_workflow_has_label_manual_and_scheduled_routes(self) -> None:
        self.assertIn("\n  pull_request:\n    types: [labeled]", self.single_gpu)
        self.assertIn("\n  schedule:", self.single_gpu)
        self.assertIn("\n  workflow_dispatch:", self.single_gpu)
        self.assertIn("github.event.label.name == 'run-gpu-ci'", self.single_gpu)
        self.assertIn("runs-on: [self-hosted, linux, x64, cuda, single-gpu]", self.single_gpu)

    def test_single_gpu_workflow_builds_both_backends_from_this_checkout(self) -> None:
        self.assertIn("actions/checkout@v5", self.single_gpu)
        self.assertIn("-m pip install --no-deps --no-build-isolation -e drjit", self.single_gpu)
        self.assertIn("-m pip install --no-deps --no-build-isolation -e torch", self.single_gpu)
        self.assertIn("RAYD_CI_BUILD_ROOT: ${{ runner.temp }}/rayd-build-", self.single_gpu)
        self.assertIn('-Cbuild-dir="$RAYD_CI_BUILD_ROOT/drjit"', self.single_gpu)
        self.assertIn('-Cbuild-dir="$RAYD_CI_BUILD_ROOT/torch"', self.single_gpu)
        self.assertNotIn("RAYD_CI_PREBUILT", self.single_gpu)

    def test_single_gpu_workflow_fails_before_skip_guarded_suites(self) -> None:
        device = self.single_gpu.index("single_gpu_ci_preflight.py device")
        optix = self.single_gpu.index("single_gpu_ci_preflight.py optix")
        suites = self.single_gpu.index("tests.parity.test_cuda_geometry")
        self.assertLess(device, optix)
        self.assertLess(optix, suites)
        self.assertIn("if count < 1:", self.preflight)
        self.assertIn("_torch_optix()", self.preflight)
        self.assertIn("_drjit_optix()", self.preflight)
        self.assertNotIn("SkipTest", self.preflight)
        self.assertNotIn("skipTest", self.preflight)
        self.assertNotIn("continue-on-error", self.single_gpu)
        self.assertNotIn("|| true", self.single_gpu)

    def test_single_gpu_workflow_covers_required_runtime_families(self) -> None:
        for marker in (
            "tests.parity.test_cuda_geometry",
            "tests.parity.test_cuda_multipath",
            "tests.native.test_multipath",
            "tests.scene.test_mesh_instancing",
            "tests.reflection.test_torch_high_level_api",
            "tests.scene.test_geometry_jit",
            "tests.reflection.test_epc_jit",
            "tests.diffraction.test_accumulation_jit",
            "tests.native.test_optix_pipeline_cold_create_jit",
            "tests.mixed.test_mixed_torch",
            "tests.mixed.test_mixed_jit",
            "tests.sdf.test_intersect",
            "tests.sdf.test_operations",
            "tests.sdf.test_operations_jit",
            "tests.parity.test_drjit",
            "tests.parity.test_share2_ad",
            'RAYD_TORCH_RUN_DR_JIT_PARITY: "1"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.single_gpu)
        self.assertIn("test_reflection_field_two_bounce_valid_path_geometry_parity", FIELD_PARITY.read_text())

    def test_single_gpu_preflight_rejects_stale_public_and_native_imports(self) -> None:
        for module in ("rayd.torch._C", "rayd.drjit._C"):
            with self.subTest(module=module):
                self.assertIn(f'_require_current_native("{module}")', self.preflight)
        self.assertIn('_origin("rayd.torch")', self.preflight)
        self.assertIn('_origin("rayd.drjit")', self.preflight)
        self.assertIn('_require_current_torch_library("_legacy_ops")', self.preflight)
        self.assertIn('_require_current_torch_library("_stable_ops")', self.preflight)
        self.assertIn("_require_editable_distribution", self.preflight)
        self.assertIn("_require_source_copy", self.preflight)
        self.assertIn("_require_current_native", self.preflight)
        self.assertIn("RAYD_CI_BUILD_MARKER", self.single_gpu)
        self.assertIn("pip uninstall -y rayd-drjit rayd-torch", self.single_gpu)

    def test_gpu_governance_is_run_by_hosted_and_release_metadata_jobs(self) -> None:
        self.assertIn("tests.governance.test_gpu_ci_contract", self.hosted)
        self.assertIn("tests.governance.test_gpu_ci_contract", self.release)

    def test_multi_gpu_workflow_runs_new_diffraction_operation_acceptance(self) -> None:
        self.assertIn("tests.test_multi_device_diffraction_paths", self.multi_gpu)
        self.assertIn("tests.test_multi_device_coherent_accum", self.multi_gpu)
        self.assertIn("tests.test_multi_device_reflection_accum", self.multi_gpu)

    def test_linux_release_matrix_runs_each_backend_lifecycle_in_cibuildwheel(self) -> None:
        self.assertEqual(self.release.count("CIBW_TEST_COMMAND:"), 2)
        for backend in ("drjit", "torch"):
            with self.subTest(backend=backend):
                self.assertIn(f"wheel_lifecycle_smoke.py installed {backend}", self.release)
                self.assertIn(f"pip uninstall -y rayd-{backend}", self.release)
                self.assertIn(f"wheel_lifecycle_smoke.py absent {backend}", self.release)

    def test_windows_release_matrix_installs_imports_and_removes_each_built_wheel(self) -> None:
        self.assertIn("Validate wheel install and uninstall lifecycle", self.release)
        self.assertIn("working-directory: ${{ runner.temp }}", self.release)
        self.assertIn("pip install --no-deps --force-reinstall $wheel", self.release)
        self.assertIn('wheel_lifecycle_smoke.py" installed $backend', self.release)
        self.assertIn('pip uninstall -y "rayd-$backend"', self.release)
        self.assertIn('wheel_lifecycle_smoke.py" absent $backend', self.release)

    def test_wheel_probe_requires_an_isolated_native_import_and_stable_abi(self) -> None:
        self.assertIn("_require_installed_path(module.__file__", self.wheel_smoke)
        self.assertIn("_require_installed_path(native.__file__", self.wheel_smoke)
        self.assertIn("module._NATIVE_AVAILABLE", self.wheel_smoke)
        self.assertIn("torch.ops.rayd_torch_stable", self.wheel_smoke)
        self.assertIn("importlib.util.find_spec(name)", self.wheel_smoke)
        self.assertGreaterEqual(self.release.count("python -I"), 6)


if __name__ == "__main__":
    unittest.main()
