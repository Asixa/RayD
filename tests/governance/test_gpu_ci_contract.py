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
TORCH_BUILD_CONSTRAINT = ROOT / ".github" / "constraints" / "torch-build.txt"


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

    def test_release_torch_build_and_lifecycle_use_one_pinned_version(self) -> None:
        self.assertEqual(TORCH_BUILD_CONSTRAINT.read_text(encoding="utf-8").strip(), "torch==2.10.0")
        torch_linux = self.release.split("  build-torch-linux:", 1)[1].split("  build-windows-wheels:", 1)[0]
        windows = self.release.split("  build-windows-wheels:", 1)[1].split(
            "  test-torch-full-wheel-compatibility:", 1
        )[0]

        self.assertIn("PIP_CONSTRAINT=/project/.github/constraints/torch-build.txt", torch_linux)
        self.assertIn('CIBW_TEST_REQUIRES: "torch==2.10.0"', torch_linux)
        self.assertNotIn("lib64/stubs", torch_linux)
        self.assertIn("torch/scripts/verify_driver_independence.py", torch_linux)
        self.assertIn("pip install --constraint .github/constraints/torch-build.txt torch", windows)
        self.assertIn("python torch/scripts/verify_driver_independence.py $wheel", windows)
        self.assertEqual(self.release.count("torch/scripts/verify_driver_independence.py"), 2)

    def test_torch_211_full_wheel_lifecycle_covers_every_published_wheel(self) -> None:
        lifecycle = self.release.split("  test-torch-full-wheel-compatibility:", 1)[1].split(
            "  test-stable-torch-abi:", 1
        )[0]
        self.assertIn("needs: [build-torch-linux, build-windows-wheels]", lifecycle)
        self.assertEqual(lifecycle.count('- {os: "'), 10)
        for os_name, artifact_os in (("ubuntu-22.04", "linux"), ("windows-2022", "windows")):
            for python_version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
                with self.subTest(os=os_name, python=python_version):
                    artifact = f"release-rayd-torch-{artifact_os}-py{python_version}"
                    self.assertIn(
                        f'{{os: "{os_name}", python-version: "{python_version}", artifact: "{artifact}"}}', lifecycle
                    )
        for marker in (
            "python-version: ${{ matrix.python-version }}",
            "name: ${{ matrix.artifact }}",
            "path: ${{ runner.temp }}/wheel-artifact",
            "working-directory: ${{ runner.temp }}",
            "torch==2.11.0",
            "https://download.pytorch.org/whl/cu128",
            "pip install --no-deps --force-reinstall $wheel",
            'wheel_lifecycle_smoke.py" installed torch',
            "pip uninstall -y rayd-torch",
            'wheel_lifecycle_smoke.py" absent torch',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, lifecycle)

        validation = self.release.split("  validate-wheel-set:", 1)[1].split("  publish-drjit:", 1)[0]
        self.assertIn("- test-torch-full-wheel-compatibility", validation)

    def test_windows_native_release_commands_fail_immediately(self) -> None:
        fail_fast = "$nativeExit=$LASTEXITCODE; if ($nativeExit -ne 0) { exit $nativeExit }"
        windows = self.release.split("  build-windows-wheels:", 1)[1].split(
            "  test-torch-full-wheel-compatibility:", 1
        )[0]
        compatibility = self.release.split("  test-torch-full-wheel-compatibility:", 1)[1].split(
            "  test-stable-torch-abi:", 1
        )[0]

        self.assertEqual(windows.count(fail_fast), 14)
        self.assertEqual(compatibility.count(fail_fast), 6)

        windows_command_tails = (
            'python -m pip install --upgrade pip build twine "scikit-build-core>=0.10" "cmake>=3.22" ninja',
            'python -m pip install --constraint .github/constraints/drjit-build.txt "nanobind==2.11.0" "drjit==1.3.1"',
            'python -m pip install --upgrade pip build twine "scikit-build-core>=0.10" "cmake>=3.26" ninja',
            "--index-url https://download.pytorch.org/whl/cu128",
            "python drjit/scripts/verify_cuda_binary_arches.py --stem _C $wheel",
            "python drjit/scripts/verify_cuda_binary_arches.py --stem _legacy_ops --stem _stable_ops $wheel",
            "python torch/scripts/verify_stable_abi.py --source-root src $wheel",
            "python torch/scripts/verify_driver_independence.py $wheel",
            "python -m pip install --no-deps --force-reinstall $wheel",
            'python -I "$env:GITHUB_WORKSPACE/tests/packaging/wheel_lifecycle_smoke.py" installed $backend',
            'python -m pip uninstall -y "rayd-$backend"',
            'python -I "$env:GITHUB_WORKSPACE/tests/packaging/wheel_lifecycle_smoke.py" absent $backend',
        )
        for command in windows_command_tails:
            with self.subTest(command=command):
                self.assertIn(f"{command}\n          {fail_fast}", windows)
        self.assertEqual(windows.count(f"python -m twine check $wheel\n          {fail_fast}"), 2)

        compatibility_command_tails = (
            "python -m pip install --upgrade pip",
            "--index-url https://download.pytorch.org/whl/cu128",
            "python -m pip install --no-deps --force-reinstall $wheel",
            'python -I "$env:GITHUB_WORKSPACE/tests/packaging/wheel_lifecycle_smoke.py" installed torch',
            "python -m pip uninstall -y rayd-torch",
            'python -I "$env:GITHUB_WORKSPACE/tests/packaging/wheel_lifecycle_smoke.py" absent torch',
        )
        for command in compatibility_command_tails:
            with self.subTest(command=command):
                self.assertIn(f"{command}\n          {fail_fast}", compatibility)

    def test_drjit_ci_nanobind_pin_matches_drjit_registry_abi(self) -> None:
        for name, workflow in (("hosted", self.hosted), ("release", self.release)):
            with self.subTest(workflow=name):
                self.assertIn('"nanobind==2.11.0" "drjit==1.3.1"', workflow)
                self.assertNotIn('"nanobind==2.9.2" "drjit==1.3.1"', workflow)

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
        self.assertIn("{extension_error!r}", self.wheel_smoke)
        self.assertIn("from extension_error", self.wheel_smoke)
        self.assertIn("torch.ops.rayd_torch_stable", self.wheel_smoke)
        self.assertIn("importlib.util.find_spec(name)", self.wheel_smoke)
        self.assertGreaterEqual(self.release.count("python -I"), 6)


if __name__ == "__main__":
    unittest.main()
