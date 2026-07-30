# Copyright Xingyu Chen.
# Tests project metadata.

import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


class DistributionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        cls.drjit = tomllib.loads((ROOT / "drjit" / "pyproject.toml").read_text(encoding="utf-8"))
        cls.torch = tomllib.loads((ROOT / "torch" / "pyproject.toml").read_text(encoding="utf-8"))

    def test_all_distributions_share_one_version(self):
        versions = {self.meta["project"]["version"], self.drjit["project"]["version"], self.torch["project"]["version"]}
        self.assertEqual(versions, {"0.8.0"})

    def test_meta_distribution_pins_both_backends(self):
        version = self.meta["project"]["version"]
        self.assertEqual(
            set(self.meta["project"]["dependencies"]), {f"rayd-drjit=={version}", f"rayd-torch=={version}"}
        )

    def test_meta_distribution_owns_no_python_package(self):
        self.assertEqual(self.meta["tool"]["setuptools"]["packages"], [])

    def test_backend_wheels_map_only_their_public_namespace_portion(self):
        self.assertEqual(
            self.drjit["tool"]["scikit-build"]["wheel"]["packages"], {"rayd/drjit": "../python/rayd/drjit"}
        )
        self.assertEqual(
            self.torch["tool"]["scikit-build"]["wheel"]["packages"], {"rayd/torch": "../python/rayd/torch"}
        )

    def test_drjit_editable_mapping_cannot_replace_runtime_modules_with_stubs(self):
        package = ROOT / "python" / "rayd" / "drjit"
        self.assertFalse((package / "__init__.pyi").exists())
        self.assertFalse((package / "_C.pyi").exists())
        self.assertTrue((package / "__init__.py").is_file())
        self.assertTrue((package / "_C" / "__init__.pyi").is_file())

    def test_release_publishes_meta_after_backend_distributions(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        self.assertIn("publish-drjit:", workflow)
        self.assertIn("publish-torch:", workflow)
        self.assertIn("publish-rayd:", workflow)
        self.assertIn("needs: [build-meta, publish-drjit, publish-torch]", workflow)

    def test_release_builds_complete_native_wheel_matrix(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(f'"{version}"', workflow)
        for marker in (
            "build-drjit-linux:",
            "build-torch-linux:",
            "build-windows-wheels:",
            "validate-wheel-set:",
            "manylinux_2_28_x86_64",
            "pypa/cibuildwheel@",
            "auditwheel repair",
            "verify_cuda_binary_arches.py",
            "verify_stable_abi.py",
        ):
            self.assertIn(marker, workflow)

    def test_release_tag_is_validated_before_metadata_can_gate_builds(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        self.assertTrue((ROOT / "scripts" / "validate_release_tag.py").is_file())
        self.assertIn("!scripts/validate_release_tag.py", (ROOT / ".gitignore").read_text(encoding="utf-8"))
        metadata_job = workflow.split("\n  metadata:", 1)[1].split("\n  build-drjit-linux:", 1)[0]
        self.assertIn("github.event.release.tag_name", metadata_job)
        self.assertIn("github.event_name == 'release' && github.event.action == 'published'", metadata_job)
        self.assertIn(
            'python scripts/validate_release_tag.py --tag "$RAYD_RELEASE_TAG" --pyproject pyproject.toml', metadata_job
        )
        self.assertIn("tests.packaging.test_release_metadata", metadata_job)
        self.assertLess(metadata_job.index("validate_release_tag.py"), metadata_job.index("test_project_metadata"))

    def test_representative_wheel_selection_accepts_dual_tags_and_fails_loudly(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        selection = workflow.split("      - name: Validate representative wheel layout", 1)[1].split(
            "\n      - run:", 1
        )[0]
        self.assertIn("rayd-*-none-any.whl", selection)
        self.assertIn("rayd_drjit-*-cp312-cp312-*manylinux_2_28_x86_64.whl", selection)
        self.assertIn("rayd_torch-*-cp312-cp312-*manylinux_2_28_x86_64.whl", selection)
        self.assertEqual(selection.count("-ne 1"), 3)
        self.assertIn('-z "$meta_wheel"', selection)
        self.assertIn('-z "$drjit_wheel"', selection)
        self.assertIn('-z "$torch_wheel"', selection)
        self.assertNotIn("-print -quit", selection)
        self.assertLess(selection.index("-ne 1"), selection.index('>> "$GITHUB_ENV"'))

    def test_pypi_publish_is_release_only(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        guard = "github.event_name == 'release' && github.event.action == 'published'"
        publish_drjit = workflow.split("\n  publish-drjit:", 1)[1].split("\n  publish-torch:", 1)[0]
        publish_torch = workflow.split("\n  publish-torch:", 1)[1].split("\n  publish-rayd:", 1)[0]
        publish_rayd = workflow.split("\n  publish-rayd:", 1)[1]
        for name, job in (("drjit", publish_drjit), ("torch", publish_torch), ("rayd", publish_rayd)):
            with self.subTest(job=name):
                self.assertEqual(job.count(guard), 1)
        self.assertIn("id-token: write", workflow)

    def test_paid_workflows_are_opt_in(self):
        workflows = ROOT / ".github" / "workflows"
        release = (workflows / "pypi.yml").read_text(encoding="utf-8")
        self.assertNotIn("\n  push:", release)
        self.assertNotIn("\n  schedule:", release)
        self.assertIn("\n  release:\n    types: [published]", release)
        self.assertIn("\n  workflow_dispatch:", release)

        label_guard = "github.event_name == 'workflow_dispatch' || github.event.label.name == 'run-ci'"
        for name in ("ci.yml", "stable-abi-ci.yml"):
            workflow = (workflows / name).read_text(encoding="utf-8")
            self.assertNotIn("\n  push:", workflow)
            self.assertNotIn("\n  schedule:", workflow)
            self.assertIn("\n  pull_request:\n    types: [labeled]", workflow)
            self.assertIn(label_guard, workflow)

    def test_metadata_gates_compile_shared_math_on_posix(self):
        workflows = ROOT / ".github" / "workflows"
        for name in ("ci.yml", "pypi.yml"):
            workflow = (workflows / name).read_text(encoding="utf-8")
            with self.subTest(workflow=name):
                self.assertIn("tests.test_rt_host_compile", workflow)

    def test_workflow_python_modules_exist_at_repository_root(self):
        module_pattern = re.compile(r"(?<![A-Za-z0-9_])(?:tests|benchmarks)(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
        workflows = ROOT / ".github" / "workflows"
        workflow_paths = (*workflows.glob("*.yml"), *workflows.glob("*.yaml"))

        for workflow_path in sorted(workflow_paths):
            workflow = workflow_path.read_text(encoding="utf-8")
            for module in sorted(set(module_pattern.findall(workflow))):
                module_path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
                package_path = ROOT.joinpath(*module.split("."), "__init__.py")
                with self.subTest(workflow=workflow_path.name, module=module):
                    self.assertTrue(
                        module_path.is_file() or package_path.is_file(),
                        f"{workflow_path.name} references missing Python module {module}",
                    )


if __name__ == "__main__":
    unittest.main()
