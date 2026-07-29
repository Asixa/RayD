# Copyright Xingyu Chen.
# Tests project metadata.

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DistributionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        cls.drjit = tomllib.loads(
            (ROOT / "drjit" / "pyproject.toml").read_text(encoding="utf-8")
        )
        cls.torch = tomllib.loads(
            (ROOT / "torch" / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_all_distributions_share_one_version(self):
        versions = {
            self.meta["project"]["version"],
            self.drjit["project"]["version"],
            self.torch["project"]["version"],
        }
        self.assertEqual(versions, {"0.7.0"})

    def test_meta_distribution_pins_both_backends(self):
        version = self.meta["project"]["version"]
        self.assertEqual(
            set(self.meta["project"]["dependencies"]),
            {f"rayd-drjit=={version}", f"rayd-torch=={version}"},
        )

    def test_meta_distribution_owns_no_python_package(self):
        self.assertEqual(self.meta["tool"]["setuptools"]["packages"], [])

    def test_backend_wheels_map_only_their_public_namespace_portion(self):
        self.assertEqual(
            self.drjit["tool"]["scikit-build"]["wheel"]["packages"],
            {"rayd/drjit": "../python/rayd/drjit"},
        )
        self.assertEqual(
            self.torch["tool"]["scikit-build"]["wheel"]["packages"],
            {"rayd/torch": "../python/rayd/torch"},
        )
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

    def test_pypi_publish_is_release_only(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        guard = "github.event_name == 'release' && github.event.action == 'published'"
        self.assertEqual(workflow.count(guard), 3)
        self.assertIn("id-token: write", workflow)

    def test_paid_workflows_are_opt_in(self):
        workflows = ROOT / ".github" / "workflows"
        release = (workflows / "pypi.yml").read_text(encoding="utf-8")
        self.assertNotIn("\n  push:", release)
        self.assertNotIn("\n  schedule:", release)
        self.assertIn("\n  release:\n    types: [published]", release)
        self.assertIn("\n  workflow_dispatch:", release)

        label_guard = (
            "github.event_name == 'workflow_dispatch' "
            "|| github.event.label.name == 'run-ci'"
        )
        for name in ("ci.yml", "stable-abi-ci.yml"):
            workflow = (workflows / name).read_text(encoding="utf-8")
            self.assertNotIn("\n  push:", workflow)
            self.assertNotIn("\n  schedule:", workflow)
            self.assertIn("\n  pull_request:\n    types: [labeled]", workflow)
            self.assertIn(label_guard, workflow)


if __name__ == "__main__":
    unittest.main()
