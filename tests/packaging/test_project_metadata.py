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
            (ROOT / "backends" / "drjit" / "pyproject.toml").read_text(encoding="utf-8")
        )
        cls.torch = tomllib.loads(
            (ROOT / "backends" / "torch" / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_all_distributions_share_one_version(self):
        versions = {
            self.meta["project"]["version"],
            self.drjit["project"]["version"],
            self.torch["project"]["version"],
        }
        self.assertEqual(versions, {"0.6.0"})

    def test_meta_distribution_pins_both_backends(self):
        version = self.meta["project"]["version"]
        self.assertEqual(
            set(self.meta["project"]["dependencies"]),
            {f"rayd-drjit=={version}", f"rayd-torch=={version}"},
        )

    def test_meta_distribution_owns_no_python_package(self):
        self.assertEqual(self.meta["tool"]["setuptools"]["packages"], [])

    def test_release_publishes_meta_after_backend_distributions(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        self.assertIn("publish-drjit:", workflow)
        self.assertIn("publish-torch:", workflow)
        self.assertIn("publish-rayd:", workflow)
        self.assertIn("needs: [scope, build-meta, publish-drjit, publish-torch]", workflow)

    def test_release_builds_complete_native_wheel_matrix(self):
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(f'"{version}"', workflow)
        for build in ("cp310", "cp311", "cp312", "cp313", "cp314"):
            self.assertIn(f"{build}-manylinux_x86_64", workflow)
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

    def test_ordinary_pushes_do_not_start_ci(self):
        """GITHUB_ACTIONS_PREBUILD_MATRIX.md: only a `run-ci` label, an explicit
        dispatch, or a published release may start a paid RayD build."""
        for name in ("pypi.yml", "stable-abi-ci.yml"):
            workflow = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
            with self.subTest(workflow=name):
                self.assertNotIn("\n  push:\n", workflow)
                self.assertIn("  pull_request:\n    types: [labeled]\n", workflow)
                self.assertIn("github.event.label.name == 'run-ci'", workflow)

    def test_reduced_architecture_artifacts_cannot_publish(self):
        """A smoke run builds sm_87/sm_120 only, so every publishing job must
        additionally require the resolved full scope."""
        workflow = (ROOT / ".github" / "workflows" / "pypi.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count("needs.scope.outputs.full == 'true'"), 4)
        self.assertIn(
            "gencode-families=--generate-code=arch=compute_87,code=sm_87 "
            "--generate-code=arch=compute_120,code=[sm_120,compute_120]",
            workflow,
        )
        self.assertIn("gencode-families=--generate-code=arch=compute_70,", workflow)


if __name__ == "__main__":
    unittest.main()
