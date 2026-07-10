import tomllib
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
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("publish-drjit:", workflow)
        self.assertIn("publish-torch:", workflow)
        self.assertIn("publish-rayd:", workflow)
        self.assertIn("needs: [build-meta, publish-drjit, publish-torch]", workflow)

    def test_release_builds_complete_native_wheel_matrix(self):
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
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
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        guard = "github.event_name == 'release' && github.event.action == 'published'"
        self.assertEqual(workflow.count(guard), 3)
        self.assertIn("id-token: write", workflow)


if __name__ == "__main__":
    unittest.main()
