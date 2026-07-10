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


if __name__ == "__main__":
    unittest.main()
