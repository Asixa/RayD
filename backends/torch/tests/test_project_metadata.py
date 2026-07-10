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

    def test_public_python_source_has_no_obsolete_product_name(self):
        source_root = Path("python") / "rayd" / "torch"
        source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
        self.assertNotIn("ray" + "dn", source.lower())
        self.assertNotIn("rayd-native", source.lower())
        self.assertNotIn("_ray" + "dn", source.lower())
