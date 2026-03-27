import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectMetadataTests(unittest.TestCase):
    def test_readme_matches_pinned_nanobind_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('nanobind==2.9.2', pyproject)
        self.assertIn('nanobind==2.9.2', readme)
        self.assertNotIn('nanobind==2.11.0', readme)


if __name__ == "__main__":
    unittest.main()
