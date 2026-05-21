import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectMetadataTests(unittest.TestCase):
    def test_torch_and_slang_frontends_are_not_shipped(self):
        removed_paths = [
            ROOT / "rayd" / "torch",
            ROOT / "rayd" / "slang",
            ROOT / "tests" / "torch",
            ROOT / "tests" / "slang",
            ROOT / "include" / "rayd" / "slang",
            ROOT / "include" / "rayd_slang.slang",
            ROOT / "src" / "slang_interop.cpp",
        ]

        for path in removed_paths:
            self.assertFalse(path.exists(), f"Unexpected frontend artifact remains: {path}")

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn("rayd.torch", readme)
        self.assertNotIn("Slang", readme)
        self.assertNotIn("torch =", pyproject)

    def test_readme_matches_pinned_nanobind_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('nanobind==2.9.2', pyproject)
        self.assertIn('nanobind==2.9.2', readme)
        self.assertNotIn('nanobind==2.11.0', readme)

    def test_reflection_trace_ptx_header_is_committed(self):
        self.assertTrue(
            (
                ROOT
                / "include"
                / "rayd"
                / "multipath"
                / "reflection_trace_ptx.h"
            ).is_file(),
            "Expected committed reflection_trace PTX header for wheel builds.",
        )

    def test_reflection_epc_ptx_header_is_committed(self):
        self.assertTrue(
            (
                ROOT
                / "include"
                / "rayd"
                / "multipath"
                / "reflection_epc_ptx.h"
            ).is_file(),
            "Expected committed reflection_epc PTX header for wheel builds.",
        )


if __name__ == "__main__":
    unittest.main()
