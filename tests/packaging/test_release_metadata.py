# Copyright Xingyu Chen.
# Tests release tag and project-version validation.

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate_release_tag.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("rayd_validate_release_tag", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load release-tag validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = _load_validator()

    def test_exact_v_prefixed_project_version_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            pyproject = Path(temporary) / "pyproject.toml"
            pyproject.write_text('[project]\nname = "rayd"\nversion = "0.10.1"\n', encoding="utf-8", newline="\n")
            self.assertEqual(self.validator.validate_release_tag("v0.10.1", pyproject), "v0.10.1")

    def test_mismatched_or_unprefixed_release_tag_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            pyproject = Path(temporary) / "pyproject.toml"
            pyproject.write_text('[project]\nname = "rayd"\nversion = "0.10.1"\n', encoding="utf-8", newline="\n")
            for tag in ("0.10.1", "v0.10.2", "release-0.10.1"):
                with self.subTest(tag=tag), self.assertRaisesRegex(RuntimeError, "does not match"):
                    self.validator.validate_release_tag(tag, pyproject)


if __name__ == "__main__":
    unittest.main()
