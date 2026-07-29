# Copyright Xingyu Chen.
# Tests release artifact matrix.

import os
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PYTHON_TAGS = ("cp310-cp310", "cp311-cp311", "cp312-cp312", "cp313-cp313", "cp314-cp314")


class ReleaseArtifactMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist_dir = os.environ.get("RAYD_RELEASE_DIST_DIR")
        if not dist_dir:
            raise unittest.SkipTest("release artifact directory is provided by distribution CI")
        cls.dist_dir = Path(dist_dir)
        root = Path(__file__).resolve().parents[2]
        cls.version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

    def wheels(self, distribution):
        normalized = distribution.replace("-", "_")
        return sorted(self.dist_dir.glob(f"{normalized}-*.whl"))

    def test_each_native_backend_has_complete_platform_matrix(self):
        for distribution in ("rayd-drjit", "rayd-torch"):
            wheels = self.wheels(distribution)
            self.assertEqual(len(wheels), 10, (distribution, wheels))
            names = [wheel.name for wheel in wheels]
            for python_tag in PYTHON_TAGS:
                matching = [name for name in names if python_tag in name]
                self.assertEqual(len(matching), 2, (distribution, python_tag, matching))
                self.assertTrue(
                    any(name.endswith("manylinux_2_28_x86_64.whl") for name in matching),
                    matching,
                )
                self.assertTrue(any(name.endswith("win_amd64.whl") for name in matching), matching)

    def test_meta_distribution_has_one_universal_wheel_and_sdist(self):
        self.assertEqual(
            [path.name for path in self.dist_dir.glob("rayd-*-none-any.whl")],
            [f"rayd-{self.version}-py3-none-any.whl"],
        )
        self.assertEqual(
            [path.name for path in self.dist_dir.glob("rayd-*.tar.gz")],
            [f"rayd-{self.version}.tar.gz"],
        )


if __name__ == "__main__":
    unittest.main()
