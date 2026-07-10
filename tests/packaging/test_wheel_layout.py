import os
import unittest
import zipfile
from pathlib import Path


class WheelLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        drjit = os.environ.get("RAYD_DRJIT_WHEEL")
        torch = os.environ.get("RAYD_TORCH_WHEEL")
        if not drjit or not torch:
            raise unittest.SkipTest("wheel paths are provided by the packaging job")
        cls.drjit_wheel = Path(drjit)
        cls.torch_wheel = Path(torch)

    @staticmethod
    def names(path):
        with zipfile.ZipFile(path) as wheel:
            return set(wheel.namelist())

    def test_namespace_root_is_implicit(self):
        for wheel in (self.drjit_wheel, self.torch_wheel):
            self.assertNotIn("rayd/__init__.py", self.names(wheel))

    def test_backend_files_are_disjoint(self):
        drjit = {name for name in self.names(self.drjit_wheel) if name.startswith("rayd/")}
        torch = {name for name in self.names(self.torch_wheel) if name.startswith("rayd/")}
        self.assertTrue(drjit)
        self.assertTrue(torch)
        self.assertFalse(drjit & torch)
        self.assertTrue(all(name.startswith("rayd/drjit/") for name in drjit))
        self.assertTrue(all(name.startswith("rayd/torch/") for name in torch))


if __name__ == "__main__":
    unittest.main()
