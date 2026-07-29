# Copyright Xingyu Chen.
# Tests wheel install matrix.

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


PROBE = r"""
import importlib
import json
import os
import sys
import sysconfig
from pathlib import Path

sys.path.append(os.environ["RAYD_ACCEPTANCE_BASE_SITE"])
import rayd

# Only inspect packages installed in this clean venv.  The base site path is
# present solely to supply heavyweight runtime dependencies such as Torch.
assert rayd.__file__ is None, rayd.__file__
rayd.__path__ = [str(Path(sysconfig.get_path("purelib")) / "rayd")]
expected = json.loads(os.environ["RAYD_ACCEPTANCE_EXPECTED"])
absent = json.loads(os.environ["RAYD_ACCEPTANCE_ABSENT"])
for backend in expected:
    module = importlib.import_module(f"rayd.{backend}")
    assert module.backend_capabilities()["backend"] == backend
for backend in absent:
    try:
        importlib.import_module(f"rayd.{backend}")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"rayd.{backend} survived wheel removal")
"""


class WheelInstallMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = os.environ.get("RAYD_META_WHEEL")
        drjit = os.environ.get("RAYD_DRJIT_WHEEL")
        torch = os.environ.get("RAYD_TORCH_WHEEL")
        if not meta or not drjit or not torch:
            raise unittest.SkipTest("wheel paths are provided by the packaging job")
        cls.meta_wheel = Path(meta).resolve()
        cls.wheels = {"drjit": Path(drjit).resolve(), "torch": Path(torch).resolve()}
        cls.base_site = sysconfig.get_path("purelib")

    @staticmethod
    def venv_python(root):
        if os.name == "nt":
            return root / "Scripts" / "python.exe"
        return root / "bin" / "python"

    def command(self, python, *args):
        completed = subprocess.run([str(python), *map(str, args)], text=True, capture_output=True, check=False)
        if completed.returncode:
            self.fail(f"command failed: {completed.args}\n{completed.stdout}\n{completed.stderr}")

    def probe(self, python, expected, absent=()):
        env = os.environ.copy()
        env["RAYD_ACCEPTANCE_BASE_SITE"] = self.base_site
        env["RAYD_ACCEPTANCE_EXPECTED"] = json.dumps(expected)
        env["RAYD_ACCEPTANCE_ABSENT"] = json.dumps(absent)
        completed = subprocess.run([str(python), "-c", PROBE], env=env, text=True, capture_output=True, check=False)
        if completed.returncode:
            self.fail(f"wheel probe failed\n{completed.stdout}\n{completed.stderr}")

    def make_venv(self, root, system_site_packages=False):
        args = [sys.executable, "-m", "venv"]
        if system_site_packages:
            args.append("--system-site-packages")
        args.append(root)
        self.command(*args)
        return self.venv_python(root)

    def install(self, python, backend):
        self.command(python, "-m", "pip", "install", "--no-deps", "--ignore-installed", self.wheels[backend])

    def install_legacy_editable(self, python, root):
        package = root / "rayd"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("LEGACY = True\n", encoding="utf-8")
        finder = f"""\
import importlib.abc
import importlib.util
import sys

class LegacyRayDFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rayd":
            return importlib.util.spec_from_file_location(
                fullname,
                {str(package / "__init__.py")!r},
                submodule_search_locations=[{str(package)!r}],
            )
        return None

sys.meta_path.insert(0, LegacyRayDFinder())
"""
        seed = f"""\
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_path("purelib"))
(site / "_rayd_editable.py").write_text({finder!r}, encoding="utf-8")
(site / "_rayd_editable.pth").write_text("import _rayd_editable\\n", encoding="utf-8")
info = site / "rayd-0.4.1.dist-info"
info.mkdir()
(info / "METADATA").write_text(
    "Metadata-Version: 2.1\\nName: rayd\\nVersion: 0.4.1\\n",
    encoding="utf-8",
)
(info / "WHEEL").write_text(
    "Wheel-Version: 1.0\\nGenerator: legacy-fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
    encoding="utf-8",
)
(info / "INSTALLER").write_text("pip\\n", encoding="utf-8")
(info / "RECORD").write_text(
    "_rayd_editable.pth,,\\n"
    "_rayd_editable.py,,\\n"
    "rayd-0.4.1.dist-info/INSTALLER,,\\n"
    "rayd-0.4.1.dist-info/METADATA,,\\n"
    "rayd-0.4.1.dist-info/RECORD,,\\n"
    "rayd-0.4.1.dist-info/WHEEL,,\\n",
    encoding="utf-8",
)
"""
        self.command(python, "-c", seed)

    def test_both_install_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, order in enumerate((("drjit", "torch"), ("torch", "drjit"))):
                python = self.make_venv(root / f"order-{index}")
                for backend in order:
                    self.install(python, backend)
                self.probe(python, order)

    def test_meta_distribution_coexists_with_both_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = self.make_venv(Path(tmp) / "meta", system_site_packages=True)
            self.install(python, "drjit")
            self.install(python, "torch")
            self.command(
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                self.meta_wheel.parent,
                "--no-deps",
                self.meta_wheel,
            )
            self.probe(python, ("drjit", "torch"))
            self.command(python, "-m", "pip", "uninstall", "-y", "rayd")
            self.probe(python, ("drjit", "torch"))

    def test_meta_upgrade_removes_legacy_editable_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = self.make_venv(root / "upgrade", system_site_packages=True)
            self.install(python, "drjit")
            self.install(python, "torch")
            self.install_legacy_editable(python, root / "legacy")
            self.command(python, "-c", "import rayd; assert rayd.__file__ is not None; assert rayd.LEGACY")
            self.command(
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                self.meta_wheel.parent,
                "--no-deps",
                "--upgrade",
                self.meta_wheel,
            )
            self.probe(python, ("drjit", "torch"))

    def test_uninstalling_one_backend_preserves_the_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = self.make_venv(Path(tmp) / "uninstall")
            self.install(python, "drjit")
            self.install(python, "torch")
            self.command(python, "-m", "pip", "uninstall", "-y", "rayd-drjit")
            self.probe(python, ("torch",), ("drjit",))
            self.install(python, "drjit")
            self.command(python, "-m", "pip", "uninstall", "-y", "rayd-torch")
            self.probe(python, ("drjit",), ("torch",))


if __name__ == "__main__":
    unittest.main()
