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
        completed = subprocess.run(
            [str(python), *map(str, args)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            self.fail(f"command failed: {completed.args}\n{completed.stdout}\n{completed.stderr}")

    def probe(self, python, expected, absent=()):
        env = os.environ.copy()
        env["RAYD_ACCEPTANCE_BASE_SITE"] = self.base_site
        env["RAYD_ACCEPTANCE_EXPECTED"] = json.dumps(expected)
        env["RAYD_ACCEPTANCE_ABSENT"] = json.dumps(absent)
        completed = subprocess.run(
            [str(python), "-c", PROBE],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
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
        self.command(
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            self.wheels[backend],
        )

    def test_both_install_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, order in enumerate((("drjit", "torch"), ("torch", "drjit"))):
                python = self.make_venv(root / f"order-{index}")
                for backend in order:
                    self.install(python, backend)
                self.probe(python, order)

    def test_meta_distribution_installs_both_backends_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = self.make_venv(Path(tmp) / "meta", system_site_packages=True)
            self.command(
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                self.meta_wheel.parent,
                self.meta_wheel,
            )
            self.probe(python, ("drjit", "torch"))
            self.command(python, "-m", "pip", "uninstall", "-y", "rayd")
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
