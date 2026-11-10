# Copyright Xingyu Chen.
# Tests that a rayd-torch native library that does not load is explained at import and on first use.

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _run(body: str, **environment: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-W", "always::RuntimeWarning", "-c", textwrap.dedent(body)],
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


@unittest.skipUnless(HAS_TORCH, "torch is required")
class NativeLoadFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.garbage = Path(cls.directory.name) / "_legacy_ops.so"
        cls.garbage.write_bytes(b"this is not a shared library")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_unloadable_legacy_library_warns_at_import_and_raises_on_first_use(self):
        result = _run(
            """
            import warnings
            import torch
            import rayd.torch as rt
            from rayd._impl import runtime

            assert not rt._NATIVE_AVAILABLE
            assert isinstance(runtime.LOAD_ERROR, ImportError), type(runtime.LOAD_ERROR)
            assert isinstance(runtime.LOAD_ERROR.__cause__, OSError), runtime.LOAD_ERROR.__cause__
            # The import stays usable for everything that needs no native code.
            assert rt.api_manifest()["backend"] == "torch"
            print(runtime.LOAD_ERROR)
            if torch.cuda.is_available():
                vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
                scene = rt.Scene()
                scene.add_mesh(rt.Mesh(vertices, torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")))
                try:
                    scene.build()
                except RuntimeError as error:
                    print("RAISED", type(error.__cause__).__name__, error.__cause__ is runtime.LOAD_ERROR)
                    # The diagnosis is in the message itself, not only in the chained cause.
                    print("MESSAGE CARRIES CAUSE", str(runtime.LOAD_ERROR) in str(error))
                else:
                    raise AssertionError("build() ran without the native dispatcher")
            """,
            RAYD_TORCH_LEGACY_LIBRARY=str(self.garbage),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RuntimeWarning: RayD Torch could not load its native library _legacy_ops.so", result.stderr)
        import torch

        if torch.cuda.is_available():
            self.assertIn("RAISED ImportError True", result.stdout)
            self.assertIn("MESSAGE CARRIES CAUSE True", result.stdout)
        for expected in (
            f"from {self.garbage.resolve()}",
            "Installed torch:",
            "Cause:",
            "--index-url https://download.pytorch.org/whl/cu",
            "RAYD_TORCH_LEGACY_LIBRARY selects",
            "python -m rayd.torch.doctor",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stdout)

    def test_override_never_falls_back_to_the_installed_library(self):
        result = _run(
            """
            import rayd.torch
            from rayd._impl import runtime
            print("AVAILABLE", runtime.AVAILABLE)
            print(runtime.LOAD_ERROR)
            """,
            RAYD_TORCH_LEGACY_LIBRARY=str(Path(self.directory.name) / "missing" / "_legacy_ops.so"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AVAILABLE False", result.stdout)
        self.assertIn("FileNotFoundError", result.stdout)
        self.assertIn("does not exist", result.stdout)

    def test_unloadable_stable_library_explains_the_camera_failure(self):
        garbage = Path(self.directory.name) / "_stable_ops.so"
        garbage.write_bytes(b"not a library either")
        result = _run(
            """
            import rayd.torch
            from rayd._impl import runtime
            try:
                runtime.camera_ops()
            except RuntimeError as error:
                print(error)
            else:
                raise AssertionError("camera_ops() ran without the stable library")
            """,
            RAYD_TORCH_STABLE_LIBRARY=str(garbage),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RayD Torch stable ABI operators are unavailable: RayD Torch could not load", result.stdout)
        self.assertIn("_stable_ops.so", result.stdout)


if __name__ == "__main__":
    unittest.main()
