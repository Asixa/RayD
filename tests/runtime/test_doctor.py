# Copyright Xingyu Chen.
# Tests the backend doctors and the warning for the automatic fallback from OptiX to the CUDA trace backend.

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DRJIT = importlib.util.find_spec("drjit") is not None
COMPILER = shutil.which("cc") or shutil.which("gcc")

# Stands in for a driver whose OptiX library predates every ABI RayD requests.
OLD_DRIVER_OPTIX = (
    "int optixQueryFunctionTable(int abi, unsigned int count, void* keys, void* values, void* table,"
    " unsigned long size) { return 7801; }\n"
)


def _cuda_available() -> bool:
    if not HAS_TORCH:
        return False
    import torch

    return torch.cuda.is_available()


def _run(arguments, **environment) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        [sys.executable, *arguments], env=env, text=True, capture_output=True, timeout=900, check=False
    )


def _assert_exit_matches_verdict(test: unittest.TestCase, result: subprocess.CompletedProcess) -> None:
    passed = "Result: all checks passed." in result.stdout
    test.assertIn("Result:", result.stdout, result.stderr)
    test.assertEqual(result.returncode, 0 if passed else 1, result.stdout + result.stderr)


class _FakeOptix:
    """Directories holding a libnvoptix.so.1 that the loader rejects or that implements no OptiX ABI."""

    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.unloadable = root / "unloadable"
        self.unloadable.mkdir()
        (self.unloadable / "libnvoptix.so.1").write_bytes(b"not a shared library")
        self.old_driver = None
        if COMPILER and sys.platform.startswith("linux"):
            source = root / "optix.c"
            source.write_text(OLD_DRIVER_OPTIX)
            target = root / "old_driver"
            target.mkdir()
            built = subprocess.run(
                [COMPILER, "-shared", "-fPIC", "-o", str(target / "libnvoptix.so.1"), str(source)],
                capture_output=True,
                check=False,
            )
            if built.returncode == 0:
                self.old_driver = target

    def library_path(self, directory: Path) -> str:
        return os.pathsep.join(part for part in (str(directory), os.environ.get("LD_LIBRARY_PATH", "")) if part)


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TorchDoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = _FakeOptix()

    @classmethod
    def tearDownClass(cls):
        cls.fake.directory.cleanup()

    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_reports_the_machine(self):
        result = _run(["-m", "rayd.torch.doctor"])
        _assert_exit_matches_verdict(self, result)
        for expected in (
            "rayd-torch doctor",
            "  OS:",
            "  Python:",
            "  driver:",
            "  GPU 0:",
            "built against torch:",
            "  torch:",
            "[  OK] native libraries",
            "OptiX library",
            "driver offers OptiX ABI:",
            "auto trace backend:",
            "auto edge backend:         cuda",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stdout)

    def test_no_visible_gpu_fails_with_the_fix(self):
        result = _run(["-m", "rayd.torch.doctor"], CUDA_VISIBLE_DEVICES="")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL] CUDA device:", result.stdout)
        self.assertIn("--gpus all", result.stdout)
        self.assertIn("[SKIP] auto backends", result.stdout)

    def test_unloadable_native_library_fails_and_says_why(self):
        with tempfile.TemporaryDirectory() as directory:
            garbage = Path(directory) / "_legacy_ops.so"
            garbage.write_bytes(b"not a shared library")
            result = _run(["-m", "rayd.torch.doctor"], RAYD_TORCH_LEGACY_LIBRARY=str(garbage))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(
            "[FAIL] native libraries: RayD Torch could not load its native library _legacy_ops", result.stdout
        )
        self.assertIn("[SKIP] auto backends: not resolved: the native libraries did not load", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "the OptiX library is replaced through LD_LIBRARY_PATH")
    def test_unloadable_optix_library_fails_with_the_docker_hint(self):
        library_path = self.fake.library_path(self.fake.unloadable)
        result = _run(["-m", "rayd.torch.doctor"], LD_LIBRARY_PATH=library_path)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL] OptiX library", result.stdout)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility", result.stdout)
        if _cuda_available():
            self.assertIn("auto trace backend:        cuda", result.stdout)
            self.assertIn("[FAIL] trace backend: OptiX is unavailable", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "the OptiX library is replaced through LD_LIBRARY_PATH")
    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_disable_optix_set_to_zero_does_not_excuse_a_missing_optix(self):
        # RAYD_DISABLE_OPTIX=0 leaves OptiX enabled in the native runtime, so the doctor must not report the CUDA
        # fallback as intended.
        library_path = self.fake.library_path(self.fake.unloadable)
        result = _run(["-m", "rayd.torch.doctor"], LD_LIBRARY_PATH=library_path, RAYD_DISABLE_OPTIX="0")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL] trace backend: OptiX is unavailable", result.stdout)
        self.assertNotIn("because RAYD_DISABLE_OPTIX is set", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "the OptiX library is replaced through LD_LIBRARY_PATH")
    def test_driver_without_the_build_abi_fails_the_abi_check(self):
        if self.fake.old_driver is None:
            self.skipTest("no C compiler to build the stand-in OptiX library")
        result = _run(["-m", "rayd.torch.doctor"], LD_LIBRARY_PATH=self.fake.library_path(self.fake.old_driver))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL] OptiX ABI: the build requests", result.stdout)
        if _cuda_available():
            self.assertIn("[FAIL] trace backend: OptiX is unavailable", result.stdout)
            self.assertIn("older than the OptiX ABI", result.stderr)


@unittest.skipUnless(_cuda_available() and sys.platform.startswith("linux"), "CUDA on Linux is required")
class TorchOptixFallbackWarningTests(unittest.TestCase):
    """`trace_backend='auto'` without OptiX warns once with the reason; an explicit OptiX request raises it.

    The warning is a Torch C++ warning raised while the native scene is created, so it reaches stderr through
    Torch's warning handler rather than Python's warnings module.
    """

    BODY = """
        import torch
        import rayd.torch as rt

        def build(**kwargs):
            vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
            faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
            scene = rt.Scene(**kwargs)
            scene.add_mesh(rt.Mesh(vertices, faces))
            scene.build()
            return scene

        print("BACKENDS", build().trace_backend, build().trace_backend)
        try:
            build(trace_backend="optix")
        except RuntimeError as error:
            print("EXPLICIT", error)
    """
    WARNING = "RayD: OptiX is unavailable on CUDA device"

    @classmethod
    def setUpClass(cls):
        cls.fake = _FakeOptix()

    @classmethod
    def tearDownClass(cls):
        cls.fake.directory.cleanup()

    def test_missing_optix_library_warns_once_with_the_docker_hint(self):
        result = _run(["-c", textwrap.dedent(self.BODY)], LD_LIBRARY_PATH=self.fake.library_path(self.fake.unloadable))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKENDS cuda cuda", result.stdout)
        self.assertEqual(result.stderr.count(self.WARNING), 1, result.stderr)
        for expected in (
            "trace_backend='auto' uses the slower CUDA trace backend",
            "OPTIX_ERROR_LIBRARY_NOT_FOUND",
            "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility",
            "python -m rayd.torch.doctor",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stderr)
        self.assertIn("EXPLICIT OptiX runtime is unavailable (OPTIX_ERROR_LIBRARY_NOT_FOUND)", result.stdout)

    def test_old_driver_names_the_driver_requirement(self):
        if self.fake.old_driver is None:
            self.skipTest("no C compiler to build the stand-in OptiX library")
        result = _run(["-c", textwrap.dedent(self.BODY)], LD_LIBRARY_PATH=self.fake.library_path(self.fake.old_driver))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKENDS cuda cuda", result.stdout)
        self.assertEqual(result.stderr.count(self.WARNING), 1, result.stderr)
        self.assertIn("OPTIX_ERROR_UNSUPPORTED_ABI_VERSION", result.stderr)
        self.assertIn("the NVIDIA driver is older than the OptiX ABI", result.stderr)
        self.assertIn("this driver supports CUDA", result.stderr)
        self.assertIn("EXPLICIT OptiX runtime is unavailable (OPTIX_ERROR_UNSUPPORTED_ABI_VERSION)", result.stdout)

    def test_disabling_optix_is_silent(self):
        result = _run(["-c", textwrap.dedent(self.BODY)], RAYD_DISABLE_OPTIX="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKENDS cuda cuda", result.stdout)
        self.assertNotIn(self.WARNING, result.stderr)


@unittest.skipUnless(HAS_DRJIT and _cuda_available() and sys.platform.startswith("linux"), "Dr.Jit CUDA on Linux")
class DrJitOptixFallbackWarningTests(unittest.TestCase):
    """`trace_backend='auto'` without OptiX warns once on stderr with the reason, like the Torch backend."""

    BODY = """
        import drjit.cuda as cuda
        import rayd.drjit as pj

        def build():
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                                   cuda.Array3i([0], [1], [2])))
            scene.build()
            return scene

        print("BACKENDS", build().trace_backend, build().trace_backend)
    """
    WARNING = "RayD: OptiX is unavailable on CUDA device"

    @classmethod
    def setUpClass(cls):
        cls.fake = _FakeOptix()

    @classmethod
    def tearDownClass(cls):
        cls.fake.directory.cleanup()

    def test_missing_optix_library_warns_once_with_the_docker_hint(self):
        result = _run(["-c", textwrap.dedent(self.BODY)], LD_LIBRARY_PATH=self.fake.library_path(self.fake.unloadable))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKENDS cuda cuda", result.stdout)
        self.assertEqual(result.stderr.count(self.WARNING), 1, result.stderr)
        for expected in (
            "trace_backend='auto' uses the slower CUDA trace backend",
            "OPTIX_ERROR_LIBRARY_NOT_FOUND",
            "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility",
            "python -m rayd.drjit.doctor",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stderr)

    def test_old_driver_names_the_driver_requirement(self):
        if self.fake.old_driver is None:
            self.skipTest("no C compiler to build the stand-in OptiX library")
        result = _run(["-c", textwrap.dedent(self.BODY)], LD_LIBRARY_PATH=self.fake.library_path(self.fake.old_driver))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKENDS cuda cuda", result.stdout)
        self.assertEqual(result.stderr.count(self.WARNING), 1, result.stderr)
        self.assertIn("OPTIX_ERROR_UNSUPPORTED_ABI_VERSION", result.stderr)
        self.assertIn("the NVIDIA driver is older than the OptiX ABI", result.stderr)

    def test_disabling_optix_is_silent(self):
        result = _run(["-c", textwrap.dedent(self.BODY)], RAYD_DISABLE_OPTIX="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKENDS cuda cuda", result.stdout)
        self.assertNotIn(self.WARNING, result.stderr)

    def test_available_optix_is_silent(self):
        result = _run(["-c", textwrap.dedent(self.BODY)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if "BACKENDS optix optix" not in result.stdout:
            self.skipTest("this machine has no usable OptiX runtime")
        self.assertNotIn(self.WARNING, result.stderr)


@unittest.skipUnless(HAS_DRJIT, "drjit is required")
class DrJitDoctorTests(unittest.TestCase):
    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_reports_the_machine(self):
        result = _run(["-m", "rayd.drjit.doctor"])
        _assert_exit_matches_verdict(self, result)
        for expected in (
            "rayd-drjit doctor",
            "  driver:",
            "  GPU 0:",
            "built against drjit:",
            "[  OK] native extension",
            "OptiX ABI:",
            "auto trace backend:",
            "auto edge backend:",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stdout)

    def test_no_visible_gpu_fails_with_the_fix(self):
        result = _run(["-m", "rayd.drjit.doctor"], CUDA_VISIBLE_DEVICES="")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL] CUDA device:", result.stdout)
        self.assertIn("[SKIP] auto backends", result.stdout)

    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_disable_optix_set_to_zero_counts_as_unset(self):
        result = _run(["-m", "rayd.drjit.doctor"], RAYD_DISABLE_OPTIX="0")
        self.assertNotIn("disabled by RAYD_DISABLE_OPTIX", result.stdout)
        self.assertNotIn("because RAYD_DISABLE_OPTIX is set", result.stdout)
        self.assertIn("auto trace backend:", result.stdout)


if __name__ == "__main__":
    unittest.main()
