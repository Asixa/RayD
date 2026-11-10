# Copyright Xingyu Chen.
# Configures the Torch backend with CMake to test its CUDA architecture selection rules.

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
NVCC = shutil.which("nvcc")
NINJA = shutil.which("ninja")
LEGACY_VARIABLES = (
    "RAYD_CUDA_ARCHS",
    "CMAKE_CUDA_ARCHITECTURES",
    "TORCH_CUDA_ARCH_LIST",
    "CUDAARCHS",
    "RAYD_TORCH_CUDA_GENCODE_FLAGS",
)


def _cmake_is_new_enough() -> bool:
    if CMAKE is None:
        return False
    output = subprocess.run([CMAKE, "--version"], capture_output=True, text=True, check=False).stdout
    match = re.search(r"cmake version (\d+)\.(\d+)", output)
    return match is not None and (int(match.group(1)), int(match.group(2))) >= (3, 26)


def _torch_has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.version.cuda is not None


@unittest.skipUnless(
    _cmake_is_new_enough() and NVCC and NINJA and _torch_has_cuda(),
    "CMake 3.26+, Ninja, nvcc, and a CUDA torch are required",
)
class TorchArchitectureSelectionTests(unittest.TestCase):
    """The rules fail during configure, before Torch's own CMake package is read, so each case runs in seconds."""

    def configure(self, build: Path, **variables: str) -> subprocess.CompletedProcess:
        env = {name: value for name, value in os.environ.items() if name not in LEGACY_VARIABLES}
        env.update(variables)
        return subprocess.run(
            [CMAKE, "-S", str(ROOT / "torch"), "-B", str(build), "-G", "Ninja", f"-DPython_EXECUTABLE={sys.executable}"]
            + ["-DRAYD_TORCH_INSTALL_SOURCE_BUNDLE=OFF"],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )

    def test_rayd_cuda_archs_refuses_an_explicit_cmake_architecture_list(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.configure(Path(directory), RAYD_CUDA_ARCHS="86", CMAKE_CUDA_ARCHITECTURES="86")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        output = " ".join((result.stdout + result.stderr).split())
        self.assertIn("Set either RAYD_CUDA_ARCHS=86 or the release-only architecture variables, not both", output)
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=86", output)

    def test_rayd_cuda_archs_refuses_torch_cuda_arch_list(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.configure(Path(directory), RAYD_CUDA_ARCHS="86", TORCH_CUDA_ARCH_LIST="8.6")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.6", " ".join((result.stdout + result.stderr).split()))

    def test_default_build_without_a_gpu_asks_for_rayd_cuda_archs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.configure(Path(directory), CUDA_VISIBLE_DEVICES="")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        output = " ".join((result.stdout + result.stderr).split())
        self.assertIn("CMake detected no NVIDIA GPU on this machine. Set RAYD_CUDA_ARCHS", output)


if __name__ == "__main__":
    unittest.main()
