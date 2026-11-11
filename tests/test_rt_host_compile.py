# Copyright Xingyu Chen.
# Tests rt host compile.

"""P4 key gate: the migrated multipath algorithm headers compile host-only.

A POSIX C++17 compile of the shared math header, and POSIX and MSVC host
compiles of tests/native/rt_host_compile_smoke.cpp, with no CUDA/OptiX device
compiler.
"""

import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "include"
SMOKE_TU = ROOT / "tests" / "native" / "rt_host_compile_smoke.cpp"


def _vswhere_install_path():
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    result = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    path = result.stdout.strip()
    return path or None


def _msvc_environment():
    """Return os.environ augmented with the x64 MSVC toolchain env, or None."""
    install_path = _vswhere_install_path()
    if not install_path:
        return None
    vsdevcmd = Path(install_path) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not vsdevcmd.is_file():
        return None
    # Pass one raw command string so Windows hands it to cmd verbatim; a list
    # would make subprocess re-quote the inner `call "..."` and break it.
    result = subprocess.run(
        f'cmd.exe /d /s /c call "{vsdevcmd}" -arch=x64 -host_arch=x64 >nul && set',
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return env or None


def _cuda_include_dir():
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and (Path(cuda_path) / "include").is_dir():
        return Path(cuda_path) / "include"
    return None


def posix_compiler():
    """Return the first POSIX C++ compiler on PATH, or None."""
    return next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)


def _posix_cuda_include_dir():
    """Return a directory holding <vector_types.h> from CUDA_PATH, CUDA_HOME, or nvcc, or None."""
    roots = [Path(value) for key in ("CUDA_PATH", "CUDA_HOME") if (value := os.environ.get(key))]
    if nvcc := shutil.which("nvcc"):
        roots.append(Path(nvcc).resolve().parent.parent)
    for root in roots:
        for candidate in (root / "include", root / "targets" / "x86_64-linux" / "include"):
            if (candidate / "vector_types.h").is_file():
                return candidate
    return None


class RtHostCompileTests(unittest.TestCase):
    @unittest.skipIf(platform.system() == "Windows", "POSIX compiler gate runs outside Windows")
    def test_smoke_translation_unit_compiles_with_posix_cxx(self):
        compiler = posix_compiler()
        if compiler is None:
            self.skipTest("no POSIX C++ compiler found")
        cuda_include = _posix_cuda_include_dir()
        if cuda_include is None:
            self.skipTest("no CUDA include directory with <vector_types.h> found")

        with tempfile.TemporaryDirectory() as directory:
            cmd = [
                compiler,
                "-std=c++17",
                "-c",
                f"-I{ROOT}",
                f"-I{SHARED_INCLUDE}",
                f"-I{cuda_include}",
                str(SMOKE_TU),
                "-o",
                str(Path(directory) / "rt_host_compile_smoke.o"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(
            result.returncode,
            0,
            f"POSIX host compile failed.\nCMD: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    @unittest.skipIf(platform.system() == "Windows", "POSIX compiler gate runs outside Windows")
    def test_shared_math_header_compiles_with_posix_cxx(self):
        compiler = posix_compiler()
        if compiler is None:
            self.skipTest("no POSIX C++ compiler found")

        source = r"""
#include <limits>
#include <rayd/math.h>

int main() {
    using rayd::shared::diffraction::Dual;
    using rayd::shared::diffraction::isfinite;
    const float infinity = std::numeric_limits<float>::infinity();
    const Dual finite_primal_infinite_tangent{1.0f, infinity};
    const Dual infinite_primal{infinity, 0.0f};
    return isfinite(1.0f) && isfinite(finite_primal_infinite_tangent) && !isfinite(infinite_primal) ? 0 : 1;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "math_host_compile_smoke"
            compile_result = subprocess.run(
                [compiler, "-std=c++17", f"-I{SHARED_INCLUDE}", "-x", "c++", "-o", str(executable), "-"],
                input=source,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                f"POSIX host compile failed.\nSTDOUT:\n{compile_result.stdout}\nSTDERR:\n{compile_result.stderr}",
            )
            run_result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    @unittest.skipUnless(platform.system() == "Windows", "host-compile gate uses MSVC cl.exe")
    def test_smoke_translation_unit_compiles_host_only(self):
        env = _msvc_environment()
        if env is None:
            self.skipTest("MSVC toolchain (vswhere / VsDevCmd) not found")
        cl = _find_cl(env)
        if cl is None:
            self.skipTest("cl.exe not on the MSVC PATH")
        cuda_include = _cuda_include_dir()
        self.assertIsNotNone(cuda_include, "CUDA_PATH/include needed for <vector_types.h>")

        out_dir = ROOT / "artifacts" / "rt_host_compile"
        out_dir.mkdir(parents=True, exist_ok=True)
        # cl.exe is passed by full path: CreateProcess resolves a bare name
        # against the current process PATH, not the captured MSVC env.
        cmd = [
            cl,
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/c",
            "/W3",
            f"/I{ROOT}",
            f"/I{SHARED_INCLUDE}",
            f"/I{cuda_include}",
            str(SMOKE_TU),
            f"/Fo{out_dir / 'rt_host_compile_smoke.obj'}",
        ]
        result = subprocess.run(cmd, cwd=str(out_dir), env=env, capture_output=True, text=True, check=False)
        self.assertEqual(
            result.returncode,
            0,
            f"host compile failed.\nCMD: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )


def _find_cl(env):
    for directory in env.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / "cl.exe"
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


if __name__ == "__main__":
    unittest.main()
