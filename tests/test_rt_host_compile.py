"""P4 key gate: the migrated multipath algorithm headers compile host-only.

The reflection-trace algorithm body was lifted out of the OptiX device header
into shared/multipath/reflection_trace_algo.h so it is host-compilable: no
optixTrace / payload register / launch-index token, all ray casts routed through
the rt::Traverser concept. This test proves that claim two ways:

* A pure token grep-gate over reflection_trace_algo.h (fast, always runs).
* An actual host compile of tests/native/rt_host_compile_smoke.cpp with the
  MSVC host compiler (cl.exe located via vswhere, mirroring
  scripts/build_local.ps1's Initialize-MSVCEnvironment), no CUDA/OptiX device
  compiler involved.
"""

import os
import platform
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "shared" / "include"
MULTIPATH_INCLUDE = SHARED_INCLUDE / "rayd" / "shared" / "multipath"
ALGO_HEADERS = (
    MULTIPATH_INCLUDE / "reflection_trace_algo.h",
    MULTIPATH_INCLUDE / "segment_visibility_algo.h",
    MULTIPATH_INCLUDE / "reflection_epc_algo.h",
)
SMOKE_TU = ROOT / "tests" / "native" / "rt_host_compile_smoke.cpp"

# Tokens that must not appear in a host-compilable algorithm header: the OptiX
# device intrinsics, the six payload-register accessors, the launch-index query,
# and the CUDA float3 vector type (the algorithm uses math::Vec3f throughout).
FORBIDDEN_ALGO_TOKENS = (
    "optixTrace",
    "optixGetPayload",
    "optixSetPayload",
    "optixGetLaunchIndex",
    "float3",
)


def _vswhere_install_path():
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    result = subprocess.run(
        [
            str(vswhere), "-latest", "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath",
        ],
        capture_output=True, text=True, check=False,
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
        capture_output=True, text=True, check=False,
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


class RtHostCompileTests(unittest.TestCase):
    def test_migrated_algo_headers_have_no_device_only_tokens(self):
        for header in ALGO_HEADERS:
            text = header.read_text(encoding="utf-8")
            for token in FORBIDDEN_ALGO_TOKENS:
                with self.subTest(header=header.name, token=token):
                    self.assertNotIn(token, text)

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

        out_dir = ROOT / "backends" / "drjit" / "build" / "rt_host_compile"
        out_dir.mkdir(parents=True, exist_ok=True)
        # cl.exe is passed by full path: CreateProcess resolves a bare name
        # against the current process PATH, not the captured MSVC env.
        cmd = [
            cl, "/nologo", "/std:c++17", "/EHsc", "/c", "/W3",
            f'/I{SHARED_INCLUDE}',
            f'/I{cuda_include}',
            str(SMOKE_TU),
            f'/Fo{out_dir / "rt_host_compile_smoke.obj"}',
        ]
        result = subprocess.run(cmd, cwd=str(out_dir), env=env,
                                capture_output=True, text=True, check=False)
        self.assertEqual(
            result.returncode, 0,
            f"host compile failed.\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
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
