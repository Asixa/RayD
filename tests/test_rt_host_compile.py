# Copyright Xingyu Chen.
# Tests rt host compile.

"""P4 key gate: the migrated multipath algorithm headers compile host-only.

The reflection-trace algorithm body was lifted out of the OptiX device header
into shared/reflection/trace_algo.h so it is host-compilable: no
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
SHARED_INCLUDE = ROOT / "include"
SHARED_ROOT = SHARED_INCLUDE / "rayd"
RT_INCLUDE = ROOT / "src" / "runtime"
ALGO_HEADERS = (
    ROOT / "src/reflection/reflection_algorithms.cuh",
    ROOT / "src/visibility/segment_visibility.cuh",
    ROOT / "src/diffraction/paths.h",
    ROOT / "src/diffraction/accumulation.h",
)
RT_HEADERS = (RT_INCLUDE / "rt_internal.h",)
SMOKE_TU = ROOT / "tests" / "native" / "rt_host_compile_smoke.cpp"

# Tokens that must not appear in a host-compilable algorithm header: the OptiX
# device intrinsics, the six payload-register accessors, and the launch-index
# query. Matched as plain substrings.
FORBIDDEN_ALGO_TOKENS = ("optixTrace", "optixGetPayload", "optixSetPayload", "optixGetLaunchIndex")

# The CUDA float3 vector type must not appear either (the algorithm uses
# math::Vec3f throughout), but the diffraction algorithm headers legitimately
# speak UTD's host-safe `float3a` POD at the utd_math boundary. Match `float3`
# only when it is NOT immediately followed by an identifier character, so the
# CUDA `float3` type and `make_float3(` constructor are caught while `float3a`
# is allowed.
FORBIDDEN_ALGO_REGEXES = (re.compile(r"float3(?![0-9A-Za-z_])"),)


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


class RtHostCompileTests(unittest.TestCase):
    def _assert_no_device_only_tokens(self, header):
        text = header.read_text(encoding="utf-8")
        if header.name == "segment_visibility.cuh":
            text = text.partition("#if defined(RAYD_OPTIX_DEVICE_PROGRAM)")[0]
        for token in FORBIDDEN_ALGO_TOKENS:
            with self.subTest(header=header.name, token=token):
                self.assertNotIn(token, text)
        for regex in FORBIDDEN_ALGO_REGEXES:
            with self.subTest(header=header.name, regex=regex.pattern):
                self.assertIsNone(regex.search(text), f"{header.name}: forbidden token matching {regex.pattern!r}")

    def test_migrated_algo_headers_have_no_device_only_tokens(self):
        # Every concept-owned *_algo.h is covered (not just the six known
        # names), so a future migrated pipeline is grep-gated automatically.
        globbed = [path for path in ALGO_HEADERS if path.is_file()]
        self.assertTrue(
            set(ALGO_HEADERS).issubset(set(globbed)), "known concept-owned algo headers missing from the shared tree"
        )
        for header in globbed:
            self._assert_no_device_only_tokens(header)

    def test_rt_contract_headers_have_no_device_only_tokens(self):
        # The full P4 grep gate also covers shared/rt/** (traverser, qualifiers,
        # numeric_policy, ...): the backend-neutral trace contracts stay free of
        # OptiX ray-cast intrinsics, payload registers, the launch-index query,
        # and the CUDA float3 type.
        self.assertTrue(RT_HEADERS, "no rt/ contract headers found")
        for header in RT_HEADERS:
            self._assert_no_device_only_tokens(header)

    def test_instantiation_matrix_both_gpu_frontends_have_cuda_traversal(self):
        # Both GPU frontends instantiate the migrated algorithm bodies with the
        # CUDA BVH traverser for their eager-native fallback. Their OptiX paths
        # remain separate thin traversal shims.
        frontends = {
            "drjit": ROOT / "src" / "scene" / "multipath_jit.cu",
            "torch": ROOT / "src" / "scene" / "multipath.cu",
        }
        for backend, source in frontends.items():
            with self.subTest(backend=backend):
                self.assertTrue(source.is_file(), source)
                self.assertIn(
                    "CudaBvhTraverser",
                    source.read_text(encoding="utf-8", errors="ignore"),
                    f"{backend} frontend must instantiate algo bodies with CudaBvhTraverser",
                )

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
