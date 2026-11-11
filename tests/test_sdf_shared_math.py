# Copyright Xingyu Chen.
# Tests sdf shared math.

"""Checks shared SDF math structure and numerical behavior."""

import json
import os
import platform
import subprocess
import unittest
from pathlib import Path

from tests.test_rt_host_compile import _msvc_environment


ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "include"
SDF_HEADER = ROOT / "src" / "sdf_device.cuh"
SMOKE_TU = ROOT / "tests" / "native" / "sdf_shared_math_smoke.cpp"
PTX_SOURCES = ROOT / "drjit" / "ptx_sources.json"


def locate_cl(env):
    """Checks shared SDF math structure and numerical behavior."""
    path = next((value for key, value in env.items() if key.upper() == "PATH"), "")
    for directory in path.split(os.pathsep):
        candidate = Path(directory) / "cl.exe"
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


class SdfSharedHeaderTests(unittest.TestCase):
    def test_headers_stay_out_of_every_committed_ptx_include_closure(self):
        # Reaching a PTX module's closure would silently stale the committed
        # `*_ptx.h` headers (ADR-0037 section 9, repository PTX identity rule).
        record = json.loads(PTX_SOURCES.read_text(encoding="utf-8"))
        closure = {source for module in record["modules"].values() for source in module["sources"]}
        for path in (SDF_HEADER,):
            relative = path.relative_to(ROOT).as_posix()
            self.assertNotIn(relative, closure)


@unittest.skipUnless(platform.system() == "Windows", "host gate uses MSVC cl.exe")
class SdfHostSmokeTests(unittest.TestCase):
    def test_smoke_translation_unit_compiles_and_passes_host_only(self):
        env = _msvc_environment()
        if env is None:
            self.skipTest("MSVC toolchain (vswhere / VsDevCmd) not found")
        cl = locate_cl(env)
        if cl is None:
            self.skipTest("cl.exe not on the MSVC PATH")

        # The headers are backend-neutral, so the artifacts land in the ignored
        # top-level build tree rather than inside either backend.
        out_dir = ROOT / "artifacts" / "sdf_host_compile"
        out_dir.mkdir(parents=True, exist_ok=True)
        executable = out_dir / "sdf_shared_math_smoke.exe"
        compile_cmd = [
            cl,
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/W3",
            f"/I{SHARED_INCLUDE}",
            f"/I{ROOT}",
            str(SMOKE_TU),
            f"/Fo{out_dir}\\",
            f"/Fe{executable}",
        ]
        built = subprocess.run(compile_cmd, cwd=str(out_dir), env=env, capture_output=True, text=True, check=False)
        self.assertEqual(
            built.returncode,
            0,
            f"host compile failed.\nCMD: {' '.join(compile_cmd)}\nSTDOUT:\n{built.stdout}\nSTDERR:\n{built.stderr}",
        )

        # Each check in the smoke TU owns a distinct exit code, so a failure
        # names the numerical claim that broke.
        ran = subprocess.run([str(executable)], cwd=str(out_dir), capture_output=True, text=True, check=False)
        self.assertEqual(
            ran.returncode,
            0,
            f"sdf_shared_math_smoke check #{ran.returncode} failed.\nSTDOUT:\n{ran.stdout}\nSTDERR:\n{ran.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
