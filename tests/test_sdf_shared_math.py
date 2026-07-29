# Copyright Xingyu Chen.
# Tests sdf shared math.

"""Checks shared SDF math structure and numerical behavior."""

import json
import os
import platform
import re
import subprocess
import unittest
from pathlib import Path

from tests.test_rt_host_compile import _msvc_environment


ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "include"
SDF_HEADER = ROOT / "src" / "sdf_device.cuh"
SMOKE_TU = ROOT / "tests" / "native" / "sdf_shared_math_smoke.cpp"
PTX_SOURCES = ROOT / "drjit" / "ptx_sources.json"

# A backend-neutral shared header may name no backend runtime, no OptiX, no CUDA
# allocation or stream API, and no device-only qualifier outside the shared
# RAYD_* macros.
FORBIDDEN_TOKENS = (
    "at::Tensor",
    "torch/",
    "drjit",
    "nanobind",
    "optix",
    "cudaStream",
    "cudaMalloc",
    "cudaFree",
    "__global__",
    "__forceinline__",
)

# The CUDA `float3` vector type is forbidden too (the headers speak
# `math::Vec3f`), matched as a token so the word "float32" is not a false hit,
# exactly as `tests/test_rt_host_compile.py` matches it.
FLOAT3_TOKEN = re.compile(r"float3(?![0-9A-Za-z_])")


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
    def test_consolidated_private_header_exists(self):
        self.assertTrue(SDF_HEADER.is_file())

    def test_headers_are_backend_neutral(self):
        for path in (SDF_HEADER,):
            source = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_TOKENS:
                self.assertNotIn(forbidden, source, f"{forbidden} in {path.relative_to(ROOT)}")
            self.assertIsNone(FLOAT3_TOKEN.search(source), f"float3 in {path.relative_to(ROOT)}")

    def test_headers_spell_the_shared_host_device_qualifier(self):
        for path in (SDF_HEADER,):
            source = path.read_text(encoding="utf-8")
            self.assertIn("<rayd/math.h>", source)
            self.assertIn("RAYD_HOST_DEVICE", source)
            # `__device__` must only ever arrive through the shared macro.
            self.assertNotIn("__device__ ", source)

    def test_grid_and_trace_surfaces_are_complete(self):
        grid = SDF_HEADER.read_text(encoding="utf-8")
        for symbol in ("grid_cells", "grid_coord", "base_index", "trilinear_cell", "sample_cell", "local_gradient"):
            self.assertIn(symbol, grid)
        trace = SDF_HEADER.read_text(encoding="utf-8")
        for symbol in (
            "make_placement",
            "world_to_local_point",
            "world_to_local_direction",
            "local_to_world_direction",
            "normalize_floor",
            "clip_ray_to_box",
            "resolve_eps_hit",
            "bisect_bracket",
            "sphere_trace",
        ):
            self.assertIn(symbol, trace)

    def test_trace_header_declares_the_adr0037_constants(self):
        trace = SDF_HEADER.read_text(encoding="utf-8")
        for declaration in (
            "kSdfEpsNorm = 1.0e-12f",
            "kSdfEpsParallel = 1.0e-7f",
            "kSdfEpsHitVoxelFraction = 1.0e-3f",
            "kSdfDefaultRelaxation = 0.9f",
            "kSdfDefaultMaxSteps = 64",
            "kSdfBisectionSteps = 32",
        ):
            self.assertIn(declaration, trace)

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
