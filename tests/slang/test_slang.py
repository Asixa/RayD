"""Tests for rayd.slang -- Slang cpp-target interop.

Test categories:
  SlangModuleTests          -- rayd.slang Python module (path helpers, import)
  SlangInteropCompileTests  -- slangc compilation of Slang code calling RayD

The interop tests use slangc to compile Slang -> C++ and verify the generated
code contains the expected rayd::slang::* function calls and type mappings.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

try:
    import slangtorch  # noqa: F401
    SLANGTORCH_AVAILABLE = True
    _SLANGC = str(Path(slangtorch.__file__).parent / "bin" /
                  ("slangc.exe" if sys.platform == "win32" else "slangc"))
except ImportError:
    SLANGTORCH_AVAILABLE = False
    _SLANGC = None


_PREAMBLE = textwrap.dedent("""\
import os, sys
scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
if scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
""")


def run_script(script: str, timeout: int = 300, check: bool = True):
    full = _PREAMBLE + textwrap.dedent(script)
    result = subprocess.run(
        [sys.executable, "-c", full],
        cwd=ROOT, text=True, capture_output=True,
        timeout=timeout, check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Subprocess failed.\nReturn code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


def run_json_case(script: str, timeout: int = 300):
    result = run_script(script, timeout=timeout, check=True)
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    if not lines:
        raise AssertionError(f"No JSON output.\nSTDERR:\n{result.stderr}")
    return json.loads(lines[-1])


def _slangc_compile(slang_file: str, output_file: str, target: str = "cpp"):
    return subprocess.run(
        [_SLANGC, slang_file, "-target", target, "-o", output_file,
         "-I", str(ROOT / "include"), "-ignore-capabilities"],
        cwd=ROOT, text=True, capture_output=True, timeout=60, check=False)


# ---------------------------------------------------------------------------
# Module-level path / import tests
# ---------------------------------------------------------------------------

class SlangModuleTests(unittest.TestCase):

    def test_import_rayd_slang(self):
        import rayd.slang as rs
        self.assertTrue(callable(rs.include_dir))
        self.assertTrue(callable(rs.load_module))

    def test_include_dir_exists(self):
        import rayd.slang as rs
        inc = rs.include_dir()
        self.assertTrue((inc / "rayd" / "slang" / "rayd.slang").is_file())

    def test_interop_header_exists(self):
        import rayd.slang as rs
        self.assertTrue((rs.include_dir() / "rayd" / "slang" / "interop.h").is_file())

    def test_load_module_without_slangtorch_gives_clear_error(self):
        data = run_json_case("""
            import importlib.abc, json, sys
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "slangtorch" or fullname.startswith("slangtorch."):
                        raise ModuleNotFoundError("No module named 'slangtorch'")
                    return None
            sys.meta_path.insert(0, Block())
            import rayd.slang as rs
            try:
                rs.load_module("x.slang")
            except ImportError as e:
                print(json.dumps({"error": str(e)}))
            """)
        self.assertIn("slangtorch", data["error"])


# ---------------------------------------------------------------------------
# Slang -> C++ compilation tests (the actual interop verification)
# ---------------------------------------------------------------------------

@unittest.skipUnless(SLANGTORCH_AVAILABLE, "Requires slangtorch (for slangc)")
class SlangInteropCompileTests(unittest.TestCase):
    """Compile test_interop.slang with slangc -target cpp and verify the
    generated C++ contains the expected rayd::slang::* calls."""

    _slang_src = str(ROOT / "tests" / "slang" / "test_interop.slang")
    _cpp_out = str(ROOT / "tests" / "slang" / "test_interop_gen.cpp")

    @classmethod
    def setUpClass(cls):
        result = _slangc_compile(cls._slang_src, cls._cpp_out)
        if result.returncode != 0:
            raise AssertionError(
                f"slangc failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        cls.gen = Path(cls._cpp_out).read_text(encoding="utf-8", errors="replace")

    def test_slangc_compiles_without_errors(self):
        self.assertIn("testIntersect", self.gen)

    def test_scene_intersect_call(self):
        self.assertIn("rayd::slang::scene_intersect", self.gen)

    def test_shadow_test_call(self):
        self.assertIn("rayd::slang::scene_shadow_test", self.gen)

    def test_closest_edge_point_call(self):
        self.assertIn("rayd::slang::scene_closest_edge_point", self.gen)

    def test_closest_edge_ray_call(self):
        self.assertIn("rayd::slang::scene_closest_edge_ray", self.gen)

    def test_camera_sample_ray_call(self):
        self.assertIn("rayd::slang::camera_sample_ray", self.gen)

    def test_type_mappings(self):
        for t in ["Float2", "Float3", "Ray", "Intersection",
                   "NearestPointEdge", "NearestRayEdge",
                   "SceneHandle", "CameraHandle"]:
            with self.subTest(t=t):
                self.assertIn(f"rayd::slang::{t}", self.gen)

    def test_all_exported_functions_present(self):
        for fn in ["testIntersect", "testShadow", "testClosestEdgePoint",
                    "testClosestEdgeRay", "testCameraSampleRay",
                    "testFloat3", "testFloat2", "testMakeRay"]:
            with self.subTest(fn=fn):
                self.assertTrue(fn in self.gen or f"{fn}_0" in self.gen,
                                f"{fn} not found in generated C++")


if __name__ == "__main__":
    unittest.main()
