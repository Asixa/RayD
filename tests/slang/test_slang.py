"""Tests for rayd.slang — Slang cpp-target interop.

SlangModuleTests         — rayd.slang Python module (paths, import)
SlangInteropCompileTests — slangc -target cpp codegen verification
SlangRaydInteropTests    — load_module(link_rayd=True) runtime queries
SlangRaydGradientTests   — fwd/bwd gradient through Slang→rayd (FD-based)
SlangRaydE2ETests        — end-to-end optimization through Slang→rayd pipeline
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).resolve().parent / "cases"

try:
    import slangtorch  # noqa: F401
    SLANGTORCH_AVAILABLE = True
    _SLANGC = str(Path(slangtorch.__file__).parent / "bin" /
                  ("slangc.exe" if sys.platform == "win32" else "slangc"))
except ImportError:
    SLANGTORCH_AVAILABLE = False
    _SLANGC = None


def _run_case(name: str, timeout: int = 300) -> dict:
    script = CASES / f"{name}.py"
    r = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT,
        text=True, capture_output=True, timeout=timeout, check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    if r.returncode != 0:
        raise AssertionError(
            f"Case {name} failed (rc={r.returncode}).\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if not lines:
        raise AssertionError(f"Case {name}: no output.\nSTDERR:\n{r.stderr}")
    return json.loads(lines[-1])


def _slangc_compile(slang_file: str, output_file: str, target: str = "cpp"):
    return subprocess.run(
        [_SLANGC, slang_file, "-target", target, "-o", output_file,
         "-I", str(ROOT / "include"), "-ignore-capabilities"],
        cwd=ROOT, text=True, capture_output=True, timeout=60, check=False)


# ---------------------------------------------------------------------------
# Module paths / import
# ---------------------------------------------------------------------------

class SlangModuleTests(unittest.TestCase):

    def test_import_rayd_slang(self):
        import rayd.slang as rs
        self.assertTrue(callable(rs.include_dir))
        self.assertTrue(callable(rs.load_module))

    def test_include_dir_exists(self):
        import rayd.slang as rs
        self.assertTrue((rs.include_dir() / "rayd" / "slang" / "rayd.slang").is_file())

    def test_interop_header_exists(self):
        import rayd.slang as rs
        self.assertTrue((rs.include_dir() / "rayd" / "slang" / "interop.h").is_file())

    def test_load_module_without_slangtorch_gives_clear_error(self):
        data = _run_case("load_module_error")
        self.assertIn("slangtorch", data["error"])


# ---------------------------------------------------------------------------
# slangc codegen verification
# ---------------------------------------------------------------------------

@unittest.skipUnless(SLANGTORCH_AVAILABLE, "Requires slangtorch (for slangc)")
class SlangInteropCompileTests(unittest.TestCase):

    _slang_src = str(ROOT / "tests" / "slang" / "test_interop.slang")
    _cpp_out = str(ROOT / "tests" / "slang" / "test_interop_gen.cpp")

    @classmethod
    def setUpClass(cls):
        r = _slangc_compile(cls._slang_src, cls._cpp_out)
        if r.returncode != 0:
            raise AssertionError(
                f"slangc failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
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


# ---------------------------------------------------------------------------
# Runtime interop: Slang calls raydSceneIntersect on a real scene
# ---------------------------------------------------------------------------

@unittest.skipUnless(SLANGTORCH_AVAILABLE, "Requires slangtorch")
class SlangRaydInteropTests(unittest.TestCase):

    def test_intersect_and_shadow(self):
        d = _run_case("slang_rayd_intersect")
        self.assertAlmostEqual(d["t_hit"], 1.0, places=4)
        self.assertTrue(d["valid_hit"])
        self.assertTrue(d["shadow_hit"])
        self.assertTrue(d["t_miss_inf"])
        self.assertFalse(d["valid_miss"])


# ---------------------------------------------------------------------------
# Gradient: fwd / bwd through Slang → rayd (FD-based backward)
# ---------------------------------------------------------------------------

@unittest.skipUnless(SLANGTORCH_AVAILABLE, "Requires slangtorch")
class SlangRaydGradientTests(unittest.TestCase):

    def test_trace_t_gradient(self):
        """d(traceRayT)/d(oz) ≈ -1 (ray z-origin → hit distance)."""
        d = _run_case("slang_rayd_gradient_fwd_bwd")
        self.assertAlmostEqual(d["t"], 1.0, places=4)
        self.assertTrue(d["grad_close_to_minus_one"],
                        f"grad_oz={d['grad_oz']}, expected ≈ -1")

    def test_inverse_depth_gradient(self):
        """d(1/t)/d(oz) ≈ 1 at oz=-1."""
        d = _run_case("slang_rayd_gradient_inverse_depth")
        self.assertAlmostEqual(d["inv_depth"], 1.0, places=4)
        self.assertTrue(d["grad_close_to_one"],
                        f"grad_oz={d['grad_oz']}, expected ≈ 1")

    def test_squared_depth_gradient(self):
        """d(t^2)/d(oz) ≈ -4 at oz=-2."""
        d = _run_case("slang_rayd_gradient_squared_depth")
        self.assertAlmostEqual(d["sq_depth"], 4.0, places=3)
        self.assertTrue(d["match"],
                        f"grad_oz={d['grad_oz']}, expected ≈ {d['expected_grad']}")


# ---------------------------------------------------------------------------
# End-to-end: optimization loop through Slang → rayd
# ---------------------------------------------------------------------------

@unittest.skipUnless(SLANGTORCH_AVAILABLE, "Requires slangtorch")
class SlangRaydE2ETests(unittest.TestCase):

    def test_optimize_origin_to_target_depth(self):
        """Adam optimizes oz so traceRayT → target depth, all through Slang."""
        d = _run_case("slang_rayd_e2e_optimize")
        self.assertTrue(d["converged"],
                        f"final_t={d['final_t']:.3f}, target=5.0")

    def test_multi_query_gradient_composition(self):
        """Two Slang→rayd queries, combined loss, independent gradients."""
        d = _run_case("slang_rayd_e2e_multi_query")
        self.assertTrue(d["oz1_close"],
                        f"grad_oz1={d['grad_oz1']}, expected ≈ 2")
        self.assertTrue(d["oz2_close"],
                        f"grad_oz2={d['grad_oz2']}, expected ≈ -2")


if __name__ == "__main__":
    unittest.main()
