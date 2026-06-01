import json
import math
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, timeout: int = 120, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "Subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def run_json_case(script: str, timeout: int = 120):
    result = run_script(script, timeout=timeout, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON output.\nSTDERR:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Failed to parse JSON from subprocess.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc


class SurfelCoreTests(unittest.TestCase):
    def test_surfel_api_is_exposed(self):
        data = run_json_case(
            """
            import json
            import rayd as pj

            print(json.dumps({
                "has_cloud": hasattr(pj, "SurfelCloud"),
                "has_scene": hasattr(pj, "SurfelScene"),
                "has_options": hasattr(pj, "SurfelTraceOptions"),
                "has_mode": hasattr(pj, "SurfelPrimitiveMode"),
                "has_reference_composite": hasattr(pj.SurfelScene, "composite_alpha_reference"),
                "single_launch_default": pj.SurfelTraceOptions().single_launch,
                "ico_name": str(pj.SurfelPrimitiveMode.Icosahedron20),
                "quad_name": str(pj.SurfelPrimitiveMode.QuadTriangles),
                "single_name": str(pj.SurfelPrimitiveMode.SingleTriangle),
            }))
            """
        )

        self.assertTrue(data["has_cloud"])
        self.assertTrue(data["has_scene"])
        self.assertTrue(data["has_options"])
        self.assertTrue(data["has_mode"])
        self.assertTrue(data["has_reference_composite"])
        self.assertTrue(data["single_launch_default"])
        self.assertIn("Icosahedron20", data["ico_name"])
        self.assertIn("QuadTriangles", data["quad_name"])
        self.assertIn("SingleTriangle", data["single_name"])

    def test_surfel_cloud_fields_are_exposed(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            cloud = pj.SurfelCloud(
                cuda.Array3f([1.0], [2.0], [3.0]),
                cuda.Array3f([0.5], [0.0], [0.0]),
                cuda.Array3f([0.0], [0.25], [0.0]),
                cuda.Float([0.75]),
            )

            print(json.dumps({
                "center_x": float(cloud.center[0][0]),
                "tangent_u_x": float(cloud.tangent_u[0][0]),
                "tangent_v_y": float(cloud.tangent_v[1][0]),
                "opacity": float(cloud.opacity[0]),
            }))
            """
        )

        self.assertAlmostEqual(data["center_x"], 1.0, places=5)
        self.assertAlmostEqual(data["tangent_u_x"], 0.5, places=5)
        self.assertAlmostEqual(data["tangent_v_y"], 0.25, places=5)
        self.assertAlmostEqual(data["opacity"], 0.75, places=5)

    def test_default_proxy_is_twenty_triangle_icosahedron(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            scene = pj.SurfelScene(pj.SurfelCloud(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
                cuda.Float([1.0]),
            ))
            scene.build()

            print(json.dumps({
                "surfel_count": scene.surfel_count,
                "triangle_count": scene.triangle_count,
            }))
            """
        )

        self.assertEqual(data["surfel_count"], 1)
        self.assertEqual(data["triangle_count"], 20)

    def test_quad_surfel_intersection_returns_2dgs_fields(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5)
            opts.primitive_mode = pj.SurfelPrimitiveMode.QuadTriangles

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
                cuda.Float([0.75]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.25], [0.25], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            its = scene.intersect(ray)

            print(json.dumps({
                "surfel_count": scene.surfel_count,
                "triangle_count": scene.triangle_count,
                "valid": bool(its.is_valid()[0]),
                "t": float(its.t[0]),
                "px": float(its.p[0][0]),
                "py": float(its.p[1][0]),
                "pz": float(its.p[2][0]),
                "nz": float(its.n[2][0]),
                "uvx": float(its.local_uv[0][0]),
                "uvy": float(its.local_uv[1][0]),
                "weight": float(its.gaussian_weight[0]),
                "opacity": float(its.opacity[0]),
                "surfel_id": int(its.surfel_id[0]),
                "triangle_id": int(its.triangle_id[0]),
            }))
            """
        )

        self.assertEqual(data["surfel_count"], 1)
        self.assertEqual(data["triangle_count"], 2)
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertAlmostEqual(data["px"], 0.25, places=5)
        self.assertAlmostEqual(data["py"], 0.25, places=5)
        self.assertAlmostEqual(data["pz"], 0.0, places=5)
        self.assertAlmostEqual(data["nz"], 1.0, places=5)
        self.assertAlmostEqual(data["uvx"], 0.25, places=5)
        self.assertAlmostEqual(data["uvy"], 0.25, places=5)
        self.assertAlmostEqual(data["weight"], math.exp(-0.5 * (0.25**2 + 0.25**2)), places=5)
        self.assertAlmostEqual(data["opacity"], 0.75, places=5)
        self.assertEqual(data["surfel_id"], 0)
        self.assertIn(data["triangle_id"], (0, 1))

    def test_quad_support_miss_and_single_triangle_mode(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            center = cuda.Array3f([0.0], [0.0], [0.0])
            tangent_u = cuda.Array3f([1.0], [0.0], [0.0])
            tangent_v = cuda.Array3f([0.0], [1.0], [0.0])
            opacity = cuda.Float([1.0])

            quad_opts = pj.SurfelTraceOptions()
            quad_opts.alpha_min = math.exp(-0.5)
            quad_opts.primitive_mode = pj.SurfelPrimitiveMode.QuadTriangles
            quad_scene = pj.SurfelScene(pj.SurfelCloud(center, tangent_u, tangent_v, opacity), quad_opts)
            quad_scene.build()

            single_opts = pj.SurfelTraceOptions()
            single_opts.alpha_min = math.exp(-0.5)
            single_opts.primitive_mode = pj.SurfelPrimitiveMode.SingleTriangle
            single_scene = pj.SurfelScene(pj.SurfelCloud(center, tangent_u, tangent_v, opacity), single_opts)
            single_scene.build()

            miss_ray = pj.Ray(
                cuda.Array3f([1.5], [1.5], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            hit_ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            miss = quad_scene.intersect(miss_ray)
            single_hit = single_scene.intersect(hit_ray)

            print(json.dumps({
                "quad_triangles": quad_scene.triangle_count,
                "single_triangles": single_scene.triangle_count,
                "miss_valid": bool(miss.is_valid()[0]),
                "single_valid": bool(single_hit.is_valid()[0]),
                "single_surfel": int(single_hit.surfel_id[0]),
            }))
            """
        )

        self.assertEqual(data["quad_triangles"], 2)
        self.assertEqual(data["single_triangles"], 1)
        self.assertFalse(data["miss_valid"])
        self.assertTrue(data["single_valid"])
        self.assertEqual(data["single_surfel"], 0)

    def test_closest_surfel_and_visibility(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5)

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, -1.0]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
                cuda.Float([1.0, 0.5]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [2.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            its = scene.intersect(ray)
            occluded = scene.shadow_test(ray)
            visible_blocked = scene.visible(
                cuda.Array3f([0.0], [0.0], [2.0]),
                cuda.Array3f([0.0], [0.0], [-2.0]),
            )
            visible_clear = scene.visible(
                cuda.Array3f([2.0], [2.0], [2.0]),
                cuda.Array3f([2.0], [2.0], [-2.0]),
            )

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "t": float(its.t[0]),
                "surfel_id": int(its.surfel_id[0]),
                "occluded": bool(occluded[0]),
                "visible_blocked": bool(visible_blocked[0]),
                "visible_clear": bool(visible_clear[0]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 2.0, places=5)
        self.assertEqual(data["surfel_id"], 0)
        self.assertTrue(data["occluded"])
        self.assertFalse(data["visible_blocked"])
        self.assertTrue(data["visible_clear"])

    def test_intersect_continues_after_invalid_proxy_candidate(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5)
            opts.max_candidate_hits = 8

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 1.1], [0.0, 0.0], [0.0, -1.0]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
                cuda.Float([1.0, 1.0]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            capped_opts = pj.SurfelTraceOptions()
            capped_opts.alpha_min = math.exp(-0.5)
            capped_opts.max_candidate_hits = 1
            capped_opts.single_launch = False
            capped_scene = pj.SurfelScene(cloud, capped_opts)
            capped_scene.build()

            ray = pj.Ray(
                cuda.Array3f([1.05], [0.0], [2.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            its = scene.intersect(ray)
            capped = capped_scene.intersect(ray)

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "surfel_id": int(its.surfel_id[0]),
                "t": float(its.t[0]),
                "capped_valid": bool(capped.is_valid()[0]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["surfel_id"], 1)
        self.assertAlmostEqual(data["t"], 3.0, places=5)
        self.assertFalse(data["capped_valid"])

    def test_single_launch_matches_legacy_retrace_and_counts_one_optix_launch(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 1.1], [0.0, 0.0], [0.0, -1.0]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
                cuda.Float([1.0, 1.0]),
            )

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5)
            opts.max_candidate_hits = 8
            opts.single_launch = True
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            legacy_opts = pj.SurfelTraceOptions()
            legacy_opts.alpha_min = math.exp(-0.5)
            legacy_opts.max_candidate_hits = 8
            legacy_opts.single_launch = False
            legacy_scene = pj.SurfelScene(cloud, legacy_opts)
            legacy_scene.build()

            ray = pj.Ray(
                cuda.Array3f([1.05], [0.0], [2.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )

            pj.native_launch_audit_clear()
            fast = scene.intersect(ray)
            fast_valid = bool(fast.is_valid()[0])
            fast_surfel = int(fast.surfel_id[0])
            fast_t = float(fast.t[0])
            fast_alpha = float(fast.alpha[0])
            audit = pj.native_launch_audit()

            legacy = legacy_scene.intersect(ray)
            print(json.dumps({
                "fast_valid": fast_valid,
                "legacy_valid": bool(legacy.is_valid()[0]),
                "fast_surfel": fast_surfel,
                "legacy_surfel": int(legacy.surfel_id[0]),
                "fast_t": fast_t,
                "legacy_t": float(legacy.t[0]),
                "fast_alpha": fast_alpha,
                "legacy_alpha": float(legacy.alpha[0]),
                "surfel_launches": audit.get("surfel_trace", {}).get("optix_launch", -1),
            }))
            """
        )

        self.assertTrue(data["fast_valid"])
        self.assertTrue(data["legacy_valid"])
        self.assertEqual(data["fast_surfel"], data["legacy_surfel"])
        self.assertAlmostEqual(data["fast_t"], data["legacy_t"], places=5)
        self.assertAlmostEqual(data["fast_alpha"], data["legacy_alpha"], places=5)
        self.assertEqual(data["surfel_launches"], 1)

    def test_single_launch_intersect_sorts_by_analytic_plane_t_not_proxy_t(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.05]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
                cuda.Float([1.0, 0.01]),
            )

            opts = pj.SurfelTraceOptions()
            opts.proxy_epsilon = 0.1
            opts.single_launch = True
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )

            pj.native_launch_audit_clear()
            hit = scene.intersect(ray)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "valid": bool(hit.is_valid()[0]),
                "surfel_id": int(hit.surfel_id[0]),
                "t": float(hit.t[0]),
                "alpha": float(hit.alpha[0]),
                "surfel_launches": audit.get("surfel_trace", {}).get("optix_launch", -1),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["surfel_id"], 1)
        self.assertAlmostEqual(data["t"], 0.95, places=5)
        self.assertAlmostEqual(data["alpha"], 0.01, places=5)
        self.assertEqual(data["surfel_launches"], 1)

    def test_alpha_composite_makes_quad_edges_transparent(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5 * (2.5 * 2.5))
            opts.alpha_cap = 0.99

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([0.2], [0.0], [0.0]),
                cuda.Array3f([0.0], [0.2], [0.0]),
                cuda.Float([1.0]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.5], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            comp = scene.composite_alpha(ray)
            expected = math.exp(-0.5 * (2.5 * 2.5))

            print(json.dumps({
                "valid": bool(comp.is_valid()[0]),
                "alpha": float(comp.alpha[0]),
                "intensity": float(comp.intensity[0]),
                "transmittance": float(comp.transmittance[0]),
                "expected": expected,
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertGreater(data["alpha"], 0.0)
        self.assertLess(data["alpha"], 0.1)
        self.assertAlmostEqual(data["alpha"], data["expected"], places=5)
        self.assertAlmostEqual(data["intensity"], data["alpha"], places=5)
        self.assertAlmostEqual(data["transmittance"], 1.0 - data["alpha"], places=5)

    def test_alpha_composite_blends_coplanar_overlapping_surfels(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5 * (2.5 * 2.5))
            opts.alpha_cap = 0.99

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.2, 0.2], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [0.2, 0.2], [0.0, 0.0]),
                cuda.Float([0.5, 0.25]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            its = scene.intersect(ray)
            comp = scene.composite_alpha(ray)

            print(json.dumps({
                "nearest_alpha": float(its.gaussian_weight[0] * its.opacity[0]),
                "composite_alpha": float(comp.alpha[0]),
                "expected": 1.0 - (1.0 - 0.5) * (1.0 - 0.25),
                "depth": float(comp.depth[0]),
            }))
            """
        )

        self.assertAlmostEqual(data["nearest_alpha"], 0.5, places=5)
        self.assertAlmostEqual(data["composite_alpha"], data["expected"], places=5)
        self.assertAlmostEqual(data["depth"], 1.0, places=5)

    def test_single_launch_alpha_composite_matches_reference_and_counts_one_launch(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5 * (2.5 * 2.5))
            opts.alpha_cap = 0.99
            opts.max_candidate_hits = 8
            opts.single_launch = True

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.5]),
                cuda.Array3f([0.2, 0.2], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [0.2, 0.2], [0.0, 0.0]),
                cuda.Float([0.5, 0.25]),
                cuda.Float([0.2, 0.8]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            legacy_opts = pj.SurfelTraceOptions()
            legacy_opts.alpha_min = opts.alpha_min
            legacy_opts.alpha_cap = opts.alpha_cap
            legacy_opts.max_candidate_hits = opts.max_candidate_hits
            legacy_opts.single_launch = False
            legacy_scene = pj.SurfelScene(cloud, legacy_opts)
            legacy_scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )

            pj.native_launch_audit_clear()
            native = scene.composite_alpha(ray)
            audit = pj.native_launch_audit()
            reference = legacy_scene.composite_alpha_reference(ray)

            print(json.dumps({
                "native_alpha": float(native.alpha[0]),
                "reference_alpha": float(reference.alpha[0]),
                "native_intensity": float(native.intensity[0]),
                "reference_intensity": float(reference.intensity[0]),
                "native_depth": float(native.depth[0]),
                "reference_depth": float(reference.depth[0]),
                "surfel_launches": audit.get("surfel_trace", {}).get("optix_launch", -1),
            }))
            """
        )

        self.assertAlmostEqual(data["native_alpha"], data["reference_alpha"], places=5)
        self.assertAlmostEqual(data["native_intensity"], data["reference_intensity"], places=5)
        self.assertAlmostEqual(data["native_depth"], data["reference_depth"], places=5)
        self.assertEqual(data["surfel_launches"], 1)

    def test_single_launch_alpha_composite_ad_uses_native_candidates_and_gradients(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda.ad as ad

            opacity = ad.Float([0.8])
            value = ad.Float([0.5])
            dr.enable_grad(opacity, value)

            cloud = pj.SurfelCloud(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([1.0]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([1.0]), ad.Float([0.0])),
                opacity,
                value,
            )
            opts = pj.SurfelTraceOptions()
            opts.single_launch = True
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.RayAD(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([1.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-1.0])),
            )

            pj.native_launch_audit_clear()
            comp = scene.composite_alpha(ray)
            loss = dr.sum(comp.intensity)
            dr.backward(loss)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "valid": bool(comp.is_valid()[0]),
                "intensity": float(comp.intensity[0]),
                "alpha": float(comp.alpha[0]),
                "grad_opacity": float(dr.grad(opacity)[0]),
                "grad_value": float(dr.grad(value)[0]),
                "surfel_launches": audit.get("surfel_trace", {}).get("optix_launch", -1),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["intensity"], 0.4, places=5)
        self.assertAlmostEqual(data["alpha"], 0.8, places=5)
        self.assertAlmostEqual(data["grad_opacity"], 0.5, places=5)
        self.assertAlmostEqual(data["grad_value"], 0.8, places=5)
        self.assertEqual(data["surfel_launches"], 1)

    def test_scalar_surfel_value_modulates_composited_intensity(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = 1.0 / 255.0
            opts.alpha_cap = 0.99

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
                cuda.Float([0.5]),
                cuda.Float([0.2]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            its = scene.intersect(ray)
            comp = scene.composite_alpha_reference(ray)

            print(json.dumps({
                "hit_value": float(its.value[0]),
                "hit_alpha": float(its.alpha[0]),
                "alpha": float(comp.alpha[0]),
                "intensity": float(comp.intensity[0]),
            }))
            """
        )

        self.assertAlmostEqual(data["hit_value"], 0.2, places=5)
        self.assertAlmostEqual(data["hit_alpha"], 0.5, places=5)
        self.assertAlmostEqual(data["alpha"], 0.5, places=5)
        self.assertAlmostEqual(data["intensity"], 0.1, places=5)

    def test_alpha_composite_sorts_surfel_hits_front_to_back(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = 1.0 / 255.0
            opts.alpha_cap = 0.99

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.5]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
                cuda.Float([0.5, 0.25]),
            )
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            comp = scene.composite_alpha(ray)

            print(json.dumps({
                "alpha": float(comp.alpha[0]),
                "expected_alpha": 1.0 - (1.0 - 0.25) * (1.0 - 0.5),
                "depth": float(comp.depth[0]),
                "expected_depth": (0.25 * 0.5 + 0.75 * 0.5 * 1.0) /
                                  (0.25 + 0.75 * 0.5),
            }))
            """
        )

        self.assertAlmostEqual(data["alpha"], data["expected_alpha"], places=5)
        self.assertAlmostEqual(data["depth"], data["expected_depth"], places=5)

    def test_alpha_min_defines_analytic_gaussian_boundary(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5 * 2.0 * 2.0)
            opts.primitive_mode = pj.SurfelPrimitiveMode.QuadTriangles

            scene = pj.SurfelScene(pj.SurfelCloud(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
                cuda.Float([1.0]),
            ), opts)
            scene.build()

            edge_ray = pj.Ray(
                cuda.Array3f([2.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            outside_ray = pj.Ray(
                cuda.Array3f([2.1], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            edge = scene.composite_alpha(edge_ray)
            outside = scene.composite_alpha(outside_ray)

            print(json.dumps({
                "edge_valid": bool(edge.is_valid()[0]),
                "edge_alpha": float(edge.alpha[0]),
                "outside_valid": bool(outside.is_valid()[0]),
                "alpha_min": math.exp(-0.5 * 2.0 * 2.0),
            }))
            """
        )

        self.assertTrue(data["edge_valid"])
        self.assertAlmostEqual(data["edge_alpha"], data["alpha_min"], places=5)
        self.assertFalse(data["outside_valid"])

    def test_ad_center_gradient_flows_through_surfel_plane(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda.ad as ad

            center = ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([0.0]))
            dr.enable_grad(center)
            cloud = pj.SurfelCloud(
                center,
                ad.Array3f(ad.Float([1.0]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([1.0]), ad.Float([0.0])),
                ad.Float([1.0]),
            )
            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5)
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.RayAD(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([1.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-1.0])),
            )
            its = scene.intersect(ray)
            dr.backward(dr.sum(its.t))
            grad = dr.grad(center)

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "t": float(its.t[0]),
                "grad_z": float(grad[2][0]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertAlmostEqual(data["grad_z"], -1.0, places=5)

    def test_ad_tangent_scale_gradient_flows_through_gaussian_weight(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda.ad as ad

            scale = ad.Float([1.0])
            dr.enable_grad(scale)
            cloud = pj.SurfelCloud(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(scale, ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([1.0]), ad.Float([0.0])),
                ad.Float([1.0]),
            )
            scene = pj.SurfelScene(cloud)
            scene.build()

            ray = pj.RayAD(
                ad.Array3f(ad.Float([0.5]), ad.Float([0.0]), ad.Float([1.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-1.0])),
            )
            its = scene.intersect(ray)
            dr.backward(dr.sum(its.gaussian_weight))

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "local_u": float(its.local_uv[0][0]),
                "weight": float(its.gaussian_weight[0]),
                "grad_scale": float(dr.grad(scale)[0]),
                "expected_grad": math.exp(-0.125) * 0.25,
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["local_u"], 0.5, places=5)
        self.assertAlmostEqual(data["weight"], math.exp(-0.125), places=5)
        self.assertAlmostEqual(data["grad_scale"], data["expected_grad"], places=5)

    def test_depth_image_fitting_recovers_surfel_center(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda.ad as ad

            xs = [-0.45, -0.15, 0.15, 0.45] * 4
            ys = [-0.45] * 4 + [-0.15] * 4 + [0.15] * 4 + [0.45] * 4
            target_z = 0.25
            target_t = 1.0 - target_z
            z_value = -0.25
            initial_rms = None
            final_rms = None

            opts = pj.SurfelTraceOptions()
            opts.alpha_min = math.exp(-0.5)

            for iteration in range(18):
                z = ad.Float([z_value])
                dr.enable_grad(z)
                cloud = pj.SurfelCloud(
                    ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), z),
                    ad.Array3f(ad.Float([1.0]), ad.Float([0.0]), ad.Float([0.0])),
                    ad.Array3f(ad.Float([0.0]), ad.Float([1.0]), ad.Float([0.0])),
                    ad.Float([1.0]),
                )
                scene = pj.SurfelScene(cloud, opts)
                scene.build()
                ray = pj.RayAD(
                    ad.Array3f(ad.Float(xs), ad.Float(ys), ad.Float([1.0] * len(xs))),
                    ad.Array3f(ad.Float([0.0] * len(xs)),
                               ad.Float([0.0] * len(xs)),
                               ad.Float([-1.0] * len(xs))),
                )
                its = scene.intersect(ray)
                residual = its.t - ad.Float([target_t] * len(xs))
                loss = dr.sum(residual * residual) / len(xs)
                dr.backward(loss)
                rms = math.sqrt(float(loss[0]))
                if initial_rms is None:
                    initial_rms = rms
                final_rms = rms
                z_value -= 0.45 * float(dr.grad(z)[0])

            print(json.dumps({
                "initial_rms": initial_rms,
                "final_rms": final_rms,
                "z": z_value,
            }))
            """,
            timeout=180,
        )

        self.assertGreater(data["initial_rms"], 0.3)
        self.assertLess(data["final_rms"], 0.02)
        self.assertAlmostEqual(data["z"], 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
