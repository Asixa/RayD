# Copyright Xingyu Chen.
# Tests surfel Dr.Jit.

import math
import unittest
from pathlib import Path

from tests.support.subprocess_cases import run_json_case


ROOT = Path(__file__).resolve().parents[2]


class SurfelCoreTests(unittest.TestCase):
    def test_surfel_api_is_exposed(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj

            print(json.dumps({
                "has_cloud": hasattr(pj, "SurfelCloud"),
                "has_geometry": hasattr(pj, "SurfelGeometry"),
                "has_appearance": hasattr(pj, "SurfelAppearance"),
                "has_scene": hasattr(pj, "SurfelScene"),
                "has_options": hasattr(pj, "SurfelTraceOptions"),
                "has_render_options": hasattr(pj, "SurfelRenderOptions"),
                "has_render_mode": hasattr(pj, "SurfelRenderMode"),
                "has_color_model": hasattr(pj, "SurfelColorModel"),
                "has_mode": hasattr(pj, "SurfelPrimitiveMode"),
                "has_reference_composite": hasattr(pj.SurfelScene, "composite_alpha_reference"),
                "has_render": hasattr(pj.SurfelScene, "render"),
                "has_update_appearance": hasattr(pj.SurfelScene, "update_appearance"),
                "single_launch_default": pj.SurfelTraceOptions().single_launch,
                "ico_name": str(pj.SurfelPrimitiveMode.Icosahedron20),
                "quad_name": str(pj.SurfelPrimitiveMode.QuadTriangles),
                "single_name": str(pj.SurfelPrimitiveMode.SingleTriangle),
                "rgb_mode": str(pj.SurfelRenderMode.RGB),
                "sh_model": str(pj.SurfelColorModel.SH),
            }))
            """
        )

        self.assertTrue(data["has_cloud"])
        self.assertTrue(data["has_geometry"])
        self.assertTrue(data["has_appearance"])
        self.assertTrue(data["has_scene"])
        self.assertTrue(data["has_options"])
        self.assertTrue(data["has_render_options"])
        self.assertTrue(data["has_render_mode"])
        self.assertTrue(data["has_color_model"])
        self.assertTrue(data["has_mode"])
        self.assertTrue(data["has_reference_composite"])
        self.assertTrue(data["has_render"])
        self.assertTrue(data["has_update_appearance"])
        self.assertTrue(data["single_launch_default"])
        self.assertIn("Icosahedron20", data["ico_name"])
        self.assertIn("QuadTriangles", data["quad_name"])
        self.assertIn("SingleTriangle", data["single_name"])
        self.assertIn("RGB", data["rgb_mode"])
        self.assertIn("SH", data["sh_model"])

    def test_surfel_native_reports_candidate_buffer_saturation(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            count = 6
            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0] * count, [0.0] * count, [0.02 * i for i in range(count)]),
                cuda.Array3f([1.0] * count, [0.0] * count, [0.0] * count),
                cuda.Array3f([0.0] * count, [1.0] * count, [0.0] * count),
                cuda.Float([0.5] * count),
                cuda.Float([1.0] * count),
            )
            opts = pj.SurfelTraceOptions()
            opts.max_candidate_hits = 2
            opts.collect_candidate_stats = True
            scene = pj.SurfelScene(cloud, opts)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, pj.SurfelRenderOptions.rgb())

            print(json.dumps({
                "alpha": float(out.alpha[0]),
                "candidate_count": int(out.candidate_count[0]),
                "buffer_full": bool(out.candidate_buffer_full[0]),
            }))
            """
        )

        self.assertEqual(data["candidate_count"], 2)
        self.assertTrue(data["buffer_full"])
        self.assertGreater(data["alpha"], 0.0)

    def test_opacity_aware_proxy_bounds_reduce_low_opacity_candidate_pressure(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0, 0.0], [0.0, 3.0], [0.0, 0.0]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
                cuda.Float([1.0, 0.01]),
                cuda.Float([1.0, 1.0]),
            )

            ray = pj.Ray(
                cuda.Array3f([0.0], [3.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )

            conservative = pj.SurfelTraceOptions()
            conservative.collect_candidate_stats = True
            conservative.opacity_aware_proxy_bounds = False
            conservative.max_candidate_hits = 8
            scene_a = pj.SurfelScene(cloud, conservative)
            scene_a.build()
            out_a = scene_a.render(ray, pj.SurfelRenderOptions.rgb())

            aware = pj.SurfelTraceOptions()
            aware.collect_candidate_stats = True
            aware.opacity_aware_proxy_bounds = True
            aware.max_candidate_hits = 8
            scene_b = pj.SurfelScene(cloud, aware)
            scene_b.build()
            out_b = scene_b.render(ray, pj.SurfelRenderOptions.rgb())

            print(json.dumps({
                "count_a": int(out_a.candidate_count[0]),
                "count_b": int(out_b.candidate_count[0]),
            }))
            """
        )

        self.assertLessEqual(data["count_b"], data["count_a"])

    def test_surfel_geometry_and_rgb_appearance_render_without_rebuild(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            geometry = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            scene = pj.SurfelScene(geometry)
            scene.build()
            before_build_count = scene.build_count

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            opts = pj.SurfelRenderOptions.rgb()

            scene.update_appearance(pj.SurfelAppearance.rgb(
                cuda.Float([0.5]),
                cuda.Array3f([0.2], [0.4], [0.6]),
            ))
            first = scene.render(ray, opts)

            scene.update_appearance(pj.SurfelAppearance.rgb(
                cuda.Float([0.5]),
                cuda.Array3f([0.8], [0.6], [0.4]),
            ))
            second = scene.render(ray, opts)

            print(json.dumps({
                "surfel_count": geometry.surfel_count,
                "first_r": float(first.rgb[0][0]),
                "first_g": float(first.rgb[1][0]),
                "first_b": float(first.rgb[2][0]),
                "second_r": float(second.rgb[0][0]),
                "second_g": float(second.rgb[1][0]),
                "second_b": float(second.rgb[2][0]),
                "alpha": float(second.alpha[0]),
                "depth": float(second.depth[0]),
                "build_count_before": before_build_count,
                "build_count_after": scene.build_count,
                "channel_count": second.channel_count,
            }))
            """
        )

        self.assertEqual(data["surfel_count"], 1)
        self.assertAlmostEqual(data["first_r"], 0.1, places=5)
        self.assertAlmostEqual(data["first_g"], 0.2, places=5)
        self.assertAlmostEqual(data["first_b"], 0.3, places=5)
        self.assertAlmostEqual(data["second_r"], 0.4, places=5)
        self.assertAlmostEqual(data["second_g"], 0.3, places=5)
        self.assertAlmostEqual(data["second_b"], 0.2, places=5)
        self.assertAlmostEqual(data["alpha"], 0.5, places=5)
        self.assertAlmostEqual(data["depth"], 1.0, places=5)
        self.assertEqual(data["build_count_before"], 1)
        self.assertEqual(data["build_count_after"], 1)
        self.assertEqual(data["channel_count"], 3)

    def test_surfel_geometry_update_rebuilds_accel_and_updates_depth(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            geometry = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            scene = pj.SurfelScene(geometry)
            scene.build()

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            before = scene.render(ray, pj.SurfelRenderOptions.rgb())

            updated = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.25]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            scene.update_geometry(updated)
            after = scene.render(ray, pj.SurfelRenderOptions.rgb())

            print(json.dumps({
                "before_depth": float(before.depth[0]),
                "after_depth": float(after.depth[0]),
                "after_count": scene.build_count,
            }))
            """
        )

        self.assertAlmostEqual(data["before_depth"], 1.0, places=5)
        self.assertAlmostEqual(data["after_depth"], 0.75, places=5)
        self.assertEqual(data["after_count"], 2)

    def test_surfel_feature_render_outputs_flat_channel_buffer(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            geometry = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            appearance = pj.SurfelAppearance.features(
                cuda.Float([0.25]),
                cuda.Float([0.1, 0.2, 0.3, 0.4]),
                4,
            )
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(appearance)

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, pj.SurfelRenderOptions.feature(4))

            print(json.dumps({
                "channel_count": out.channel_count,
                "c0": float(out.channels[0]),
                "c1": float(out.channels[1]),
                "c2": float(out.channels[2]),
                "c3": float(out.channels[3]),
                "alpha": float(out.alpha[0]),
            }))
            """
        )

        self.assertEqual(data["channel_count"], 4)
        self.assertAlmostEqual(data["c0"], 0.025, places=5)
        self.assertAlmostEqual(data["c1"], 0.05, places=5)
        self.assertAlmostEqual(data["c2"], 0.075, places=5)
        self.assertAlmostEqual(data["c3"], 0.1, places=5)
        self.assertAlmostEqual(data["alpha"], 0.25, places=5)

    def test_surfel_sh_render_evaluates_degree_one_in_native_render(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            y10 = 0.4886025119029199
            geometry = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            # Layout is [surfel][basis][rgb], basis order:
            # Y00, Y1-1(y), Y10(z), Y11(x).
            coeffs = cuda.Float([
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0,
            ])
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(pj.SurfelAppearance.sh(cuda.Float([0.5]), coeffs, 1))

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, pj.SurfelRenderOptions.rgb(sh_degree=1))

            print(json.dumps({
                "r": float(out.rgb[0][0]),
                "g": float(out.rgb[1][0]),
                "b": float(out.rgb[2][0]),
                "expected_r": 0.5 * y10,
                "expected_g": 0.5 * 2.0 * y10,
                "expected_b": 0.5 * 3.0 * y10,
            }))
            """
        )

        self.assertAlmostEqual(data["r"], data["expected_r"], places=5)
        self.assertAlmostEqual(data["g"], data["expected_g"], places=5)
        self.assertAlmostEqual(data["b"], data["expected_b"], places=5)

    def test_surfel_sh_render_options_degree_limits_evaluation(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            geometry = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            coeffs = cuda.Float([
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0,
            ])
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(pj.SurfelAppearance.sh(cuda.Float([0.5]), coeffs, 1))

            opts = pj.SurfelRenderOptions()
            opts.mode = pj.SurfelRenderMode.RGB
            opts.color_model = pj.SurfelColorModel.SH
            opts.sh_degree = 0

            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, opts)

            print(json.dumps({
                "r": float(out.rgb[0][0]),
                "g": float(out.rgb[1][0]),
                "b": float(out.rgb[2][0]),
                "alpha": float(out.alpha[0]),
            }))
            """
        )

        self.assertAlmostEqual(data["r"], 0.0, places=5)
        self.assertAlmostEqual(data["g"], 0.0, places=5)
        self.assertAlmostEqual(data["b"], 0.0, places=5)
        self.assertAlmostEqual(data["alpha"], 0.5, places=5)

    def test_surfel_sh_lower_degree_render_uses_storage_stride(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            y00 = 0.28209479177387814
            geometry = pj.SurfelGeometry(
                cuda.Array3f([-2.0, 2.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
                cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
            )
            # Two surfels, degree-1 SH: 4 basis values * RGB per surfel.
            # The first surfel's Y1-1 red coefficient is deliberately large;
            # a degree-0 render of surfel 1 must not use it as surfel 1's Y00.
            coeffs = cuda.Float([
                1.0, 0.0, 0.0,
                123.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                4.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(pj.SurfelAppearance.sh(cuda.Float([0.5, 0.5]), coeffs, 1))

            opts = pj.SurfelRenderOptions()
            opts.mode = pj.SurfelRenderMode.RGB
            opts.color_model = pj.SurfelColorModel.SH
            opts.sh_degree = 0

            ray = pj.Ray(
                cuda.Array3f([2.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, opts)

            print(json.dumps({
                "r": float(out.rgb[0][0]),
                "expected_r": 0.5 * 4.0 * y00,
                "alpha": float(out.alpha[0]),
            }))
            """
        )

        self.assertAlmostEqual(data["r"], data["expected_r"], places=5)
        self.assertAlmostEqual(data["alpha"], 0.5, places=5)

    def test_surfel_sh_ad_replay_lower_degree_uses_storage_stride(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit as dr
            import drjit.cuda.ad as ad

            y00 = 0.28209479177387814
            coeffs = ad.Float([
                1.0, 0.0, 0.0,
                123.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                4.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])
            dr.enable_grad(coeffs)
            geometry = pj.SurfelGeometry(
                ad.Array3f(ad.Float([-2.0, 2.0]), ad.Float([0.0, 0.0]), ad.Float([0.0, 0.0])),
                ad.Array3f(ad.Float([1.0, 1.0]), ad.Float([0.0, 0.0]), ad.Float([0.0, 0.0])),
                ad.Array3f(ad.Float([0.0, 0.0]), ad.Float([1.0, 1.0]), ad.Float([0.0, 0.0])),
            )
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(pj.SurfelAppearance.sh(ad.Float([0.5, 0.5]), coeffs, 1))

            opts = pj.SurfelRenderOptions()
            opts.mode = pj.SurfelRenderMode.RGB
            opts.color_model = pj.SurfelColorModel.SH
            opts.sh_degree = 0

            ray = pj.RayAD(
                ad.Array3f(ad.Float([2.0]), ad.Float([0.0]), ad.Float([1.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-1.0])),
            )
            out = scene.render(ray, opts)
            dr.backward(dr.sum(out.rgb[0]))
            grad = dr.grad(coeffs)

            print(json.dumps({
                "r": float(out.rgb[0][0]),
                "expected_r": 0.5 * 4.0 * y00,
                "grad_surfel1_y00_r": float(grad[12]),
                "expected_grad": 0.5 * y00,
                "grad_wrong_stride": float(grad[3]),
            }))
            """
        )

        self.assertAlmostEqual(data["r"], data["expected_r"], places=5)
        self.assertAlmostEqual(data["grad_surfel1_y00_r"], data["expected_grad"], places=5)
        self.assertAlmostEqual(data["grad_wrong_stride"], 0.0, places=5)

    def test_surfel_render_inactive_lane_does_not_emit_background(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            geometry = pj.SurfelGeometry(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
            )
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(pj.SurfelAppearance.rgb(
                cuda.Float([0.5]),
                cuda.Array3f([0.2], [0.4], [0.6]),
            ))

            opts = pj.SurfelRenderOptions.rgb(background_rgb=[0.9, 0.8, 0.7])
            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, opts, False)

            print(json.dumps({
                "r": float(out.rgb[0][0]),
                "g": float(out.rgb[1][0]),
                "b": float(out.rgb[2][0]),
                "alpha": float(out.alpha[0]),
            }))
            """
        )

        self.assertAlmostEqual(data["r"], 0.0, places=5)
        self.assertAlmostEqual(data["g"], 0.0, places=5)
        self.assertAlmostEqual(data["b"], 0.0, places=5)
        self.assertAlmostEqual(data["alpha"], 0.0, places=5)

    def test_surfel_rgb_render_ad_replays_appearance_gradients(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit as dr
            import drjit.cuda.ad as ad

            opacity = ad.Float([0.5])
            red = ad.Float([0.2])
            dr.enable_grad(opacity, red)

            geometry = pj.SurfelGeometry(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([1.0]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([1.0]), ad.Float([0.0])),
            )
            appearance = pj.SurfelAppearance.rgb(
                opacity,
                ad.Array3f(red, ad.Float([0.4]), ad.Float([0.6])),
            )
            scene = pj.SurfelScene(geometry)
            scene.build()
            scene.update_appearance(appearance)

            ray = pj.RayAD(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([1.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-1.0])),
            )

            pj.native_launch_audit_clear()
            out = scene.render(ray, pj.SurfelRenderOptions.rgb())
            loss = dr.sum(out.rgb[0])
            dr.backward(loss)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "r": float(out.rgb[0][0]),
                "grad_opacity": float(dr.grad(opacity)[0]),
                "grad_red": float(dr.grad(red)[0]),
                "surfel_launches": audit.get("surfel_trace", {}).get("optix_launch", -1),
            }))
            """
        )

        self.assertAlmostEqual(data["r"], 0.1, places=5)
        self.assertAlmostEqual(data["grad_opacity"], 0.2, places=5)
        self.assertAlmostEqual(data["grad_red"], 0.5, places=5)
        self.assertEqual(data["surfel_launches"], 1)

    def test_surfel_render_outputs_alpha_weighted_normal(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Array3f([0.0], [1.0], [0.0]),
                cuda.Float([0.75]),
                cuda.Float([1.0]),
            )
            scene = pj.SurfelScene(cloud)
            scene.build()
            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )
            out = scene.render(ray, pj.SurfelRenderOptions.rgb(normal=True))

            print(json.dumps({
                "nx": float(out.normal.x[0]),
                "ny": float(out.normal.y[0]),
                "nz": float(out.normal.z[0]),
                "alpha": float(out.alpha[0]),
            }))
            """
        )

        self.assertGreater(data["alpha"], 0.0)
        self.assertAlmostEqual(data["nx"], 0.0, places=5)
        self.assertAlmostEqual(data["ny"], 0.0, places=5)
        self.assertAlmostEqual(data["nz"], 1.0, places=5)

    def test_surfel_cloud_fields_are_exposed(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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

    def test_continuation_matches_reference_better_than_capped_k(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            count = 10
            cloud = pj.SurfelCloud(
                cuda.Array3f([0.0] * count, [0.0] * count, [0.03 * i for i in range(count)]),
                cuda.Array3f([1.0] * count, [0.0] * count, [0.0] * count),
                cuda.Array3f([0.0] * count, [1.0] * count, [0.0] * count),
                cuda.Float([0.2] * count),
                cuda.Float([1.0] * count),
            )
            ray = pj.Ray(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
            )

            capped_opts = pj.SurfelTraceOptions()
            capped_opts.max_candidate_hits = 2
            capped = pj.SurfelScene(cloud, capped_opts)
            capped.build()
            capped_out = capped.composite_alpha(ray)

            cont_opts = pj.SurfelTraceOptions()
            cont_opts.max_candidate_hits = 2
            cont_opts.continue_after_full_buffer = True
            cont_opts.max_trace_segments = 8
            cont = pj.SurfelScene(cloud, cont_opts)
            cont.build()
            cont_out = cont.composite_alpha(ray)

            ref_opts = pj.SurfelTraceOptions()
            ref_opts.max_candidate_hits = 16
            ref_opts.single_launch = False
            ref = pj.SurfelScene(cloud, ref_opts)
            ref.build()
            ref_out = ref.composite_alpha_reference(ray)

            print(json.dumps({
                "capped_error": abs(float(capped_out.alpha[0]) - float(ref_out.alpha[0])),
                "cont_error": abs(float(cont_out.alpha[0]) - float(ref_out.alpha[0])),
            }))
            """
        )

        self.assertLess(data["cont_error"], data["capped_error"])

    def test_single_launch_alpha_composite_ad_uses_native_candidates_and_gradients(self):
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
            import rayd.drjit as pj
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
