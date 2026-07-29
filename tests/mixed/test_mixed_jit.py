# Copyright Xingyu Chen.
# Tests Dr.Jit mixed mesh, SDF, and surfel queries, gradients, and operation scope.

from __future__ import annotations

import unittest

from tests.support.subprocess_cases import run_json_case


class DrJitMixedSceneTests(unittest.TestCase):
    def test_mixed_scene_is_unified_differentiable_and_excludes_diffraction(self) -> None:
        data = run_json_case(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd.drjit as rt

            vertices = ad.Array3f(
                ad.Float([-2.75, -1.25, -2.0]),
                ad.Float([-0.75, -0.75, 0.75]),
                ad.Float([0.0, 0.0, 0.0]),
            )
            dr.enable_grad(vertices)
            faces = cuda.Array3i(cuda.Int([0]), cuda.Int([1]), cuda.Int([2]))
            mesh = rt.Mesh(vertices, faces)
            mesh.edges_enabled = False

            n = 16
            values = ad.Float([(-0.5 + k / (n - 1)) for _i in range(n) for _j in range(n) for k in range(n)])
            sdf_position = ad.Array3f(0.0)
            dr.enable_grad(sdf_position)
            sdf = rt.SdfGrid(
                values, n, n, n, sdf_position,
                ad.Float([1.0, 0.0, 0.0, 0.0]), ad.Array3f(1.0),
            )

            center = ad.Array3f(ad.Float([2.0]), ad.Float([0.0]), ad.Float([0.0]))
            opacity = ad.Float([0.5])
            dr.enable_grad(center, opacity)
            cloud = rt.SurfelCloud(
                center,
                ad.Array3f(ad.Float([0.25]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.25]), ad.Float([0.0])),
                opacity, ad.Float([1.0]),
            )
            options = rt.SurfelTraceOptions()
            options.max_candidate_hits = 1
            options.transmittance_min = 0.0

            scene = rt.MixedScene()
            scene.add_mesh(mesh)
            scene.add_sdf(sdf)
            scene.add_surfel(cloud, options)
            scene.build()
            ray = rt.RayAD(
                ad.Array3f(ad.Float([-2.0, 0.0, 2.0]), ad.Float([0.0, 0.0, 0.0]), ad.Float([-2.0, -2.0, -2.0])),
                ad.Array3f(ad.Float([0.0, 0.0, 0.0]), ad.Float([0.0, 0.0, 0.0]), ad.Float([1.0, 1.0, 1.0])),
            )
            hit = scene.intersect(ray)
            minimal = scene.intersect(ray, flags=getattr(rt.RayFlags, "None"))
            active = cuda.Bool([True, False, True])
            masked_hit = scene.intersect(ray, active)
            masked_visibility = scene.visible(ray.o, ray.o + 4.0 * ray.d, active).visible
            masked_transmission = scene.transmittance(ray, active)
            visibility = scene.visible(ray.o, ray.o + 4.0 * ray.d).visible
            transmission = scene.transmittance(ray)
            chain = scene.trace_reflections(ray, 1)
            reflection_t_grad_enabled = bool(dr.grad_enabled(chain.t))
            dr.backward(dr.sum(hit.t) + dr.sum(chain.t))
            combined_gradients = {
                "mesh": float(dr.norm(dr.grad(vertices))[0]),
                "sdf_z": float(dr.grad(sdf_position)[2][0]),
                "surfel_z": float(dr.grad(center)[2][0]),
            }
            dr.clear_grad(opacity)
            surfel_ray = rt.RayAD(
                ad.Array3f(ad.Float([2.0]), ad.Float([0.0]), ad.Float([-2.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([1.0])),
            )
            surfel_transmission = scene.transmittance(surfel_ray)
            dr.backward(dr.sum(surfel_transmission))
            dr.enable_grad(center)
            forward_hit = scene.intersect(surfel_ray)
            dr.set_grad(center.z, ad.Float([1.0]))
            dr.forward(center.z)
            sdf_ray = rt.RayAD(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-2.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([1.0])),
            )
            dr.enable_grad(sdf_position)
            sdf_forward_hit = scene.intersect(sdf_ray)
            dr.set_grad(sdf_position.z, ad.Float([1.0]))
            dr.forward(sdf_position.z)
            print(json.dumps({
                "shape_ids": [int(hit.shape_id[i]) for i in range(3)],
                "global_ids": [int(hit.global_prim_id[i]) for i in range(3)],
                "minimal_field_norms": [
                    float(dr.norm(minimal.p)[0]), float(dr.norm(minimal.n)[0]), float(dr.norm(minimal.geo_n)[0])
                ],
                "masked_ids": [int(masked_hit.global_prim_id[i]) for i in range(3)],
                "masked_visible": [bool(masked_visibility[i]) for i in range(3)],
                "masked_transmission": [float(masked_transmission[i]) for i in range(3)],
                "t": [float(hit.t[i]) for i in range(3)],
                "visible": [bool(visibility[i]) for i in range(3)],
                "transmission": [float(transmission[i]) for i in range(3)],
                "reflection_ids": [int(chain.global_prim_ids[i]) for i in range(3)],
                "combined_gradients": combined_gradients,
                "reflection_t_grad_enabled": reflection_t_grad_enabled,
                "opacity_gradient": float(dr.grad(opacity)[0]),
                "surfel_center_forward_tangent": float(dr.grad(forward_hit.t)[0]),
                "sdf_position_forward_tangent": float(dr.grad(sdf_forward_hit.t)[0]),
                "has_diffraction": hasattr(scene, "trace_dfr_paths") or hasattr(scene, "trace_diffraction"),
            }))
            """
        )
        self.assertEqual(data["shape_ids"], [0, 1, 2])
        self.assertEqual(data["global_ids"], [0, 1, 2])
        self.assertEqual(data["minimal_field_norms"], [0.0, 0.0, 0.0])
        self.assertEqual(data["masked_ids"], [0, -1, 2])
        self.assertEqual(data["masked_visible"], [False, False, False])
        self.assertAlmostEqual(data["masked_transmission"][0], 0.0, places=5)
        self.assertAlmostEqual(data["masked_transmission"][1], 1.0, places=5)
        self.assertAlmostEqual(data["masked_transmission"][2], 0.5, places=4)
        for value in data["t"]:
            self.assertAlmostEqual(value, 2.0, places=3)
        self.assertEqual(data["visible"], [False, False, False])
        self.assertAlmostEqual(data["transmission"][0], 0.0, places=5)
        self.assertAlmostEqual(data["transmission"][1], 1.0, places=5)
        self.assertAlmostEqual(data["transmission"][2], 0.5, places=4)
        self.assertEqual(data["reflection_ids"], [0, 1, 2])
        self.assertTrue(data["reflection_t_grad_enabled"])
        self.assertGreater(data["combined_gradients"]["mesh"], 0.2)
        self.assertAlmostEqual(data["combined_gradients"]["sdf_z"], 2.0, places=3)
        self.assertAlmostEqual(data["combined_gradients"]["surfel_z"], 2.0, places=4)
        self.assertAlmostEqual(data["opacity_gradient"], -1.0, places=4)
        self.assertAlmostEqual(data["surfel_center_forward_tangent"], 1.0, places=4)
        self.assertAlmostEqual(data["sdf_position_forward_tangent"], 1.0, places=3)
        self.assertFalse(data["has_diffraction"])


if __name__ == "__main__":
    unittest.main()
