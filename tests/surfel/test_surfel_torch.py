# Copyright Xingyu Chen.
# Tests Torch surfel LOS, reflection, transmission, and fixed-winner AD.

from __future__ import annotations

import unittest

import torch


try:
    import rayd.torch as rt
except (ImportError, RuntimeError):
    rt = None


@unittest.skipUnless(torch.cuda.is_available() and rt is not None and rt._NATIVE_AVAILABLE, "CUDA Torch RayD required")
class TorchSurfelTests(unittest.TestCase):
    def setUp(self) -> None:
        device = torch.device("cuda")
        self.center = torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float32, device=device, requires_grad=True)
        self.tangent_u = torch.tensor(((1.0, 0.0, 0.0),), dtype=torch.float32, device=device)
        self.tangent_v = torch.tensor(((0.0, 1.0, 0.0),), dtype=torch.float32, device=device)
        self.opacity = torch.tensor((0.5,), dtype=torch.float32, device=device)
        self.value = torch.tensor((2.0,), dtype=torch.float32, device=device)
        cloud = rt.SurfelCloud(self.center, self.tangent_u, self.tangent_v, self.opacity, self.value)
        self.scene = rt.SurfelScene(cloud)
        self.scene.build()

    def test_intersection_los_reflection_and_ad(self) -> None:
        origin = torch.tensor(((0.0, 0.0, -1.0),), dtype=torch.float32, device="cuda")
        direction = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda")
        ray = rt.Ray(origin, direction)
        hit = self.scene.intersect(ray)
        self.assertEqual(hit.surfel_id.tolist(), [0])
        self.assertAlmostEqual(hit.t.item(), 1.0, places=5)
        self.assertAlmostEqual(hit.alpha.item(), 0.5, places=5)
        hit.t.sum().backward()
        self.assertAlmostEqual(self.center.grad[0, 2].item(), 1.0, places=5)

        crossing = self.scene.visible(origin, torch.tensor(((0.0, 0.0, 1.0),), device="cuda"))
        clear = self.scene.visible(
            torch.tensor(((5.0, 0.0, -1.0),), device="cuda"), torch.tensor(((5.0, 0.0, 1.0),), device="cuda")
        )
        self.assertEqual(crossing.tolist(), [False])
        self.assertEqual(clear.tolist(), [True])

        chain = self.scene.trace_reflections(ray, 1)
        self.assertEqual(chain.valid.tolist(), [[True]])
        self.assertEqual(chain.prim_ids.tolist(), [[0]])
        self.assertAlmostEqual(chain.t[0, 0].item(), 1.0, places=5)

    def test_surfel_close_behind_a_rejected_one_is_found(self) -> None:
        # Surfel 0 is crossed at 2 sigma with opacity 0.01, so its candidate is rejected (alpha < alpha_min); surfel 1
        # lies 5e-4 behind it at its centre. The proxy march must resume closer than that, also far from the origin.
        tangent_u = torch.tensor(((1.0, 0.0, 0.0),) * 2, dtype=torch.float32, device="cuda")
        tangent_v = torch.tensor(((0.0, 1.0, 0.0),) * 2, dtype=torch.float32, device="cuda")
        opacity = torch.tensor((0.01, 0.9), dtype=torch.float32, device="cuda")
        value = torch.tensor((1.0, 2.0), dtype=torch.float32, device="cuda")
        for offset in (0.0, 30.0):
            with self.subTest(offset=offset):
                center = torch.tensor(((-2.0, 0.0, offset), (0.0, 0.0, offset + 5.0e-4)), device="cuda")
                scene = rt.SurfelScene(rt.SurfelCloud(center, tangent_u, tangent_v, opacity, value))
                scene.build()
                origin = torch.tensor(((0.0, 0.0, offset - 1.0),), dtype=torch.float32, device="cuda")
                direction = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda")
                hit = scene.intersect(rt.Ray(origin, direction))
                self.assertEqual(hit.surfel_id.tolist(), [1])
                self.assertAlmostEqual(hit.t.item(), 1.0005, places=4)

    def test_alpha_transmission_keeps_coplanar_surfels(self) -> None:
        center = self.center.detach().repeat(2, 1).contiguous()
        tangent_u = self.tangent_u.repeat(2, 1).contiguous()
        tangent_v = self.tangent_v.repeat(2, 1).contiguous()
        opacity = torch.tensor((0.5, 0.5), dtype=torch.float32, device="cuda")
        value = torch.tensor((2.0, 4.0), dtype=torch.float32, device="cuda")
        scene = rt.SurfelScene(
            rt.SurfelCloud(center, tangent_u, tangent_v, opacity, value),
            rt.SurfelTraceOptions(max_candidate_hits=2, transmittance_min=0.0),
        )
        scene.build()
        ray = rt.Ray(
            torch.tensor(((0.0, 0.0, -1.0),), dtype=torch.float32, device="cuda"),
            torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda"),
        )
        composite = scene.composite_alpha(ray)
        self.assertAlmostEqual(composite.transmittance.item(), 0.25, places=5)
        self.assertAlmostEqual(composite.alpha.item(), 0.75, places=5)
        self.assertAlmostEqual(composite.intensity.item(), 2.0, places=5)
        self.assertEqual(composite.candidate_count.tolist(), [2])
        self.assertEqual(composite.candidate_buffer_full.tolist(), [True])
        self.assertAlmostEqual(scene.transmittance(ray).item(), 0.25, places=5)
        self.assertFalse(hasattr(scene, "trace_diffraction"))


if __name__ == "__main__":
    unittest.main()
