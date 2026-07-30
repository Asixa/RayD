# Copyright Xingyu Chen.
# Tests Torch mixed mesh, SDF, and surfel queries, gradients, and operation scope.

from __future__ import annotations

import unittest

import torch
from torch.autograd import forward_ad


try:
    import rayd.torch as rt
except (ImportError, RuntimeError):
    rt = None


@unittest.skipUnless(torch.cuda.is_available() and rt is not None and rt._NATIVE_AVAILABLE, "CUDA Torch RayD required")
class TorchMixedSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        device = torch.device("cuda")
        self.mesh_vertices = torch.tensor(
            ((-2.75, -0.75, 0.0), (-1.25, -0.75, 0.0), (-2.0, 0.75, 0.0)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        faces = torch.tensor(((0, 1, 2),), dtype=torch.int32, device=device)
        resolution = 16
        axis = torch.linspace(-0.5, 0.5, resolution, dtype=torch.float32, device=device)
        _x, _y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
        self.sdf_position = torch.zeros((3,), dtype=torch.float32, device=device, requires_grad=True)
        self.sdf = rt.SdfGrid(
            z.contiguous(),
            self.sdf_position,
            torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=device),
            torch.ones((3,), dtype=torch.float32, device=device),
        )
        self.surfel_center = torch.tensor(((2.0, 0.0, 0.0),), dtype=torch.float32, device=device, requires_grad=True)
        self.opacity = torch.tensor((0.5,), dtype=torch.float32, device=device, requires_grad=True)
        self.cloud = rt.SurfelCloud(
            self.surfel_center,
            torch.tensor(((0.25, 0.0, 0.0),), dtype=torch.float32, device=device),
            torch.tensor(((0.0, 0.25, 0.0),), dtype=torch.float32, device=device),
            self.opacity,
            torch.ones((1,), dtype=torch.float32, device=device),
        )
        self.scene = rt.MixedScene()
        self.scene.add_mesh(rt.Mesh(self.mesh_vertices, faces, edges_enabled=False))
        self.scene.add_sdf(self.sdf)
        self.scene.add_surfel(self.cloud, rt.SurfelTraceOptions(max_candidate_hits=1, transmittance_min=0.0))
        self.scene.build()

    @staticmethod
    def _ray() -> rt.Ray:
        origin = torch.tensor(
            ((-2.0, 0.0, -2.0), (0.0, 0.0, -2.0), (2.0, 0.0, -2.0)), dtype=torch.float32, device="cuda"
        )
        direction = torch.tensor(((0.0, 0.0, 1.0),) * 3, dtype=torch.float32, device="cuda")
        return rt.Ray(origin, direction)

    def _sdf_batch(self, positions: tuple[tuple[float, float, float], ...]) -> rt.SdfGridBatch:
        count = len(positions)
        return rt.SdfGridBatch(
            torch.stack((self.sdf.values.detach(),) * count).contiguous(),
            torch.tensor(positions, dtype=torch.float32, device="cuda"),
            self.sdf.rotation.detach().expand(count, -1).contiguous(),
            self.sdf.scale.detach().expand(count, -1).contiguous(),
        )

    def test_all_geometry_families_share_one_closest_hit_and_reflection_query(self) -> None:
        ray = self._ray()
        hit = self.scene.intersect(ray)
        self.assertEqual(hit.shape_id.tolist(), [0, 1, 2])
        self.assertEqual(hit.global_prim_id.tolist(), [0, 1, 2])
        torch.testing.assert_close(hit.t, torch.full_like(hit.t, 2.0), atol=2.0e-3, rtol=0.0)
        self.assertEqual(self.scene.visible(ray.o, ray.o + 4.0 * ray.d).tolist(), [False, False, False])
        transmission = self.scene.transmittance(ray)
        torch.testing.assert_close(transmission, torch.tensor((0.0, 1.0, 0.5), device="cuda"), atol=2.0e-4, rtol=0.0)
        chain = self.scene.trace_reflections(ray, 1)
        self.assertEqual(chain.valid.tolist(), [[True], [True], [True]])
        self.assertEqual(chain.prim_ids.tolist(), [[0], [1], [2]])
        self.assertFalse(hasattr(self.scene, "trace_dfr_paths"))
        self.assertFalse(hasattr(self.scene, "trace_diffraction"))
        minimal = self.scene.intersect(ray, flags=getattr(rt.RayFlags, "None"))
        torch.testing.assert_close(minimal.p, torch.zeros_like(minimal.p))
        torch.testing.assert_close(minimal.n, torch.zeros_like(minimal.n))
        torch.testing.assert_close(minimal.geo_n, torch.zeros_like(minimal.geo_n))
        self.assertEqual(minimal.global_prim_id.tolist(), [0, 1, 2])
        active = torch.tensor((True, False, True), dtype=torch.bool, device="cuda")
        self.assertEqual(self.scene.intersect(ray, active).global_prim_id.tolist(), [0, -1, 2])
        self.assertEqual(self.scene.visible(ray.o, ray.o + 4.0 * ray.d, active).tolist(), [False, False, False])
        torch.testing.assert_close(
            self.scene.transmittance(ray, active), torch.tensor((0.0, 1.0, 0.5), device="cuda"), atol=2.0e-4, rtol=0.0
        )

    def test_fixed_winners_propagate_geometry_and_transmission_gradients(self) -> None:
        hit = self.scene.intersect(self._ray())
        hit.t.sum().backward()
        self.assertGreater(torch.linalg.vector_norm(self.mesh_vertices.grad).item(), 0.1)
        self.assertAlmostEqual(self.sdf_position.grad[2].item(), 1.0, places=3)
        self.assertAlmostEqual(self.surfel_center.grad[0, 2].item(), 1.0, places=4)

        self.mesh_vertices.grad = None
        self.sdf_position.grad = None
        self.surfel_center.grad = None
        self.scene.trace_reflections(self._ray(), 1).t.sum().backward()
        self.assertGreater(torch.linalg.vector_norm(self.mesh_vertices.grad).item(), 0.1)
        self.assertAlmostEqual(self.sdf_position.grad[2].item(), 1.0, places=3)
        self.assertAlmostEqual(self.surfel_center.grad[0, 2].item(), 1.0, places=4)

        self.opacity.grad = None
        surfel_ray = rt.Ray(self._ray().o[2:3].contiguous(), self._ray().d[2:3].contiguous())
        self.scene.transmittance(surfel_ray).sum().backward()
        self.assertAlmostEqual(self.opacity.grad.item(), -1.0, places=4)

    def test_ray_forward_ad_crosses_sdf_and_surfel(self) -> None:
        scene = rt.MixedScene()
        scene.add_sdf(rt.SdfGrid(self.sdf.values, self.sdf.position.detach(), self.sdf.rotation, self.sdf.scale))
        scene.add_surfel(
            rt.SurfelCloud(
                self.cloud.center.detach(),
                self.cloud.tangent_u,
                self.cloud.tangent_v,
                self.cloud.opacity.detach(),
                self.cloud.value,
            ),
            rt.SurfelTraceOptions(max_candidate_hits=1),
        )
        scene.build()
        primal = self._ray().o[1:].contiguous()
        direction = self._ray().d[1:].contiguous()
        tangent = torch.zeros_like(primal)
        tangent[:, 2] = 1.0
        with torch.autograd.forward_ad.dual_level():
            dual_origin = torch.autograd.forward_ad.make_dual(primal, tangent)
            hit = scene.intersect(rt.Ray(dual_origin, direction))
            unpacked = torch.autograd.forward_ad.unpack_dual(hit.t)
            torch.testing.assert_close(unpacked.tangent, torch.full_like(primal[:, 0], -1.0), atol=2.0e-3, rtol=0.0)

    def test_sdf_oriented_bbox_rejects_outside_rays(self) -> None:
        scene = rt.MixedScene()
        scene.add_sdf(self.sdf)
        scene.build()
        ray = rt.Ray(
            torch.tensor(((2.0, 0.0, -2.0),), dtype=torch.float32, device="cuda"),
            torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda"),
        )
        self.assertFalse(scene.intersect(ray).is_valid().item())

    def test_packed_sdf_batch_matches_individual_grids_and_preserves_ids(self) -> None:
        values = torch.stack((self.sdf.values.detach(), self.sdf.values.detach())).contiguous()
        positions = torch.tensor(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=torch.float32, device="cuda")
        rotations = self.sdf.rotation.detach().expand(2, -1).contiguous()
        scales = self.sdf.scale.detach().expand(2, -1).contiguous()
        batch = rt.SdfGridBatch(values, positions, rotations, scales)
        ray = rt.Ray(
            torch.tensor(((-1.0, 0.0, -2.0), (1.0, 0.0, -2.0)), dtype=torch.float32, device="cuda"),
            torch.tensor(((0.0, 0.0, 1.0),) * 2, dtype=torch.float32, device="cuda"),
        )
        batched = batch.intersect(ray)
        individual = tuple(batch.grid(index).intersect(ray) for index in range(2))
        for batch_hit, individual_hit in zip(batched, individual):
            for field in ("t", "hit_mask", "position", "normal", "steps"):
                self.assertTrue(torch.equal(getattr(batch_hit, field), getattr(individual_hit, field)), field)

        scene = rt.MixedScene()
        self.assertEqual(scene.add_sdf_batch(batch), 0)
        scene.build()
        hit = scene.intersect(ray)
        self.assertEqual(hit.shape_id.tolist(), [0, 1])
        self.assertEqual(hit.global_prim_id.tolist(), [0, 1])

    def test_packed_sdf_batch_ad_falls_back_to_fixed_winner_per_grid(self) -> None:
        values = torch.stack((self.sdf.values.detach(), self.sdf.values.detach())).contiguous()
        positions = torch.tensor(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float32, device="cuda", requires_grad=True
        )
        rotations = self.sdf.rotation.detach().expand(2, -1).contiguous()
        scales = self.sdf.scale.detach().expand(2, -1).contiguous()
        scene = rt.MixedScene()
        scene.add_sdf_batch(rt.SdfGridBatch(values, positions, rotations, scales))
        scene.build()
        origin = torch.tensor(((0.0, 0.0, -2.0),), dtype=torch.float32, device="cuda")
        direction = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda")
        scene.intersect(rt.Ray(origin, direction)).t.sum().backward()
        torch.testing.assert_close(positions.grad[0], torch.tensor((0.0, 0.0, 1.0), device="cuda"), atol=2e-3, rtol=0.0)
        self.assertTrue(torch.equal(positions.grad[1], torch.zeros(3, device="cuda")))

        tangent = torch.zeros_like(origin)
        tangent[:, 2] = 1.0
        with forward_ad.dual_level():
            dual_origin = forward_ad.make_dual(origin, tangent)
            hit = scene.intersect(rt.Ray(dual_origin, direction))
            unpacked = forward_ad.unpack_dual(hit.t)
            torch.testing.assert_close(unpacked.tangent, torch.tensor((-1.0,), device="cuda"), atol=2e-3, rtol=0.0)

    def test_packed_sdf_batch_honors_empty_active_and_per_ray_tmax(self) -> None:
        batch = self._sdf_batch(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        empty3 = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        empty_hits = batch.intersect(rt.Ray(empty3, empty3))
        self.assertEqual([tuple(hit.t.shape) for hit in empty_hits], [(0,), (0,)])

        ray = rt.Ray(
            torch.tensor(((-1.0, 0.0, -2.0),) * 3, dtype=torch.float32, device="cuda"),
            torch.tensor(((0.0, 0.0, 1.0),) * 3, dtype=torch.float32, device="cuda"),
            torch.tensor((3.0, 3.0, 1.0), dtype=torch.float32, device="cuda"),
        )
        first, second = batch.intersect(ray, active=torch.tensor((True, False, True), device="cuda"))
        self.assertEqual(first.hit_mask.tolist(), [True, False, False])
        self.assertEqual(second.hit_mask.tolist(), [False, False, False])
        torch.testing.assert_close(first.position[1:], torch.zeros_like(first.position[1:]))
        self.assertEqual(int(first.steps[1].item()), 0)
        self.assertTrue(torch.isinf(first.t[1:]).all())

    def test_packed_sdf_batch_keeps_mixed_ids_and_services_reflection_and_los(self) -> None:
        scene = rt.MixedScene()
        faces = torch.tensor(((0, 1, 2),), dtype=torch.int32, device="cuda")
        scene.add_mesh(rt.Mesh(self.mesh_vertices.detach(), faces, edges_enabled=False))
        scene.add_sdf(
            rt.SdfGrid(
                self.sdf.values.detach(),
                torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device="cuda"),
                self.sdf.rotation,
                self.sdf.scale,
            )
        )
        scene.add_sdf_batch(self._sdf_batch(((2.0, 0.0, 0.0), (4.0, 0.0, 0.0))))
        scene.add_surfel(
            rt.SurfelCloud(
                torch.tensor(((6.0, 0.0, 0.0),), dtype=torch.float32, device="cuda"),
                self.cloud.tangent_u,
                self.cloud.tangent_v,
                self.cloud.opacity.detach(),
                self.cloud.value,
            ),
            rt.SurfelTraceOptions(max_candidate_hits=1),
        )
        scene.build()
        xs = (-2.0, 0.0, 2.0, 4.0, 6.0)
        ray = rt.Ray(
            torch.tensor(tuple((x, 0.0, -2.0) for x in xs), dtype=torch.float32, device="cuda"),
            torch.tensor(((0.0, 0.0, 1.0),) * len(xs), dtype=torch.float32, device="cuda"),
        )
        hit = scene.intersect(ray)
        self.assertEqual(hit.shape_id.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(hit.global_prim_id.tolist(), [0, 1, 2, 3, 4])
        chain = scene.trace_reflections(ray, 1)
        self.assertEqual(chain.valid[:, 0].tolist(), [True] * len(xs))
        self.assertEqual(chain.prim_ids[:, 0].tolist(), [0, 1, 2, 3, 4])
        start = ray.o[2:4].contiguous()
        end = (start + 4.0 * ray.d[2:4]).contiguous()
        self.assertEqual(scene.visible(start, end).tolist(), [False, False])

    def test_packed_sdf_batch_uses_one_kernel_for_primal_and_reflection(self) -> None:
        batch = self._sdf_batch(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        ray = rt.Ray(
            torch.tensor(((-1.0, 0.0, -2.0), (1.0, 0.0, -2.0)), dtype=torch.float32, device="cuda"),
            torch.tensor(((0.0, 0.0, 1.0),) * 2, dtype=torch.float32, device="cuda"),
        )

        def kernel_count(profile, name: str) -> int:
            return sum(event.count for event in profile.key_averages() if name in event.key)

        activities = (torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile:
            batch.intersect(ray)
            torch.cuda.synchronize()
        self.assertEqual(kernel_count(profile, "sdf_batch_intersect_forward_kernel"), 1)

        scene = rt.MixedScene()
        scene.add_sdf_batch(batch)
        scene.build()
        with torch.profiler.profile(activities=activities) as profile:
            scene.trace_reflections(ray, 1)
            torch.cuda.synchronize()
        self.assertEqual(kernel_count(profile, "sdf_batch_intersect_forward_kernel"), 1)

        with torch.profiler.profile(activities=activities) as profile:
            scene.visible(ray.o, (ray.o + 4.0 * ray.d).contiguous())
            torch.cuda.synchronize()
        self.assertEqual(kernel_count(profile, "sdf_batch_intersect_forward_kernel"), 0)
        self.assertEqual(kernel_count(profile, "sdf_intersect_forward_kernel"), 2)


if __name__ == "__main__":
    unittest.main()
