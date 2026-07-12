import unittest

import torch
import rayd.torch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchTopKNearestEdgeTests(unittest.TestCase):
    @staticmethod
    def make_scene(*, requires_grad=False):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            device="cuda",
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        faces = torch.tensor(
            [[0, 1, 2], [0, 2, 3]], device="cuda", dtype=torch.int32
        )
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
        scene.build()
        return scene, vertices

    def test_forward_is_sorted_deterministic_and_has_public_shapes(self):
        scene, _ = self.make_scene()
        point = torch.tensor(
            [[0.5, 0.5, 0.25], [0.5, 0.5, 0.25]],
            device="cuda",
            dtype=torch.float32,
        )
        result = scene.nearest_edges(point, 4)
        self.assertEqual(result.query_count, 2)
        self.assertEqual(result.k, 4)
        self.assertEqual(tuple(result.distances.shape), (2, 4))
        self.assertEqual(tuple(result.points.shape), (2, 4, 3))
        self.assertEqual(tuple(result.edge_points.shape), (2, 4, 3))
        self.assertTrue((result.distances[:, 1:] >= result.distances[:, :-1]).all().item())
        torch.testing.assert_close(result.global_edge_ids[0], result.global_edge_ids[1])
        equal_distance = result.distances[:, 1:] == result.distances[:, :-1]
        tied_ids_are_ordered = result.global_edge_ids[:, 1:] > result.global_edge_ids[:, :-1]
        self.assertTrue((~equal_distance | tied_ids_are_ordered).all().item())

    def test_active_invalid_empty_and_k_validation(self):
        scene, _ = self.make_scene()
        point = torch.tensor(
            [[float("nan"), 0.0, 0.0], [0.25, -0.2, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        active = torch.tensor([True, False], device="cuda")
        result = scene.nearest_edges(point, 2, active=active)
        self.assertFalse(result.is_valid.any().item())
        self.assertTrue(torch.isinf(result.distances).all().item())
        empty = scene.nearest_edges(point[:0], 2)
        self.assertEqual(tuple(empty.distances.shape), (0, 2))
        with self.assertRaisesRegex(ValueError, r"\[1, 16\]"):
            scene.nearest_edges(point, 0)
        with self.assertRaisesRegex(ValueError, r"\[1, 16\]"):
            scene.nearest_edges(point, 17)

    def test_edge_mask_filters_winners_without_rebuilding_scene(self):
        scene, _ = self.make_scene()
        point = torch.tensor([[0.5, -0.1, 0.0]], device="cuda")
        before = scene.nearest_edges(point, 2)
        masked = before.global_edge_ids[0, 0]
        edge_mask = scene.edge_mask()
        edge_mask[masked] = False
        version = scene.version
        scene.set_edge_mask(edge_mask)
        after = scene.nearest_edges(point, 2)
        self.assertGreater(scene.version, version)
        self.assertFalse((after.global_edge_ids == masked).any().item())

    def test_fixed_winner_vjp_and_jvp_shapes(self):
        scene, vertices = self.make_scene(requires_grad=True)
        point = torch.tensor(
            [[0.2, -0.25, 0.1], [0.8, 1.25, -0.1]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        result = scene.nearest_edges(point, 2)
        (result.distances.sum() + result.edge_points.sum() + result.points.sum()).backward()
        self.assertEqual(vertices.grad.shape, vertices.shape)
        self.assertEqual(point.grad.shape, point.shape)
        self.assertTrue(torch.isfinite(vertices.grad).all().item())
        self.assertTrue(torch.isfinite(point.grad).all().item())

        detached_point = point.detach()
        tangent = torch.ones_like(detached_point)

        def query(value):
            return scene.nearest_edges(value, 2).distances

        primal, jvp = torch.func.jvp(query, (detached_point,), (tangent,))
        self.assertEqual(primal.shape, (2, 2))
        self.assertEqual(jvp.shape, (2, 2))
        self.assertTrue(torch.isfinite(jvp).all().item())

    def test_torch_compile_matches_eager(self):
        scene, _ = self.make_scene()
        point = torch.tensor([[0.2, 0.3, 0.4]], device="cuda")

        def query(value):
            return scene.nearest_edges(value, 3).distances

        eager = query(point)
        compiled = torch.compile(query, backend="eager")(point)
        torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    unittest.main()
