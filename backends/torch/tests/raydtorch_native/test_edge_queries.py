import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class EdgeQueryTests(unittest.TestCase):
    def test_nearest_edge_point_forward_and_grad(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.5, -0.25, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        result = scene.nearest_edge(point)
        torch.testing.assert_close(result.distance, torch.tensor([0.25], device="cuda"), atol=1e-5, rtol=1e-5)
        result.distance.sum().backward()
        self.assertIsNotNone(point.grad)
        self.assertIsNotNone(verts.grad)

    def test_nearest_edge_point_edge_t_vjp_matches_interior_edge(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.25, 0.2, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        result = scene.nearest_edge(point)
        self.assertEqual(int(result.edge_id[0].item()), 0)
        result.edge_t.sum().backward()
        torch.testing.assert_close(point.grad, torch.tensor([[1.0, 0.0, 0.0]], device="cuda"))

    def test_nearest_edge_point_edge_point_vjp_reaches_query_point(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.25, 0.2, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        result = scene.nearest_edge(point)
        self.assertEqual(int(result.edge_id[0].item()), 0)
        result.edge_point[:, 0].sum().backward()
        torch.testing.assert_close(point.grad, torch.tensor([[1.0, 0.0, 0.0]], device="cuda"))

    def test_nearest_edge_ray_forward(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.5, -0.25, 1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        )
        result = scene.nearest_edge(ray)
        self.assertIsInstance(result, rt.NearestRayEdge)
        torch.testing.assert_close(result.distance, torch.tensor([0.25], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.ray_t, torch.tensor([1.0], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            result.point,
            torch.tensor([[0.5, -0.25, 0.0]], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(result.edge_t, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            result.edge_point,
            torch.tensor([[0.5, 0.0, 0.0]], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(int(result.edge_id[0].item()), 0)

    def test_large_grid_edge_query_returns_finite_distances(self):
        n = 64
        xs, ys = torch.meshgrid(
            torch.linspace(0, 1, n, device="cuda"),
            torch.linspace(0, 1, n, device="cuda"),
            indexing="ij",
        )
        verts = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(n * n, device="cuda")], dim=1).contiguous()
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b = a + 1
                c = a + n
                d = c + 1
                faces.append([a, b, c])
                faces.append([b, d, c])
        faces_t = torch.tensor(faces, device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces_t))
        scene.build()
        q = torch.rand((4096, 3), device="cuda", dtype=torch.float32)
        out = scene.nearest_edge(q)
        self.assertTrue(torch.isfinite(out.distance).all().item())


if __name__ == "__main__":
    unittest.main()
