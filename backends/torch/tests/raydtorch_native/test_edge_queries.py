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
