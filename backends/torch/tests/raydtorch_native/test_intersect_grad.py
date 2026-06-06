import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class IntersectGradientTests(unittest.TestCase):
    def test_vertex_gradient_exact_values_through_t(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        its = scene.intersect(ray)
        its.t.sum().backward()
        torch.testing.assert_close(
            verts.grad[:, 2],
            torch.tensor([0.5, 0.25, 0.25], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(verts.grad[:, 0], torch.zeros(3, device="cuda"))
        torch.testing.assert_close(verts.grad[:, 1], torch.zeros(3, device="cuda"))

    def test_ray_origin_gradient_through_t(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        origin = torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        direction = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        its = scene.intersect(rt.Ray(origin, direction))
        its.t.sum().backward()
        torch.testing.assert_close(origin.grad, torch.tensor([[0.0, 0.0, -1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
