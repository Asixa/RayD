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


if __name__ == "__main__":
    unittest.main()
