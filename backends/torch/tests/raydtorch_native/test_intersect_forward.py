import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class IntersectForwardTests(unittest.TestCase):
    def test_single_triangle_hit_and_miss(self):
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
            torch.tensor([[0.25, 0.25, -1.0], [2.0, 2.0, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        its = scene.intersect(ray)
        torch.testing.assert_close(its.t[0], torch.tensor(1.0, device="cuda"))
        torch.testing.assert_close(its.p[0], torch.tensor([0.25, 0.25, 0.0], device="cuda"))
        torch.testing.assert_close(its.barycentric[0], torch.tensor([0.5, 0.25, 0.25], device="cuda"))
        self.assertEqual(int(its.shape_id[0].item()), 0)
        self.assertEqual(int(its.shape_id[1].item()), -1)
        self.assertTrue(torch.isinf(its.t[1]))


if __name__ == "__main__":
    unittest.main()
