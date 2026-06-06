import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class MultipathTests(unittest.TestCase):
    def test_visibility_returns_bool_tensor(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        start = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        end = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        visible = scene.visible(start, end)
        self.assertEqual(visible.dtype, torch.bool)
        self.assertFalse(bool(visible[0].item()))

    def test_single_reflection_t_has_gradient(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        chain = scene.trace_reflections(ray, max_bounces=1)
        chain.t.sum().backward()
        self.assertIsNotNone(verts.grad)
        self.assertGreater(float(verts.grad.abs().sum().item()), 0.0)
