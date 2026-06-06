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

    def test_reflection_epc_field_backward_reaches_vertices(self):
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
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        receiver = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        out = scene.trace_refl_epc_field(source, receiver, max_bounces=1)
        loss = out.field_real.sum() + out.field_imag.sum()
        loss.backward()
        self.assertIsNotNone(verts.grad)

    def test_dfr_direct_accum_backward_reaches_state_tensors(self):
        scene = rt.Scene()
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        src = torch.tensor([[0.0, -1.0, 0.2]], device="cuda", dtype=torch.float32, requires_grad=True)
        out = scene.accum_dfr_direct(edge_pos=edge_pos, edge_dir=edge_dir, src=src)
        out.power.sum().backward()
        self.assertIsNotNone(edge_pos.grad)
        self.assertIsNotNone(edge_dir.grad)
        self.assertIsNotNone(src.grad)
