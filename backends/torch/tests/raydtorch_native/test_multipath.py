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

    def test_two_bounce_reflection_trace_fills_subsequent_bounces(self):
        verts = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, -1.0, 2.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[1.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        )
        chain = scene.trace_reflections(ray, max_bounces=2)
        self.assertTrue(bool(chain.valid[0, 0].item()))
        self.assertTrue(bool(chain.valid[0, 1].item()))
        self.assertEqual([int(v) for v in chain.prim_ids[0].tolist()], [0, 1])
        torch.testing.assert_close(chain.t[0], torch.tensor([1.0, 1.0], device="cuda"), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            chain.image_sources[0],
            torch.tensor([[0.0, 0.0, -1.0], [4.0, 0.0, -1.0]], device="cuda"),
            atol=1e-3,
            rtol=1e-3,
        )

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

    def test_reflection_dedup_native_binding_smoke(self):
        ray_count = 2
        max_bounces = 1
        slot_count = ray_count * max_bounces
        device = "cuda"
        bounce_count = torch.ones((ray_count,), device=device, dtype=torch.int32)
        shape_ids = torch.zeros((slot_count,), device=device, dtype=torch.int32)
        prim_ids = torch.zeros((slot_count,), device=device, dtype=torch.int32)
        t = torch.ones((slot_count,), device=device, dtype=torch.float32)
        zeros = torch.zeros((slot_count,), device=device, dtype=torch.float32)
        norm_z = torch.ones((slot_count,), device=device, dtype=torch.float32)
        out = rt._C.reflection_dedup_forward(
            bounce_count,
            shape_ids,
            prim_ids,
            t,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            norm_z,
            zeros,
            zeros,
            zeros,
            max_bounces,
            1e-5,
        )
        unique_count = int(out[0])
        discovery_count = out[-2]
        self.assertEqual(unique_count, 1)
        self.assertEqual(int(discovery_count[0].item()), 2)

    def test_reflection_accumulation_native_binding_smoke(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray_o = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        ray_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        ray_tmax = torch.tensor([2.0], device="cuda", dtype=torch.float32)
        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        tx_pol = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        out = rt._C.reflection_accumulation_forward(
            scene._native_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            ray_o,
            tx_pol,
            1,
            2,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            4,
            4,
            1.0,
        )
        self.assertEqual(out[0].shape, (4, 4))
        self.assertEqual(out[-1].dtype, torch.int32)

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
        out = scene.accum_dfr_legacy_direct(edge_pos=edge_pos, edge_dir=edge_dir, src=src)
        out.power.sum().backward()
        self.assertIsNotNone(edge_pos.grad)
        self.assertIsNotNone(edge_dir.grad)
        self.assertIsNotNone(src.grad)

    def test_diffraction_paths_order1_native_binding_smoke(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        tx_pos = torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32)
        rx_pos = torch.tensor([[0.0, 1.0, 0.25]], device="cuda", dtype=torch.float32)
        state_edge_index = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_t_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
        state_edge_t_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        state_n0 = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        state_n1 = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        state_prim0 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_prim1 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_exterior_angle = torch.tensor([torch.pi], device="cuda", dtype=torch.float32)
        state_src = tx_pos.clone()
        state_src_power = torch.ones((1,), device="cuda", dtype=torch.float32)
        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        material_gain = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_valid = torch.ones((1,), device="cuda", dtype=torch.bool)

        out = rt._C.diffraction_paths_order1_forward(
            scene._native_handle,
            tx_pos,
            rx_pos,
            active,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            8,
            1.0,
        )
        count = int(out[0].item())
        self.assertGreaterEqual(count, 0)
        self.assertEqual(out[1].shape, (8,))
        self.assertEqual(out[8].dtype, torch.float32)

    def test_diffraction_accumulation_native_binding_smoke(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        state_edge_index = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_t_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
        state_edge_t_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        state_n0 = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        state_n1 = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        state_prim0 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_prim1 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_exterior_angle = torch.tensor([torch.pi], device="cuda", dtype=torch.float32)
        state_src = torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32)
        state_src_power = torch.ones((1,), device="cuda", dtype=torch.float32)
        zeros_vec = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
        material_eta_r = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_sigma = torch.zeros((1,), device="cuda", dtype=torch.float32)
        material_mu_r = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_gain = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_valid = torch.ones((1,), device="cuda", dtype=torch.bool)

        out = rt._C.diffraction_accumulation_forward(
            scene._native_handle,
            active,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            zeros_vec,
            zeros_vec,
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            2,
            0.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            4,
            4,
            0.25,
            1.0,
            4,
            0,
        )
        self.assertEqual(out[0].shape, (4, 4))
        self.assertEqual(out[1].dtype, torch.float32)

    def test_scene_accum_dfr_direct_native_api(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        states = rt.DfrStates(
            edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
            edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_t_min=torch.tensor([-1.0], device="cuda", dtype=torch.float32),
            edge_t_max=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            n0=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([0], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([0], device="cuda", dtype=torch.int32),
            exterior_angle=torch.tensor([torch.pi], device="cuda", dtype=torch.float32),
            src=torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32),
            src_power=torch.ones((1,), device="cuda", dtype=torch.float32),
        )
        grid = rt.DfrGrid(
            axis=2,
            position=0.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=4,
            resolution1=4,
        )
        out = scene.accum_dfr_direct(states=states, grid=grid, wavelength=1.0, direct_samples=4)
        self.assertEqual(out.power.shape, (4, 4))
        self.assertEqual(out.field_x_re.dtype, torch.float32)
