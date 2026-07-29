# Copyright Xingyu Chen.
# Tests cuda multipath.

import unittest

import torch
import rayd.torch as rt


def _scene(backend: str):
    vertices = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene(trace_backend=backend, edge_bvh_backend=backend)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


def _dfr_inputs(power: float = 1.0):
    states = rt.DfrStates(
        edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
        edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device="cuda"),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        edge_t_min=torch.tensor([-1.0], device="cuda"),
        edge_t_max=torch.tensor([1.0], device="cuda"),
        n0=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        n1=torch.tensor([[0.0, 0.0, -1.0]], device="cuda"),
        prim0=torch.tensor([0], device="cuda", dtype=torch.int32),
        prim1=torch.tensor([0], device="cuda", dtype=torch.int32),
        exterior_angle=torch.tensor([torch.pi], device="cuda"),
        src=torch.tensor([[0.0, -1.0, 0.25]], device="cuda"),
        src_power=torch.tensor([power], device="cuda"),
    )
    grid = rt.DfrGrid(
        axis=2,
        position=0.5,
        coord0_min=-2.0,
        coord0_max=2.0,
        coord1_min=-2.0,
        coord1_max=2.0,
        resolution0=8,
        resolution1=8,
    )
    material = rt.DfrMaterial(
        eta_r=torch.ones(1, device="cuda"),
        sigma=torch.zeros(1, device="cuda"),
        mu_r=torch.ones(1, device="cuda"),
        gain=torch.ones(1, device="cuda"),
        valid=torch.ones(1, device="cuda", dtype=torch.bool),
    )
    return states, grid, material


def _accumulate(scene, states, grid, material, *, direct=256, suffix=0):
    return scene.accum_dfr_direct(
        states=states,
        grid=grid,
        material=material,
        wavelength=1.0,
        direct_samples=direct,
        suffix_samples=suffix,
        seed=17,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class CudaMultipathParityTests(unittest.TestCase):
    def test_reflection_trace_nonzero_matches_optix(self):
        cuda_scene = _scene("cuda")
        try:
            optix_scene = _scene("optix")
        except RuntimeError as error:
            self.skipTest(f"OptiX is unavailable: {error}")
        ray = rt.Ray(
            torch.tensor([[0.0, 0.0, -1.0], [0.2, 0.1, -1.0]], device="cuda"),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda"),
        )
        cuda = cuda_scene.trace_reflections(ray, max_bounces=1)
        optix = optix_scene.trace_reflections(ray, max_bounces=1)
        self.assertTrue(torch.equal(cuda.valid, optix.valid))
        self.assertTrue(torch.equal(cuda.prim_ids, optix.prim_ids))
        torch.testing.assert_close(cuda.t, optix.t, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(cuda.image_sources, optix.image_sources, atol=2e-6, rtol=2e-6)

    def test_reflection_accumulation_nonzero_matches_optix(self):
        cuda_scene = _scene("cuda")
        try:
            optix_scene = _scene("optix")
        except RuntimeError as error:
            self.skipTest(f"OptiX is unavailable: {error}")
        ray_count = 64
        ray_o = torch.zeros((ray_count, 3), device="cuda")
        ray_o[:, 2] = -1.0
        ray_d = torch.zeros_like(ray_o)
        ray_d[:, 2] = 1.0
        ray_tmax = torch.full((ray_count,), 4.0, device="cuda")
        active = torch.ones(ray_count, device="cuda", dtype=torch.bool)
        tx_pol = torch.zeros_like(ray_o)
        tx_pol[:, 0] = 1.0
        one = torch.ones(1, device="cuda")
        zero = torch.zeros(1, device="cuda")
        valid = torch.ones(1, device="cuda", dtype=torch.bool)

        def run(scene):
            return torch.ops.rayd_torch.reflection_accumulation_forward(
                scene._native_scene,
                ray_o,
                ray_d,
                ray_tmax,
                active,
                ray_o,
                tx_pol,
                one,
                zero,
                one,
                one,
                valid,
                1,
                2,
                0.0,
                -2.0,
                2.0,
                -2.0,
                2.0,
                8,
                8,
                1.0,
                1.0,
                False,
                False,
                0,
                1,
                1,
                0,
                0,
                0,
                True,
            )

        cuda = run(cuda_scene)
        optix = run(optix_scene)
        self.assertGreater(float(cuda[0].sum().item()), 0.0)
        self.assertTrue(torch.equal(cuda[7], optix[7]))
        for index in range(7):
            torch.testing.assert_close(cuda[index], optix[index], atol=2e-6, rtol=2e-6)

    def test_diffraction_direct_and_unreachable_suffix_match_optix(self):
        cuda_scene = _scene("cuda")
        try:
            optix_scene = _scene("optix")
        except RuntimeError as error:
            self.skipTest(f"OptiX is unavailable: {error}")
        states, grid, material = _dfr_inputs()
        cuda = _accumulate(cuda_scene, states, grid, material, suffix=4)
        optix = _accumulate(optix_scene, states, grid, material, suffix=4)

        self.assertGreater(int(cuda.direct_count.sum().item()), 0)
        self.assertEqual(int(cuda.suffix_count.sum().item()), 0)
        self.assertTrue(torch.equal(cuda.direct_count, optix.direct_count))
        self.assertTrue(torch.equal(cuda.keller_count, optix.keller_count))
        self.assertTrue(torch.equal(cuda.suffix_count, optix.suffix_count))
        for name in ("power", "field_x_re", "field_x_im", "field_y_re", "field_y_im", "field_z_re", "field_z_im"):
            torch.testing.assert_close(getattr(cuda, name), getattr(optix, name), atol=2e-8, rtol=2e-6)

    def _assert_diffraction_params_are_isolated_across_streams(self, backend: str):
        try:
            scene_a = _scene(backend)
            scene_b = _scene(backend)
        except RuntimeError as error:
            self.skipTest(f"{backend} is unavailable: {error}")
        states_a, grid, material = _dfr_inputs(1.0)
        states_b, _, _ = _dfr_inputs(9.0)
        reference_a = _accumulate(scene_a, states_a, grid, material, direct=2048)
        reference_b = _accumulate(scene_b, states_b, grid, material, direct=2048)
        torch.cuda.synchronize()

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        concurrent_a_results = []
        concurrent_b_results = []
        # Reuse the bounded OptiX launch-param ring more than once while the
        # two streams carry different scene parameters.
        for _ in range(8):
            with torch.cuda.stream(stream_a):
                concurrent_a_results.append(_accumulate(scene_a, states_a, grid, material, direct=2048))
            with torch.cuda.stream(stream_b):
                concurrent_b_results.append(_accumulate(scene_b, states_b, grid, material, direct=2048))
        stream_a.synchronize()
        stream_b.synchronize()

        for concurrent_a, concurrent_b in zip(concurrent_a_results, concurrent_b_results):
            self.assertTrue(torch.equal(concurrent_a.direct_count, reference_a.direct_count))
            self.assertTrue(torch.equal(concurrent_b.direct_count, reference_b.direct_count))
            torch.testing.assert_close(concurrent_a.power, reference_a.power, atol=0, rtol=0)
            torch.testing.assert_close(concurrent_b.power, reference_b.power, atol=0, rtol=0)
            torch.testing.assert_close(concurrent_a.field_x_re, reference_a.field_x_re, atol=0, rtol=0)
            torch.testing.assert_close(concurrent_b.field_x_re, reference_b.field_x_re, atol=0, rtol=0)
        concurrent_a = concurrent_a_results[-1]
        concurrent_b = concurrent_b_results[-1]
        self.assertNotEqual(float(concurrent_a.power.sum().item()), float(concurrent_b.power.sum().item()))

    def test_diffraction_params_are_isolated_across_cuda_streams(self):
        self._assert_diffraction_params_are_isolated_across_streams("cuda")

    def test_diffraction_params_are_isolated_across_optix_streams(self):
        self._assert_diffraction_params_are_isolated_across_streams("optix")


if __name__ == "__main__":
    unittest.main()
