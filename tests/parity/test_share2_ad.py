# Copyright Xingyu Chen.
# Tests share2 ad.

"""Cross-backend AD parity for smooth, fixed-discrete-choice public operations.

These tests intentionally keep edge winners, reflection hits, and diffraction
strategy choices away from non-differentiable boundaries. Run them explicitly
with ``RAYD_TORCH_RUN_DR_JIT_PARITY=1`` after both backends are installed.
"""

import importlib
import math
import os
import unittest

import torch

from .test_drjit import (
    _FIELD_ABS,
    _load_backends,
    _rayd_dfr_grid,
    _rayd_dfr_scene,
    _torch_dfr_grid,
    _torch_dfr_scene,
    _torch_dfr_states,
)


_FORWARD_ATOL = 1.0e-5
_GRAD_ATOL = 5.0e-4
_FIELD_GRAD_ATOL = 2.0e-3


def _torch_triangle_scene(rt, vertices):
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


def _drjit_triangle_scene(dr_backend, cuda, vertices=None):
    ad = importlib.import_module("drjit.cuda.ad")
    faces = cuda.Array3i([0], [1], [2])
    if vertices is None:
        vertices = ad.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    mesh = dr_backend.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]), faces)
    mesh.vertex_positions = vertices
    scene = dr_backend.Scene()
    scene.add_mesh(mesh)
    scene.build()
    return scene


def _dr_vec3(value):
    return [float(value[axis][0]) for axis in range(3)]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
@unittest.skipUnless(os.environ.get("RAYD_TORCH_RUN_DR_JIT_PARITY") == "1", "external RayD parity is opt-in")
class Share2AdParityTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, *, delta=_GRAD_ATOL):
        self.assertEqual(len(actual), len(expected))
        for lhs, rhs in zip(actual, expected):
            self.assertAlmostEqual(float(lhs), float(rhs), delta=delta)

    def test_nearest_edge_point_fixed_winner_forward_vjp_jvp(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("drjit")
        ad = importlib.import_module("drjit.cuda.ad")
        base_vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32
        )
        tangent = torch.tensor([[0.07, -0.03, 0.05]], device="cuda")

        point_t = torch.tensor([[0.25, -0.2, 0.3]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene_t = _torch_triangle_scene(rt, base_vertices)
        hit_t = scene_t.nearest_edge(point_t)
        self.assertEqual(int(hit_t.edge_id[0]), 0)
        hit_t.distance.sum().backward()
        grad_t = point_t.grad.detach().flatten().tolist()

        def torch_distance(point):
            return scene_t.nearest_edge(point).distance

        _, jvp_t = torch.func.jvp(torch_distance, (point_t.detach(),), (tangent,))

        point_d = ad.Array3f([0.25], [-0.2], [0.3])
        dr.enable_grad(point_d)
        scene_d = _drjit_triangle_scene(dr_backend, cuda)
        hit_d = scene_d.nearest_edge(point_d)
        self.assertEqual(int(hit_d.edge_id[0]), 0)
        forward_d = float(hit_d.distance[0])
        dr.backward(hit_d.distance)
        grad_d = _dr_vec3(dr.grad(point_d))

        point_j = ad.Array3f([0.25], [-0.2], [0.3])
        dr.enable_grad(point_j)
        scene_j = _drjit_triangle_scene(dr_backend, cuda)
        hit_j = scene_j.nearest_edge(point_j)
        dr.set_grad(point_j, ad.Array3f([0.07], [-0.03], [0.05]))
        dr.forward_from(point_j)
        jvp_d = float(dr.grad(hit_j.distance)[0])

        self.assertAlmostEqual(float(hit_t.distance[0].detach()), forward_d, delta=_FORWARD_ATOL)
        self.assertVectorClose(grad_t, grad_d)
        self.assertAlmostEqual(float(jvp_t[0]), jvp_d, delta=_GRAD_ATOL)

    def test_nearest_edge_ray_fixed_winner_forward_vjp_jvp(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("drjit")
        ad = importlib.import_module("drjit.cuda.ad")
        vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32)
        direction_t = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", requires_grad=True)
        origin_t = torch.tensor([[0.25, -0.4, 1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene_t = _torch_triangle_scene(rt, vertices)
        hit_t = scene_t.nearest_edge(rt.Ray(origin_t, direction_t))
        self.assertEqual(int(hit_t.edge_id[0]), 0)
        hit_t.distance.sum().backward()
        grad_origin_t = origin_t.grad.detach().flatten().tolist()
        grad_direction_t = direction_t.grad.detach().flatten().tolist()

        def torch_distance(origin):
            return scene_t.nearest_edge(rt.Ray(origin, direction_t.detach())).distance

        tangent_origin_t = torch.tensor([[0.0, 0.07, 0.0]], device="cuda")
        _, jvp_t = torch.func.jvp(torch_distance, (origin_t.detach(),), (tangent_origin_t,))

        origin_d = ad.Array3f([0.25], [-0.4], [1.0])
        direction_d = ad.Array3f([0.0], [0.0], [-1.0])
        dr.enable_grad(origin_d, direction_d)
        scene_d = _drjit_triangle_scene(dr_backend, cuda)
        ray_d = dr_backend.RayAD(origin_d, direction_d)
        ray_d.tmax = ad.Float([2.0])
        hit_d = scene_d.nearest_edge(ray_d)
        self.assertEqual(int(hit_d.edge_id[0]), 0)
        forward_d = float(hit_d.distance[0])
        dr.backward(hit_d.distance)
        grad_origin_d = _dr_vec3(dr.grad(origin_d))
        grad_direction_d = _dr_vec3(dr.grad(direction_d))

        origin_j = ad.Array3f([0.25], [-0.4], [1.0])
        dr.enable_grad(origin_j)
        ray_j = dr_backend.RayAD(origin_j, ad.Array3f([0.0], [0.0], [-1.0]))
        ray_j.tmax = ad.Float([2.0])
        hit_j = scene_d.nearest_edge(ray_j)
        dr.set_grad(origin_j, ad.Array3f([0.0], [0.07], [0.0]))
        dr.forward_from(origin_j)
        jvp_d = float(dr.grad(hit_j.distance)[0])

        self.assertAlmostEqual(float(hit_t.distance[0]), forward_d, delta=_FORWARD_ATOL)
        self.assertAlmostEqual(float(hit_t.ray_t[0]), float(hit_d.ray_t[0]), delta=_FORWARD_ATOL)
        self.assertAlmostEqual(float(hit_t.edge_t[0]), float(hit_d.edge_t[0]), delta=_FORWARD_ATOL)
        self.assertVectorClose(grad_origin_t, grad_origin_d)
        self.assertVectorClose(grad_direction_t, grad_direction_d)
        self.assertAlmostEqual(float(jvp_t[0]), jvp_d, delta=_GRAD_ATOL)

    def test_reflection_geometry_fixed_hit_forward_vjp_jvp(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("drjit")
        ad = importlib.import_module("drjit.cuda.ad")
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        base_t = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], device="cuda", dtype=torch.float32
        )
        tangent_t = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.1], [0.0, 0.0, 0.05]], device="cuda")
        ray_t = rt.Ray(
            torch.tensor([[-0.2, -0.2, -1.0]], device="cuda"), torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
        )

        vertices_t = base_t.clone().requires_grad_(True)
        scene_t = rt.Scene()
        scene_t.add_mesh(rt.Mesh(vertices_t, faces))
        scene_t.build()
        chain_t = scene_t.trace_reflections(ray_t, max_bounces=1)
        self.assertTrue(bool(chain_t.valid[0, 0]))
        chain_t.t.sum().backward()
        grad_t = vertices_t.grad.detach().flatten().tolist()

        def torch_t(vertices):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(vertices.contiguous(), faces))
            scene.build()
            return scene.trace_reflections(ray_t, max_bounces=1).t[:, 0]

        _, jvp_t = torch.func.jvp(torch_t, (base_t,), (tangent_t,))

        vertices_d = ad.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0])
        dr.enable_grad(vertices_d)
        scene_d = _drjit_triangle_scene(dr_backend, cuda, vertices_d)
        ray_d = dr_backend.RayAD(ad.Array3f([-0.2], [-0.2], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
        chain_d = scene_d.trace_reflections(ray_d, max_bounces=1, symbolic=False)
        self.assertTrue(bool(chain_d.is_valid()[0]))
        forward_d = float(chain_d.t[0])
        dr.backward(chain_d.t)
        grad_vec_d = dr.grad(vertices_d)
        grad_d = [float(grad_vec_d[axis][vertex]) for vertex in range(3) for axis in range(3)]

        vertices_j = ad.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0])
        dr.enable_grad(vertices_j)
        scene_j = _drjit_triangle_scene(dr_backend, cuda, vertices_j)
        chain_j = scene_j.trace_reflections(ray_d, max_bounces=1, symbolic=False)
        dr.set_grad(vertices_j, ad.Array3f([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, -0.1, 0.05]))
        dr.forward_from(vertices_j)
        jvp_d = float(dr.grad(chain_j.t)[0])

        self.assertAlmostEqual(float(chain_t.t[0, 0].detach()), forward_d, delta=_FORWARD_ATOL)
        self.assertVectorClose(grad_t, grad_d)
        self.assertAlmostEqual(float(jvp_t[0]), jvp_d, delta=_GRAD_ATOL)

    @unittest.skip(
        "Torch trace_refl_epc_field currently returns an invalid sentinel for the public "
        "single-plane fixtures; enable this parity gate when a valid shared fixture exists."
    )
    def test_reflection_field_fixed_hit_forward_vjp_jvp(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("drjit")
        ad = importlib.import_module("drjit.cuda.ad")
        vertices_t = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], device="cuda")
        source_t = torch.tensor([[-0.2, -0.2, -1.0]], device="cuda")
        receiver_t = torch.tensor([[-0.2, -0.2, 1.0]], device="cuda", requires_grad=True)
        tangent_t = torch.tensor([[0.03, -0.02, 0.04]], device="cuda")
        scene_t = _torch_triangle_scene(rt, vertices_t)
        field_t = scene_t.trace_refl_epc_field(source_t, receiver_t, max_bounces=1)
        self.assertTrue(bool(field_t.valid[0]))
        loss_t = field_t.field_real.sum() + 0.25 * field_t.field_imag.sum()
        loss_t.backward()
        grad_t = receiver_t.grad.detach().flatten().tolist()

        def torch_field(receiver):
            out = scene_t.trace_refl_epc_field(source_t, receiver, max_bounces=1)
            return out.field_real + 0.25 * out.field_imag

        _, jvp_t = torch.func.jvp(torch_field, (receiver_t.detach(),), (tangent_t,))

        source_d = ad.Array3f([-0.2], [-0.2], [-1.0])
        receiver_d = ad.Array3f([-0.2], [-0.2], [1.0])
        dr.enable_grad(receiver_d)
        scene_d = _drjit_triangle_scene(dr_backend, cuda)
        options_d = dr_backend.ReflEpcFieldOptionsAD()
        field_d = scene_d.trace_refl_epc_field(source_d, receiver_d, 1, options_d, ad.Bool([True]))
        self.assertTrue(bool(field_d.valid[0]))
        scalar_d = field_d.field_x_re + 0.25 * field_d.field_x_im
        forward_d = float(scalar_d[0])
        dr.backward(scalar_d)
        grad_d = _dr_vec3(dr.grad(receiver_d))

        receiver_j = ad.Array3f([-0.2], [-0.2], [1.0])
        dr.enable_grad(receiver_j)
        field_j = scene_d.trace_refl_epc_field(source_d, receiver_j, 1, options_d, ad.Bool([True]))
        scalar_j = field_j.field_x_re + 0.25 * field_j.field_x_im
        dr.set_grad(receiver_j, ad.Array3f([0.03], [-0.02], [0.04]))
        dr.forward_from(receiver_j)
        jvp_d = float(dr.grad(scalar_j)[0])

        self.assertAlmostEqual(float(torch_field(receiver_t.detach())[0]), forward_d, delta=_FIELD_ABS)
        self.assertVectorClose(grad_t, grad_d, delta=_FIELD_GRAD_ATOL)
        self.assertAlmostEqual(float(jvp_t[0]), jvp_d, delta=_FIELD_GRAD_ATOL)

    def test_utd_direct_gain_forward_vjp_jvp(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("drjit")
        ad = importlib.import_module("drjit.cuda.ad")
        scene_t = _torch_dfr_scene(rt)
        states_t = _torch_dfr_states(rt, src_power=2.0)
        grid_t = _torch_dfr_grid(rt)
        gain_t = torch.tensor([1.2], device="cuda", requires_grad=True)

        def torch_power(gain):
            material = rt.DfrMaterial(
                eta_r=torch.tensor([4.0], device="cuda"),
                sigma=torch.tensor([0.0], device="cuda"),
                mu_r=torch.tensor([1.0], device="cuda"),
                gain=gain,
                valid=torch.tensor([True], device="cuda"),
            )
            return scene_t.accum_dfr_direct(
                states=states_t, grid=grid_t, material=material, wavelength=0.125, seed=17, direct_samples=64
            ).power

        power_t = torch_power(gain_t)
        power_t.sum().backward()
        grad_t = float(gain_t.grad[0])
        _, jvp_t = torch.func.jvp(torch_power, (gain_t.detach(),), (torch.tensor([0.3], device="cuda"),))

        scene_d = _rayd_dfr_scene(dr_backend, cuda)

        def drjit_states():
            states = dr_backend.DfrStatesAD()
            states.count = 1
            states.edge_index = ad.Int([0])
            states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
            states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
            states.edge_t_min = ad.Float([-0.5])
            states.edge_t_max = ad.Float([0.5])
            states.n0 = ad.Array3f([0.0], [1.0], [0.0])
            states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
            states.prim0 = ad.Int([-1])
            states.prim1 = ad.Int([-1])
            states.exterior_angle = ad.Float([1.5 * math.pi])
            states.src = ad.Array3f([0.0], [0.0], [1.0])
            states.src_power = ad.Float([2.0])
            states.wi = ad.Array3f([0.0], [0.0], [-1.0])
            states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
            states.prefix_depth = ad.Int([0])
            return states

        states_d = drjit_states()
        grid_d = _rayd_dfr_grid(dr_backend)
        options_d = dr_backend.DfrOptions()
        options_d.wavelength = 0.125
        options_d.k = 2.0 * math.pi / 0.125
        options_d.seed = 17
        options_d.samples = 64
        options_d.max_order = 1
        options_d.direct_samples = 64
        options_d.keller_samples = 0
        options_d.strategy_mask = dr_backend.RAYD_DFR_DIRECT
        options_d.sample_sequence = dr_backend.RAYD_DFR_HASH
        options_d.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO

        gain_d = ad.Float([1.2])
        dr.enable_grad(gain_d)
        material_d = dr_backend.DfrMaterialAD()
        material_d.eta_r = ad.Float([4.0])
        material_d.sigma = ad.Float([0.0])
        material_d.mu_r = ad.Float([1.0])
        material_d.gain = gain_d
        material_d.valid = ad.Bool([True])
        out_d = scene_d.accum_dfr_direct(states_d, grid_d, material_d, options_d, True)
        forward_d = float(out_d.power[0])
        dr.backward(out_d.power, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad_d = float(dr.grad(gain_d)[0])

        gain_j = ad.Float([1.2])
        dr.enable_grad(gain_j)
        material_j = dr_backend.DfrMaterialAD()
        material_j.eta_r = ad.Float([4.0])
        material_j.sigma = ad.Float([0.0])
        material_j.mu_r = ad.Float([1.0])
        material_j.gain = gain_j
        material_j.valid = ad.Bool([True])
        out_j = scene_d.accum_dfr_direct(drjit_states(), grid_d, material_j, options_d, True)
        dr.set_grad(gain_j, ad.Float([0.3]))
        dr.forward_from(gain_j)
        jvp_d = float(dr.grad(out_j.power)[0])

        self.assertAlmostEqual(float(power_t[0].detach()), forward_d, delta=_FIELD_ABS)
        self.assertAlmostEqual(grad_t, grad_d, delta=_FIELD_GRAD_ATOL)
        self.assertAlmostEqual(float(jvp_t[0]), jvp_d, delta=_FIELD_GRAD_ATOL)


if __name__ == "__main__":
    unittest.main()
