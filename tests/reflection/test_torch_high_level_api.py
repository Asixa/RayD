# Copyright Xingyu Chen.
# Tests the Torch high-level reflection accumulation and EPC APIs.

import inspect
import unittest

import torch

import rayd.torch as rt


class TorchReflectionApiContractTests(unittest.TestCase):
    def test_public_records_are_exported(self):
        for name in (
            "AccumGrid",
            "AccumOptions",
            "AccumResult",
            "ReflEpc",
            "ReflEpcFieldOptions",
            "ReflEpcOptions",
            "ReflMaterial",
            "WedgeEvents",
        ):
            self.assertIn(name, rt.__all__)
            self.assertTrue(hasattr(rt, name))

    def test_epc_options_default_to_none_as_in_drjit(self):
        for method in (rt.Scene.trace_refl_epc, rt.Scene.trace_refl_epc_field):
            parameters = inspect.signature(method).parameters
            self.assertEqual(list(parameters)[1:], ["source", "receiver", "max_bounces", "options", "active"])
            self.assertIsNone(parameters["options"].default)

    def test_multi_device_epc_paths_fail_loudly(self):
        class MultiStub:
            def unsupported(self, operation):
                raise NotImplementedError(operation)

        scene = object.__new__(rt.Scene)
        scene._multi = MultiStub()
        placeholder = torch.empty((0,))
        with self.assertRaisesRegex(NotImplementedError, "trace_refl_epc"):
            scene.trace_refl_epc(placeholder, placeholder, 1, object())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchReflectionApiNativeTests(unittest.TestCase):
    @staticmethod
    def _scene(*, vertices_require_grad=False):
        vertices = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=vertices_require_grad,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        return scene

    @staticmethod
    def _epc_options():
        return rt.ReflEpcOptions(
            expected_prim_ids=torch.tensor([[0]], device="cuda", dtype=torch.int32),
            direct_plane_points=torch.tensor([[[0.0, 0.0, 0.0]]], device="cuda"),
            direct_plane_normals=torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda"),
            surface_group_id=torch.tensor([0], device="cuda", dtype=torch.int32),
            surface_group_size=torch.tensor([1], device="cuda", dtype=torch.int32),
            surface_group_members=torch.tensor([0], device="cuda", dtype=torch.int32),
        )

    def test_accum_refl_wraps_native_result(self):
        scene = self._scene()
        ray_o = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
        ray = rt.Ray(ray_o, torch.tensor([[0.0, 0.0, 1.0]], device="cuda"), torch.tensor([2.0], device="cuda"))
        grid = rt.AccumGrid(
            axis=2,
            position=-1.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=4,
            resolution1=4,
        )
        result = scene.accum_refl(ray, ray_o, grid, rt.ReflMaterial.default(1, device=ray_o.device), max_bounces=1)
        self.assertIsInstance(result, rt.AccumResult)
        self.assertEqual(result.ray_count, 1)
        self.assertEqual(result.grid_cell_count, 16)
        self.assertEqual(result.reflection_power.shape, (4, 4))
        self.assertEqual(result.reflection_field_x.dtype, torch.complex64)
        self.assertEqual(result.wedge_events.capacity, 0)

    def test_trace_refl_epc_wraps_native_result(self):
        scene = self._scene()
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
        receiver = torch.tensor([[0.5, 0.0, -1.0]], device="cuda")
        result = scene.trace_refl_epc(source, receiver, 1, self._epc_options())
        self.assertIsInstance(result, rt.ReflEpc)
        self.assertEqual(result.ray_count, 1)
        self.assertEqual(result.max_bounces, 1)
        self.assertEqual(result.valid.shape, (1,))
        self.assertEqual(result.hit_points.shape, (1, 1, 3))

    def test_forward_only_operations_reject_reverse_ad(self):
        scene = self._scene()
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", requires_grad=True)
        receiver = torch.tensor([[0.5, 0.0, -1.0]], device="cuda")
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            scene.trace_refl_epc(source, receiver, 1, self._epc_options())

        ray = rt.Ray(source, torch.tensor([[0.0, 0.0, 1.0]], device="cuda"), torch.tensor([2.0], device="cuda"))
        grid = rt.AccumGrid(resolution0=1, resolution1=1)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            scene.accum_refl(ray, source, grid, rt.ReflMaterial.default(1, device=source.device), 1)

    def test_forward_only_operations_reject_jvp(self):
        scene = self._scene()
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
        receiver = torch.tensor([[0.5, 0.0, -1.0]], device="cuda")
        with torch.autograd.forward_ad.dual_level():
            dual_source = torch.autograd.forward_ad.make_dual(source, torch.ones_like(source))
            with self.assertRaisesRegex(RuntimeError, "JVP"):
                scene.trace_refl_epc(dual_source, receiver, 1, self._epc_options())
            ray = rt.Ray(
                dual_source, torch.tensor([[0.0, 0.0, 1.0]], device="cuda"), torch.tensor([2.0], device="cuda")
            )
            with self.assertRaisesRegex(RuntimeError, "JVP"):
                scene.accum_refl(
                    ray,
                    dual_source,
                    rt.AccumGrid(resolution0=1, resolution1=1),
                    rt.ReflMaterial.default(1, device=source.device),
                    1,
                )

    def test_forward_only_operations_reject_mesh_ad(self):
        scene = self._scene(vertices_require_grad=True)
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
        receiver = torch.tensor([[0.5, 0.0, -1.0]], device="cuda")
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            scene.trace_refl_epc(source, receiver, 1, self._epc_options())

    def _ray_toward_receiver(self, source, receiver):
        direction = receiver - source
        return rt.Ray(source, direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True))

    def test_trace_refl_epc_from_a_ray_follows_the_ray(self):
        scene = self._scene()
        source = torch.tensor([[-0.5, -0.5, -1.0], [-0.5, -0.5, -1.0]], device="cuda")
        receiver = torch.tensor([[0.0, -0.5, -1.0], [0.0, -0.5, -1.0]], device="cuda")
        planes = scene.trace_refl_epc(source[:1], receiver[:1], 1, self._epc_options())
        # Row 0 aims at the specular point (-0.25, -0.5, 0); row 1 points away from the mirror, which a Ray respects.
        toward = torch.tensor([[0.25, 0.0, 1.0], [0.0, 0.0, -1.0]], device="cuda")
        traced = scene.trace_refl_epc(rt.Ray(source, toward), receiver, 1)
        self.assertEqual(traced.valid.tolist(), [True, False])
        self.assertTrue(planes.valid.item())
        torch.testing.assert_close(traced.path_length[:1], planes.path_length, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(traced.hit_points[:1], planes.hit_points, rtol=1e-6, atol=1e-6)
        self.assertEqual(traced.resolved_prim_ids[0].tolist(), [0])

    def test_trace_refl_epc_from_a_ray_honours_expected_primitives_and_tmax(self):
        scene = self._scene()
        source = torch.tensor([[-0.5, -0.5, -1.0]], device="cuda")
        receiver = torch.tensor([[0.0, -0.5, -1.0]], device="cuda")
        ray = rt.Ray(source, torch.tensor([[0.25, 0.0, 1.0]], device="cuda"))
        free = rt.ReflEpcOptions(expected_prim_ids=torch.tensor([[-1]], device="cuda", dtype=torch.int32))
        other = rt.ReflEpcOptions(expected_prim_ids=torch.tensor([[5]], device="cuda", dtype=torch.int32))
        self.assertTrue(scene.trace_refl_epc(ray, receiver, 1, free).valid.item())
        self.assertFalse(scene.trace_refl_epc(ray, receiver, 1, other).valid.item())
        short = rt.Ray(ray.o, ray.d, torch.tensor([0.5], device="cuda"))
        self.assertFalse(scene.trace_refl_epc(short, receiver, 1).valid.item())
        inactive = scene.trace_refl_epc(ray, receiver, 1, active=torch.tensor([False], device="cuda"))
        self.assertFalse(inactive.valid.item())

    def test_trace_refl_epc_rejects_mixed_modes(self):
        scene = self._scene()
        source = torch.tensor([[-0.5, -0.5, -1.0]], device="cuda")
        receiver = torch.tensor([[0.0, -0.5, -1.0]], device="cuda")
        with self.assertRaisesRegex(ValueError, "direct_plane_points"):
            scene.trace_refl_epc(self._ray_toward_receiver(source, receiver), receiver, 1, self._epc_options())
        with self.assertRaisesRegex(ValueError, "requires ReflEpcOptions"):
            scene.trace_refl_epc(source, receiver, 1)
        with self.assertRaisesRegex(TypeError, "ReflEpcOptions"):
            scene.trace_refl_epc(source, receiver, 1, torch.tensor([True], device="cuda"))

    def test_trace_refl_epc_from_a_ray_rejects_ad(self):
        scene = self._scene()
        source = torch.tensor([[-0.5, -0.5, -1.0]], device="cuda", requires_grad=True)
        receiver = torch.tensor([[0.0, -0.5, -1.0]], device="cuda")
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            scene.trace_refl_epc(rt.Ray(source, torch.tensor([[0.25, 0.0, 1.0]], device="cuda")), receiver, 1)
