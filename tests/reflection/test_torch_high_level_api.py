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
            "ReflEpcOptions",
            "ReflMaterial",
            "WedgeEvents",
        ):
            self.assertIn(name, rt.__all__)
            self.assertTrue(hasattr(rt, name))

    def test_epc_options_are_required(self):
        options = inspect.signature(rt.Scene.trace_refl_epc).parameters["options"]
        self.assertIs(options.default, inspect.Parameter.empty)

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

    def test_accumulate_reflections_wraps_native_result(self):
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
        result = scene.accumulate_reflections(
            ray, ray_o, grid, rt.ReflMaterial.default(1, device=ray_o.device), max_bounces=1
        )
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
            scene.accumulate_reflections(ray, source, grid, rt.ReflMaterial.default(1, device=source.device), 1)

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
                scene.accumulate_reflections(
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
