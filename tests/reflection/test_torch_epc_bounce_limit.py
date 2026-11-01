# Copyright Xingyu Chen.
# Tests that the Torch EPC forwards reject bounce counts beyond the native per-lane chain capacity.

import unittest

import torch

import rayd.torch as rt

# The discovery raygen stores each chain in per-lane arrays of this size (ReflEpcMaxBounces).
NATIVE_EPC_MAX_BOUNCES = 8


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchEpcBounceLimitTests(unittest.TestCase):
    @staticmethod
    def _box_scene():
        vertices = torch.tensor(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        return scene

    @staticmethod
    def _closed_box_rays(count):
        generator = torch.Generator(device="cpu").manual_seed(7)
        source = ((torch.rand(count, 3, generator=generator) - 0.5) * 1.6).cuda()
        direction = torch.nn.functional.normalize(torch.randn(count, 3, generator=generator), dim=1).cuda()
        # Receivers far outside the closed box make every lane discover a full-depth chain.
        return source, source + 100.0 * direction

    @staticmethod
    def _epc_options(count, bounces):
        return rt.ReflEpcOptions(
            expected_prim_ids=torch.full((count, bounces), -1, device="cuda", dtype=torch.int32),
            direct_plane_points=torch.zeros((count, bounces, 3), device="cuda"),
            direct_plane_normals=torch.zeros((count, bounces, 3), device="cuda"),
            surface_group_id=torch.zeros((12,), device="cuda", dtype=torch.int32),
            surface_group_size=torch.ones((1,), device="cuda", dtype=torch.int32),
            surface_group_members=torch.zeros((1,), device="cuda", dtype=torch.int32),
        )

    def test_trace_refl_epc_field_accepts_the_native_limit(self):
        scene = self._box_scene()
        source, receiver = self._closed_box_rays(256)
        result = scene.trace_refl_epc_field(source, receiver, NATIVE_EPC_MAX_BOUNCES)
        torch.cuda.synchronize()
        self.assertEqual(result.valid.shape, (256,))

    def test_trace_refl_epc_field_rejects_bounces_beyond_native_limit(self):
        scene = self._box_scene()
        source, receiver = self._closed_box_rays(256)
        for bounces in (NATIVE_EPC_MAX_BOUNCES + 1, 64):
            with self.subTest(bounces=bounces):
                with self.assertRaisesRegex(RuntimeError, "native EPC limit"):
                    scene.trace_refl_epc_field(source, receiver, bounces)
        torch.cuda.synchronize()

    def test_trace_refl_epc_rejects_bounces_beyond_native_limit(self):
        scene = self._box_scene()
        source, receiver = self._closed_box_rays(16)
        bounces = NATIVE_EPC_MAX_BOUNCES + 1
        with self.assertRaisesRegex(RuntimeError, "native EPC limit"):
            scene.trace_refl_epc(source, receiver, bounces, self._epc_options(16, bounces))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
