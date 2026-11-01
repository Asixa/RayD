# Copyright Xingyu Chen.
# Tests that Torch reflection tracing rejects batches whose per-bounce output index overflows int32.

import unittest

import torch

import rayd.torch as rt

from tests.support.geometry import box_mesh

INT32_MAX = 2**31 - 1
# 4096 rays bouncing 174,763 times inside a closed box address image-source row 3 * 715,829,247, past INT32_MAX.
# Unguarded, that launch wrote through a wrapped negative index and raised an illegal memory access.
OVERFLOW_RAYS = 4096
OVERFLOW_BOUNCES = 174_763


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchReflectionIndexRangeTests(unittest.TestCase):
    def setUp(self):
        vertices, faces = box_mesh(torch.device("cuda"))
        self.scene = rt.Scene()
        self.scene.add_mesh(rt.Mesh(vertices, faces))
        self.scene.build()

    def _rays(self, count):
        origin = torch.zeros((count, 3), device="cuda")
        direction = torch.tensor([[0.3, 0.4, 1.0]], device="cuda").expand(count, 3).contiguous()
        return rt.Ray(origin, direction)

    def test_rejects_image_source_index_overflow_before_allocating(self):
        chain = self.scene.trace_reflections(self._rays(OVERFLOW_RAYS), OVERFLOW_BOUNCES)
        with self.assertRaisesRegex(RuntimeError, "int32 output index range"):
            _ = chain.image_sources
        torch.cuda.synchronize()

    def test_rejects_scalar_output_index_overflow_before_allocating(self):
        for ray_count, max_bounces in ((1, INT32_MAX + 1), (1, 2**40), (1024, INT32_MAX // 1024 + 1)):
            with self.subTest(ray_count=ray_count, max_bounces=max_bounces):
                chain = self.scene.trace_reflections(self._rays(ray_count), max_bounces)
                with self.assertRaisesRegex(RuntimeError, "int32 output index range"):
                    _ = chain.t
        torch.cuda.synchronize()

    def test_rejects_differentiable_trace_index_overflow(self):
        origin = torch.zeros((OVERFLOW_RAYS, 3), device="cuda", requires_grad=True)
        direction = torch.tensor([[0.3, 0.4, 1.0]], device="cuda").expand(OVERFLOW_RAYS, 3).contiguous()
        with self.assertRaisesRegex(RuntimeError, "int32 output index range"):
            self.scene.trace_reflections(rt.Ray(origin, direction), OVERFLOW_BOUNCES)
        torch.cuda.synchronize()

    def test_accepts_ordinary_bounce_counts(self):
        chain = self.scene.trace_reflections(self._rays(64), 16)
        self.assertEqual(tuple(chain.image_sources.shape), (64, 16, 3))
        self.assertTrue(bool(chain.valid.all()))


if __name__ == "__main__":
    unittest.main()
