# Copyright Xingyu Chen.
# Tests Torch SDF LOS and reflection operation scope.

from __future__ import annotations

import unittest

import torch


try:
    import rayd.torch as rt
except (ImportError, RuntimeError):
    rt = None


@unittest.skipUnless(torch.cuda.is_available() and rt is not None and rt._NATIVE_AVAILABLE, "CUDA Torch RayD required")
class TorchSdfOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        resolution = 16
        axis = torch.linspace(-1.0, 1.0, resolution, dtype=torch.float32, device="cuda")
        _x, _y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
        values = z.contiguous()
        self.grid = rt.SdfGrid(
            values,
            torch.zeros((3,), dtype=torch.float32, device="cuda"),
            torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device="cuda"),
            torch.tensor((2.0, 2.0, 2.0), dtype=torch.float32, device="cuda"),
        )

    def test_los_reflection_and_closed_scope(self) -> None:
        origin = torch.tensor(((0.0, 0.0, -1.0),), dtype=torch.float32, device="cuda", requires_grad=True)
        direction = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device="cuda")
        ray = rt.Ray(origin, direction)
        hit = self.grid.intersect(ray)
        self.assertEqual(hit.hit_mask.tolist(), [True])
        self.assertAlmostEqual(hit.t.item(), 1.0, places=3)
        self.assertEqual(
            self.grid.visible(origin.detach(), torch.tensor(((0.0, 0.0, 1.0),), device="cuda")).tolist(), [False]
        )
        chain = self.grid.trace_reflections(ray, 1)
        self.assertEqual(chain.valid.tolist(), [[True]])
        self.assertEqual(chain.prim_ids.tolist(), [[0]])
        chain.t.sum().backward()
        self.assertAlmostEqual(origin.grad[0, 2].item(), -1.0, places=3)
        self.assertFalse(hasattr(self.grid, "transmittance"))
        self.assertFalse(hasattr(self.grid, "trace_diffraction"))


if __name__ == "__main__":
    unittest.main()
