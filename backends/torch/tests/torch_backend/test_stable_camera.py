from __future__ import annotations

import os
from pathlib import Path
import unittest

import torch

import rayd.torch  # noqa: F401 - registers the legacy dispatcher for parity checks
from rayd.torch import _stable


class StableCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("RAYD_TORCH_STABLE_LIBRARY")
        if override and Path(override).is_file():
            torch.ops.load_library(str(Path(override).resolve()))
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if not hasattr(torch.ops.rayd_torch_stable, "camera_sample_to_world"):
            raise unittest.SkipTest("RayD Stable ABI library is not built")
        if os.environ.get("RAYD_TORCH_STABLE_LIBRARY"):
            if not _stable.AVAILABLE:
                raise AssertionError(f"Stable ABI loader failed: {_stable.LOAD_ERROR}")

    def assert_close(self, stable, legacy):
        if isinstance(stable, (tuple, list)):
            self.assertEqual(len(stable), len(legacy))
            for lhs, rhs in zip(stable, legacy):
                torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(stable, legacy, rtol=0.0, atol=0.0)

    def test_forward_and_backward_ops_match_legacy_dispatcher(self):
        torch.manual_seed(7)
        sample = torch.rand((257, 2), device="cuda", dtype=torch.float32)
        point = torch.randn((257, 3), device="cuda", dtype=torch.float32)
        point[:, 2].abs_().add_(0.25)
        grad_world = torch.randn((257, 3), device="cuda", dtype=torch.float32)
        grad_sample = torch.randn((257, 2), device="cuda", dtype=torch.float32)
        grad_direction = torch.randn((257, 3), device="cuda", dtype=torch.float32)
        stable = torch.ops.rayd_torch_stable
        legacy = torch.ops.rayd_torch

        calls = (
            ("camera_sample_to_world", (sample, 0.9, 0.7, 2.5)),
            ("camera_sample_to_world_backward", (grad_world, 257, 0.9, 0.7, 2.5)),
            ("camera_world_to_sample", (point, 0.9, 0.7)),
            ("camera_world_to_sample_backward", (point, grad_sample, 0.9, 0.7)),
            ("camera_sample_ray", (sample, 0.9, 0.7)),
            ("camera_sample_ray_backward", (sample, grad_direction, 0.9, 0.7)),
        )
        for name, args in calls:
            with self.subTest(op=name):
                self.assert_close(getattr(stable, name)(*args), getattr(legacy, name)(*args))


if __name__ == "__main__":
    unittest.main()
