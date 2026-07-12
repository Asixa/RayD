from __future__ import annotations

import os
from pathlib import Path
import unittest

import torch

import rayd.torch  # noqa: F401 - loads stable and legacy dispatcher libraries
from rayd.torch import _legacy, _stable


class StableCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if not hasattr(torch.ops.rayd_torch_stable, "intersection_valid"):
            raise unittest.SkipTest("RayD Stable ABI core operators are not built")

    def test_intersection_valid_matches_legacy_dispatcher(self):
        self.assertTrue(_legacy.is_registered())
        t = torch.tensor([1.0, float("inf"), float("nan")], device="cuda")
        shape_id = torch.tensor([0, -1, 3], device="cuda", dtype=torch.int32)
        empty_shape = torch.empty((0,), device="cuda", dtype=torch.int32)
        stable = torch.ops.rayd_torch_stable
        legacy = torch.ops.rayd_torch
        torch.testing.assert_close(
            stable.intersection_valid(t, shape_id),
            legacy.intersection_valid(t, shape_id),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            stable.intersection_valid(t, empty_shape),
            legacy.intersection_valid(t, empty_shape),
            rtol=0.0,
            atol=0.0,
        )

    def test_loader_override_reports_failures(self):
        override = os.environ.get("RAYD_TORCH_STABLE_LIBRARY")
        if override and Path(override).is_file():
            self.assertTrue(_stable.AVAILABLE, _stable.LOAD_ERROR)


if __name__ == "__main__":
    unittest.main()
