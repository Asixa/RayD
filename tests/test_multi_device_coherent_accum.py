# Copyright Xingyu Chen.
# Tests replicated coherent diffraction grid assembly.

from __future__ import annotations

from pathlib import Path
import unittest

import torch

import rayd.torch as rt
from rayd._impl.multi import _add_coherent_accum, _add_coherent_accum_in_place, _coherent_accum_to


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "direct_field_x_re",
    "direct_field_x_im",
    "direct_field_y_re",
    "direct_field_y_im",
    "direct_field_z_re",
    "direct_field_z_im",
    "multi_field_x_re",
    "multi_field_x_im",
    "multi_field_y_re",
    "multi_field_y_im",
    "multi_field_z_re",
    "multi_field_z_im",
    "direct_count",
    "multi_count",
    "visibility_reject_count",
    "utd_reject_count",
)


def _partial(value: int) -> rt.DfrCoherentAccum:
    floats = [torch.full((2,), float(value + index), dtype=torch.float32) for index in range(12)]
    counts = [torch.full((2,), value + index, dtype=torch.int32) for index in range(12, 16)]
    return rt.DfrCoherentAccum(2, *floats, *counts)


class MultiDeviceCoherentAccumTests(unittest.TestCase):
    def test_partial_grids_add_field_by_field(self) -> None:
        left = _partial(1)
        right = _partial(10)
        merged = _add_coherent_accum(left, right)

        for name in FIELDS:
            self.assertTrue(torch.equal(getattr(merged, name), getattr(left, name) + getattr(right, name)))

        in_place_left = _partial(1)
        returned = _add_coherent_accum_in_place(in_place_left, right)
        self.assertIs(returned, in_place_left)
        for name in FIELDS:
            self.assertTrue(torch.equal(getattr(returned, name), getattr(merged, name)))
        self.assertIs(_coherent_accum_to(returned, torch.device("cpu")), returned)

    def test_scene_routes_coherent_accumulation_through_lane_shards(self) -> None:
        scene = (ROOT / "python" / "rayd" / "_impl" / "scene.py").read_text(encoding="utf-8")
        multi = (ROOT / "python" / "rayd" / "_impl" / "multi.py").read_text(encoding="utf-8")

        scene_body = scene.split("    def accum_dfr_coherent_direct(", 1)[1].split("    def update_mesh_vertices(", 1)[
            0
        ]
        multi_body = multi.split("    def accum_dfr_coherent_direct(", 1)[1].split("    def accum_dfr(", 1)[0]
        self.assertIn("self._multi.accum_dfr_coherent_direct(", scene_body)
        self.assertNotIn('self._multi.unsupported("accum_dfr_coherent_direct")', scene_body)
        self.assertIn('"accum_dfr_coherent_direct"', multi_body)
        self.assertIn("int(states.state_count) * grid_cell_count", multi_body)
        self.assertIn("lane_offset=begin", multi_body)
        self.assertIn("lane_count=count", multi_body)
        self.assertIn("_add_coherent_accum", multi_body)
        self.assertIn("lane_row_bytes=_DFR_COHERENT_STAGED_LANE_BYTES", multi_body)
        self.assertIn("_DFR_COHERENT_STAGED_LANE_BYTES = 36", multi)


if __name__ == "__main__":
    unittest.main()
