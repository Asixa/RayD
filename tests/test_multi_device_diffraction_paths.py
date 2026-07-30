# Copyright Xingyu Chen.
# Tests deterministic multi-device diffraction path row assembly.

from __future__ import annotations

from pathlib import Path
import unittest

import torch

import rayd.torch as rt
from rayd._impl.multi import _merge_source_lane_paths, _validate_source_lane_active
from rayd._impl.multipath import _reject_forward_only_ad


ROOT = Path(__file__).resolve().parents[1]


def _paths(*, tx_ids: list[int], valid: list[bool], base: float) -> rt.DfrPaths:
    rows = len(tx_ids)
    scalar = torch.arange(rows, dtype=torch.float32) + base
    points = torch.stack((scalar, scalar + 1.0, scalar + 2.0), dim=1)
    ids = torch.arange(rows, dtype=torch.int32)
    return rt.DfrPaths(
        rows,
        torch.tensor([sum(valid)], dtype=torch.int32),
        torch.tensor(valid, dtype=torch.bool),
        torch.tensor(tx_ids, dtype=torch.int32),
        ids,
        ids,
        ids,
        ids,
        ids,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        points,
        points,
        points,
        layout=rt.DfrPathLayout.SourceLane,
    )


class MultiDeviceDiffractionPathTests(unittest.TestCase):
    def test_forward_only_guard_rejects_vjp_and_jvp_inputs(self) -> None:
        tracked = torch.ones(1, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            _reject_forward_only_ad("operation", tracked)

        primal = torch.ones(1)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal, torch.ones_like(primal))
            with self.assertRaisesRegex(RuntimeError, "JVP"):
                _reject_forward_only_ad("operation", dual)

    def test_active_contract_is_checked_before_replication(self) -> None:
        master = torch.device("cuda", 0)
        with self.assertRaisesRegex(RuntimeError, "must be CUDA"):
            _validate_source_lane_active(torch.ones(2, dtype=torch.bool), 2, master)

        if not torch.cuda.is_available():
            return

        contiguous = torch.ones(2, dtype=torch.bool, device=master)
        _validate_source_lane_active(contiguous, 2, master)
        cases = (
            (torch.ones(4, dtype=torch.bool, device=master)[::2], 2, "contiguous"),
            (torch.ones(2, dtype=torch.float32, device=master), 2, "wrong dtype"),
            (torch.ones((2, 1), dtype=torch.bool, device=master), 2, "wrong rank"),
            (torch.ones(3, dtype=torch.bool, device=master), 2, r"shape \[state_limit\]"),
        )
        for active, state_limit, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                _validate_source_lane_active(active, state_limit, master)

    def test_merge_preserves_source_lane_blocks_and_global_tx_ids(self) -> None:
        first = _paths(tx_ids=[0, -1, 1, 1], valid=[True, False, True, True], base=0.0)
        second = _paths(tx_ids=[0, 0], valid=[True, True], base=4.0)

        merged = _merge_source_lane_paths([(0, first), (2, second)], capacity=6, master=torch.device("cpu"))

        self.assertEqual(merged.layout, rt.DfrPathLayout.SourceLane)
        self.assertEqual(merged.capacity, 6)
        self.assertEqual(merged.count.dtype, torch.int32)
        self.assertTrue(torch.equal(merged.count, torch.tensor([5], dtype=torch.int32)))
        self.assertTrue(torch.equal(merged.tx_id, torch.tensor([0, -1, 1, 1, 2, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(merged.delay, torch.arange(6, dtype=torch.float32)))
        self.assertEqual(tuple(merged.p0.shape), (6, 3))

    def test_public_and_dispatcher_contracts_expose_layout(self) -> None:
        geometry = (ROOT / "python" / "rayd" / "_impl" / "geometry.py").read_text(encoding="utf-8")
        multipath = (ROOT / "python" / "rayd" / "_impl" / "multipath.py").read_text(encoding="utf-8")
        scene = (ROOT / "python" / "rayd" / "_impl" / "scene.py").read_text(encoding="utf-8")
        library = (ROOT / "src" / "bindings" / "library.cpp").read_text(encoding="utf-8")

        self.assertIn("class DfrPathLayout(IntEnum):", geometry)
        self.assertIn("layout: DfrPathLayout = DfrPathLayout.Compact", geometry)
        self.assertIn("int(resolved_layout)", multipath)
        self.assertIn("DfrPathLayout.SourceLane if self._multi is not None", scene)
        self.assertIn("int output_layout=0", library)

    def test_multi_device_path_uses_tx_aligned_source_lane_shards(self) -> None:
        multi = (ROOT / "python" / "rayd" / "_impl" / "multi.py").read_text(encoding="utf-8")

        body = multi.split("    def trace_dfr_paths(", 1)[1].split("    def accum_dfr_direct(", 1)[0]
        self.assertIn("lanes_per_tx = rx_count * state_limit", body)
        self.assertIn("self._shards(tx_count, operation)", body)
        self.assertIn("layout=DfrPathLayout.SourceLane", body)
        self.assertIn("_merge_source_lane_paths", body)
        self.assertIn('raise RuntimeError("state_limit must be non-negative.")', body)
        self.assertLess(body.index("_validate_source_lane_devices"), body.index("self._shards(tx_count, operation)"))
        self.assertLess(body.index("_validate_source_lane_active"), body.index("self._shards(tx_count, operation)"))
        self.assertLess(body.index("if self.chunked:"), body.index("if tx_count == 0 or lanes_per_tx == 0:"))
        self.assertIn('raise RuntimeError("capacity does not fit in int32.")', body)
        self.assertNotIn(".cpu(", body)
        self.assertNotIn(".item(", body)


if __name__ == "__main__":
    unittest.main()
