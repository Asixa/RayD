# Copyright Xingyu Chen.
# Tests deterministic multi-device diffraction path row assembly.

from __future__ import annotations

import dataclasses
from pathlib import Path
import re
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

    def test_default_layout_matches_the_dispatcher_default(self) -> None:
        defaults = {field.name: field.default for field in dataclasses.fields(rt.DfrPaths)}
        self.assertIs(defaults["layout"], rt.DfrPathLayout.Compact)
        library = (ROOT / "src" / "bindings" / "library.cpp").read_text(encoding="utf-8")
        schema_defaults = re.findall(r"\bint output_layout=(-?\d+)", library)
        self.assertTrue(schema_defaults)
        self.assertEqual({int(value) for value in schema_defaults}, {int(rt.DfrPathLayout.Compact)})


if __name__ == "__main__":
    unittest.main()


class DfrPathMaxPathsTests(unittest.TestCase):
    """`DfrPathOptions.max_paths`: `None` or `0` exports every state, a negative value is refused."""

    def test_zero_and_none_mean_every_state_and_negative_raises(self) -> None:
        from rayd._impl.geometry import _resolve_dfr_max_paths

        self.assertEqual(_resolve_dfr_max_paths(None, 5), 5)
        self.assertEqual(_resolve_dfr_max_paths(0, 5), 5)
        self.assertEqual(_resolve_dfr_max_paths(3, 5), 3)
        with self.assertRaisesRegex(ValueError, "max_paths must be None or non-negative"):
            _resolve_dfr_max_paths(-1, 5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
    def test_scene_export_capacity_follows_the_resolved_limit(self) -> None:
        from tests.scene.test_multi_device_scene import _accum_scene, _dfr_material, _dfr_states

        device = torch.device("cuda", 0)
        scene = _accum_scene(device)
        states = _dfr_states(device)
        active = torch.ones(states.state_count, dtype=torch.bool, device=device)
        tx = torch.tensor([[0.0, -1.0, 0.25]], device=device)
        rx = torch.tensor([[0.0, 1.0, 0.25], [0.1, 1.0, 0.25]], device=device)

        def trace(max_paths, mask=active):
            options = rt.DfrPathOptions(max_paths=max_paths)
            return scene.trace_dfr_paths(tx, rx, states, _dfr_material(device), options, active=mask)

        every = trace(None)
        zero = trace(0)
        self.assertEqual(every.capacity, 2 * states.state_count)
        self.assertEqual(zero.capacity, every.capacity)
        self.assertTrue(torch.equal(zero.count, every.count))
        # `active` has one entry per exported state.
        self.assertEqual(trace(1, active[:1].contiguous()).capacity, 2)
        with self.assertRaisesRegex(ValueError, "max_paths must be None or non-negative"):
            trace(-1)
