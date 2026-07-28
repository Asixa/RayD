"""Policy and memory contracts for replicated multi-device execution.

These tests deliberately exercise the Python policy layer without constructing
native scenes where possible.  That keeps topology, sharding, and dispatch
regressions visible on CPU-only CI; the final peak-allocation test is optional
CUDA evidence for the budget contract.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import textwrap
import unittest
from unittest import mock

import torch

import rayd.torch as rt
from rayd.torch._multi import (
    _ReplicatedScene,
    _add_accum,
    _add_accum_in_place,
    _device_index,
    calibrate_chunk_size,
    plan as plan_multi_device,
)
from rayd.torch.types import DfrAccum


def _planned(options: rt.MultiDeviceOptions) -> _ReplicatedScene:
    """Create a policy layer without touching CUDA streams or native scenes."""
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=2),
        mock.patch.object(
            torch.cuda, "can_device_access_peer", return_value=True, create=True
        ),
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(name="Test GPU", major=9, minor=0),
        ),
    ):
        layer = plan_multi_device(
            [0, 1],
            options,
            trace_backend="optix",
            edge_bvh_backend="optix",
        )
    if not isinstance(layer, _ReplicatedScene):
        raise AssertionError("two devices must create a replicated policy layer")
    # Policy helpers need replica identity but do not invoke replica methods.
    layer._replicas = ("master", "remote")
    return layer


class PeerTopologyPolicyTests(unittest.TestCase):
    def test_bare_cuda_uses_the_current_device(self) -> None:
        with mock.patch.object(torch.cuda, "current_device", return_value=3):
            self.assertEqual(_device_index("cuda", 0), 3)

    def test_peer_access_is_required_by_default(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(
                torch.cuda, "can_device_access_peer", return_value=False, create=True
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(name="Test GPU", major=9, minor=0),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "(?i)peer|p2p"):
                plan_multi_device(
                    [0, 1],
                    rt.MultiDeviceOptions(warm_up=False),
                    trace_backend="optix",
                    edge_bvh_backend="optix",
                )

    def test_the_caller_can_explicitly_accept_a_non_peer_link(self) -> None:
        options = rt.MultiDeviceOptions(
            warm_up=False,
            require_peer_access=False,
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(
                torch.cuda, "can_device_access_peer", return_value=False, create=True
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(name="Test GPU", major=9, minor=0),
            ),
        ):
            layer = plan_multi_device(
                [0, 1],
                options,
                trace_backend="optix",
                edge_bvh_backend="optix",
            )
        self.assertIsInstance(layer, _ReplicatedScene)

    def test_heterogeneous_devices_are_an_explicit_opt_in(self) -> None:
        properties = {
            0: SimpleNamespace(name="GPU A", major=9, minor=0),
            1: SimpleNamespace(name="GPU B", major=8, minor=6),
        }
        common = (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(
                torch.cuda, "can_device_access_peer", return_value=True, create=True
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect=lambda index: properties[index],
            ),
        )
        with common[0], common[1], common[2], common[3]:
            with self.assertRaisesRegex(RuntimeError, "(?i)identical|heterogeneous"):
                plan_multi_device(
                    [0, 1],
                    rt.MultiDeviceOptions(warm_up=False),
                    trace_backend="optix",
                    edge_bvh_backend="optix",
                )
        common = (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(
                torch.cuda, "can_device_access_peer", return_value=True, create=True
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect=lambda index: properties[index],
            ),
        )
        with common[0], common[1], common[2], common[3]:
            layer = plan_multi_device(
                [0, 1],
                rt.MultiDeviceOptions(
                    warm_up=False,
                    require_homogeneous_devices=False,
                ),
                trace_backend="optix",
                edge_bvh_backend="optix",
            )
        self.assertIsInstance(layer, _ReplicatedScene)


class OperationPolicyTests(unittest.TestCase):
    def test_default_intersection_probe_uses_the_private_dispatch_signature(self) -> None:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(_ReplicatedScene._default_probe))
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "intersect"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            len(calls[0].args),
            3,
            "the built-in probe must work for both Scene and _ReplicatedScene",
        )

    def test_builtin_probe_cannot_be_mislabeled_as_another_operation(self) -> None:
        layer = _planned(rt.MultiDeviceOptions(warm_up=False))
        with self.assertRaisesRegex(ValueError, "(?i)built-in|custom probe"):
            layer.calibrate(
                operation="visible",
                rows=100,
                repeats=1,
                warm_up=0,
            )

    def test_operation_weights_are_isolated(self) -> None:
        layer = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                min_rays_per_device=1,
                operation_weights={
                    "intersect": [1.0, 0.0],
                    "visible": [1.0, 3.0],
                },
            )
        )

        self.assertEqual(layer._weights_for("intersect"), (1.0, 0.0))
        self.assertEqual(layer._weights_for("visible"), (1.0, 3.0))
        self.assertEqual(layer._weights_for("nearest_edge"), (1.0, 1.0))

        intersect = layer._shards(100, "intersect")
        visible = layer._shards(100, "visible")
        intersect_again = layer._shards(100, "intersect")
        self.assertEqual([(start, stop) for _r, _d, start, stop in intersect], [(0, 100)])
        self.assertEqual(
            [(start, stop) for _r, _d, start, stop in visible],
            [(0, 25), (25, 100)],
        )
        self.assertEqual(
            [(start, stop) for _r, _d, start, stop in intersect_again],
            [(0, 100)],
        )

    def test_calibration_updates_only_the_named_operation(self) -> None:
        layer = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                operation_weights={"intersect": [1.0, 0.0]},
            )
        )
        with (
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(
                torch.cuda, "device", side_effect=lambda _d: nullcontext()
            ),
            mock.patch(
                "rayd.torch._multi.time.perf_counter",
                side_effect=[0.0, 1.0, 2.0, 4.0],
            ),
        ):
            record = layer.calibrate(
                operation="visible",
                rows=100,
                probe=lambda _scene, _device: None,
                repeats=1,
                warm_up=0,
                refine=False,
            )

        self.assertEqual(layer._weights_for("visible"), record.weights)
        self.assertEqual(layer._weights_for("intersect"), (1.0, 0.0))
        self.assertEqual(layer._weights_for("nearest_edge"), (1.0, 1.0))

    def test_dispatch_floor_uses_the_actual_remote_shard(self) -> None:
        layer = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                min_rays_per_device=10,
                weights=[91.0, 9.0],
            )
        )
        # The old total-row check sharded this batch because 100 >= 2 * 10,
        # even though the remote launch received only nine rows.
        self.assertEqual(layer._dispatch_mode("visible", 100, 100), "master")

        enough_remote_work = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                min_rays_per_device=10,
                weights=[9.0, 1.0],
            )
        )
        self.assertEqual(
            enough_remote_work._dispatch_mode("visible", 100, 100),
            "pipelined",
        )

    def test_wide_rows_raise_the_copy_amortization_floor(self) -> None:
        layer = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                min_rays_per_device=10,
                weights=[1.0, 1.0],
            )
        )
        narrow = layer._dispatch_mode("visible", 100, 1)
        wide = layer._dispatch_mode("intersect", 100, 1 << 20)
        self.assertEqual(narrow, "pipelined")
        self.assertEqual(wide, "master")

    def test_small_grid_reduce_window_runs_on_the_master(self) -> None:
        layer = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                min_lanes_per_device=128,
                operation_weights={"accum_dfr_direct": [1.0, 1.0]},
            )
        )
        calls: list[tuple[object, torch.device, int, int]] = []
        answer = object()

        def scatter(device: torch.device) -> torch.device:
            return device

        def call(replica, inputs, begin: int, count: int):
            calls.append((replica, inputs, begin, count))
            return answer

        result = layer._run_lane_shards(
            "accum_dfr_direct",
            64,
            0,
            -1,
            scatter,
            call,
        )
        self.assertIs(result, answer)
        self.assertEqual(calls, [("master", torch.device("cuda", 0), 0, 64)])
        self.assertEqual(layer.last_dispatch, "master")

    def test_grid_reduce_master_only_weight_never_launches_a_remote(self) -> None:
        layer = _planned(
            rt.MultiDeviceOptions(
                warm_up=False,
                min_lanes_per_device=1,
                operation_weights={"accum_dfr_direct": [1.0, 0.0]},
            )
        )
        calls: list[object] = []
        answer = object()

        def call(replica, _inputs, _begin: int, _count: int):
            calls.append(replica)
            return answer

        result = layer._run_lane_shards(
            "accum_dfr_direct",
            4096,
            0,
            -1,
            lambda device: device,
            call,
        )
        self.assertIs(result, answer)
        self.assertEqual(calls, ["master"])
        self.assertEqual(layer.last_dispatch, "master")


class MemoryBudgetPolicyTests(unittest.TestCase):
    @staticmethod
    def _accum(*, requires_grad: bool) -> DfrAccum:
        values = [
            torch.ones(1, requires_grad=requires_grad)
            for _ in range(14)
        ]
        return DfrAccum(1, *values)

    def test_budget_accounts_for_three_resident_pipeline_chunks(self) -> None:
        row_bytes = 142
        budget = row_bytes * 3000
        plan = calibrate_chunk_size(
            "trace_reflections",
            1_000_000,
            row_bytes=row_bytes,
            budget_bytes=budget,
            resident_chunks=3,
        )
        self.assertLessEqual(plan.chunk_rays * row_bytes * 3, budget)
        self.assertGreater((plan.chunk_rays + 1) * row_bytes * 3, budget)

    def test_fixed_concatenated_output_is_reserved_before_chunk_sizing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "(?i)fixed|returned output|budget"):
            calibrate_chunk_size(
                "intersect",
                100,
                row_bytes=76,
                budget_bytes=8192,
                resident_chunks=3,
                fixed_output_bytes=8192,
            )

    def test_grid_accumulation_autograd_graph_grows_with_chunk_count(self) -> None:
        """The value buffer is fixed-size, but ordinary ``+`` retains every tape."""

        def graph_nodes(value: torch.Tensor) -> int:
            seen: set[object] = set()
            pending = [value.grad_fn]
            while pending:
                node = pending.pop()
                if node is None or node in seen:
                    continue
                seen.add(node)
                pending.extend(parent for parent, _index in node.next_functions)
            return len(seen)

        two = _add_accum(
            self._accum(requires_grad=True),
            self._accum(requires_grad=True),
        )
        many = self._accum(requires_grad=True)
        for _ in range(7):
            many = _add_accum(many, self._accum(requires_grad=True))
        self.assertGreater(graph_nodes(many.power), graph_nodes(two.power))

    def test_inference_grid_merge_reuses_the_left_buffers(self) -> None:
        left = self._accum(requires_grad=False)
        right = self._accum(requires_grad=False)
        power = left.power
        merged = _add_accum_in_place(left, right)
        self.assertIs(merged, left)
        self.assertIs(merged.power, power)
        torch.testing.assert_close(merged.power, torch.full((1,), 2.0))

    def test_in_place_grid_merge_refuses_autograd(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "(?i)autograd"):
            _add_accum_in_place(
                self._accum(requires_grad=True),
                self._accum(requires_grad=True),
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class CudaPeakMemoryBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda", 0)
        vertices = torch.tensor(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        vertices.requires_grad_()
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=self.device)
        self.mesh = rt.Mesh(vertices, faces)

    def _scene(self, **options) -> rt.Scene:
        scene = rt.Scene(
            devices=[0],
            options=rt.MultiDeviceOptions(warm_up=False, **options),
        )
        scene.add_mesh(self.mesh)
        scene.build()
        return scene

    def _ray(self, rows: int) -> rt.Ray:
        origins = torch.zeros((rows, 3), dtype=torch.float32, device=self.device)
        origins[:, 2] = -1.0
        directions = torch.zeros_like(origins)
        directions[:, 2] = 1.0
        return rt.Ray(origins.contiguous(), directions.contiguous())

    def test_a_concatenated_result_larger_than_the_budget_fails_loudly(self) -> None:
        scene = self._scene(tape_memory_budget_bytes=4096)
        with self.assertRaisesRegex(RuntimeError, "(?i)budget|offload"):
            scene.intersect(self._ray(4096), flags=rt.RayFlags.All)

    def test_streaming_peak_increment_respects_the_budget(self) -> None:
        budget = 16 << 20
        scene = self._scene(
            tape_memory_budget_bytes=budget,
            offload=lambda _start, _chunk: None,
        )
        ray = self._ray(250_000)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        self.assertIsNone(scene.intersect(ray, flags=rt.RayFlags.All))
        torch.cuda.synchronize(self.device)
        peak_increment = torch.cuda.max_memory_allocated(self.device) - baseline
        # CUDA allocator block rounding and small fixed Python/Torch metadata
        # are outside the row estimate.  Eight MiB is a deliberately coarse
        # allowance; a missing three-resident-chunk factor exceeds it here.
        self.assertLessEqual(peak_increment, budget + (8 << 20))

    def test_budgeted_multichunk_autograd_requires_per_chunk_backward(self) -> None:
        scene = self._scene(tape_memory_budget_bytes=1 << 20)
        with self.assertRaisesRegex(RuntimeError, "(?i)differentiable|backward|offload"):
            scene.intersect(self._ray(4096), flags=rt.RayFlags.All)


if __name__ == "__main__":
    unittest.main()
