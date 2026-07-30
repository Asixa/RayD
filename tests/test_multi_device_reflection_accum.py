# Copyright Xingyu Chen.
# Tests replicated Torch reflection-accumulation grid reduction.

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import unittest

import torch

import rayd.torch as rt
from rayd._impl.multi import MultiDeviceOptions, _ReplicatedScene


ROOT = Path(__file__).resolve().parents[1]


def _empty_wedges(device: torch.device) -> rt.WedgeEvents:
    empty_i = torch.empty((0,), device=device, dtype=torch.int32)
    empty_f = torch.empty((0,), device=device, dtype=torch.float32)
    empty_v = torch.empty((0, 3), device=device, dtype=torch.float32)
    return rt.WedgeEvents(
        0,
        torch.zeros((1,), device=device, dtype=torch.int32),
        empty_i,
        empty_v,
        empty_v,
        empty_i,
        empty_v,
        empty_v,
        empty_f,
        empty_v,
        empty_i,
    )


def _partial(rows: int, max_bounces: int, grid: rt.AccumGrid, device: torch.device) -> rt.AccumResult:
    shape = (int(grid.resolution1), int(grid.resolution0))
    values = [torch.full(shape, float(rows), device=device) for _ in range(7)]
    count = torch.tensor([rows], device=device, dtype=torch.int32)
    return rt.AccumResult(
        rows, max_bounces, int(grid.resolution0) * int(grid.resolution1), *values, count, _empty_wedges(device)
    )


class _FakeReplica:
    def __init__(self):
        self.calls = []
        self.results = []

    def accumulate_reflections(self, ray, tx_position, grid, material, max_bounces, options, active, tx_polarization):
        self.calls.append((ray, tx_position, material, options, active, tx_polarization))
        result = _partial(int(ray.o.shape[0]), int(max_bounces), grid, ray.o.device)
        self.results.append(result)
        return result


class _CpuRay(NamedTuple):
    o: torch.Tensor
    d: torch.Tensor
    tmax: torch.Tensor


class _CpuReplicatedScene(_ReplicatedScene):
    __slots__ = ()

    def _require_master_tensors(self, operation, values):
        del operation, values

    def _shard_ray(self, ray, start: int, stop: int, device: torch.device):
        return _CpuRay(
            self._slice(ray.o, start, stop, device),
            self._slice(ray.d, start, stop, device),
            self._slice(ray.tmax, start, stop, device),
        )


def _orchestrator(device_count: int, options: MultiDeviceOptions) -> tuple[_ReplicatedScene, list[_FakeReplica]]:
    orchestrator = object.__new__(_CpuReplicatedScene)
    devices = tuple(torch.device("cpu", index) for index in range(device_count))
    replicas = [_FakeReplica() for _ in devices]
    orchestrator.devices = devices
    orchestrator.options = options
    orchestrator.weights = tuple(1.0 for _ in devices)
    orchestrator._base_weights = orchestrator.weights
    orchestrator._operation_weights = {}
    orchestrator._calibration_override = None
    orchestrator.chunked = any(
        value is not None for value in (options.chunk_rays, options.tape_memory_budget_bytes, options.offload)
    )
    orchestrator.pipelined = False
    orchestrator.min_rays_per_device = int(options.min_rays_per_device)
    orchestrator.min_lanes_per_device = int(options.min_lanes_per_device)
    orchestrator.last_chunk_plan = None
    orchestrator.last_dispatch = None
    orchestrator.last_calibration = None
    orchestrator._trace_backend = "optix"
    orchestrator._edge_bvh_backend = "optix"
    orchestrator._replicas = tuple(replicas)
    orchestrator._streams = {}
    orchestrator._active_stream = None
    orchestrator._poisoned = None
    return orchestrator, replicas


def _inputs(rows: int):
    device = torch.device("cpu")
    ray = _CpuRay(
        torch.zeros((rows, 3), device=device), torch.ones((rows, 3), device=device), torch.ones((rows,), device=device)
    )
    tx = torch.zeros((rows, 3), device=device)
    polarization = torch.zeros((rows, 3), device=device)
    active = torch.ones((rows,), device=device, dtype=torch.bool)
    material = rt.ReflMaterial.default(1, device=device)
    grid = rt.AccumGrid(resolution0=1, resolution1=1)
    return ray, tx, polarization, active, material, grid


class ReflectionAccumulationFakeOrchestratorTests(unittest.TestCase):
    def test_two_shards_merge_grid_and_global_metadata_in_device_order(self):
        orchestrator, replicas = _orchestrator(2, MultiDeviceOptions(warm_up=False, min_lanes_per_device=1))
        ray, tx, polarization, active, material, grid = _inputs(64)
        result = orchestrator.accumulate_reflections(
            ray=ray,
            tx_position=tx,
            grid=grid,
            material=material,
            max_bounces=1,
            options=rt.AccumOptions(),
            active=active,
            tx_polarization=polarization,
        )

        self.assertEqual([call[0].o.shape[0] for replica in replicas for call in replica.calls], [32, 32])
        self.assertTrue(all(call[3].accumulation_strategy == 1 for replica in replicas for call in replica.calls))
        self.assertEqual(result.ray_count, 64)
        self.assertEqual(result.max_bounces, 1)
        self.assertEqual(result.grid_cell_count, 1)
        self.assertEqual(result.reflection_power.item(), 64.0)
        self.assertEqual(result.reflection_count.item(), 64)
        self.assertEqual(result.wedge_events.capacity, 0)
        self.assertEqual(orchestrator.last_dispatch, "batch-sharded")

    def test_default_one_device_path_is_the_exact_master_result(self):
        orchestrator, replicas = _orchestrator(1, MultiDeviceOptions(warm_up=False, min_lanes_per_device=1))
        ray, tx, polarization, active, material, grid = _inputs(64)
        result = orchestrator.accumulate_reflections(
            ray=ray,
            tx_position=tx,
            grid=grid,
            material=material,
            max_bounces=1,
            options=rt.AccumOptions(),
            active=active,
            tx_polarization=polarization,
        )

        self.assertIs(result, replicas[0].results[0])
        self.assertEqual(len(replicas[0].calls), 1)
        self.assertEqual(replicas[0].calls[0][0].o.shape[0], 64)
        self.assertEqual(orchestrator.last_dispatch, "master")

    def test_auto_strategy_is_resolved_from_the_unsplit_batch(self):
        orchestrator, replicas = _orchestrator(2, MultiDeviceOptions(warm_up=False, min_lanes_per_device=1))
        ray, tx, polarization, active, material, grid = _inputs(2048)
        orchestrator.accumulate_reflections(
            ray=ray,
            tx_position=tx,
            grid=grid,
            material=material,
            max_bounces=0,
            options=rt.AccumOptions(),
            active=active,
            tx_polarization=polarization,
        )
        self.assertTrue(all(call[3].accumulation_strategy == 2 for replica in replicas for call in replica.calls))

    def test_wedge_collection_preserves_native_global_buffer_on_master(self):
        orchestrator, replicas = _orchestrator(2, MultiDeviceOptions(warm_up=False, min_lanes_per_device=1))
        ray, tx, polarization, active, material, grid = _inputs(64)
        orchestrator.accumulate_reflections(
            ray=ray,
            tx_position=tx,
            grid=grid,
            material=material,
            max_bounces=1,
            options=rt.AccumOptions(collect_wedges=True, wedge_capacity=64),
            active=active,
            tx_polarization=polarization,
        )

        self.assertEqual(len(replicas[0].calls), 1)
        self.assertEqual(len(replicas[1].calls), 0)
        self.assertEqual(replicas[0].calls[0][0].o.shape[0], 64)
        self.assertEqual(orchestrator.last_dispatch, "master")

    def test_wedge_collection_rejects_chunking_instead_of_changing_capacity(self):
        orchestrator, _replicas = _orchestrator(
            2, MultiDeviceOptions(warm_up=False, min_lanes_per_device=1, chunk_rays=32)
        )
        ray, tx, polarization, active, material, grid = _inputs(64)
        with self.assertRaisesRegex(NotImplementedError, "global capacity"):
            orchestrator.accumulate_reflections(
                ray=ray,
                tx_position=tx,
                grid=grid,
                material=material,
                max_bounces=1,
                options=rt.AccumOptions(collect_wedges=True, wedge_capacity=64),
                active=active,
                tx_polarization=polarization,
            )

    def test_empty_active_tensor_keeps_single_device_absent_mask_semantics(self):
        orchestrator, replicas = _orchestrator(2, MultiDeviceOptions(warm_up=False, min_lanes_per_device=1))
        ray, tx, polarization, _active, material, grid = _inputs(64)
        orchestrator.accumulate_reflections(
            ray=ray,
            tx_position=tx,
            grid=grid,
            material=material,
            max_bounces=1,
            options=rt.AccumOptions(),
            active=torch.empty((0,), dtype=torch.bool),
            tx_polarization=polarization,
        )
        self.assertTrue(all(call[4].numel() == 0 for replica in replicas for call in replica.calls))


class ReflectionAccumulationStaticContractTests(unittest.TestCase):
    def test_scene_routes_after_the_forward_only_guard(self):
        scene = (ROOT / "python" / "rayd" / "_impl" / "scene.py").read_text(encoding="utf-8")
        body = scene.split("    def accumulate_reflections(", 1)[1].split("    def _default_dfr_material(", 1)[0]
        self.assertLess(body.index("_has_reverse_or_forward_ad"), body.index("self._multi.accumulate_reflections"))
        self.assertNotIn('self._multi.unsupported("accumulate_reflections")', body)

    def test_orchestrator_has_no_native_lane_or_kernel_extension(self):
        multi = (ROOT / "python" / "rayd" / "_impl" / "multi.py").read_text(encoding="utf-8")
        body = multi.split("    def accumulate_reflections(", 1)[1].split("    def _run_lane_shards(", 1)[0]
        self.assertIn("self._lane_shards(0, total, operation)", body)
        self.assertIn("_add_reflection_accum_in_place", body)
        self.assertIn("return master()", body)
        self.assertNotIn("torch.ops.", body)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices are required")
class ReflectionAccumulationTwoGpuAcceptanceTests(unittest.TestCase):
    @staticmethod
    def _scene(*, devices=None):
        vertices = torch.tensor(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]], device="cuda:0", dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda:0", dtype=torch.int32)
        options = rt.MultiDeviceOptions(warm_up=False, min_lanes_per_device=1) if devices else None
        scene = rt.Scene(devices=devices, options=options)
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        return scene

    def test_two_gpu_grid_matches_single_device_with_explicit_active_and_polarization(self):
        rows = 64
        single = self._scene()
        multi = self._scene(devices=[0, 1])
        ray_o = torch.zeros((rows, 3), device="cuda:0")
        ray_o[:, :2] = -0.25
        ray_o[:, 2] = -1.0
        ray_d = torch.zeros_like(ray_o)
        ray_d[:, 2] = 1.0
        ray = rt.Ray(ray_o, ray_d, torch.full((rows,), 2.0, device="cuda:0"))
        active = torch.arange(rows, device="cuda:0") % 3 != 0
        polarization = torch.zeros_like(ray_o)
        polarization[:, 0] = 1.0
        grid = rt.AccumGrid(
            axis=2,
            position=-2.0,
            coord0_min=-2.0,
            coord0_max=2.0,
            coord1_min=-2.0,
            coord1_max=2.0,
            resolution0=2,
            resolution1=2,
        )
        material = rt.ReflMaterial(
            eta_r=torch.full((1,), 4.0, device=ray_o.device),
            sigma=torch.zeros((1,), device=ray_o.device),
            mu_r=torch.ones((1,), device=ray_o.device),
            gain=torch.ones((1,), device=ray_o.device),
            valid=torch.ones((1,), device=ray_o.device, dtype=torch.bool),
        )
        options = rt.AccumOptions(wavelength=12.566370614359172)
        reference = single.accumulate_reflections(ray, ray_o, grid, material, 1, options, active, polarization)
        merged = multi.accumulate_reflections(ray, ray_o, grid, material, 1, options, active, polarization)

        for name in (
            "reflection_power",
            "reflection_field_x_re",
            "reflection_field_x_im",
            "reflection_field_y_re",
            "reflection_field_y_im",
            "reflection_field_z_re",
            "reflection_field_z_im",
        ):
            torch.testing.assert_close(getattr(merged, name), getattr(reference, name), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(merged.reflection_count, reference.reflection_count, rtol=0, atol=0)
        self.assertEqual(merged.ray_count, rows)
        self.assertEqual(multi._multi.last_dispatch, "batch-sharded")

    def test_empty_batch_uses_the_master_native_contract(self):
        multi = self._scene(devices=[0, 1])
        empty3 = torch.empty((0, 3), device="cuda:0")
        empty1 = torch.empty((0,), device="cuda:0")
        result = multi.accumulate_reflections(
            rt.Ray(empty3, empty3, empty1),
            empty3,
            rt.AccumGrid(resolution0=1, resolution1=1),
            rt.ReflMaterial.default(1, device=empty3.device),
            1,
            active=torch.empty((0,), device="cuda:0", dtype=torch.bool),
            tx_polarization=empty3,
        )
        self.assertEqual(result.ray_count, 0)
        self.assertEqual(multi._multi.last_dispatch, "master")


if __name__ == "__main__":
    unittest.main()
