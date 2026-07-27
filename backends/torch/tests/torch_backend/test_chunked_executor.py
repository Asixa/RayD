"""Phase 2b of `docs/dev/multi_gpu_plan.md`: the chunked executor.

The contract under test is decision D7: a large batch is executed as a stream
of chunks per device, and that changes nothing a caller can observe except
memory. Concretely -- a chunked `per_ray` result is bitwise the unchunked
result at any chunk size, an `offload` hook sees every chunk exactly once and
in order per device while the operation itself returns `None`, a tape budget
really does shrink the launches, and a chunked forward with a per-chunk
backward accumulates the same gradient the unchunked backward produces (D4).

Chunking is engaged for a one-device scene too, so everything here except the
two-device gather runs on a single GPU.
"""

import unittest

import torch
import rayd.torch as rt

from rayd.torch._multi import (
    _NEAREST_EDGES_TOPK_FIELDS,
    _NEAREST_POINT_EDGE_FIELDS,
    _NEAREST_RAY_EDGE_FIELDS,
    _REFL_EPC_FIELD_FIELDS,
    _SEGMENT_PAIR_FIELDS,
    ChunkPlan,
    calibrate_chunk_size,
)


# 33 rows is ragged under every chunk size used here (1, 7, 11, 64) and under
# the 17/16 two-device split, so no case degenerates into an even division.
_BATCH = 33


def _grid_mesh(device: torch.device, cells: int = 8, span: float = 2.0):
    """Deterministic z=0 triangle grid; identical bits on every CUDA device."""
    axis = torch.linspace(-0.5 * span, 0.5 * span, cells + 1, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    flat_x = x.reshape(-1)
    vertices = torch.stack((flat_x, y.reshape(-1), torch.zeros_like(flat_x)), dim=1)
    index = torch.arange((cells + 1) * (cells + 1), dtype=torch.int32).reshape(
        cells + 1, cells + 1
    )
    a = index[:-1, :-1].reshape(-1)
    b = index[:-1, 1:].reshape(-1)
    c = index[1:, :-1].reshape(-1)
    d = index[1:, 1:].reshape(-1)
    faces = torch.cat((torch.stack((a, b, c), dim=1), torch.stack((b, d, c), dim=1)))
    return vertices.contiguous().to(device), faces.contiguous().to(device)


def _query_inputs(device: torch.device, count: int = _BATCH) -> dict:
    """One deterministic query batch with hits, misses and blocked segments."""
    generator = torch.Generator().manual_seed(20260727)
    origins = torch.rand((count, 3), generator=generator) * 1.8 - 0.9
    origins[:, 2] = -1.0
    directions = torch.zeros((count, 3))
    directions[:, 2] = 1.0
    points = torch.rand((count, 3), generator=generator) * 2.4 - 1.2
    active = torch.zeros((count,), dtype=torch.bool)
    active[::2] = True

    origins = origins.contiguous().to(device)
    directions = directions.contiguous().to(device)
    points = points.contiguous().to(device)
    end = (origins + torch.tensor([[0.0, 0.0, 2.0]], device=device)).contiguous()
    return {
        "ray": rt.Ray(origins, directions),
        "origins": origins,
        "points": points,
        "active": active.to(device),
        "end": end,
    }


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Host copy compared bit-for-bit, so NaN, -0.0, and inf all compare exactly."""
    host = tensor.detach().contiguous().cpu()
    if host.dtype == torch.float32:
        return host.view(torch.int32)
    if host.dtype == torch.float64:
        return host.view(torch.int64)
    return host


def _build_scene(device: torch.device, **kwargs) -> rt.Scene:
    vertices, faces = _grid_mesh(device)
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


def _chunked_ops(scene: rt.Scene, inputs: dict) -> dict[str, torch.Tensor]:
    """The three operations the plan names for chunked coverage, field by field."""
    ray = inputs["ray"]
    active = inputs["active"]
    reduced = scene.intersect(ray, flags=getattr(rt.RayFlags, "None"))
    full = scene.intersect(ray, active, flags=rt.RayFlags.All)
    chain = scene.trace_reflections(ray, max_bounces=2, active=active)
    visible = scene.visible(inputs["origins"], inputs["end"], active)
    results = {
        "intersect_reduced.t": reduced.t,
        "trace_reflections.valid": chain.valid,
        "trace_reflections.t": chain.t,
        "trace_reflections.prim_ids": chain.prim_ids,
        "trace_reflections.image_sources": chain.image_sources,
        "visible": visible,
    }
    for name in (
        "t",
        "p",
        "n",
        "geo_n",
        "uv",
        "barycentric",
        "shape_id",
        "prim_id",
        "local_prim_id",
        "global_prim_id",
    ):
        results[f"intersect_full.{name}"] = getattr(full, name)
    return results


class ChunkSizeCalibrationTests(unittest.TestCase):
    """`calibrate_chunk_size()` is pure: no device needed to pin its decisions."""

    def test_an_explicit_request_wins_and_is_clamped_to_the_batch(self):
        plan = calibrate_chunk_size("intersect", 33, row_bytes=76, chunk_rays=7)
        self.assertEqual((plan.chunk_rays, plan.source), (7, "requested"))
        clamped = calibrate_chunk_size("intersect", 33, row_bytes=76, chunk_rays=64)
        self.assertEqual(clamped.chunk_rays, 33)
        # A request also beats a budget that would have chosen something else.
        both = calibrate_chunk_size(
            "intersect", 33, row_bytes=76, chunk_rays=7, budget_bytes=1 << 30
        )
        self.assertEqual((both.chunk_rays, both.source), (7, "requested"))

    def test_a_budget_picks_the_largest_chunk_that_fits(self):
        plan = calibrate_chunk_size(
            "trace_reflections", 1_000_000, row_bytes=142, budget_bytes=142 * 1000
        )
        self.assertEqual((plan.chunk_rays, plan.source), (1000, "budget"))
        self.assertEqual(plan.budget_bytes, 142 * 1000)

    def test_a_budget_below_one_row_still_makes_progress(self):
        plan = calibrate_chunk_size("trace_reflections", 64, row_bytes=142, budget_bytes=1)
        self.assertEqual((plan.chunk_rays, plan.source), (1, "budget"))

    def test_without_a_request_or_a_budget_a_chunk_is_the_whole_shard(self):
        plan = calibrate_chunk_size("visible", 33, row_bytes=1)
        self.assertEqual((plan.chunk_rays, plan.source), (33, "shard"))
        empty = calibrate_chunk_size("visible", 0, row_bytes=1)
        self.assertEqual(empty.chunk_rays, 1)

    def test_the_plan_reports_what_it_was_asked(self):
        plan = calibrate_chunk_size("intersect", 33, row_bytes=76, chunk_rays=7)
        self.assertIsInstance(plan, ChunkPlan)
        self.assertEqual(plan.operation, "intersect")
        self.assertEqual(plan.total_rows, 33)
        self.assertEqual(plan.row_bytes, 76)
        self.assertIsNone(plan.budget_bytes)
        self.assertIsNone(plan.measured_row_bytes)


class ChunkOptionValidationTests(unittest.TestCase):
    def test_the_new_option_defaults_leave_chunking_off(self):
        options = rt.MultiDeviceOptions()
        self.assertIsNone(options.chunk_rays)
        self.assertIsNone(options.offload)
        self.assertIsNone(options.tape_memory_budget_bytes)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
    def test_the_chunking_knobs_are_validated(self):
        for options in (
            rt.MultiDeviceOptions(chunk_rays=0),
            rt.MultiDeviceOptions(chunk_rays=-4),
            rt.MultiDeviceOptions(tape_memory_budget_bytes=0),
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    rt.Scene(devices=[0], options=options)
        for options in (
            rt.MultiDeviceOptions(chunk_rays=8.0),
            rt.MultiDeviceOptions(chunk_rays=True),
            rt.MultiDeviceOptions(tape_memory_budget_bytes="1MB"),
            rt.MultiDeviceOptions(offload=object()),
        ):
            with self.subTest(options=options):
                with self.assertRaises(TypeError):
                    rt.Scene(devices=[0], options=options)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class ChunkedEngagementTests(unittest.TestCase):
    """Who gets an orchestrator, and who keeps the untouched fast path (D9)."""

    def test_a_scene_that_asks_for_nothing_keeps_the_single_device_path(self):
        self.assertIsNone(rt.Scene()._multi)
        self.assertIsNone(rt.Scene(devices=[0])._multi)
        self.assertIsNone(
            rt.Scene(devices=[0], options=rt.MultiDeviceOptions(warm_up=False))._multi
        )

    def test_any_chunking_knob_engages_the_layer_on_one_device(self):
        for options in (
            rt.MultiDeviceOptions(warm_up=False, chunk_rays=8),
            rt.MultiDeviceOptions(warm_up=False, tape_memory_budget_bytes=1 << 20),
            rt.MultiDeviceOptions(warm_up=False, offload=lambda start, result: None),
        ):
            with self.subTest(options=options):
                scene = rt.Scene(devices=[0], options=options)
                self.assertIsNotNone(scene._multi)
                self.assertTrue(scene._multi.chunked)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class ChunkedSingleDeviceTests(unittest.TestCase):
    """One device, many chunks: the memory story of D7 without any sharding."""

    def setUp(self) -> None:
        self.device = torch.device("cuda", 0)
        self.inputs = _query_inputs(self.device)
        self.reference = _chunked_ops(_build_scene(self.device), self.inputs)

    def _scene(self, **options) -> rt.Scene:
        return _build_scene(
            self.device,
            devices=[0],
            options=rt.MultiDeviceOptions(warm_up=False, **options),
        )

    def test_chunked_results_are_bitwise_the_unchunked_results(self):
        # 1 and 7 are ragged, 11 divides the batch exactly, 64 exceeds it.
        for chunk_rays in (1, 7, 11, 64):
            with self.subTest(chunk_rays=chunk_rays):
                scene = self._scene(chunk_rays=chunk_rays)
                results = _chunked_ops(scene, self.inputs)
                self.assertEqual(sorted(results), sorted(self.reference))
                for name, value in results.items():
                    expected = self.reference[name]
                    self.assertEqual(value.dtype, expected.dtype, name)
                    self.assertEqual(value.shape, expected.shape, name)
                    self.assertEqual(value.device, expected.device, name)
                    self.assertTrue(
                        torch.equal(_bits(value), _bits(expected)),
                        f"chunk_rays={chunk_rays}: {name} is not bitwise equal",
                    )
                plan = scene._multi.last_chunk_plan
                self.assertEqual(plan.chunk_rays, min(chunk_rays, _BATCH))
                self.assertEqual(
                    plan.chunk_count, -(-_BATCH // min(chunk_rays, _BATCH))
                )

    def test_every_wrapped_operation_survives_a_ragged_chunking(self):
        """The executor is generic; the ops the plan does not name work too."""
        scene = self._scene(chunk_rays=7)
        single = _build_scene(self.device)
        points = self.inputs["points"]
        active = self.inputs["active"]
        cases = {
            "nearest_edge_point": (
                lambda target: target.nearest_edge(points),
                _NEAREST_POINT_EDGE_FIELDS,
            ),
            "nearest_edge_ray": (
                lambda target: target.nearest_edge(self.inputs["ray"]),
                _NEAREST_RAY_EDGE_FIELDS,
            ),
            "nearest_edges": (
                lambda target: target.nearest_edges(points, 3, active),
                _NEAREST_EDGES_TOPK_FIELDS,
            ),
            "visible_pair": (
                lambda target: target.visible_pair(
                    self.inputs["origins"], self.inputs["end"], self.inputs["end"] + 4.0
                ),
                _SEGMENT_PAIR_FIELDS,
            ),
            "trace_refl_epc_field": (
                lambda target: target.trace_refl_epc_field(
                    self.inputs["origins"], self.inputs["end"], 2
                ),
                _REFL_EPC_FIELD_FIELDS,
            ),
        }
        for name, (call, fields) in cases.items():
            with self.subTest(operation=name):
                expected = call(single)
                actual = call(scene)
                for field in fields:
                    self.assertTrue(
                        torch.equal(
                            _bits(getattr(expected, field)),
                            _bits(getattr(actual, field)),
                        ),
                        f"{name}.{field} is not bitwise equal",
                    )

    def test_an_empty_batch_is_still_one_empty_result(self):
        scene = self._scene(chunk_rays=4)
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        ray = rt.Ray(empty, empty)
        self.assertEqual(tuple(scene.intersect(ray).t.shape), (0,))
        self.assertEqual(tuple(scene.visible(empty, empty).shape), (0,))
        self.assertEqual(
            tuple(scene.trace_reflections(ray, max_bounces=2).valid.shape), (0, 2)
        )
        self.assertEqual(scene._multi.last_chunk_plan.chunk_count, 0)

    def test_a_tiny_tape_budget_picks_a_small_chunk_and_still_completes(self):
        """The synthetic memory case: the budget, not the batch, sizes the launch."""
        # 2 bounces at the plan's 50 B/ray/bounce tape estimate plus the chain's
        # own 21 B/ray/bounce output is 142 B/ray, so this budget buys 4 rays.
        scene = self._scene(tape_memory_budget_bytes=142 * 4)
        chain = scene.trace_reflections(
            self.inputs["ray"], max_bounces=2, active=self.inputs["active"]
        )
        plan = scene._multi.last_chunk_plan
        self.assertEqual(plan.source, "budget")
        self.assertEqual(plan.chunk_rays, 4)
        self.assertEqual(plan.chunk_count, 9)
        self.assertLess(plan.chunk_rays, _BATCH)
        self.assertTrue(
            torch.equal(_bits(chain.t), _bits(self.reference["trace_reflections.t"]))
        )
        # The estimate covers the tape as well, so the measured output row is
        # smaller; what matters is that a real chunk was measured at all.
        self.assertIsNotNone(plan.measured_row_bytes)
        self.assertGreater(plan.measured_row_bytes, 0.0)
        self.assertLess(plan.measured_row_bytes, plan.row_bytes)

        # The same budget on a cheap per-row operation buys a bigger chunk.
        cheap = _build_scene(
            self.device,
            devices=[0],
            options=rt.MultiDeviceOptions(warm_up=False, tape_memory_budget_bytes=142 * 4),
        )
        cheap.visible(self.inputs["origins"], self.inputs["end"])
        self.assertEqual(cheap._multi.last_chunk_plan.chunk_rays, _BATCH)

    def test_the_offload_hook_sees_every_chunk_once_and_in_order(self):
        seen = []

        def consume(start, result):
            seen.append((start, result.t.detach().clone()))

        scene = self._scene(chunk_rays=7, offload=consume)
        self.assertIsNone(scene.intersect(self.inputs["ray"], self.inputs["active"]))
        self.assertEqual([start for start, _t in seen], [0, 7, 14, 21, 28])
        self.assertEqual([int(t.shape[0]) for _start, t in seen], [7, 7, 7, 7, 5])
        streamed = torch.cat([t for _start, t in seen])
        self.assertTrue(
            torch.equal(_bits(streamed), _bits(self.reference["intersect_full.t"]))
        )
        for _start, t in seen:
            self.assertEqual(t.device, self.device)

    def test_every_streamed_operation_returns_none(self):
        counts = {"n": 0}

        def consume(start, result):
            counts["n"] += 1

        scene = self._scene(chunk_rays=11, offload=consume)
        ray = self.inputs["ray"]
        self.assertIsNone(scene.intersect(ray, flags=getattr(rt.RayFlags, "None")))
        self.assertIsNone(scene.trace_reflections(ray, max_bounces=2))
        self.assertIsNone(scene.visible(self.inputs["origins"], self.inputs["end"]))
        self.assertEqual(counts["n"], 9)

    def test_a_streamed_chunk_carries_the_whole_result_type(self):
        chains = []
        scene = self._scene(chunk_rays=11, offload=lambda start, result: chains.append(result))
        scene.trace_reflections(
            self.inputs["ray"], max_bounces=2, active=self.inputs["active"]
        )
        self.assertEqual(len(chains), 3)
        for name in ("valid", "t", "prim_ids", "image_sources"):
            streamed = torch.cat([getattr(chain, name) for chain in chains])
            self.assertTrue(
                torch.equal(_bits(streamed), _bits(self.reference[f"trace_reflections.{name}"])),
                f"streamed trace_reflections.{name} is not bitwise equal",
            )

    def test_a_chunked_forward_with_a_per_chunk_backward_accumulates_the_gradient(self):
        """D4/D7: chunked training is gradient accumulation, chunk by chunk."""
        vertices, faces = _grid_mesh(self.device)
        weight = (
            torch.arange(_BATCH, device=self.device, dtype=torch.float32) + 1.0
        ) / _BATCH
        ray = self.inputs["ray"]

        def loss(t, start):
            finite = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
            return (finite * weight[start : start + t.shape[0]]).sum()

        def unchunked_gradient():
            leaf = vertices.clone().requires_grad_(True)
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(leaf, faces))
            scene.build()
            loss(scene.intersect(ray).t, 0).backward()
            return leaf.grad

        def chunked_gradient(chunk_rays):
            leaf = vertices.clone().requires_grad_(True)
            chunks = {"n": 0}

            def consume(start, result):
                # The whole point of chunking under training: the chunk's tape
                # is consumed and released here, not held until the batch ends.
                loss(result.t, start).backward()
                chunks["n"] += 1

            scene = rt.Scene(
                devices=[0],
                options=rt.MultiDeviceOptions(
                    warm_up=False, chunk_rays=chunk_rays, offload=consume
                ),
            )
            scene.add_mesh(rt.Mesh(leaf, faces))
            scene.build()
            self.assertIsNone(scene.intersect(ray))
            self.assertEqual(chunks["n"], -(-_BATCH // chunk_rays))
            return leaf.grad

        expected = unchunked_gradient()
        self.assertGreater(float(expected.abs().max()), 0.0)
        for chunk_rays in (7, 11):
            with self.subTest(chunk_rays=chunk_rays):
                accumulated = chunked_gradient(chunk_rays)
                self.assertIsNotNone(accumulated)
                # Per-chunk backward sums the same per-ray contributions in a
                # different order through the same vertex atomics, so only
                # float32 rounding may differ.
                torch.testing.assert_close(
                    accumulated, expected, rtol=1e-5, atol=1e-6
                )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "two CUDA devices are required",
)
class ChunkedTwoDeviceTests(unittest.TestCase):
    """The overlap smoke test: chunks on two devices still gather one result."""

    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        self.inputs = _query_inputs(self.device)
        self.reference = _chunked_ops(_build_scene(self.device), self.inputs)

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def _scene(self, **options) -> rt.Scene:
        return _build_scene(
            self.device,
            devices=[0, 1],
            options=rt.MultiDeviceOptions(warm_up=False, **options),
        )

    def test_chunked_two_device_results_match_the_single_device_results(self):
        if torch.cuda.get_device_name(0) != torch.cuda.get_device_name(1):
            self.skipTest("bitwise cross-device equality needs identical devices")
        for chunk_rays in (1, 7, 64):
            with self.subTest(chunk_rays=chunk_rays):
                scene = self._scene(chunk_rays=chunk_rays)
                results = _chunked_ops(scene, self.inputs)
                for name, value in results.items():
                    self.assertTrue(
                        torch.equal(_bits(value), _bits(self.reference[name])),
                        f"chunk_rays={chunk_rays}: {name} is not bitwise equal",
                    )
                    self.assertEqual(value.device, self.device, name)
                # Both devices really took chunks: 33 rows split 17/16 needs
                # ceil(17/c) + ceil(16/c) launches, never one.
                plan = scene._multi.last_chunk_plan
                size = min(chunk_rays, _BATCH)
                self.assertEqual(
                    plan.chunk_count, -(-16 // size) + -(-17 // size)
                )

    def test_the_offload_hook_streams_each_device_in_row_order(self):
        seen = []
        scene = self._scene(chunk_rays=7, offload=lambda start, result: seen.append((start, result)))
        self.assertIsNone(scene.visible(self.inputs["origins"], self.inputs["end"]))

        starts = [start for start, _result in seen]
        self.assertEqual(sorted(starts), [0, 7, 14, 16, 23, 30])
        self.assertEqual(len(set(starts)), len(starts))
        # Per device the chunks arrive in increasing row order; across devices
        # they interleave, which is what keeps both devices busy.
        first = [start for start in starts if start < 16]
        second = [start for start in starts if start >= 16]
        self.assertEqual(first, sorted(first))
        self.assertEqual(second, sorted(second))

        streamed = torch.cat(
            [result for _start, result in sorted(seen, key=lambda piece: piece[0])]
        )
        self.assertTrue(torch.equal(_bits(streamed), _bits(self.reference["visible"])))

    def test_the_two_gather_modes_agree(self):
        """The pipelined gather has two shapes; only one of them is differentiable.

        Without gradients the executor allocates the operation's whole output
        once and every chunk copies its rows straight into it; with gradients
        it falls back to a per-chunk copy and a concatenation, because filling
        one buffer slice by slice would make each chunk's backward walk the
        whole buffer's `CopySlices` chain. The two must produce the same bits.
        """
        vertices, faces = _grid_mesh(self.device)
        ray = self.inputs["ray"]

        def run(requires_grad: bool):
            leaf = vertices.clone().requires_grad_(requires_grad)
            scene = rt.Scene(
                devices=[0, 1],
                options=rt.MultiDeviceOptions(
                    warm_up=False, min_rays_per_device=1, pipeline_chunks_per_device=3
                ),
            )
            scene.add_mesh(rt.Mesh(leaf, faces))
            scene.build()
            hit = scene.intersect(ray, flags=rt.RayFlags.All)
            self.assertEqual(scene._multi.last_dispatch, "pipelined")
            return hit

        direct = run(False)
        traced = run(True)
        self.assertFalse(direct.t.requires_grad)
        self.assertTrue(traced.t.requires_grad)
        for name in ("t", "p", "n", "prim_id", "barycentric"):
            self.assertTrue(
                torch.equal(_bits(getattr(direct, name)), _bits(getattr(traced, name))),
                f"{name} differs between the buffered and the concatenated gather",
            )

    def test_a_pipelined_backward_reaches_the_master_leaf(self):
        """The executor's streams are the ones backward runs on; the sum still lands."""
        vertices, faces = _grid_mesh(self.device)
        ray = self.inputs["ray"]

        def gradient(devices, **options):
            leaf = vertices.clone().requires_grad_(True)
            if devices is None:
                scene = rt.Scene()
            else:
                scene = rt.Scene(
                    devices=devices,
                    options=rt.MultiDeviceOptions(warm_up=False, **options),
                )
            scene.add_mesh(rt.Mesh(leaf, faces))
            scene.build()
            hit = scene.intersect(ray)
            chain = scene.trace_reflections(ray, max_bounces=2)
            reduced = torch.where(chain.valid, chain.t, torch.zeros_like(chain.t)).sum()
            (
                torch.where(torch.isfinite(hit.t), hit.t, torch.zeros_like(hit.t)).sum()
                + reduced
            ).backward()
            return leaf.grad, getattr(scene._multi, "last_dispatch", None)

        expected, _ = gradient(None)
        self.assertGreater(float(expected.abs().max()), 0.0)
        for chunks in (2, 5):
            with self.subTest(chunks=chunks):
                piped, dispatch = gradient(
                    [0, 1], min_rays_per_device=1, pipeline_chunks_per_device=chunks
                )
                self.assertEqual(dispatch, "pipelined")
                torch.testing.assert_close(piped, expected, rtol=1e-5, atol=1e-6)

    def test_the_pipeline_streams_are_created_once_and_reused(self):
        """A per-call stream would leak one CUDA stream per query."""
        scene = self._scene(min_rays_per_device=1)
        scene.intersect(self.inputs["ray"])
        streams = dict(scene._multi._streams)
        self.assertEqual(sorted(streams), [0, 1])
        # The master needs no copy streams; the other device needs one per
        # direction on each side of the pair.
        self.assertIsNone(streams[0].scatter_src)
        self.assertIsNone(streams[0].gather_dst)
        self.assertIsNotNone(streams[1].scatter_src)
        self.assertIsNotNone(streams[1].gather_dst)
        self.assertEqual(streams[1].scatter_src.device.index, 0)
        self.assertEqual(streams[1].scatter_dst.device.index, 1)
        self.assertEqual(streams[1].gather_src.device.index, 1)
        self.assertEqual(streams[1].gather_dst.device.index, 0)
        for _ in range(3):
            scene.visible(self.inputs["origins"], self.inputs["end"])
        self.assertEqual(scene._multi._streams, streams)

    def test_a_two_device_chunked_gradient_matches_the_single_device_gradient(self):
        # Torch warns that the master leaf's AccumulateGrad node runs on a
        # different stream than the chunk that produced the gradient; that is
        # this executor by construction, and the engine inserts the ordering.
        vertices, faces = _grid_mesh(self.device)
        ray = self.inputs["ray"]

        def gradient(devices, **options):
            leaf = vertices.clone().requires_grad_(True)
            if devices is None:
                scene = rt.Scene()
            else:
                scene = rt.Scene(
                    devices=devices,
                    options=rt.MultiDeviceOptions(warm_up=False, **options),
                )
            scene.add_mesh(rt.Mesh(leaf, faces))
            scene.build()
            hit = scene.intersect(ray)
            torch.where(torch.isfinite(hit.t), hit.t, torch.zeros_like(hit.t)).sum().backward()
            return leaf.grad

        expected = gradient(None)
        self.assertGreater(float(expected.abs().max()), 0.0)
        chunked = gradient([0, 1], chunk_rays=7)
        torch.testing.assert_close(chunked, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
