"""Phase 2a of `docs/dev/multi_gpu_plan.md`: replicated multi-device `Scene`.

The contract under test is that `Scene(devices=[...])` is a composition layer
and nothing else: a scene that did not ask for several devices never reaches
the layer at all (D9), a one-device scene is the pre-existing path, and a
two-device scene answers every wrapped `per_ray` operation with the
field-for-field result the single-device scene produces (D1), with vertex
gradients reduced back onto the master leaf by autograd (D4).

The diffraction accumulation operations join it from Phase 2c as the first
`grid_reduce` members: they shard the Monte-Carlo lane space through the
`lane_offset` window of D5 instead of a batch axis, so their contract is the
weaker merge-layer one of D3/D6 -- the merged grid is the single-device grid up
to float32 summation order, the integer sample counters merge exactly, and a
fixed (devices, weights, chunk size) merges in a fixed order.
"""

import os
import pathlib
import subprocess
import sys
import unittest

import torch
import rayd.torch as rt


_TESTS_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

# 33 queries split ragged under every weighting used here (17/16 at 50/50,
# 29/4 at 90/10), so no test accidentally exercises only an even split.
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
    """One deterministic query batch, built on the host and moved to `device`.

    The origins are scattered over the mesh so that hits, misses and partially
    blocked segments all occur, which is what makes a bitwise comparison of a
    sharded result meaningful.
    """
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
    end = origins + torch.tensor([[0.0, 0.0, 2.0]], device=device)
    return {
        "ray": rt.Ray(origins, directions),
        "origins": origins,
        "directions": directions,
        "points": points,
        "active": active.to(device),
        "end": end.contiguous(),
        "end_b": (end + 4.0).contiguous(),
        "receiver": (origins + torch.tensor([[0.3, 0.3, 2.0]], device=device)).contiguous(),
        "edge_t_min": torch.zeros((count,), device=device),
        "edge_t_max": torch.ones((count,), device=device),
        "chain_points": torch.stack(
            (origins, origins + 0.5 * directions, origins + 2.0 * directions), dim=1
        ).contiguous(),
        "chain_length": torch.full((count,), 2, dtype=torch.int32, device=device),
    }


def _covered_op_results(scene: rt.Scene, inputs: dict) -> dict[str, torch.Tensor]:
    """Every `per_ray` operation the multi-device layer wraps, field by field."""
    ray = inputs["ray"]
    points = inputs["points"]
    active = inputs["active"]

    reduced = scene.intersect(ray, flags=getattr(rt.RayFlags, "None"))
    full = scene.intersect(ray, flags=rt.RayFlags.All)
    masked = scene.intersect(ray, active)
    nearest_point = scene.nearest_edge(points)
    nearest_ray = scene.nearest_edge(ray)
    topk = scene.nearest_edges(points, 3, active)
    visible = scene.visible(inputs["origins"], inputs["end"], active)
    pair = scene.visible_pair(inputs["origins"], inputs["end"], inputs["end_b"])
    axial = scene.visible_edge(
        inputs["origins"],
        points,
        inputs["directions"],
        inputs["edge_t_min"],
        inputs["edge_t_max"],
    )
    chain = scene.visible_chain(inputs["chain_points"], inputs["chain_length"])
    reflections = scene.trace_reflections(ray, max_bounces=2, active=active)
    epc = scene.trace_refl_epc_field(
        inputs["origins"], inputs["receiver"], max_bounces=2
    )

    results = {
        "intersect_reduced.t": reduced.t,
        "intersect_reduced.p": reduced.p,
        "intersect_masked.t": masked.t,
        "nearest_edge_point.distance": nearest_point.distance,
        "nearest_edge_point.edge_point": nearest_point.edge_point,
        "nearest_edge_point.edge_t": nearest_point.edge_t,
        "nearest_edge_point.shape_id": nearest_point.shape_id,
        "nearest_edge_point.edge_id": nearest_point.edge_id,
        "nearest_edge_point.global_edge_id": nearest_point.global_edge_id,
        "nearest_edge_ray.distance": nearest_ray.distance,
        "nearest_edge_ray.ray_t": nearest_ray.ray_t,
        "nearest_edge_ray.point": nearest_ray.point,
        "nearest_edge_ray.edge_t": nearest_ray.edge_t,
        "nearest_edge_ray.edge_point": nearest_ray.edge_point,
        "nearest_edge_ray.shape_id": nearest_ray.shape_id,
        "nearest_edge_ray.edge_id": nearest_ray.edge_id,
        "nearest_edge_ray.global_edge_id": nearest_ray.global_edge_id,
        "nearest_edges.is_valid": topk.is_valid,
        "nearest_edges.distances": topk.distances,
        "nearest_edges.points": topk.points,
        "nearest_edges.edge_t": topk.edge_t,
        "nearest_edges.edge_points": topk.edge_points,
        "nearest_edges.shape_ids": topk.shape_ids,
        "nearest_edges.edge_ids": topk.edge_ids,
        "nearest_edges.global_edge_ids": topk.global_edge_ids,
        "nearest_edges.is_boundary": topk.is_boundary,
        "visible": visible,
        "visible_pair.visible_a": pair.visible_a,
        "visible_pair.visible_b": pair.visible_b,
        "visible_edge.any_visible": axial.any_visible,
        "visible_chain.all_visible": chain.all_visible,
        "visible_chain.first_blocked_segment": chain.first_blocked_segment,
        "visible_chain.first_blocked_prim": chain.first_blocked_prim,
        "trace_reflections.valid": reflections.valid,
        "trace_reflections.t": reflections.t,
        "trace_reflections.prim_ids": reflections.prim_ids,
        "trace_reflections.image_sources": reflections.image_sources,
        "trace_refl_epc_field.field_real": epc.field_real,
        "trace_refl_epc_field.field_imag": epc.field_imag,
        "trace_refl_epc_field.path_length": epc.path_length,
        "trace_refl_epc_field.valid": epc.valid,
        "trace_refl_epc_field.resolved_prim_ids": epc.resolved_prim_ids,
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


def _covered_op_headers(scene: rt.Scene, inputs: dict) -> dict[str, int]:
    """The non-tensor header fields the sharded results have to rebuild."""
    topk = scene.nearest_edges(inputs["points"], 3)
    pair = scene.visible_pair(inputs["origins"], inputs["end"], inputs["end_b"])
    axial = scene.visible_edge(
        inputs["origins"],
        inputs["points"],
        inputs["directions"],
        inputs["edge_t_min"],
        inputs["edge_t_max"],
    )
    chain = scene.visible_chain(inputs["chain_points"], inputs["chain_length"])
    return {
        "nearest_edges.query_count": topk.query_count,
        "nearest_edges.k": topk.k,
        "visible_pair.ray_count": pair.ray_count,
        "visible_edge.state_count": axial.state_count,
        "visible_chain.chain_count": chain.chain_count,
        "visible_chain.max_segments": chain.max_segments,
    }


def _build_scene(device: torch.device, **kwargs) -> rt.Scene:
    vertices, faces = _grid_mesh(device)
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


def run_single_device_probe() -> None:
    """Exercise the whole single-device surface; used by the engagement probe.

    This is a module-level entry point on purpose: the subprocess that proves
    the multi-device layer is never imported runs exactly the operations a
    single-device caller runs, without duplicating them in a program string.
    """
    device = torch.device("cuda", 0)
    vertices, faces = _grid_mesh(device)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
    scene.build()
    inputs = _query_inputs(device)
    for value in _covered_op_results(scene, inputs).values():
        value.cpu()
    _covered_op_headers(scene, inputs)
    scene.update_mesh_vertices(0, vertices + 0.01)
    scene.sync()
    scene.set_edge_mask(scene.edge_mask())
    scene.global_geometry()
    scene.intersect(inputs["ray"]).t.cpu()


_ENGAGE_PROBE = """
import sys

sys.path.insert(0, {tests_root!r})

import torch
from torch_backend import test_multi_device_scene as harness

harness.run_single_device_probe()

assert harness.rt.Scene()._multi is None, "a default Scene planned a replica set"
loaded = sorted(name for name in sys.modules if name.startswith("rayd.torch._multi"))
assert not loaded, "single-device run imported {{}}".format(loaded)
print("MULTI-LAYER-UNTOUCHED")
"""


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Host copy compared bit-for-bit, so NaN, -0.0, and inf all compare exactly."""
    host = tensor.detach().contiguous().cpu()
    if host.dtype == torch.float32:
        return host.view(torch.int32)
    if host.dtype == torch.float64:
        return host.view(torch.int64)
    return host


class MultiDeviceResultMixin:
    def assert_same_results(self, left, right, context: str) -> None:
        self.assertEqual(sorted(left), sorted(right))
        for name, value in left.items():
            other = right[name]
            self.assertEqual(value.dtype, other.dtype, f"{context}: {name} dtype")
            self.assertEqual(value.shape, other.shape, f"{context}: {name} shape")
            self.assertEqual(
                value.device, other.device, f"{context}: {name} device"
            )
            self.assertTrue(
                torch.equal(_bits(value), _bits(other)),
                f"{context}: {name} is not bitwise equal",
            )


class MultiDeviceOptionsExportTests(unittest.TestCase):
    """`MultiDeviceOptions` is the whole public surface of the multi layer."""

    def test_the_options_dataclass_is_exported_from_the_package(self):
        from rayd.torch import MultiDeviceOptions
        from rayd.torch._multi import MultiDeviceOptions as private

        self.assertIs(MultiDeviceOptions, private)
        self.assertIs(rt.MultiDeviceOptions, private)
        self.assertIn("MultiDeviceOptions", rt.__all__)

    def test_unknown_package_attributes_still_raise(self):
        with self.assertRaises(AttributeError):
            rt.NoSuchPublicName


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class SingleDeviceStaysOnThePreExistingPathTests(
    MultiDeviceResultMixin, unittest.TestCase
):
    def test_default_scene_never_imports_the_multi_device_layer(self):
        """D9: the multi layer is dead code for a caller who never asks for it."""
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        package_root = str(
            pathlib.Path(__file__).resolve().parents[3] / "torch" / "python"
        )
        environment["PYTHONPATH"] = (
            package_root if not existing else os.pathsep.join([package_root, existing])
        )
        completed = subprocess.run(
            [sys.executable, "-c", _ENGAGE_PROBE.format(tests_root=_TESTS_ROOT)],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"probe failed:\n{completed.stdout}\n{completed.stderr}",
        )
        self.assertIn("MULTI-LAYER-UNTOUCHED", completed.stdout)

    def test_one_device_scene_matches_the_default_scene_bitwise(self):
        """A one-device `Scene(devices=[d])` is the pre-existing path, not a replica set."""
        device = torch.device("cuda", 0)
        inputs = _query_inputs(device)
        default = _build_scene(device)
        explicit = _build_scene(device, devices=[0])

        self.assertIsNone(explicit._multi)
        self.assert_same_results(
            _covered_op_results(default, inputs),
            _covered_op_results(explicit, inputs),
            "Scene(devices=[0])",
        )
        self.assertEqual(
            _covered_op_headers(default, inputs),
            _covered_op_headers(explicit, inputs),
        )

    def test_device_and_option_arguments_are_validated(self):
        with self.assertRaises(TypeError):
            rt.Scene(devices=0)
        with self.assertRaises(TypeError):
            rt.Scene(devices=[None])
        with self.assertRaises(ValueError):
            rt.Scene(devices=[])
        with self.assertRaises(ValueError):
            rt.Scene(devices=["cpu"])
        with self.assertRaises(ValueError):
            rt.Scene(devices=[0, 0])
        with self.assertRaises(ValueError):
            rt.Scene(devices=[torch.cuda.device_count()])
        with self.assertRaises(TypeError):
            rt.Scene(options=rt.MultiDeviceOptions())
        with self.assertRaises(TypeError):
            rt.Scene(devices=[0], options={"weights": [1.0]})
        with self.assertRaises(ValueError):
            rt.Scene(devices=[0], options=rt.MultiDeviceOptions(weights=[1.0, 1.0]))
        with self.assertRaises(ValueError):
            rt.Scene(devices=[0], options=rt.MultiDeviceOptions(weights=[0.0]))
        with self.assertRaises(ValueError):
            rt.Scene(devices=[0], options=rt.MultiDeviceOptions(weights=[-1.0]))

    def test_options_defaults_are_the_documented_ones(self):
        options = rt.MultiDeviceOptions()
        self.assertIsNone(options.weights)
        self.assertTrue(options.warm_up)

    def test_chunked_one_device_accumulation_matches_the_unchunked_scene(self):
        """The lane executor is a memory story on one device too (D7)."""
        device = torch.device("cuda", 0)
        reference = _accum_scene(device).accum_dfr_direct(
            states=_dfr_states(device),
            grid=_dfr_grid(),
            material=_dfr_material(device),
            wavelength=1.0,
            direct_samples=_ACCUM_SAMPLES,
            seed=17,
        )
        scene = _accum_scene(
            device,
            devices=[0],
            options=rt.MultiDeviceOptions(warm_up=False, chunk_rays=2048),
        )
        self.assertIsNotNone(scene._multi)
        merged = scene.accum_dfr_direct(
            states=_dfr_states(device),
            grid=_dfr_grid(),
            material=_dfr_material(device),
            wavelength=1.0,
            direct_samples=_ACCUM_SAMPLES,
            seed=17,
        )
        self.assertEqual(scene._multi.last_chunk_plan.chunk_count, _ACCUM_SAMPLES // 2048)
        self.assertGreater(float(reference.power.sum().item()), 0.0)
        torch.testing.assert_close(merged.power, reference.power, rtol=1e-4, atol=1e-9)
        self.assertTrue(torch.equal(merged.direct_count, reference.direct_count))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "two CUDA devices are required",
)
class MultiDeviceSceneTests(MultiDeviceResultMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        self.inputs = _query_inputs(self.device)
        self.single = _build_scene(self.device)
        self.reference = _covered_op_results(self.single, self.inputs)

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def _multi_scene(self, weights=None, warm_up: bool = False) -> rt.Scene:
        return _build_scene(
            self.device,
            devices=[0, 1],
            options=rt.MultiDeviceOptions(weights=weights, warm_up=warm_up),
        )

    def test_every_covered_op_matches_single_device_at_several_weightings(self):
        """The whole point of D1: a sharded `per_ray` result is the single-device result."""
        if torch.cuda.get_device_name(0) != torch.cuda.get_device_name(1):
            self.skipTest("bitwise cross-device equality needs identical devices")
        expected_split = {
            None: [(0, 16), (16, 33)],
            (0.5, 0.5): [(0, 16), (16, 33)],
            (9.0, 1.0): [(0, 29), (29, 33)],
            (1.0, 0.0): [(0, 33)],
        }
        for weights in (None, [0.5, 0.5], [9.0, 1.0], [1.0, 0.0]):
            with self.subTest(weights=weights):
                scene = self._multi_scene(weights)
                # Guard against a vacuous comparison: the split really is the
                # weighted one, and the degenerate weighting really does leave
                # the second device idle.
                self.assertIsNotNone(scene._multi)
                self.assertEqual(
                    [(start, stop) for _replica, _device, start, stop in scene._multi._shards(_BATCH)],
                    expected_split[None if weights is None else tuple(weights)],
                )
                self.assert_same_results(
                    self.reference,
                    _covered_op_results(scene, self.inputs),
                    f"weights={weights}",
                )

    def test_result_headers_and_types_survive_the_gather(self):
        scene = self._multi_scene([9.0, 1.0])
        self.assertEqual(
            _covered_op_headers(self.single, self.inputs),
            _covered_op_headers(scene, self.inputs),
        )
        reduced = scene.intersect(self.inputs["ray"], flags=getattr(rt.RayFlags, "None"))
        self.assertEqual(type(reduced), type(self.single.intersect(
            self.inputs["ray"], flags=getattr(rt.RayFlags, "None")
        )))
        self.assertEqual(tuple(reduced.p.shape), (0, 3))
        self.assertEqual(
            type(scene.trace_reflections(self.inputs["ray"], max_bounces=2)),
            type(self.single.trace_reflections(self.inputs["ray"], max_bounces=2)),
        )
        self.assertEqual(scene.num_meshes, self.single.num_meshes)
        self.assertTrue(scene.is_ready())

    def test_default_options_warm_up_and_still_match(self):
        """The default path (`Scene(devices=[...])`, warm-up on) is covered too."""
        if torch.cuda.get_device_name(0) != torch.cuda.get_device_name(1):
            self.skipTest("bitwise cross-device equality needs identical devices")
        scene = _build_scene(self.device, devices=[0, 1])
        self.assert_same_results(
            self.reference,
            _covered_op_results(scene, self.inputs),
            "default options",
        )

    def test_master_vertex_gradient_matches_single_device(self):
        """D4: every replica's gradient is reduced onto the master leaf by autograd."""
        vertices, faces = _grid_mesh(self.device)
        weight = (torch.arange(_BATCH, device=self.device, dtype=torch.float32) + 1.0) / _BATCH

        def gradient(devices):
            leaf = vertices.clone().requires_grad_(True)
            if devices is None:
                scene = rt.Scene()
            else:
                scene = rt.Scene(
                    devices=devices, options=rt.MultiDeviceOptions(warm_up=False)
                )
            scene.add_mesh(rt.Mesh(leaf, faces))
            scene.build()
            hit = scene.intersect(self.inputs["ray"])
            t = torch.where(torch.isfinite(hit.t), hit.t, torch.zeros_like(hit.t))
            chain = scene.trace_reflections(self.inputs["ray"], max_bounces=2)
            bounces = torch.where(chain.valid, chain.t, torch.zeros_like(chain.t))
            ((t * weight).sum() + bounces.sum()).backward()
            return leaf.grad

        single = gradient(None)
        multi = gradient([0, 1])
        self.assertIsNotNone(multi)
        self.assertGreater(float(single.abs().max()), 0.0)
        # The shards accumulate their vertex gradients through per-device
        # atomics and are then summed on the master, so the summation order
        # differs from the single-launch order. Only float32 rounding may
        # differ: measured max deviation is ~5e-7 on gradients of order 1.
        torch.testing.assert_close(multi, single, rtol=1e-5, atol=1e-6)

    def test_broadcast_update_and_sync_match_a_freshly_built_scene(self):
        vertices, faces = _grid_mesh(self.device)
        moved = (vertices + torch.tensor([[0.05, -0.02, 0.15]], device=self.device)).contiguous()

        scene = rt.Scene(devices=[0, 1], options=rt.MultiDeviceOptions(warm_up=False))
        scene.add_mesh(rt.Mesh(vertices.clone(), faces), dynamic=True)
        scene.build()
        scene.update_mesh_vertices(0, moved)
        self.assertTrue(scene.has_pending_updates())
        scene.sync()
        self.assertFalse(scene.has_pending_updates())

        fresh = rt.Scene()
        fresh.add_mesh(rt.Mesh(moved.clone(), faces))
        fresh.build()

        self.assert_same_results(
            _covered_op_results(fresh, self.inputs),
            _covered_op_results(scene, self.inputs),
            "after update_mesh_vertices + sync",
        )
        # Every replica really took the update: the version counters stay in
        # lockstep, and the replica on the second device answers with the
        # updated geometry (it owns the second shard of every query above).
        versions = [replica.version for replica in scene._multi._replicas]
        self.assertEqual(len(set(versions)), 1)

    def test_edge_mask_is_broadcast_to_every_replica(self):
        scene = self._multi_scene()
        mask = scene.edge_mask().clone()
        self.assertGreater(mask.numel(), 0)
        mask[: mask.numel() // 2] = False
        scene.set_edge_mask(mask)

        single_mask = self.single.edge_mask().clone()
        single_mask[: single_mask.numel() // 2] = False
        self.single.set_edge_mask(single_mask)

        self.assert_same_results(
            _covered_op_results(self.single, self.inputs),
            _covered_op_results(scene, self.inputs),
            "after set_edge_mask",
        )
        for replica, device in zip(scene._multi._replicas, scene._multi.devices):
            self.assertTrue(
                torch.equal(_bits(replica.edge_mask()), _bits(mask)),
                f"edge mask did not reach {device}",
            )

    def test_operations_without_multi_device_semantics_raise(self):
        """What is left after Phase 2c wired the lane-windowed accumulation ops.

        `trace_dfr_paths` places exporter rows by batch position, and
        `accum_dfr_coherent_direct` has no lane window at all, so both still
        need the per-shard contract of D6.
        """
        scene = self._multi_scene()
        states = _dfr_states(self.device)
        grid = rt.DfrGrid(axis=2, position=0.5, resolution0=2, resolution1=2)
        active = torch.ones(states.state_count, dtype=torch.bool, device=self.device)

        cases = {
            "trace_dfr_paths": lambda: scene.trace_dfr_paths(
                tx_positions=self.inputs["origins"],
                rx_positions=self.inputs["receiver"],
                states=states,
                active=active,
            ),
            "accum_dfr_coherent_direct": lambda: scene.accum_dfr_coherent_direct(
                states=states, grid=grid
            ),
        }
        for name, call in cases.items():
            with self.subTest(operation=name):
                with self.assertRaises(NotImplementedError) as raised:
                    call()
                message = str(raised.exception)
                self.assertIn(name, message)
                self.assertIn("docs/dev/multi_gpu_plan.md Phase 2c", message)

    def test_mesh_tensors_must_live_on_the_master_device(self):
        vertices, faces = _grid_mesh(torch.device("cuda", 1))
        scene = rt.Scene(devices=[0, 1], options=rt.MultiDeviceOptions(warm_up=False))
        scene.add_mesh(rt.Mesh(vertices, faces))
        with self.assertRaises(ValueError) as raised:
            scene.build()
        self.assertIn("master device", str(raised.exception))

    def test_querying_before_build_fails_loudly(self):
        scene = rt.Scene(devices=[0, 1], options=rt.MultiDeviceOptions(warm_up=False))
        scene.add_mesh(rt.Mesh(*_grid_mesh(self.device)))
        with self.assertRaises(RuntimeError):
            scene.intersect(self.inputs["ray"])

    def test_empty_batches_take_the_master_replica(self):
        scene = self._multi_scene()
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        ray = rt.Ray(empty, empty)
        self.assertEqual(tuple(scene.intersect(ray).t.shape), (0,))
        self.assertEqual(tuple(scene.visible(empty, empty).shape), (0,))
        self.assertEqual(
            tuple(scene.trace_reflections(ray, max_bounces=2).valid.shape), (0, 2)
        )


def _dfr_states(device: torch.device, requires_grad: bool = False) -> rt.DfrStates:
    """Caller-owned order-1 diffraction states over the accumulation fixture."""

    def leaf(values):
        return torch.tensor(
            values, dtype=torch.float32, device=device, requires_grad=requires_grad
        )

    def f32(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return rt.DfrStates(
        edge_index=i32([0, 1]),
        edge_pos=leaf([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        edge_dir=leaf([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_t_min=leaf([-1.0, -1.0]),
        edge_t_max=leaf([1.0, 1.0]),
        n0=f32([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        n1=f32([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        prim0=i32([0, 0]),
        prim1=i32([0, 0]),
        exterior_angle=leaf([torch.pi, torch.pi]),
        src=leaf([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25]]),
        src_power=leaf([1.0, 1.0]),
    )


def _dfr_material(device: torch.device, requires_grad: bool = False) -> rt.DfrMaterial:
    """A one-face material; only `gain` carries an accumulation gradient."""
    return rt.DfrMaterial(
        eta_r=torch.ones((1,), device=device),
        sigma=torch.zeros((1,), device=device),
        mu_r=torch.ones((1,), device=device),
        gain=torch.ones((1,), device=device, requires_grad=requires_grad),
        valid=torch.ones((1,), device=device, dtype=torch.bool),
    )


def _dfr_grid() -> rt.DfrGrid:
    return rt.DfrGrid(axis=2, position=0.0, resolution0=4, resolution1=4)


def _accum_scene(device: torch.device, **kwargs) -> rt.Scene:
    """One triangle in the `z = 0` plane, the fixture `test_lane_offset` uses.

    The states above diffract over its edges into the `z = 0` grid, so every
    accumulation below lands real power in real cells; each test asserts that
    explicitly rather than trusting the fixture.
    """
    vertices = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


def _chain_fixture(device: torch.device, **kwargs):
    """Order-2 chain fixture: one initial and one recursive diffraction state."""
    vertices = torch.tensor(
        [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()

    def f32(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def states(index, edge_pos, wi):
        return rt.DfrStates(
            edge_index=torch.tensor([index], dtype=torch.int32, device=device),
            edge_pos=f32(edge_pos),
            edge_dir=f32([[1.0, 0.0, 0.0]]),
            edge_t_min=f32([-0.5]),
            edge_t_max=f32([0.5]),
            n0=f32([[0.0, 1.0, 0.0]]),
            n1=f32([[0.0, -1.0, 0.0]]),
            prim0=torch.tensor([-1], dtype=torch.int32, device=device),
            prim1=torch.tensor([-1], dtype=torch.int32, device=device),
            exterior_angle=f32([1.5 * torch.pi]),
            src=f32([[0.0, 0.0, 1.0]]),
            src_power=f32([2.0]),
            wi=f32(wi),
            d0=f32([[0.0, 0.0, -1.0]]),
            count=1,
        )

    initial = states(0, [[0.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]])
    recursive = states(1, [[0.0, 0.5, 0.0]], [[0.0, 1.0, 0.0]])
    return scene, initial, recursive


# The lane space of every accumulation comparison below. 8192 lanes is over the
# 2048-lane threshold at which a no-AD launch takes the staged sort/reduce route
# instead of plain atomics -- on both a whole device's shard and every chunk cut
# here -- which is what makes those launches, and therefore the merge, bitwise
# reproducible. The atomics route is reproducible only to the last ULP, on one
# device as much as on two, so it is compared with a tolerance instead.
_ACCUM_SAMPLES = 8192


class AccumulationMixin:
    def assert_accum_close(self, merged, reference, context: str) -> None:
        """A merged grid is the single-launch grid up to float32 summation order."""
        self.assertGreater(
            float(reference.power.sum().item()), 0.0, f"{context}: vacuous fixture"
        )
        self.assertEqual(merged.grid_cell_count, reference.grid_cell_count)
        self.assertEqual(merged.power.device, reference.power.device)
        for name in ("power", "field_x_re", "field_x_im", "field_y_re", "field_z_re"):
            torch.testing.assert_close(
                getattr(merged, name),
                getattr(reference, name),
                rtol=1e-4,
                atol=1e-9,
                msg=f"{context}: {name} mismatch",
            )
        # The counters are integers, so the split has to be exact, not close:
        # every sample the single launch drew is drawn once by some shard.
        for name in ("direct_count", "keller_count", "suffix_count"):
            self.assertTrue(
                torch.equal(getattr(merged, name), getattr(reference, name)),
                f"{context}: {name} is not the single-launch count",
            )

    def assert_accum_identical(self, left, right, context: str) -> None:
        # Imported here, not at module scope: this module is also the body of
        # the subprocess probe that proves a single-device run never loads the
        # multi-device layer at all.
        from rayd.torch._multi import _DFR_ACCUM_FIELDS

        for name in _DFR_ACCUM_FIELDS:
            self.assertTrue(
                torch.equal(_bits(getattr(left, name)), _bits(getattr(right, name))),
                f"{context}: {name} is not bitwise reproducible",
            )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "two CUDA devices are required",
)
class MultiDeviceAccumulationTests(AccumulationMixin, unittest.TestCase):
    """Phase 2c: `grid_reduce` accumulation sharded over the lane space."""

    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        self.grid = _dfr_grid()
        self.single = _accum_scene(self.device)

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def _scene(self, weights=None, **options) -> rt.Scene:
        return _accum_scene(
            self.device,
            devices=[0, 1],
            options=rt.MultiDeviceOptions(weights=weights, warm_up=False, **options),
        )

    def _accum(
        self,
        scene,
        *,
        samples: int = _ACCUM_SAMPLES,
        states=None,
        material=None,
        **kwargs,
    ):
        return scene.accum_dfr_direct(
            states=_dfr_states(self.device) if states is None else states,
            grid=self.grid,
            material=_dfr_material(self.device) if material is None else material,
            wavelength=1.0,
            direct_samples=samples,
            seed=17,
            **kwargs,
        )

    def test_two_device_accum_matches_single_device_at_several_weightings(self):
        reference = self._accum(self.single)
        for weights in (None, [3.0, 1.0], [1.0, 0.0], [0.0, 1.0]):
            with self.subTest(weights=weights):
                scene = self._scene(weights)
                self.assert_accum_close(
                    self._accum(scene), reference, f"weights={weights}"
                )

    def test_the_lane_windows_partition_the_caller_s_window(self):
        """D5: the shards are a partition of the lane space, warp by warp.

        The `(tape_state_idx, tape_cell, tape_edge_u)` multiset comparison the
        plan asks for is not reachable from here: the AD tape is internal to the
        accumulation autograd function and never leaves it, so the native-level
        multiset test lives in `test_lane_offset` instead. What the public API
        does expose is the integer sample counters, which are checked to be the
        single-launch counters exactly by `assert_accum_close`, plus the windows
        themselves.
        """
        for weights in (None, [3.0, 1.0], [1.0, 1.0, 0.0][:2]):
            with self.subTest(weights=weights):
                layer = self._scene(weights)._multi
                windows = [
                    (begin, count)
                    for _replica, _device, begin, count in layer._lane_shards(
                        0, _ACCUM_SAMPLES
                    )
                ]
                self.assertTrue(windows)
                covered = 0
                for begin, count in windows:
                    self.assertEqual(begin, covered, "windows are not contiguous")
                    self.assertGreater(count, 0, "an empty window was launched")
                    covered += count
                self.assertEqual(covered, _ACCUM_SAMPLES)
                for begin, _count in windows[1:]:
                    self.assertEqual(begin % 32, 0, "a window boundary splits a warp")

    def test_a_caller_lane_window_is_sharded_inside_itself(self):
        """A caller that shards by hand may hand the layer a sub-window."""
        scene = self._scene()
        whole = self._accum(scene)
        halves = [
            self._accum(scene, lane_offset=offset, lane_count=_ACCUM_SAMPLES // 2)
            for offset in (0, _ACCUM_SAMPLES // 2)
        ]
        merged = halves[0].power + halves[1].power
        torch.testing.assert_close(merged, whole.power, rtol=1e-4, atol=1e-9)
        self.assertEqual(
            int(halves[0].direct_count.item()) + int(halves[1].direct_count.item()),
            int(whole.direct_count.item()),
        )
        with self.assertRaises(RuntimeError):
            self._accum(scene, lane_offset=_ACCUM_SAMPLES + 1)
        with self.assertRaises(RuntimeError):
            self._accum(scene, lane_offset=1, lane_count=_ACCUM_SAMPLES)

    def test_a_fixed_split_merges_in_a_fixed_order(self):
        """D3/D6: the merge order is pinned, so the run reproduces itself."""
        scene = self._scene([3.0, 1.0])
        first = self._accum(scene)
        second = self._accum(scene)
        self.assert_accum_identical(first, second, "weights=[3.0, 1.0]")

    def test_chunked_two_device_accum_matches_and_reproduces(self):
        """D7 on the lane axis: chunks fold into their device's partial grid."""
        reference = self._accum(self.single)
        # 2048 is also the staged route's threshold, so every chunk here is
        # individually reproducible and the bitwise comparison below tests the
        # merge order rather than the kernel's atomics.
        scene = self._scene(chunk_rays=2048)
        merged = self._accum(scene)
        plan = scene._multi.last_chunk_plan
        self.assertEqual(plan.operation, "accum_dfr_direct")
        self.assertEqual(plan.chunk_rays, 2048)
        self.assertEqual(plan.chunk_count, _ACCUM_SAMPLES // 2048)
        self.assert_accum_close(merged, reference, "chunk_rays=2048")
        self.assert_accum_identical(merged, self._accum(scene), "chunk_rays=2048")

    def test_a_chunk_size_is_rounded_up_to_a_whole_warp(self):
        scene = self._scene(chunk_rays=100)
        self._accum(scene, samples=1024)
        plan = scene._multi.last_chunk_plan
        self.assertEqual(plan.chunk_rays, 128)
        self.assertEqual(plan.chunk_count, 1024 // 128)

    def test_a_window_narrower_than_a_warp_runs_on_one_device(self):
        """Sub-warp windows are not split; the leading devices simply idle."""
        scene = self._scene()
        windows = scene._multi._lane_shards(0, 16)
        self.assertEqual([(begin, count) for _r, _d, begin, count in windows], [(0, 16)])
        self.assert_accum_close(
            self._accum(scene, samples=16),
            self._accum(self.single, samples=16),
            "samples=16",
        )

    def test_an_empty_lane_space_runs_once_on_the_master(self):
        scene = self._scene()
        merged = self._accum(scene, samples=0)
        reference = self._accum(self.single, samples=0)
        self.assertEqual(merged.power.device, self.device)
        self.assertTrue(torch.equal(_bits(merged.power), _bits(reference.power)))
        self.assertEqual(int(merged.direct_count.item()), 0)
        self.assertEqual(scene._multi.last_chunk_plan.chunk_count, 1)

    def test_order_two_chain_accumulation_matches_single_device(self):
        single, initial, recursive = _chain_fixture(self.device)
        multi, _initial, _recursive = _chain_fixture(
            self.device, devices=[0, 1], options=rt.MultiDeviceOptions(warm_up=False)
        )

        def accum(scene):
            return scene.accum_dfr(
                initial_states=initial,
                recursive_states=recursive,
                grid=rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2),
                material=_dfr_material(self.device),
                wavelength=0.125,
                seed=17,
                direct_samples=2048,
                keller_samples=2048,
                max_order=2,
            )

        reference = accum(single)
        merged = accum(multi)
        self.assertEqual(
            [(begin, count) for _r, _d, begin, count in multi._multi._lane_shards(0, 4096)],
            [(0, 2048), (2048, 2048)],
        )
        # The chain path reduces through plain atomics on one device as much as
        # on two, so this is the tolerance comparison of D3, never a bitwise one.
        self.assert_accum_close(merged, reference, "accum_dfr")

    def test_backward_reaches_the_master_state_and_material_gradients(self):
        """D4 on the lane axis: every shard's backward lands on the caller's leaves."""

        def gradients(scene):
            states = _dfr_states(self.device, requires_grad=True)
            material = _dfr_material(self.device, requires_grad=True)
            # The AD route writes a tape and reduces through atomics, so this
            # runs at the smaller sample count the tape budget likes.
            merged = self._accum(
                scene, samples=1024, states=states, material=material
            )
            merged.power.sum().backward()
            return {
                "edge_pos": states.edge_pos.grad,
                "edge_dir": states.edge_dir.grad,
                "exterior_angle": states.exterior_angle.grad,
                "src": states.src.grad,
                "src_power": states.src_power.grad,
                "gain": material.gain.grad,
            }

        expected = gradients(self.single)
        actual = gradients(self._scene([3.0, 1.0]))
        for name, value in expected.items():
            with self.subTest(gradient=name):
                self.assertIsNotNone(actual[name], "the shard gradients never arrived")
                self.assertEqual(actual[name].device, self.device)
                self.assertGreater(
                    float(value.abs().sum().item()), 0.0, "vacuous gradient"
                )
                torch.testing.assert_close(
                    actual[name], value, rtol=1e-4, atol=1e-9, msg=f"{name} mismatch"
                )


if __name__ == "__main__":
    unittest.main()
