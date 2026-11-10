# Copyright Xingyu Chen.
# Compares rayd.torch and rayd.drjit geometry queries on identical inputs and pins their known divergences.

"""Cross-backend parity for geometry queries.

Every case builds the same host-side scene (`tests.support.backend_scenes`) in both backends and feeds both
the same rays, points and masks. Ids are compared exactly wherever a tie is impossible; floats use the
contract tolerances from `contracts/operations.json`. Inputs keep every hit distance far above both
backends' `ray_tmin` so the frozen tmin divergence cannot leak into the comparisons; dedicated tests pin
each divergence instead. Set `RAYD_PARITY_REPORT=1` to print the largest deviation seen per operation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from tests.support import backend_scenes as bs
from tests.support.backends import both_backends_available, drjit_modules


CONTRACT = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "operations.json").read_text())
TOL = CONTRACT["tolerances"]
ABS, REL = TOL["default_abs"], TOL["default_rel"]
PROFILES = CONTRACT["numeric_policy"]["backend_profiles"]
RECORDER = bs.ParityRecorder()

# Torch and Dr.Jit edge backends that answer the same nearest-edge queries; "auto" resolves differently.
EDGE_BACKEND_PAIRS = (("cuda", "drjit"), ("optix", "optix"), ("cuda", "optix_drjit"))
INTERSECT_FIELDS_CLOSE = ("t", "p", "geo_n", "barycentric")
INTERSECT_FIELDS_EQUAL = ("shape_id", "local_prim_id", "global_prim_id")


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = end - start
    t = np.clip(
        np.einsum("ij,ij->i", point - start, axis) / np.maximum(np.einsum("ij,ij->i", axis, axis), 1e-30), 0.0, 1.0
    )
    return np.linalg.norm(point - (start + t[:, None] * axis), axis=1)


def _untied_slots(distances_k_plus_1, k: int) -> np.ndarray:
    """Flattened (query, rank) slots of a top-k result whose distance is separated from both neighbours.

    The Torch top-(k+1) distances also expose a tie between rank k-1 and the first excluded edge.
    """
    distances = bs.host(distances_k_plus_1).astype(np.float64)
    gap = 1.0e-5
    lower = np.concatenate((np.full((distances.shape[0], 1), -np.inf), distances[:, :k]), axis=1)
    below = np.abs(distances[:, :k] - lower[:, :k]) > gap
    above = ~(np.abs(distances[:, 1 : k + 1] - distances[:, :k]) <= gap)
    return (below & above & np.isfinite(distances[:, :k])).reshape(-1)


def tearDownModule() -> None:
    if os.environ.get("RAYD_PARITY_REPORT") == "1":
        print("\nmax |Torch - Dr.Jit| per operation (geometry):\n" + RECORDER.report(), file=sys.stderr)


def _torch():
    import torch

    import rayd.torch as rt

    return torch, rt


class _ParityBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mods = drjit_modules()
        cls.torch, cls.rt = _torch()
        cls._scene_cache = {}

    def scenes(
        self, spec: bs.SceneSpec, trace_backend: str = "optix", torch_edge: str = "cuda", drjit_edge: str = "drjit"
    ):
        key = (spec.name, trace_backend, torch_edge, drjit_edge)
        if key not in self._scene_cache:
            self._scene_cache[key] = (
                bs.torch_scene(self.rt, spec, trace_backend=trace_backend, edge_bvh_backend=torch_edge),
                bs.drjit_scene(self.mods, spec, trace_backend=trace_backend, edge_bvh_backend=drjit_edge),
            )
        return self._scene_cache[key]

    def bool_t(self, values):
        return self.torch.tensor(list(values), device="cuda", dtype=self.torch.bool)

    def int_t(self, values):
        return self.torch.tensor(list(values), device="cuda", dtype=self.torch.int32)

    def float_t(self, values):
        return self.torch.tensor(list(values), device="cuda", dtype=self.torch.float32)

    def close(self, op, name, a, b, atol=ABS, rtol=REL, rows=None):
        RECORDER.close(self, op, name, a, b, atol, rtol, rows)

    def equal(self, op, name, a, b, rows=None):
        RECORDER.equal(self, op, name, a, b, rows)

    def compare_edge_winners(self, op, near_t, near_d, scene_d):
        """Compare nearest-edge ids exactly, except where the two winners are a genuine geometric tie.

        A query closest to a vertex ties every edge incident on it, and a non-manifold or collinear-degenerate
        edge is stored once per incident face pair, so equally near edges can win in either backend. A differing
        id is accepted only when the shared closest point lies on both winning segments (scene-global edge
        numbering is identical in both backends; Dr.Jit `edge_info()` supplies the segments).
        """
        ids_t = bs.host(near_t.global_edge_id).reshape(-1)
        ids_d = bs.host(near_d.global_edge_id).reshape(-1)
        same = ids_t == ids_d
        info = scene_d.edge_info()
        start, end = bs.host(info.start), bs.host(info.end)
        differ = np.flatnonzero(~same)
        if differ.size:
            self.assertTrue(np.all((ids_t[differ] >= 0) & (ids_d[differ] >= 0)), f"{op}: valid/invalid winner")
            point = bs.host(near_t.edge_point).reshape(-1, 3)[differ].astype(np.float64)
            for ids in (ids_t[differ], ids_d[differ]):
                gap = _point_segment_distance(point, start[ids].astype(np.float64), end[ids].astype(np.float64))
                self.assertLess(
                    float(gap.max()),
                    1.0e-5,
                    f"{op}: differing winners {ids_t[differ][:4]} vs {ids_d[differ][:4]} are not a geometric tie",
                )
        self.assertGreater(int(same.sum()), same.size // 2, f"{op}: too few untied rows to compare ids")
        self.close(op, "edge_t", near_t.edge_t, near_d.edge_t, rows=same)
        for name in ("shape_id", "edge_id", "global_edge_id"):
            self.equal(op, name, getattr(near_t, name), getattr(near_d, name), same)


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class IntersectParityTests(_ParityBase):
    def _compare_intersections(self, op, hit_t, hit_d):
        for name in INTERSECT_FIELDS_CLOSE:
            self.close(op, name, getattr(hit_t, name), getattr(hit_d, name))
        for name in INTERSECT_FIELDS_EQUAL:
            self.equal(op, name, getattr(hit_t, name), getattr(hit_d, name))
        self.equal(op, "valid", bs.host(hit_t.shape_id) >= 0, hit_d.is_valid())

    def test_intersect_hits_and_misses_on_every_scene(self):
        for trace_backend in ("optix", "cuda"):
            for spec in bs.all_scenes():
                with self.subTest(scene=spec.name, trace_backend=trace_backend):
                    scene_t, scene_d = self.scenes(spec, trace_backend)
                    origins, directions = bs.scene_rays(spec, 256, seed=1)
                    hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
                    hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
                    valid = bs.host(hit_t.shape_id) >= 0
                    self.assertGreater(int(valid.sum()), 16, "the ray set must contain hits")
                    self.assertGreater(int((~valid).sum()), 16, "the ray set must contain misses")
                    self._compare_intersections("intersect", hit_t, hit_d)

    def test_intersect_finite_tmax_cuts_hits_on_both_sides(self):
        for spec in (bs.closed_box(), bs.overlapping_meshes(), bs.non_manifold_fan()):
            with self.subTest(scene=spec.name):
                scene_t, scene_d = self.scenes(spec)
                origins, directions = bs.scene_rays(spec, 256, seed=2)
                first = bs.host(scene_t.intersect(bs.torch_ray(self.rt, origins, directions)).t)
                # Half of the hit rows stop at 60% of their hit distance and the others at 140%, so no tmax lies
                # within float32 rounding of a hit; misses keep an infinite extent.
                scale = np.where(np.arange(first.size) % 2 == 0, 0.6, 1.4)
                tmax = np.where(np.isfinite(first), first * scale, np.inf).tolist()
                hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions, tmax))
                hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions, tmax))
                cut = np.isfinite(first) & (scale < 1.0)
                self.assertTrue(np.all(~np.isfinite(bs.host(hit_t.t)[cut])), "tmax must cut the hits it precedes")
                self._compare_intersections("intersect[tmax]", hit_t, hit_d)

    def test_intersect_active_mask_disables_lanes_identically(self):
        for spec in (bs.grid_sheet(), bs.overlapping_meshes()):
            with self.subTest(scene=spec.name):
                scene_t, scene_d = self.scenes(spec)
                origins, directions = bs.scene_rays(spec, 128, seed=3)
                active = [index % 3 != 1 for index in range(128)]
                hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions), self.bool_t(active))
                hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions), self.mods.cuda.Bool(active))
                self.assertTrue(np.all(~np.isfinite(bs.host(hit_t.t)[~np.asarray(active)])))
                self._compare_intersections("intersect[active]", hit_t, hit_d)

    def test_intersect_ignores_degenerate_faces(self):
        # The degenerate scene is a unit quad plus zero-area faces; every hit must land on faces 0 or 1.
        spec = bs.degenerate_faces()
        scene_t, scene_d = self.scenes(spec)
        origins, directions = bs.scene_rays(spec, 256, seed=4)
        for hit in (
            scene_t.intersect(bs.torch_ray(self.rt, origins, directions)),
            scene_d.intersect(bs.drjit_ray(self.mods, origins, directions)),
        ):
            ids = bs.host(hit.global_prim_id)
            self.assertTrue(set(ids[ids >= 0].tolist()) <= {0, 1}, sorted(set(ids.tolist())))


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class EdgeQueryParityTests(_ParityBase):
    def test_nearest_edge_point_on_every_scene_and_edge_backend(self):
        for torch_edge, drjit_edge in EDGE_BACKEND_PAIRS:
            for spec in bs.all_scenes():
                with self.subTest(scene=spec.name, torch_edge=torch_edge, drjit_edge=drjit_edge):
                    scene_t, scene_d = self.scenes(spec, torch_edge=torch_edge, drjit_edge=drjit_edge)
                    points = bs.scene_points(spec, 128, seed=5)
                    near_t = scene_t.nearest_edge(bs.torch_rows(points))
                    near_d = scene_d.nearest_edge(bs.drjit_vec3(self.mods, points))
                    for name in ("distance", "edge_point"):
                        self.close("nearest_edge[point]", name, getattr(near_t, name), getattr(near_d, name))
                    self.compare_edge_winners("nearest_edge[point]", near_t, near_d, scene_d)

    def test_nearest_edge_ray_finite_and_infinite_tmax(self):
        for torch_edge, drjit_edge in EDGE_BACKEND_PAIRS:
            for spec in bs.all_scenes():
                for finite in (False, True):
                    with self.subTest(scene=spec.name, torch_edge=torch_edge, finite_tmax=finite):
                        scene_t, scene_d = self.scenes(spec, torch_edge=torch_edge, drjit_edge=drjit_edge)
                        origins = bs.scene_points(spec, 128, seed=6)
                        directions = bs.unit_vectors(128, seed=7)
                        tmax = [0.7] * 128 if finite else None
                        near_t = scene_t.nearest_edge(bs.torch_ray(self.rt, origins, directions, tmax))
                        near_d = scene_d.nearest_edge(bs.drjit_ray(self.mods, origins, directions, tmax))
                        op = "nearest_edge[ray,finite]" if finite else "nearest_edge[ray,inf]"
                        for name in ("distance", "ray_t", "point", "edge_point"):
                            self.close(op, name, getattr(near_t, name), getattr(near_d, name))
                        self.compare_edge_winners(op, near_t, near_d, scene_d)
                        if finite:
                            self.assertTrue(np.all(bs.host(near_t.ray_t) <= 0.7 + 1e-6))

    def test_nearest_edges_topk_without_ties(self):
        for torch_edge, drjit_edge in EDGE_BACKEND_PAIRS:
            for spec in bs.all_scenes():
                for k in (1, 3, 5):
                    with self.subTest(scene=spec.name, torch_edge=torch_edge, k=k):
                        scene_t, scene_d = self.scenes(spec, torch_edge=torch_edge, drjit_edge=drjit_edge)
                        points = bs.scene_points(spec, 64, seed=8)
                        active = [index % 5 != 2 for index in range(64)]
                        top_t = scene_t.nearest_edges(bs.torch_rows(points), k, self.bool_t(active))
                        top_d = scene_d.nearest_edges(bs.drjit_vec3(self.mods, points), k, self.mods.cuda.Bool(active))
                        self.assertEqual((top_t.query_count, top_t.k), (int(top_d.query_count), int(top_d.k)))
                        self.equal("nearest_edges", "is_valid", top_t.is_valid, top_d.is_valid)
                        self.close("nearest_edges", "distances", top_t.distances, top_d.distances)
                        wider = scene_t.nearest_edges(bs.torch_rows(points), k + 1, self.bool_t(active))
                        slots = _untied_slots(wider.distances, k)
                        self.assertGreater(int(slots.sum()), 16)
                        for name in ("edge_t", "edge_points"):
                            self.close("nearest_edges", name, getattr(top_t, name), getattr(top_d, name), rows=slots)
                        for name in ("global_edge_ids", "edge_ids", "shape_ids", "is_boundary"):
                            self.equal("nearest_edges", name, getattr(top_t, name), getattr(top_d, name), slots)
                        inactive = ~np.asarray(active)
                        self.assertFalse(bs.host(top_t.is_valid).reshape(64, k)[inactive].any())

    def test_edge_mask_filters_the_same_edges(self):
        for spec in (bs.closed_box(), bs.grid_sheet(), bs.non_manifold_fan()):
            with self.subTest(scene=spec.name):
                scene_t = bs.torch_scene(self.rt, spec)
                scene_d = bs.drjit_scene(self.mods, spec)
                self.equal("edge_mask", "default", scene_t.edge_mask(), scene_d.edge_mask())
                count = int(bs.host(scene_t.edge_mask()).size)
                mask = [index % 3 != 0 for index in range(count)]
                scene_t.set_edge_mask(self.bool_t(mask))
                scene_d.set_edge_mask(self.mods.cuda.Bool(mask))
                scene_d.sync()  # Dr.Jit commits an edge mask on sync(); see test_edge_mask_commit_point_differs.
                self.equal("edge_mask", "readback", scene_t.edge_mask(), scene_d.edge_mask())
                points = bs.scene_points(spec, 96, seed=9)
                near_t = scene_t.nearest_edge(bs.torch_rows(points))
                near_d = scene_d.nearest_edge(bs.drjit_vec3(self.mods, points))
                ids = bs.host(near_t.global_edge_id)
                self.assertTrue(all(mask[index] for index in ids[ids >= 0]), "a masked edge was returned")
                self.close("edge_mask", "distance", near_t.distance, near_d.distance)
                self.equal("edge_mask", "global_edge_id", near_t.global_edge_id, near_d.global_edge_id)
                top_t = scene_t.nearest_edges(bs.torch_rows(points), 2)
                top_d = scene_d.nearest_edges(bs.drjit_vec3(self.mods, points), 2)
                self.equal("edge_mask", "topk_ids", top_t.global_edge_ids, top_d.global_edge_ids)

    def test_edge_mask_of_wrong_length_is_rejected_by_both(self):
        spec = bs.single_triangle()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        with self.assertRaises(Exception):
            scene_t.set_edge_mask(self.bool_t([True] * 5))
        with self.assertRaises(Exception):
            scene_d.set_edge_mask(self.mods.cuda.Bool([True] * 5))


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class VisibilityParityTests(_ParityBase):
    @staticmethod
    def _segments(spec, count, seed):
        low, high = bs.bounds(spec)
        generator = np.random.default_rng(seed)
        draw = lambda: (low - 0.4 + (high - low + 0.8) * generator.uniform(size=(count, 3))).tolist()  # noqa: E731
        return draw(), draw(), draw()

    def test_visible_pair_and_chain_on_every_scene(self):
        for spec in bs.all_scenes():
            with self.subTest(scene=spec.name):
                scene_t, scene_d = self.scenes(spec)
                count = 128
                start, end_a, end_b = self._segments(spec, count, seed=10)
                active = [index % 7 != 3 for index in range(count)]
                face_count = sum(len(mesh.faces) for mesh in spec.meshes)
                generator = np.random.default_rng(11)
                ignore = generator.integers(-1, face_count, size=count).tolist()

                visible_t = scene_t.visible(bs.torch_rows(start), bs.torch_rows(end_a), self.bool_t(active))
                visible_d = scene_d.visible(
                    bs.drjit_vec3(self.mods, start),
                    bs.drjit_vec3(self.mods, end_a),
                    self.mods.cuda.Int([]),
                    self.mods.cuda.Bool(active),
                )
                self.equal("visible", "visible", visible_t, visible_d.visible)
                blocked = ~bs.host(visible_t) & np.asarray(active)
                self.assertGreater(int(blocked.sum()), 0, "the segment set must contain blocked segments")

                pair_t = scene_t.visible_pair(
                    bs.torch_rows(start),
                    bs.torch_rows(end_a),
                    bs.torch_rows(end_b),
                    self.int_t(ignore),
                    self.bool_t(active),
                )
                pair_d = scene_d.visible_pair(
                    bs.drjit_vec3(self.mods, start),
                    bs.drjit_vec3(self.mods, end_a),
                    bs.drjit_vec3(self.mods, end_b),
                    self.mods.cuda.Int(ignore),
                    self.mods.cuda.Bool(active),
                )
                self.equal("visible_pair", "visible_a", pair_t.visible_a, pair_d.visible_a)
                self.equal("visible_pair", "visible_b", pair_t.visible_b, pair_d.visible_b)

                segments = 3
                points = (
                    np.asarray(bs.bounds(spec)[0])
                    - 0.4
                    + (np.asarray(bs.bounds(spec)[1]) - np.asarray(bs.bounds(spec)[0]) + 0.8)
                    * generator.uniform(size=(count, segments + 1, 3))
                )
                lengths = generator.integers(1, segments + 1, size=count).tolist()
                ignore_chain = generator.integers(-1, face_count, size=(count, segments)).tolist()
                chain_t = scene_t.visible_chain(
                    self.torch.tensor(points, device="cuda", dtype=self.torch.float32),
                    self.int_t(lengths),
                    self.torch.tensor(ignore_chain, device="cuda", dtype=self.torch.int32),
                    self.bool_t(active),
                )
                chain_d = scene_d.visible_chain(
                    bs.drjit_vec3(self.mods, points.reshape(-1, 3).tolist()),
                    self.mods.cuda.Int(lengths),
                    self.mods.cuda.Int(np.asarray(ignore_chain).reshape(-1).tolist()),
                    self.mods.cuda.Bool(active),
                )
                self.assertEqual(
                    (chain_t.chain_count, chain_t.max_segments), (int(chain_d.chain_count), int(chain_d.max_segments))
                )
                for name in ("all_visible", "first_blocked_segment", "first_blocked_prim"):
                    self.equal("visible_chain", name, getattr(chain_t, name), getattr(chain_d, name))

    def test_visible_edge_on_every_scene(self):
        for spec in bs.all_scenes():
            with self.subTest(scene=spec.name):
                scene_t, scene_d = self.scenes(spec)
                count = 96
                source, edge_position, _ = self._segments(spec, count, seed=12)
                direction = bs.unit_vectors(count, seed=13)
                generator = np.random.default_rng(14)
                t_min = (-generator.uniform(0.1, 0.6, size=count)).tolist()
                t_max = generator.uniform(0.1, 0.6, size=count).tolist()
                active = [index % 4 != 1 for index in range(count)]
                fractions = (0.0, 0.3, 0.6, 1.0)
                edge_t = scene_t.visible_edge(
                    bs.torch_rows(source),
                    bs.torch_rows(edge_position),
                    bs.torch_rows(direction),
                    self.float_t(t_min),
                    self.float_t(t_max),
                    fractions,
                    self.bool_t(active),
                )
                edge_d = scene_d.visible_edge(
                    bs.drjit_vec3(self.mods, source),
                    bs.drjit_vec3(self.mods, edge_position),
                    bs.drjit_vec3(self.mods, direction),
                    self.mods.cuda.Float(t_min),
                    self.mods.cuda.Float(t_max),
                    list(fractions),
                    self.mods.cuda.Bool(active),
                )
                self.assertEqual(edge_t.state_count, int(edge_d.state_count))
                self.equal("visible_edge", "any_visible", edge_t.any_visible, edge_d.any_visible)


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class DynamicSceneParityTests(_ParityBase):
    def test_update_mesh_vertices_then_sync(self):
        base = bs.grid_sheet(6)
        spec = bs.SceneSpec(
            "dynamic_grid",
            (bs.MeshSpec(base.meshes[0].vertices, base.meshes[0].faces, dynamic=True), bs.single_triangle().meshes[0]),
        )
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        vertices = np.asarray(base.meshes[0].vertices)
        for step, amplitude in enumerate((0.15, -0.3)):
            with self.subTest(step=step):
                moved = vertices.copy()
                moved[:, 2] = amplitude * np.sin(2.0 * moved[:, 0]) + 0.1 * moved[:, 1]
                version_t, version_d = scene_t.version, scene_d.version
                scene_t.update_mesh_vertices(0, bs.torch_rows(moved.tolist()))
                scene_d.update_mesh_vertices(0, bs.drjit_vec3(self.mods, moved.tolist(), ad=True))
                self.assertTrue(scene_t.has_pending_updates())
                self.assertTrue(scene_d.has_pending_updates())
                origins, directions = bs.scene_rays(spec, 64, seed=20 + step)
                with self.assertRaisesRegex(RuntimeError, "pending updates"):
                    scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
                with self.assertRaisesRegex(RuntimeError, "pending updates"):
                    scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
                scene_t.sync()
                scene_d.sync()
                self.assertFalse(scene_t.has_pending_updates() or scene_d.has_pending_updates())
                self.assertGreater(scene_t.version, version_t)
                self.assertGreater(scene_d.version, version_d)

                hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
                hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
                for name in INTERSECT_FIELDS_CLOSE:
                    self.close("update_mesh_vertices+sync", name, getattr(hit_t, name), getattr(hit_d, name))
                for name in INTERSECT_FIELDS_EQUAL:
                    self.equal("update_mesh_vertices+sync", name, getattr(hit_t, name), getattr(hit_d, name))
                points = bs.scene_points(spec, 64, seed=30 + step)
                near_t = scene_t.nearest_edge(bs.torch_rows(points))
                near_d = scene_d.nearest_edge(bs.drjit_vec3(self.mods, points))
                self.close("update_mesh_vertices+sync", "edge_distance", near_t.distance, near_d.distance)
                self.equal("update_mesh_vertices+sync", "edge_id", near_t.global_edge_id, near_d.global_edge_id)
                start, end = bs.scene_points(spec, 64, seed=40), bs.scene_points(spec, 64, seed=41)
                self.equal(
                    "update_mesh_vertices+sync",
                    "visible",
                    scene_t.visible(bs.torch_rows(start), bs.torch_rows(end)),
                    scene_d.visible(bs.drjit_vec3(self.mods, start), bs.drjit_vec3(self.mods, end)).visible,
                )

    def test_add_instance_matches_ids_and_hits(self):
        spec = bs.grid_sheet(4)
        transform = [[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, -1.0, 0.25], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        scene_t = bs.torch_scene(self.rt, spec, build=False)
        scene_d = bs.drjit_scene(self.mods, spec, build=False)
        instance_t = scene_t.add_instance(0, self.torch.tensor(transform, device="cuda"), dynamic=True)
        instance_d = scene_d.add_instance(0, self.mods.ad.Matrix4f(transform), True)
        self.assertEqual(instance_t, instance_d)
        scene_t.build()
        scene_d.build()
        self.assertEqual(scene_t.num_meshes, scene_d.num_meshes)
        rotated = [
            [
                (np.asarray(transform)[:3, :3] @ np.asarray(row) + np.asarray(transform)[:3, 3]).tolist()
                for row in spec.meshes[0].vertices
            ]
        ]
        world = bs.SceneSpec("instanced", (spec.meshes[0], bs.MeshSpec(rotated[0], spec.meshes[0].faces)))
        origins, directions = bs.scene_rays(world, 256, seed=50)
        hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
        hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        self.assertGreater(int((bs.host(hit_t.shape_id) == instance_t).sum()), 8, "rays must hit the instance")
        for name in INTERSECT_FIELDS_CLOSE:
            self.close("add_instance", name, getattr(hit_t, name), getattr(hit_d, name))
        for name in INTERSECT_FIELDS_EQUAL:
            self.equal("add_instance", name, getattr(hit_t, name), getattr(hit_d, name))
        points = bs.scene_points(world, 64, seed=51)
        near_t = scene_t.nearest_edge(bs.torch_rows(points))
        near_d = scene_d.nearest_edge(bs.drjit_vec3(self.mods, points))
        self.close("add_instance", "edge_distance", near_t.distance, near_d.distance)
        self.equal("add_instance", "global_edge_id", near_t.global_edge_id, near_d.global_edge_id)


def _sphere_sdf(resolution: int, radius: float) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, resolution)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return (np.sqrt(x * x + y * y + z * z) - radius).astype(np.float32)


def _surfel_cloud_rows(count: int, seed: int, offset=(0.0, 0.0, 0.0)):
    generator = np.random.default_rng(seed)
    centre = generator.uniform(-1.0, 1.0, size=(count, 3)) + np.asarray(offset)
    tangent_u = generator.normal(size=(count, 3)) * 0.3
    tangent_v = np.cross(tangent_u, generator.normal(size=(count, 3)))
    tangent_v = 0.25 * tangent_v / np.linalg.norm(tangent_v, axis=1, keepdims=True)
    opacity = generator.uniform(0.3, 0.9, size=count)
    value = generator.uniform(0.0, 3.0, size=count)
    return centre, tangent_u, tangent_v, opacity, value


def _shell_rays(count: int, seed: int, radius: float, extent):
    generator = np.random.default_rng(seed)
    origins = generator.normal(size=(count, 3))
    origins = radius * origins / np.linalg.norm(origins, axis=1, keepdims=True)
    targets = generator.uniform(-1.0, 1.0, size=(count, 3)) * np.asarray(extent)
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins.tolist(), directions.tolist(), targets.tolist()


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class StandaloneGeometryParityTests(_ParityBase):
    """SDF grids, surfel scenes and MixedScene: the operations both backends expose."""

    def _sdf_pair(self, values, position, rotation, scale):
        torch, rt, cuda = self.torch, self.rt, self.mods.cuda
        grid_t = rt.SdfGrid(
            torch.tensor(values, device="cuda"), self.float_t(position), self.float_t(rotation), self.float_t(scale)
        )
        nx, ny, nz = values.shape
        grid_d = self.mods.pj.SdfGrid(
            cuda.Float(values.reshape(-1).tolist()),
            nx,
            ny,
            nz,
            bs.drjit_vec3(self.mods, [position]),
            cuda.Float(list(rotation)),
            bs.drjit_vec3(self.mods, [scale]),
        )
        return grid_t, grid_d

    def _surfel_pair(self, rows, options_d=None):
        torch, rt, cuda = self.torch, self.rt, self.mods.cuda
        centre, tangent_u, tangent_v, opacity, value = rows
        cloud_t = rt.SurfelCloud(
            *(
                torch.tensor(np.asarray(x), device="cuda", dtype=torch.float32)
                for x in (centre, tangent_u, tangent_v, opacity, value)
            )
        )
        cloud_d = self.mods.pj.SurfelCloud(
            bs.drjit_vec3(self.mods, centre.tolist()),
            bs.drjit_vec3(self.mods, tangent_u.tolist()),
            bs.drjit_vec3(self.mods, tangent_v.tolist()),
            cuda.Float(opacity.tolist()),
            cuda.Float(value.tolist()),
        )
        return cloud_t, cloud_d

    def _compare_chains(self, op, chain_t, chain_d, prim_ids: str = "prim_ids"):
        self.equal(op, "valid", chain_t.valid, chain_d.is_valid())
        self.close(op, "t", chain_t.t, chain_d.t)
        self.close(op, "image_sources", chain_t.image_sources, chain_d.image_sources)
        self.equal(op, "prim_ids", chain_t.prim_ids, getattr(chain_d, prim_ids))

    def test_sdf_intersect_visible_and_reflections(self):
        rotation = [math.cos(0.2), 0.0, math.sin(0.2), 0.0]
        grid_t, grid_d = self._sdf_pair(_sphere_sdf(16, 0.6), [0.1, -0.2, 0.05], rotation, [2.0, 1.6, 2.0])
        origins, directions, targets = _shell_rays(128, seed=60, radius=3.0, extent=(0.8, 0.8, 0.8))
        active = [index % 6 != 5 for index in range(128)]
        hit_t = grid_t.intersect(bs.torch_ray(self.rt, origins, directions), active=self.bool_t(active))
        hit_d = grid_d.intersect(bs.drjit_ray(self.mods, origins, directions), active=self.mods.cuda.Bool(active))
        self.assertGreater(int(bs.host(hit_t.hit_mask).sum()), 16)
        self.equal("sdf.intersect", "hit_mask", hit_t.hit_mask, hit_d.hit_mask)
        self.equal("sdf.intersect", "steps", hit_t.steps, hit_d.steps)
        for name in ("t", "position", "normal"):
            self.close("sdf.intersect", name, getattr(hit_t, name), getattr(hit_d, name))
        self.equal(
            "sdf.visible",
            "visible",
            grid_t.visible(bs.torch_rows(origins), bs.torch_rows(targets)),
            grid_d.visible(bs.drjit_vec3(self.mods, origins), bs.drjit_vec3(self.mods, targets)),
        )
        for bounces in (1, 2, 3):
            with self.subTest(bounces=bounces):
                self._compare_chains(
                    "sdf.trace_reflections",
                    grid_t.trace_reflections(bs.torch_ray(self.rt, origins, directions), bounces),
                    grid_d.trace_reflections(bs.drjit_ray(self.mods, origins, directions), bounces),
                )

    def test_surfel_intersect_composite_visible_and_reflections(self):
        # The cloud has no rejected surfel lying within RayEpsilon in front of an accepted one along any of these
        # rays; that configuration is the Torch defect pinned by test_surfel_hit_behind_a_rejected_surfel.
        cloud_t, cloud_d = self._surfel_pair(_surfel_cloud_rows(16, seed=61))
        scene_t = self.rt.SurfelScene(cloud_t)
        scene_t.build()
        options_d = self.mods.pj.SurfelTraceOptions()
        options_d.collect_candidate_stats = True  # Torch always reports candidate counts.
        scene_d = self.mods.pj.SurfelScene(cloud_d, options_d)
        scene_d.build()
        origins, directions, targets = _shell_rays(128, seed=62, radius=4.0, extent=(1.0, 1.0, 1.0))
        ray_t = bs.torch_ray(self.rt, origins, directions)
        ray_d = bs.drjit_ray(self.mods, origins, directions)
        hit_t, hit_d = scene_t.intersect(ray_t), scene_d.intersect(ray_d)
        self.assertGreater(int((bs.host(hit_t.surfel_id) >= 0).sum()), 16)
        self.equal("surfel.intersect", "surfel_id", hit_t.surfel_id, hit_d.surfel_id)
        self.equal("surfel.intersect", "is_valid", hit_t.is_valid(), hit_d.is_valid())
        for name in ("t", "p", "n", "alpha", "gaussian_weight", "local_uv", "opacity", "value"):
            self.close("surfel.intersect", name, getattr(hit_t, name), getattr(hit_d, name))
        composite_t, composite_d = scene_t.composite_alpha(ray_t), scene_d.composite_alpha(ray_d)
        for name in ("alpha", "transmittance", "intensity", "depth"):
            self.close("surfel.composite_alpha", name, getattr(composite_t, name), getattr(composite_d, name))
        for name in ("candidate_count", "candidate_buffer_full"):
            self.equal("surfel.composite_alpha", name, getattr(composite_t, name), getattr(composite_d, name))
        self.equal("surfel.composite_alpha", "is_valid", composite_t.is_valid(), composite_d.is_valid())
        self.close("surfel.transmittance", "transmittance", scene_t.transmittance(ray_t), composite_d.transmittance)
        self.equal(
            "surfel.visible",
            "visible",
            scene_t.visible(bs.torch_rows(origins), bs.torch_rows(targets)),
            scene_d.visible(bs.drjit_vec3(self.mods, origins), bs.drjit_vec3(self.mods, targets)),
        )
        for bounces in (1, 2):
            with self.subTest(bounces=bounces):
                self._compare_chains(
                    "surfel.trace_reflections",
                    scene_t.trace_reflections(ray_t, bounces),
                    scene_d.trace_reflections(ray_d, bounces),
                )

    def test_mixed_scene_intersect_visible_transmittance_and_reflections(self):
        rotation = [1.0, 0.0, 0.0, 0.0]
        grid_t, grid_d = self._sdf_pair(_sphere_sdf(12, 0.6), [1.5, 0.0, 0.0], rotation, [1.0, 1.0, 1.0])
        cloud_t, cloud_d = self._surfel_pair(_surfel_cloud_rows(6, seed=63, offset=(-1.5, 0.0, 0.0)))
        mesh = bs.grid_sheet(4).meshes[0]
        mixed_t = self.rt.MixedScene()
        mixed_d = self.mods.pj.MixedScene()
        for scene, mesh_obj, grid, cloud in (
            (mixed_t, bs.torch_mesh(self.rt, mesh), grid_t, cloud_t),
            (mixed_d, bs.drjit_mesh(self.mods, mesh), grid_d, cloud_d),
        ):
            scene.add_mesh(mesh_obj)
            scene.add_sdf(grid)
            scene.add_surfel(cloud)
            scene.build()
        origins, directions, targets = _shell_rays(256, seed=64, radius=5.0, extent=(2.0, 1.0, 1.0))
        ray_t = bs.torch_ray(self.rt, origins, directions)
        ray_d = bs.drjit_ray(self.mods, origins, directions)
        hit_t, hit_d = mixed_t.intersect(ray_t), mixed_d.intersect(ray_d)
        self.assertEqual(set(bs.host(hit_t.shape_id).tolist()), {-1, 0, 1, 2}, "every geometry kind must be hit")
        for name in INTERSECT_FIELDS_CLOSE + ("n",):
            self.close("mixed.intersect", name, getattr(hit_t, name), getattr(hit_d, name))
        for name in INTERSECT_FIELDS_EQUAL + ("prim_id",):
            self.equal("mixed.intersect", name, getattr(hit_t, name), getattr(hit_d, name))
        self.equal(
            "mixed.visible",
            "visible",
            mixed_t.visible(bs.torch_rows(origins), bs.torch_rows(targets)),
            mixed_d.visible(bs.drjit_vec3(self.mods, origins), bs.drjit_vec3(self.mods, targets)).visible,
        )
        self.close("mixed.transmittance", "transmittance", mixed_t.transmittance(ray_t), mixed_d.transmittance(ray_d))
        for bounces in (1, 2):
            with self.subTest(bounces=bounces):
                # MixedScene reflection prim_ids are mixed-global in Torch; Dr.Jit names them global_prim_ids.
                self._compare_chains(
                    "mixed.trace_reflections",
                    mixed_t.trace_reflections(ray_t, bounces),
                    mixed_d.trace_reflections(ray_d, bounces),
                    prim_ids="global_prim_ids",
                )


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class KnownDivergenceTests(_ParityBase):
    """Each test pins one cross-backend divergence so that changing either backend or the contract is noticed."""

    def test_intersect_ray_tmin_follows_each_backend_profile(self):
        tmin_t, tmin_d = PROFILES["torch"]["ray_tmin"], PROFILES["drjit"]["ray_tmin"]
        self.assertEqual(
            CONTRACT["operations"]["intersect"]["backend_overrides"],
            {"drjit": {"ray_tmin": tmin_d}, "torch": {"ray_tmin": tmin_t}},
        )
        scene_t, scene_d = self.scenes(bs.single_triangle())
        # Probe distances sit a factor of ~3 or more from each threshold so float32 rounding cannot decide.
        probes = [tmin_t / 3.0, math.sqrt(tmin_t * tmin_d), tmin_d / 2.0, 3.0 * tmin_d]
        origins = [[0.25, 0.25, -distance] for distance in probes]
        directions = [[0.0, 0.0, 1.0]] * len(probes)
        hit_t = bs.host(scene_t.intersect(bs.torch_ray(self.rt, origins, directions)).t)
        hit_d = bs.host(scene_d.intersect(bs.drjit_ray(self.mods, origins, directions)).t)
        self.assertEqual(np.isfinite(hit_t).tolist(), [distance > tmin_t for distance in probes])
        self.assertEqual(np.isfinite(hit_d).tolist(), [distance > tmin_d for distance in probes])
        self.assertNotEqual(np.isfinite(hit_t).tolist(), np.isfinite(hit_d).tolist(), "the divergence is gone")

    def test_shadow_tmin_drjit_shadow_test_versus_shared_segment_visibility(self):
        shadow_t, shadow_d = PROFILES["torch"]["shadow_tmin"], PROFILES["drjit"]["shadow_tmin"]
        offset = PROFILES["torch"]["endpoint_offset"]
        self.assertEqual(offset, PROFILES["drjit"]["endpoint_offset"])
        self.assertFalse(hasattr(self.rt.Scene, "shadow_test"), "Torch gained a scene-level shadow_test")
        scene_t, scene_d = self.scenes(bs.single_triangle())
        probes = [(shadow_t + offset) / 6.0, 3.0 * (shadow_t + offset), shadow_d / 2.0, 3.0 * shadow_d]
        starts = [[0.25, 0.25, -distance] for distance in probes]
        ends = [[0.25, 0.25, 1.0]] * len(probes)
        visible_t = bs.host(scene_t.visible(bs.torch_rows(starts), bs.torch_rows(ends)))
        visible_d = bs.host(scene_d.visible(bs.drjit_vec3(self.mods, starts), bs.drjit_vec3(self.mods, ends)).visible)
        expected_segment = [distance <= shadow_t + offset for distance in probes]
        self.assertEqual(visible_t.tolist(), expected_segment)
        self.assertEqual(visible_d.tolist(), expected_segment, "segment visibility is shared by both backends")
        directions = [[0.0, 0.0, 1.0]] * len(probes)
        occluded_d = bs.host(scene_d.shadow_test(bs.drjit_ray(self.mods, starts, directions, [1.0] * len(probes))))
        self.assertEqual(occluded_d.tolist(), [distance > shadow_d for distance in probes])

    def test_topk_tie_break_matches_the_contract_record(self):
        record = CONTRACT["edge_topk_tie_break"]
        self.assertEqual(record["contract"], "(distance_squared, global_edge_id)")
        spec = bs.grid_sheet(4)
        # Above an interior grid vertex, its six incident edges are exactly equidistant.
        vertex = [0.0, 0.0, 0.25]
        for torch_edge, drjit_edge in EDGE_BACKEND_PAIRS:
            with self.subTest(torch_edge=torch_edge, drjit_edge=drjit_edge):
                scene_t, scene_d = self.scenes(spec, torch_edge=torch_edge, drjit_edge=drjit_edge)
                full_t = scene_t.nearest_edges(bs.torch_rows([vertex]), 8)
                distances = bs.host(full_t.distances).reshape(-1)
                tied = sorted(bs.host(full_t.global_edge_ids).reshape(-1)[distances == distances[0]].tolist())
                self.assertEqual(len(tied), 6, "six incident edges must tie")
                top_t = bs.host(scene_t.nearest_edges(bs.torch_rows([vertex]), 3).global_edge_ids).reshape(-1)
                top_d = bs.host(scene_d.nearest_edges(bs.drjit_vec3(self.mods, [vertex]), 3).global_edge_ids)
                self.assertEqual(top_t.tolist(), tied[:3], "Torch implements the (d^2, global id) tie-break")
                self.assertTrue(set(top_d.tolist()) <= set(tied), "Dr.Jit must still return tied edges")
                drjit_paths = {"optix": "drjit_optix", "drjit": "drjit_native_bvh", "optix_drjit": "drjit_native_bvh"}
                if drjit_paths[drjit_edge] in record["implemented_in"]:
                    self.assertEqual(top_d.tolist(), tied[:3])
                else:
                    self.assertIn(drjit_paths[drjit_edge], record["not_implemented_in"])

    def test_auto_edge_backend_resolves_differently(self):
        spec = bs.single_triangle()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        self.assertEqual(scene_t.edge_bvh_backend, "cuda")
        self.assertEqual(scene_d.edge_bvh_backend, "drjit")
        self.assertEqual(scene_t.trace_backend, scene_d.trace_backend_name())

    def test_edge_mask_commit_point_differs(self):
        # Torch applies set_edge_mask() immediately; Dr.Jit queues it and refuses queries until sync().
        spec = bs.single_triangle()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        point = [[0.5, -0.3, 0.0]]
        self.assertEqual(int(bs.host(scene_t.nearest_edge(bs.torch_rows(point)).global_edge_id)[0]), 0)
        scene_t.set_edge_mask(self.bool_t([False, True, True]))
        scene_d.set_edge_mask(self.mods.cuda.Bool([False, True, True]))
        self.assertFalse(scene_t.has_pending_updates())
        self.assertTrue(scene_d.has_pending_updates())
        self.assertNotEqual(int(bs.host(scene_t.nearest_edge(bs.torch_rows(point)).global_edge_id)[0]), 0)
        with self.assertRaisesRegex(RuntimeError, "pending updates"):
            scene_d.nearest_edge(bs.drjit_vec3(self.mods, point))
        scene_d.sync()
        self.equal(
            "edge_mask",
            "after_sync",
            scene_t.nearest_edge(bs.torch_rows(point)).global_edge_id,
            scene_d.nearest_edge(bs.drjit_vec3(self.mods, point)).global_edge_id,
        )

    def test_prim_id_is_global_in_torch_and_local_in_drjit(self):
        # Undocumented: contracts list `prim_id` for both backends without an id space.
        scene_t, scene_d = self.scenes(bs.overlapping_meshes())
        origins, directions = [[0.5, 0.5, 3.0], [1.5, 1.5, 3.0]], [[0.0, 0.0, -1.0]] * 2
        hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
        hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        self.assertEqual(bs.host(hit_t.shape_id).tolist(), [1, 1])
        self.assertEqual(bs.host(hit_t.prim_id).tolist(), bs.host(hit_t.global_prim_id).tolist())
        self.assertEqual(bs.host(hit_d.prim_id).tolist(), bs.host(hit_d.local_prim_id).tolist())
        self.assertNotEqual(bs.host(hit_t.prim_id).tolist(), bs.host(hit_d.prim_id).tolist())
        # ReflectionChain.prim_ids follows the same split: global in Torch, shape-local in Dr.Jit.
        chain_t = scene_t.trace_reflections(bs.torch_ray(self.rt, origins, directions), 1)
        chain_d = scene_d.trace_reflections(bs.drjit_ray(self.mods, origins, directions), 1, symbolic=False)
        self.assertEqual(bs.host(chain_t.prim_ids).reshape(-1).tolist(), bs.host(chain_d.global_prim_ids).tolist())
        self.assertEqual(bs.host(chain_d.prim_ids).tolist(), bs.host(chain_d.local_prim_ids).tolist())
        self.assertNotEqual(bs.host(chain_t.prim_ids).reshape(-1).tolist(), bs.host(chain_d.prim_ids).tolist())

    def test_shading_normal_is_the_face_normal_in_torch_and_interpolated_in_drjit(self):
        # Undocumented: Torch ignores Mesh.use_face_normals=False and always returns the face normal.
        scene_t, scene_d = self.scenes(bs.closed_box())
        origins, directions = [[0.3, 0.2, 3.0], [0.0, 0.0, 3.0]], [[0.0, 0.0, -1.0]] * 2
        hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
        hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        np.testing.assert_allclose(bs.host(hit_t.n), bs.host(hit_t.geo_n), atol=1e-7)
        self.assertGreater(float(np.abs(bs.host(hit_d.n) - bs.host(hit_d.geo_n)).max()), 0.1)
        face_t = bs.torch_scene(self.rt, bs.closed_box())
        face_spec = bs.closed_box()
        face_d = self.mods.pj.Scene()
        mesh = bs.drjit_mesh(self.mods, face_spec.meshes[0])
        mesh.use_face_normals = True
        face_d.add_mesh(mesh)
        face_d.build()
        self.close(
            "intersect[use_face_normals]",
            "n",
            face_t.intersect(bs.torch_ray(self.rt, origins, directions)).n,
            face_d.intersect(bs.drjit_ray(self.mods, origins, directions)).n,
        )

    def test_surfel_proxy_triangle_ids_and_candidate_stats_differ(self):
        # Torch proxies each surfel with two triangles and always counts candidates; Dr.Jit uses a 20-triangle
        # icosahedron proxy and reports candidate counts only with SurfelTraceOptions.collect_candidate_stats.
        rows = _surfel_cloud_rows(12, seed=70)
        centre, tangent_u, tangent_v, opacity, value = rows
        torch = self.torch
        scene_t = self.rt.SurfelScene(
            self.rt.SurfelCloud(*(torch.tensor(np.asarray(x), device="cuda", dtype=torch.float32) for x in rows))
        )
        scene_t.build()
        cloud_d = self.mods.pj.SurfelCloud(
            bs.drjit_vec3(self.mods, centre.tolist()),
            bs.drjit_vec3(self.mods, tangent_u.tolist()),
            bs.drjit_vec3(self.mods, tangent_v.tolist()),
            self.mods.cuda.Float(opacity.tolist()),
            self.mods.cuda.Float(value.tolist()),
        )
        options_d = self.mods.pj.SurfelTraceOptions()
        self.assertEqual(options_d.primitive_mode, self.mods.pj.SurfelPrimitiveMode.Icosahedron20)
        self.assertFalse(options_d.collect_candidate_stats)
        scene_d = self.mods.pj.SurfelScene(cloud_d, options_d)
        scene_d.build()
        origins, directions, _ = _shell_rays(64, seed=71, radius=4.0, extent=(1.0, 1.0, 1.0))
        hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
        hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        ids = bs.host(hit_t.surfel_id)
        rows_hit = ids >= 0
        self.assertGreater(int(rows_hit.sum()), 8)
        self.assertEqual((bs.host(hit_t.triangle_id)[rows_hit] // 2).tolist(), ids[rows_hit].tolist())
        self.assertEqual((bs.host(hit_d.triangle_id)[rows_hit] // 20).tolist(), ids[rows_hit].tolist())
        ray_t = bs.torch_ray(self.rt, origins, directions)
        self.assertGreater(int(bs.host(scene_t.composite_alpha(ray_t).candidate_count).sum()), 0)
        counts_d = bs.host(scene_d.composite_alpha(bs.drjit_ray(self.mods, origins, directions)).candidate_count)
        self.assertEqual(int(counts_d.sum()), 0)

    def test_mixed_scene_mesh_prim_id_is_global_in_torch_and_local_in_drjit(self):
        mesh = bs.overlapping_meshes()
        mixed_t, mixed_d = self.rt.MixedScene(), self.mods.pj.MixedScene()
        for item in mesh.meshes:
            mixed_t.add_mesh(bs.torch_mesh(self.rt, item))
            mixed_d.add_mesh(bs.drjit_mesh(self.mods, item))
        mixed_t.build()
        mixed_d.build()
        origins, directions = [[0.5, 0.5, 3.0], [1.5, 1.5, 3.0]], [[0.0, 0.0, -1.0]] * 2
        hit_t = mixed_t.intersect(bs.torch_ray(self.rt, origins, directions))
        hit_d = mixed_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        self.assertEqual(bs.host(hit_t.shape_id).tolist(), [1, 1])
        self.assertEqual(bs.host(hit_t.prim_id).tolist(), bs.host(hit_t.global_prim_id).tolist())
        self.assertEqual(bs.host(hit_d.prim_id).tolist(), bs.host(hit_d.local_prim_id).tolist())
        self.equal("mixed.intersect", "global_prim_id", hit_t.global_prim_id, hit_d.global_prim_id)

    def test_uv_without_mesh_uv_is_barycentric_in_torch_and_zero_in_drjit(self):
        scene_t, scene_d = self.scenes(bs.single_triangle())
        origins, directions = [[0.2, 0.3, -1.0]], [[0.0, 0.0, 1.0]]
        uv_t = bs.host(scene_t.intersect(bs.torch_ray(self.rt, origins, directions)).uv)
        uv_d = bs.host(scene_d.intersect(bs.drjit_ray(self.mods, origins, directions)).uv)
        np.testing.assert_allclose(uv_t, [[0.2, 0.3]], atol=1e-6)
        np.testing.assert_allclose(uv_d, [[0.0, 0.0]], atol=0.0)

    def test_supplied_mesh_uv_is_interpolated_by_both(self):
        # Regression: the Torch intersect kernel once wrote the raw barycentric (u, v) into `uv` and ignored a
        # caller-supplied UV map; both backends now interpolate it.
        uv = [[0.5, 0.5], [1.0, 0.5], [0.5, 1.0]]
        spec = bs.SceneSpec("uv_triangle", (bs.MeshSpec(bs.single_triangle().meshes[0].vertices, [[0, 1, 2]], uv),))
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        origins, directions = [[0.2, 0.3, -1.0]], [[0.0, 0.0, 1.0]]
        uv_d = bs.host(scene_d.intersect(bs.drjit_ray(self.mods, origins, directions)).uv)
        np.testing.assert_allclose(uv_d, [[0.6, 0.65]], atol=1e-6)
        uv_t = bs.host(scene_t.intersect(bs.torch_ray(self.rt, origins, directions)).uv)
        np.testing.assert_allclose(uv_t, uv_d, atol=1e-6)

    def test_surfel_hit_behind_a_rejected_surfel(self):
        # Regression: the Torch proxy march once restarted `t + RayEpsilon` (1e-3) past a rejected proxy hit and
        # never examined a surfel closer behind it. Here surfel 0 is crossed at 2 sigma with opacity 0.01
        # (alpha < alpha_min, rejected) and surfel 1 lies 5e-4 behind it at its centre.
        rows = (
            np.asarray([[-2.0, 0.0, 0.0], [0.0, 0.0, 5.0e-4]]),
            np.asarray([[1.0, 0.0, 0.0]] * 2),
            np.asarray([[0.0, 1.0, 0.0]] * 2),
            np.asarray([0.01, 0.9]),
            np.asarray([1.0, 2.0]),
        )
        cloud_t, cloud_d = StandaloneGeometryParityTests._surfel_pair(self, rows)
        scene_t, scene_d = self.rt.SurfelScene(cloud_t), self.mods.pj.SurfelScene(cloud_d)
        scene_t.build()
        scene_d.build()
        origins, directions = [[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]]
        hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        self.assertEqual(bs.host(hit_d.surfel_id).tolist(), [1])
        hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
        self.assertEqual(bs.host(hit_t.surfel_id).tolist(), [1])

    def test_shading_normal_is_finite_at_a_non_manifold_fan(self):
        # At a fan edge whose incident face normals sum to zero (three fins at 120 degrees) the vertex normal has no
        # direction; Dr.Jit shades the corners that reference it with their face normal, and Torch always returns
        # the face normal. Neither backend may return NaN for a finite hit.
        scene_t, scene_d = self.scenes(bs.non_manifold_fan())
        origins, directions = [[0.3, 0.2, 3.0]], [[0.0, 0.0, -1.0]]
        hit_t = scene_t.intersect(bs.torch_ray(self.rt, origins, directions))
        hit_d = scene_d.intersect(bs.drjit_ray(self.mods, origins, directions))
        self.assertTrue(np.isfinite(bs.host(hit_t.n)).all())
        self.assertTrue(np.isfinite(bs.host(hit_d.n)).all(), f"Dr.Jit n = {bs.host(hit_d.n).tolist()}")
        np.testing.assert_allclose(np.abs(bs.host(hit_d.n)).ravel(), np.abs(bs.host(hit_d.geo_n)).ravel(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
