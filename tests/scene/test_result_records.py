# Copyright Xingyu Chen.
# Checks the Torch result-record helpers, backend enums, and keyword-only scene selectors.

from __future__ import annotations

import dataclasses
import unittest

import torch

import rayd.torch as rt

from tests.support.geometry import build_scene


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class ResultRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.scene = build_scene(self.device)
        origins = torch.tensor([[0.1, 0.2, 1.0], [5.0, 5.0, 1.0]], device=self.device)
        directions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device=self.device)
        self.ray = rt.Ray(origins, directions)

    def test_every_intersection_result_is_an_intersection(self) -> None:
        for flags in (rt.RayFlags.All, rt.RayFlags.Empty):
            with self.subTest(flags=flags):
                hit = self.scene.intersect(self.ray, flags=flags)
                self.assertIsInstance(hit, rt.Intersection)
                self.assertEqual(hit.valid.tolist(), [True, False])
                self.assertIs(hit.instance_id, hit.shape_id)

    def test_repr_describes_tensors_without_their_values(self) -> None:
        text = repr(self.scene.intersect(self.ray))
        self.assertTrue(text.startswith("Intersection(t=float32[2] cuda:"), text)
        self.assertNotIn("tensor(", text)
        chain = self.scene.trace_reflections(self.ray, 1)
        self.assertIn("<not loaded>", repr(chain))

    def test_to_and_detach_return_new_records(self) -> None:
        hit = self.scene.intersect(self.ray)
        moved = hit.to("cpu")
        self.assertIs(type(moved), rt.Intersection)
        self.assertEqual(moved.t.device.type, "cpu")
        torch.testing.assert_close(moved.t, hit.t.cpu())
        chain = self.scene.trace_reflections(self.ray, 2).detach()
        self.assertIsInstance(chain, rt.ReflectionChain)
        self.assertTrue(torch.equal(chain.is_valid(), chain.valid))

    def test_nearest_edge_masks_inactive_rows(self) -> None:
        points = torch.tensor([[0.1, 0.2, 0.5], [0.3, 0.1, 0.5]], device=self.device)
        active = torch.tensor([True, False], device=self.device)
        point_edges = self.scene.nearest_edge(points, active)
        self.assertTrue(torch.isfinite(point_edges.distance[0]))
        self.assertEqual(float(point_edges.distance[1]), float("inf"))
        self.assertEqual(int(point_edges.global_edge_id[1]), -1)
        ray_edges = self.scene.nearest_edge(self.ray, active)
        self.assertEqual(int(ray_edges.global_edge_id[1]), -1)

    def test_split_complex_fields_have_complex_views(self) -> None:
        zeros = torch.zeros(2)
        field = rt.ReflEpcField(zeros, torch.ones(2), zeros, zeros.bool(), zeros.int())
        self.assertTrue(torch.equal(field.field, torch.complex(zeros, torch.ones(2))))


class OptionsAndEnumTests(unittest.TestCase):
    def test_backend_selectors_are_keyword_only_and_exact(self) -> None:
        with self.assertRaises(TypeError):
            rt.Scene("optix")  # type: ignore[misc]
        with self.assertRaises(TypeError):
            rt.MixedScene("optix")  # type: ignore[misc]
        for name in ("trace_backend", "edge_bvh_backend"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                rt.Scene(**{name: "OptiX"})
        self.assertEqual(rt.TraceBackend.CUDA, "cuda")
        self.assertEqual(rt.EdgeBVHBackend("optix"), rt.EdgeBVHBackend.OPTIX)

    def test_accum_strategy_is_an_enum(self) -> None:
        self.assertIs(rt.AccumOptions(accumulation_strategy=2).accumulation_strategy, rt.AccumStrategy.STAGED)
        with self.assertRaises(ValueError):
            rt.AccumOptions(accumulation_strategy=7)

    def test_dfr_options_follow_the_drjit_defaults(self) -> None:
        defaults = {field.name: field.default for field in dataclasses.fields(rt.DfrOptions)}
        self.assertEqual(defaults["wavelength"], 1.0)
        self.assertEqual(defaults["max_order"], 1)
        self.assertEqual((defaults["lane_offset"], defaults["lane_count"]), (0, -1))
        coherent = rt.DfrCoherentOptions()
        self.assertTrue(coherent.select_diffraction_point and coherent.prefilter_visibility)
        self.assertIsNone(rt.DfrPathOptions().max_paths)

    def test_ray_flags_has_no_none_member(self) -> None:
        self.assertEqual(int(rt.RayFlags.Empty), 0)
        self.assertNotIn("None", rt.RayFlags.__members__)


if __name__ == "__main__":
    unittest.main()
