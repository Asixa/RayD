"""Assert the golden suite maps every RAY_TRACING_BACKEND_ARCHITECTURE section 16 case.

Pure metadata checks over ``coverage.json`` and ``scenes.py``; no GPU required.
"""

import json
import unittest
from pathlib import Path

from tests.golden import scenes as scene_defs

HERE = Path(__file__).resolve().parent

# The canonical section 16 test-scope cases (geometry + edge). Extending this
# set is a deliberate act; the coverage manifest must account for every entry.
CANONICAL_SECTION16_CASES = {
    # geometry
    "miss",
    "front_back_face",
    "shared_edge_vertex",
    "degenerate_triangle",
    "large_coordinates",
    "finite_tmax",
    "self_intersection",
    "multi_mesh",
    "instance",
    "dynamic_refit",
    "id_mapping",
    "ignore_primitive",
    "inactive_lane",
    "empty_or_large_batch",
    # edge
    "point_ray_nearest",
    "finite_infinite_ray",
    "boundary_edge",
    "topk",
    "equal_distance_tie",
    "mask_update",
    "edge_dynamic_refit",
    "edge_vjp_jvp",
}


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.coverage = json.loads((HERE / "coverage.json").read_text(encoding="utf-8"))
        self.scene_names = {scene["name"] for scene in scene_defs.SCENES}

    def test_every_section16_case_is_accounted_for_exactly_once(self):
        golden = set(self.coverage["golden"].keys())
        not_golden = set(self.coverage["not_golden"].keys())
        self.assertEqual(golden & not_golden, set(), "a case is both golden and not_golden")
        self.assertEqual(golden | not_golden, CANONICAL_SECTION16_CASES)

    def test_golden_cases_reference_existing_scenes(self):
        for case, entry in self.coverage["golden"].items():
            self.assertIn("scenes", entry, case)
            self.assertTrue(entry["scenes"], f"{case}: empty scene list")
            for name in entry["scenes"]:
                self.assertIn(name, self.scene_names, f"{case} references unknown scene {name!r}")

    def test_instances_are_marked_not_applicable(self):
        self.assertEqual(self.coverage["not_golden"]["instance"]["status"], "na")
        self.assertIn("no instancing", self.coverage["not_golden"]["instance"]["rationale"])

    def test_not_golden_entries_have_a_status_and_rationale(self):
        for case, entry in self.coverage["not_golden"].items():
            self.assertIn(entry["status"], {"na", "covered_elsewhere"}, case)
            self.assertTrue(entry.get("rationale"), f"{case}: missing rationale")

    def test_scene_case_tags_are_declared_in_the_manifest(self):
        golden_cases = set(self.coverage["golden"].keys())
        for scene in scene_defs.SCENES:
            for case in scene.get("cases", []):
                self.assertIn(case, golden_cases, f"scene {scene['name']} tags undeclared case {case!r}")
                self.assertIn(
                    scene["name"],
                    self.coverage["golden"][case]["scenes"],
                    f"scene {scene['name']} tags case {case!r} but is not listed under it",
                )

    def test_every_scene_name_is_unique(self):
        names = [scene["name"] for scene in scene_defs.SCENES]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
