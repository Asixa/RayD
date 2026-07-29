# Copyright Xingyu Chen.
# Verifies the unified mixed-geometry scene decision, contract, API, and performance evidence.

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MixedSceneContractTests(unittest.TestCase):
    def test_cross_backend_capability_and_closed_effect_scope(self) -> None:
        public_api = json.loads((ROOT / "contracts" / "public_api.json").read_text(encoding="utf-8"))
        operations = json.loads((ROOT / "contracts" / "operations.json").read_text(encoding="utf-8"))
        self.assertTrue(public_api["backends"]["torch"]["capabilities"]["mixed_scene"])
        self.assertTrue(public_api["backends"]["drjit"]["capabilities"]["mixed_scene"])
        scope = operations["mixed_scene_scope"]
        self.assertEqual(scope["does_not_affect"], ["diffraction_direct", "diffraction_chain"])
        self.assertIn("oriented bounding box", scope["sdf_broad_phase"])
        self.assertIn("no host read", scope["execution"])
        self.assertIn("SDF grids are ignored", scope["transmission"])
        self.assertEqual(
            scope["geometry_order_on_exact_ties"],
            ["mesh", "sdf_by_insertion", "surfel_scene_by_insertion", "surfel_id"],
        )

    def test_public_surfaces_expose_mixed_scene_without_diffraction(self) -> None:
        torch_source = (ROOT / "python" / "rayd" / "torch" / "__init__.py").read_text(encoding="utf-8")
        jit_header = (ROOT / "include" / "rayd" / "jit" / "mixed_scene.h").read_text(encoding="utf-8")
        self.assertIn('"MixedScene"', torch_source)
        self.assertIn("class MixedScene final", jit_header)
        for forbidden in ("trace_dfr", "diffraction", "nearest_edge"):
            self.assertNotIn(forbidden, jit_header)

    def test_accepted_adr_and_stress_evidence_are_complete(self) -> None:
        adr = (ROOT / "docs" / "adr" / "0043-unified-mixed-geometry-scene.md").read_text(encoding="utf-8")
        self.assertIn("- Status: Accepted", adr)
        self.assertIn("This is intentionally not a single fused OptiX launch", adr)
        evidence = json.loads(
            (ROOT / "benchmarks" / "baselines" / "mixed_geometry_20260729.json").read_text(encoding="utf-8")
        )
        for backend in ("torch", "drjit"):
            self.assertEqual(set(evidence["cases"][backend]), {"65536", "262144", "1048576"})
            for case in evidence["cases"][backend].values():
                self.assertTrue(case["all_gates_passed"])
                self.assertTrue(case["stress_sanity_passed"])


if __name__ == "__main__":
    unittest.main()
