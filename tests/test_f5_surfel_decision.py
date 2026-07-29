# Copyright Xingyu Chen.
# Tests the superseding cross-backend surfel decision.

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = ROOT / "contracts/surfel_backend_decision.json"
MANIFEST_PATH = ROOT / "contracts/public_api.json"


class SurfelBackendDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

    def test_decision_is_accepted_cross_backend_and_has_an_adr(self) -> None:
        self.assertEqual(self.decision["schema_version"], 2)
        self.assertEqual(self.decision["decision_id"], "cross-backend-implicit-geometry")
        self.assertEqual(self.decision["status"], "accepted")
        self.assertEqual(self.decision["decision"], "cross_backend")
        adr = ROOT / self.decision["adr"]
        self.assertTrue(adr.is_file())
        adr_text = adr.read_text(encoding="utf-8")
        self.assertIn("Status: Accepted", adr_text)
        self.assertIn("Cross-backend implicit and surfel geometry", adr_text)
        self.assertEqual(self.decision["supersedes"], "docs/adr/0001-surfel-remains-drjit-only.md")

    def test_capability_and_operation_scope_are_explicit(self) -> None:
        capability = self.decision["capability"]
        self.assertEqual(capability["key"], "surfel")
        self.assertEqual(capability["category"], "surfel")
        self.assertEqual(capability["stability"], "experimental")
        self.assertEqual(capability["backends"], {"drjit": True, "torch": True})
        self.assertEqual(self.decision["operation_scope"]["sdf"], ["visibility", "reflection_trace"])
        self.assertEqual(self.decision["operation_scope"]["surfel"], ["visibility", "reflection_trace", "transmission"])
        self.assertEqual(self.decision["operation_scope"]["diffraction"], [])

    def test_repository_shape_supports_both_backends(self) -> None:
        self.assertTrue((ROOT / "src/surfel_optix_jit.cu").is_file())
        self.assertTrue((ROOT / "src/surfel_jit.cpp").is_file())
        self.assertTrue((ROOT / "python/rayd/_impl/surfel.py").is_file())
        self.assertTrue((ROOT / "src/sdf_jit.cpp").is_file())
        self.assertTrue((ROOT / "src/sdf_jit.cu").is_file())
        self.assertTrue((ROOT / "python/rayd/_impl/sdf.py").is_file())

    def test_manifest_matches_the_superseding_decision(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = self.decision["minimal_manifest"]
        surfel_api = manifest["apis"]["surfel"]
        self.assertEqual(surfel_api["category"], expected["api"]["category"])
        self.assertEqual(surfel_api["stability"], expected["api"]["stability"])
        self.assertEqual(
            manifest["backends"]["drjit"]["capabilities"]["surfel"], expected["backend_capabilities"]["drjit"]
        )
        self.assertEqual(
            manifest["backends"]["torch"]["capabilities"]["surfel"], expected["backend_capabilities"]["torch"]
        )
        self.assertEqual(expected["new_capability_keys"], [])


if __name__ == "__main__":
    unittest.main()
