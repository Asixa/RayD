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

    def test_decision_is_accepted_drjit_only_and_has_an_adr(self) -> None:
        self.assertEqual(self.decision["schema_version"], 1)
        self.assertEqual(self.decision["decision_id"], "F5-surfel-backend-scope")
        self.assertEqual(self.decision["status"], "accepted")
        self.assertEqual(self.decision["decision"], "drjit_only")
        adr = ROOT / self.decision["adr"]
        self.assertTrue(adr.is_file())
        adr_text = adr.read_text(encoding="utf-8")
        self.assertIn("Status: Accepted", adr_text)
        self.assertIn("Keep surfel support Dr.Jit-only", adr_text)

    def test_capability_contract_is_explicit_and_asymmetric(self) -> None:
        capability = self.decision["capability"]
        self.assertEqual(capability["key"], "surfel")
        self.assertEqual(capability["category"], "surfel")
        self.assertEqual(capability["stability"], "experimental")
        self.assertEqual(capability["backends"], {"drjit": True, "torch": False})

    def test_repository_shape_supports_the_decision(self) -> None:
        evidence = self.decision["evidence"]
        self.assertGreaterEqual(evidence["repository"]["focused_tests"], 50)
        self.assertGreaterEqual(evidence["repository"]["surfel_commits"], 10)
        self.assertFalse(evidence["torch_demand"]["evidence_sufficient_for_port"])
        self.assertTrue(evidence["reuse"]["raw_pointer_params_candidate"])
        self.assertTrue(evidence["reuse"]["optix_device_core_candidate"])
        self.assertFalse(evidence["reuse"]["host_scene_owner_backend_neutral"])
        self.assertFalse(evidence["reuse"]["ad_replay_backend_neutral"])

        self.assertTrue((ROOT / "src/surfel/surfel_optix_jit.cu").is_file())
        self.assertTrue((ROOT / "src/surfel/surfel_jit.cpp").is_file())
        torch_frontend_sources = ROOT / "torch" / "src"
        self.assertFalse(torch_frontend_sources.exists())
        torch_cmake = (ROOT / "torch" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("src/surfel/", torch_cmake)

    def test_reconsideration_is_gated_and_core_parity_stays_first(self) -> None:
        gates = set(self.decision["reconsideration_gates"])
        self.assertIn("named_torch_workflow_with_fixture_and_owner", gates)
        self.assertIn("shared_params_and_device_core_used_by_both_backends", gates)
        self.assertIn("forward_and_ad_parity_matrix", gates)
        self.assertIn("measured_build_wheel_cold_create_runtime_memory_cost", gates)
        self.assertIn("core_edge_visibility_group_acceptance_passing", gates)
        self.assertTrue(
            self.decision["evidence"]["priority"]
            ["core_edge_visibility_before_surfel_port"]
        )

    def test_phase_f2_manifest_matches_the_f5_decision(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = self.decision["minimal_manifest"]
        surfel_api = manifest["apis"]["surfel"]
        self.assertEqual(surfel_api["category"], expected["api"]["category"])
        self.assertEqual(surfel_api["stability"], expected["api"]["stability"])
        self.assertEqual(
            manifest["backends"]["drjit"]["capabilities"]["surfel"],
            expected["backend_capabilities"]["drjit"],
        )
        self.assertEqual(
            manifest["backends"]["torch"]["capabilities"]["surfel"],
            expected["backend_capabilities"]["torch"],
        )
        self.assertEqual(expected["new_capability_keys"], [])


if __name__ == "__main__":
    unittest.main()
