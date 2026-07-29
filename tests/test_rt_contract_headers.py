# Copyright Xingyu Chen.
# Tests consolidated runtime contracts and their owners.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT / "include/rayd/math.h"
CONTRACTS = ROOT / "include/rayd/contracts.h"
JIT_SCENE = ROOT / "include/rayd/jit/scene.h"
RT_INTERNAL = ROOT / "src/runtime/rt_internal.h"
RT_DEVICE = ROOT / "src/runtime/rt_device.cuh"


class RtContractHeaderTests(unittest.TestCase):
    def test_runtime_contracts_have_one_owner(self):
        self.assertFalse((ROOT / "include/rayd/rt").exists())
        for path in (MATH, CONTRACTS, JIT_SCENE, RT_INTERNAL, RT_DEVICE):
            self.assertTrue(path.is_file(), path)

    def test_math_owns_numeric_policy_and_qualifiers(self):
        text = MATH.read_text(encoding="utf-8")
        for symbol in (
            "struct NumericPolicy",
            "kDrJitLegacyProfile",
            "kTorchLegacyProfile",
            "kMultipathTraceTMin",
            "define RAYD_DEVICE",
            "define RAYD_HOST_DEVICE",
        ):
            self.assertIn(symbol, text)

    def test_device_owner_contains_traverser_and_primitive_helper(self):
        text = RT_DEVICE.read_text(encoding="utf-8")
        for symbol in ("struct TriangleHit", "is_traverser", "is_traverser_v", "struct TraceConfig"):
            self.assertIn(symbol, text)
        self.assertIn("global_primitive_id", text)

    def test_internal_owner_contains_pipeline_counts(self):
        text = RT_INTERNAL.read_text(encoding="utf-8")
        for symbol in (
            "SceneIntersectionPayloadCount",
            "TriangleHitPayloadCount",
            "VisibilityPayloadCount",
            "DiffractionPayloadCount",
            "EdgeTopKPayloadCount",
        ):
            self.assertIn(symbol, text)

    def test_sbt_contract_is_shared_and_backend_free(self):
        text = CONTRACTS.read_text(encoding="utf-8")
        for symbol in ("SbtRecordAlignment", "SbtRecordHeaderSize", "SbtRecord", "EmptySbtRecord"):
            self.assertIn(symbol, text)
        self.assertNotIn("drjit-core", text)
        self.assertNotIn("torch/", text)

    def test_backend_descriptors_follow_the_jit_scene_api(self):
        text = JIT_SCENE.read_text(encoding="utf-8")
        for symbol in ("TraceBackendKind", "IntegrationMode", "TraceCapabilities"):
            self.assertIn(symbol, text)


if __name__ == "__main__":
    unittest.main()
