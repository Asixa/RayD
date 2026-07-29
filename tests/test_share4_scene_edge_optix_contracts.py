# Copyright Xingyu Chen.
# Tests share4 scene edge optix contracts.

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCENE_CONTRACT = ROOT / "include/rayd/scene/optix_contracts.h"
EDGE_CONTRACT = ROOT / "include/rayd/edge/optix_contracts.h"
SBT_CONTRACT = ROOT / "include/rayd/rt/optix_sbt.h"
PIPELINE_CONTRACT = ROOT / "include/rayd/rt/optix_pipeline_contracts.h"
EDGE_DEVICE = ROOT / "include/rayd/edge/optix_device.cuh"
SCENE_DEVICE = ROOT / "include/rayd/scene/optix_device.cuh"
DRJIT_EDGE = ROOT / "src/edge/edge_optix_jit.cu"
TORCH_EDGE = ROOT / "src/edge/edge_optix.cu"
TORCH_SCENE = ROOT / "src/scene/intersection_optix.cu"
DRJIT_SCENE = ROOT / "src/scene/scene_jit.cpp"


class Share4SceneEdgeOptixContractsTests(unittest.TestCase):
    def test_contracts_are_backend_neutral_and_pod(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in (SCENE_CONTRACT, EDGE_CONTRACT, SBT_CONTRACT))
        self.assertNotIn("rayd/torch", combined)
        self.assertNotIn("rayd/drjit", combined)
        self.assertNotIn("#include <optix.h>", combined)
        for type_name in ("SceneIntersectionPayload", "EmptySbtRecord"):
            self.assertIn(type_name, combined)
            self.assertIn(f"std::is_standard_layout_v<{type_name}>", combined)
        for type_name in ("EdgeGeometrySoAView", "EdgeQuerySoAView", "EdgeQueryOutputView"):
            self.assertIn(type_name, combined)
            self.assertIn(f"RAYD_SHARED_EDGE_OPTIX_ASSERT_POD({type_name})", combined)

    def test_payload_and_sbt_abi_are_explicit(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in (SCENE_CONTRACT, EDGE_CONTRACT, SBT_CONTRACT))
        for token in (
            "SbtRecordAlignment = 16u",
            "SbtRecordHeaderSize = 32u",
            "EdgePayloadTopKMax = 8",
            "SceneIntersectionPayloadSlot::Count) == 5u",
            "SceneHitObjectFieldSlot::Count) == 6u",
            "EdgePointPayloadSlot::Count) == 4u",
            "EdgeRayPayloadSlot::CommonCount) == 4u",
            "DrJitEdgeRayPayloadSlot::Valid) == 4u",
            "TorchEdgeRayPayloadSlot::TierRadius) == 4u",
        ):
            self.assertIn(token, combined)

    def test_pipeline_compile_counts_derive_from_shared_contract(self):
        source = PIPELINE_CONTRACT.read_text(encoding="utf-8")
        for token in (
            "SceneIntersectionPayloadCount = 5u",
            "TriangleHitPayloadCount = 6u",
            "VisibilityPayloadCount = 3u",
            "DiffractionPayloadCount = 4u",
            "EdgePointRayPayloadCount = 5u",
            "EdgeTopKPayloadCount = 16u",
            "TriangleAttributeCount = 2u",
            "EdgeAttributeCount = 3u",
        ):
            self.assertIn(token, source)

        consumers = tuple(
            ROOT / path
            for path in (
                "src/runtime/runtime_jit.cpp",
                "src/runtime/optix.cpp",
                "src/visibility/visibility.cpp",
                "src/reflection/reflection.cpp",
                "src/diffraction/diffraction.cpp",
            )
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in consumers)
        self.assertGreaterEqual(combined.count("shared::optix::"), 16)
        self.assertNotRegex(combined, r"numPayloadValues\s*=\s*(?:5|6|16)\s*;")
        self.assertNotRegex(combined, r"num_payload_values\s*=\s*(?:3|4|6)\s*;")

    def test_edge_programs_use_shared_device_helpers(self):
        for path in (DRJIT_EDGE, TORCH_EDGE):
            source = path.read_text(encoding="utf-8")
            self.assertIn("rayd/edge/optix_device.cuh", source)
            for symbol in (
                "shared::optix::edge_query_active",
                "shared::optix::edge_geometry_active",
                "shared::optix::write_invalid_edge_result",
                "shared::optix::set_edge_point_payload",
                "shared::optix::insert_edge_topk_payload_candidate",
            ):
                self.assertIn(symbol, source)
            self.assertNotIn("get_topk_payload_id", source)
            self.assertNotIn("set_topk_payload_slot", source)

    def test_backend_specific_edge_outer_params_remain_adapters(self):
        drjit_params = (ROOT / "include/rayd/jit/edge_optix_params.h").read_text(encoding="utf-8")
        torch_params = (ROOT / "src/edge/optix_params.h").read_text(encoding="utf-8")
        self.assertIn("struct EdgeOptixQueryParams", drjit_params)
        self.assertIn("struct EdgeOptixQueryParams", torch_params)
        self.assertNotIn("tier_handles", drjit_params)
        self.assertIn("tier_handles", torch_params)
        self.assertIn("write_point_outputs", torch_params)

    def test_scene_payload_and_hitobject_order_share_one_contract(self):
        torch_scene = TORCH_SCENE.read_text(encoding="utf-8")
        drjit_scene = DRJIT_SCENE.read_text(encoding="utf-8")
        self.assertIn("rayd/scene/optix_device.cuh", torch_scene)
        self.assertIn("shared::optix::SceneIntersectionPayload", torch_scene)
        self.assertIn("shared::optix::set_scene_intersection_payload", torch_scene)
        self.assertIn("SceneHitObjectFieldSlot::Count", drjit_scene)
        self.assertIn("OptixHitObjectField::Attribute0", drjit_scene)
        self.assertIn("OptixHitObjectField::InstanceId", drjit_scene)


if __name__ == "__main__":
    unittest.main()
