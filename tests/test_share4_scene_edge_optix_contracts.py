from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "include/rayd/shared/optix/scene_edge_contracts.h"
PIPELINE_CONTRACT = ROOT / "include/rayd/shared/optix/pipeline_contracts.h"
DEVICE = ROOT / "include/rayd/shared/optix/scene_edge_device.cuh"
DRJIT_EDGE = ROOT / "src/edge/edge_optix_jit.cu"
TORCH_EDGE = ROOT / "src/edge/edge_optix.cu"
TORCH_SCENE = ROOT / "src/scene/intersection_optix.cu"
DRJIT_SCENE = ROOT / "src/scene/scene_jit.cpp"


class Share4SceneEdgeOptixContractsTests(unittest.TestCase):
    def test_contract_is_backend_neutral_and_pod(self):
        source = CONTRACT.read_text(encoding="utf-8")
        self.assertNotIn("rayd/torch", source)
        self.assertNotIn("rayd/drjit", source)
        self.assertNotIn("optix.h", source)
        for type_name in (
            "SceneIntersectionPayload",
            "EdgeGeometrySoAView",
            "EdgeQuerySoAView",
            "EdgeQueryOutputView",
            "EmptySbtRecord",
        ):
            self.assertIn(f"RAYD_SHARED_SCENE_EDGE_ASSERT_POD({type_name})", source)

    def test_payload_and_sbt_abi_are_explicit(self):
        source = CONTRACT.read_text(encoding="utf-8")
        self.assertIn("SbtRecordAlignment = 16u", source)
        self.assertIn("SbtRecordHeaderSize = 32u", source)
        self.assertIn("EdgePayloadTopKMax = 8", source)
        self.assertIn("SceneIntersectionPayloadSlot::Count) == 5u", source)
        self.assertIn("SceneHitObjectFieldSlot::Count) == 6u", source)
        self.assertIn("EdgePointPayloadSlot::Count) == 4u", source)
        self.assertIn("EdgeRayPayloadSlot::CommonCount) == 4u", source)
        self.assertIn("DrJitEdgeRayPayloadSlot::Valid) == 4u", source)
        self.assertIn("TorchEdgeRayPayloadSlot::TierRadius) == 4u", source)

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

        consumers = (
            ROOT / "src/runtime/runtime_jit.cpp",
            ROOT / "src/runtime/optix.cpp",
            ROOT / "src/visibility/visibility.cpp",
            ROOT / "src/reflection/reflection.cpp",
            ROOT / "src/diffraction/diffraction.cpp",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in consumers)
        self.assertGreaterEqual(combined.count("shared::optix::"), 16)
        self.assertNotRegex(combined, r"numPayloadValues\s*=\s*(?:5|6|16)\s*;")
        self.assertNotRegex(combined, r"num_payload_values\s*=\s*(?:3|4|6)\s*;")

    def test_edge_programs_use_shared_device_helpers(self):
        for path in (DRJIT_EDGE, TORCH_EDGE):
            source = path.read_text(encoding="utf-8")
            self.assertIn("rayd/shared/optix/scene_edge_device.cuh", source)
            self.assertIn("shared::optix::edge_query_active", source)
            self.assertIn("shared::optix::edge_geometry_active", source)
            self.assertIn("shared::optix::write_invalid_edge_result", source)
            self.assertIn("shared::optix::set_edge_point_payload", source)
            self.assertIn("shared::optix::insert_edge_topk_payload_candidate", source)
            self.assertNotIn("get_topk_payload_id", source)
            self.assertNotIn("set_topk_payload_slot", source)

    def test_backend_specific_edge_outer_params_remain_adapters(self):
        drjit_params = (ROOT / "include/rayd/edge/edge_optix_params.h").read_text(
            encoding="utf-8"
        )
        torch_params = (ROOT / "include/rayd/torch/edge/optix_params.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("struct EdgeOptixQueryParams", drjit_params)
        self.assertIn("struct EdgeOptixQueryParams", torch_params)
        self.assertNotIn("tier_handles", drjit_params)
        self.assertIn("tier_handles", torch_params)
        self.assertIn("write_point_outputs", torch_params)

    def test_scene_payload_and_hitobject_order_share_one_contract(self):
        torch_scene = TORCH_SCENE.read_text(encoding="utf-8")
        drjit_scene = DRJIT_SCENE.read_text(encoding="utf-8")
        self.assertIn("shared::optix::SceneIntersectionPayload", torch_scene)
        self.assertIn("shared::optix::set_scene_intersection_payload", torch_scene)
        self.assertIn("SceneHitObjectFieldSlot::Count", drjit_scene)
        self.assertIn("OptixHitObjectField::Attribute0", drjit_scene)
        self.assertIn("OptixHitObjectField::InstanceId", drjit_scene)


if __name__ == "__main__":
    unittest.main()
