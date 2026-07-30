# Copyright Xingyu Chen.
# Tests source lane diffraction paths.

from __future__ import annotations

from pathlib import Path

from tests.support.source_inspection import read_text as read, struct_body
import unittest


ROOT = Path(__file__).resolve().parents[1]
RAYD_INCLUDE = ROOT / "include" / "rayd"
TORCH_SOURCE = ROOT / "src"


class Adr0032SourceLaneDiffractionPathTests(unittest.TestCase):
    def test_api6_exposes_stable_layout_without_a_new_boundary(self):
        integration = read(RAYD_INCLUDE / "integration.h")
        self.assertIn("kIntegrationApiVersion = 8", integration)
        self.assertIn('"rayd.torch.integration"', integration)
        diffraction = read(RAYD_INCLUDE / "diffraction.h")
        self.assertIn("enum class DiffractionPathLayout", diffraction)
        enum = diffraction.split("enum class DiffractionPathLayout", 1)[1].split("};", 1)[0]
        self.assertIn("Compact = 0", enum)
        self.assertIn("SourceLane = 1", enum)
        config = struct_body(diffraction, "DiffractionPathConfig")
        self.assertIn("DiffractionPathLayout layout = DiffractionPathLayout::Compact;", config)
        self.assertFalse((RAYD_INCLUDE / "integration" / "torch_v2.h").exists())

    def test_typed_impl_validates_and_threads_layout(self):
        params = read(ROOT / "src" / "diffraction" / "paths_params.h")
        self.assertIn("int output_layout;", struct_body(params, "DfrPathParams"))

        ops = read(TORCH_SOURCE / "diffraction" / "diffraction.cpp")
        body = ops.split("DiffractionPathOutputs diffraction_paths_order1_forward_impl", 1)[1]
        body = body.split("py::tuple diffraction_path_outputs_to_tuple", 1)[0]
        self.assertIn("int output_layout", body)
        self.assertIn("output_layout != kDiffractionPathLayoutCompact", body)
        self.assertIn("output_layout != kDiffractionPathLayoutSourceLane", body)
        self.assertIn("params.output_layout = output_layout", body)
        dispatcher = ops.split("py::tuple diffraction_paths_order1_forward_op", 1)[1]
        dispatcher = dispatcher.split("struct DiffractionAccumulationOutputs", 1)[0]
        self.assertIn('checked_i32(output_layout, "output_layout")', dispatcher)
        typed = ops.split("DiffractionPathResult diffraction_paths_order1_forward", 1)[1]
        self.assertIn("static_cast<int>(config.layout)", typed)

    def test_shared_exporter_uses_lane_only_for_source_lane(self):
        shared = read(TORCH_SOURCE / "diffraction" / "paths.h")
        reserve = shared.split("RAYD_DEVICE int reserve_path_output", 1)[1]
        reserve = reserve.split("/// Combined first-order", 1)[0]
        self.assertIn("path_output_layout(params, 0) == kOutputLayoutSourceLane", reserve)
        self.assertIn("atomic_add(params.out_count, 1);", reserve)
        self.assertIn("return static_cast<int>(lane);", reserve)
        self.assertIn("return atomic_add(params.out_count, 1);", reserve)
        self.assertEqual(shared.count("reserve_path_output(params, lane)"), 2)

    def test_live_torch_raygen_uses_source_lane_reservation(self):
        params = read(ROOT / "src" / "diffraction" / "paths_params.h")
        self.assertIn("kDiffractionPathLayoutCompact = 0", params)
        self.assertIn("kDiffractionPathLayoutSourceLane = 1", params)

        optix = read(TORCH_SOURCE / "diffraction" / "paths_optix.cu")
        reserve = optix.split("static __forceinline__ __device__ int reserve_path_slot", 1)[1].split(
            "} // namespace", 1
        )[0]
        self.assertIn("params.output_layout == kDiffractionPathLayoutSourceLane", reserve)
        self.assertIn("base = atomicAdd(params.out_count, count);", reserve)
        self.assertIn("return static_cast<int>(logical_lane);", reserve)
        self.assertIn("const unsigned int mask = __activemask();", reserve)
        self.assertEqual(optix.count("reserve_path_slot(lane)"), 2)

    def test_direct_contract_covers_sparse_fixed_lane(self):
        direct = read(ROOT / "tests" / "native" / "integration_test.cpp")
        self.assertIn("DiffractionPathLayout::SourceLane", direct)
        self.assertIn("source-lane export must preserve the pair/state lane index", direct)
        self.assertIn("compact and source-lane payloads must be bit-identical", direct)
        self.assertIn("source-lane inactive identity must remain canonical", direct)
        self.assertIn("invalid diffraction path layout must fail loudly", direct)

    def test_governance_files_are_identical_and_link_decision(self):
        self.assertEqual((ROOT / "AGENTS.md").read_bytes(), (ROOT / "CLAUDE.md").read_bytes())
        link = "0032-source-lane-diffraction-path-layout.md"
        self.assertIn(link, read(ROOT / "AGENTS.md"))
        self.assertIn(link, read(ROOT / "torch" / "README.md"))


if __name__ == "__main__":
    unittest.main()
