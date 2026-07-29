# Copyright Xingyu Chen.
# Tests share6 diffraction accumulation device.

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "include/rayd/diffraction/accumulation_optix_device.cuh"
ALGO = ROOT / "include/rayd/diffraction/accumulation_algo.h"
REFLECTION = ROOT / "include/rayd/reflection/accumulation_optix_device.cuh"
DRJIT = ROOT / "src/diffraction/accumulation_optix_jit.cu"
TORCH = ROOT / "src/diffraction/accumulation_optix.cu"
COEXIST = ROOT / "tests/native/share6_accumulation_headers_coexist.cu"
ACCEPTANCE = (
    ROOT
    / "benchmarks/baselines/share6_diffraction_accumulation_20260711.json"
)


class SharedDiffractionAccumulationDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shared = SHARED.read_text(encoding="utf-8")
        cls.algo = ALGO.read_text(encoding="utf-8")
        cls.reflection = REFLECTION.read_text(encoding="utf-8")
        cls.drjit = DRJIT.read_text(encoding="utf-8")
        cls.torch = TORCH.read_text(encoding="utf-8")

    def test_shared_header_owns_complete_operations(self) -> None:
        # Since P4c the algorithm body lives in the host-compilable algo header;
        # the device header keeps only the OptiX entry layer and delegates.
        algorithm_tokens = (
            "trace_scene_impl(",
            "visible_segment_impl(",
            "run_coherent_utd_lane(",
            "suffix_reflection_connection(",
            "diffraction_weight(",
        )
        for token in algorithm_tokens:
            self.assertIn(token, self.algo)
            self.assertNotIn(token, self.drjit)
            self.assertNotIn(token, self.torch)
        entry_tokens = (
            "run_diffraction_order1_accumulation_raygen()",
            "run_diffraction_order1_source_visibility_raygen()",
            "run_diffraction_order1_no_suffix_target_accumulation_raygen()",
            "run_diffraction_order1_suffix_first_visibility_raygen()",
            "run_diffraction_order1_suffix_target_accumulation_raygen()",
            "run_diffraction_order1_coherent_accumulation_raygen()",
            "run_diffraction_chain_accumulation_raygen()",
        )
        for token in entry_tokens:
            self.assertIn(token, self.shared)
        self.assertIn("accumulation_algo.h", self.shared)

    def test_hit_payload_is_the_canonical_triangle_hit(self) -> None:
        # The former file-local 4-field HitPayload dissolved into rt::TriangleHit;
        # forbid it from reappearing anywhere in the pipeline pair.
        self.assertIn("TriangleHit", self.algo)
        for text in (self.algo, self.shared):
            self.assertNotIn("struct HitPayload", text)

    def test_adapters_only_own_params_policy_and_entries(self) -> None:
        forbidden = (
            "trace_scene_impl(",
            "visible_segment_impl(",
            "run_coherent_utd_lane(",
            "suffix_reflection_connection(",
            "diffraction_weight(",
        )
        for source in (self.drjit, self.torch):
            for token in forbidden:
                self.assertNotIn(token, source)
            self.assertIn("struct DiffractionAccumulationPolicy", source)
            definitions = re.findall(
                r"(?m)^__constant__ DfrAccumParams params;$", source
            )
            self.assertEqual(len(definitions), 1)
            self.assertLess(len(source.splitlines()), 400)

    def test_optix_entry_names_and_order_are_identical(self) -> None:
        pattern = re.compile(r'extern "C" __global__ void\s+(__\w+)\(\)')
        expected = pattern.findall(self.drjit)
        self.assertEqual(pattern.findall(self.torch), expected)
        self.assertEqual(len(expected), 16)
        self.assertEqual(len(set(expected)), 16)
        self.assertIn("__raygen__diffraction_order1_accumulation", expected)
        self.assertIn("__raygen__diffraction_order1_coherent_accumulation", expected)
        self.assertIn("__raygen__diffraction_chain_accumulation", expected)
        # Both adapters dispatch all 14 raygen entries into the shared device
        # struct and own no algorithm body. Dr.Jit uses its raygen wrappers,
        # which read the launch index themselves; the Torch adapter enters the
        # same struct through make_algo() because it offsets the launch index
        # into the global Monte-Carlo lane space first (DfrAccumParams::
        # lane_offset), which the shared wrappers cannot express.
        self.assertEqual(self.drjit.count("Device::run_diffraction_"), 14)
        self.assertEqual(self.torch.count("Device::make_algo()"), 14)
        self.assertEqual(self.torch.count("_algo<"), 14)
        self.assertEqual(self.torch.count("params.lane_offset"), 1)
        for source in (self.drjit, self.torch):
            self.assertIn("Device::closesthit();", source)
            self.assertIn("Device::miss();", source)

    def test_policy_preserves_backend_extensions(self) -> None:
        compact_drjit = re.sub(r"\s+", " ", self.drjit)
        self.assertIn(
            "lane % static_cast<unsigned int>(params().state_count)",
            compact_drjit,
        )
        self.assertIn("return edge_length /", compact_drjit)
        self.assertIn("atomicAdd(base + i, value);", self.drjit)
        self.assertIn("return false;", self.drjit)
        for token in (
            "sample_state_index_stride",
            "sample_edge_weight_stride",
            "active_width",
            "coherent_stage_key",
            "stage_value",
            "warp_cell_group",
        ):
            self.assertIn(token, self.torch)
            self.assertNotIn(token, self.shared)

    def test_shared_core_adds_no_runtime_ownership(self) -> None:
        for token in (
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "<<<",
        ):
            self.assertNotIn(token, self.shared)
            self.assertNotIn(token, self.algo)

    def test_reflection_and_diffraction_namespaces_coexist(self) -> None:
        self.assertIn(
            "namespace rayd::shared::multipath::diffraction_accumulation",
            self.shared,
        )
        self.assertIn(
            "namespace rayd::shared::multipath::reflection_accumulation",
            self.reflection,
        )
        coexist = COEXIST.read_text(encoding="utf-8")
        self.assertIn("diffraction/accumulation_optix_device.cuh", coexist)
        self.assertIn("reflection/accumulation_optix_device.cuh", coexist)

    def test_acceptance_record_is_machine_readable(self) -> None:
        record = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        self.assertEqual(record["phase"], "Share-6 diffraction accumulation operation core")
        self.assertFalse(record["implementation"]["params_abi_changed"])
        self.assertEqual(record["implementation"]["new_launches"], 0)
        self.assertEqual(record["implementation"]["new_synchronizations"], 0)
        self.assertEqual(record["implementation"]["new_global_buffers"], 0)


if __name__ == "__main__":
    unittest.main()
