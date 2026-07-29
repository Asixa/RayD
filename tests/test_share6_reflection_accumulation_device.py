from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "include/rayd/detail/reflection/accumulation_optix_device.cuh"
ALGO = ROOT / "include/rayd/detail/reflection/accumulation_algo.h"
DRJIT = ROOT / "src/reflection/accumulation_optix_jit.cu"
TORCH = ROOT / "src/reflection/accumulation_optix.cu"


class SharedReflectionAccumulationDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shared = SHARED.read_text(encoding="utf-8")
        cls.algo = ALGO.read_text(encoding="utf-8")
        cls.drjit = DRJIT.read_text(encoding="utf-8")
        cls.torch = TORCH.read_text(encoding="utf-8")

    def test_shared_header_owns_complete_device_operation(self) -> None:
        # Since P4c the algorithm body lives in the host-compilable algo header;
        # the device header keeps only the OptiX entry layer and delegates.
        algorithm_tokens = (
            "trace_scene(",
            "Complex3 reflect_field_vector(",
            "void store_wedge_event(",
            "bool accumulate_plane(",
            "for (int depth = 0; depth <= params.max_bounces; ++depth)",
            "Policy::include_depth(params, depth)",
            "Policy::commit(",
        )
        for token in algorithm_tokens:
            self.assertIn(token, self.algo)
            self.assertNotIn(token, self.drjit)
            self.assertNotIn(token, self.torch)
        entry_tokens = (
            "void closest_hit()",
            "void miss()",
            "void raygen(const Params &params)",
        )
        for token in entry_tokens:
            self.assertIn(token, self.shared)
        self.assertIn("accumulation_algo.h", self.shared)
        # The former file-local HitPayload duplication dissolved into the
        # canonical shared types; forbid it from reappearing.
        self.assertIn("TriangleHit", self.algo)
        for text in (self.algo, self.shared):
            self.assertNotIn("struct HitPayload", text)

    def test_adapters_are_only_params_policy_and_entry_wrappers(self) -> None:
        forbidden = (
            "HitPayload trace_scene(",
            "Complex3 reflect_field_vector(",
            "void store_wedge_event(",
            "bool accumulate_plane(",
            "for (int depth = 0; depth <= params.max_bounces; ++depth)",
        )
        for source in (self.drjit, self.torch):
            for token in forbidden:
                self.assertNotIn(token, source)
            self.assertLess(len(source.splitlines()), 100)
            self.assertEqual(source.count("__constant__ AccumParams params"), 1)

    def test_optix_entry_identity_is_shared(self) -> None:
        entry_pattern = re.compile(
            r'extern "C" __global__ void (__\w+__reflection_accumulation)\(\)'
        )
        expected = [
            "__closesthit__reflection_accumulation",
            "__miss__reflection_accumulation",
            "__raygen__reflection_accumulation",
        ]
        for source in (self.drjit, self.torch):
            self.assertEqual(entry_pattern.findall(source), expected)
            self.assertIn("shared_accum::closest_hit();", source)
            self.assertIn("shared_accum::miss();", source)
            self.assertIn("shared_accum::raygen<AccumParams, ReflectionAccumulationPolicy>(params);", source)

    def test_policy_preserves_backend_specific_semantics(self) -> None:
        self.assertIn("return depth > 0;", self.drjit)
        self.assertIn("return depth > 0 || params.include_los != 0;", self.torch)
        self.assertIn("params.stage_cell != nullptr", self.torch)
        self.assertIn("ReflAccumStagedValue value", self.torch)
        self.assertIn("const WarpCellGroup group = warp_cell_group(cell);", self.torch)
        self.assertIn("atomic_add_same_cell", self.torch)
        self.assertIn("atomic_add_warp", self.torch)
        self.assertIn("atomicAdd(params.out_field_x_re + cell", self.drjit)
        self.assertNotIn("include_los", self.shared)
        self.assertNotIn("stage_cell", self.shared)
        self.assertNotIn("WarpCellGroup", self.shared)

    def test_shared_core_adds_no_resource_or_launch_ownership(self) -> None:
        for token in (
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "<<<",
        ):
            self.assertNotIn(token, self.shared)

    def test_params_abi_headers_are_unchanged_and_backend_local(self) -> None:
        self.assertNotIn("reflection_accumulation_params.h", self.shared)
        self.assertNotIn("torch/reflection/accum_params.h", self.shared)
        self.assertIn(
            "#include <src/reflection/accumulation_params_jit.h>", self.drjit
        )
        self.assertIn(
            "#include <src/reflection/accum_params.h>", self.torch
        )


if __name__ == "__main__":
    unittest.main()
