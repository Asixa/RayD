# Copyright Xingyu Chen.
# Tests share4 shared optix contracts.

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFLECTION = ROOT / "src/reflection"
VISIBILITY = ROOT / "src/visibility"


class Share4SharedOptixContractsTests(unittest.TestCase):
    def test_shared_contract_headers_are_backend_neutral_pods(self):
        for path, struct_name in (
            (REFLECTION / "reflection_internal.h", "ReflectionTraceParams"),
            (VISIBILITY / "segment_visibility.cuh", "SegmentVisibilityParams"),
            (REFLECTION / "reflection_internal.h", "ReflEpcParams"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertIn(f"struct {struct_name}", source)
            self.assertIn(f"std::is_standard_layout_v<{struct_name}>", source)
            self.assertIn(f"std::is_trivially_copyable_v<{struct_name}>", source)
            self.assertNotRegex(source, re.compile(r"rayd/(torch|multipath|optix\.h)"))
            self.assertNotIn("at::Tensor", source)
            self.assertNotIn("drjit", source.lower())

    def test_backend_headers_are_thin_shared_aliases(self):
        reflection = (REFLECTION / "reflection_internal.h").read_text(encoding="utf-8")
        self.assertEqual(reflection.count("struct ReflectionTraceParams"), 1)
        self.assertEqual(reflection.count("struct ReflEpcParams"), 1)
        self.assertEqual(reflection.count("using ReflectionTraceParams = shared::optix::ReflectionTraceParams;"), 2)
        self.assertEqual(reflection.count("using ReflEpcParams = shared::optix::ReflEpcParams;"), 2)
        self.assertNotIn("rayd/reflection/", reflection)

        for relative in ("src/visibility/segment_params_jit.h", "src/visibility/visibility_params.h"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("src/visibility/segment_visibility.cuh", source)
            self.assertIn("using SegmentVisibilityParams = shared::optix::SegmentVisibilityParams;", source)
            self.assertNotIn("struct SegmentVisibilityParams", source)

    def test_shared_trace_superset_preserves_backend_specific_optional_fields(self):
        source = (REFLECTION / "reflection_internal.h").read_text(encoding="utf-8")
        for field in (
            "tri_p0_x",
            "tri_p0_packed",
            "ray_ox",
            "ray_o_aos",
            "out_bary_u",
            "out_bary",
            "output_layout",
            "out_valid",
            "out_trailing_origin_z",
        ):
            self.assertRegex(source, rf"\b{field}\b")

        visibility = (VISIBILITY / "segment_visibility.cuh").read_text(encoding="utf-8")
        for field in ("start_aos", "start_x", "sample_fractions", "out_first_blocked_prim", "out_t"):
            self.assertRegex(visibility, rf"\b{field}\b")

    def test_device_programs_have_one_shared_implementation(self):
        helper = (REFLECTION / "reflection_optix_common.cuh").read_text(encoding="utf-8")
        for symbol in (
            "TriangleHitPayload",
            "VisibilityPayload",
            "clear_triangle_hit",
            "set_triangle_hit_payload",
            "choose_nearest_hit",
        ):
            self.assertIn(symbol, helper)

        primitive_id = (ROOT / "src/runtime/rt_device.cuh").read_text(encoding="utf-8")
        self.assertIn("global_primitive_id", primitive_id)
        self.assertIn("src/runtime/rt_device.cuh", helper)
        self.assertIn("src/runtime/rt_device.cuh", (VISIBILITY / "segment_visibility.cuh").read_text(encoding="utf-8"))

        shared_programs = (
            (REFLECTION / "reflection_trace_optix.cuh", "reflection_trace_raygen"),
            (REFLECTION / "reflection_epc_optix.cuh", "run_reflection_epc_raygen"),
            (VISIBILITY / "segment_visibility.cuh", "raygen_segment_chain"),
        )
        for path, entry in shared_programs:
            source = path.read_text(encoding="utf-8")
            self.assertIn(entry, source)

        reflection_trace = (REFLECTION / "reflection_trace_optix.cuh").read_text(encoding="utf-8")
        self.assertIn("src/reflection/reflection_optix_common.cuh", reflection_trace)
        self.assertIn("OptixTraverser", reflection_trace)
        traverser = (REFLECTION / "reflection_optix_common.cuh").read_text(encoding="utf-8")
        self.assertIn("optixTrace", traverser)
        for path in (REFLECTION / "reflection_epc_optix.cuh", VISIBILITY / "segment_visibility.cuh"):
            self.assertIn("optixTrace", path.read_text(encoding="utf-8"))

        adapters = (
            ("src/reflection/trace_optix_jit.cu", "src/reflection/reflection_trace_optix.cuh"),
            ("src/reflection/trace_optix.cu", "src/reflection/reflection_trace_optix.cuh"),
            ("src/reflection/epc_optix_jit.cu", "src/reflection/reflection_epc_optix.cuh"),
            ("src/reflection/epc_optix.cu", "src/reflection/reflection_epc_optix.cuh"),
            ("src/visibility/visibility_optix_jit.cu", "src/visibility/segment_visibility.cuh"),
            ("src/visibility/visibility_optix.cu", "src/visibility/segment_visibility.cuh"),
        )
        for relative, shared_header in adapters:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(shared_header, source)
            self.assertIn("__constant__", source)
            self.assertNotIn("optixTrace", source)
            self.assertNotIn("struct HitPayload {", source)
            self.assertNotIn("struct VisibilityPayload {", source)

    def test_backend_policies_preserve_existing_device_semantics(self):
        drjit_trace = (ROOT / "src/reflection/trace_optix_jit.cu").read_text(encoding="utf-8")
        torch_trace = (ROOT / "src/reflection/trace_optix.cu").read_text(encoding="utf-8")
        self.assertIn("DrJitReflectionTracePolicy", drjit_trace)
        self.assertIn("TorchReflectionTracePolicy", torch_trace)

        drjit_visibility = (ROOT / "src/visibility/visibility_optix_jit.cu").read_text(encoding="utf-8")
        torch_visibility = (ROOT / "src/visibility/visibility_optix.cu").read_text(encoding="utf-8")
        self.assertIn("SegmentVisibilityDevicePolicy<false, false>", drjit_visibility)
        self.assertIn("SegmentVisibilityDevicePolicy<true, true>", torch_visibility)

        drjit_epc = (ROOT / "src/reflection/epc_optix_jit.cu").read_text(encoding="utf-8")
        torch_epc = (ROOT / "src/reflection/epc_optix.cu").read_text(encoding="utf-8")
        self.assertIn("DisableAnyHitWithoutIgnore = false", drjit_epc)
        self.assertIn("DisableAnyHitWithoutIgnore = true", torch_epc)

    def test_ptx_builds_depend_on_shared_device_programs(self):
        drjit_cmake = (ROOT / "drjit/CMakeLists.txt").read_text(encoding="utf-8")
        torch_cmake = (ROOT / "torch/CMakeLists.txt").read_text(encoding="utf-8")
        for header in ("reflection_trace_optix.cuh", "reflection_epc_optix.cuh", "segment_visibility.cuh"):
            self.assertGreaterEqual(drjit_cmake.count(header), 1)
            self.assertGreaterEqual(torch_cmake.count(header), 1)

    def test_shared_headers_do_not_take_host_pipeline_ownership(self):
        paths = tuple(
            REFLECTION / name
            for name in (
                "reflection_internal.h",
                "reflection_algorithms.cuh",
                "reflection_optix_common.cuh",
                "reflection_trace_optix.cuh",
                "reflection_accumulation_optix.cuh",
                "reflection_epc_optix.cuh",
                "epc_field_fragment.cuh",
            )
        ) + (VISIBILITY / "segment_visibility.cuh",)
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "OptixPipeline",
            "OptixModule",
            "OptixProgramGroup",
            "optixPipelineCreate",
            "cudaMalloc",
            "cudaFree",
            "cudaDeviceSynchronize",
            "at::Tensor",
            "CudaBuffer",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
