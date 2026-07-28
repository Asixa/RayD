import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFLECTION = ROOT / "include/rayd/shared/reflection"
VISIBILITY = ROOT / "include/rayd/shared/visibility"


class Share4SharedOptixContractsTests(unittest.TestCase):
    def test_shared_contract_headers_are_backend_neutral_pods(self):
        for path, struct_name in (
            (REFLECTION / "trace_params.h", "ReflectionTraceParams"),
            (VISIBILITY / "segment_params.h", "SegmentVisibilityParams"),
            (REFLECTION / "epc_params.h", "ReflEpcParams"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertIn(f"struct {struct_name}", source)
            self.assertIn(f"std::is_standard_layout_v<{struct_name}>", source)
            self.assertIn(f"std::is_trivially_copyable_v<{struct_name}>", source)
            self.assertNotRegex(source, re.compile(r"rayd/(torch|multipath|optix\.h)"))
            self.assertNotIn("at::Tensor", source)
            self.assertNotIn("drjit", source.lower())

    def test_backend_headers_are_thin_shared_aliases(self):
        headers = (
            ("src/reflection/trace_params_jit.h", "rayd/shared/reflection/", "ReflectionTraceParams"),
            ("src/reflection/trace_params.h", "rayd/shared/reflection/", "ReflectionTraceParams"),
            ("src/visibility/segment_params_jit.h", "rayd/shared/visibility/", "SegmentVisibilityParams"),
            ("src/visibility/visibility_params.h", "rayd/shared/visibility/", "SegmentVisibilityParams"),
            ("src/reflection/epc_params_jit.h", "rayd/shared/reflection/", "ReflEpcParams"),
            ("src/reflection/epc_params.h", "rayd/shared/reflection/", "ReflEpcParams"),
        )
        for relative, owner, type_name in headers:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(owner, source)
            self.assertIn(f"using {type_name} = shared::optix::{type_name};", source)
            self.assertNotIn(f"struct {type_name}", source)

    def test_shared_trace_superset_preserves_backend_specific_optional_fields(self):
        source = (REFLECTION / "trace_params.h").read_text(encoding="utf-8")
        for field in (
            "tri_p0_x", "tri_p0_packed", "ray_ox", "ray_o_aos", "out_bary_u",
            "out_bary", "output_layout", "out_valid", "out_trailing_origin_z",
        ):
            self.assertRegex(source, rf"\b{field}\b")

        visibility = (VISIBILITY / "segment_params.h").read_text(encoding="utf-8")
        for field in ("start_aos", "start_x", "sample_fractions", "out_first_blocked_prim", "out_t"):
            self.assertRegex(visibility, rf"\b{field}\b")

    def test_device_programs_have_one_shared_implementation(self):
        helper = (REFLECTION / "optix_hit.h").read_text(encoding="utf-8")
        for symbol in (
            "TriangleHitPayload", "VisibilityPayload", "clear_triangle_hit",
            "set_triangle_hit_payload", "choose_nearest_hit",
        ):
            self.assertIn(symbol, helper)

        primitive_id = (
            ROOT / "include/rayd/shared/rt/optix_primitive_id.h"
        ).read_text(encoding="utf-8")
        self.assertIn("global_primitive_id", primitive_id)
        self.assertIn("rayd/shared/rt/optix_primitive_id.h", helper)
        self.assertIn(
            "rayd/shared/rt/optix_primitive_id.h",
            (VISIBILITY / "segment_optix_device.cuh").read_text(encoding="utf-8"),
        )

        shared_programs = (
            (REFLECTION / "trace_optix_device.cuh", "reflection_trace_raygen"),
            (REFLECTION / "epc_optix_device.cuh", "run_reflection_epc_raygen"),
            (VISIBILITY / "segment_optix_device.cuh", "raygen_segment_chain"),
        )
        for path, entry in shared_programs:
            source = path.read_text(encoding="utf-8")
            self.assertIn(entry, source)

        reflection_trace = (REFLECTION / "trace_optix_device.cuh").read_text(encoding="utf-8")
        self.assertIn("rayd/shared/reflection/optix_traverser.h", reflection_trace)
        self.assertIn("OptixTraverser", reflection_trace)
        traverser = (REFLECTION / "optix_traverser.h").read_text(encoding="utf-8")
        self.assertIn("optixTrace", traverser)
        for path in (REFLECTION / "epc_optix_device.cuh", VISIBILITY / "segment_optix_device.cuh"):
            self.assertIn("optixTrace", path.read_text(encoding="utf-8"))

        adapters = (
            ("src/reflection/trace_optix_jit.cu", "rayd/shared/reflection/trace_optix_device.cuh"),
            ("src/reflection/trace_optix.cu", "rayd/shared/reflection/trace_optix_device.cuh"),
            ("src/reflection/epc_optix_jit.cu", "rayd/shared/reflection/epc_optix_device.cuh"),
            ("src/reflection/epc_optix.cu", "rayd/shared/reflection/epc_optix_device.cuh"),
            ("src/visibility/visibility_optix_jit.cu", "rayd/shared/visibility/segment_optix_device.cuh"),
            ("src/visibility/visibility_optix.cu", "rayd/shared/visibility/segment_optix_device.cuh"),
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
        for header in (
            "trace_optix_device.cuh",
            "epc_optix_device.cuh",
            "segment_optix_device.cuh",
        ):
            self.assertGreaterEqual(drjit_cmake.count(header), 1)
            self.assertGreaterEqual(torch_cmake.count(header), 1)

    def test_shared_headers_do_not_take_host_pipeline_ownership(self):
        paths = tuple(REFLECTION.glob("*.h")) + tuple(REFLECTION.glob("*.cuh")) + tuple(VISIBILITY.glob("*.h")) + tuple(VISIBILITY.glob("*.cuh"))
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "OptixPipeline", "OptixModule", "OptixProgramGroup", "optixPipelineCreate",
            "cudaMalloc", "cudaFree", "cudaDeviceSynchronize", "at::Tensor", "CudaBuffer",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()