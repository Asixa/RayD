import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared/include/rayd/shared/optix"


class Share4SharedOptixContractsTests(unittest.TestCase):
    def test_shared_contract_headers_are_backend_neutral_pods(self):
        for name, struct_name in (
            ("reflection_trace_params.h", "ReflectionTraceParams"),
            ("segment_visibility_params.h", "SegmentVisibilityParams"),
            ("reflection_epc_params.h", "ReflEpcParams"),
        ):
            source = (SHARED / name).read_text(encoding="utf-8")
            self.assertIn(f"struct {struct_name}", source)
            self.assertIn(f"std::is_standard_layout_v<{struct_name}>", source)
            self.assertIn(f"std::is_trivially_copyable_v<{struct_name}>", source)
            self.assertNotRegex(source, re.compile(r"rayd/(torch|multipath|optix\.h)"))
            self.assertNotIn("at::Tensor", source)
            self.assertNotIn("drjit", source.lower())

    def test_backend_headers_are_thin_shared_aliases(self):
        headers = (
            ("backends/drjit/include/rayd/multipath/reflection_trace_params.h", "ReflectionTraceParams"),
            ("backends/torch/include/rayd/torch/reflection/trace_params.h", "ReflectionTraceParams"),
            ("backends/drjit/include/rayd/multipath/segment_visibility_params.h", "SegmentVisibilityParams"),
            ("backends/torch/include/rayd/torch/reflection/visibility_params.h", "SegmentVisibilityParams"),
            ("backends/drjit/include/rayd/multipath/reflection_epc_params.h", "ReflEpcParams"),
            ("backends/torch/include/rayd/torch/reflection/epc_params.h", "ReflEpcParams"),
        )
        for relative, type_name in headers:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("rayd/shared/optix/", source)
            self.assertIn(f"using {type_name} = shared::optix::{type_name};", source)
            self.assertNotIn(f"struct {type_name}", source)

    def test_shared_trace_superset_preserves_backend_specific_optional_fields(self):
        source = (SHARED / "reflection_trace_params.h").read_text(encoding="utf-8")
        for field in (
            "tri_p0_x", "tri_p0_packed", "ray_ox", "ray_o_aos", "out_bary_u",
            "out_bary", "output_layout", "out_valid", "out_trailing_origin_z",
        ):
            self.assertRegex(source, rf"\b{field}\b")

        visibility = (SHARED / "segment_visibility_params.h").read_text(encoding="utf-8")
        for field in ("start_aos", "start_x", "sample_fractions", "out_first_blocked_prim", "out_t"):
            self.assertRegex(visibility, rf"\b{field}\b")

    def test_device_programs_have_one_shared_implementation(self):
        helper = (SHARED / "device_hit.h").read_text(encoding="utf-8")
        for symbol in (
            "TriangleHitPayload", "VisibilityPayload", "clear_triangle_hit",
            "set_triangle_hit_payload", "choose_nearest_hit", "global_primitive_id",
        ):
            self.assertIn(symbol, helper)

        shared_programs = (
            ("reflection_trace_device.cuh", "reflection_trace_raygen"),
            ("reflection_epc_device.cuh", "run_reflection_epc_raygen"),
            ("segment_visibility_device.cuh", "raygen_segment_chain"),
        )
        for name, entry in shared_programs:
            source = (SHARED / name).read_text(encoding="utf-8")
            self.assertIn("rayd/shared/optix/device_hit.h", source)
            self.assertIn(entry, source)

        # P4 Stage A funnels the reflection trace's single optixTrace through the
        # shared OptixTraverser shim (still one shared implementation, one include
        # deeper); the entry header instantiates it. Not-yet-migrated pipelines
        # keep optixTrace inline in their device header.
        reflection_trace = (SHARED / "reflection_trace_device.cuh").read_text(encoding="utf-8")
        self.assertIn("rayd/shared/optix/optix_traverser.h", reflection_trace)
        self.assertIn("OptixTraverser", reflection_trace)
        traverser = (SHARED / "optix_traverser.h").read_text(encoding="utf-8")
        self.assertIn("optixTrace", traverser)
        for name in ("reflection_epc_device.cuh", "segment_visibility_device.cuh"):
            self.assertIn("optixTrace", (SHARED / name).read_text(encoding="utf-8"))

        adapters = (
            ("backends/drjit/src/multipath/reflection_trace.cu", "reflection_trace_device.cuh"),
            ("backends/torch/src/torch_ext/reflection/trace_optix.cu", "reflection_trace_device.cuh"),
            ("backends/drjit/src/multipath/reflection_epc.cu", "reflection_epc_device.cuh"),
            ("backends/torch/src/torch_ext/reflection/epc_optix.cu", "reflection_epc_device.cuh"),
            ("backends/drjit/src/multipath/segment_visibility.cu", "segment_visibility_device.cuh"),
            ("backends/torch/src/torch_ext/reflection/visibility_optix.cu", "segment_visibility_device.cuh"),
        )
        for relative, shared_header in adapters:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(f"rayd/shared/optix/{shared_header}", source)
            self.assertIn("__constant__", source)
            self.assertNotIn("optixTrace", source)
            self.assertNotIn("struct HitPayload {", source)
            self.assertNotIn("struct VisibilityPayload {", source)

    def test_backend_policies_preserve_existing_device_semantics(self):
        drjit_trace = (ROOT / "backends/drjit/src/multipath/reflection_trace.cu").read_text(
            encoding="utf-8"
        )
        torch_trace = (ROOT / "backends/torch/src/torch_ext/reflection/trace_optix.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("DrJitReflectionTracePolicy", drjit_trace)
        self.assertIn("TorchReflectionTracePolicy", torch_trace)

        drjit_visibility = (ROOT / "backends/drjit/src/multipath/segment_visibility.cu").read_text(
            encoding="utf-8"
        )
        torch_visibility = (ROOT / "backends/torch/src/torch_ext/reflection/visibility_optix.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("SegmentVisibilityDevicePolicy<false, false>", drjit_visibility)
        self.assertIn("SegmentVisibilityDevicePolicy<true, true>", torch_visibility)

        drjit_epc = (ROOT / "backends/drjit/src/multipath/reflection_epc.cu").read_text(
            encoding="utf-8"
        )
        torch_epc = (ROOT / "backends/torch/src/torch_ext/reflection/epc_optix.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("DisableAnyHitWithoutIgnore = false", drjit_epc)
        self.assertIn("DisableAnyHitWithoutIgnore = true", torch_epc)

    def test_ptx_builds_depend_on_shared_device_programs(self):
        drjit_cmake = (ROOT / "backends/drjit/CMakeLists.txt").read_text(encoding="utf-8")
        torch_cmake = (ROOT / "backends/torch/CMakeLists.txt").read_text(encoding="utf-8")
        for header in (
            "reflection_trace_device.cuh",
            "reflection_epc_device.cuh",
            "segment_visibility_device.cuh",
        ):
            self.assertGreaterEqual(drjit_cmake.count(header), 1)
            self.assertGreaterEqual(torch_cmake.count(header), 1)

    def test_shared_headers_do_not_take_host_pipeline_ownership(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for pattern in ("*.h", "*.cuh")
            for path in SHARED.glob(pattern)
        )
        for forbidden in (
            "OptixPipeline", "OptixModule", "OptixProgramGroup", "optixPipelineCreate",
            "cudaMalloc", "cudaFree", "cudaDeviceSynchronize", "at::Tensor", "CudaBuffer",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
