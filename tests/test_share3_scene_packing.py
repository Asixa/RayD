# Copyright Xingyu Chen.
# Tests share3 scene packing.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "rayd" / "scene" / "packing.h"
SOURCE = ROOT / "src" / "scene" / "packing_shared.cu"
TORCH_ADAPTER = ROOT / "src" / "scene" / "cache.cu"


class Share3ScenePackingTests(unittest.TestCase):
    def test_shared_contract_is_backend_neutral_and_enqueue_only(self):
        header = HEADER.read_text(encoding="utf-8")
        source = SOURCE.read_text(encoding="utf-8")
        combined = header + source
        for forbidden in (
            "at::Tensor",
            "torch/",
            "drjit",
            "nanobind",
            "cudaMalloc",
            "cudaFree",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "throw ",
            "std::vector",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertEqual(header.count("cudaStream_t stream;"), 3)
        self.assertEqual(header.count(") noexcept;"), 3)

    def test_contract_keeps_packed_float4_layout(self):
        header = HEADER.read_text(encoding="utf-8")
        source = SOURCE.read_text(encoding="utf-8")
        for required in (
            "struct alignas(16) PackedFloat4",
            "sizeof(PackedFloat4) == 4u * sizeof(float)",
            "alignof(PackedFloat4) == 4u * alignof(float)",
            "offsetof(PackedFloat4, w) == 3u * sizeof(float)",
            "RAYD_SHARED_SCENE_ASSERT_POD(PackedFloat4)",
        ):
            self.assertIn(required, header)
        self.assertIn("sizeof(PackedFloat4) == sizeof(float4)", source)
        self.assertIn("alignof(PackedFloat4) == alignof(float4)", source)

    def test_raw_pointer_count_offset_contract_is_complete(self):
        header = HEADER.read_text(encoding="utf-8")
        for struct_name in (
            "GlobalGeometryPackingParams",
            "GlobalVertexTangentPackingParams",
            "GlobalVertexTangentZeroParams",
        ):
            self.assertIn(f"struct {struct_name}", header)
            self.assertIn(f"RAYD_SHARED_SCENE_ASSERT_POD({struct_name})", header)
        for field in (
            "const float* mesh_vertices;",
            "const std::int32_t* mesh_faces;",
            "std::int32_t vertex_count;",
            "std::int32_t face_count;",
            "std::int32_t vertex_offset;",
            "std::int32_t face_offset;",
            "float* global_vertices;",
            "std::int32_t* global_faces;",
            "const float* mesh_tangent;",
            "float* global_tangent;",
        ):
            self.assertIn(field, header)

    def test_torch_adapter_owns_validation_stream_and_errors(self):
        source = TORCH_ADAPTER.read_text(encoding="utf-8")
        self.assertIn("<rayd/scene/packing.h>", source)
        for launcher in (
            "shared::scene::launch_pack_global_geometry_async",
            "shared::scene::launch_pack_global_vertex_tangent_async",
            "shared::scene::launch_zero_global_vertex_tangent_range_async",
        ):
            self.assertIn(launcher, source)
        for retained in (
            "launch_require_count",
            "current_torch_cuda_context",
            "cuda_check",
            "vertex range exceeds int32",
        ):
            self.assertIn(retained, source)
        for removed_kernel in (
            "__global__ void pack_global_geometry_kernel",
            "__global__ void pack_global_vertex_tangent_kernel",
            "__global__ void zero_global_vertex_tangent_range_kernel",
        ):
            self.assertNotIn(removed_kernel, source)


if __name__ == "__main__":
    unittest.main()
