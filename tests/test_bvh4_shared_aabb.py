from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_HEADER = ROOT / "shared/include/rayd/shared/edge/edge_aabb.h"
SHARED_SOURCE = ROOT / "shared/src/edge/edge_aabb.cu"
TORCH_HEADER = ROOT / "backends/torch/include/rayd/torch/edge/bvh.h"
TORCH_SOURCE = ROOT / "backends/torch/src/torch_ext/edge/bvh.cu"
TORCH_CMAKE = ROOT / "backends/torch/CMakeLists.txt"


class SharedEdgeAabbSourceTests(unittest.TestCase):
    def test_shared_api_is_raw_pointer_stream_only(self) -> None:
        header = SHARED_HEADER.read_text(encoding="utf-8")
        self.assertEqual(header.count("void launch_edge_aabb("), 1)
        self.assertIn("cudaStream_t stream", header)
        for backend_type in ("at::Tensor", "drjit", "nanobind", "Scene"):
            self.assertNotIn(backend_type, header)

    def test_shared_implementation_is_async_and_allocation_free(self) -> None:
        source = SHARED_SOURCE.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(r"compute_edge_aabbs_kernel<<<[^>]+,\s*stream>>>", re.DOTALL),
        )
        for forbidden in (
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaMalloc",
            "cudaFree",
            "at::Tensor",
            "drjit",
            "nanobind",
        ):
            self.assertNotIn(forbidden, source)

    def test_torch_adapter_uses_shared_api_on_current_stream(self) -> None:
        source = TORCH_SOURCE.read_text(encoding="utf-8")
        header = TORCH_HEADER.read_text(encoding="utf-8")
        cmake = TORCH_CMAKE.read_text(encoding="utf-8")
        self.assertIn("rayd/shared/edge/edge_aabb.h", source)
        self.assertIn("current_torch_cuda_context()", source)
        self.assertRegex(
            source,
            re.compile(r"launch_edge_aabb\([\s\S]+?torch_ctx\.stream\s*\);"),
        )
        self.assertNotIn("__global__", source)
        self.assertNotIn("compute_edge_optix_aabbs_gpu", header)
        self.assertIn("shared/src/edge/edge_aabb.cu", cmake)


if __name__ == "__main__":
    unittest.main()
