from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_HEADER = ROOT / "include/rayd/shared/edge/edge_aabb.h"
SHARED_SOURCE = ROOT / "src/edge/edge_shared.cu"
TORCH_HEADER = ROOT / "include/rayd/torch/edge/bvh.h"
TORCH_SOURCE = ROOT / "src/edge/edge_bvh.cu"
TORCH_CMAKE = ROOT / "torch/CMakeLists.txt"


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

    def test_torch_adapter_uses_shared_api_on_the_buffers_device_stream(self) -> None:
        # Since the multi-GPU Phase 0 hardening (f643336) the adapter derives
        # its device and stream from the scene-owned buffers it operates on
        # instead of the ambient current_torch_cuda_context().
        source = TORCH_SOURCE.read_text(encoding="utf-8")
        header = TORCH_HEADER.read_text(encoding="utf-8")
        cmake = TORCH_CMAKE.read_text(encoding="utf-8")
        self.assertIn("rayd/shared/edge/edge_aabb.h", source)
        self.assertNotIn("current_torch_cuda_context()", source)
        self.assertIn("c10::cuda::CUDAGuard guard(out_aabbs.device());", source)
        self.assertIn(
            "at::cuda::getCurrentCUDAStream(out_aabbs.get_device()).stream();",
            source,
        )
        self.assertRegex(
            source,
            re.compile(r"launch_edge_aabb\([\s\S]+?stream\s*\);"),
        )
        # F1 keeps a Torch-only raw-BVH encoding kernel in this adapter.  The
        # edge AABB implementation itself must remain exclusively shared.
        global_kernels = re.findall(r"__global__\s+void\s+(\w+)", source)
        self.assertEqual(global_kernels, ["encode_raw_bvh_kernel"])
        self.assertNotIn("compute_edge_aabbs_kernel", source)
        self.assertNotIn("compute_edge_optix_aabbs_gpu", header)
        self.assertIn("src/edge/edge_shared.cu", cmake)


if __name__ == "__main__":
    unittest.main()
