import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIR = ROOT / "shared" / "include" / "rayd" / "shared" / "edge"
SOURCE_DIR = ROOT / "shared" / "src" / "edge"
CONTRACT_HEADERS = (
    "bvh_types.h",
    "bvh_build.h",
    "bvh_query.h",
    "edge_distance.h",
)


class BVH4SharedEdgeCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {
            name: (INCLUDE_DIR / name).read_text(encoding="utf-8")
            for name in CONTRACT_HEADERS
        }
        cls.combined = "\n".join(cls.sources.values())

    def test_contract_headers_exist_in_shared_tree(self):
        for name in CONTRACT_HEADERS:
            self.assertTrue((INCLUDE_DIR / name).is_file(), name)

    def test_contracts_do_not_depend_on_backend_frameworks(self):
        forbidden = (
            "at::Tensor",
            "torch/",
            "torch::",
            "drjit",
            "nanobind",
            "nb::",
            "python.h",
            "PyObject",
            "SceneHandle",
        )
        lowered = self.combined.lower()
        for token in forbidden:
            self.assertNotIn(token.lower(), lowered)

    def test_contracts_have_no_allocation_or_synchronization_api(self):
        forbidden = (
            "cudaMalloc",
            "cudaFree",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "operator new",
            "std::vector",
            "std::unique_ptr",
            "std::shared_ptr",
        )
        for token in forbidden:
            self.assertNotIn(token, self.combined)

    def test_build_query_and_distance_contracts_carry_streams(self):
        for name in ("bvh_build.h", "bvh_query.h", "edge_distance.h"):
            self.assertIn("cudaStream_t stream;", self.sources[name])
        self.assertIn("launch_mark_dirty_ancestors_async", self.sources["bvh_build.h"])
        self.assertIn("launch_point_bvh_query_async", self.sources["bvh_query.h"])
        self.assertIn("launch_ray_bvh_query_async", self.sources["bvh_query.h"])
        self.assertIn("launch_point_edge_distances_async", self.sources["edge_distance.h"])
        self.assertIn("launch_ray_edge_distances_async", self.sources["edge_distance.h"])

    def test_shared_build_source_is_backend_neutral_and_enqueue_only(self):
        source = (SOURCE_DIR / "bvh_build.cu").read_text(encoding="utf-8")
        forbidden = (
            "cudaMalloc",
            "cudaFree",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "throw ",
            "std::vector",
            "at::Tensor",
            "drjit",
            "nanobind",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("params.stream", source)

    def test_storage_is_raw_pointer_count_and_caller_owned(self):
        pointer_fields = re.findall(
            r"(?:const\s+)?(?:float|void|std::int32_t)\s*\*\w+",
            self.combined,
        )
        count_fields = re.findall(
            r"std::size_t\s+(?:count|\w+_count|capacity|\w+_stride|size_bytes)",
            self.combined,
        )
        self.assertGreaterEqual(len(pointer_fields), 20)
        self.assertGreaterEqual(len(count_fields), 12)
        self.assertGreaterEqual(self.combined.lower().count("caller-owned"), 6)

    def test_every_contract_struct_has_layout_assertions(self):
        struct_names = re.findall(r"^struct\s+(\w+)\s*\{", self.combined, re.MULTILINE)
        self.assertGreater(len(struct_names), 0)
        for name in struct_names:
            macro_assertion = f"RAYD_SHARED_EDGE_ASSERT_POD({name})"
            self.assertTrue(
                macro_assertion in self.combined
                or f"is_standard_layout_v<{name}>" in self.combined,
                f"missing standard-layout assertion for {name}",
            )
            self.assertTrue(
                macro_assertion in self.combined
                or f"is_trivially_copyable_v<{name}>" in self.combined,
                f"missing trivially-copyable assertion for {name}",
            )


if __name__ == "__main__":
    unittest.main()
