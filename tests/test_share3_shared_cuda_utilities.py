# Copyright Xingyu Chen.
# Tests share3 shared cuda utilities.

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "include" / "rayd"
SHARED_SOURCE = ROOT / "src"

# Local builds drop generated trees under backend build directories (notably the
# Torch rayd-source-bundle, which contains a verbatim canonical source copy). Those copies are
# build output, not a second definition, so ownership scans must skip them.
_BUILD_OUTPUT_DIRS = frozenset({"build", "artifacts", "_skbuild", "dist"})


def _is_build_output(path: Path) -> bool:
    return any(part in _BUILD_OUTPUT_DIRS for part in path.relative_to(ROOT).parts)


UNITS = {
    "aabb": (
        SHARED_INCLUDE / "edge" / "edge_aabb.h",
        SHARED_SOURCE / "edge" / "edge_shared.cu",
    ),
    "dedup": (
        SHARED_INCLUDE / "reflection" / "dedup.h",
        SHARED_SOURCE / "reflection" / "dedup_shared.cu",
    ),
    "packing": (
        SHARED_INCLUDE / "scene" / "packing.h",
        SHARED_SOURCE / "scene" / "packing_shared.cu",
    ),
}


class Share3SharedCudaUtilitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = {}
        for name, (header, source) in UNITS.items():
            cls.text[name] = {
                "header": header.read_text(encoding="utf-8"),
                "source": source.read_text(encoding="utf-8"),
            }

    def test_expected_shared_units_exist(self):
        for header, source in UNITS.values():
            self.assertTrue(header.is_file(), header)
            self.assertTrue(source.is_file(), source)

    def test_public_apis_use_raw_pointers_counts_and_explicit_streams(self):
        aabb = self.text["aabb"]["header"]
        self.assertRegex(aabb, r"void\s+launch_edge_aabb\s*\(")
        self.assertGreaterEqual(aabb.count("const float *"), 6)
        self.assertIn("int edge_count", aabb)
        self.assertIn("cudaStream_t stream", aabb)

        dedup = self.text["dedup"]["header"]
        for name in (
            "launch_reflection_dedup_build_keys",
            "launch_reflection_dedup_mark_boundaries",
            "launch_reflection_dedup_zero_base_ids",
            "launch_reflection_dedup_sub_cluster",
            "launch_reflection_dedup_compact",
            "launch_reflection_dedup_sequence",
        ):
            self.assertRegex(dedup, rf"\b{name}\s*\(")
        for params in (
            "ReflectionDedupBuildKeysParams",
            "ReflectionDedupSubClusterParams",
            "ReflectionDedupCompactParams",
            "ReflectionDedupSequenceParams",
        ):
            body = self._struct_body(dedup, params)
            self.assertIn("*", body)
            self.assertRegex(body, r"(?:count|ray_count|item_count|max_bounces)")
            self.assertIn("cudaStream_t stream", body)

        packing = self.text["packing"]["header"]
        for name in (
            "launch_pack_global_geometry_async",
            "launch_pack_global_vertex_tangent_async",
            "launch_zero_global_vertex_tangent_range_async",
        ):
            self.assertRegex(packing, rf"cudaError_t\s+{name}\s*\(")
        for params in (
            "GlobalGeometryPackingParams",
            "GlobalVertexTangentPackingParams",
            "GlobalVertexTangentZeroParams",
        ):
            body = self._struct_body(packing, params)
            self.assertIn("*", body)
            self.assertRegex(body, r"(?:count|offset)")
            self.assertIn("cudaStream_t stream", body)

    def test_shared_layer_is_enqueue_only_and_backend_neutral(self):
        combined = "\n".join(
            unit[kind]
            for unit in self.text.values()
            for kind in ("header", "source")
        )
        for forbidden in (
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaMemset",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "throw ",
            "at::Tensor",
            "torch::",
            "drjit",
            "nanobind",
            "SceneHandle",
        ):
            self.assertNotIn(forbidden, combined)
        for include in re.findall(r"#include\s*[<\"]([^>\"]+)", combined):
            lowered = include.lower()
            self.assertNotIn("torch", lowered)
            self.assertNotIn("drjit", lowered)
            self.assertNotIn("nanobind", lowered)

    def test_contract_structs_are_pod_and_packing_layout_is_frozen(self):
        for unit_name in ("dedup", "packing"):
            header = self.text[unit_name]["header"]
            structs = re.findall(r"^struct\s+(\w+)\s*\{", header, re.MULTILINE)
            self.assertTrue(structs)
            for struct in structs:
                explicit = (
                    f"is_standard_layout_v<{struct}>" in header
                    and f"is_trivially_copyable_v<{struct}>" in header
                )
                macro = re.search(rf"ASSERT_POD\({struct}\)", header) is not None
                self.assertTrue(explicit or macro, struct)
        packing = self.text["packing"]["header"]
        self.assertRegex(packing, r"struct\s+alignas\(16\)\s+PackedFloat4")
        self.assertRegex(
            packing,
            r"sizeof\(PackedFloat4\)\s*==\s*(?:16|4u?\s*\*\s*sizeof\(float\))",
        )
        self.assertRegex(
            packing,
            r"alignof\(PackedFloat4\)\s*==\s*(?:16|4u?\s*\*\s*alignof\(float\))",
        )
        self.assertRegex(
            packing,
            r"offsetof\(PackedFloat4,\s*w\)\s*==\s*(?:12|3u?\s*\*\s*sizeof\(float\))",
        )

    def test_duplicate_cuda_kernels_have_one_shared_definition(self):
        expected_locations = {
            "compute_edge_aabbs_kernel": "src/edge/edge_shared.cu",
            "reflection_dedup_build_keys_kernel": "src/reflection/dedup_shared.cu",
            "reflection_dedup_mark_boundaries_kernel": "src/reflection/dedup_shared.cu",
            "reflection_dedup_zero_base_ids_kernel": "src/reflection/dedup_shared.cu",
            "reflection_dedup_sub_cluster_kernel": "src/reflection/dedup_shared.cu",
            "reflection_dedup_compact_kernel": "src/reflection/dedup_shared.cu",
            "pack_global_geometry_kernel": "src/scene/packing_shared.cu",
            "pack_global_vertex_tangent_kernel": "src/scene/packing_shared.cu",
        }
        cuda_sources = [
            path
            for path in (ROOT / "src").rglob("*.cu")
            if not _is_build_output(path)
        ]
        for kernel, expected in expected_locations.items():
            definitions = []
            pattern = re.compile(rf"__global__\s+void\s+{kernel}\s*\(")
            for path in cuda_sources:
                if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                    definitions.append(path.relative_to(ROOT).as_posix())
            self.assertEqual(definitions, [expected], kernel)

    def test_cmake_and_callers_name_shared_paths_explicitly(self):
        drjit_cmake = (ROOT / "drjit" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        ).replace("\\", "/")
        torch_cmake = (ROOT / "torch" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        ).replace("\\", "/")
        for cmake in (drjit_cmake, torch_cmake):
            self.assertTrue(
                "edge/edge_shared.cu" in cmake,
                "backend CMake is missing the shared AABB source path",
            )
            self.assertTrue(
                "reflection/dedup_shared.cu" in cmake,
                "backend CMake is missing the shared reflection-dedup source path",
            )
        self.assertTrue(
            "scene/packing_shared.cu" in torch_cmake,
            "Torch CMake is missing the shared scene-packing source path",
        )

        callers = {
            "drjit_aabb": ROOT / "src" / "edge" / "edge_bvh_jit.cu",
            "torch_aabb": ROOT / "src" / "edge" / "edge_bvh.cu",
            "drjit_dedup": ROOT / "src" / "reflection" / "reflection_kernels_jit.cu",
            "torch_dedup": ROOT / "src" / "reflection" / "reflection_kernels.cu",
            "torch_packing": ROOT / "src" / "scene" / "cache.cu",
        }
        caller_text = {
            name: path.read_text(encoding="utf-8") for name, path in callers.items()
        }
        self.assertIn("<rayd/edge/edge_aabb.h>", caller_text["drjit_aabb"])
        self.assertIn("<rayd/edge/edge_aabb.h>", caller_text["torch_aabb"])
        self.assertIn("<rayd/reflection/dedup.h>", caller_text["drjit_dedup"])
        self.assertIn("<rayd/reflection/dedup.h>", caller_text["torch_dedup"])
        self.assertIn("<rayd/scene/packing.h>", caller_text["torch_packing"])

    def test_aabb_reference_covers_reversed_degenerate_and_negative_inflation(self):
        def reference(p0, edge, inflation):
            p1 = tuple(a + b for a, b in zip(p0, edge))
            radius = max(inflation, 0.0)
            return tuple(min(a, b) - radius for a, b in zip(p0, p1)) + tuple(
                max(a, b) + radius for a, b in zip(p0, p1)
            )

        self.assertEqual(
            reference((3.0, -1.0, 2.0), (-5.0, 4.0, -2.0), 0.25),
            (-2.25, -1.25, -0.25, 3.25, 3.25, 2.25),
        )
        self.assertEqual(
            reference((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), -4.0),
            (1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
        )
        source = self.text["aabb"]["source"]
        self.assertIn("const float radius = fmaxf(inflation, 0.0f);", source)
        self.assertIn("if (edge_count == 0)", source)

    @staticmethod
    def _struct_body(header, name):
        match = re.search(rf"struct\s+(?:alignas\([^)]*\)\s+)?{name}\s*\{{(.*?)\n\}};", header, re.DOTALL)
        if match is None:
            raise AssertionError(f"missing struct {name}")
        return match.group(1)


if __name__ == "__main__":
    unittest.main()
