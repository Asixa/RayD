import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "shared/include/rayd/shared/edge"
SHARED_SOURCE = ROOT / "shared/src/edge"


class Share5EdgeBvhCoreTests(unittest.TestCase):
    def test_raw_and_compact_topologies_are_distinct_pods(self):
        source = (SHARED_INCLUDE / "bvh_types.h").read_text(encoding="utf-8")
        for type_name in (
            "RawBvhTopologyView",
            "MutableRawBvhTopologyView",
            "CompactBvhTopologyView",
        ):
            self.assertIn(f"struct {type_name}", source)
            self.assertIn(f"RAYD_SHARED_EDGE_ASSERT_POD({type_name})", source)
        self.assertIn("left_child[node] = -leaf_begin - 1", source)
        self.assertIn("leaf_primitives", source)
        self.assertNotIn("struct BvhTopologyView", source)

    def test_product_treelet_constants_have_one_shared_definition(self):
        shared = (SHARED_INCLUDE / "bvh_types.h").read_text(encoding="utf-8")
        adapter = (ROOT / "backends/drjit/include/rayd/edge/edge_bvh_config.h").read_text(
            encoding="utf-8"
        )
        for token in (
            "kBvhTreeletMaxLeaves = 7",
            "kBvhTreeletMinPrimitives = 65536",
            "kBvhTreeletMinSubtreeLeaves = 32",
            "kBvhTreeletCostInflationRatio = 1e-4f",
            "kBvhLeafSize = 4",
            "kBvhTraversalStackDepth = 64",
            "kBvhTopKMax = 16",
        ):
            self.assertIn(token, shared)
        self.assertGreaterEqual(adapter.count("shared::edge::kBvhTreelet"), 4)
        self.assertIn("shared::edge::kBvhLeafSize", adapter)
        scene_edge = (ROOT / "backends/drjit/src/edge/scene_edge.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("shared::edge::kBvhTraversalStackDepth", scene_edge)

    def test_build_stages_are_shared_and_drjit_is_an_adapter(self):
        header = (SHARED_INCLUDE / "bvh_build.h").read_text(encoding="utf-8")
        shared = (SHARED_SOURCE / "bvh_build.cu").read_text(encoding="utf-8")
        adapter = (ROOT / "backends/drjit/src/edge/edge_bvh.cu").read_text(encoding="utf-8")
        launchers = (
            "launch_compute_primitive_bounds_async",
            "launch_init_sequence_async",
            "launch_compute_morton_codes_async",
            "launch_build_radix_tree_async",
            "launch_finalize_leaves_and_bounds_async",
            "launch_initialize_leaf_costs_async",
            "launch_initialize_internal_costs_async",
            "launch_optimize_selected_treelets_async",
        )
        for launcher in launchers:
            self.assertIn(launcher, header)
            self.assertIn(launcher, shared)
            self.assertIn(f"shared::edge::{launcher}", adapter)
        self.assertNotIn("__global__", adapter)

    def test_shared_cuda_owns_no_resources_or_host_barriers(self):
        combined = "\n".join(
            (SHARED_SOURCE / name).read_text(encoding="utf-8")
            for name in ("bvh_build.cu", "bvh_query.cu", "edge_distance.cu")
        )
        for forbidden in (
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaMemset",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cub::",
            "std::vector",
            "at::Tensor",
            "drjit",
            "nanobind",
            "throw ",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertGreaterEqual(combined.count("params.stream"), 12)

    def test_query_contract_freezes_masks_topk_stack_and_tie_break(self):
        header = (SHARED_INCLUDE / "bvh_query.h").read_text(encoding="utf-8")
        source = (SHARED_SOURCE / "bvh_query.cu").read_text(encoding="utf-8")
        for token in (
            "CompactBvhTopologyView topology",
            "const std::uint8_t *active_mask",
            "const std::uint8_t *edge_mask",
            "EdgeBvhTopKMax =",
            "query_stride",
            "stack_depth",
            "overflow_capacity",
            "launch_point_bvh_query_async",
            "launch_ray_bvh_query_async",
        ):
            self.assertIn(token, header)
        self.assertIn("distance == slot_distance && edge < slot_edge", source)
        self.assertIn("depth * scratch.query_stride + query", source)
        self.assertIn("params.scratch.overflow[query] = 1u", source)

    def test_topk_runtime_dispatch_uses_bucketed_local_state(self):
        header = (SHARED_INCLUDE / "bvh_query.h").read_text(encoding="utf-8")
        source = (SHARED_SOURCE / "bvh_query.cu").read_text(encoding="utf-8")
        expected_buckets = {
            0: 0,
            1: 1,
            2: 2,
            3: 4,
            4: 4,
            5: 8,
            6: 8,
            7: 8,
            8: 8,
            9: 16,
            10: 16,
            11: 16,
            12: 16,
            13: 16,
            14: 16,
            15: 16,
            16: 16,
            17: 0,
        }
        for k, capacity in expected_buckets.items():
            mapping = f"edge_bvh_topk_capacity({k}) == {capacity}"
            self.assertIn(mapping, header)
        for capacity in (1, 2, 4, 8, 16):
            self.assertIn(
                f"launch_bvh_query_capacity<{capacity}, RayQuery>", source
            )
        for local_array in (
            "int edge_ids[TopKCapacity]",
            "float distances[TopKCapacity]",
            "float edge_parameters[TopKCapacity]",
            "float query_parameters[TopKCapacity]",
        ):
            self.assertIn(local_array, source)
        self.assertNotIn("edge_ids[EdgeBvhTopKMax]", source)
        self.assertNotIn("distances[EdgeBvhTopKMax]", source)

    def test_exact_distance_contract_is_masked_and_async(self):
        header = (SHARED_INCLUDE / "edge_distance.h").read_text(encoding="utf-8")
        source = (SHARED_SOURCE / "edge_distance.cu").read_text(encoding="utf-8")
        for token in (
            "launch_point_edge_distances_async",
            "launch_ray_edge_distances_async",
            "const std::uint8_t *active_mask",
            "const std::uint8_t *edge_mask",
        ):
            self.assertIn(token, header)
        self.assertIn("point_segment_distance", source)
        self.assertIn("ray_segment_distance", source)

    def test_both_backends_compile_the_shared_units(self):
        drjit = (ROOT / "backends/drjit/CMakeLists.txt").read_text(encoding="utf-8")
        torch = (ROOT / "backends/torch/CMakeLists.txt").read_text(encoding="utf-8")
        for unit in ("bvh_build.cu", "bvh_query.cu", "edge_distance.cu"):
            self.assertIn(f"shared/src/edge/{unit}", drjit)
            self.assertIn(f"shared/src/edge/{unit}", torch)

    def test_removed_strategies_do_not_reappear_in_shared_core(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in list(SHARED_INCLUDE.glob("*.h")) + list(SHARED_SOURCE.glob("*.cu"))
        ).lower()
        for forbidden in (
            "hlbvh",
            "top-level sah",
            "gpu_emit",
            "host_upload_exact",
            "per_level_uploads",
            "packed node mirror",
        ):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
