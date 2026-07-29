# Copyright Xingyu Chen.
# Tests share5 edge bvh core.

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_INCLUDE = ROOT / "include/rayd/edge"
SHARED_SOURCE = ROOT / "src/edge"
# P3 Stage A moved the primitive-agnostic BVH machinery into shared/bvh/. The
# edge headers now re-export it, so the pins below live on the new locations and
# additionally assert that the edge layer keeps delegating to the core.
BVH_CORE_INCLUDE = ROOT / "include/rayd/bvh"
BVH_CORE_SOURCE = ROOT / "src/bvh"


class Share5EdgeBvhCoreTests(unittest.TestCase):
    def test_raw_and_compact_topologies_are_distinct_pods(self):
        topology = (BVH_CORE_INCLUDE / "topology.h").read_text(encoding="utf-8")
        edge = (SHARED_INCLUDE / "bvh_types.h").read_text(encoding="utf-8")
        for type_name in ("RawBvhTopologyView", "MutableRawBvhTopologyView", "CompactBvhTopologyView"):
            # The single struct definition now lives in the primitive-agnostic core.
            self.assertIn(f"struct {type_name}", topology)
            self.assertIn(f"RAYD_SHARED_BVH_ASSERT_POD({type_name})", topology)
            # The edge layer re-exports it so rayd::shared::edge:: stays valid.
            self.assertIn(f"using bvh::{type_name};", edge)
            self.assertNotIn(f"struct {type_name}", edge)
        self.assertIn("left_child[node] = -leaf_begin - 1", topology)
        self.assertIn("leaf_primitives", topology)
        self.assertNotIn("struct BvhTopologyView", topology)
        self.assertNotIn("struct BvhTopologyView", edge)

    def test_product_treelet_constants_have_one_shared_definition(self):
        shared = (BVH_CORE_INCLUDE / "topology.h").read_text(encoding="utf-8")
        edge = (SHARED_INCLUDE / "bvh_types.h").read_text(encoding="utf-8")
        adapter = (ROOT / "include/rayd/jit/edge_bvh_config.h").read_text(encoding="utf-8")
        for token in (
            "kBvhTreeletMaxLeaves = 7",
            "kBvhTreeletMinPrimitives = 65536",
            "kBvhTreeletMaxPrimitives = 500000",
            "kBvhTreeletMinSubtreeLeaves = 32",
            "kBvhTreeletCostInflationRatio = 1e-4f",
            "kBvhLeafSize = 4",
            "kBvhTraversalStackDepth = 64",
            "kBvhTopKMax = 16",
        ):
            # Exactly one literal definition, in the shared core.
            self.assertIn(token, shared)
            self.assertNotIn(token, edge)
        # The edge layer re-exports the constants so shared::edge::kBvh* resolves.
        for name in (
            "kBvhTreeletMaxLeaves",
            "kBvhTreeletMaxPrimitives",
            "kBvhLeafSize",
            "kBvhTraversalStackDepth",
            "kBvhTopKMax",
        ):
            self.assertIn(f"using bvh::{name};", edge)
        self.assertGreaterEqual(adapter.count("shared::edge::kBvhTreelet"), 4)
        self.assertIn("shared::edge::kBvhLeafSize", adapter)
        scene_edge = (ROOT / "src/edge/edge_jit.cpp").read_text(encoding="utf-8")
        self.assertIn("shared::edge::kBvhTraversalStackDepth", scene_edge)
        drjit_build = (ROOT / "src/edge/edge_bvh_jit.cu").read_text(encoding="utf-8")
        torch_build = (ROOT / "src/scene/scene.cpp").read_text(encoding="utf-8")
        self.assertIn("primitive_count <= EdgeBVHTreeletMaxPrimitives", drjit_build)
        self.assertIn("primitive_count <= rayd::shared::edge::kBvhTreeletMaxPrimitives", torch_build)

    def test_build_stages_are_shared_and_drjit_is_an_adapter(self):
        header = (SHARED_INCLUDE / "bvh_build.h").read_text(encoding="utf-8")
        shared = (SHARED_SOURCE / "edge_shared.cu").read_text(encoding="utf-8")
        core = (BVH_CORE_SOURCE / "build_shared.cu").read_text(encoding="utf-8")
        adapter = (ROOT / "src/edge/edge_bvh_jit.cu").read_text(encoding="utf-8")
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
            # The edge header/adapter symbol surface is unchanged.
            self.assertIn(launcher, header)
            self.assertIn(launcher, shared)
            self.assertIn(f"shared::edge::{launcher}", adapter)
        self.assertNotIn("__global__", adapter)
        # The primitive-agnostic launchers have their one real definition in the
        # shared BVH core, and the edge unit is a thin forwarder to it. The
        # edge-only primitive-bounds kernel does not leak into the core.
        self.assertNotIn("launch_compute_primitive_bounds_async", core)
        self.assertIn("__global__", core)
        for launcher in launchers[1:]:
            self.assertIn(launcher, core)
            self.assertIn(f"bvh::{launcher}", shared)

    def test_shared_cuda_owns_no_resources_or_host_barriers(self):
        combined = "\n".join((SHARED_SOURCE / name).read_text(encoding="utf-8") for name in ("edge_shared.cu",))
        combined += "\n" + (BVH_CORE_SOURCE / "build_shared.cu").read_text(encoding="utf-8")
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
        source = (SHARED_SOURCE / "edge_shared.cu").read_text(encoding="utf-8")
        traversal = (BVH_CORE_INCLUDE / "traversal_common.cuh").read_text(encoding="utf-8")
        for token in (
            "CompactBvhTopologyView topology",
            "const std::uint8_t* active_mask",
            "const std::uint8_t* edge_mask",
            "EdgeBvhTopKMax =",
            "query_stride",
            "stack_depth",
            "overflow_capacity",
            "launch_point_bvh_query_async",
            "launch_ray_bvh_query_async",
        ):
            self.assertIn(token, header)
        self.assertIn("distance == slot_distance && edge < slot_edge", source)
        self.assertIn("params.scratch.overflow[query] = 1u", source)
        # Depth-major stack indexing moved into the shared traversal helper, and
        # the edge query consumes it rather than keeping a private copy.
        self.assertIn("depth * scratch.query_stride + query", traversal)
        self.assertIn("traversal_common.cuh", source)
        self.assertIn("bvh::stack_push", source)
        self.assertIn("bvh::stack_load", source)

    def test_topk_runtime_dispatch_uses_bucketed_local_state(self):
        header = (SHARED_INCLUDE / "bvh_query.h").read_text(encoding="utf-8")
        source = (SHARED_SOURCE / "edge_shared.cu").read_text(encoding="utf-8")
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
            self.assertIn(f"launch_bvh_query_capacity<{capacity}, RayQuery>", source)
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
        source = (SHARED_SOURCE / "edge_shared.cu").read_text(encoding="utf-8")
        for token in (
            "launch_point_edge_distances_async",
            "launch_ray_edge_distances_async",
            "const std::uint8_t* active_mask",
            "const std::uint8_t* edge_mask",
        ):
            self.assertIn(token, header)
        self.assertIn("point_segment_distance", source)
        self.assertIn("ray_segment_distance", source)

    def test_both_backends_compile_the_shared_units(self):
        drjit = (ROOT / "drjit/CMakeLists.txt").read_text(encoding="utf-8")
        torch = (ROOT / "torch/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("edge/edge_shared.cu", drjit)
        self.assertIn("edge/edge_shared.cu", torch)
        # Both backends also compile the shared primitive-agnostic BVH core.
        self.assertIn("bvh/build_shared.cu", drjit)
        self.assertIn("bvh/build_shared.cu", torch)

    def test_removed_strategies_do_not_reappear_in_shared_core(self):
        paths = (
            list(SHARED_INCLUDE.glob("*.h"))
            + list(SHARED_SOURCE.glob("*.cu"))
            + list(BVH_CORE_INCLUDE.glob("*.h"))
            + list(BVH_CORE_INCLUDE.glob("*.cuh"))
            + list(BVH_CORE_SOURCE.glob("*.cu"))
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
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
