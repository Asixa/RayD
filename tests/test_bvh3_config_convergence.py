import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "include" / "rayd" / "edge" / "edge_bvh_config.h"
BUILD_SOURCE = ROOT / "src" / "edge" / "edge_bvh_jit.cu"
SCENE_SOURCE = ROOT / "src" / "edge" / "edge_jit.cpp"
BENCHMARK = ROOT / "benchmarks" / "drjit" / "benchmark_edge_bvh_stages.py"


class BVH3ConfigConvergenceTests(unittest.TestCase):
    def test_dominated_build_modes_are_removed_from_active_surfaces(self):
        forbidden = (
            "EdgeBVHFinalizeMode",
            "EdgeBVHTreeletScheduleMode",
            "EdgeBVHCompactionMode",
            "EdgeBVHNodeLayoutMode",
            "RAYD_EDGE_BVH_FINALIZE_MODE",
            "RAYD_EDGE_BVH_TREELET_SCHEDULE_MODE",
            "RAYD_EDGE_BVH_COMPACTION_MODE",
            "RAYD_EDGE_BVH_NODE_LAYOUT_MODE",
            "active_edge_bvh_finalize_mode",
            "active_edge_bvh_treelet_schedule_mode",
            "active_edge_bvh_compaction_mode",
            "active_edge_bvh_node_layout_mode",
            "finalize_mode",
            "treelet_schedule_mode",
            "compaction_mode",
            "node_layout_mode",
            "--finalize-mode",
            "--treelet-schedule-mode",
            "--compaction-mode",
            "--node-layout-mode",
            "finalize_leaf_nodes_kernel",
            "finalize_raw_subtree_leaf_counts_kernel",
            "mark_collapsible_raw_nodes_kernel",
            "emit_collapsed_raw_nodes_kernel",
            "emit_compacted_bvh_preorder_exact",
            "compact_edge_bvh_gpu",
            "CompactedEdgeBVHPlan",
            "packed_node",
        )
        for path in (CONFIG, BUILD_SOURCE, SCENE_SOURCE, BENCHMARK):
            text = path.read_text(encoding="utf-8")
            for symbol in forbidden:
                self.assertNotIn(symbol, text, f"{symbol} remains in {path.relative_to(ROOT)}")

    def test_supported_build_controls_remain(self):
        config = CONFIG.read_text(encoding="utf-8")
        benchmark = BENCHMARK.read_text(encoding="utf-8")

        required_config = (
            "EdgeBVHPostBuildStrategy::GpuTreelet",
            "EdgeBVHPostBuildStrategy::None",
            "RAYD_EDGE_BVH_POST_BUILD_STRATEGY",
            "EdgeBVHBuildStreamMode::Overlap",
            "EdgeBVHBuildStreamMode::Serial",
            "RAYD_EDGE_BVH_BUILD_STREAM_MODE",
        )
        for symbol in required_config:
            self.assertIn(symbol, config)

        required_benchmark = (
            '"post_build_strategy": "gpu_treelet"',
            '"build_stream_mode": "overlap"',
            'choices=("none", "gpu_treelet")',
            'choices=("serial", "overlap")',
        )
        for symbol in required_benchmark:
            self.assertIn(symbol, benchmark)

    def test_auto_full_and_dirty_ancestor_refit_remain(self):
        source = SCENE_SOURCE.read_text(encoding="utf-8")
        build_source = BUILD_SOURCE.read_text(encoding="utf-8")
        required = (
            "EdgeBVHRefitStrategy::Auto",
            "EdgeBVHRefitStrategy::Full",
            "EdgeBVHRefitStrategy::DirtyAncestors",
            "RAYD_EDGE_BVH_REFIT_STRATEGY",
            'normalized == "auto"',
            'normalized == "full"',
            'normalized == "dirty_ancestors"',
        )
        for symbol in required:
            self.assertIn(symbol, source)
        self.assertIn("compact_and_refit_edge_bvh_level_gpu", build_source)


if __name__ == "__main__":
    unittest.main()
