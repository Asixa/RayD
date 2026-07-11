import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backends" / "drjit" / "src" / "edge" / "edge_bvh.cu"


class BVH2RemovalTests(unittest.TestCase):
    def test_dead_gpu_flat_treelet_prepare_path_is_removed(self):
        source = SOURCE.read_text(encoding="utf-8")
        forbidden = (
            "gpu_flat_treelet_prepare",
            "EdgeBVHMaxRadixTreeHeight",
            "compute_subtree_leaf_count_host",
            "update_internal_nodes_kernel",
            "finalize_treelet_metrics_kernel",
            "count_treelet_level_nodes_kernel",
            "scatter_treelet_level_nodes_kernel",
            "update_flat_treelet_level_kernel",
            "optimize_flat_treelet_level_kernel",
            "failed to init treelet subtree leaf counts",
            "failed to launch treelet level counting",
            "failed to launch flat GPU treelet optimization",
        )
        for symbol in forbidden:
            self.assertNotIn(symbol, source)

    def test_supported_host_prepared_treelet_schedule_remains(self):
        source = SOURCE.read_text(encoding="utf-8")
        required = (
            "EdgeBVHTreeletScheduleMode::FlatLevels",
            "flatten_node_levels(optimize_levels)",
            "optimize_selected_treelets_kernel",
            "failed to upload treelet schedule",
            "failed to launch GPU treelet optimization",
        )
        for symbol in required:
            self.assertIn(symbol, source)

    def test_cuda_source_has_no_unreferenced_kernel_helpers(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("bbox_surface_area_bounds", source)


if __name__ == "__main__":
    unittest.main()
