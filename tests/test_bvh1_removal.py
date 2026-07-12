import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BVH1RemovalTests(unittest.TestCase):
    def test_hlbvh_runtime_and_benchmark_symbols_are_removed(self):
        forbidden = (
            "HybridTopLevelSAH",
            "hybrid_top_level_sah",
            "TopLevelBuildRecord",
            "SAHBin",
            "select_lbvh_clusters",
            "collect_top_level_nodes",
            "choose_sah_split",
            "build_top_level_bvh_recursive",
        )
        active_paths = (
            ROOT / "backends" / "drjit" / "include" / "rayd" / "edge" / "edge_bvh_config.h",
            ROOT / "backends" / "drjit" / "src" / "edge" / "scene_edge.cpp",
            ROOT / "backends" / "drjit" / "tests" / "benchmark_edge_bvh_stages.py",
        )

        for path in active_paths:
            text = path.read_text(encoding="utf-8")
            for symbol in forbidden:
                self.assertNotIn(symbol, text, f"{symbol} remains in {path.relative_to(ROOT)}")

    def test_historical_hlbvh_baseline_is_retained_as_a_rejected_strategy(self):
        path = (
            ROOT
            / "shared"
            / "benchmarks"
            / "baselines"
            / "bvh0_strategy_stage_20260711.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("hybrid_top_level_sah", payload["single_mesh"]["strategies"])
        self.assertTrue(
            any("deletion decision baseline" in note for note in payload["observations"])
        )

    def test_combined_backend_documentation_uses_canonical_name(self):
        agent_guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("combined public backend is named `optix_drjit`", agent_guide)
        self.assertIn("`hybrid` is a deprecated compatibility alias", agent_guide)

    def test_atomic_bottom_up_builds_publish_before_arrival(self):
        source = (
            ROOT / "shared" / "src" / "edge" / "bvh_build.cu"
        ).read_text(encoding="utf-8")
        kernels = {
            "finalize_leaves_and_bounds_kernel": (
                r"while \(current >= 0\) \{"
                r".*?__threadfence\(\);"
                r"\s*if \(atomicAdd\((?:params\.)?merge_counters \+ current, 1\) == 0\)"
            ),
        }
        for kernel, pattern in kernels.items():
            with self.subTest(kernel=kernel):
                start = source.index(f"__global__ void {kernel}")
                end = source.find("__global__ void ", start + 1)
                body = source[start:] if end < 0 else source[start:end]
                self.assertRegex(body, re.compile(pattern, re.DOTALL))


if __name__ == "__main__":
    unittest.main()
