# Copyright Xingyu Chen.
# Tests benchmark script Dr.Jit.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "drjit" / "benchmark_surfel_queries.py"


class SurfelBenchmarkScriptTests(unittest.TestCase):
    def test_benchmark_reports_irgs_followup_metrics(self):
        source = BENCHMARK.read_text(encoding="utf-8")

        self.assertIn("def benchmark_normal_output", source)
        self.assertIn("def benchmark_geometry_update", source)
        self.assertIn("def benchmark_miss_prepass", source)
        self.assertIn("--skip-normal-output", source)
        self.assertIn("--skip-geometry-update", source)
        self.assertIn("--skip-miss-prepass", source)
        self.assertIn('"normal_output"', source)
        self.assertIn('"geometry_update"', source)
        self.assertIn('"miss_prepass"', source)
        self.assertIn("rd.SurfelRenderOptions.rgb(normal=True)", source)


if __name__ == "__main__":
    unittest.main()
