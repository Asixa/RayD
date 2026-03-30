import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


TORCH_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())


def _run_pressure_script(*args: str, timeout: int = 240):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "benchmark_edge_bvh_pressure.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Pressure benchmark subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "Pressure benchmark did not emit JSON.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        ) from exc


@unittest.skipUnless(TORCH_CUDA_AVAILABLE, "torch with CUDA is required for RayD edge BVH pressure tests")
class EdgeBvhPressureTests(unittest.TestCase):
    def test_large_scene_pressure_benchmark_reports_sane_counts(self):
        data = _run_pressure_script(
            "--mesh-resolution",
            "16",
            "--tiles-x",
            "4",
            "--tiles-y",
            "4",
            "--tile-spacing",
            "1.25",
            "--row-z-stride",
            "0.02",
            "--query-grid-side",
            "48",
            "--mask-keep-stride",
            "4",
            "--repeats",
            "1",
            "--warmup",
            "0",
        )

        self.assertGreater(data["scene"]["tile_count"], 1)
        self.assertGreater(data["scene"]["edge_count"], 0)
        self.assertGreater(data["queries"]["query_count"], 0)
        self.assertGreater(data["sanity"]["point_valid_count"], 0)
        self.assertGreater(data["sanity"]["finite_ray_valid_count"], 0)
        self.assertGreater(data["sanity"]["infinite_ray_valid_count"], 0)
        self.assertGreater(data["build"]["avg_ms"], 0.0)
        self.assertGreater(data["timings"]["point_query"]["avg_ms"], 0.0)
        self.assertGreater(data["timings"]["finite_ray_query"]["avg_ms"], 0.0)
        self.assertGreater(data["timings"]["infinite_ray_query"]["avg_ms"], 0.0)

        scenarios = {entry["name"]: entry for entry in data["mask_scenarios"]}
        self.assertEqual(
            set(scenarios),
            {
                "full",
                "checker_tiles",
                "boundary_only",
                "interior_only",
                "stride_sparse",
                "empty",
            },
        )

        self.assertEqual(
            scenarios["full"]["active_edge_count"],
            data["scene"]["edge_count"],
        )
        self.assertAlmostEqual(scenarios["full"]["keep_ratio"], 1.0)
        self.assertEqual(scenarios["empty"]["active_edge_count"], 0)
        self.assertEqual(scenarios["empty"]["keep_ratio"], 0.0)
        self.assertEqual(scenarios["empty"]["sanity"]["point_valid_count"], 0)
        self.assertEqual(scenarios["empty"]["sanity"]["finite_ray_valid_count"], 0)
        self.assertEqual(scenarios["empty"]["sanity"]["infinite_ray_valid_count"], 0)

        for name, scenario in scenarios.items():
            self.assertGreaterEqual(scenario["active_edge_count"], 0)
            self.assertGreaterEqual(scenario["keep_ratio"], 0.0)
            self.assertLessEqual(scenario["keep_ratio"], 1.0)
            self.assertGreater(scenario["queries"]["point_query"]["avg_ms"], 0.0)
            self.assertGreater(scenario["queries"]["finite_ray_query"]["avg_ms"], 0.0)
            self.assertGreater(scenario["queries"]["infinite_ray_query"]["avg_ms"], 0.0)
            if name == "full":
                self.assertIsNone(scenario["sync_to_mask"])
                self.assertIsNone(scenario["restore_full"])
                continue

            self.assertIsNotNone(scenario["sync_to_mask"])
            self.assertIsNotNone(scenario["restore_full"])
            self.assertGreaterEqual(scenario["sync_to_mask"]["wall_ms"]["avg_ms"], 0.0)
            self.assertGreaterEqual(scenario["restore_full"]["wall_ms"]["avg_ms"], 0.0)
            self.assertGreaterEqual(
                scenario["sync_to_mask"]["profile_avg"]["edge_refit_ms"],
                0.0,
            )
            self.assertGreaterEqual(
                scenario["restore_full"]["profile_avg"]["edge_refit_ms"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
