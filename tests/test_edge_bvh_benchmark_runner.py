import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from backends.drjit.tests.benchmark_edge_bvh_matrix import (
    WORKER_PREFIX,
    _component_edge_counts,
    _mesh_components,
    aggregate_case,
    case_id,
    run_worker,
    summarize,
)
from tests.performance.edge_bvh_gate import ContractError


class EdgeBVHBenchmarkRunnerContractTests(unittest.TestCase):
    def test_worker_failure_reports_the_case_id(self):
        dimensions = {
            "edge_count": 1000,
            "query_count": 1,
            "query_kind": "point",
            "top_k": 1,
            "update_mode": "static",
            "mask": "sparse",
            "distribution": "long_thin",
        }
        failed = CompletedProcess(
            args=[],
            returncode=1,
            stdout=f'{WORKER_PREFIX}{{"error":"build failed"}}\n',
            stderr="",
        )
        with patch(
            "backends.drjit.tests.benchmark_edge_bvh_matrix.subprocess.run",
            return_value=failed,
        ):
            with self.assertRaisesRegex(
                ContractError,
                "1000-1-point-1-static-sparse-long_thin: build failed",
            ):
                run_worker(dimensions, Path("matrix.json"))

    def test_component_partition_preserves_every_full_profile_edge_count(self):
        for edge_count in (1000, 16000, 64000, 111000, 500000, 2000000):
            with self.subTest(edge_count=edge_count):
                components = _component_edge_counts(edge_count)
                self.assertEqual(len(components), 100)
                self.assertEqual(sum(components), edge_count)
                self.assertTrue(all(value >= 3 and value % 2 == 1 for value in components))

    def test_generated_strip_topology_has_the_requested_edge_count(self):
        components = _mesh_components(1000, "grid", 20260711)
        total = 0
        for component in components:
            edges = set()
            for triangle in zip(component["i0"], component["i1"], component["i2"]):
                for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                    edges.add(tuple(sorted((a, b))))
            total += len(edges)
        self.assertEqual(total, 1000)

    def test_summary_reports_median_and_p95_from_real_samples(self):
        self.assertEqual(
            summarize([1.0, 2.0, 3.0, 4.0, 5.0], "ms"),
            {"unit": "ms", "samples": [1.0, 2.0, 3.0, 4.0, 5.0], "median": 3.0, "p95": 5.0},
        )

    def test_case_id_and_aggregation_are_deterministic(self):
        dimensions = {
            "edge_count": 1000,
            "query_count": 1,
            "query_kind": "point",
            "top_k": 1,
            "update_mode": "static",
            "mask": "sparse",
            "distribution": "grid",
        }
        sample = {
            "performance": {
                "hot_query_ms": [1.0] * 5,
                "build_ms": [2.0] * 5,
                "refit_ms": [3.0] * 5,
                "peak_device_memory_bytes": [4.0] * 5,
                "cold_create_ms": [5.0] * 5,
            },
            "launch_audit": {
                "method": "independent_stable_audit",
                "timing_isolated": True,
                "runs": 1,
                "sampling": "single_deterministic_pass_not_timing_sample",
                "state": "fresh_scene_build_warm_queries_and_refit",
                "stages": {
                    stage: {
                        "drjit_kernel_launches": 1,
                        "drjit_optix_launches": 0,
                        "native_cuda_kernel_launches": 2,
                        "native_cub_launches": 0,
                        "native_optix_launches": 0,
                        "native_optix_accel_operations": 0,
                        "total_observed_launches": 3,
                    }
                    for stage in (
                        "build", "refit", "query_point",
                        "query_finite_ray", "query_infinite_ray",
                    )
                },
            },
            "correctness": {"max_abs_error": 1e-7, "max_rel_error": 2e-7},
            "ad": {
                "vjp_max_abs_error": 1e-6,
                "vjp_max_rel_error": 2e-6,
                "jvp_max_abs_error": 3e-6,
                "jvp_max_rel_error": 4e-6,
            },
        }
        case = aggregate_case(dimensions, sample)
        self.assertEqual(case["case_id"], case_id(dimensions))
        self.assertEqual(case["performance"]["cold_create_ms"]["median"], 5.0)
        self.assertEqual(case["launch_audit"], sample["launch_audit"])
        self.assertEqual(case["ad"], sample["ad"])


if __name__ == "__main__":
    unittest.main()
