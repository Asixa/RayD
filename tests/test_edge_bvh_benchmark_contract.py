# Copyright Xingyu Chen.
# Tests edge bvh benchmark contract.

import copy
import unittest

from tests.performance.edge_bvh_gate import (
    ContractError,
    LAUNCH_COUNT_FIELDS,
    LAUNCH_STAGES,
    LAUNCH_TOTAL_FIELDS,
    PERFORMANCE_METRICS,
    evaluate_gate,
    expected_case_dimensions,
    load_matrix,
    validate_result,
)


def summary(unit, scale=1.0):
    samples = [9.0 * scale, 10.0 * scale, 10.0 * scale, 10.0 * scale, 11.0 * scale]
    return {"unit": unit, "samples": samples, "median": 10.0 * scale, "p95": 11.0 * scale}


def launch_audit(count=1):
    stages = {}
    for stage in LAUNCH_STAGES:
        counts = {field: count for field in LAUNCH_COUNT_FIELDS}
        counts["total_observed_launches"] = count * len(LAUNCH_TOTAL_FIELDS)
        stages[stage] = counts
    return {
        "method": "independent_stable_audit",
        "timing_isolated": True,
        "runs": 1,
        "sampling": "single_deterministic_pass_not_timing_sample",
        "state": "fresh_scene_build_warm_queries_and_refit",
        "stages": stages,
    }


def result(matrix):
    cases = []
    for index, dimensions in enumerate(expected_case_dimensions(matrix, "smoke")):
        performance = {
            metric: summary(unit)
            for metric, (unit, _) in PERFORMANCE_METRICS.items()
        }
        cases.append({
            "case_id": f"smoke-{index:04d}",
            "dimensions": dimensions,
            "performance": performance,
            "launch_audit": launch_audit(),
            "correctness": {"max_abs_error": 0.0, "max_rel_error": 0.0},
            "ad": {
                "vjp_max_abs_error": 0.0,
                "vjp_max_rel_error": 0.0,
                "jvp_max_abs_error": 0.0,
                "jvp_max_rel_error": 0.0,
            },
        })
    return {
        "schema_version": matrix["schema_version"],
        "matrix_id": matrix["matrix_id"],
        "benchmark": matrix["benchmark"],
        "seed": matrix["seed"],
        "profile": "smoke",
        "environment": {
            "gpu_name": "Test GPU",
            "gpu_compute_capability": "12.0",
            "cuda_runtime_version": "12.9",
            "cuda_driver_version": "580.0",
            "optix_version": "9.0",
            "compiler_id": "MSVC",
            "compiler_version": "19.44",
            "build_type": "Release",
            "git_commit": "abcdef0",
        },
        "tolerances": copy.deepcopy(matrix["tolerances"]),
        "cases": cases,
    }


def scale_metric(payload, metric, factor):
    current = payload["cases"][0]["performance"][metric]
    current["samples"] = [value * factor for value in current["samples"]]
    current["median"] *= factor
    current["p95"] *= factor


class EdgeBVHBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load_matrix()

    def test_full_matrix_is_frozen_and_complete(self):
        full = self.matrix["profiles"]["full"]
        self.assertEqual(full["edge_counts"], [1000, 16000, 64000, 111000, 500000, 2000000])
        self.assertEqual(full["query_counts"], [1, 256, 4096, 65536, 1000000])
        self.assertEqual(full["query_kinds"], ["point", "finite_ray", "infinite_ray"])
        self.assertEqual(full["top_k"], [1, 4, 8, 16])
        self.assertEqual(
            full["update_modes"],
            ["static", "full_refit", "dirty_refit_1pct", "dirty_refit_10pct", "dirty_refit_100pct"],
        )
        self.assertEqual(full["masks"], ["sparse", "dense"])
        self.assertEqual(full["distributions"], ["grid", "random", "long_thin"])
        self.assertEqual(self.matrix["seed"], 20260711)

    def test_smoke_profile_is_a_strict_subset(self):
        full = self.matrix["profiles"]["full"]
        smoke = self.matrix["profiles"]["smoke"]
        for dimension in full:
            self.assertLessEqual(set(smoke[dimension]), set(full[dimension]))
        self.assertLess(len(smoke["edge_counts"]), len(full["edge_counts"]))
        self.assertLess(len(smoke["query_counts"]), len(full["query_counts"]))

    def test_schema_and_matrix_require_reproducibility_metadata(self):
        fields = set(self.matrix["required_environment_fields"])
        self.assertTrue({
            "gpu_name", "cuda_runtime_version", "cuda_driver_version", "optix_version",
            "compiler_id", "compiler_version", "build_type", "git_commit",
        }.issubset(fields))
        self.assertGreaterEqual(self.matrix["measurement"]["minimum_timed_runs"], 5)
        self.assertEqual(self.matrix["measurement"]["required_statistics"], ["median", "p95"])
        self.assertEqual(
            self.matrix["measurement"]["launch_audit"]["method"],
            "independent_stable_audit",
        )
        self.assertTrue(self.matrix["measurement"]["launch_audit"]["timing_isolated"])

    def test_valid_result_passes_schema_validation(self):
        payload = result(self.matrix)
        self.assertEqual(len(payload["cases"]), 11)
        validate_result(payload, self.matrix)

    def test_full_and_smoke_expected_case_counts_are_fixed(self):
        self.assertEqual(len(expected_case_dimensions(self.matrix, "smoke")), 11)
        self.assertEqual(len(expected_case_dimensions(self.matrix, "full")), 22)

    def test_validation_rejects_incomplete_profile_coverage(self):
        payload = result(self.matrix)
        payload["cases"].pop()
        with self.assertRaisesRegex(ContractError, "expected 11 cases, got 10"):
            validate_result(payload, self.matrix)

    def test_validation_rejects_duplicate_dimensions(self):
        payload = result(self.matrix)
        payload["cases"][1]["dimensions"] = copy.deepcopy(payload["cases"][0]["dimensions"])
        with self.assertRaisesRegex(ContractError, "duplicate case dimensions"):
            validate_result(payload, self.matrix)

    def test_validation_rejects_missing_environment_field(self):
        payload = result(self.matrix)
        del payload["environment"]["optix_version"]
        with self.assertRaisesRegex(ContractError, "optix_version"):
            validate_result(payload, self.matrix)

    def test_validation_rejects_fewer_than_five_rounds(self):
        payload = result(self.matrix)
        metric = payload["cases"][0]["performance"]["hot_query_ms"]
        metric.update(samples=[1.0] * 4, median=1.0, p95=1.0)
        with self.assertRaisesRegex(ContractError, "needs 5 samples"):
            validate_result(payload, self.matrix)

    def test_validation_rejects_unreported_or_inconsistent_statistics(self):
        payload = result(self.matrix)
        del payload["cases"][0]["performance"]["build_ms"]["p95"]
        with self.assertRaisesRegex(ContractError, "build_ms.p95"):
            validate_result(payload, self.matrix)

        payload = result(self.matrix)
        payload["cases"][0]["performance"]["refit_ms"]["median"] = 99.0
        with self.assertRaisesRegex(ContractError, "inconsistent"):
            validate_result(payload, self.matrix)

    def test_launch_audit_requires_all_stages_and_consistent_counts(self):
        payload = result(self.matrix)
        del payload["cases"][0]["launch_audit"]["stages"]["query_infinite_ray"]
        with self.assertRaisesRegex(ContractError, "launch_audit stages are incomplete"):
            validate_result(payload, self.matrix)

        payload = result(self.matrix)
        payload["cases"][0]["launch_audit"]["stages"]["build"]["total_observed_launches"] += 1
        with self.assertRaisesRegex(ContractError, "total_observed_launches is inconsistent"):
            validate_result(payload, self.matrix)

    def test_legacy_launch_baseline_requires_explicit_compatibility_mode(self):
        baseline = result(self.matrix)
        candidate = copy.deepcopy(baseline)
        for case in baseline["cases"]:
            del case["launch_audit"]
        with self.assertRaisesRegex(ContractError, "legacy result"):
            evaluate_gate(baseline, candidate, self.matrix)

        report = evaluate_gate(
            baseline,
            candidate,
            self.matrix,
            allow_legacy_launch_baseline=True,
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(report["warnings"]), len(baseline["cases"]))

    def test_unexplained_launch_increase_fails_and_explanation_is_reported(self):
        baseline = result(self.matrix)
        candidate = copy.deepcopy(baseline)
        stage = candidate["cases"][0]["launch_audit"]["stages"]["query_point"]
        stage["drjit_kernel_launches"] += 1
        stage["total_observed_launches"] += 1
        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            failure["metric"] == "launch_audit.query_point.drjit_kernel_launches"
            and failure["reason"] == "unexplained launch-count regression"
            for failure in report["failures"]
        ))

        stage["increase_explanation"] = "One fused dispatch was split to preserve fixed-winner AD semantics."
        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertTrue(report["passed"], report)
        comparison = next(
            item for item in report["comparisons"]
            if item["metric"] == "launch_audit.query_point.drjit_kernel_launches"
        )
        self.assertEqual(comparison["increase"], 1)
        self.assertTrue(comparison["increase_explanation"])

    def test_top_k_is_restricted_to_point_queries(self):
        payload = result(self.matrix)
        payload["cases"][0]["dimensions"].update(query_kind="finite_ray", top_k=4)
        with self.assertRaisesRegex(ContractError, "top_k only applies"):
            validate_result(payload, self.matrix)

    def test_default_performance_limits_pass_just_below_threshold(self):
        baseline = result(self.matrix)
        candidate = copy.deepcopy(baseline)
        factors = {
            "hot_query_ms": 1.029,
            "build_ms": 1.049,
            "refit_ms": 1.049,
            "peak_device_memory_bytes": 1.049,
            "cold_create_ms": 1.099,
        }
        for metric, factor in factors.items():
            scale_metric(candidate, metric, factor)
        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertTrue(report["passed"], report)

    def test_exact_default_performance_limits_pass(self):
        baseline = result(self.matrix)
        candidate = copy.deepcopy(baseline)
        factors = {
            "hot_query_ms": 1.03,
            "build_ms": 1.05,
            "refit_ms": 1.05,
            "peak_device_memory_bytes": 1.05,
            "cold_create_ms": 1.10,
        }
        for metric, factor in factors.items():
            scale_metric(candidate, metric, factor)
        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertTrue(report["passed"], report)

    def test_each_default_performance_limit_is_enforced(self):
        limits = {
            "hot_query_ms": 0.03,
            "build_ms": 0.05,
            "refit_ms": 0.05,
            "peak_device_memory_bytes": 0.05,
            "cold_create_ms": 0.10,
        }
        for metric, limit in limits.items():
            with self.subTest(metric=metric):
                baseline = result(self.matrix)
                candidate = copy.deepcopy(baseline)
                scale_metric(candidate, metric, 1.0 + limit + 0.001)
                report = evaluate_gate(baseline, candidate, self.matrix)
                self.assertFalse(report["passed"])
                self.assertTrue(any(failure["metric"] == metric for failure in report["failures"]))

    def test_sub_10_microsecond_timing_jitter_is_below_the_noise_floor(self):
        baseline = result(self.matrix)
        candidate = copy.deepcopy(baseline)
        baseline["cases"][0]["performance"]["refit_ms"] = summary("ms", 0.00022)
        candidate["cases"][0]["performance"]["refit_ms"] = summary("ms", 0.00026)

        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertTrue(report["passed"], report)

        candidate["cases"][0]["performance"]["refit_ms"] = summary("ms", 0.002)
        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertFalse(report["passed"])
        failure = next(item for item in report["failures"] if item["metric"] == "refit_ms")
        self.assertEqual(failure["absolute_noise_floor"], 0.01)

    def test_correctness_and_ad_tolerances_are_enforced(self):
        for group, field in (
            ("correctness", "max_abs_error"),
            ("ad", "vjp_max_abs_error"),
            ("ad", "jvp_max_rel_error"),
        ):
            with self.subTest(group=group, field=field):
                baseline = result(self.matrix)
                candidate = copy.deepcopy(baseline)
                candidate["cases"][0][group][field] = self.matrix["tolerances"][group][field] * 2.0
                report = evaluate_gate(baseline, candidate, self.matrix)
                self.assertFalse(report["passed"])
                self.assertTrue(
                    any(failure["metric"] == f"{group}.{field}" for failure in report["failures"])
                )

    def test_zero_baseline_regression_is_well_defined(self):
        baseline = result(self.matrix)
        candidate = copy.deepcopy(baseline)
        for payload in (baseline, candidate):
            payload["cases"][0]["performance"]["hot_query_ms"] = summary("ms", 0.0)
        self.assertTrue(evaluate_gate(baseline, candidate, self.matrix)["passed"])

        candidate["cases"][0]["performance"]["hot_query_ms"] = summary("ms", 1.0)
        report = evaluate_gate(baseline, candidate, self.matrix)
        self.assertFalse(report["passed"])
        regression = next(
            failure["regression"]
            for failure in report["failures"]
            if failure["metric"] == "hot_query_ms"
        )
        self.assertIsNone(regression)


if __name__ == "__main__":
    unittest.main()
