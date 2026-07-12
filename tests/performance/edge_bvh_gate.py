from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "shared" / "benchmarks" / "edge_bvh_matrix.json"

PROFILE_KEYS = (
    "edge_counts",
    "query_counts",
    "query_kinds",
    "top_k",
    "update_modes",
    "masks",
    "distributions",
)
PERFORMANCE_METRICS = {
    "hot_query_ms": ("ms", "hot_query_max_regression"),
    "build_ms": ("ms", "build_max_regression"),
    "refit_ms": ("ms", "refit_max_regression"),
    "peak_device_memory_bytes": ("bytes", "memory_max_regression"),
    "cold_create_ms": ("ms", "cold_create_max_regression"),
}
LAUNCH_STAGES = (
    "build", "refit", "query_point", "query_finite_ray", "query_infinite_ray",
)
LAUNCH_COUNT_FIELDS = (
    "drjit_kernel_launches",
    "drjit_optix_launches",
    "native_cuda_kernel_launches",
    "native_cub_launches",
    "native_optix_launches",
    "native_optix_accel_operations",
)
LAUNCH_TOTAL_FIELDS = tuple(
    field for field in LAUNCH_COUNT_FIELDS if field != "drjit_optix_launches"
)
LAUNCH_AUDIT_CONTRACT = {
    "method": "independent_stable_audit",
    "timing_isolated": True,
    "runs": 1,
    "sampling": "single_deterministic_pass_not_timing_sample",
    "state": "fresh_scene_build_warm_queries_and_refit",
    "gate": "no_unexplained_component_increase",
}


class ContractError(ValueError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_matrix(path: str | Path = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = load_json(path)
    validate_matrix(matrix, Path(path).parent)
    return matrix


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_matrix(matrix: dict[str, Any], schema_dir: Path | None = None) -> None:
    _require(matrix.get("schema_version") == 1, "matrix.schema_version must be 1")
    _require(matrix.get("matrix_id") == "rayd_edge_bvh_v1", "unexpected matrix_id")
    _require(matrix.get("benchmark") == "rayd_edge_bvh", "unexpected benchmark name")
    _require(isinstance(matrix.get("seed"), int), "matrix.seed must be an integer")

    profiles = matrix.get("profiles", {})
    _require(set(profiles) == {"full", "smoke"}, "matrix must define full and smoke profiles")
    for profile_name, profile in profiles.items():
        _require(set(profile) == set(PROFILE_KEYS), f"{profile_name} profile dimensions are incomplete")
        for key in PROFILE_KEYS:
            values = profile[key]
            _require(isinstance(values, list) and values, f"{profile_name}.{key} must be non-empty")
            _require(len(values) == len(set(values)), f"{profile_name}.{key} contains duplicates")

    full = profiles["full"]
    expected_full = {
        "edge_counts": [1000, 16000, 64000, 111000, 500000, 2000000],
        "query_counts": [1, 256, 4096, 65536, 1000000],
        "query_kinds": ["point", "finite_ray", "infinite_ray"],
        "top_k": [1, 4, 8, 16],
        "update_modes": [
            "static", "full_refit", "dirty_refit_1pct",
            "dirty_refit_10pct", "dirty_refit_100pct",
        ],
        "masks": ["sparse", "dense"],
        "distributions": ["grid", "random", "long_thin"],
    }
    _require(full == expected_full, "full profile does not match the frozen BVH-0 matrix")
    for key in PROFILE_KEYS:
        _require(
            set(profiles["smoke"][key]).issubset(full[key]),
            f"smoke.{key} must be a subset of full.{key}",
        )

    contracts = matrix.get("dimension_contracts", {})
    for key in ("update_modes", "masks", "distributions"):
        _require(set(contracts.get(key, {})) == set(full[key]), f"dimension contract mismatch for {key}")

    measurement = matrix.get("measurement", {})
    _require(measurement.get("minimum_timed_runs", 0) >= 5, "at least five timed runs are required")
    _require(measurement.get("required_statistics") == ["median", "p95"], "median and p95 are required")
    _require(measurement.get("gate_statistic") == "median", "the default gate statistic must be median")
    _require(
        measurement.get("launch_audit") == LAUNCH_AUDIT_CONTRACT,
        "launch audit measurement contract does not match the stable independent audit",
    )

    required_environment = matrix.get("required_environment_fields", [])
    _require(len(required_environment) == len(set(required_environment)), "environment fields contain duplicates")
    for required in (
        "gpu_name", "cuda_runtime_version", "optix_version", "compiler_id",
        "compiler_version", "build_type", "git_commit",
    ):
        _require(required in required_environment, f"missing environment field: {required}")

    gate = matrix.get("performance_gate", {})
    expected_gate = {
        "hot_query_max_regression": 0.03,
        "build_max_regression": 0.05,
        "refit_max_regression": 0.05,
        "memory_max_regression": 0.05,
        "cold_create_max_regression": 0.10,
        "absolute_noise_floor_ms": 0.01,
    }
    _require(gate == expected_gate, "performance gate thresholds do not match BVH-0 defaults")

    tolerances = matrix.get("tolerances", {})
    _require(set(tolerances) == {"correctness", "ad"}, "correctness and AD tolerances are required")
    for group in tolerances.values():
        _require(all(_is_number(value) and value >= 0 for value in group.values()), "invalid tolerance")

    if schema_dir is not None:
        schema_path = schema_dir / matrix.get("result_schema", "")
        _require(schema_path.is_file(), f"result schema does not exist: {schema_path}")
        schema = load_json(schema_path)
        schema_environment = schema["properties"]["environment"]["required"]
        _require(schema_environment == required_environment, "schema and matrix environment fields differ")
        summary = schema["$defs"]["summary"]
        _require(
            summary["properties"]["samples"]["minItems"] == measurement["minimum_timed_runs"],
            "schema and matrix timed-run minimum differ",
        )
        launch_audit = schema["$defs"]["launch_audit"]
        _require(
            set(launch_audit["properties"]["stages"]["required"]) == set(LAUNCH_STAGES),
            "schema and matrix launch stages differ",
        )


def _validate_summary(summary: Any, metric: str, expected_unit: str, minimum_runs: int) -> None:
    _require(isinstance(summary, dict), f"{metric} summary must be an object")
    _require(summary.get("unit") == expected_unit, f"{metric} must use {expected_unit}")
    samples = summary.get("samples")
    _require(isinstance(samples, list) and len(samples) >= minimum_runs, f"{metric} needs {minimum_runs} samples")
    _require(all(_is_number(value) and value >= 0 for value in samples), f"{metric} samples must be finite and non-negative")
    for statistic in ("median", "p95"):
        _require(_is_number(summary.get(statistic)), f"{metric}.{statistic} is required")
    expected_median = statistics.median(samples)
    ordered = sorted(samples)
    expected_p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    _require(math.isclose(summary["median"], expected_median, rel_tol=1e-12, abs_tol=1e-12), f"{metric}.median is inconsistent with samples")
    _require(math.isclose(summary["p95"], expected_p95, rel_tol=1e-12, abs_tol=1e-12), f"{metric}.p95 is inconsistent with samples")


def expected_case_dimensions(matrix: dict[str, Any], profile_name: str) -> list[dict[str, Any]]:
    profile = matrix["profiles"][profile_name]
    base = {
        "edge_count": profile["edge_counts"][0],
        "query_count": profile["query_counts"][0],
        "query_kind": "point",
        "top_k": 1,
        "update_mode": profile["update_modes"][0],
        "mask": profile["masks"][0],
        "distribution": profile["distributions"][0],
    }
    cases = [base]
    axes = (
        ("edge_count", "edge_counts"),
        ("query_count", "query_counts"),
        ("query_kind", "query_kinds"),
        ("top_k", "top_k"),
        ("update_mode", "update_modes"),
        ("mask", "masks"),
        ("distribution", "distributions"),
    )
    for result_key, profile_key in axes:
        for value in profile[profile_key]:
            if value == base[result_key]:
                continue
            dimensions = dict(base)
            dimensions[result_key] = value
            if result_key == "query_kind" and value != "point":
                dimensions["top_k"] = 1
            elif result_key == "top_k":
                dimensions["query_kind"] = "point"
            cases.append(dimensions)
    return cases


def dimension_key(dimensions: dict[str, Any]) -> tuple[Any, ...]:
    return (
        dimensions["edge_count"], dimensions["query_count"],
        dimensions["query_kind"], dimensions["top_k"],
        dimensions["update_mode"], dimensions["mask"],
        dimensions["distribution"],
    )


def _validate_launch_audit(audit: Any, case_id: str, allow_missing: bool) -> bool:
    if audit is None:
        _require(
            allow_missing,
            f"{case_id}.launch_audit is required; this is a legacy result. "
            "Regenerate it, or use --allow-legacy-launch-baseline only for the baseline.",
        )
        return False
    _require(isinstance(audit, dict), f"{case_id}.launch_audit must be an object")
    for field in ("method", "timing_isolated", "runs", "sampling", "state"):
        _require(
            audit.get(field) == LAUNCH_AUDIT_CONTRACT[field],
            f"{case_id}.launch_audit.{field} does not match the measurement contract",
        )
    stages = audit.get("stages")
    _require(isinstance(stages, dict) and set(stages) == set(LAUNCH_STAGES), f"{case_id}.launch_audit stages are incomplete")
    for stage_name, stage in stages.items():
        _require(isinstance(stage, dict), f"{case_id}.launch_audit.{stage_name} must be an object")
        allowed = set(LAUNCH_COUNT_FIELDS) | {"total_observed_launches", "increase_explanation"}
        _require(set(stage).issubset(allowed), f"{case_id}.launch_audit.{stage_name} has unknown fields")
        _require(set(LAUNCH_COUNT_FIELDS) | {"total_observed_launches"} <= set(stage), f"{case_id}.launch_audit.{stage_name} counts are incomplete")
        for field in (*LAUNCH_COUNT_FIELDS, "total_observed_launches"):
            value = stage[field]
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{case_id}.launch_audit.{stage_name}.{field} must be a non-negative integer")
        _require(
            stage["total_observed_launches"] == sum(stage[field] for field in LAUNCH_TOTAL_FIELDS),
            f"{case_id}.launch_audit.{stage_name}.total_observed_launches is inconsistent",
        )
        if "increase_explanation" in stage:
            explanation = stage["increase_explanation"]
            _require(isinstance(explanation, str) and explanation.strip(), f"{case_id}.launch_audit.{stage_name}.increase_explanation must be non-empty")
    return True


def validate_result(
    result: dict[str, Any],
    matrix: dict[str, Any],
    *,
    allow_legacy_launch_audit: bool = False,
) -> None:
    for key in ("schema_version", "matrix_id", "benchmark", "seed"):
        _require(result.get(key) == matrix[key], f"result.{key} does not match matrix")
    profile_name = result.get("profile")
    _require(profile_name in matrix["profiles"], "result.profile is invalid")
    profile = matrix["profiles"][profile_name]

    environment = result.get("environment")
    _require(isinstance(environment, dict), "result.environment must be an object")
    for field in matrix["required_environment_fields"]:
        _require(isinstance(environment.get(field), str) and environment[field], f"missing environment field: {field}")
    _require(re.fullmatch(r"[0-9a-fA-F]{7,40}", environment["git_commit"]) is not None, "git_commit must be a 7-40 digit hex hash")
    _require(result.get("tolerances") == matrix["tolerances"], "result tolerances must match the matrix")

    cases = result.get("cases")
    _require(isinstance(cases, list) and cases, "result.cases must be non-empty")
    case_ids: set[str] = set()
    actual_dimensions: set[tuple[Any, ...]] = set()
    minimum_runs = matrix["measurement"]["minimum_timed_runs"]
    for case in cases:
        case_id = case.get("case_id")
        _require(isinstance(case_id, str) and case_id, "case_id must be a non-empty string")
        _require(case_id not in case_ids, f"duplicate case_id: {case_id}")
        case_ids.add(case_id)

        dimensions = case.get("dimensions", {})
        dimension_map = {
            "edge_count": "edge_counts", "query_count": "query_counts",
            "query_kind": "query_kinds", "top_k": "top_k",
            "update_mode": "update_modes", "mask": "masks",
            "distribution": "distributions",
        }
        for result_key, profile_key in dimension_map.items():
            _require(dimensions.get(result_key) in profile[profile_key], f"{case_id}.{result_key} is outside the {profile_name} profile")
        if dimensions["query_kind"] != "point":
            _require(dimensions["top_k"] == 1, f"{case_id}: top_k only applies to point queries")
        key = dimension_key(dimensions)
        _require(key not in actual_dimensions, f"duplicate case dimensions: {dimensions}")
        actual_dimensions.add(key)

        performance = case.get("performance", {})
        _require(set(performance) == set(PERFORMANCE_METRICS), f"{case_id}.performance metrics are incomplete")
        for metric, (unit, _) in PERFORMANCE_METRICS.items():
            _validate_summary(performance[metric], metric, unit, minimum_runs)

        _validate_launch_audit(case.get("launch_audit"), case_id, allow_legacy_launch_audit)

        for group_name in ("correctness", "ad"):
            group = case.get(group_name)
            expected_fields = matrix["tolerances"][group_name]
            _require(isinstance(group, dict) and set(group) == set(expected_fields), f"{case_id}.{group_name} fields are incomplete")
            _require(all(_is_number(value) and value >= 0 for value in group.values()), f"{case_id}.{group_name} values are invalid")

    expected_dimensions = {
        dimension_key(dimensions)
        for dimensions in expected_case_dimensions(matrix, profile_name)
    }
    _require(
        actual_dimensions == expected_dimensions,
        f"{profile_name} profile coverage mismatch: expected {len(expected_dimensions)} cases, got {len(actual_dimensions)}",
    )


def _regression(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - baseline) / baseline


def evaluate_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    matrix: dict[str, Any],
    *,
    allow_legacy_launch_baseline: bool = False,
) -> dict[str, Any]:
    validate_result(
        baseline,
        matrix,
        allow_legacy_launch_audit=allow_legacy_launch_baseline,
    )
    validate_result(candidate, matrix)
    _require(baseline["profile"] == candidate["profile"], "profiles differ")
    for field in matrix["required_environment_fields"]:
        if field != "git_commit":
            _require(baseline["environment"][field] == candidate["environment"][field], f"environment mismatch: {field}")

    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    candidate_cases = {case["case_id"]: case for case in candidate["cases"]}
    _require(set(baseline_cases) == set(candidate_cases), "baseline and candidate case sets differ")

    failures: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    gate_statistic = matrix["measurement"]["gate_statistic"]
    for case_id in sorted(baseline_cases):
        base_case = baseline_cases[case_id]
        candidate_case = candidate_cases[case_id]
        _require(base_case["dimensions"] == candidate_case["dimensions"], f"dimensions differ: {case_id}")
        for metric, (unit, threshold_key) in PERFORMANCE_METRICS.items():
            baseline_value = base_case["performance"][metric][gate_statistic]
            candidate_value = candidate_case["performance"][metric][gate_statistic]
            regression = _regression(baseline_value, candidate_value)
            limit = matrix["performance_gate"][threshold_key]
            absolute_regression = candidate_value - baseline_value
            absolute_noise_floor = (
                matrix["performance_gate"]["absolute_noise_floor_ms"]
                if unit == "ms"
                else 0.0
            )
            comparison = {
                "case_id": case_id,
                "metric": metric,
                "statistic": gate_statistic,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "regression": regression,
                "limit": limit,
                "absolute_regression": absolute_regression,
                "absolute_noise_floor": absolute_noise_floor,
            }
            comparisons.append(comparison)
            exceeds_relative_limit = regression is None or (
                regression > limit
                and not math.isclose(regression, limit, rel_tol=1e-12, abs_tol=1e-12)
            )
            exceeds_limit = (
                exceeds_relative_limit
                and absolute_regression > absolute_noise_floor
                and not math.isclose(
                    absolute_regression,
                    absolute_noise_floor,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            if exceeds_limit:
                failures.append({**comparison, "reason": "performance regression"})

        baseline_audit = base_case.get("launch_audit")
        if baseline_audit is None:
            warning = (
                "legacy baseline has no launch_audit; launch regression checks were skipped "
                f"for {case_id}"
            )
            warnings.append(warning)
        else:
            for stage_name in LAUNCH_STAGES:
                base_stage = baseline_audit["stages"][stage_name]
                candidate_stage = candidate_case["launch_audit"]["stages"][stage_name]
                explanation = candidate_stage.get("increase_explanation")
                for field in LAUNCH_COUNT_FIELDS:
                    baseline_value = base_stage[field]
                    candidate_value = candidate_stage[field]
                    increased = candidate_value > baseline_value
                    comparison = {
                        "case_id": case_id,
                        "metric": f"launch_audit.{stage_name}.{field}",
                        "baseline": baseline_value,
                        "candidate": candidate_value,
                        "increase": candidate_value - baseline_value,
                        "increase_explanation": explanation,
                    }
                    comparisons.append(comparison)
                    if increased and explanation is None:
                        failures.append({**comparison, "reason": "unexplained launch-count regression"})

        for group_name in ("correctness", "ad"):
            for field, limit in matrix["tolerances"][group_name].items():
                value = candidate_case[group_name][field]
                if value > limit:
                    failures.append({
                        "case_id": case_id,
                        "metric": f"{group_name}.{field}",
                        "candidate": value,
                        "limit": limit,
                        "reason": "tolerance exceeded",
                    })

    return {
        "passed": not failures,
        "failures": failures,
        "comparisons": comparisons,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and gate RayD edge BVH benchmark JSON.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--allow-legacy-launch-baseline",
        action="store_true",
        help="accept a baseline without launch_audit and skip only its launch-count comparisons",
    )
    args = parser.parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
        report = evaluate_gate(
            load_json(args.baseline),
            load_json(args.candidate),
            matrix,
            allow_legacy_launch_baseline=args.allow_legacy_launch_baseline,
        )
    except (ContractError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "validation_error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
