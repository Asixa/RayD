"""Comparison policy for golden scene results.

Discrete fields are compared for exact equality; continuous fields use the
tolerances declared in ``shared/contracts/operations.json``; informative fields
are recorded in the baseline but never compared. These helpers run in the
parent (host) process and need no GPU.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = ROOT / "shared" / "contracts" / "operations.json"
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"


def continuous_tolerances():
    tolerances = json.loads(OPERATIONS.read_text(encoding="utf-8"))["tolerances"]
    return tolerances["default_abs"], tolerances["default_rel"]


def baseline_scene(name, backend="optix"):
    return json.loads((BASELINE_DIR / backend / f"{name}.json").read_text(encoding="utf-8"))


def baseline_scene_names(backend="optix"):
    return sorted(p.stem for p in (BASELINE_DIR / backend).glob("*.json") if p.stem != "manifest")


def _stripped(run):
    """Drop informative buckets and query metadata, keep discrete + continuous."""
    out = {}
    for scene_name, scene_data in run.items():
        queries = {}
        for query_name, record in scene_data["queries"].items():
            queries[query_name] = {
                "discrete": record["discrete"],
                "continuous": record["continuous"],
            }
        out[scene_name] = queries
    return out


def assert_run_to_run_identical(testcase, run_a, run_b):
    """Discrete AND continuous fields must be bitwise identical across two runs."""
    text_a = json.dumps(_stripped(run_a), sort_keys=True)
    text_b = json.dumps(_stripped(run_b), sort_keys=True)
    testcase.assertEqual(text_a, text_b, "golden collection is not deterministic run-to-run")


def _assert_continuous(testcase, actual, expected, path, abs_tol, rel_tol):
    if isinstance(expected, dict):
        testcase.assertIsInstance(actual, dict, path)
        testcase.assertEqual(set(actual.keys()), set(expected.keys()), path)
        for key in sorted(expected.keys()):
            _assert_continuous(testcase, actual[key], expected[key], f"{path}.{key}", abs_tol, rel_tol)
        return
    if isinstance(expected, list):
        testcase.assertIsInstance(actual, list, path)
        testcase.assertEqual(len(actual), len(expected), path)
        for index, (a, e) in enumerate(zip(actual, expected)):
            _assert_continuous(testcase, a, e, f"{path}[{index}]", abs_tol, rel_tol)
        return
    actual_value = float(actual)
    expected_value = float(expected)
    if math.isnan(expected_value):
        testcase.assertTrue(math.isnan(actual_value), path)
        return
    if math.isinf(expected_value):
        testcase.assertEqual(math.isinf(actual_value), True, path)
        testcase.assertEqual(math.copysign(1.0, actual_value), math.copysign(1.0, expected_value), path)
        return
    testcase.assertTrue(
        math.isclose(actual_value, expected_value, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{path}: actual={actual_value!r}, expected={expected_value!r}, "
        f"abs_tol={abs_tol}, rel_tol={rel_tol}",
    )


def assert_scene_matches_baseline(testcase, actual_scene, baseline_scene_data, name, abs_tol, rel_tol):
    actual_queries = actual_scene["queries"]
    baseline_queries = baseline_scene_data["queries"]
    testcase.assertEqual(
        set(actual_queries.keys()), set(baseline_queries.keys()), f"{name}: query set drift"
    )
    for query_name, baseline_record in baseline_queries.items():
        actual_record = actual_queries[query_name]
        path = f"{name}.{query_name}"
        testcase.assertEqual(actual_record["kind"], baseline_record["kind"], f"{path}: kind")
        testcase.assertEqual(
            actual_record["discrete"], baseline_record["discrete"], f"{path}: discrete"
        )
        _assert_continuous(
            testcase, actual_record["continuous"], baseline_record["continuous"],
            f"{path}.continuous", abs_tol, rel_tol,
        )
