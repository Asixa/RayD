import json
import math
import numbers
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "tests" / "baselines" / "drjit_v0_4_6"

DEFAULT_TOLERANCE_POLICY = {
    "default_abs": 1e-6,
    "default_rel": 1e-6,
    "gradient_abs": 5e-5,
    "gradient_rel": 5e-5,
    "matrix_abs": 1e-5,
    "matrix_rel": 1e-5,
}

SECTION_FILES = {
    "geometry/constant_hit.json": ("geometry", "constant_hit"),
    "geometry/miss.json": ("geometry", "miss"),
    "geometry/multi_mesh.json": ("geometry", "multi_mesh"),
    "geometry/uv.json": ("geometry", "uv"),
    "geometry/batched_hits.json": ("geometry", "batched_hits"),
    "geometry/degenerate.json": ("geometry", "degenerate"),
    "gradients/vertex_gradients.json": ("gradients", "vertex_gradients"),
    "gradients/transform_gradients.json": ("gradients", "transform_gradients"),
    "transforms/matrix_entry_checks.json": ("transforms", "matrix_entry_checks"),
    "transforms/point_transform_checks.json": ("transforms", "point_transform_checks"),
    "transforms/direction_transform_checks.json": ("transforms", "direction_transform_checks"),
    "transforms/composition_left_right.json": ("transforms", "composition_left_right"),
    "camera/perspective_fov.json": ("camera", "perspective_fov"),
    "camera/perspective_intrinsic.json": ("camera", "perspective_intrinsic"),
    "camera/sample_primary_ray.json": ("camera", "sample_primary_ray"),
    "camera/world_to_sample.json": ("camera", "world_to_sample"),
    "camera/sample_to_world.json": ("camera", "sample_to_world"),
    "edges/secondary_edges.json": ("edges", "secondary_edges"),
    "edges/primary_edge_sampling.json": ("edges", "primary_edge_sampling"),
    "stress/repeated_run_summary.json": ("stress", "repeated_run_summary"),
}


def run_script(script: str, timeout: int = 180, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "Subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def run_json_case(script: str, timeout: int = 180):
    result = run_script(script, timeout=timeout, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON output.\nSTDERR:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Failed to parse JSON from subprocess.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc


def bundle_to_file_map(data):
    result = {}
    for rel_path, key_path in SECTION_FILES.items():
        value = data
        for key in key_path:
            value = value[key]
        result[rel_path] = value
    return result


def write_baseline_tree(manifest, data, out_dir=BASELINE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for rel_path, payload in bundle_to_file_map(data).items():
        path = out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_expected_data(base_dir=BASELINE_DIR):
    result = {}
    for rel_path, key_path in SECTION_FILES.items():
        payload = json.loads((base_dir / rel_path).read_text(encoding="utf-8"))
        cursor = result
        for key in key_path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[key_path[-1]] = payload
    return result


def load_manifest(base_dir=BASELINE_DIR):
    return json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))


def tolerance_for(path: str, policy):
    if path.startswith("gradients."):
        return policy["gradient_abs"], policy["gradient_rel"]
    if path.startswith("transforms.") or path.startswith("camera."):
        return policy["matrix_abs"], policy["matrix_rel"]
    return policy["default_abs"], policy["default_rel"]


def assert_close(testcase, actual, expected, path="root", policy=None):
    if policy is None:
        policy = DEFAULT_TOLERANCE_POLICY

    if isinstance(expected, dict):
        testcase.assertIsInstance(actual, dict, path)
        testcase.assertEqual(set(actual.keys()), set(expected.keys()), path)
        for key in sorted(expected.keys()):
            next_path = key if path == "root" else f"{path}.{key}"
            assert_close(testcase, actual[key], expected[key], next_path, policy)
        return

    if isinstance(expected, list):
        testcase.assertIsInstance(actual, list, path)
        testcase.assertEqual(len(actual), len(expected), path)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            next_path = f"{path}[{index}]"
            assert_close(testcase, actual_item, expected_item, next_path, policy)
        return

    if isinstance(expected, bool):
        testcase.assertIsInstance(actual, bool, path)
        testcase.assertEqual(actual, expected, path)
        return

    if isinstance(expected, numbers.Integral):
        testcase.assertEqual(actual, expected, path)
        return

    if isinstance(expected, numbers.Real):
        testcase.assertIsInstance(actual, numbers.Real, path)
        if math.isnan(expected):
            testcase.assertTrue(math.isnan(actual), path)
            return
        if math.isinf(expected):
            testcase.assertEqual(math.isinf(actual), True, path)
            testcase.assertEqual(math.copysign(1.0, actual), math.copysign(1.0, expected), path)
            return

        abs_tol, rel_tol = tolerance_for(path, policy)
        testcase.assertTrue(
            math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
            f"{path}: actual={actual!r}, expected={expected!r}, abs_tol={abs_tol}, rel_tol={rel_tol}",
        )
        return

    testcase.assertEqual(actual, expected, path)
