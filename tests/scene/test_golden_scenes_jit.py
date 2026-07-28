"""Golden-scene regression test for the drjit backend.

Collects the declarative golden scenes (``tests/golden``) twice in two isolated
subprocesses and asserts (a) the discrete and continuous results are bitwise
identical run-to-run and (b) they match the checked-in OptiX baselines under the
comparison policy: discrete exact, continuous within operations.json tolerances,
informative fields skipped.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `tests.golden` lives at the repository root; see the root tests package
# for why it resolves from here under both documented invocations.
from tests.golden import compare  # noqa: E402  (needs ROOT on sys.path first)


def _collect_subprocess(timeout: int = 300):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    script = textwrap.dedent(
        """
        import json
        from tests.golden.runner import collect_golden
        print(json.dumps(collect_golden(), sort_keys=True))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Golden collection subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Golden collection produced no JSON.\nSTDERR:\n{result.stderr}")
    return json.loads(lines[-1])


class GoldenSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_a = _collect_subprocess()
        cls.run_b = _collect_subprocess()

    def test_collection_is_bit_identical_run_to_run(self):
        compare.assert_run_to_run_identical(self, self.run_a, self.run_b)

    def test_results_match_checked_in_baselines(self):
        abs_tol, rel_tol = compare.continuous_tolerances()
        baseline_names = compare.baseline_scene_names()
        self.assertEqual(
            set(self.run_a.keys()), set(baseline_names), "scene set drift vs checked-in baselines"
        )
        for name in baseline_names:
            baseline = compare.baseline_scene(name)
            compare.assert_scene_matches_baseline(
                self, self.run_a[name], baseline, name, abs_tol, rel_tol
            )


if __name__ == "__main__":
    unittest.main()
