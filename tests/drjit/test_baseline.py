import unittest

from tests.baseline_utils import (
    BASELINE_DIR,
    DEFAULT_TOLERANCE_POLICY,
    assert_close,
    load_expected_data,
    load_manifest,
    run_json_case,
)


class BaselineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BASELINE_DIR.exists():
            raise AssertionError(f"Missing baseline directory: {BASELINE_DIR}")

        cls.expected_data = load_expected_data()
        cls.baseline_manifest = load_manifest()
        cls.actual_data = run_json_case(
            """
            import json
            from tests.baseline_cases import collect_baseline_data

            print(json.dumps(collect_baseline_data(), sort_keys=True))
            """,
            timeout=300,
        )
        cls.tolerance_policy = cls.baseline_manifest.get("tolerance_policy", DEFAULT_TOLERANCE_POLICY)

    def test_baseline_manifest_is_present(self):
        self.assertEqual(self.baseline_manifest["baseline_version"], "drjit_v0_4_6")
        self.assertIn("drjit_version", self.baseline_manifest)
        self.assertIn("rayd_commit", self.baseline_manifest)

    def test_baseline_outputs_match(self):
        for section, expected_section in sorted(self.expected_data.items()):
            with self.subTest(section=section):
                assert_close(
                    self,
                    self.actual_data[section],
                    expected_section,
                    path=section,
                    policy=self.tolerance_policy,
                )


if __name__ == "__main__":
    unittest.main()
