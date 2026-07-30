# Copyright Xingyu Chen.
# Tests contract schema validation.

import json
import unittest
from pathlib import Path

from tests._schema_validate import SchemaValidationError, validate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "contracts"


def _load(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


class ContractSchemaValidationTests(unittest.TestCase):
    def test_public_api_manifest_matches_its_schema(self):
        validate(_load("public_api.json"), _load("public_api.schema.json"))

    def test_path_exchange_matches_its_schema(self):
        validate(_load("path_exchange.json"), _load("path_exchange.schema.json"))

    def test_broken_instances_are_rejected(self):
        schema = _load("public_api.schema.json")

        missing_required = _load("public_api.json")
        del missing_required["trace"]

        wrong_version_type = _load("public_api.json")
        wrong_version_type["version"] = "2"

        unexpected_property = _load("public_api.json")
        unexpected_property["unexpected"] = True

        invalid_typing_enum = _load("public_api.json")
        invalid_typing_enum["backends"]["drjit"]["typing"] = "full"

        invalid_integration_mode = _load("public_api.json")
        invalid_integration_mode["trace"]["integration_modes"] = ["jit_symbolic", "reactive_native"]

        invalid_derivative_status = _load("public_api.json")
        invalid_derivative_status["backends"]["torch"]["derivatives"]["reflection_trace"]["trace_refl_epc"][
            "input_domains"
        ]["receiver"]["vjp"] = "partial"

        missing_derivative_mode = _load("public_api.json")
        del missing_derivative_mode["backends"]["torch"]["derivatives"]["reflection_trace"]["trace_reflections"][
            "input_domains"
        ]["ray"]["jvp"]

        empty_input_domains = _load("public_api.json")
        empty_input_domains["backends"]["torch"]["derivatives"]["intersect"]["intersect"]["input_domains"] = {}

        cases = {
            "missing_required_trace": missing_required,
            "wrong_version_type": wrong_version_type,
            "unexpected_top_level_property": unexpected_property,
            "invalid_typing_enum": invalid_typing_enum,
            "invalid_integration_mode": invalid_integration_mode,
            "invalid_derivative_status": invalid_derivative_status,
            "missing_derivative_mode": missing_derivative_mode,
            "empty_input_domains": empty_input_domains,
        }
        for label, broken in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(SchemaValidationError):
                    validate(broken, schema)

    def test_validator_rejects_unimplemented_schema_growth(self):
        with self.assertRaises(NotImplementedError):
            validate({"a": 1}, {"type": "object", "patternProperties": {"^a$": {"type": "integer"}}})
        with self.assertRaises(NotImplementedError):
            validate(None, {"type": "null"})


if __name__ == "__main__":
    unittest.main()
