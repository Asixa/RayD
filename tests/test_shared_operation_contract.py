import json
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "shared" / "contracts" / "operations.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class SharedOperationContractTests(unittest.TestCase):
    def test_schema_v2_has_per_operation_contracts(self):
        self.assertEqual(CONTRACT["version"], 2)
        operations = CONTRACT["operations"]
        self.assertEqual(
            set(operations),
            {
                "intersect",
                "nearest_edge_point",
                "nearest_edge_ray",
                "nearest_edges_topk",
                "visibility",
                "visibility_pair",
                "reflection_trace",
                "reflection_accumulation",
                "diffraction_direct",
                "diffraction_chain",
            },
        )
        for name, operation in operations.items():
            with self.subTest(operation=name):
                self.assertTrue(operation["inputs"])
                self.assertIn("result", operation)
                self.assertIn("ad", operation)

    def test_capability_names_cover_operation_names(self):
        required = set(CONTRACT["required_capability_keys"])
        self.assertLessEqual(set(CONTRACT["operations"]), required)

    def test_invalid_values_and_ray_flags_are_canonical(self):
        constants = CONTRACT["constants"]
        self.assertEqual(constants["invalid_signed_id"], -1)
        self.assertEqual(constants["invalid_unsigned_id"], 0xFFFFFFFF)
        self.assertEqual(
            constants["ray_flags"],
            {"None": 0, "Geometric": 1, "ShadingN": 2, "UV": 4, "All": 7},
        )

    def test_intersection_field_order_matches_torch_public_result(self):
        source = (ROOT / "backends" / "torch" / "python" / "rayd" / "torch" / "types.py").read_text(
            encoding="utf-8"
        )
        block = source[source.index("class Intersection:") : source.index("    def is_valid", source.index("class Intersection:"))]
        fields = re.findall(r"^    ([a-z][a-z0-9_]*): torch\.Tensor$", block, re.MULTILINE)
        self.assertEqual(fields, CONTRACT["result_contracts"]["intersection"]["canonical_fields"])

    def test_backend_specific_intersection_epsilon_is_explicit(self):
        overrides = CONTRACT["operations"]["intersect"]["backend_overrides"]
        self.assertEqual(overrides["drjit"]["ray_tmin"], 1e-3)
        self.assertEqual(overrides["torch"]["ray_tmin"], 1e-6)
        self.assertNotEqual(overrides["drjit"], overrides["torch"])

    def test_result_differences_are_extensions_not_canonical_reordering(self):
        point = CONTRACT["result_contracts"]["nearest_edge_point"]
        ray = CONTRACT["result_contracts"]["nearest_edge_ray"]
        self.assertNotIn("is_boundary", point["backend_fields"]["torch"])
        self.assertNotIn("is_boundary", ray["backend_fields"]["torch"])
        self.assertIn("is_boundary", point["backend_fields"]["drjit"])
        self.assertIn("is_boundary", ray["backend_fields"]["drjit"])
        self.assertLessEqual(set(point["canonical_semantics"]), set(point["backend_fields"]["torch"]))
        self.assertLessEqual(set(ray["canonical_semantics"]), set(ray["backend_fields"]["torch"]))

    def test_tensor_invalid_ids_match_constant_contract(self):
        invalid = CONTRACT["tensor_contract"]["invalid"]
        signed_invalid = CONTRACT["constants"]["invalid_signed_id"]
        self.assertEqual(invalid["shape_id"], signed_invalid)
        self.assertEqual(invalid["prim_id"], signed_invalid)
        self.assertEqual(invalid["edge_id"], signed_invalid)


if __name__ == "__main__":
    unittest.main()
