# Copyright Xingyu Chen.
# Tests shared contracts.

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "include" / "rayd" / "contracts.h"


class SharedContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = CONTRACT.read_text(encoding="utf-8")

    def test_header_is_backend_neutral_and_pod_checked(self):
        self.assertTrue(CONTRACT.is_file())
        lowered = self.header.lower()
        for forbidden in ("torch/", "at::tensor", "drjit", "nanobind", "#include <optix", "cuda_runtime"):
            self.assertNotIn(forbidden, lowered)
        for enum_name in (
            "RayFlagBits",
            "IntersectionField",
            "NearestPointEdgeField",
            "NearestRayEdgeField",
            "NearestEdgesTopKField",
        ):
            self.assertIn(f"RAYD_SHARED_ASSERT_ENUM_POD({enum_name})", self.header)

    def test_scalar_and_field_contracts_are_frozen(self):
        required = {
            "InvalidSignedId": "-1",
            "InvalidUnsignedId": "0xffffffffu",
            "GeneralEpsilon": "1.0e-5f",
            "RayEpsilon": "1.0e-3f",
            "ShadowEpsilon": "1.0e-3f",
            "EdgeEpsilon": "1.0e-5f",
            "SmallEpsilon": "1.0e-6f",
            "VacuumPermittivity": "8.854187817e-12f",
        }
        for name, value in required.items():
            self.assertRegex(self.header, rf"\b{name}\s*=\s*{re.escape(value)}\s*;")
        self.assertIn("IntersectionField::Count) == 10u", self.header)
        self.assertIn("NearestPointEdgeField::Count) == 8u", self.header)
        self.assertIn("NearestRayEdgeField::Count) == 9u", self.header)

    def test_torch_python_flags_come_from_contract_mirror(self):
        source_path = ROOT / "python" / "rayd" / "_impl" / "geometry.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assignments = {
            target.id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        mirror = ast.literal_eval(assignments["_CONTRACT_VALUES"])
        self.assertEqual(mirror["invalid_signed_id"], -1)
        self.assertEqual(mirror["invalid_unsigned_id"], 0xFFFFFFFF)
        self.assertEqual(mirror["ray_flags_all"], 0x07)
        self.assertEqual(mirror["intersection_field_count"], 10)

    def test_backends_consume_shared_values_without_merging_ray_tmin(self):
        sources = {
            "drjit_rayd": ROOT / "include" / "rayd" / "jit" / "core.h",
            "torch_forward": ROOT / "src" / "scene" / "intersection.cu",
            "torch_backward": ROOT / "src" / "scene" / "intersection.cu",
            "torch_intersect": ROOT / "src" / "scene" / "intersection_optix.cu",
            "torch_edge": ROOT / "src" / "edge" / "edge_optix.cu",
            "drjit_edge": ROOT / "src" / "edge" / "edge_optix_jit.cu",
        }
        text = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}
        self.assertIn("shared::RayEpsilon", text["drjit_rayd"])
        self.assertIn("shared::RayFlagBits", text["torch_forward"])
        self.assertIn("shared::RayFlagBits", text["torch_backward"])
        self.assertIn("shared::SmallEpsilon", text["torch_intersect"])
        self.assertIn("shared::EdgeEpsilon", text["torch_edge"])
        self.assertIn("shared::EdgeEpsilon", text["drjit_edge"])

    def test_drjit_core_header_owns_transcendental_definitions(self):
        core = (ROOT / "include" / "rayd" / "jit" / "core.h").read_text(encoding="utf-8")
        math_include = core.find("#include <drjit/math.h>")
        public_include = core.find("#include <rayd/contracts.h>")
        transform_namespace = core.find("namespace transform {")
        self.assertEqual(core.count("#include <drjit/math.h>"), 1)
        self.assertGreaterEqual(math_include, 0)
        self.assertGreater(public_include, math_include)
        self.assertGreater(transform_namespace, math_include)


if __name__ == "__main__":
    unittest.main()
