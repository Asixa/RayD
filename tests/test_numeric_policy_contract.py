import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_H = ROOT / "include" / "rayd" / "detail" / "contracts.h"
POLICY_H = ROOT / "include" / "rayd" / "detail" / "rt" / "numeric_policy.h"
OPERATIONS = json.loads(
    (ROOT / "contracts" / "operations.json").read_text(encoding="utf-8")
)


def parse_float_literal(token: str) -> float:
    return float(token.strip().rstrip("fF"))


def contracts_constants() -> dict[str, float]:
    text = CONTRACTS_H.read_text(encoding="utf-8")
    names = ("GeneralEpsilon", "RayEpsilon", "ShadowEpsilon", "SmallEpsilon")
    values = {}
    for name in names:
        match = re.search(rf"\b{name}\s*=\s*([0-9.eE+-]+f)\s*;", text)
        values[name] = parse_float_literal(match.group(1))
    return values


def resolve_token(token: str, consts: dict[str, float]):
    token = token.strip()
    if token == "false":
        return False
    if token == "true":
        return True
    if token in consts:
        return consts[token]
    return parse_float_literal(token)


class NumericPolicyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = POLICY_H.read_text(encoding="utf-8")
        cls.consts = contracts_constants()
        cls.policy = OPERATIONS["numeric_policy"]

    def profile_fields(self, name: str) -> list:
        match = re.search(rf"{name}\{{([^}}]*)\}}", self.header)
        tokens = [token.strip() for token in match.group(1).split(",")]
        return [resolve_token(token, self.consts) for token in tokens]

    def family_constant(self, name: str) -> float:
        match = re.search(rf"\b{name}\s*=\s*([0-9.eE+-]+f)\s*;", self.header)
        return parse_float_literal(match.group(1))

    def test_header_backend_profiles_match_operations_json(self):
        field_order = [
            "ray_tmin",
            "shadow_tmin",
            "endpoint_offset",
            "parallel_epsilon",
            "watertight_triangles",
        ]
        drjit = dict(zip(field_order, self.profile_fields("kDrJitLegacyProfile")))
        torch = dict(zip(field_order, self.profile_fields("kTorchLegacyProfile")))
        json_drjit = self.policy["backend_profiles"]["drjit"]
        json_torch = self.policy["backend_profiles"]["torch"]
        for field in field_order:
            self.assertEqual(drjit[field], json_drjit[field])
            self.assertEqual(torch[field], json_torch[field])

    def test_header_family_constants_match_operations_json(self):
        mapping = {
            "kMultipathTraceTMin": "trace_tmin",
            "kTraceTMaxFinite": "trace_tmax_finite",
            "kMultipathRayBias": "ray_bias",
            "kMinSegmentLength": "min_segment_length",
            "kEpcBarycentricSlack": "epc_barycentric_slack",
            "kNormalizeFloor": "normalize_floor",
            "kEdgeDistanceEpsilon": "edge_distance_epsilon",
        }
        shared = self.policy["shared_multipath"]
        for header_name, json_key in mapping.items():
            self.assertEqual(self.family_constant(header_name), shared[json_key])

    def test_surfel_endpoint_offset_is_shadow_epsilon(self):
        match = re.search(r"kSurfelEndpointOffset\s*=\s*(\w+)\s*;", self.header)
        self.assertEqual(match.group(1), "ShadowEpsilon")
        self.assertEqual(
            self.consts["ShadowEpsilon"],
            self.policy["backend_profiles"]["drjit"]["surfel_endpoint_offset"],
        )

    def test_frozen_ray_tmin_divergence(self):
        field_order = ["ray_tmin"]
        drjit = dict(zip(field_order, self.profile_fields("kDrJitLegacyProfile")[:1]))
        torch = dict(zip(field_order, self.profile_fields("kTorchLegacyProfile")[:1]))
        self.assertEqual(drjit["ray_tmin"], 1e-3)
        self.assertEqual(torch["ray_tmin"], 1e-6)
        self.assertNotEqual(drjit["ray_tmin"], torch["ray_tmin"])

    def test_reflection_trace_miss_distance_is_not_infinity(self):
        self.assertIn(
            "std::numeric_limits<float>::infinity()", self.header
        )
        reflection_miss = self.family_constant("kReflectionTraceMissDistance")
        self.assertEqual(reflection_miss, 1e8)
        self.assertEqual(
            reflection_miss, self.policy["shared_multipath"]["trace_tmax_finite"]
        )
        self.assertEqual(
            reflection_miss, OPERATIONS["miss_sentinels"]["reflection_trace_distance"]
        )


if __name__ == "__main__":
    unittest.main()
