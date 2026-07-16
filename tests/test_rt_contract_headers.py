import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RT_DIR = ROOT / "shared" / "include" / "rayd" / "shared" / "rt"
NUMERIC_POLICY = RT_DIR / "numeric_policy.h"
HIT_TYPES = RT_DIR / "hit_types.h"
RAY_TYPES = RT_DIR / "ray_types.h"

# rt/ headers are backend-neutral and host-safe: no backend, CUDA, or OptiX
# tokens may leak in, otherwise a third backend cannot include them cleanly.
FORBIDDEN_TOKENS = ("__device__", "__host__", "optix", "float3", "cuda_runtime")


def struct_fields(header: str, struct_name: str) -> list[str]:
    match = re.search(rf"struct {struct_name}\s*\{{([^}}]*)\}}", header)
    return re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*;", match.group(1))


class RtContractHeaderTests(unittest.TestCase):
    def test_rt_headers_exist(self):
        self.assertTrue(NUMERIC_POLICY.is_file())
        self.assertTrue(HIT_TYPES.is_file())
        self.assertTrue(RAY_TYPES.is_file())

    def test_headers_are_host_safe(self):
        for path in (NUMERIC_POLICY, HIT_TYPES, RAY_TYPES):
            lowered = path.read_text(encoding="utf-8").lower()
            for token in FORBIDDEN_TOKENS:
                with self.subTest(header=path.name, token=token):
                    self.assertNotIn(token, lowered)

    def test_numeric_policy_struct_field_order(self):
        header = NUMERIC_POLICY.read_text(encoding="utf-8")
        self.assertEqual(
            struct_fields(header, "NumericPolicy"),
            [
                "ray_tmin",
                "shadow_tmin",
                "endpoint_offset",
                "parallel_epsilon",
                "watertight_triangles",
            ],
        )

    def test_hit_types_struct_field_order(self):
        header = HIT_TYPES.read_text(encoding="utf-8")
        self.assertEqual(
            struct_fields(header, "RawHit"),
            ["t", "bary_u", "bary_v", "global_prim_id", "shape_id", "local_prim_id"],
        )
        self.assertEqual(struct_fields(header, "RawBlocker"), ["global_prim_id"])
        self.assertIn("sizeof(RawHit) == 24", header)
        self.assertIn("sizeof(RawBlocker) == 4", header)

    def test_ray_types_struct_field_order(self):
        header = RAY_TYPES.read_text(encoding="utf-8")
        self.assertEqual(
            struct_fields(header, "RayBatchView"),
            [
                "origin_x",
                "origin_y",
                "origin_z",
                "direction_x",
                "direction_y",
                "direction_z",
                "tmax",
                "active",
                "count",
            ],
        )
        self.assertEqual(
            struct_fields(header, "SegmentBatchView"),
            [
                "start_x",
                "start_y",
                "start_z",
                "end_x",
                "end_y",
                "end_z",
                "active",
                "count",
            ],
        )


if __name__ == "__main__":
    unittest.main()
