# Copyright Xingyu Chen.
# Tests Dr.Jit SDF LOS and reflection operation scope.

from __future__ import annotations

import unittest

from tests.support.subprocess_cases import run_json_case


class DrJitSdfOperationTests(unittest.TestCase):
    def test_sdf_los_reflection_and_closed_scope(self) -> None:
        data = run_json_case(
            """
            import json
            import rayd.drjit as pj
            import drjit.cuda as cuda

            n = 8
            values = cuda.Float([(-1.0 + 2.0 * k / (n - 1)) for _i in range(n) for _j in range(n) for k in range(n)])
            grid = pj.SdfGrid(
                values, n, n, n,
                cuda.Array3f([0.0], [0.0], [0.0]),
                cuda.Float([1.0, 0.0, 0.0, 0.0]),
                cuda.Array3f([2.0], [2.0], [2.0]),
            )
            ray = pj.Ray(cuda.Array3f([0.0], [0.0], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
            hit = grid.intersect(ray)
            visible = grid.visible(
                cuda.Array3f([0.0], [0.0], [-1.5]),
                cuda.Array3f([0.0], [0.0], [1.5]),
            )
            chain = grid.trace_reflections(ray, 1)
            print(json.dumps({
                "hit": bool(hit.hit_mask[0]),
                "t": float(hit.t[0]),
                "visible": bool(visible[0]),
                "bounce_count": int(chain.bounce_count[0]),
                "prim_id": int(chain.prim_ids[0]),
                "has_transmission": hasattr(grid, "transmittance"),
                "has_diffraction": hasattr(grid, "trace_diffraction"),
            }))
            """
        )
        self.assertTrue(data["hit"])
        self.assertAlmostEqual(data["t"], 1.0, places=3)
        self.assertFalse(data["visible"])
        self.assertEqual(data["bounce_count"], 1)
        self.assertEqual(data["prim_id"], 0)
        self.assertFalse(data["has_transmission"])
        self.assertFalse(data["has_diffraction"])


if __name__ == "__main__":
    unittest.main()
