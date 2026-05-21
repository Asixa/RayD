import json
import math
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, *, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
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


def run_json(script: str):
    result = run_script(script)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON.\nSTDERR:\n{result.stderr}")
    return json.loads(lines[-1])


class ReflectionEpcTests(unittest.TestCase):
    def test_one_bounce_method_of_images_path_is_extracted_in_one_launch(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            mirror = pj.Mesh(
                cuda.Array3f([-2.0, 2.0, 2.0, -2.0],
                             [-2.0, -2.0, 2.0, 2.0],
                             [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(mirror)
            scene.build()

            tx = cuda.Array3f([-0.5], [0.0], [-1.0])
            rx = cuda.Array3f([0.5], [0.0], [-1.0])
            inv_len = 1.0 / math.sqrt(1.25)
            ray = pj.RayDetached(
                tx,
                cuda.Array3f([0.5 * inv_len], [0.0], [1.0 * inv_len]),
            )

            pj.native_launch_audit_clear()
            result = scene.trace_reflection_epc(ray, rx, max_bounces=1)
            dr.eval(result.valid,
                    result.bounce_count,
                    result.path_length,
                    result.reflection_points,
                    result.prim_ids)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "ray_count": result.ray_count,
                "max_bounces": result.max_bounces,
                "valid": bool(result.valid[0]),
                "bounce_count": int(result.bounce_count[0]),
                "path_length": float(result.path_length[0]),
                "point": [
                    float(result.reflection_points[0][0]),
                    float(result.reflection_points[1][0]),
                    float(result.reflection_points[2][0]),
                ],
                "prim": int(result.prim_ids[0]),
                "trace_reflections_launches": audit["trace_reflections"]["optix_launch"],
            }))
            """
        )

        self.assertEqual(data["ray_count"], 1)
        self.assertEqual(data["max_bounces"], 1)
        self.assertTrue(data["valid"])
        self.assertEqual(data["bounce_count"], 1)
        self.assertAlmostEqual(data["path_length"], 2.0 * math.sqrt(1.25), places=4)
        self.assertAlmostEqual(data["point"][0], 0.0, places=4)
        self.assertAlmostEqual(data["point"][1], 0.0, places=4)
        self.assertAlmostEqual(data["point"][2], 0.0, places=4)
        self.assertGreaterEqual(data["prim"], 0)
        self.assertEqual(data["trace_reflections_launches"], 1)

    def test_one_bounce_path_reports_blocked_receiver_segment(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            mirror = pj.Mesh(
                cuda.Array3f([-2.0, 2.0, 2.0, -2.0],
                             [-2.0, -2.0, 2.0, 2.0],
                             [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )
            blocker = pj.Mesh(
                cuda.Array3f([0.25, 0.25, 0.25, 0.25],
                             [-0.4, 0.4, 0.4, -0.4],
                             [-0.75, -0.75, -0.25, -0.25]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(mirror)
            scene.add_mesh(blocker)
            scene.build()

            tx = cuda.Array3f([-0.5], [0.0], [-1.0])
            rx = cuda.Array3f([0.5], [0.0], [-1.0])
            inv_len = 1.0 / math.sqrt(1.25)
            ray = pj.RayDetached(
                tx,
                cuda.Array3f([0.5 * inv_len], [0.0], [1.0 * inv_len]),
            )

            result = scene.trace_reflection_epc(ray, rx, max_bounces=1)
            dr.eval(result.valid,
                    result.bounce_count,
                    result.first_blocked_segment,
                    result.first_blocked_prim)

            print(json.dumps({
                "valid": bool(result.valid[0]),
                "bounce_count": int(result.bounce_count[0]),
                "blocked_segment": int(result.first_blocked_segment[0]),
                "blocked_prim": int(result.first_blocked_prim[0]),
            }))
            """
        )

        self.assertFalse(data["valid"])
        self.assertEqual(data["bounce_count"], 1)
        self.assertEqual(data["blocked_segment"], 1)
        self.assertGreaterEqual(data["blocked_prim"], 2)

    def test_two_bounce_method_of_images_extracts_ordered_reflection_points(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            wall = pj.Mesh(
                cuda.Array3f([1.0, 1.0, 1.0, 1.0],
                             [-2.0, 2.0, 2.0, -2.0],
                             [0.0, 0.0, 3.0, 3.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )
            ceiling = pj.Mesh(
                cuda.Array3f([-2.0, 2.0, 2.0, -2.0],
                             [-2.0, -2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0, 2.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(wall)
            scene.add_mesh(ceiling)
            scene.build()

            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            ray = pj.RayDetached(
                cuda.Array3f([0.0], [0.0], [0.5]),
                cuda.Array3f([inv_sqrt2], [0.0], [inv_sqrt2]),
            )
            rx = cuda.Array3f([0.0], [0.0], [1.5])

            result = scene.trace_reflection_epc(ray, rx, max_bounces=2)
            dr.eval(result.valid,
                    result.bounce_count,
                    result.path_length,
                    result.reflection_points,
                    result.prim_ids)

            print(json.dumps({
                "valid": bool(result.valid[0]),
                "bounce_count": int(result.bounce_count[0]),
                "path_length": float(result.path_length[0]),
                "p0": [
                    float(result.reflection_points[0][0]),
                    float(result.reflection_points[1][0]),
                    float(result.reflection_points[2][0]),
                ],
                "p1": [
                    float(result.reflection_points[0][1]),
                    float(result.reflection_points[1][1]),
                    float(result.reflection_points[2][1]),
                ],
                "prims": [int(result.prim_ids[0]), int(result.prim_ids[1])],
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["bounce_count"], 2)
        self.assertAlmostEqual(data["path_length"], 2.0 * math.sqrt(2.0), places=4)
        self.assertAlmostEqual(data["p0"][0], 1.0, places=4)
        self.assertAlmostEqual(data["p0"][1], 0.0, places=4)
        self.assertAlmostEqual(data["p0"][2], 1.5, places=4)
        self.assertAlmostEqual(data["p1"][0], 0.5, places=4)
        self.assertAlmostEqual(data["p1"][1], 0.0, places=4)
        self.assertAlmostEqual(data["p1"][2], 2.0, places=4)
        self.assertGreaterEqual(data["prims"][0], 0)
        self.assertGreaterEqual(data["prims"][1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
