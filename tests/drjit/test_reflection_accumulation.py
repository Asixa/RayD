import json
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


class ReflectionAccumulationTests(unittest.TestCase):
    def test_trace_reflections_accumulating_writes_grid_and_wedge_events(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [0.0, 0.0, 0.0])
            faces = cuda.Array3i([0], [1], [2])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, faces))
            scene.build()

            ray = pj.RayDetached(cuda.Array3f([0.0], [0.0], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))
            tx = cuda.Array3f([0.0], [0.0], [-1.0])

            grid = pj.ReflectionAccumulationGrid()
            grid.axis = 2
            grid.position = -2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1

            material = pj.PrimitiveMaterialPayloadDetached()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.gain = cuda.Float([1.0])
            material.mu_r = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.ReflectionAccumulationOptions()
            options.wavelength = 12.566370614359172
            options.k = 0.5
            options.solid_angle_per_ray = 1.0
            options.cell_area = 1.0
            options.seed = 17
            options.rr_depth = 0
            options.rr_prob = 1.0
            options.stop_threshold = 0.0
            options.collect_wedges = True
            options.collect_wedge_prefixes = True
            options.wedge_capacity = 4

            pj.native_launch_audit_clear()
            result = scene.trace_reflections_accumulating(
                ray, tx, grid, material, 1, options
            )
            dr.eval(result.reflection_power,
                    result.reflection_count,
                    result.wedge_events.count,
                    result.wedge_events.prim_id,
                    result.wedge_events.bounce_depth)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "ray_count": result.ray_count,
                "max_bounces": result.max_bounces,
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.reflection_power[0]),
                "reflection_count": int(result.reflection_count[0]),
                "wedge_capacity": result.wedge_events.capacity,
                "wedge_count": int(result.wedge_events.count[0]),
                "wedge_prim0": int(result.wedge_events.prim_id[0]),
                "wedge_depth0": int(result.wedge_events.bounce_depth[0]),
                "trace_reflections_launches": audit["trace_reflections"]["optix_launch"],
                "trace_reflections_accumulating_launches": (
                    audit["trace_reflections_accumulating"]["optix_launch"]
                ),
            }))
            """
        )

        self.assertEqual(data["ray_count"], 1)
        self.assertEqual(data["max_bounces"], 1)
        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertEqual(data["reflection_count"], 1)
        self.assertEqual(data["wedge_capacity"], 4)
        self.assertEqual(data["wedge_count"], 1)
        self.assertEqual(data["wedge_prim0"], 0)
        self.assertEqual(data["wedge_depth0"], 0)
        self.assertEqual(data["trace_reflections_launches"], 0)
        self.assertEqual(data["trace_reflections_accumulating_launches"], 1)

    def test_trace_reflections_accumulating_rejects_ad_inputs(self):
        result = run_script(
            """
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [0.0, 0.0, 0.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            ray = pj.Ray(ad.Array3f([0.0], [0.0], [-1.0]),
                         ad.Array3f([0.0], [0.0], [1.0]))
            grid = pj.ReflectionAccumulationGrid()
            material = pj.PrimitiveMaterialPayload()
            options = pj.ReflectionAccumulationOptions()
            try:
                scene.trace_reflections_accumulating(
                    ray, ad.Array3f([0.0], [0.0], [-1.0]), grid, material, 1, options
                )
            except RuntimeError as exc:
                if "non-AD native fast path" in str(exc):
                    raise SystemExit(0)
                raise
            raise AssertionError("expected AD rejection")
            """,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
