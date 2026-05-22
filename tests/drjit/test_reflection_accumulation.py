import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, *, check: bool = True):
    env = os.environ.copy()
    env["PYTHONSAFEPATH"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        env=env,
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
    def test_accumulate_reflections_writes_grid_and_wedge_events(self):
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

            ray = pj.Ray(cuda.Array3f([0.0], [0.0], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))
            tx = cuda.Array3f([0.0], [0.0], [-1.0])

            grid = pj.AccumGrid()
            grid.axis = 2
            grid.position = -2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1

            material = pj.Material()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.gain = cuda.Float([1.0])
            material.mu_r = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.AccumOptions()
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
            result = scene.accumulate_reflections(
                ray, tx, grid, material, 1, options
            )
            dr.eval(result.reflection_power,
                    result.reflection_field_x.real,
                    result.reflection_field_x.imag,
                    result.reflection_field_y.real,
                    result.reflection_field_y.imag,
                    result.reflection_field_z.real,
                    result.reflection_field_z.imag,
                    result.reflection_count,
                    result.wedge_events.count,
                    result.wedge_events.prim_id,
                    result.wedge_events.directions.z,
                    result.wedge_events.source_points.z,
                    result.wedge_events.source_power,
                    result.wedge_events.initial_directions.z,
                    result.wedge_events.bounce_depth)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "ray_count": result.ray_count,
                "max_bounces": result.max_bounces,
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.reflection_power[0]),
                "field_x_re": float(result.reflection_field_x.real[0]),
                "field_x_im": float(result.reflection_field_x.imag[0]),
                "field_y_re": float(result.reflection_field_y.real[0]),
                "field_y_im": float(result.reflection_field_y.imag[0]),
                "field_z_re": float(result.reflection_field_z.real[0]),
                "field_z_im": float(result.reflection_field_z.imag[0]),
                "reflection_count": int(result.reflection_count[0]),
                "wedge_capacity": result.wedge_events.capacity,
                "wedge_count": int(result.wedge_events.count[0]),
                "wedge_prim0": int(result.wedge_events.prim_id[0]),
                "wedge_depth0": int(result.wedge_events.bounce_depth[0]),
                "wedge_dir_z0": float(result.wedge_events.directions.z[0]),
                "wedge_source_z0": float(result.wedge_events.source_points.z[0]),
                "wedge_source_power0": float(result.wedge_events.source_power[0]),
                "wedge_initial_dir_z0": float(result.wedge_events.initial_directions.z[0]),
                "trace_reflections_launches": audit["trace_reflections"]["optix_launch"],
                "accumulate_reflections_launches": (
                    audit["accumulate_reflections"]["optix_launch"]
                ),
            }))
            """
        )

        self.assertEqual(data["ray_count"], 1)
        self.assertEqual(data["max_bounces"], 1)
        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["field_x_re"] ** 2 + data["field_x_im"] ** 2, 0.0)
        self.assertLess(data["field_y_re"] ** 2 + data["field_y_im"] ** 2, 1e-8)
        self.assertLess(data["field_z_re"] ** 2 + data["field_z_im"] ** 2, 1e-8)
        self.assertEqual(data["reflection_count"], 1)
        self.assertEqual(data["wedge_capacity"], 4)
        self.assertEqual(data["wedge_count"], 1)
        self.assertEqual(data["wedge_prim0"], 0)
        self.assertEqual(data["wedge_depth0"], 0)
        self.assertGreater(data["wedge_dir_z0"], 0.0)
        self.assertAlmostEqual(data["wedge_source_z0"], -1.0)
        self.assertGreater(data["wedge_source_power0"], 0.0)
        self.assertGreater(data["wedge_initial_dir_z0"], 0.0)
        self.assertEqual(data["trace_reflections_launches"], 0)
        self.assertEqual(data["accumulate_reflections_launches"], 1)

    def test_accumulate_reflections_rejects_ad_inputs(self):
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

            ray = pj.RayAD(ad.Array3f([0.0], [0.0], [-1.0]),
                         ad.Array3f([0.0], [0.0], [1.0]))
            grid = pj.AccumGrid()
            material = pj.MaterialAD()
            options = pj.AccumOptions()
            try:
                scene.accumulate_reflections(
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

    def test_accumulate_reflections_accepts_tx_polarization(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [0.0, 0.0, 0.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            ray = pj.Ray(cuda.Array3f([0.0], [0.0], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))
            tx = cuda.Array3f([0.0], [0.0], [-1.0])
            tx_pol = cuda.Array3f([0.0], [1.0], [0.0])

            grid = pj.AccumGrid()
            grid.axis = 2
            grid.position = -2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1

            material = pj.Material()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.gain = cuda.Float([1.0])
            material.mu_r = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.AccumOptions()
            options.wavelength = 12.566370614359172
            options.k = 0.5
            options.cell_area = 1.0

            result = scene.accumulate_reflections(
                ray, tx, grid, material, 1, options, True, tx_pol
            )
            dr.eval(result.reflection_field_x.real,
                    result.reflection_field_x.imag,
                    result.reflection_field_y.real,
                    result.reflection_field_y.imag)

            print(json.dumps({
                "x2": float(result.reflection_field_x.real[0]) ** 2
                      + float(result.reflection_field_x.imag[0]) ** 2,
                "y2": float(result.reflection_field_y.real[0]) ** 2
                      + float(result.reflection_field_y.imag[0]) ** 2,
            }))
            """
        )

        self.assertLess(data["x2"], 1e-8)
        self.assertGreater(data["y2"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
