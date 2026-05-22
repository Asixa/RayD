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


class DiffractionAccumulationTests(unittest.TestCase):
    def test_diffraction_accumulation_abi_fields_are_bound(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            grid = pj.DiffractionGrid()
            grid.axis = 1
            grid.position = 2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 3.0
            grid.coord1_min = -2.0
            grid.coord1_max = 4.0
            grid.resolution0 = 5
            grid.resolution1 = 7
            grid.cell_area = 0.25

            material = pj.DiffractionMaterial()
            material.eta_r = cuda.Float([4.0, 5.0])
            material.sigma = cuda.Float([0.01, 0.02])
            material.mu_r = cuda.Float([1.0, 1.1])
            material.gain = cuda.Float([0.8, 0.9])
            material.valid = cuda.Bool([True, False])

            states = pj.DiffractionStateTable()
            states.count = 2
            states.edge_index = cuda.Int([3, 4])
            states.edge_pos = cuda.Array3f([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
            states.edge_dir = cuda.Array3f([0.0, 1.0], [1.0, 0.0], [0.0, 0.0])
            states.edge_line_min = cuda.Float([-0.5, -0.25])
            states.edge_line_max = cuda.Float([0.5, 0.25])
            states.face0_normal = cuda.Array3f([1.0, 0.0], [0.0, 1.0], [0.0, 0.0])
            states.face1_normal = cuda.Array3f([0.0, -1.0], [1.0, 0.0], [0.0, 0.0])
            states.face0_prim_id = cuda.Int([9, 10])
            states.face1_prim_id = cuda.Int([11, 12])
            states.exterior_angle = cuda.Float([1.5, 2.5])
            states.source_pos = cuda.Array3f([7.0, 8.0], [9.0, 10.0], [11.0, 12.0])
            states.source_power = cuda.Float([0.1, 0.2])
            states.incident_direction = cuda.Array3f([0.0, 0.0], [0.0, 1.0], [1.0, 0.0])
            states.initial_direction = cuda.Array3f([1.0, 0.0], [0.0, 1.0], [0.0, 0.0])
            states.prefix_reflection_depth = cuda.Int([0, 2])

            options = pj.DiffractionAccumOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 13
            options.samples = 1024
            options.max_order = 1
            options.direct_samples = 512
            options.keller_samples = 256
            options.suffix_samples = 0
            options.strategy_mask = pj.RAYD_DIFF_DIRECT | pj.RAYD_DIFF_KELLER
            options.sample_sequence = pj.RAYD_DIFF_SOBOL
            options.receiver_model = pj.RAYD_DIFF_MATCHED_ISOTROPIC
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = pj.DiffractionAccumResult()
            result.grid_cell_count = 2
            result.diffraction_power = cuda.Float([1.0, 2.0])
            result.diffraction_field_x = cuda.Complex2f([1.0, 0.0], [0.0, 1.0])
            result.diffraction_field_y = cuda.Complex2f([2.0, 0.0], [0.0, 2.0])
            result.diffraction_field_z = cuda.Complex2f([3.0, 0.0], [0.0, 3.0])
            result.direct_count = cuda.Int([4, 5])
            result.keller_count = cuda.Int([6, 7])
            result.suffix_count = cuda.Int([8, 9])
            result.visibility_reject_count = cuda.Int([10, 11])
            result.utd_reject_count = cuda.Int([12, 13])
            result.edge_use_count = cuda.Int([14, 15])

            dr.eval(
                material.eta_r,
                states.edge_index,
                states.edge_pos,
                states.prefix_reflection_depth,
                result.diffraction_power,
                result.diffraction_field_x.real,
                result.direct_count,
                result.edge_use_count,
            )

            print(json.dumps({
                "grid_axis": grid.axis,
                "grid_area": grid.cell_area,
                "material_count": len(material.eta_r),
                "state_count": states.count,
                "edge0": int(states.edge_index[0]),
                "edge_pos_y1": float(states.edge_pos.y[1]),
                "prefix_depth1": int(states.prefix_reflection_depth[1]),
                "strategy_mask": options.strategy_mask,
                "collect_edge_use": options.collect_edge_use,
                "result_cells": result.grid_cell_count,
                "power1": float(result.diffraction_power[1]),
                "field_x_im1": float(result.diffraction_field_x.imag[1]),
                "direct0": int(result.direct_count[0]),
                "edge_use1": int(result.edge_use_count[1]),
            }))
            """
        )

        self.assertEqual(data["grid_axis"], 1)
        self.assertEqual(data["grid_area"], 0.25)
        self.assertEqual(data["material_count"], 2)
        self.assertEqual(data["state_count"], 2)
        self.assertEqual(data["edge0"], 3)
        self.assertEqual(data["edge_pos_y1"], 4.0)
        self.assertEqual(data["prefix_depth1"], 2)
        self.assertEqual(data["strategy_mask"], 3)
        self.assertTrue(data["collect_edge_use"])
        self.assertEqual(data["result_cells"], 2)
        self.assertEqual(data["power1"], 2.0)
        self.assertEqual(data["field_x_im1"], 1.0)
        self.assertEqual(data["direct0"], 4)
        self.assertEqual(data["edge_use1"], 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
