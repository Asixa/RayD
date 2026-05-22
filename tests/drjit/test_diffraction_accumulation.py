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

    def test_accumulate_diffraction_order1_writes_grid(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            states = pj.DiffractionStateTable()
            states.count = 1
            states.edge_index = cuda.Int([0])
            states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            states.edge_line_min = cuda.Float([-0.5])
            states.edge_line_max = cuda.Float([0.5])
            states.face0_normal = cuda.Array3f([0.0], [1.0], [0.0])
            states.face1_normal = cuda.Array3f([0.0], [-1.0], [0.0])
            states.face0_prim_id = cuda.Int([-1])
            states.face1_prim_id = cuda.Int([-1])
            states.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            states.source_pos = cuda.Array3f([0.0], [0.0], [1.0])
            states.source_power = cuda.Float([2.0])
            states.incident_direction = cuda.Array3f([0.0], [0.0], [-1.0])
            states.initial_direction = cuda.Array3f([0.0], [0.0], [-1.0])
            states.prefix_reflection_depth = cuda.Int([0])

            grid = pj.DiffractionGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DiffractionMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DiffractionAccumOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 17
            options.samples = 64
            options.max_order = 1
            options.direct_samples = 64
            options.keller_samples = 0
            options.strategy_mask = pj.RAYD_DIFF_DIRECT
            options.sample_sequence = pj.RAYD_DIFF_HASH
            options.receiver_model = pj.RAYD_DIFF_MATCHED_ISOTROPIC
            options.collect_edge_use = True
            options.collect_debug_counts = True

            pj.native_launch_audit_clear()
            result = scene.accumulate_diffraction_order1(
                states, grid, material, options, True
            )
            dr.eval(
                result.diffraction_power,
                result.diffraction_field_x.real,
                result.diffraction_field_x.imag,
                result.direct_count,
                result.keller_count,
                result.visibility_reject_count,
                result.edge_use_count,
            )
            audit = pj.native_launch_audit()

            keller_options = pj.DiffractionAccumOptions()
            keller_options.wavelength = 0.125
            keller_options.k = 50.26548245743669
            keller_options.seed = 23
            keller_options.samples = 64
            keller_options.max_order = 1
            keller_options.direct_samples = 0
            keller_options.keller_samples = 64
            keller_options.strategy_mask = pj.RAYD_DIFF_KELLER
            keller_options.sample_sequence = pj.RAYD_DIFF_HASH
            keller_options.receiver_model = pj.RAYD_DIFF_MATCHED_ISOTROPIC
            keller_options.collect_edge_use = True
            keller_options.collect_debug_counts = True

            pj.native_launch_audit_clear()
            keller_result = scene.accumulate_diffraction_order1(
                states, grid, material, keller_options, True
            )
            dr.eval(
                keller_result.diffraction_power,
                keller_result.diffraction_field_x.real,
                keller_result.direct_count,
                keller_result.keller_count,
                keller_result.visibility_reject_count,
                keller_result.utd_reject_count,
                keller_result.edge_use_count,
            )
            keller_audit = pj.native_launch_audit()

            print(json.dumps({
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.diffraction_power[0]),
                "field_x_re": float(result.diffraction_field_x.real[0]),
                "field_x_im": float(result.diffraction_field_x.imag[0]),
                "direct_count": int(result.direct_count[0]),
                "keller_count": int(result.keller_count[0]),
                "visibility_reject_count": int(result.visibility_reject_count[0]),
                "edge_use_count": int(result.edge_use_count[0]),
                "accumulate_diffraction_launches": (
                    audit["accumulate_diffraction"]["optix_launch"]
                ),
                "keller_power": float(keller_result.diffraction_power[0]),
                "keller_field_x_re": float(keller_result.diffraction_field_x.real[0]),
                "keller_direct_count": int(keller_result.direct_count[0]),
                "keller_path_count": int(keller_result.keller_count[0]),
                "keller_visibility_reject_count": int(keller_result.visibility_reject_count[0]),
                "keller_utd_reject_count": int(keller_result.utd_reject_count[0]),
                "keller_edge_use_count": int(keller_result.edge_use_count[0]),
                "keller_launches": (
                    keller_audit["accumulate_diffraction"]["optix_launch"]
                ),
            }))
            """
        )

        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["field_x_re"], 0.0)
        self.assertEqual(data["field_x_im"], 0.0)
        self.assertEqual(data["direct_count"], 64)
        self.assertEqual(data["keller_count"], 0)
        self.assertEqual(data["visibility_reject_count"], 0)
        self.assertEqual(data["edge_use_count"], 64)
        self.assertEqual(data["accumulate_diffraction_launches"], 1)
        self.assertGreater(data["keller_power"], 0.0)
        self.assertGreater(data["keller_field_x_re"], 0.0)
        self.assertEqual(data["keller_direct_count"], 0)
        self.assertGreater(data["keller_path_count"], 0)
        self.assertLessEqual(data["keller_path_count"], 64)
        self.assertEqual(data["keller_visibility_reject_count"], 0)
        self.assertEqual(
            data["keller_path_count"] + data["keller_utd_reject_count"], 64
        )
        self.assertEqual(data["keller_edge_use_count"], data["keller_path_count"])
        self.assertEqual(data["keller_launches"], 1)

    def test_accumulate_diffraction_order1_accepts_vector_active_mask(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            states = pj.DiffractionStateTable()
            states.count = 2
            states.edge_index = cuda.Int([0, 1])
            states.edge_pos = cuda.Array3f([0.0, 0.5], [0.0, 0.0], [0.0, 0.0])
            states.edge_dir = cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
            states.edge_line_min = cuda.Float([-0.5, -0.5])
            states.edge_line_max = cuda.Float([0.5, 0.5])
            states.face0_normal = cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
            states.face1_normal = cuda.Array3f([0.0, 0.0], [-1.0, -1.0], [0.0, 0.0])
            states.face0_prim_id = cuda.Int([-1, -1])
            states.face1_prim_id = cuda.Int([-1, -1])
            states.exterior_angle = cuda.Float([1.5 * 3.141592653589793, 1.5 * 3.141592653589793])
            states.source_pos = cuda.Array3f([0.0, 0.5], [0.0, 0.0], [1.0, 1.0])
            states.source_power = cuda.Float([2.0, 2.0])
            states.incident_direction = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
            states.initial_direction = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
            states.prefix_reflection_depth = cuda.Int([0, 0])

            grid = pj.DiffractionGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DiffractionMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DiffractionAccumOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 31
            options.samples = 16
            options.max_order = 1
            options.direct_samples = 8
            options.keller_samples = 8
            options.strategy_mask = pj.RAYD_DIFF_DIRECT | pj.RAYD_DIFF_KELLER
            options.sample_sequence = pj.RAYD_DIFF_HASH
            options.receiver_model = pj.RAYD_DIFF_MATCHED_ISOTROPIC
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = scene.accumulate_diffraction_order1(
                states, grid, material, options, cuda.Bool([True, False])
            )
            dr.eval(result.diffraction_power, result.direct_count, result.keller_count)
            print(json.dumps({
                "finite_power": bool(dr.all(dr.isfinite(result.diffraction_power))),
                "direct_count": int(result.direct_count[0]),
                "keller_count": int(result.keller_count[0]),
            }))
            """
        )

        self.assertTrue(data["finite_power"])
        self.assertGreater(data["direct_count"] + data["keller_count"], 0)

    def test_accumulate_diffraction_chains_order2_direct_writes_grid(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            initial = pj.DiffractionStateTable()
            initial.count = 1
            initial.edge_index = cuda.Int([0])
            initial.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            initial.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            initial.edge_line_min = cuda.Float([-0.5])
            initial.edge_line_max = cuda.Float([0.5])
            initial.face0_normal = cuda.Array3f([0.0], [1.0], [0.0])
            initial.face1_normal = cuda.Array3f([0.0], [-1.0], [0.0])
            initial.face0_prim_id = cuda.Int([-1])
            initial.face1_prim_id = cuda.Int([-1])
            initial.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            initial.source_pos = cuda.Array3f([0.0], [0.0], [1.0])
            initial.source_power = cuda.Float([2.0])
            initial.incident_direction = cuda.Array3f([0.0], [0.0], [-1.0])
            initial.initial_direction = cuda.Array3f([0.0], [0.0], [-1.0])
            initial.prefix_reflection_depth = cuda.Int([0])

            recursive = pj.DiffractionStateTable()
            recursive.count = 1
            recursive.edge_index = cuda.Int([1])
            recursive.edge_pos = cuda.Array3f([0.0], [0.5], [0.0])
            recursive.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            recursive.edge_line_min = cuda.Float([-0.5])
            recursive.edge_line_max = cuda.Float([0.5])
            recursive.face0_normal = cuda.Array3f([0.0], [1.0], [0.0])
            recursive.face1_normal = cuda.Array3f([0.0], [-1.0], [0.0])
            recursive.face0_prim_id = cuda.Int([-1])
            recursive.face1_prim_id = cuda.Int([-1])
            recursive.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            recursive.source_pos = cuda.Array3f([0.0], [0.0], [1.0])
            recursive.source_power = cuda.Float([1.0])
            recursive.incident_direction = cuda.Array3f([0.0], [1.0], [0.0])
            recursive.initial_direction = cuda.Array3f([0.0], [0.0], [-1.0])
            recursive.prefix_reflection_depth = cuda.Int([0])

            grid = pj.DiffractionGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DiffractionMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DiffractionAccumOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 41
            options.samples = 32
            options.max_order = 2
            options.direct_samples = 32
            options.keller_samples = 0
            options.strategy_mask = pj.RAYD_DIFF_DIRECT
            options.sample_sequence = pj.RAYD_DIFF_HASH
            options.receiver_model = pj.RAYD_DIFF_MATCHED_ISOTROPIC
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = scene.accumulate_diffraction_chains(
                initial, recursive, grid, material, options, True
            )
            dr.eval(
                result.diffraction_power,
                result.direct_count,
                result.inter_edge_visibility_reject_count,
                result.edge_use_count,
            )
            print(json.dumps({
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.diffraction_power[0]),
                "direct_count": int(result.direct_count[0]),
                "inter_edge_rejects": int(result.inter_edge_visibility_reject_count[0]),
                "edge_use_count": int(result.edge_use_count[0]),
            }))
            """
        )

        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["direct_count"], 0)
        self.assertEqual(data["edge_use_count"], data["direct_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
