import json
import math
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


class DfrAccumulationTests(unittest.TestCase):
    def test_diffraction_accumulation_abi_fields_are_bound(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            grid = pj.DfrGrid()
            grid.axis = 1
            grid.position = 2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 3.0
            grid.coord1_min = -2.0
            grid.coord1_max = 4.0
            grid.resolution0 = 5
            grid.resolution1 = 7
            grid.cell_area = 0.25

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0, 5.0])
            material.sigma = cuda.Float([0.01, 0.02])
            material.mu_r = cuda.Float([1.0, 1.1])
            material.gain = cuda.Float([0.8, 0.9])
            material.valid = cuda.Bool([True, False])

            states = pj.DfrStates()
            states.count = 2
            states.edge_index = cuda.Int([3, 4])
            states.edge_pos = cuda.Array3f([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
            states.edge_dir = cuda.Array3f([0.0, 1.0], [1.0, 0.0], [0.0, 0.0])
            states.edge_t_min = cuda.Float([-0.5, -0.25])
            states.edge_t_max = cuda.Float([0.5, 0.25])
            states.n0 = cuda.Array3f([1.0, 0.0], [0.0, 1.0], [0.0, 0.0])
            states.n1 = cuda.Array3f([0.0, -1.0], [1.0, 0.0], [0.0, 0.0])
            states.prim0 = cuda.Int([9, 10])
            states.prim1 = cuda.Int([11, 12])
            states.exterior_angle = cuda.Float([1.5, 2.5])
            states.src = cuda.Array3f([7.0, 8.0], [9.0, 10.0], [11.0, 12.0])
            states.src_power = cuda.Float([0.1, 0.2])
            states.wi = cuda.Array3f([0.0, 0.0], [0.0, 1.0], [1.0, 0.0])
            states.d0 = cuda.Array3f([1.0, 0.0], [0.0, 1.0], [0.0, 0.0])
            states.prefix_depth = cuda.Int([0, 2])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 13
            options.samples = 1024
            options.max_order = 1
            options.direct_samples = 512
            options.keller_samples = 256
            options.suffix_samples = 0
            options.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
            options.sample_sequence = pj.RAYD_DFR_SOBOL
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = pj.DfrAccum()
            result.grid_cell_count = 2
            result.power = cuda.Float([1.0, 2.0])
            result.field_x = cuda.Complex2f([1.0, 0.0], [0.0, 1.0])
            result.field_y = cuda.Complex2f([2.0, 0.0], [0.0, 2.0])
            result.field_z = cuda.Complex2f([3.0, 0.0], [0.0, 3.0])
            result.direct_count = cuda.Int([4, 5])
            result.keller_count = cuda.Int([6, 7])
            result.suffix_count = cuda.Int([8, 9])
            result.vis_rejects = cuda.Int([10, 11])
            result.utd_rejects = cuda.Int([12, 13])
            result.edge_uses = cuda.Int([14, 15])

            dr.eval(
                material.eta_r,
                states.edge_index,
                states.edge_pos,
                states.prefix_depth,
                result.power,
                result.field_x.real,
                result.direct_count,
                result.edge_uses,
            )

            print(json.dumps({
                "grid_axis": grid.axis,
                "grid_area": grid.cell_area,
                "material_count": len(material.eta_r),
                "state_count": states.count,
                "edge0": int(states.edge_index[0]),
                "edge_pos_y1": float(states.edge_pos.y[1]),
                "prefix_depth1": int(states.prefix_depth[1]),
                "strategy_mask": options.strategy_mask,
                "collect_edge_use": options.collect_edge_use,
                "result_cells": result.grid_cell_count,
                "power1": float(result.power[1]),
                "field_x_im1": float(result.field_x.imag[1]),
                "direct0": int(result.direct_count[0]),
                "edge_use1": int(result.edge_uses[1]),
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

    def test_diffraction_path_export_abi_fields_are_bound(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            options = pj.DfrPathOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 19
            options.max_order = 1
            options.max_paths = 8
            options.max_rx = 3
            options.strategy_mask = pj.RAYD_DFR_DIRECT
            options.sample_count = 64
            options.return_geom = 1
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

            result = pj.DfrPaths()
            result.capacity = 2
            result.count = cuda.Int([1])
            result.valid = cuda.Bool([True, False])
            result.tx_id = cuda.Int([0, -1])
            result.rx_id = cuda.Int([1, -1])
            result.order = cuda.Int([1, 0])
            result.edge0 = cuda.Int([7, -1])
            result.edge1 = cuda.Int([-1, -1])
            result.edge2 = cuda.Int([-1, -1])
            result.delay = cuda.Float([2.0e-9, 0.0])
            result.field_x = cuda.Complex2f([1.0, 0.0], [0.25, 0.0])
            result.field_y = cuda.Complex2f([0.0, 0.0], [0.0, 0.0])
            result.field_z = cuda.Complex2f([0.0, 0.0], [0.0, 0.0])
            result.p0 = cuda.Array3f([1.0, 0.0], [2.0, 0.0], [3.0, 0.0])
            result.p1 = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
            result.p2 = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
            dr.eval(result.count, result.valid, result.field_x.real, result.p0.z)

            print(json.dumps({
                "max_paths": options.max_paths,
                "return_geom": options.return_geom,
                "capacity": result.capacity,
                "count": int(result.count[0]),
                "rx0": int(result.rx_id[0]),
                "edge0": int(result.edge0[0]),
                "field_x_im0": float(result.field_x.imag[0]),
                "p0_z0": float(result.p0.z[0]),
            }))
            """
        )

        self.assertEqual(data["max_paths"], 8)
        self.assertEqual(data["return_geom"], 1)
        self.assertEqual(data["capacity"], 2)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["rx0"], 1)
        self.assertEqual(data["edge0"], 7)
        self.assertAlmostEqual(data["field_x_im0"], 0.25, places=6)
        self.assertAlmostEqual(data["p0_z0"], 3.0, places=6)

    def test_accum_dfr_direct_writes_grid(self):
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

            states = pj.DfrStates()
            states.count = 1
            states.edge_index = cuda.Int([0])
            states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            states.edge_t_min = cuda.Float([-0.5])
            states.edge_t_max = cuda.Float([0.5])
            states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            states.prim0 = cuda.Int([-1])
            states.prim1 = cuda.Int([-1])
            states.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            states.src = cuda.Array3f([0.0], [0.0], [1.0])
            states.src_power = cuda.Float([2.0])
            states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            states.prefix_depth = cuda.Int([0])

            grid = pj.DfrGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 17
            options.samples = 64
            options.max_order = 1
            options.direct_samples = 64
            options.keller_samples = 0
            options.strategy_mask = pj.RAYD_DFR_DIRECT
            options.sample_sequence = pj.RAYD_DFR_HASH
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_edge_use = True
            options.collect_debug_counts = True

            pj.native_launch_audit_clear()
            result = scene.accum_dfr_direct(
                states, grid, material, options, True
            )
            dr.eval(
                result.power,
                result.field_x.real,
                result.field_x.imag,
                result.direct_count,
                result.keller_count,
                result.vis_rejects,
                result.edge_uses,
            )
            audit = pj.native_launch_audit()

            keller_options = pj.DfrOptions()
            keller_options.wavelength = 0.125
            keller_options.k = 50.26548245743669
            keller_options.seed = 23
            keller_options.samples = 64
            keller_options.max_order = 1
            keller_options.direct_samples = 0
            keller_options.keller_samples = 64
            keller_options.strategy_mask = pj.RAYD_DFR_KELLER
            keller_options.sample_sequence = pj.RAYD_DFR_HASH
            keller_options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            keller_options.collect_edge_use = True
            keller_options.collect_debug_counts = True

            pj.native_launch_audit_clear()
            keller_result = scene.accum_dfr_direct(
                states, grid, material, keller_options, True
            )
            dr.eval(
                keller_result.power,
                keller_result.field_x.real,
                keller_result.direct_count,
                keller_result.keller_count,
                keller_result.vis_rejects,
                keller_result.utd_rejects,
                keller_result.edge_uses,
            )
            keller_audit = pj.native_launch_audit()

            print(json.dumps({
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.power[0]),
                "field_x_re": float(result.field_x.real[0]),
                "field_x_im": float(result.field_x.imag[0]),
                "direct_count": int(result.direct_count[0]),
                "keller_count": int(result.keller_count[0]),
                "vis_rejects": int(result.vis_rejects[0]),
                "edge_uses": int(result.edge_uses[0]),
                "accum_dfr_launches": (
                    audit["accum_dfr"]["optix_launch"]
                ),
                "keller_power": float(keller_result.power[0]),
                "keller_field_x_re": float(keller_result.field_x.real[0]),
                "keller_direct_count": int(keller_result.direct_count[0]),
                "keller_path_count": int(keller_result.keller_count[0]),
                "keller_vis_rejects": int(keller_result.vis_rejects[0]),
                "keller_utd_rejects": int(keller_result.utd_rejects[0]),
                "keller_edge_uses": int(keller_result.edge_uses[0]),
                "keller_launches": (
                    keller_audit["accum_dfr"]["optix_launch"]
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
        self.assertEqual(data["vis_rejects"], 0)
        self.assertEqual(data["edge_uses"], 64)
        self.assertEqual(data["accum_dfr_launches"], 1)
        self.assertGreater(data["keller_power"], 0.0)
        self.assertGreater(data["keller_field_x_re"], 0.0)
        self.assertEqual(data["keller_direct_count"], 0)
        self.assertGreater(data["keller_path_count"], 0)
        self.assertLessEqual(data["keller_path_count"], 64)
        self.assertEqual(data["keller_vis_rejects"], 0)
        self.assertEqual(
            data["keller_path_count"] + data["keller_utd_rejects"], 64
        )
        self.assertEqual(data["keller_edge_uses"], data["keller_path_count"])
        self.assertEqual(data["keller_launches"], 1)

    def test_accum_dfr_direct_supports_ad_inputs(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z, enable_grad=False):
                src = ad.Array3f([0.0], [0.0], [src_z])
                if enable_grad:
                    dr.enable_grad(src)
                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.5])
                states.edge_t_max = ad.Float([0.5])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([-1])
                states.prim1 = ad.Int([-1])
                states.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                states.src = src
                states.src_power = ad.Float([2.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 17
                options.samples = 64
                options.max_order = 1
                options.direct_samples = 64
                options.keller_samples = 0
                options.strategy_mask = pj.RAYD_DFR_DIRECT
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src = dr.grad(src)
                    dr.eval(result.power, result.direct_count, grad_src)
                    return {
                        "result_type": type(result).__name__,
                        "grid_cell_count": result.grid_cell_count,
                        "power": float(result.power[0]),
                        "direct_count": int(result.direct_count[0]),
                        "grad_src_z": float(grad_src.z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(1.0, enable_grad=True)
            fd = (run_case(1.0 + step)["loss"] - run_case(1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertEqual(data["direct_count"], 64)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=2.0e-3)

    def test_accum_dfr_direct_forward_jvp_matches_finite_difference(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z, enable_grad=False):
                src = ad.Array3f([0.0], [0.0], [src_z])
                if enable_grad:
                    dr.enable_grad(src)
                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.5])
                states.edge_t_max = ad.Float([0.5])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([-1])
                states.prim1 = ad.Int([-1])
                states.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                states.src = src
                states.src_power = ad.Float([2.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([1.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 7
                options.samples = 64
                options.max_order = 1
                options.direct_samples = 64
                options.keller_samples = 0
                options.suffix_samples = 0
                options.strategy_mask = pj.RAYD_DFR_DIRECT
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.set_grad(src.z, ad.Float([1.0]))
                    dr.forward(src.z)
                    jvp = dr.grad(loss)
                    dr.eval(result.power, result.direct_count, jvp)
                    return {
                        "power": float(result.power[0]),
                        "direct_count": int(result.direct_count[0]),
                        "jvp_src_z": float(jvp[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(1.0, enable_grad=True)
            fd = (run_case(1.0 + step)["loss"] - run_case(1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertGreater(data["power"], 0.0)
        self.assertEqual(data["direct_count"], 64)
        self.assertAlmostEqual(data["jvp_src_z"], data["fd_src_z"], delta=2.0e-3)

    def test_accum_dfr_direct_keller_forward_jvp_matches_finite_difference(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(wi_x_value, enable_grad=False):
                wi_x = ad.Float([wi_x_value])
                if enable_grad:
                    dr.enable_grad(wi_x)
                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.5])
                states.edge_t_max = ad.Float([0.5])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([-1])
                states.prim1 = ad.Int([-1])
                states.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                states.src = ad.Array3f([0.0], [0.0], [1.0])
                states.src_power = ad.Float([2.0])
                states.wi = ad.Array3f(wi_x, [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 17
                options.samples = 64
                options.max_order = 1
                options.direct_samples = 0
                options.keller_samples = 64
                options.suffix_samples = 0
                options.strategy_mask = pj.RAYD_DFR_KELLER
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.set_grad(wi_x, ad.Float([1.0]))
                    dr.forward(wi_x)
                    jvp = dr.grad(loss)
                    dr.eval(result.power, result.keller_count, jvp)
                    return {
                        "power": float(result.power[0]),
                        "keller_count": int(result.keller_count[0]),
                        "jvp_wi_x": float(jvp[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            base = 0.1
            step = 1.0e-3
            ad_result = run_case(base, enable_grad=True)
            fd = (run_case(base + step)["loss"] - run_case(base - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_wi_x": fd,
            }))
            """
        )

        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["keller_count"], 0)
        self.assertAlmostEqual(data["jvp_wi_x"], data["fd_wi_x"], delta=2.0e-3)

    def test_accum_dfr_direct_ad_matches_detached_native_primal(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def make_grid():
                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0
                return grid

            def make_options():
                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 17
                options.samples = 64
                options.max_order = 1
                options.direct_samples = 64
                options.keller_samples = 0
                options.strategy_mask = pj.RAYD_DFR_DIRECT
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True
                return options

            states = pj.DfrStates()
            states.count = 1
            states.edge_index = cuda.Int([0])
            states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            states.edge_t_min = cuda.Float([-0.5])
            states.edge_t_max = cuda.Float([0.5])
            states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            states.prim0 = cuda.Int([-1])
            states.prim1 = cuda.Int([-1])
            states.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            states.src = cuda.Array3f([0.0], [0.0], [1.0])
            states.src_power = cuda.Float([2.0])
            states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            states.prefix_depth = cuda.Int([0])

            states_ad = pj.DfrStatesAD()
            states_ad.count = 1
            states_ad.edge_index = ad.Int([0])
            states_ad.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
            states_ad.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
            states_ad.edge_t_min = ad.Float([-0.5])
            states_ad.edge_t_max = ad.Float([0.5])
            states_ad.n0 = ad.Array3f([0.0], [1.0], [0.0])
            states_ad.n1 = ad.Array3f([0.0], [-1.0], [0.0])
            states_ad.prim0 = ad.Int([-1])
            states_ad.prim1 = ad.Int([-1])
            states_ad.exterior_angle = ad.Float([1.5 * 3.141592653589793])
            states_ad.src = ad.Array3f([0.0], [0.0], [1.0])
            states_ad.src_power = ad.Float([2.0])
            states_ad.wi = ad.Array3f([0.0], [0.0], [-1.0])
            states_ad.d0 = ad.Array3f([0.0], [0.0], [-1.0])
            states_ad.prefix_depth = ad.Int([0])

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            material_ad = pj.DfrMaterialAD()
            material_ad.eta_r = ad.Float([4.0])
            material_ad.sigma = ad.Float([0.0])
            material_ad.mu_r = ad.Float([1.0])
            material_ad.gain = ad.Float([1.0])
            material_ad.valid = ad.Bool([True])

            differentiable = scene.accum_dfr_direct(
                states_ad, make_grid(), material_ad, make_options(), True
            )
            detached = scene.accum_dfr_direct(
                states, make_grid(), material, make_options(), True
            )
            dr.eval(
                differentiable.power,
                detached.power,
                differentiable.direct_count,
                detached.direct_count,
            )
            print(json.dumps({
                "ad_power": float(differentiable.power[0]),
                "detached_power": float(detached.power[0]),
                "ad_direct_count": int(differentiable.direct_count[0]),
                "detached_direct_count": int(detached.direct_count[0]),
            }))
            """
        )

        self.assertEqual(data["ad_direct_count"], 64)
        self.assertEqual(data["detached_direct_count"], 64)
        self.assertAlmostEqual(data["ad_power"], data["detached_power"], delta=2.0e-6)

    def test_accum_dfr_direct_supports_ad_keller_inputs(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z, enable_grad=False):
                src = ad.Array3f([0.0], [0.0], [src_z])
                if enable_grad:
                    dr.enable_grad(src)
                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.5])
                states.edge_t_max = ad.Float([0.5])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([-1])
                states.prim1 = ad.Int([-1])
                states.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                states.src = src
                states.src_power = ad.Float([2.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 17
                options.samples = 64
                options.max_order = 1
                options.direct_samples = 0
                options.keller_samples = 64
                options.strategy_mask = pj.RAYD_DFR_KELLER
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src = dr.grad(src)
                    dr.eval(result.power, result.keller_count, grad_src)
                    return {
                        "result_type": type(result).__name__,
                        "power": float(result.power[0]),
                        "direct_count": int(result.direct_count[0]),
                        "keller_count": int(result.keller_count[0]),
                        "grad_src_z": float(grad_src.z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(1.0, enable_grad=True)
            fd = (run_case(1.0 + step)["loss"] - run_case(1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertGreater(data["power"], 0.0)
        self.assertEqual(data["direct_count"], 0)
        self.assertGreater(data["keller_count"], 0)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=2.0e-3)

    def test_accum_dfr_direct_suffix_backward_matches_finite_difference(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-2.0, 2.0, -2.0],
                                    [0.0, 0.0, 0.0],
                                    [-2.0, -2.0, 2.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z_value, gain_value, enable_grad=False):
                src_z = ad.Float([src_z_value])
                gain = ad.Float([gain_value])
                if enable_grad:
                    dr.enable_grad(src_z, gain)

                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [-1.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.25])
                states.edge_t_max = ad.Float([0.25])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([0])
                states.prim1 = ad.Int([0])
                states.exterior_angle = ad.Float([1.5 * math.pi])
                states.src = ad.Array3f([0.0], [-1.0], src_z)
                states.src_power = ad.Float([1.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 1
                grid.position = -2.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = gain
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 41
                options.samples = 16
                options.max_order = 1
                options.direct_samples = 0
                options.keller_samples = 0
                options.suffix_samples = 16
                options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src = dr.grad(src_z)
                    grad_gain = dr.grad(gain)
                    dr.eval(result.power, result.suffix_count, grad_src, grad_gain)
                    return {
                        "result_type": type(result).__name__,
                        "power": float(result.power[0]),
                        "suffix_count": int(result.suffix_count[0]),
                        "grad_src_z": float(grad_src[0]),
                        "grad_gain": float(grad_gain[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            src_step = 1.0e-3
            gain_step = 1.0e-3
            ad_result = run_case(1.0, 1.0, enable_grad=True)
            fd_src = (
                run_case(1.0 + src_step, 1.0)["loss"] -
                run_case(1.0 - src_step, 1.0)["loss"]
            ) / (2.0 * src_step)
            fd_gain = (
                run_case(1.0, 1.0 + gain_step)["loss"] -
                run_case(1.0, 1.0 - gain_step)["loss"]
            ) / (2.0 * gain_step)
            print(json.dumps({
                **ad_result,
                "fd_src_z": fd_src,
                "fd_gain": fd_gain,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["suffix_count"], 0)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=2.0e-3)
        self.assertAlmostEqual(data["grad_gain"], data["fd_gain"], delta=2.0e-3)

    def test_accum_dfr_direct_suffix_forward_jvp_matches_finite_difference(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-2.0, 2.0, -2.0],
                                    [0.0, 0.0, 0.0],
                                    [-2.0, -2.0, 2.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z_value, enable_grad=False):
                src_z = ad.Float([src_z_value])
                if enable_grad:
                    dr.enable_grad(src_z)

                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [-1.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.25])
                states.edge_t_max = ad.Float([0.25])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([0])
                states.prim1 = ad.Int([0])
                states.exterior_angle = ad.Float([1.5 * math.pi])
                states.src = ad.Array3f([0.0], [-1.0], src_z)
                states.src_power = ad.Float([1.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 1
                grid.position = -2.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 41
                options.samples = 16
                options.max_order = 1
                options.direct_samples = 0
                options.keller_samples = 0
                options.suffix_samples = 16
                options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.set_grad(src_z, ad.Float([1.0]))
                    dr.forward(src_z)
                    jvp = dr.grad(loss)
                    dr.eval(result.power, result.suffix_count, jvp)
                    return {
                        "power": float(result.power[0]),
                        "suffix_count": int(result.suffix_count[0]),
                        "jvp_src_z": float(jvp[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(1.0, enable_grad=True)
            fd = (run_case(1.0 + step)["loss"] - run_case(1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["suffix_count"], 0)
        self.assertAlmostEqual(data["jvp_src_z"], data["fd_src_z"], delta=2.0e-3)

    def test_accum_dfr_direct_suffix_mesh_vertex_backward_matches_finite_difference(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            def run_case(vertex_y_offset, enable_grad=False):
                ty = ad.Float([vertex_y_offset])
                if enable_grad:
                    dr.enable_grad(ty)

                mesh = pj.Mesh(cuda.Array3f([-2.0, 2.0, -2.0],
                                           [0.0, 0.0, 0.0],
                                           [-2.0, -2.0, 2.0]),
                               cuda.Array3i([0], [1], [2]))
                mesh.vertex_positions = ad.Array3f(
                    [-2.0, 2.0, -2.0],
                    ad.Float([0.0, 0.0, 0.0]) + ty,
                    [-2.0, -2.0, 2.0],
                )
                scene = pj.Scene()
                scene.add_mesh(mesh)
                scene.build()

                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [-1.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.25])
                states.edge_t_max = ad.Float([0.25])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([0])
                states.prim1 = ad.Int([0])
                states.exterior_angle = ad.Float([1.5 * math.pi])
                states.src = ad.Array3f([0.0], [-1.0], [1.0])
                states.src_power = ad.Float([1.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 1
                grid.position = -2.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 41
                options.samples = 16
                options.max_order = 1
                options.direct_samples = 0
                options.keller_samples = 0
                options.suffix_samples = 16
                options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

                result = scene.accum_dfr_direct(states, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_ty = dr.grad(ty)
                    dr.eval(result.power, result.suffix_count, grad_ty)
                    return {
                        "power": float(result.power[0]),
                        "suffix_count": int(result.suffix_count[0]),
                        "grad_ty": float(grad_ty[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(0.0, enable_grad=True)
            fd = (run_case(step)["loss"] - run_case(-step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_ty": fd,
            }))
            """
        )

        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["suffix_count"], 0)
        self.assertAlmostEqual(data["grad_ty"], data["fd_ty"], delta=2.0e-6)

    def test_trace_dfr_paths_order1_exports_compact_paths(self):
        data = run_json(
            """
            import json
            import math
            import numpy as np
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            states = pj.DfrStates()
            states.count = 1
            states.edge_index = cuda.Int([0])
            states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            states.edge_t_min = cuda.Float([-0.5])
            states.edge_t_max = cuda.Float([0.5])
            states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            states.prim0 = cuda.Int([-1])
            states.prim1 = cuda.Int([-1])
            states.exterior_angle = cuda.Float([1.5 * math.pi])
            states.src = cuda.Array3f([0.0], [0.0], [1.0])
            states.src_power = cuda.Float([1.0])
            states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            states.prefix_depth = cuda.Int([0])

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DfrPathOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 17
            options.max_order = 1
            options.max_paths = 4
            options.max_rx = 1
            options.strategy_mask = pj.RAYD_DFR_DIRECT
            options.sample_count = 4
            options.return_geom = 1
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

            result = scene.trace_dfr_paths(
                cuda.Array3f([0.0], [0.0], [1.0]),
                cuda.Array3f([0.0], [0.0], [-1.0]),
                states,
                material,
                options,
                cuda.Bool([True]),
            )
            dr.eval(
                result.count,
                result.valid,
                result.rx_id,
                result.edge0,
                result.delay,
                result.field_x.real,
                result.field_x.imag,
                result.p0.x,
            )

            print(json.dumps({
                "capacity": result.capacity,
                "count": int(np.asarray(result.count, dtype=np.int32)[0]),
                "valid0": bool(np.asarray(result.valid, dtype=np.bool_)[0]),
                "rx0": int(np.asarray(result.rx_id, dtype=np.int32)[0]),
                "edge0": int(np.asarray(result.edge0, dtype=np.int32)[0]),
                "delay0": float(np.asarray(result.delay, dtype=np.float32)[0]),
                "field_x_re0": float(np.asarray(result.field_x.real, dtype=np.float32)[0]),
                "field_x_im0": float(np.asarray(result.field_x.imag, dtype=np.float32)[0]),
                "p0_x0": float(np.asarray(result.p0.x, dtype=np.float32)[0]),
            }))
            """
        )

        self.assertEqual(data["capacity"], 1)
        self.assertEqual(data["count"], 1)
        self.assertTrue(data["valid0"])
        self.assertEqual(data["rx0"], 0)
        self.assertEqual(data["edge0"], 0)
        self.assertTrue(math.isfinite(data["delay0"]))
        self.assertGreater(data["delay0"], 0.0)
        self.assertTrue(math.isfinite(data["field_x_re0"]))
        self.assertTrue(math.isfinite(data["field_x_im0"]))
        self.assertAlmostEqual(data["p0_x0"], 0.0, places=5)

    def test_trace_dfr_paths_order1_supports_ad_inputs(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(rx_z, enable_grad=False):
                states = pj.DfrStatesAD()
                states.count = 1
                states.edge_index = ad.Int([0])
                states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                states.edge_t_min = ad.Float([-0.5])
                states.edge_t_max = ad.Float([0.5])
                states.n0 = ad.Array3f([0.0], [1.0], [0.0])
                states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                states.prim0 = ad.Int([-1])
                states.prim1 = ad.Int([-1])
                states.exterior_angle = ad.Float([1.5 * math.pi])
                states.src = ad.Array3f([0.0], [0.0], [1.0])
                states.src_power = ad.Float([1.0])
                states.wi = ad.Array3f([0.0], [0.0], [-1.0])
                states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                states.prefix_depth = ad.Int([0])

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrPathOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 17
                options.max_order = 1
                options.max_paths = 4
                options.max_rx = 1
                options.strategy_mask = pj.RAYD_DFR_DIRECT
                options.sample_count = 4
                options.return_geom = 1
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO

                rx = ad.Array3f([0.0], [0.0], [rx_z])
                if enable_grad:
                    dr.enable_grad(rx)
                result = scene.trace_dfr_paths(
                    ad.Array3f([0.0], [0.0], [1.0]),
                    rx,
                    states,
                    material,
                    options,
                    ad.Bool([True]),
                )
                loss = dr.sum(result.delay)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_rx = dr.grad(rx)
                    dr.eval(result.count, result.valid, result.delay, result.field_x.real, result.p0.x, grad_rx)
                    return {
                        "result_type": type(result).__name__,
                        "capacity": result.capacity,
                        "count": int(result.count[0]),
                        "valid0": bool(result.valid[0]),
                        "delay0": float(result.delay[0]),
                        "field_x_re0": float(result.field_x.real[0]),
                        "p0_x0": float(result.p0.x[0]),
                        "grad_rx_z": float(grad_rx.z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(-1.0, enable_grad=True)
            fd = (run_case(-1.0 + step)["loss"] - run_case(-1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_rx_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrPathsAD")
        self.assertEqual(data["capacity"], 1)
        self.assertEqual(data["count"], 1)
        self.assertTrue(data["valid0"])
        self.assertGreater(data["delay0"], 0.0)
        self.assertTrue(math.isfinite(data["field_x_re0"]))
        self.assertAlmostEqual(data["p0_x0"], 0.0, places=5)
        self.assertAlmostEqual(data["grad_rx_z"], data["fd_rx_z"], delta=2.0e-10)

    def test_accum_dfr_direct_suffix_reflection_writes_grid(self):
        data = run_json(
            """
            import json
            import math
            import numpy as np
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            vertices = cuda.Array3f([-2.0, 2.0, -2.0],
                                    [0.0, 0.0, 0.0],
                                    [-2.0, -2.0, 2.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            states = pj.DfrStates()
            states.count = 1
            states.edge_index = cuda.Int([0])
            states.edge_pos = cuda.Array3f([0.0], [-1.0], [0.0])
            states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            states.edge_t_min = cuda.Float([-0.25])
            states.edge_t_max = cuda.Float([0.25])
            states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            states.prim0 = cuda.Int([0])
            states.prim1 = cuda.Int([0])
            states.exterior_angle = cuda.Float([1.5 * math.pi])
            states.src = cuda.Array3f([0.0], [-1.0], [1.0])
            states.src_power = cuda.Float([1.0])
            states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            states.prefix_depth = cuda.Int([0])

            grid = pj.DfrGrid()
            grid.axis = 1
            grid.position = -2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 41
            options.samples = 16
            options.max_order = 1
            options.direct_samples = 0
            options.keller_samples = 0
            options.suffix_samples = 16
            options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
            options.sample_sequence = pj.RAYD_DFR_HASH
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_debug_counts = True

            result = scene.accum_dfr_direct(
                states,
                grid,
                material,
                options,
                cuda.Bool([True]),
            )
            dr.eval(
                result.power,
                result.direct_count,
                result.keller_count,
                result.suffix_count,
            )
            print(json.dumps({
                "power": float(np.asarray(result.power, dtype=np.float32)[0]),
                "direct": int(np.asarray(result.direct_count, dtype=np.int32)[0]),
                "keller": int(np.asarray(result.keller_count, dtype=np.int32)[0]),
                "suffix": int(np.asarray(result.suffix_count, dtype=np.int32)[0]),
            }))
            """
        )

        self.assertEqual(data["direct"], 0)
        self.assertEqual(data["keller"], 0)
        self.assertGreater(data["suffix"], 0)
        self.assertTrue(math.isfinite(data["power"]))
        self.assertGreater(data["power"], 0.0)

    def test_accum_dfr_direct_suffix_ignores_unrelated_global_candidates(self):
        data = run_json(
            """
            import json
            import math
            import numpy as np
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            # Triangle 0 can specularly connect the diffraction point and grid cell.
            # Triangle 1 is unrelated to the connection and is the only local
            # adjacent face referenced by the diffraction state.
            vertices = cuda.Array3f([-2.0, 2.0, -2.0, 10.0, 11.0, 10.0],
                                    [0.0, 0.0, 0.0, 10.0, 10.0, 11.0],
                                    [-2.0, -2.0, 2.0, 10.0, 10.0, 10.0])
            faces = cuda.Array3i([0, 3], [1, 4], [2, 5])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, faces))
            scene.build()

            states = pj.DfrStates()
            states.count = 1
            states.edge_index = cuda.Int([0])
            states.edge_pos = cuda.Array3f([0.0], [-1.0], [0.0])
            states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            states.edge_t_min = cuda.Float([-0.25])
            states.edge_t_max = cuda.Float([0.25])
            states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            states.prim0 = cuda.Int([1])
            states.prim1 = cuda.Int([1])
            states.exterior_angle = cuda.Float([1.5 * math.pi])
            states.src = cuda.Array3f([0.0], [-1.0], [1.0])
            states.src_power = cuda.Float([1.0])
            states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            states.prefix_depth = cuda.Int([0])

            grid = pj.DfrGrid()
            grid.axis = 1
            grid.position = -2.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0, 4.0])
            material.sigma = cuda.Float([0.0, 0.0])
            material.mu_r = cuda.Float([1.0, 1.0])
            material.gain = cuda.Float([1.0, 1.0])
            material.valid = cuda.Bool([True, False])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 41
            options.samples = 16
            options.max_order = 1
            options.direct_samples = 0
            options.keller_samples = 0
            options.suffix_samples = 16
            options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
            options.sample_sequence = pj.RAYD_DFR_HASH
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_debug_counts = True

            result = scene.accum_dfr_direct(
                states,
                grid,
                material,
                options,
                cuda.Bool([True]),
            )
            dr.eval(result.power, result.suffix_count)
            print(json.dumps({
                "power": float(np.asarray(result.power, dtype=np.float32)[0]),
                "suffix": int(np.asarray(result.suffix_count, dtype=np.int32)[0]),
            }))
            """
        )

        self.assertEqual(data["suffix"], 0)
        self.assertEqual(data["power"], 0.0)

    def test_accum_dfr_direct_accepts_vector_active_mask(self):
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

            states = pj.DfrStates()
            states.count = 2
            states.edge_index = cuda.Int([0, 1])
            states.edge_pos = cuda.Array3f([0.0, 0.5], [0.0, 0.0], [0.0, 0.0])
            states.edge_dir = cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
            states.edge_t_min = cuda.Float([-0.5, -0.5])
            states.edge_t_max = cuda.Float([0.5, 0.5])
            states.n0 = cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
            states.n1 = cuda.Array3f([0.0, 0.0], [-1.0, -1.0], [0.0, 0.0])
            states.prim0 = cuda.Int([-1, -1])
            states.prim1 = cuda.Int([-1, -1])
            states.exterior_angle = cuda.Float([1.5 * 3.141592653589793, 1.5 * 3.141592653589793])
            states.src = cuda.Array3f([0.0, 0.5], [0.0, 0.0], [1.0, 1.0])
            states.src_power = cuda.Float([2.0, 2.0])
            states.wi = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
            states.d0 = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
            states.prefix_depth = cuda.Int([0, 0])

            grid = pj.DfrGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 31
            options.samples = 16
            options.max_order = 1
            options.direct_samples = 8
            options.keller_samples = 8
            options.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
            options.sample_sequence = pj.RAYD_DFR_HASH
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = scene.accum_dfr_direct(
                states, grid, material, options, cuda.Bool([True, False])
            )
            dr.eval(result.power, result.direct_count, result.keller_count)
            print(json.dumps({
                "finite_power": bool(dr.all(dr.isfinite(result.power))),
                "direct_count": int(result.direct_count[0]),
                "keller_count": int(result.keller_count[0]),
            }))
            """
        )

        self.assertTrue(data["finite_power"])
        self.assertGreater(data["direct_count"] + data["keller_count"], 0)

    def test_accum_dfr_order2_direct_and_keller_writes_grid(self):
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

            initial = pj.DfrStates()
            initial.count = 1
            initial.edge_index = cuda.Int([0])
            initial.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            initial.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            initial.edge_t_min = cuda.Float([-0.5])
            initial.edge_t_max = cuda.Float([0.5])
            initial.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            initial.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            initial.prim0 = cuda.Int([-1])
            initial.prim1 = cuda.Int([-1])
            initial.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            initial.src = cuda.Array3f([0.0], [0.0], [1.0])
            initial.src_power = cuda.Float([2.0])
            initial.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            initial.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            initial.prefix_depth = cuda.Int([0])

            recursive = pj.DfrStates()
            recursive.count = 1
            recursive.edge_index = cuda.Int([1])
            recursive.edge_pos = cuda.Array3f([0.0], [0.5], [0.0])
            recursive.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            recursive.edge_t_min = cuda.Float([-0.5])
            recursive.edge_t_max = cuda.Float([0.5])
            recursive.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            recursive.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            recursive.prim0 = cuda.Int([-1])
            recursive.prim1 = cuda.Int([-1])
            recursive.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            recursive.src = cuda.Array3f([0.0], [0.0], [1.0])
            recursive.src_power = cuda.Float([1.0])
            recursive.wi = cuda.Array3f([0.0], [1.0], [0.0])
            recursive.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            recursive.prefix_depth = cuda.Int([0])

            grid = pj.DfrGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 41
            options.samples = 288
            options.max_order = 2
            options.direct_samples = 32
            options.keller_samples = 256
            options.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
            options.sample_sequence = pj.RAYD_DFR_HASH
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = scene.accum_dfr(
                initial, recursive, grid, material, options, True
            )
            dr.eval(
                result.power,
                result.direct_count,
                result.keller_count,
                result.edge_vis_rejects,
                result.edge_uses,
            )
            print(json.dumps({
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.power[0]),
                "direct_count": int(result.direct_count[0]),
                "keller_count": int(result.keller_count[0]),
                "inter_edge_rejects": int(result.edge_vis_rejects[0]),
                "edge_uses": int(result.edge_uses[0]),
            }))
            """
        )

        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["direct_count"], 0)
        self.assertGreater(data["keller_count"], 0)
        self.assertEqual(data["edge_uses"], data["direct_count"] + data["keller_count"])

    def test_accum_dfr_order2_supports_ad_inputs(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z, enable_grad=False):
                src = ad.Array3f([0.0], [0.0], [src_z])
                if enable_grad:
                    dr.enable_grad(src)

                initial = pj.DfrStatesAD()
                initial.count = 1
                initial.edge_index = ad.Int([0])
                initial.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                initial.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                initial.edge_t_min = ad.Float([-0.5])
                initial.edge_t_max = ad.Float([0.5])
                initial.n0 = ad.Array3f([0.0], [1.0], [0.0])
                initial.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                initial.prim0 = ad.Int([-1])
                initial.prim1 = ad.Int([-1])
                initial.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                initial.src = src
                initial.src_power = ad.Float([2.0])
                initial.wi = ad.Array3f([0.0], [0.0], [-1.0])
                initial.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                initial.prefix_depth = ad.Int([0])

                recursive = pj.DfrStatesAD()
                recursive.count = 1
                recursive.edge_index = ad.Int([1])
                recursive.edge_pos = ad.Array3f([0.0], [0.0], [-0.75])
                recursive.edge_dir = ad.Array3f([0.0], [1.0], [0.0])
                recursive.edge_t_min = ad.Float([-0.5])
                recursive.edge_t_max = ad.Float([0.5])
                recursive.n0 = ad.Array3f([1.0], [0.0], [0.0])
                recursive.n1 = ad.Array3f([-1.0], [0.0], [0.0])
                recursive.prim0 = ad.Int([-1])
                recursive.prim1 = ad.Int([-1])
                recursive.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                recursive.src = ad.Array3f([0.0], [0.0], [0.0])
                recursive.src_power = ad.Float([1.0])
                recursive.wi = ad.Array3f([0.0], [0.0], [-1.0])
                recursive.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                recursive.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.5
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 17
                options.samples = 64
                options.max_order = 2
                options.direct_samples = 32
                options.keller_samples = 32
                options.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr(initial, recursive, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src = dr.grad(src)
                    dr.eval(result.power, result.direct_count, result.keller_count, grad_src)
                    return {
                        "result_type": type(result).__name__,
                        "power": float(result.power[0]),
                        "direct_count": int(result.direct_count[0]),
                        "keller_count": int(result.keller_count[0]),
                        "grad_src_z": float(grad_src.z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(1.0, enable_grad=True)
            fd = (run_case(1.0 + step)["loss"] - run_case(1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["direct_count"], 0)
        self.assertGreater(data["keller_count"], 0)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=3.0e-3)

    def test_accum_dfr_order2_suffix_supports_ad_inputs(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-2.0, 2.0, -2.0],
                                    [0.0, 0.0, 0.0],
                                    [-2.0, -2.0, 2.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z_value, mode=None):
                src_z = ad.Float([src_z_value])
                if mode is not None:
                    dr.enable_grad(src_z)

                initial = pj.DfrStatesAD()
                initial.count = 1
                initial.edge_index = ad.Int([0])
                initial.edge_pos = ad.Array3f([0.0], [-1.0], [0.75])
                initial.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                initial.edge_t_min = ad.Float([-0.25])
                initial.edge_t_max = ad.Float([0.25])
                initial.n0 = ad.Array3f([0.0], [1.0], [0.0])
                initial.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                initial.prim0 = ad.Int([0])
                initial.prim1 = ad.Int([0])
                initial.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                initial.src = ad.Array3f([0.0], [-1.0], src_z)
                initial.src_power = ad.Float([1.0])
                initial.wi = ad.Array3f([0.0], [0.0], [-1.0])
                initial.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                initial.prefix_depth = ad.Int([0])

                recursive = pj.DfrStatesAD()
                recursive.count = 1
                recursive.edge_index = ad.Int([1])
                recursive.edge_pos = ad.Array3f([0.0], [-1.0], [0.0])
                recursive.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                recursive.edge_t_min = ad.Float([-0.25])
                recursive.edge_t_max = ad.Float([0.25])
                recursive.n0 = ad.Array3f([0.0], [1.0], [0.0])
                recursive.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                recursive.prim0 = ad.Int([0])
                recursive.prim1 = ad.Int([0])
                recursive.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                recursive.src = ad.Array3f([0.0], [0.0], [0.0])
                recursive.src_power = ad.Float([1.0])
                recursive.wi = ad.Array3f([0.0], [0.0], [-1.0])
                recursive.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                recursive.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 1
                grid.position = -2.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 41
                options.samples = 32
                options.max_order = 2
                options.direct_samples = 0
                options.keller_samples = 0
                options.suffix_samples = 32
                options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr(initial, recursive, grid, material, options, True)
                loss = dr.sum(result.power)
                if mode == "backward":
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src_z = dr.grad(src_z)
                    dr.eval(result.power, result.suffix_count, grad_src_z)
                    return {
                        "result_type": type(result).__name__,
                        "power": float(result.power[0]),
                        "suffix_count": int(result.suffix_count[0]),
                        "grad_src_z": float(grad_src_z[0]),
                    }
                if mode == "forward":
                    dr.set_grad(src_z, ad.Float([1.0]))
                    dr.forward(src_z)
                    jvp_src_z = dr.grad(loss)
                    dr.eval(result.power, result.suffix_count, jvp_src_z)
                    return {
                        "jvp_src_z": float(jvp_src_z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            backward_result = run_case(1.5, mode="backward")
            forward_result = run_case(1.5, mode="forward")
            fd = (run_case(1.5 + step)["loss"] - run_case(1.5 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **backward_result,
                **forward_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["suffix_count"], 0)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=2.0e-5)
        self.assertAlmostEqual(data["jvp_src_z"], data["fd_src_z"], delta=2.0e-5)

    def test_accum_dfr_order2_suffix_mesh_vertex_backward_matches_finite_difference(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            def run_case(vertex_y_offset, enable_grad=False):
                ty = ad.Float([vertex_y_offset])
                if enable_grad:
                    dr.enable_grad(ty)

                mesh = pj.Mesh(cuda.Array3f([-2.0, 2.0, -2.0],
                                           [0.0, 0.0, 0.0],
                                           [-2.0, -2.0, 2.0]),
                               cuda.Array3i([0], [1], [2]))
                mesh.vertex_positions = ad.Array3f(
                    [-2.0, 2.0, -2.0],
                    ad.Float([0.0, 0.0, 0.0]) + ty,
                    [-2.0, -2.0, 2.0],
                )
                scene = pj.Scene()
                scene.add_mesh(mesh)
                scene.build()

                initial = pj.DfrStatesAD()
                initial.count = 1
                initial.edge_index = ad.Int([0])
                initial.edge_pos = ad.Array3f([0.0], [-1.0], [0.75])
                initial.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                initial.edge_t_min = ad.Float([-0.25])
                initial.edge_t_max = ad.Float([0.25])
                initial.n0 = ad.Array3f([0.0], [1.0], [0.0])
                initial.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                initial.prim0 = ad.Int([0])
                initial.prim1 = ad.Int([0])
                initial.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                initial.src = ad.Array3f([0.0], [-1.0], [1.5])
                initial.src_power = ad.Float([1.0])
                initial.wi = ad.Array3f([0.0], [0.0], [-1.0])
                initial.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                initial.prefix_depth = ad.Int([0])

                recursive = pj.DfrStatesAD()
                recursive.count = 1
                recursive.edge_index = ad.Int([1])
                recursive.edge_pos = ad.Array3f([0.0], [-1.0], [0.0])
                recursive.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                recursive.edge_t_min = ad.Float([-0.25])
                recursive.edge_t_max = ad.Float([0.25])
                recursive.n0 = ad.Array3f([0.0], [1.0], [0.0])
                recursive.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                recursive.prim0 = ad.Int([0])
                recursive.prim1 = ad.Int([0])
                recursive.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                recursive.src = ad.Array3f([0.0], [0.0], [0.0])
                recursive.src_power = ad.Float([1.0])
                recursive.wi = ad.Array3f([0.0], [0.0], [-1.0])
                recursive.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                recursive.prefix_depth = ad.Int([0])

                grid = pj.DfrGrid()
                grid.axis = 1
                grid.position = -2.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 41
                options.samples = 32
                options.max_order = 2
                options.direct_samples = 0
                options.keller_samples = 0
                options.suffix_samples = 32
                options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr(initial, recursive, grid, material, options, True)
                loss = dr.sum(result.power)
                if enable_grad:
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_ty = dr.grad(ty)
                    dr.eval(result.power, result.suffix_count, grad_ty)
                    return {
                        "power": float(result.power[0]),
                        "suffix_count": int(result.suffix_count[0]),
                        "grad_ty": float(grad_ty[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            ad_result = run_case(0.0, enable_grad=True)
            fd = (run_case(step)["loss"] - run_case(-step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **ad_result,
                "fd_ty": fd,
            }))
            """
        )

        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["suffix_count"], 0)
        self.assertAlmostEqual(data["grad_ty"], data["fd_ty"], delta=2.0e-6)

    def test_accum_dfr_order3_direct_and_keller_writes_grid(self):
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

            initial = pj.DfrStates()
            initial.count = 1
            initial.edge_index = cuda.Int([0])
            initial.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
            initial.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
            initial.edge_t_min = cuda.Float([-0.5])
            initial.edge_t_max = cuda.Float([0.5])
            initial.n0 = cuda.Array3f([0.0], [1.0], [0.0])
            initial.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
            initial.prim0 = cuda.Int([-1])
            initial.prim1 = cuda.Int([-1])
            initial.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
            initial.src = cuda.Array3f([0.0], [0.0], [1.0])
            initial.src_power = cuda.Float([2.0])
            initial.wi = cuda.Array3f([0.0], [0.0], [-1.0])
            initial.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
            initial.prefix_depth = cuda.Int([0])

            recursive = pj.DfrStates()
            recursive.count = 2
            recursive.edge_index = cuda.Int([1, 2])
            recursive.edge_pos = cuda.Array3f([0.0, 0.0], [0.5, 1.0], [0.0, 0.0])
            recursive.edge_dir = cuda.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
            recursive.edge_t_min = cuda.Float([-0.5, -0.5])
            recursive.edge_t_max = cuda.Float([0.5, 0.5])
            recursive.n0 = cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
            recursive.n1 = cuda.Array3f([0.0, 0.0], [-1.0, -1.0], [0.0, 0.0])
            recursive.prim0 = cuda.Int([-1, -1])
            recursive.prim1 = cuda.Int([-1, -1])
            recursive.exterior_angle = cuda.Float([1.5 * 3.141592653589793,
                                                   1.5 * 3.141592653589793])
            recursive.src = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
            recursive.src_power = cuda.Float([1.0, 1.0])
            recursive.wi = cuda.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
            recursive.d0 = cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
            recursive.prefix_depth = cuda.Int([0, 0])

            grid = pj.DfrGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0

            material = pj.DfrMaterial()
            material.eta_r = cuda.Float([4.0])
            material.sigma = cuda.Float([0.0])
            material.mu_r = cuda.Float([1.0])
            material.gain = cuda.Float([1.0])
            material.valid = cuda.Bool([True])

            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.k = 50.26548245743669
            options.seed = 43
            options.samples = 320
            options.max_order = 3
            options.direct_samples = 64
            options.keller_samples = 256
            options.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
            options.sample_sequence = pj.RAYD_DFR_HASH
            options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
            options.collect_edge_use = True
            options.collect_debug_counts = True

            result = scene.accum_dfr(
                initial, recursive, grid, material, options, True
            )
            dr.eval(
                result.power,
                result.direct_count,
                result.keller_count,
                result.edge_vis_rejects,
                result.edge_uses,
            )
            print(json.dumps({
                "grid_cell_count": result.grid_cell_count,
                "power": float(result.power[0]),
                "direct_count": int(result.direct_count[0]),
                "keller_count": int(result.keller_count[0]),
                "inter_edge_rejects": int(result.edge_vis_rejects[0]),
                "edge_uses": int(result.edge_uses[0]),
            }))
            """
        )

        self.assertEqual(data["grid_cell_count"], 1)
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["direct_count"], 0)
        self.assertGreater(data["keller_count"], 0)
        self.assertEqual(data["edge_uses"], data["direct_count"] + data["keller_count"])

    def test_accum_dfr_order3_supports_ad_inputs(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, 1.0],
                                    [10.0, 10.0, 10.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z_value, mode=None):
                src_z = ad.Float([src_z_value])
                if mode is not None:
                    dr.enable_grad(src_z)

                initial = pj.DfrStatesAD()
                initial.count = 1
                initial.edge_index = ad.Int([0])
                initial.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
                initial.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                initial.edge_t_min = ad.Float([-0.5])
                initial.edge_t_max = ad.Float([0.5])
                initial.n0 = ad.Array3f([0.0], [1.0], [0.0])
                initial.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                initial.prim0 = ad.Int([-1])
                initial.prim1 = ad.Int([-1])
                initial.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                initial.src = ad.Array3f([0.0], [0.0], src_z)
                initial.src_power = ad.Float([2.0])
                initial.wi = ad.Array3f([0.0], [0.0], [-1.0])
                initial.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                initial.prefix_depth = ad.Int([0])

                recursive = pj.DfrStatesAD()
                recursive.count = 2
                recursive.edge_index = ad.Int([1, 2])
                recursive.edge_pos = ad.Array3f([0.0, 0.0], [0.5, 1.0], [0.0, 0.0])
                recursive.edge_dir = ad.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
                recursive.edge_t_min = ad.Float([-0.5, -0.5])
                recursive.edge_t_max = ad.Float([0.5, 0.5])
                recursive.n0 = ad.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
                recursive.n1 = ad.Array3f([0.0, 0.0], [-1.0, -1.0], [0.0, 0.0])
                recursive.prim0 = ad.Int([-1, -1])
                recursive.prim1 = ad.Int([-1, -1])
                recursive.exterior_angle = ad.Float([1.5 * 3.141592653589793,
                                                   1.5 * 3.141592653589793])
                recursive.src = ad.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
                recursive.src_power = ad.Float([1.0, 1.0])
                recursive.wi = ad.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
                recursive.d0 = ad.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
                recursive.prefix_depth = ad.Int([0, 0])

                grid = pj.DfrGrid()
                grid.axis = 2
                grid.position = -1.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 43
                options.samples = 320
                options.max_order = 3
                options.direct_samples = 64
                options.keller_samples = 256
                options.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr(initial, recursive, grid, material, options, True)
                loss = dr.sum(result.power)
                if mode == "backward":
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src_z = dr.grad(src_z)
                    dr.eval(result.power, result.direct_count, result.keller_count, grad_src_z)
                    return {
                        "result_type": type(result).__name__,
                        "power": float(result.power[0]),
                        "direct_count": int(result.direct_count[0]),
                        "keller_count": int(result.keller_count[0]),
                        "grad_src_z": float(grad_src_z[0]),
                    }
                if mode == "forward":
                    dr.set_grad(src_z, ad.Float([1.0]))
                    dr.forward(src_z)
                    jvp_src_z = dr.grad(loss)
                    dr.eval(result.power, result.direct_count, result.keller_count, jvp_src_z)
                    return {
                        "jvp_src_z": float(jvp_src_z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            backward_result = run_case(1.0, mode="backward")
            forward_result = run_case(1.0, mode="forward")
            fd = (run_case(1.0 + step)["loss"] - run_case(1.0 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **backward_result,
                **forward_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["direct_count"], 0)
        self.assertGreater(data["keller_count"], 0)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=3.0e-3)
        self.assertAlmostEqual(data["jvp_src_z"], data["fd_src_z"], delta=3.0e-3)

    def test_accum_dfr_order3_suffix_supports_ad_inputs(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad
            import rayd as pj

            vertices = cuda.Array3f([-2.0, 2.0, -2.0],
                                    [0.0, 0.0, 0.0],
                                    [-2.0, -2.0, 2.0])
            scene = pj.Scene()
            scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
            scene.build()

            def run_case(src_z_value, mode=None):
                src_z = ad.Float([src_z_value])
                if mode is not None:
                    dr.enable_grad(src_z)

                initial = pj.DfrStatesAD()
                initial.count = 1
                initial.edge_index = ad.Int([0])
                initial.edge_pos = ad.Array3f([0.0], [-1.0], [0.75])
                initial.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
                initial.edge_t_min = ad.Float([-0.25])
                initial.edge_t_max = ad.Float([0.25])
                initial.n0 = ad.Array3f([0.0], [1.0], [0.0])
                initial.n1 = ad.Array3f([0.0], [-1.0], [0.0])
                initial.prim0 = ad.Int([0])
                initial.prim1 = ad.Int([0])
                initial.exterior_angle = ad.Float([1.5 * 3.141592653589793])
                initial.src = ad.Array3f([0.0], [-1.0], src_z)
                initial.src_power = ad.Float([1.0])
                initial.wi = ad.Array3f([0.0], [0.0], [-1.0])
                initial.d0 = ad.Array3f([0.0], [0.0], [-1.0])
                initial.prefix_depth = ad.Int([0])

                recursive = pj.DfrStatesAD()
                recursive.count = 2
                recursive.edge_index = ad.Int([1, 2])
                recursive.edge_pos = ad.Array3f([0.0, 0.0], [-1.0, -1.0], [0.25, 0.0])
                recursive.edge_dir = ad.Array3f([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
                recursive.edge_t_min = ad.Float([-0.25, -0.25])
                recursive.edge_t_max = ad.Float([0.25, 0.25])
                recursive.n0 = ad.Array3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
                recursive.n1 = ad.Array3f([0.0, 0.0], [-1.0, -1.0], [0.0, 0.0])
                recursive.prim0 = ad.Int([0, 0])
                recursive.prim1 = ad.Int([0, 0])
                recursive.exterior_angle = ad.Float([1.5 * 3.141592653589793,
                                                   1.5 * 3.141592653589793])
                recursive.src = ad.Array3f([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
                recursive.src_power = ad.Float([1.0, 1.0])
                recursive.wi = ad.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
                recursive.d0 = ad.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0])
                recursive.prefix_depth = ad.Int([0, 0])

                grid = pj.DfrGrid()
                grid.axis = 1
                grid.position = -2.0
                grid.coord0_min = -1.0
                grid.coord0_max = 1.0
                grid.coord1_min = -1.0
                grid.coord1_max = 1.0
                grid.resolution0 = 1
                grid.resolution1 = 1
                grid.cell_area = 4.0

                material = pj.DfrMaterialAD()
                material.eta_r = ad.Float([4.0])
                material.sigma = ad.Float([0.0])
                material.mu_r = ad.Float([1.0])
                material.gain = ad.Float([1.0])
                material.valid = ad.Bool([True])

                options = pj.DfrOptions()
                options.wavelength = 0.125
                options.k = 50.26548245743669
                options.seed = 43
                options.samples = 32
                options.max_order = 3
                options.direct_samples = 0
                options.keller_samples = 0
                options.suffix_samples = 32
                options.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
                options.sample_sequence = pj.RAYD_DFR_HASH
                options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
                options.collect_edge_use = True
                options.collect_debug_counts = True

                result = scene.accum_dfr(initial, recursive, grid, material, options, True)
                loss = dr.sum(result.power)
                if mode == "backward":
                    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                    grad_src_z = dr.grad(src_z)
                    dr.eval(result.power, result.suffix_count, grad_src_z)
                    return {
                        "result_type": type(result).__name__,
                        "power": float(result.power[0]),
                        "suffix_count": int(result.suffix_count[0]),
                        "grad_src_z": float(grad_src_z[0]),
                    }
                if mode == "forward":
                    dr.set_grad(src_z, ad.Float([1.0]))
                    dr.forward(src_z)
                    jvp_src_z = dr.grad(loss)
                    dr.eval(result.power, result.suffix_count, jvp_src_z)
                    return {
                        "jvp_src_z": float(jvp_src_z[0]),
                    }
                dr.eval(loss)
                return {"loss": float(loss[0])}

            step = 1.0e-3
            backward_result = run_case(1.5, mode="backward")
            forward_result = run_case(1.5, mode="forward")
            fd = (run_case(1.5 + step)["loss"] - run_case(1.5 - step)["loss"]) / (2.0 * step)
            print(json.dumps({
                **backward_result,
                **forward_result,
                "fd_src_z": fd,
            }))
            """
        )

        self.assertEqual(data["result_type"], "DfrAccumAD")
        self.assertGreater(data["power"], 0.0)
        self.assertGreater(data["suffix_count"], 0)
        self.assertAlmostEqual(data["grad_src_z"], data["fd_src_z"], delta=2.0e-5)
        self.assertAlmostEqual(data["jvp_src_z"], data["fd_src_z"], delta=2.0e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
