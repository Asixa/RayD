import json
import math
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, timeout: int = 120, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
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


def run_json_case(script: str, timeout: int = 120):
    result = run_script(script, timeout=timeout, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON output.\nSTDERR:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Failed to parse JSON from subprocess.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc


class GeometryCoreTests(unittest.TestCase):
    def test_import_does_not_force_disable_symbolic_loops(self):
        data = run_json_case(
            """
            import json
            import drjit as dr

            before = bool(dr.flag(dr.JitFlag.SymbolicLoops))
            import rayd as pj
            after = bool(dr.flag(dr.JitFlag.SymbolicLoops))

            print(json.dumps({
                "before": before,
                "after": after,
                "has_scene": hasattr(pj, "Scene"),
            }))
            """
        )

        self.assertTrue(data["before"])
        self.assertTrue(data["after"])
        self.assertTrue(data["has_scene"])

    def test_device_selection_api_round_trips_and_preserves_optix_queries(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            device_count = pj.device_count()
            current_before = pj.current_device()
            current_after = pj.set_device(current_before)

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)

            print(json.dumps({
                "device_count": int(device_count),
                "current_before": int(current_before),
                "current_after": int(current_after),
                "valid": bool(its.is_valid()[0]),
                "t": float(its.t[0]),
            }))
            """
        )

        self.assertGreaterEqual(data["device_count"], 1)
        self.assertEqual(data["current_before"], data["current_after"])
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)

    def test_legacy_type_aliases_are_not_exposed(self):
        data = run_json_case(
            """
            import json
            import rayd as pj

            print(json.dumps({
                "has_rayc": hasattr(pj, "RayC"),
                "has_rayd": hasattr(pj, "RayD"),
                "has_intersectionc": hasattr(pj, "IntersectionC"),
                "has_intersectiond": hasattr(pj, "IntersectionD"),
            }))
            """
        )

        self.assertFalse(data["has_rayc"])
        self.assertFalse(data["has_rayd"])
        self.assertFalse(data["has_intersectionc"])
        self.assertFalse(data["has_intersectiond"])

    def test_constant_hit_returns_minimal_intersection(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                          cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "shape": int(its.shape_id[0]),
                "prim": int(its.prim_id[0]),
                "local_prim": int(its.local_prim_id[0]),
                "global_prim": int(its.global_prim_id[0]),
                "t": float(its.t[0]),
                "p": [float(its.p[0][0]), float(its.p[1][0]), float(its.p[2][0])],
                "n": [float(its.n[0][0]), float(its.n[1][0]), float(its.n[2][0])],
                "bary": [float(its.barycentric[0][0]), float(its.barycentric[1][0]), float(its.barycentric[2][0])]
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["shape"], 0)
        self.assertEqual(data["prim"], 0)
        self.assertEqual(data["local_prim"], 0)
        self.assertEqual(data["global_prim"], 0)
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertAlmostEqual(data["p"][0], 0.25, places=5)
        self.assertAlmostEqual(data["p"][1], 0.25, places=5)
        self.assertAlmostEqual(data["p"][2], 0.0, places=5)
        self.assertAlmostEqual(data["n"][2], 1.0, places=5)
        self.assertAlmostEqual(data["bary"][0], 0.5, places=5)
        self.assertAlmostEqual(data["bary"][1], 0.25, places=5)
        self.assertAlmostEqual(data["bary"][2], 0.25, places=5)

    def test_miss_returns_stable_sentinel_values(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            ray = pj.RayDetached(cuda.Array3f([2.0], [2.0], [-1.0]),
                          cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "shape": int(its.shape_id[0]),
                "prim": int(its.prim_id[0]),
                "t_is_inf": math.isinf(float(its.t[0]))
            }))
            """
        )

        self.assertFalse(data["valid"])
        self.assertEqual(data["shape"], -1)
        self.assertEqual(data["prim"], -1)
        self.assertTrue(data["t_is_inf"])

    def test_intersect_preserves_batch_shapes_across_flags_and_miss_modes(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            none_flag = getattr(pj.RayFlags, "None")

            def vec3_rows(vec):
                return [
                    [float(vec[0][i]), float(vec[1][i]), float(vec[2][i])]
                    for i in range(dr.width(vec[0]))
                ]

            def vec2_rows(vec):
                return [
                    [float(vec[0][i]), float(vec[1][i])]
                    for i in range(dr.width(vec[0]))
                ]

            def summarize(its):
                return {
                    "widths": {
                        "t": dr.width(its.t),
                        "p": dr.width(its.p[0]),
                        "n": dr.width(its.n[0]),
                        "geo_n": dr.width(its.geo_n[0]),
                        "uv": dr.width(its.uv[0]),
                        "barycentric": dr.width(its.barycentric[0]),
                        "shape_id": dr.width(its.shape_id),
                        "prim_id": dr.width(its.prim_id),
                    },
                    "valid": [bool(v) for v in list(its.is_valid())],
                    "t_inf": [math.isinf(float(v)) for v in list(its.t)],
                    "p": vec3_rows(its.p),
                    "n": vec3_rows(its.n),
                    "geo_n": vec3_rows(its.geo_n),
                    "uv": vec2_rows(its.uv),
                    "bary": vec3_rows(its.barycentric),
                }

            mixed_rays = pj.RayDetached(cuda.Array3f([0.25, 2.0], [0.25, 2.0], [-1.0, -1.0]),
                                        cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))
            miss_rays = pj.RayDetached(cuda.Array3f([2.0, 3.0], [2.0, 3.0], [-1.0, -1.0]),
                                       cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))
            tmax_rays = pj.RayDetached(cuda.Array3f([0.25, 0.75], [0.25, 0.1], [-1.0, -1.0]),
                                       cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))
            tmax_rays.tmax = cuda.Float([0.5, 0.5])

            ad_rays = pj.Ray(ad.Array3f([0.25, 2.0], [0.25, 2.0], [-1.0, -1.0]),
                             ad.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))

            print(json.dumps({
                "detached_none": summarize(scene.intersect(mixed_rays, flags=none_flag)),
                "detached_geometric": summarize(scene.intersect(mixed_rays, flags=pj.RayFlags.Geometric)),
                "detached_spatial_miss": summarize(scene.intersect(miss_rays, flags=pj.RayFlags.Geometric)),
                "detached_tmax_miss": summarize(scene.intersect(tmax_rays, flags=pj.RayFlags.Geometric)),
                "ad_none": summarize(scene.intersect(ad_rays, flags=none_flag)),
            }))
            """
        )

        expected_fields = {"t", "p", "n", "geo_n", "uv", "barycentric", "shape_id", "prim_id"}
        for case_name, case in data.items():
            self.assertEqual(set(case["widths"].keys()), expected_fields, msg=case_name)
            for field_name, width in case["widths"].items():
                self.assertEqual(width, 2, msg=f"{case_name}:{field_name}")

        self.assertEqual(data["detached_none"]["valid"], [True, False])
        self.assertEqual(data["ad_none"]["valid"], [True, False])
        self.assertEqual(data["detached_none"]["n"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(data["detached_none"]["geo_n"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(data["detached_none"]["uv"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(data["ad_none"]["n"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(data["ad_none"]["geo_n"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(data["ad_none"]["uv"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertAlmostEqual(data["detached_geometric"]["geo_n"][0][2], 1.0, places=5)
        self.assertEqual(data["detached_geometric"]["uv"], [[0.0, 0.0], [0.0, 0.0]])

        for miss_case in ("detached_spatial_miss", "detached_tmax_miss"):
            self.assertEqual(data[miss_case]["valid"], [False, False], msg=miss_case)
            self.assertEqual(data[miss_case]["t_inf"], [True, True], msg=miss_case)
            self.assertEqual(data[miss_case]["p"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], msg=miss_case)
            self.assertEqual(data[miss_case]["n"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], msg=miss_case)
            self.assertEqual(data[miss_case]["geo_n"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], msg=miss_case)
            self.assertEqual(data[miss_case]["uv"], [[0.0, 0.0], [0.0, 0.0]], msg=miss_case)
            self.assertEqual(data[miss_case]["bary"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], msg=miss_case)

    def test_shadow_test_returns_hit_mask_without_intersection_payload(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            rays = pj.RayDetached(cuda.Array3f([0.25, 2.0], [0.25, 2.0], [-1.0, -1.0]),
                                  cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))
            shadow = scene.shadow_test(rays)

            print(json.dumps({
                "type": type(shadow).__name__,
                "values": [bool(v) for v in list(shadow)],
            }))
            """
        )

        self.assertEqual(data["values"], [True, False])

    def test_split_queries_remain_lazy_until_results_are_consumed(self):
        data = run_json_case(
            """
            import os
            os.environ["RAYD_OPTIX_SPLIT_MODE"] = "on"

            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda

            static_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0],
                                               [0.0, 0.0, 0.0]),
                                  cuda.Array3i([0], [1], [2]))
            dynamic_mesh = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                                [0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0]),
                                   cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(static_mesh, dynamic=False)
            scene.add_mesh(dynamic_mesh, dynamic=True)
            scene.build()

            static_ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                        cuda.Array3f([0.0], [0.0], [1.0]))
            dynamic_ray = pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                                         cuda.Array3f([0.0], [0.0], [1.0]))

            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                its = scene.intersect(static_ray)
                shadow = scene.shadow_test(dynamic_ray)
                lazy_hist = dr.kernel_history()

                t_value = float(its.t[0])
                shadow_value = bool(shadow[0])
                consume_hist = dr.kernel_history()

            print(json.dumps({
                "lazy_jit": sum(1 for h in lazy_hist if str(h.get("type")) == "KernelType.JIT"),
                "lazy_optix": sum(1 for h in lazy_hist if bool(h.get("uses_optix", False))),
                "consume_jit": sum(1 for h in consume_hist if str(h.get("type")) == "KernelType.JIT"),
                "consume_optix": sum(1 for h in consume_hist if bool(h.get("uses_optix", False))),
                "t": t_value,
                "shadow": shadow_value,
            }))
            """
        )

        self.assertEqual(data["lazy_jit"], 0)
        self.assertEqual(data["lazy_optix"], 0)
        self.assertGreaterEqual(data["consume_jit"], 1)
        self.assertGreaterEqual(data["consume_optix"], 1)
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertTrue(data["shadow"])

    def test_split_queries_work_in_default_symbolic_loop_mode_without_manual_eval(self):
        data = run_json_case(
            """
            import os
            os.environ["RAYD_OPTIX_SPLIT_MODE"] = "on"

            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            static_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0],
                                               [0.0, 0.0, 0.0]),
                                  cuda.Array3i([0], [1], [2]))
            dynamic_mesh = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                                [0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0]),
                                   cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(static_mesh, dynamic=False)
            scene.add_mesh(dynamic_mesh, dynamic=True)
            scene.build()

            direction = ad.Array3f([0.0], [0.0], [1.0])

            def body(i, origin, acc):
                ray = pj.Ray(origin, direction)
                its = scene.intersect(ray)
                shadow = scene.shadow_test(ray)
                next_origin = its.p + ad.Array3f([0.0], [0.0], [-1.0])
                return i + 1, next_origin, acc + its.t + dr.select(shadow, 1.0, 0.0)

            iterations, origin_out, acc = dr.while_loop(
                state=(
                    cuda.Int([0]),
                    ad.Array3f([0.25], [0.25], [-1.0]),
                    ad.Float([0.0]),
                ),
                cond=lambda i, origin, acc: i < 2,
                body=body,
            )

            print(json.dumps({
                "symbolic_loops": bool(dr.flag(dr.JitFlag.SymbolicLoops)),
                "iterations": int(iterations[0]),
                "acc": float(acc[0]),
                "origin_z": float(origin_out[2][0]),
            }))
            """
        )

        self.assertTrue(data["symbolic_loops"])
        self.assertEqual(data["iterations"], 2)
        self.assertAlmostEqual(data["acc"], 4.0, places=5)
        self.assertAlmostEqual(data["origin_z"], -1.0, places=5)

    def test_nearest_edge_queries_remain_correct_when_symbolic_loops_enabled(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda

            dr.set_flag(dr.JitFlag.SymbolicLoops, True)

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0]),
                          cuda.Array3i([0, 0], [1, 2], [2, 3]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            point = cuda.Array3f([0.5], [-0.2], [0.0])
            point_hit = scene.nearest_edge(point)

            ray = pj.RayDetached(cuda.Array3f([0.5], [0.5], [1.0]),
                                 cuda.Array3f([0.0], [0.0], [-1.0]))
            ray_hit = scene.nearest_edge(ray)

            print(json.dumps({
                "symbolic_loops": bool(dr.flag(dr.JitFlag.SymbolicLoops)),
                "point_valid": bool(point_hit.is_valid()[0]),
                "point_distance": float(point_hit.distance[0]),
                "point_edge": int(point_hit.edge_id[0]),
                "ray_valid": bool(ray_hit.is_valid()[0]),
                "ray_distance": float(ray_hit.distance[0]),
                "ray_edge": int(ray_hit.edge_id[0]),
            }))
            """
        )

        self.assertTrue(data["symbolic_loops"])
        self.assertTrue(data["point_valid"])
        self.assertGreaterEqual(data["point_distance"], 0.0)
        self.assertGreaterEqual(data["point_edge"], 0)
        self.assertTrue(data["ray_valid"])
        self.assertGreaterEqual(data["ray_distance"], 0.0)
        self.assertGreaterEqual(data["ray_edge"], 0)

    def test_trace_reflections_defaults_to_symbolic_trace_in_symbolic_loop(self):
        data = run_json_case(
            """
            import os
            os.environ["RAYD_OPTIX_SPLIT_MODE"] = "on"

            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            static_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0],
                                               [0.0, 0.0, 0.0]),
                                  cuda.Array3i([0], [1], [2]))
            dynamic_mesh = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                                [0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0]),
                                   cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(static_mesh, dynamic=False)
            scene.add_mesh(dynamic_mesh, dynamic=True)
            scene.build()

            direction = ad.Array3f([0.0], [0.0], [1.0])

            def body(i, origin, acc):
                trace = scene.trace_reflections(pj.Ray(origin, direction), max_bounces=1)
                bounce = trace.bounce(0)
                next_origin = bounce.hit_points + ad.Array3f([0.0], [0.0], [-1.0])
                return i + 1, next_origin, acc + bounce.t + dr.select(bounce.is_valid(), 1.0, 0.0)

            iterations, origin_out, acc = dr.while_loop(
                state=(
                    cuda.Int([0]),
                    ad.Array3f([2.25], [0.25], [-1.0]),
                    ad.Float([0.0]),
                ),
                cond=lambda i, origin, acc: i < 2,
                body=body,
            )

            print(json.dumps({
                "symbolic_loops": bool(dr.flag(dr.JitFlag.SymbolicLoops)),
                "iterations": int(iterations[0]),
                "acc": float(acc[0]),
                "origin_z": float(origin_out[2][0]),
            }))
            """
        )

        self.assertTrue(data["symbolic_loops"])
        self.assertEqual(data["iterations"], 2)
        self.assertAlmostEqual(data["acc"], 4.0, places=5)
        self.assertAlmostEqual(data["origin_z"], -1.0, places=5)

    def test_trace_reflections_returns_expected_two_bounce_chain(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda

            wall = pj.Mesh(
                cuda.Array3f([1.0, 1.0, 1.0, 1.0],
                             [-1.0, 1.0, 1.0, -1.0],
                             [0.0, 0.0, 2.0, 2.0]),
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
            chain = scene.trace_reflections(ray, max_bounces=3, symbolic=False)
            dr.eval(chain.bounce_count, chain.t, chain.hit_points,
                    chain.image_sources, chain.shape_ids, chain.prim_ids,
                    chain.local_prim_ids, chain.global_prim_ids)
            dr.sync_thread()

            print(json.dumps({
                "bounce_count": int(chain.bounce_count[0]),
                "valid": [bool(v) for v in list(chain.is_valid())],
                "face_offsets": [int(v) for v in list(scene.mesh_face_offsets())],
                "shape_ids": [int(v) for v in list(chain.shape_ids)],
                "local_prim_ids": [int(v) for v in list(chain.local_prim_ids)],
                "global_prim_ids": [int(v) for v in list(chain.global_prim_ids)],
                "compat_prim_ids": [int(v) for v in list(chain.prim_ids)],
                "t": [float(v) for v in list(chain.t)],
                "hit0": [float(chain.hit_points[0][0]), float(chain.hit_points[1][0]), float(chain.hit_points[2][0])],
                "hit1": [float(chain.hit_points[0][1]), float(chain.hit_points[1][1]), float(chain.hit_points[2][1])],
                "img0": [float(chain.image_sources[0][0]), float(chain.image_sources[1][0]), float(chain.image_sources[2][0])],
                "img1": [float(chain.image_sources[0][1]), float(chain.image_sources[1][1]), float(chain.image_sources[2][1])],
            }))
            """
        )

        self.assertEqual(data["bounce_count"], 2)
        self.assertEqual(data["valid"], [True, True, False])
        self.assertEqual(data["shape_ids"][:2], [0, 1])
        self.assertEqual(data["compat_prim_ids"], data["local_prim_ids"])
        for shape_id, local_prim_id, global_prim_id in zip(
            data["shape_ids"], data["local_prim_ids"], data["global_prim_ids"]
        ):
            if local_prim_id < 0:
                self.assertEqual(global_prim_id, -1)
            else:
                self.assertEqual(global_prim_id, data["face_offsets"][shape_id] + local_prim_id)
        self.assertAlmostEqual(data["t"][0], math.sqrt(2.0), places=4)
        self.assertAlmostEqual(data["t"][1], math.sqrt(0.5), places=4)
        self.assertAlmostEqual(data["hit0"][0], 1.0, places=4)
        self.assertAlmostEqual(data["hit0"][1], 0.0, places=4)
        self.assertAlmostEqual(data["hit0"][2], 1.5, places=4)
        self.assertAlmostEqual(data["hit1"][0], 0.5, places=4)
        self.assertAlmostEqual(data["hit1"][1], 0.0, places=4)
        self.assertAlmostEqual(data["hit1"][2], 2.0, places=4)
        self.assertAlmostEqual(data["img0"][0], 2.0, places=4)
        self.assertAlmostEqual(data["img0"][1], 0.0, places=4)
        self.assertAlmostEqual(data["img0"][2], 0.5, places=4)
        self.assertAlmostEqual(data["img1"][0], 2.0, places=4)
        self.assertAlmostEqual(data["img1"][1], 0.0, places=4)
        self.assertAlmostEqual(data["img1"][2], 3.5, places=4)

    def test_trace_reflections_preserves_gradients_for_ad_inputs(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 0.0, 0.0]),
                           cuda.Array3i([0], [1], [2]))

            verts = ad.Array3f([0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0])
            dr.enable_grad(verts)
            mesh.vertex_positions = verts

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                         ad.Array3f([0.0], [0.0], [1.0]))
            chain = scene.trace_reflections(ray, max_bounces=1, symbolic=False)
            dr.backward(dr.sum(chain.t))
            grad = dr.grad(verts)

            print(json.dumps({
                "bounce_count": int(chain.bounce_count[0]),
                "valid": bool(chain.is_valid()[0]),
                "t": float(chain.t[0]),
                "grad_z_sum": float(grad[2][0] + grad[2][1] + grad[2][2]),
            }))
            """
        )

        self.assertEqual(data["bounce_count"], 1)
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertGreater(data["grad_z_sum"], 0.0)

    def test_trace_reflections_preserves_gradients_for_ad_mesh_constructor_inputs(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            verts = ad.Array3f([0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0])
            dr.enable_grad(verts)

            mesh = pj.Mesh(verts, cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                         ad.Array3f([0.0], [0.0], [1.0]))
            chain = scene.trace_reflections(ray, max_bounces=1, symbolic=False)
            dr.backward(dr.sum(chain.t))
            grad = dr.grad(verts)

            print(json.dumps({
                "bounce_count": int(chain.bounce_count[0]),
                "valid": bool(chain.is_valid()[0]),
                "t": float(chain.t[0]),
                "grad_z_sum": float(grad[2][0] + grad[2][1] + grad[2][2]),
            }))
            """
        )

        self.assertEqual(data["bounce_count"], 1)
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertGreater(data["grad_z_sum"], 0.0)

    def test_trace_reflections_supports_multi_bounce_symbolic_trace(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            wall = pj.Mesh(
                cuda.Array3f([1.0, 1.0, 1.0, 1.0],
                             [-1.0, 1.0, 1.0, -1.0],
                             [0.0, 0.0, 2.0, 2.0]),
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
            ray_o = ad.Array3f([0.0], [0.0], [0.5])
            ray_d = ad.Array3f([inv_sqrt2], [0.0], [inv_sqrt2])

            def body(i, acc):
                trace = scene.trace_reflections(
                    pj.Ray(ray_o, ray_d),
                    max_bounces=2,
                    symbolic=True,
                )
                bounce0 = trace.bounce(0)
                bounce1 = trace.bounce(1)
                total = bounce0.t + dr.select(bounce1.is_valid(), bounce1.t, 0.0)
                return i + 1, acc + total

            iterations, acc = dr.while_loop(
                state=(cuda.Int([0]), ad.Float([0.0])),
                cond=lambda i, acc: i < 2,
                body=body,
            )

            print(json.dumps({
                "symbolic_loops": bool(dr.flag(dr.JitFlag.SymbolicLoops)),
                "iterations": int(iterations[0]),
                "acc": float(acc[0]),
            }))
            """
        )

        self.assertTrue(data["symbolic_loops"])
        self.assertEqual(data["iterations"], 2)
        self.assertAlmostEqual(data["acc"], 2.0 * (math.sqrt(2.0) + math.sqrt(0.5)), places=4)

    def test_trace_reflections_returns_expected_two_bounce_symbolic_trace(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda

            wall = pj.Mesh(
                cuda.Array3f([1.0, 1.0, 1.0, 1.0],
                             [-1.0, 1.0, 1.0, -1.0],
                             [0.0, 0.0, 2.0, 2.0]),
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
            trace = scene.trace_reflections(ray, max_bounces=3, symbolic=True)
            bounce0 = trace.bounce(0)
            bounce1 = trace.bounce(1)
            bounce2 = trace.bounce(2)
            dr.eval(trace.bounce_count,
                    trace.dedup_keep_mask,
                    trace.discovery_count,
                    trace.representative_ray_index,
                    bounce0.t,
                    bounce0.hit_points,
                    bounce0.image_sources,
                    bounce1.t,
                    bounce1.hit_points,
                    bounce1.image_sources,
                    bounce2.t)
            dr.sync_thread()

            print(json.dumps({
                "max_bounces": int(trace.max_bounces),
                "num_bounces": len(trace.bounces),
                "bounce_count": int(trace.bounce_count[0]),
                "keep_mask": bool(trace.dedup_keep_mask[0]),
                "discovery_count": int(trace.discovery_count[0]),
                "representative_ray_index": int(trace.representative_ray_index[0]),
                "bounce0_valid": bool(bounce0.is_valid()[0]),
                "bounce1_valid": bool(bounce1.is_valid()[0]),
                "bounce2_valid": bool(bounce2.is_valid()[0]),
                "bounce0_t": float(bounce0.t[0]),
                "bounce1_t": float(bounce1.t[0]),
                "hit0": [float(bounce0.hit_points[0][0]), float(bounce0.hit_points[1][0]), float(bounce0.hit_points[2][0])],
                "hit1": [float(bounce1.hit_points[0][0]), float(bounce1.hit_points[1][0]), float(bounce1.hit_points[2][0])],
                "img0": [float(bounce0.image_sources[0][0]), float(bounce0.image_sources[1][0]), float(bounce0.image_sources[2][0])],
                "img1": [float(bounce1.image_sources[0][0]), float(bounce1.image_sources[1][0]), float(bounce1.image_sources[2][0])],
            }))
            """
        )

        self.assertEqual(data["max_bounces"], 3)
        self.assertEqual(data["num_bounces"], 3)
        self.assertEqual(data["bounce_count"], 2)
        self.assertTrue(data["keep_mask"])
        self.assertEqual(data["discovery_count"], 1)
        self.assertEqual(data["representative_ray_index"], 0)
        self.assertTrue(data["bounce0_valid"])
        self.assertTrue(data["bounce1_valid"])
        self.assertFalse(data["bounce2_valid"])
        self.assertAlmostEqual(data["bounce0_t"], math.sqrt(2.0), places=4)
        self.assertAlmostEqual(data["bounce1_t"], math.sqrt(0.5), places=4)
        self.assertAlmostEqual(data["hit0"][0], 1.0, places=4)
        self.assertAlmostEqual(data["hit0"][1], 0.0, places=4)
        self.assertAlmostEqual(data["hit0"][2], 1.5, places=4)
        self.assertAlmostEqual(data["hit1"][0], 0.5, places=4)
        self.assertAlmostEqual(data["hit1"][1], 0.0, places=4)
        self.assertAlmostEqual(data["hit1"][2], 2.0, places=4)
        self.assertAlmostEqual(data["img0"][0], 2.0, places=4)
        self.assertAlmostEqual(data["img0"][1], 0.0, places=4)
        self.assertAlmostEqual(data["img0"][2], 0.5, places=4)
        self.assertAlmostEqual(data["img1"][0], 2.0, places=4)
        self.assertAlmostEqual(data["img1"][1], 0.0, places=4)
        self.assertAlmostEqual(data["img1"][2], 3.5, places=4)

    def test_trace_reflections_symbolic_trace_preserves_gradients_for_ad_inputs(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 0.0, 0.0]),
                           cuda.Array3i([0], [1], [2]))

            verts = ad.Array3f([0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0])
            dr.enable_grad(verts)
            mesh.vertex_positions = verts

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                         ad.Array3f([0.0], [0.0], [1.0]))
            trace = scene.trace_reflections(ray, max_bounces=2, symbolic=True)
            bounce0 = trace.bounce(0)
            dr.backward(dr.sum(bounce0.t))
            grad = dr.grad(verts)

            print(json.dumps({
                "bounce_count": int(trace.bounce_count[0]),
                "valid": bool(bounce0.is_valid()[0]),
                "t": float(bounce0.t[0]),
                "grad_z_sum": float(grad[2][0] + grad[2][1] + grad[2][2]),
            }))
            """
        )

        self.assertEqual(data["bounce_count"], 1)
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertGreater(data["grad_z_sum"], 0.0)

    def test_trace_reflections_symbolic_trace_rejects_deduplicate_for_now(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 0.0, 0.0]),
                           cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))

            error = ""
            try:
                scene.trace_reflections(ray, max_bounces=1, deduplicate=True, symbolic=True)
            except RuntimeError as exc:
                error = str(exc)

            print(json.dumps({"error": error}))
            """
        )

        self.assertIn("deduplicate=true is not implemented with symbolic=true yet", data["error"])

    def test_trace_reflections_gpu_deduplicate_merges_duplicate_and_canonical_paths(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda

            wall = pj.Mesh(
                cuda.Array3f([1.0, 1.0, 1.0, 1.0],
                             [-1.0, 1.0, 1.0, -1.0],
                             [0.0, 0.0, 2.0, 2.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(wall)
            scene.build()

            ray = pj.RayDetached(
                cuda.Array3f([0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0]),
                cuda.Array3f([1.0, 1.0, 1.0],
                             [0.2, 0.2, -0.6],
                             [0.2, 0.2, 0.2]),
            )

            chain = scene.trace_reflections(ray, max_bounces=1, deduplicate=True, symbolic=False)
            chain_canon = scene.trace_reflections(
                ray,
                max_bounces=1,
                deduplicate=True,
                canonical_prim_table=cuda.Int([0, 0]),
                symbolic=False,
            )
            dr.eval(chain.discovery_count,
                    chain.hit_points,
                    chain.plane_points,
                    chain.geo_normals,
                    chain.plane_normals,
                    chain_canon.discovery_count)
            dr.sync_thread()

            plane_match = all(
                abs(float(chain.hit_points[axis][i]) - float(chain.plane_points[axis][i])) < 1e-6
                for axis in range(3)
                for i in range(dr.width(chain.hit_points[0]))
            )
            normal_match = all(
                abs(float(chain.geo_normals[axis][i]) - float(chain.plane_normals[axis][i])) < 1e-6
                for axis in range(3)
                for i in range(dr.width(chain.geo_normals[0]))
            )

            print(json.dumps({
                "ray_count": int(chain.ray_count),
                "discovery": sorted(int(v) for v in list(chain.discovery_count)),
                "plane_match": plane_match,
                "normal_match": normal_match,
                "canon_ray_count": int(chain_canon.ray_count),
                "canon_discovery": [int(v) for v in list(chain_canon.discovery_count)],
            }))
            """
        )

        self.assertEqual(data["ray_count"], 2)
        self.assertEqual(data["discovery"], [1, 2])
        self.assertTrue(data["plane_match"])
        self.assertTrue(data["normal_match"])
        self.assertEqual(data["canon_ray_count"], 1)
        self.assertEqual(data["canon_discovery"], [3])

    def test_shadow_test_honors_active_mask_and_pending_update_guard(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            rays = pj.RayDetached(cuda.Array3f([0.25, 0.25], [0.25, 0.25], [-1.0, -1.0]),
                                  cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))
            active = cuda.Bool([True, False])
            masked = scene.shadow_test(rays, active)

            scene.update_mesh_vertices(
                mesh_id,
                cuda.Array3f([2.0, 3.0, 2.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0])
            )

            pending_error = False
            try:
                scene.shadow_test(
                    pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                                   cuda.Array3f([0.0], [0.0], [1.0]))
                )
            except Exception as e:
                pending_error = "pending updates" in str(e)

            print(json.dumps({
                "masked": [bool(v) for v in list(masked)],
                "pending_error": pending_error,
            }))
            """
        )

        self.assertEqual(data["masked"], [True, False])
        self.assertTrue(data["pending_error"])

    def test_recreating_scenes_keeps_optix_traces_valid(self):
        data = run_json_case(
            """
            import gc
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            def make_scene():
                mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0],
                                           [0.0, 0.0, 0.0]),
                              cuda.Array3i([0], [1], [2]))
                scene = pj.Scene()
                scene.add_mesh(mesh)
                scene.build()
                return scene

            def run_forward():
                scene = make_scene()
                ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                     cuda.Array3f([0.0], [0.0], [1.0]))
                its = scene.intersect(ray)
                dr.eval(its.t, its.shape_id, its.prim_id)
                dr.sync_thread()
                result = {
                    "valid": bool(its.is_valid()[0]),
                    "shape": int(its.shape_id[0]),
                    "prim": int(its.prim_id[0]),
                }
                del scene
                gc.collect()
                dr.sync_thread()
                return result

            def run_gradient():
                mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0],
                                           [0.0, 0.0, 0.0]),
                              cuda.Array3i([0], [1], [2]))
                verts = ad.Array3f([0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0])
                dr.enable_grad(verts)
                mesh.vertex_positions = verts
                scene = pj.Scene()
                scene.add_mesh(mesh)
                scene.build()
                ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                             ad.Array3f([0.0], [0.0], [1.0]))
                its = scene.intersect(ray)
                loss = dr.sum(its.t)
                dr.backward(loss)
                grad = dr.grad(verts)
                dr.eval(loss, grad)
                dr.sync_thread()
                result = {
                    "loss": float(loss[0]),
                    "grad_z_sum": float(grad[2][0] + grad[2][1] + grad[2][2]),
                }
                del scene
                gc.collect()
                dr.sync_thread()
                return result

            first_forward = run_forward()
            second_forward = run_forward()
            first_gradient = run_gradient()
            second_gradient = run_gradient()

            print(json.dumps({
                "first_forward": first_forward,
                "second_forward": second_forward,
                "first_gradient": first_gradient,
                "second_gradient": second_gradient,
            }))
            """
        )

        self.assertTrue(data["first_forward"]["valid"])
        self.assertTrue(data["second_forward"]["valid"])
        self.assertEqual(data["first_forward"]["shape"], 0)
        self.assertEqual(data["second_forward"]["shape"], 0)
        self.assertEqual(data["first_forward"]["prim"], 0)
        self.assertEqual(data["second_forward"]["prim"], 0)
        self.assertGreater(data["first_gradient"]["loss"], 0.0)
        self.assertGreater(data["second_gradient"]["loss"], 0.0)
        self.assertAlmostEqual(data["first_gradient"]["loss"], data["second_gradient"]["loss"], places=5)
        self.assertAlmostEqual(
            data["first_gradient"]["grad_z_sum"],
            data["second_gradient"]["grad_z_sum"],
            places=5,
        )

    def test_multi_mesh_hit_reports_shape_and_local_prim(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh_a = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                            cuda.Array3i([0], [1], [2]))

            mesh_b = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                            cuda.Array3i([0], [1], [2]))
            mesh_b.vertex_positions = ad.Array3f([2.0, 3.0, 2.0],
                                                 [0.0, 0.0, 1.0],
                                                 [0.0, 0.0, 0.0])

            scene = pj.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            ray = pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                          cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "shape": int(its.shape_id[0]),
                "prim": int(its.prim_id[0])
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["shape"], 1)
        self.assertEqual(data["prim"], 0)

    def test_uv_queries_handle_present_and_missing_uvs(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                          cuda.Array3f([0.0], [0.0], [1.0]))

            mesh0 = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 0.0, 0.0]),
                           cuda.Array3i([0], [1], [2]))
            scene0 = pj.Scene()
            scene0.add_mesh(mesh0)
            scene0.build()
            its0 = scene0.intersect(ray)

            mesh1 = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 0.0, 0.0]),
                           cuda.Array3i([0], [1], [2]),
                           cuda.Array2f([0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]))
            scene1 = pj.Scene()
            scene1.add_mesh(mesh1)
            scene1.build()
            its1 = scene1.intersect(ray)

            print(json.dumps({
                "uv0": [float(its0.uv[0][0]), float(its0.uv[1][0])],
                "uv1": [float(its1.uv[0][0]), float(its1.uv[1][0])]
            }))
            """
        )

        self.assertAlmostEqual(data["uv0"][0], 0.0, places=6)
        self.assertAlmostEqual(data["uv0"][1], 0.0, places=6)
        self.assertAlmostEqual(data["uv1"][0], 0.25, places=5)
        self.assertAlmostEqual(data["uv1"][1], 0.25, places=5)

    def test_vertex_gradients_flow_through_intersection(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            verts = ad.Array3f([0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0])
            dr.enable_grad(verts)
            mesh.vertex_positions = verts

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                          ad.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            dr.backward(its.t)
            grad = dr.grad(verts)

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "grad_x_sum": float(grad[0][0] + grad[0][1] + grad[0][2]),
                "grad_y_sum": float(grad[1][0] + grad[1][1] + grad[1][2]),
                "grad_z_sum": float(grad[2][0] + grad[2][1] + grad[2][2])
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["grad_x_sum"], 0.0, places=5)
        self.assertAlmostEqual(data["grad_y_sum"], 0.0, places=5)
        self.assertGreater(data["grad_z_sum"], 0.0)

    def test_transform_gradients_flow_through_intersection(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            tz = ad.Float([0.0])
            dr.enable_grad(tz)
            mesh.to_world_left = ad.Matrix4f([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ])

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                          ad.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            dr.backward(its.t)

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "grad_tz": float(dr.grad(tz)[0]),
                "t": float(its.t[0])
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["grad_tz"], 1.0, places=5)
        self.assertAlmostEqual(data["t"], 1.0, places=5)

    def test_edge_queries_and_primary_edge_sampling_work(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0]),
                          cuda.Array3i([0, 0], [1, 2], [2, 3]))
            mesh.build()
            edge_indices = mesh.edge_indices()
            secondary_edges = mesh.secondary_edges()

            front_mesh = pj.Mesh(cuda.Array3f([-0.5, 0.5, 0.0],
                                             [-0.5, -0.5, 0.5],
                                             [3.0, 3.0, 3.0]),
                                cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(front_mesh)
            scene.build()

            camera = pj.Camera(45.0, 1e-4, 1e4)
            camera.width = 32
            camera.height = 32
            camera.build()
            camera.prepare_edges(scene)
            sample = camera.sample_edge(cuda.Float([0.25]))

            print(json.dumps({
                "edge_count": len(list(edge_indices[0])),
                "boundary_count": sum(bool(v) for v in list(secondary_edges.is_boundary)),
                "sample_idx": int(sample.idx[0]),
                "sample_pdf": float(sample.pdf[0]),
                "sample_x_dot_n": float(sample.x_dot_n[0]),
                "sample_x_dot_n_finite": math.isfinite(float(sample.x_dot_n[0]))
            }))
            """
        )

        self.assertEqual(data["edge_count"], 5)
        self.assertEqual(data["boundary_count"], 4)
        self.assertGreaterEqual(data["sample_idx"], 0)
        self.assertGreater(data["sample_pdf"], 0.0)
        self.assertTrue(data["sample_x_dot_n_finite"])

    def test_scene_nearest_edge_point_queries_return_expected_fields_and_batches(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            points = cuda.Array3f([0.25, -0.1],
                                  [-0.2, 0.25],
                                  [0.1, 0.0])
            edges = scene.nearest_edge(points)

            print(json.dumps({
                "type": type(edges).__name__,
                "valid": [bool(v) for v in list(edges.is_valid())],
                "shape_id": [int(v) for v in list(edges.shape_id)],
                "edge_id": [int(v) for v in list(edges.edge_id)],
                "global_edge_id": [int(v) for v in list(edges.global_edge_id)],
                "distance": [float(v) for v in list(edges.distance)],
                "edge_t": [float(v) for v in list(edges.edge_t)],
                "point": [[float(v) for v in list(edges.point[i])] for i in range(3)],
                "edge_point": [[float(v) for v in list(edges.edge_point[i])] for i in range(3)],
                "is_boundary": [bool(v) for v in list(edges.is_boundary)],
            }))
            """
        )

        self.assertEqual(data["type"], "NearestPointEdgeDetached")
        self.assertEqual(data["valid"], [True, True])
        self.assertEqual(data["shape_id"], [0, 0])
        self.assertEqual(data["edge_id"], [0, 1])
        self.assertEqual(data["global_edge_id"], [0, 1])
        self.assertAlmostEqual(data["distance"][0], math.sqrt(0.05), places=5)
        self.assertAlmostEqual(data["distance"][1], 0.1, places=5)
        self.assertAlmostEqual(data["edge_t"][0], 0.25, places=5)
        self.assertAlmostEqual(data["edge_t"][1], 0.25, places=5)
        self.assertAlmostEqual(data["point"][0][0], 0.25, places=5)
        self.assertAlmostEqual(data["point"][1][0], -0.2, places=5)
        self.assertAlmostEqual(data["point"][2][0], 0.1, places=5)
        self.assertAlmostEqual(data["edge_point"][0][0], 0.25, places=5)
        self.assertAlmostEqual(data["edge_point"][1][0], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][2][0], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][0][1], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][1][1], 0.25, places=5)
        self.assertAlmostEqual(data["edge_point"][2][1], 0.0, places=5)
        self.assertEqual(data["is_boundary"], [True, True])

    def test_scene_nearest_edge_ray_queries_use_segment_semantics_and_batches(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            rays = pj.RayDetached(cuda.Array3f([0.25, -0.2],
                                               [0.0, 0.25],
                                               [1.0, 0.0]),
                                  cuda.Array3f([0.0, 1.0],
                                               [0.0, 0.0],
                                               [-1.0, 0.0]))
            rays.tmax = cuda.Float([0.5, 1.0])
            edges = scene.nearest_edge(rays)

            print(json.dumps({
                "type": type(edges).__name__,
                "valid": [bool(v) for v in list(edges.is_valid())],
                "shape_id": [int(v) for v in list(edges.shape_id)],
                "edge_id": [int(v) for v in list(edges.edge_id)],
                "global_edge_id": [int(v) for v in list(edges.global_edge_id)],
                "distance": [float(v) for v in list(edges.distance)],
                "ray_t": [float(v) for v in list(edges.ray_t)],
                "edge_t": [float(v) for v in list(edges.edge_t)],
                "point": [[float(v) for v in list(edges.point[i])] for i in range(3)],
                "edge_point": [[float(v) for v in list(edges.edge_point[i])] for i in range(3)],
            }))
            """
        )

        self.assertEqual(data["type"], "NearestRayEdgeDetached")
        self.assertEqual(data["valid"], [True, True])
        self.assertEqual(data["shape_id"], [0, 0])
        self.assertEqual(data["edge_id"], [0, 1])
        self.assertEqual(data["global_edge_id"], [0, 1])
        self.assertAlmostEqual(data["distance"][0], 0.5, places=5)
        self.assertAlmostEqual(data["distance"][1], 0.0, places=6)
        self.assertAlmostEqual(data["ray_t"][0], 0.5, places=5)
        self.assertAlmostEqual(data["ray_t"][1], 0.2, places=5)
        self.assertAlmostEqual(data["edge_t"][0], 0.25, places=5)
        self.assertAlmostEqual(data["edge_t"][1], 0.25, places=5)
        self.assertAlmostEqual(data["point"][0][0], 0.25, places=5)
        self.assertAlmostEqual(data["point"][1][0], 0.0, places=5)
        self.assertAlmostEqual(data["point"][2][0], 0.5, places=5)
        self.assertAlmostEqual(data["edge_point"][0][0], 0.25, places=5)
        self.assertAlmostEqual(data["edge_point"][1][0], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][2][0], 0.0, places=5)
        self.assertAlmostEqual(data["point"][0][1], 0.0, places=5)
        self.assertAlmostEqual(data["point"][1][1], 0.25, places=5)
        self.assertAlmostEqual(data["point"][2][1], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][0][1], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][1][1], 0.25, places=5)
        self.assertAlmostEqual(data["edge_point"][2][1], 0.0, places=5)

    def test_scene_nearest_edge_multi_mesh_maps_shape_and_global_ids(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh_a = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                            cuda.Array3i([0], [1], [2]))

            mesh_b = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                            cuda.Array3i([0], [1], [2]))
            mesh_b.vertex_positions = cuda.Array3f([2.0, 3.0, 2.0],
                                                   [0.0, 0.0, 1.0],
                                                   [0.0, 0.0, 0.0])

            scene = pj.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            edge = scene.nearest_edge(cuda.Array3f([2.2], [0.2], [0.3]))
            offsets = scene.mesh_edge_offsets()
            print(json.dumps({
                "valid": bool(edge.is_valid()[0]),
                "shape_id": int(edge.shape_id[0]),
                "edge_id": int(edge.edge_id[0]),
                "global_edge_id": int(edge.global_edge_id[0]),
                "mesh_edge_offset": int(offsets[1]),
                "distance": float(edge.distance[0]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["shape_id"], 1)
        self.assertEqual(data["edge_id"], 0)
        self.assertEqual(data["global_edge_id"], data["mesh_edge_offset"] + data["edge_id"])
        self.assertAlmostEqual(data["distance"], math.sqrt(0.13), places=5)

    def test_scene_nearest_edge_dynamic_updates_require_commit_and_refit(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            scene.update_mesh_vertices(mesh_id,
                                       cuda.Array3f([2.0, 3.0, 2.0],
                                                    [0.0, 0.0, 1.0],
                                                    [0.0, 0.0, 0.0]))

            pending_error = False
            try:
                scene.nearest_edge(cuda.Array3f([2.25], [-0.2], [0.1]))
            except Exception as e:
                pending_error = "pending updates" in str(e)

            scene.sync()
            edge = scene.nearest_edge(cuda.Array3f([2.25], [-0.2], [0.1]))

            print(json.dumps({
                "pending_error": pending_error,
                "valid": bool(edge.is_valid()[0]),
                "shape_id": int(edge.shape_id[0]),
                "edge_id": int(edge.edge_id[0]),
                "global_edge_id": int(edge.global_edge_id[0]),
                "distance": float(edge.distance[0]),
                "edge_t": float(edge.edge_t[0]),
                "edge_point": [float(edge.edge_point[0][0]), float(edge.edge_point[1][0]), float(edge.edge_point[2][0])],
                "pending_after_commit": bool(scene.has_pending_updates()),
            }))
            """
        )

        self.assertTrue(data["pending_error"])
        self.assertTrue(data["valid"])
        self.assertEqual(data["shape_id"], 0)
        self.assertEqual(data["edge_id"], 0)
        self.assertEqual(data["global_edge_id"], 0)
        self.assertAlmostEqual(data["distance"], math.sqrt(0.05), places=5)
        self.assertAlmostEqual(data["edge_t"], 0.25, places=5)
        self.assertAlmostEqual(data["edge_point"][0], 2.25, places=5)
        self.assertAlmostEqual(data["edge_point"][1], 0.0, places=5)
        self.assertAlmostEqual(data["edge_point"][2], 0.0, places=5)
        self.assertFalse(data["pending_after_commit"])

    def test_scene_edge_mask_requires_build_filters_queries_and_preserves_metadata(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh)

            prebuild_error = False
            try:
                scene.set_edge_mask(cuda.Bool([True, True, True]))
            except Exception as exc:
                prebuild_error = "not built" in str(exc)

            scene.build()
            default_mask = [bool(v) for v in list(scene.edge_mask())]
            edge_info_before = scene.edge_info()
            version_before = int(scene.version)
            edge_version_before = int(scene.edge_version)

            wrong_size_error = False
            try:
                scene.set_edge_mask(cuda.Bool([True, True]))
            except Exception as exc:
                wrong_size_error = "mask size must match" in str(exc)
            scene.set_edge_mask(cuda.Bool([True, True, True]))
            pending_after_same_mask = bool(scene.has_pending_updates())

            scene.set_edge_mask(cuda.Bool([False, True, False]))
            pending_after_mask_change = bool(scene.has_pending_updates())
            pending_query_error = False
            try:
                scene.nearest_edge(cuda.Array3f([0.25], [-0.2], [0.1]))
            except Exception as exc:
                pending_query_error = "pending updates" in str(exc)
            scene.sync()
            version_after_mask_sync = int(scene.version)
            edge_version_after_mask_sync = int(scene.edge_version)

            masked_edge = scene.nearest_edge(cuda.Array3f([0.25], [-0.2], [0.1]))
            scene.set_edge_mask(cuda.Bool([False, False, False]))
            scene.sync()
            invalid_edge = scene.nearest_edge(cuda.Array3f([0.25], [-0.2], [0.1]))
            edge_info_after = scene.edge_info()

            print(json.dumps({
                "prebuild_error": prebuild_error,
                "wrong_size_error": wrong_size_error,
                "default_mask": default_mask,
                "pending_after_same_mask": pending_after_same_mask,
                "pending_after_mask_change": pending_after_mask_change,
                "pending_query_error": pending_query_error,
                "version_before": version_before,
                "version_after_mask_sync": version_after_mask_sync,
                "edge_version_before": edge_version_before,
                "edge_version_after_mask_sync": edge_version_after_mask_sync,
                "masked_valid": bool(masked_edge.is_valid()[0]),
                "masked_edge_id": int(masked_edge.edge_id[0]),
                "masked_global_edge_id": int(masked_edge.global_edge_id[0]),
                "invalid_valid": bool(invalid_edge.is_valid()[0]),
                "invalid_global_edge_id": int(invalid_edge.global_edge_id[0]),
                "edge_info_before": [int(v) for v in list(edge_info_before.global_edge_id)],
                "edge_info_after": [int(v) for v in list(edge_info_after.global_edge_id)],
            }))
            """
        )

        self.assertTrue(data["prebuild_error"])
        self.assertTrue(data["wrong_size_error"])
        self.assertEqual(data["default_mask"], [True, True, True])
        self.assertFalse(data["pending_after_same_mask"])
        self.assertTrue(data["pending_after_mask_change"])
        self.assertTrue(data["pending_query_error"])
        self.assertEqual(data["version_after_mask_sync"], data["version_before"])
        self.assertEqual(data["edge_version_after_mask_sync"], data["edge_version_before"] + 1)
        self.assertTrue(data["masked_valid"])
        self.assertEqual(data["masked_edge_id"], 1)
        self.assertEqual(data["masked_global_edge_id"], 1)
        self.assertFalse(data["invalid_valid"])
        self.assertEqual(data["invalid_global_edge_id"], -1)
        self.assertEqual(data["edge_info_before"], [0, 1, 2])
        self.assertEqual(data["edge_info_after"], [0, 1, 2])

    def test_scene_edge_ploc_gpu_treelet_mode_builds_and_queries(self):
        data = run_json_case(
            """
            import json
            import os

            os.environ["RAYD_EDGE_BVH_BUILD_ALGORITHM"] = "ploc"
            os.environ["RAYD_EDGE_BVH_POST_BUILD_STRATEGY"] = "gpu_treelet"
            os.environ["RAYD_EDGE_BVH_COMPACTION_MODE"] = "gpu_emit"

            import rayd as pj
            import drjit.cuda as cuda
            from tests.benchmark_support import _make_grid_mesh_data

            mesh_data = _make_grid_mesh_data(192)
            mesh = pj.Mesh(cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
                           cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            edge = scene.nearest_edge(cuda.Array3f([0.5], [0.5], [0.1]))

            print(json.dumps({
                "valid": bool(edge.is_valid()[0]),
                "global_edge_id": int(edge.global_edge_id[0]),
                "distance": float(edge.distance[0]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertGreaterEqual(data["global_edge_id"], 0)
        self.assertGreater(data["distance"], 0.0)

    def test_scene_edge_ploc_stats_report_gpu_collapsed_leaves(self):
        data = run_json_case(
            """
            import json
            import os

            os.environ["RAYD_EDGE_BVH_BUILD_ALGORITHM"] = "ploc"
            os.environ["RAYD_EDGE_BVH_POST_BUILD_STRATEGY"] = "none"
            os.environ["RAYD_EDGE_BVH_COMPACTION_MODE"] = "gpu_emit"

            import rayd as pj
            import drjit.cuda as cuda
            from tests.benchmark_support import _make_grid_mesh_data

            mesh_data = _make_grid_mesh_data(32)
            mesh = pj.Mesh(cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
                           cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            stats = scene.edge_bvh_stats()

            print(json.dumps({
                "primitive_count": int(stats.primitive_count),
                "node_count": int(stats.node_count),
                "max_leaf_size": int(stats.max_leaf_size),
                "leaf_histogram": [int(v) for v in list(stats.leaf_size_histogram)],
            }))
            """
        )

        raw_node_count = 2 * data["primitive_count"] - 1
        multi_primitive_leaves = sum(data["leaf_histogram"][2:]) if len(data["leaf_histogram"]) > 2 else 0

        self.assertGreater(data["primitive_count"], 0)
        self.assertGreater(data["node_count"], 0)
        self.assertLess(data["node_count"], raw_node_count)
        self.assertLessEqual(data["max_leaf_size"], 4)
        self.assertGreater(multi_primitive_leaves, 0)

    def test_scene_edge_bvh_stats_report_compacted_tree_metrics(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda
            from tests.benchmark_support import _make_grid_mesh_data

            mesh_data = _make_grid_mesh_data(8)
            mesh = pj.Mesh(cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
                           cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            stats = scene.edge_bvh_stats()

            print(json.dumps({
                "primitive_count": int(stats.primitive_count),
                "node_count": int(stats.node_count),
                "internal_node_count": int(stats.internal_node_count),
                "leaf_node_count": int(stats.leaf_node_count),
                "max_height": int(stats.max_height),
                "refit_level_count": int(stats.refit_level_count),
                "leaf_histogram": [int(v) for v in list(stats.leaf_size_histogram)],
                "hist_leaf_count": int(sum(stats.leaf_size_histogram)),
                "hist_primitive_count": int(sum(i * v for i, v in enumerate(stats.leaf_size_histogram))),
                "normalized_overlap": float(stats.normalized_sibling_overlap),
            }))
            """
        )

        self.assertGreater(data["primitive_count"], 0)
        self.assertGreater(data["node_count"], 0)
        self.assertGreater(data["internal_node_count"], 0)
        self.assertGreater(data["leaf_node_count"], 0)
        self.assertGreaterEqual(data["max_height"], 1)
        self.assertGreaterEqual(data["refit_level_count"], 1)
        self.assertEqual(data["hist_leaf_count"], data["leaf_node_count"])
        self.assertEqual(data["hist_primitive_count"], data["primitive_count"])
        self.assertGreaterEqual(data["normalized_overlap"], 0.0)

    def test_scene_edge_masked_refit_updates_active_subset(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()
            scene.set_edge_mask(cuda.Bool([False, True, False]))
            scene.sync()

            scene.update_mesh_vertices(mesh_id,
                                       cuda.Array3f([2.0, 3.0, 2.0],
                                                    [0.0, 0.0, 1.0],
                                                    [0.0, 0.0, 0.0]))
            scene.sync()
            edge = scene.nearest_edge(cuda.Array3f([1.8], [0.25], [0.1]))

            print(json.dumps({
                "valid": bool(edge.is_valid()[0]),
                "shape_id": int(edge.shape_id[0]),
                "edge_id": int(edge.edge_id[0]),
                "global_edge_id": int(edge.global_edge_id[0]),
                "distance": float(edge.distance[0]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["shape_id"], 0)
        self.assertEqual(data["edge_id"], 1)
        self.assertEqual(data["global_edge_id"], 1)
        self.assertAlmostEqual(data["distance"], math.sqrt(0.05), places=5)

    def test_scene_edge_metadata_and_topology_are_exposed(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh_a = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0, 1.0],
                                         [0.0, 0.0, 0.0, 0.0]),
                             cuda.Array3i([0, 0], [1, 2], [2, 3]))
            mesh_b = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                             cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            edge_info = scene.edge_info()
            topology = scene.edge_topology()
            tri0_edges = scene.triangle_edge_indices(cuda.Int([0]))
            tri1_edges = scene.triangle_edge_indices(cuda.Int([1]))
            tri2_edges_local = scene.triangle_edge_indices(cuda.Int([2]), False)
            adj_faces = scene.edge_adjacent_faces(cuda.Int([1]))
            adj_faces_local = scene.edge_adjacent_faces(cuda.Int([1]), False)

            print(json.dumps({
                "version": int(scene.version),
                "edge_version": int(scene.edge_version),
                "edge_count": edge_info.size(),
                "topology_count": topology.size(),
                "shape_ids": [int(v) for v in list(edge_info.shape_id)],
                "local_edge_ids": [int(v) for v in list(edge_info.local_edge_id)],
                "global_edge_ids": [int(v) for v in list(edge_info.global_edge_id)],
                "length_edge0": float(edge_info.length[0]),
                "length_edge1": float(edge_info.length[1]),
                "edge1_start": [float(edge_info.start[0][1]), float(edge_info.start[1][1]), float(edge_info.start[2][1])],
                "edge1_end": [float(edge_info.end[0][1]), float(edge_info.end[1][1]), float(edge_info.end[2][1])],
                "edge1_boundary": bool(edge_info.is_boundary[1]),
                "face_offsets": [int(v) for v in list(scene.mesh_face_offsets())],
                "edge_offsets": [int(v) for v in list(scene.mesh_edge_offsets())],
                "vertex_offsets": [int(v) for v in list(scene.mesh_vertex_offsets())],
                "topology_edge1": {
                    "v0": int(topology.v0[1]),
                    "v1": int(topology.v1[1]),
                    "v0_global": int(topology.v0_global[1]),
                    "v1_global": int(topology.v1_global[1]),
                    "face0_global": int(topology.face0_global[1]),
                    "face1_global": int(topology.face1_global[1]),
                    "opposite0": int(topology.opposite_vertex0[1]),
                    "opposite1": int(topology.opposite_vertex1[1]),
                    "opposite0_global": int(topology.opposite_vertex0_global[1]),
                    "opposite1_global": int(topology.opposite_vertex1_global[1]),
                },
                "tri0_edges": [int(tri0_edges[i][0]) for i in range(3)],
                "tri1_edges": [int(tri1_edges[i][0]) for i in range(3)],
                "tri2_edges_local": [int(tri2_edges_local[i][0]) for i in range(3)],
                "adj_faces": [int(adj_faces[i][0]) for i in range(2)],
                "adj_faces_local": [int(adj_faces_local[i][0]) for i in range(2)],
            }))
            """
        )

        self.assertEqual(data["version"], 1)
        self.assertEqual(data["edge_version"], 1)
        self.assertEqual(data["edge_count"], 8)
        self.assertEqual(data["topology_count"], 8)
        self.assertEqual(data["shape_ids"], [0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(data["local_edge_ids"], [0, 1, 2, 3, 4, 0, 1, 2])
        self.assertEqual(data["global_edge_ids"], list(range(8)))
        self.assertAlmostEqual(data["length_edge0"], 1.0, places=5)
        self.assertAlmostEqual(data["length_edge1"], math.sqrt(2.0), places=5)
        self.assertEqual(data["edge1_start"], [0.0, 0.0, 0.0])
        self.assertEqual(data["edge1_end"], [1.0, 1.0, 0.0])
        self.assertFalse(data["edge1_boundary"])
        self.assertEqual(data["face_offsets"], [0, 2, 3])
        self.assertEqual(data["edge_offsets"], [0, 5, 8])
        self.assertEqual(data["vertex_offsets"], [0, 4, 7])
        self.assertEqual(data["topology_edge1"], {
            "v0": 0,
            "v1": 2,
            "v0_global": 0,
            "v1_global": 2,
            "face0_global": 0,
            "face1_global": 1,
            "opposite0": 1,
            "opposite1": 3,
            "opposite0_global": 1,
            "opposite1_global": 3,
        })
        self.assertEqual(data["tri0_edges"], [0, 3, 1])
        self.assertEqual(data["tri1_edges"], [1, 4, 2])
        self.assertEqual(data["tri2_edges_local"], [0, 2, 1])
        self.assertEqual(data["adj_faces"], [0, 1])
        self.assertEqual(data["adj_faces_local"], [0, 1])

    def test_scene_global_geometry_exposes_scene_global_vertices_faces_and_metadata(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh_a = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0, 1.0],
                                         [0.0, 0.0, 0.0, 0.0]),
                             cuda.Array3i([0, 0], [1, 2], [2, 3]))
            mesh_b = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                             cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            geometry = scene.global_geometry()

            print(json.dumps({
                "vertex_count": int(geometry.vertex_count()),
                "face_count": int(geometry.face_count()),
                "vertices": [
                    [float(geometry.vertices[0][i]), float(geometry.vertices[1][i]), float(geometry.vertices[2][i])]
                    for i in range(geometry.vertex_count())
                ],
                "faces": [
                    [int(geometry.faces[0][i]), int(geometry.faces[1][i]), int(geometry.faces[2][i])]
                    for i in range(geometry.face_count())
                ],
                "face_normal": [
                    [float(geometry.face_normal[0][i]), float(geometry.face_normal[1][i]), float(geometry.face_normal[2][i])]
                    for i in range(geometry.face_count())
                ],
                "shape_id": [int(v) for v in list(geometry.shape_id)],
                "local_prim_id": [int(v) for v in list(geometry.local_prim_id)],
                "global_prim_id": [int(v) for v in list(geometry.global_prim_id)],
            }))
            """
        )

        self.assertEqual(data["vertex_count"], 7)
        self.assertEqual(data["face_count"], 3)
        self.assertEqual(data["vertices"], [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ])
        self.assertEqual(data["faces"], [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
        ])
        self.assertEqual(data["shape_id"], [0, 0, 1])
        self.assertEqual(data["local_prim_id"], [0, 1, 0])
        self.assertEqual(data["global_prim_id"], [0, 1, 2])
        for normal in data["face_normal"]:
            self.assertAlmostEqual(normal[0], 0.0, places=5)
            self.assertAlmostEqual(normal[1], 0.0, places=5)
            self.assertAlmostEqual(normal[2], 1.0, places=5)

    def test_scene_edge_version_and_commit_profile_track_dynamic_updates(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            version_before = int(scene.version)
            edge_version_before = int(scene.edge_version)
            pending_edge_info_error = False

            scene.update_mesh_vertices(mesh_id,
                                       cuda.Array3f([2.0, 3.0, 2.0],
                                                    [0.0, 0.0, 1.0],
                                                    [0.0, 0.0, 0.0]))
            try:
                scene.edge_info()
            except Exception as exc:
                pending_edge_info_error = "pending updates" in str(exc)

            scene.sync()
            edge_info = scene.edge_info()
            profile = scene.last_sync_profile

            print(json.dumps({
                "pending_edge_info_error": pending_edge_info_error,
                "version_before": version_before,
                "version_after": int(scene.version),
                "edge_version_before": edge_version_before,
                "edge_version_after": int(scene.edge_version),
                "edge0_start_after": [float(edge_info.start[0][0]), float(edge_info.start[1][0]), float(edge_info.start[2][0])],
                "profile_edge_scatter_ms": float(profile.edge_scatter_ms),
                "profile_edge_refit_ms": float(profile.edge_refit_ms),
                "profile_updated_edge_meshes": int(profile.updated_edge_meshes),
                "profile_updated_edges": int(profile.updated_edges),
            }))
            """
        )

        self.assertTrue(data["pending_edge_info_error"])
        self.assertEqual(data["version_after"], data["version_before"] + 1)
        self.assertEqual(data["edge_version_after"], data["edge_version_before"] + 1)
        self.assertEqual(data["edge0_start_after"], [2.0, 0.0, 0.0])
        self.assertGreaterEqual(data["profile_edge_scatter_ms"], 0.0)
        self.assertGreaterEqual(data["profile_edge_refit_ms"], 0.0)
        self.assertEqual(data["profile_updated_edge_meshes"], 1)
        self.assertEqual(data["profile_updated_edges"], 3)

    def test_scene_edge_interfaces_handle_edges_disabled_and_empty_edge_sets(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            mesh.edges_enabled = False

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            edge_info = scene.edge_info()
            topology = scene.edge_topology()
            tri_edges = scene.triangle_edge_indices(cuda.Int([0]))
            adj_faces = scene.edge_adjacent_faces(cuda.Int([0]))
            nearest = scene.nearest_edge(cuda.Array3f([0.2], [0.1], [0.3]))

            print(json.dumps({
                "edge_count": edge_info.size(),
                "topology_count": topology.size(),
                "version": int(scene.version),
                "edge_version": int(scene.edge_version),
                "edge_offsets": [int(v) for v in list(scene.mesh_edge_offsets())],
                "tri_edges": [int(tri_edges[i][0]) for i in range(3)],
                "adj_faces": [int(adj_faces[i][0]) for i in range(2)],
                "nearest_valid": bool(nearest.is_valid()[0]),
                "nearest_shape": int(nearest.shape_id[0]),
                "nearest_edge": int(nearest.edge_id[0]),
                "nearest_distance_inf": math.isinf(float(nearest.distance[0])),
            }))
            """
        )

        self.assertEqual(data["edge_count"], 0)
        self.assertEqual(data["topology_count"], 0)
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["edge_version"], 1)
        self.assertEqual(data["edge_offsets"], [0, 0])
        self.assertEqual(data["tri_edges"], [-1, -1, -1])
        self.assertEqual(data["adj_faces"], [-1, -1])
        self.assertFalse(data["nearest_valid"])
        self.assertEqual(data["nearest_shape"], -1)
        self.assertEqual(data["nearest_edge"], -1)
        self.assertTrue(data["nearest_distance_inf"])

    def test_scene_edge_index_queries_support_batched_valid_and_invalid_ids(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh_a = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0, 1.0],
                                         [0.0, 0.0, 0.0, 0.0]),
                             cuda.Array3i([0, 0], [1, 2], [2, 3]))
            mesh_b = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                         [0.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0]),
                             cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            prim_ids = cuda.Int([-1, 0, 1, 2, 9])
            edge_ids = cuda.Int([-1, 1, 6, 99])
            tri_edges_global = scene.triangle_edge_indices(prim_ids)
            tri_edges_local = scene.triangle_edge_indices(prim_ids, False)
            adj_faces_global = scene.edge_adjacent_faces(edge_ids)
            adj_faces_local = scene.edge_adjacent_faces(edge_ids, False)

            print(json.dumps({
                "tri_edges_global": [[int(tri_edges_global[i][j]) for j in range(5)] for i in range(3)],
                "tri_edges_local": [[int(tri_edges_local[i][j]) for j in range(5)] for i in range(3)],
                "adj_faces_global": [[int(adj_faces_global[i][j]) for j in range(4)] for i in range(2)],
                "adj_faces_local": [[int(adj_faces_local[i][j]) for j in range(4)] for i in range(2)],
            }))
            """
        )

        self.assertEqual(data["tri_edges_global"], [
            [-1, 0, 1, 5, -1],
            [-1, 3, 4, 7, -1],
            [-1, 1, 2, 6, -1],
        ])
        self.assertEqual(data["tri_edges_local"], [
            [-1, 0, 1, 0, -1],
            [-1, 3, 4, 2, -1],
            [-1, 1, 2, 1, -1],
        ])
        self.assertEqual(data["adj_faces_global"], [
            [-1, 0, 2, -1],
            [-1, 1, -1, -1],
        ])
        self.assertEqual(data["adj_faces_local"], [
            [-1, 0, 0, -1],
            [-1, 1, -1, -1],
        ])

    def test_scene_edge_topology_stays_stable_across_pending_and_transform_updates(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0]),
                          cuda.Array3i([0, 0], [1, 2], [2, 3]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            version_before = int(scene.version)
            edge_version_before = int(scene.edge_version)
            tri_edges_before = scene.triangle_edge_indices(cuda.Int([0, 1]))
            topology_before = scene.edge_topology()

            translate = cuda.Matrix4f([[1.0, 0.0, 0.0, 3.0],
                                       [0.0, 1.0, 0.0, 0.5],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 1.0]])
            scene.set_mesh_transform(mesh_id, translate)

            topology_pending = scene.edge_topology()
            tri_edges_pending = scene.triangle_edge_indices(cuda.Int([0, 1]))
            pending_edge_info_error = False
            try:
                scene.edge_info()
            except Exception as exc:
                pending_edge_info_error = "pending updates" in str(exc)

            scene.sync()
            edge_info_after = scene.edge_info()
            topology_after = scene.edge_topology()
            tri_edges_after = scene.triangle_edge_indices(cuda.Int([0, 1]))
            profile = scene.last_sync_profile
            profile_updated_transform_meshes = int(profile.updated_transform_meshes)
            profile_updated_edge_meshes = int(profile.updated_edge_meshes)
            profile_updated_edges = int(profile.updated_edges)
            version_after = int(scene.version)
            edge_version_after = int(scene.edge_version)

            scene.sync()
            version_after_noop = int(scene.version)
            edge_version_after_noop = int(scene.edge_version)
            noop_profile = scene.last_sync_profile

            print(json.dumps({
                "pending_edge_info_error": pending_edge_info_error,
                "version_before": version_before,
                "version_after": version_after,
                "edge_version_before": edge_version_before,
                "edge_version_after": edge_version_after,
                "version_after_noop": version_after_noop,
                "edge_version_after_noop": edge_version_after_noop,
                "tri_edges_before": [[int(tri_edges_before[i][j]) for j in range(2)] for i in range(3)],
                "tri_edges_pending": [[int(tri_edges_pending[i][j]) for j in range(2)] for i in range(3)],
                "tri_edges_after": [[int(tri_edges_after[i][j]) for j in range(2)] for i in range(3)],
                "topology_before_face1": int(topology_before.face1_global[1]),
                "topology_pending_face1": int(topology_pending.face1_global[1]),
                "topology_after_face1": int(topology_after.face1_global[1]),
                "edge0_start_after": [float(edge_info_after.start[0][0]), float(edge_info_after.start[1][0]), float(edge_info_after.start[2][0])],
                "profile_updated_transform_meshes": profile_updated_transform_meshes,
                "profile_updated_edge_meshes": profile_updated_edge_meshes,
                "profile_updated_edges": profile_updated_edges,
                "noop_profile_total_ms": float(noop_profile.total_ms),
                "noop_profile_updated_edges": int(noop_profile.updated_edges),
            }))
            """
        )

        self.assertTrue(data["pending_edge_info_error"])
        self.assertEqual(data["tri_edges_before"], data["tri_edges_pending"])
        self.assertEqual(data["tri_edges_before"], data["tri_edges_after"])
        self.assertEqual(data["topology_before_face1"], 1)
        self.assertEqual(data["topology_pending_face1"], 1)
        self.assertEqual(data["topology_after_face1"], 1)
        self.assertEqual(data["edge0_start_after"], [3.0, 0.5, 1.0])
        self.assertEqual(data["version_after"], data["version_before"] + 1)
        self.assertEqual(data["edge_version_after"], data["edge_version_before"] + 1)
        self.assertEqual(data["version_after_noop"], data["version_after"])
        self.assertEqual(data["edge_version_after_noop"], data["edge_version_after"])
        self.assertEqual(data["profile_updated_transform_meshes"], 1)
        self.assertEqual(data["profile_updated_edge_meshes"], 1)
        self.assertEqual(data["profile_updated_edges"], 5)
        self.assertEqual(data["noop_profile_total_ms"], 0.0)
        self.assertEqual(data["noop_profile_updated_edges"], 0)

    def test_scene_global_geometry_preserves_gradients_for_ad_vertices(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            verts = ad.Array3f([0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0])
            dr.enable_grad(verts)

            mesh = pj.Mesh(verts, cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            geometry = scene.global_geometry()
            dr.backward(dr.sum(geometry.vertices[2]))
            grad = dr.grad(verts)

            print(json.dumps({
                "vertex_count": int(geometry.vertex_count()),
                "face_count": int(geometry.face_count()),
                "grad_z": [float(grad[2][i]) for i in range(3)],
                "grad_z_sum": float(dr.sum(grad[2])[0]),
            }))
            """
        )

        self.assertEqual(data["vertex_count"], 3)
        self.assertEqual(data["face_count"], 1)
        self.assertEqual(data["grad_z"], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(data["grad_z_sum"], 3.0, places=5)

    def test_scene_nearest_edge_gradients_flow_for_point_and_ray_queries(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            point_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [0.0, 0.0, 0.0]),
                                cuda.Array3i([0], [1], [2]))
            point_verts = ad.Array3f([0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0],
                                     [0.0, 0.0, 0.0])
            point_query = ad.Array3f([0.2], [0.1], [0.3])
            dr.enable_grad(point_verts)
            dr.enable_grad(point_query)
            point_mesh.vertex_positions = point_verts

            point_scene = pj.Scene()
            point_scene.add_mesh(point_mesh)
            point_scene.build()
            point_edge = point_scene.nearest_edge(point_query)
            dr.backward(point_edge.distance)
            point_grad = dr.grad(point_query)
            point_vert_grad = dr.grad(point_verts)

            ray_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0],
                                           [0.0, 0.0, 0.0]),
                              cuda.Array3i([0], [1], [2]))
            tz = ad.Float([0.0])
            ray_origin = ad.Array3f([0.2], [0.4], [1.0])
            dr.enable_grad(tz)
            dr.enable_grad(ray_origin)
            ray_mesh.to_world_left = ad.Matrix4f([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ])

            ray_scene = pj.Scene()
            ray_scene.add_mesh(ray_mesh)
            ray_scene.build()
            ray = pj.Ray(ray_origin, ad.Array3f([0.0], [0.0], [-1.0]))
            ray.tmax = ad.Float([2.0])
            ray_edge = ray_scene.nearest_edge(ray)
            dr.backward(ray_edge.ray_t)
            ray_origin_grad = dr.grad(ray_origin)

            print(json.dumps({
                "point_result_type": type(point_edge).__name__,
                "ray_result_type": type(ray_edge).__name__,
                "point_valid": bool(point_edge.is_valid()[0]),
                "point_edge_id": int(point_edge.edge_id[0]),
                "point_distance": float(point_edge.distance[0]),
                "point_grad": [float(point_grad[0][0]), float(point_grad[1][0]), float(point_grad[2][0])],
                "point_vert_grad_z_sum": float(point_vert_grad[2][0] + point_vert_grad[2][1] + point_vert_grad[2][2]),
                "ray_valid": bool(ray_edge.is_valid()[0]),
                "ray_edge_id": int(ray_edge.edge_id[0]),
                "ray_t": float(ray_edge.ray_t[0]),
                "ray_origin_grad_z": float(ray_origin_grad[2][0]),
                "ray_tz_grad": float(dr.grad(tz)[0]),
            }))
            """
        )

        self.assertEqual(data["point_result_type"], "NearestPointEdge")
        self.assertEqual(data["ray_result_type"], "NearestRayEdge")
        self.assertTrue(data["point_valid"])
        self.assertEqual(data["point_edge_id"], 0)
        self.assertAlmostEqual(data["point_distance"], math.sqrt(0.1), places=5)
        self.assertAlmostEqual(data["point_grad"][0], 0.0, places=5)
        self.assertGreater(data["point_grad"][1], 0.0)
        self.assertGreater(data["point_grad"][2], 0.0)
        self.assertLess(data["point_vert_grad_z_sum"], 0.0)
        self.assertTrue(data["ray_valid"])
        self.assertEqual(data["ray_edge_id"], 1)
        self.assertAlmostEqual(data["ray_t"], 1.0, places=5)
        self.assertAlmostEqual(data["ray_origin_grad_z"], 1.0, places=5)
        self.assertAlmostEqual(data["ray_tz_grad"], -1.0, places=5)

    def test_scene_nearest_edge_invalid_inputs_return_sentinels(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            zero_tmax_ray = pj.RayDetached(cuda.Array3f([0.0], [0.0], [0.0]),
                                           cuda.Array3f([0.0], [0.0], [1.0]))
            zero_tmax_ray.tmax = cuda.Float([0.0])

            cases = {
                "nan_point": scene.nearest_edge(cuda.Array3f([float("nan")], [0.0], [0.0])),
                "inf_point": scene.nearest_edge(cuda.Array3f([float("inf")], [0.0], [0.0])),
                "zero_dir_ray": scene.nearest_edge(
                    pj.RayDetached(cuda.Array3f([0.0], [0.0], [0.0]),
                                   cuda.Array3f([0.0], [0.0], [0.0]))
                ),
                "zero_tmax_ray": scene.nearest_edge(zero_tmax_ray),
            }

            summary = {}
            for key, edge in cases.items():
                summary[key] = {
                    "valid": bool(edge.is_valid()[0]),
                    "shape_id": int(edge.shape_id[0]),
                    "edge_id": int(edge.edge_id[0]),
                    "distance_is_inf": math.isinf(float(edge.distance[0])),
                }

            print(json.dumps(summary))
            """
        )

        for case in data.values():
            self.assertFalse(case["valid"])
            self.assertEqual(case["shape_id"], -1)
            self.assertEqual(case["edge_id"], -1)
            self.assertTrue(case["distance_is_inf"])

    def test_precondition_failures_raise_clean_exceptions(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            result = {}

            scene = pj.Scene()
            try:
                scene.build()
            except Exception as e:
                result["empty_scene_build"] = "missing meshes" in str(e)

            try:
                ray = pj.RayDetached(cuda.Array3f([0.0], [0.0], [0.0]),
                              cuda.Array3f([0.0], [0.0], [1.0]))
                scene.intersect(ray)
            except Exception as e:
                result["unbuilt_scene_intersect"] = "not built" in str(e)

            camera = pj.Camera(45.0, 1e-4, 1e4)
            camera.width = 0
            camera.height = 32
            try:
                camera.build()
            except Exception as e:
                result["camera_resolution"] = "width and height must be positive" in str(e)

            mesh = pj.Mesh(cuda.Array3f([0.0], [0.0], [0.0]),
                          cuda.Array3i([], [], []))
            zero_face_scene = pj.Scene()
            zero_face_scene.add_mesh(mesh)
            try:
                zero_face_scene.build()
            except Exception as e:
                result["zero_face_scene"] = "no faces" in str(e)

            print(json.dumps(result))
            """
        )

        self.assertTrue(data["empty_scene_build"])
        self.assertTrue(data["unbuilt_scene_intersect"])
        self.assertTrue(data["camera_resolution"])
        self.assertTrue(data["zero_face_scene"])

    def test_invalid_mesh_inputs_raise_clean_exceptions(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            result = {}

            try:
                pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0],
                                     [0.0, 0.0, 0.0]),
                        cuda.Array3i([0], [1], [2]),
                        cuda.Array2f([0.0, 1.0],
                                     [0.0, 0.0]))
            except Exception as e:
                result["uv_vertex_count"] = "UV count must match vertex count" in str(e)

            try:
                pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0, 1.0],
                                     [0.0, 0.0, 1.0, 1.0],
                                     [0.0, 0.0, 0.0, 0.0]),
                        cuda.Array3i([0, 0], [1, 2], [2, 3]),
                        cuda.Array2f([0.0, 1.0, 0.0, 1.0],
                                     [0.0, 0.0, 1.0, 1.0]),
                        cuda.Array3i([0], [1], [2]))
            except Exception as e:
                result["face_uv_count"] = "face_uv_indices must match the number of faces" in str(e)

            print(json.dumps(result))
            """
        )

        self.assertTrue(data["uv_vertex_count"])
        self.assertTrue(data["face_uv_count"])

    def test_nonfinite_and_invalid_rays_return_miss_without_crashing(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            def summarize(ray):
                its = scene.intersect(ray)
                return {
                    "valid": bool(its.is_valid()[0]),
                    "shape": int(its.shape_id[0]),
                    "prim": int(its.prim_id[0]),
                    "t_inf": math.isinf(float(its.t[0])),
                }

            nan_tmax = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                               cuda.Array3f([0.0], [0.0], [1.0]))
            nan_tmax.tmax = cuda.Float([float("nan")])

            zero_tmax = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                cuda.Array3f([0.0], [0.0], [1.0]))
            zero_tmax.tmax = cuda.Float([0.0])

            cases = {
                "nan_origin": summarize(pj.RayDetached(cuda.Array3f([float("nan")], [0.25], [-1.0]),
                                                cuda.Array3f([0.0], [0.0], [1.0]))),
                "nan_dir": summarize(pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                             cuda.Array3f([0.0], [float("nan")], [1.0]))),
                "inf_origin": summarize(pj.RayDetached(cuda.Array3f([float("inf")], [0.25], [-1.0]),
                                                cuda.Array3f([0.0], [0.0], [1.0]))),
                "inf_dir": summarize(pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                             cuda.Array3f([0.0], [float("inf")], [1.0]))),
                "zero_dir": summarize(pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                              cuda.Array3f([0.0], [0.0], [0.0]))),
                "nan_tmax": summarize(nan_tmax),
                "zero_tmax": summarize(zero_tmax),
            }

            print(json.dumps(cases))
            """
        )

        for case in data.values():
            self.assertFalse(case["valid"])
            self.assertEqual(case["shape"], -1)
            self.assertEqual(case["prim"], -1)
            self.assertTrue(case["t_inf"])

    def test_degenerate_geometry_is_stable_and_finite(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 2.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))
            mesh.build()
            secondary_edges = mesh.secondary_edges()

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            ray = pj.RayDetached(cuda.Array3f([0.5], [0.0], [-1.0]),
                          cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)

            finite_secondary = True
            for arr in (secondary_edges.start, secondary_edges.edge, secondary_edges.normal0, secondary_edges.normal1, secondary_edges.opposite):
                for axis in arr:
                    for value in list(axis):
                        finite_secondary = finite_secondary and math.isfinite(float(value))

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "shape": int(its.shape_id[0]),
                "prim": int(its.prim_id[0]),
                "t_inf": math.isinf(float(its.t[0])),
                "finite_secondary": finite_secondary,
                "edge_count": secondary_edges.size()
            }))
            """
        )

        self.assertFalse(data["valid"])
        self.assertEqual(data["shape"], -1)
        self.assertEqual(data["prim"], -1)
        self.assertTrue(data["t_inf"])
        self.assertTrue(data["finite_secondary"])
        self.assertGreaterEqual(data["edge_count"], 0)

    def test_batched_intersections_keep_finite_outputs_and_miss_sentinels(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            xs = []
            ys = []
            zs = []
            for iy in range(12):
                for ix in range(12):
                    xs.append(-0.2 + ix * 0.12)
                    ys.append(-0.2 + iy * 0.12)
                    zs.append(-1.0)

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0]),
                          cuda.Array3i([0, 0], [1, 2], [2, 3]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            rays = pj.RayDetached(cuda.Array3f(xs, ys, zs),
                           cuda.Array3f([0.0] * len(xs), [0.0] * len(xs), [1.0] * len(xs)))
            its = scene.intersect(rays)

            valid = list(its.is_valid())
            shape = list(its.shape_id)
            prim = list(its.prim_id)
            t = list(its.t)

            valid_nonfinite = 0
            miss_sentinel_errors = 0
            for i, flag in enumerate(valid):
                if flag:
                    values = [
                        float(t[i]),
                        float(its.p[0][i]), float(its.p[1][i]), float(its.p[2][i]),
                        float(its.n[0][i]), float(its.n[1][i]), float(its.n[2][i]),
                        float(its.geo_n[0][i]), float(its.geo_n[1][i]), float(its.geo_n[2][i]),
                        float(its.uv[0][i]), float(its.uv[1][i]),
                        float(its.barycentric[0][i]), float(its.barycentric[1][i]), float(its.barycentric[2][i]),
                    ]
                    if not all(math.isfinite(v) for v in values):
                        valid_nonfinite += 1
                else:
                    if shape[i] != -1 or prim[i] != -1 or not math.isinf(float(t[i])):
                        miss_sentinel_errors += 1

            print(json.dumps({
                "num_rays": len(xs),
                "num_hits": sum(bool(v) for v in valid),
                "valid_nonfinite": valid_nonfinite,
                "miss_sentinel_errors": miss_sentinel_errors
            }))
            """
        )

        self.assertEqual(data["num_rays"], 144)
        self.assertGreater(data["num_hits"], 0)
        self.assertEqual(data["valid_nonfinite"], 0)
        self.assertEqual(data["miss_sentinel_errors"], 0)

    def test_stress_subprocess_does_not_crash_on_repeated_build_intersect_and_edges(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            total_hits = 0
            total_samples = 0
            max_abs_grad = 0.0

            for it in range(20):
                mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0]),
                              cuda.Array3i([0, 0], [1, 2], [2, 3]))

                scene = pj.Scene()
                scene.add_mesh(mesh)
                scene.build()

                camera = pj.Camera(45.0, 1e-4, 1e4)
                camera.width = 32
                camera.height = 32
                camera.build()
                camera.prepare_edges(scene)
                sample = camera.sample_edge(cuda.Float([0.25]))
                total_samples += int(sample.idx[0] >= 0)

                xs = [0.1 + 0.02 * (i % 8) for i in range(64)]
                ys = [0.1 + 0.02 * (i // 8) for i in range(64)]
                rays = pj.RayDetached(cuda.Array3f(xs, ys, [-1.0] * 64),
                               cuda.Array3f([0.0] * 64, [0.0] * 64, [1.0] * 64))
                its = scene.intersect(rays)
                total_hits += sum(bool(v) for v in list(its.is_valid()))

                grad_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0]),
                                   cuda.Array3i([0], [1], [2]))
                tz = ad.Float([0.0])
                dr.enable_grad(tz)
                grad_mesh.to_world_left = ad.Matrix4f([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, tz],
                    [0.0, 0.0, 0.0, 1.0],
                ])
                grad_scene = pj.Scene()
                grad_scene.add_mesh(grad_mesh)
                grad_scene.build()
                ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                              ad.Array3f([0.0], [0.0], [1.0]))
                hit = grad_scene.intersect(ray)
                dr.backward(hit.t)
                max_abs_grad = max(max_abs_grad, abs(float(dr.grad(tz)[0])))

            print(json.dumps({
                "total_hits": total_hits,
                "total_samples": total_samples,
                "max_abs_grad": max_abs_grad
            }))
            """,
            timeout=180,
        )

        self.assertGreater(data["total_hits"], 0)
        self.assertGreater(data["total_samples"], 0)
        self.assertGreater(data["max_abs_grad"], 0.0)

    def test_dynamic_mesh_vertex_updates_require_commit_and_refresh_hits(self):
        data = run_json_case(
            """
            import json
            import math
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            moved = cuda.Array3f([2.0, 3.0, 2.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0])
            scene.update_mesh_vertices(mesh_id, moved)

            pending_error = False
            try:
                scene.intersect(
                    pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                                   cuda.Array3f([0.0], [0.0], [1.0]))
                )
            except Exception as e:
                pending_error = "pending updates" in str(e)

            scene.sync()
            ray = pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)

            print(json.dumps({
                "pending_error": pending_error,
                "valid": bool(its.is_valid()[0]),
                "shape": int(its.shape_id[0]),
                "prim": int(its.prim_id[0]),
                "t": float(its.t[0]),
                "p": [float(its.p[0][0]), float(its.p[1][0]), float(its.p[2][0])],
                "pending_after_commit": bool(scene.has_pending_updates()),
            }))
            """
        )

        self.assertTrue(data["pending_error"])
        self.assertTrue(data["valid"])
        self.assertEqual(data["shape"], 0)
        self.assertEqual(data["prim"], 0)
        self.assertAlmostEqual(data["t"], 1.0, places=5)
        self.assertAlmostEqual(data["p"][0], 2.25, places=5)
        self.assertAlmostEqual(data["p"][1], 0.25, places=5)
        self.assertFalse(data["pending_after_commit"])

    def test_dynamic_update_rejects_static_mesh_and_invalid_vertex_count(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            static_scene = pj.Scene()
            static_id = static_scene.add_mesh(mesh, dynamic=False)
            static_scene.build()

            dynamic_scene = pj.Scene()
            dynamic_id = dynamic_scene.add_mesh(mesh, dynamic=True)
            dynamic_scene.build()

            result = {}
            try:
                static_scene.update_mesh_vertices(
                    static_id,
                    cuda.Array3f([2.0, 3.0, 2.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0])
                )
            except Exception as e:
                result["static_mesh"] = "not dynamic" in str(e)

            try:
                dynamic_scene.update_mesh_vertices(
                    dynamic_id,
                    cuda.Array3f([0.0, 1.0],
                                 [0.0, 0.0],
                                 [0.0, 0.0])
                )
            except Exception as e:
                result["vertex_count"] = "vertex count must remain unchanged" in str(e)

            print(json.dumps(result))
            """
        )

        self.assertTrue(data["static_mesh"])
        self.assertTrue(data["vertex_count"])

    def test_dynamic_transform_update_keeps_local_primitive_id(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            scene.set_mesh_transform(
                mesh_id,
                cuda.Matrix4f([
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ])
            )
            scene.sync()

            ray = pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                                 cuda.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "shape": int(its.shape_id[0]),
                "prim": int(its.prim_id[0]),
                "p": [float(its.p[0][0]), float(its.p[1][0]), float(its.p[2][0])],
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["shape"], 0)
        self.assertEqual(data["prim"], 0)
        self.assertAlmostEqual(data["p"][0], 2.25, places=5)
        self.assertAlmostEqual(data["p"][1], 0.25, places=5)

    def test_mixed_static_dynamic_optix_split_keeps_hits_and_shadow_queries_correct(self):
        data = run_json_case(
            """
            import os
            os.environ["RAYD_OPTIX_SPLIT_MODE"] = "on"

            import json
            import rayd as pj
            import drjit.cuda as cuda

            static_mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0],
                                               [0.0, 0.0, 0.0]),
                                  cuda.Array3i([0], [1], [2]))
            dynamic_mesh = pj.Mesh(cuda.Array3f([2.0, 3.0, 2.0],
                                                [0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0]),
                                   cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            scene.add_mesh(static_mesh, dynamic=False)
            dynamic_id = scene.add_mesh(dynamic_mesh, dynamic=True)
            scene.build()

            static_ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]),
                                        cuda.Array3f([0.0], [0.0], [1.0]))
            dynamic_ray_before = pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]),
                                                cuda.Array3f([0.0], [0.0], [1.0]))

            static_before = scene.intersect(static_ray)
            dynamic_before = scene.intersect(dynamic_ray_before)
            static_before_valid = bool(static_before.is_valid()[0])
            static_before_shape = int(static_before.shape_id[0])
            dynamic_before_valid = bool(dynamic_before.is_valid()[0])
            dynamic_before_shape = int(dynamic_before.shape_id[0])

            scene.set_mesh_transform(
                dynamic_id,
                cuda.Matrix4f([
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ])
            )
            scene.sync()

            dynamic_ray_after = pj.RayDetached(cuda.Array3f([4.25], [0.25], [-1.0]),
                                               cuda.Array3f([0.0], [0.0], [1.0]))
            static_after = scene.intersect(static_ray)
            dynamic_after = scene.intersect(dynamic_ray_after)
            old_position_after = scene.intersect(dynamic_ray_before)
            shadow_old = scene.shadow_test(dynamic_ray_before)
            shadow_new = scene.shadow_test(dynamic_ray_after)

            print(json.dumps({
                "static_before_valid": static_before_valid,
                "static_before_shape": static_before_shape,
                "dynamic_before_valid": dynamic_before_valid,
                "dynamic_before_shape": dynamic_before_shape,
                "static_after_valid": bool(static_after.is_valid()[0]),
                "static_after_shape": int(static_after.shape_id[0]),
                "dynamic_after_valid": bool(dynamic_after.is_valid()[0]),
                "dynamic_after_shape": int(dynamic_after.shape_id[0]),
                "dynamic_after_x": float(dynamic_after.p[0][0]),
                "old_position_after_valid": bool(old_position_after.is_valid()[0]),
                "shadow_old": bool(shadow_old[0]),
                "shadow_new": bool(shadow_new[0]),
            }))
            """
        )

        self.assertTrue(data["static_before_valid"])
        self.assertEqual(data["static_before_shape"], 0)
        self.assertTrue(data["dynamic_before_valid"])
        self.assertEqual(data["dynamic_before_shape"], 1)
        self.assertTrue(data["static_after_valid"])
        self.assertEqual(data["static_after_shape"], 0)
        self.assertTrue(data["dynamic_after_valid"])
        self.assertEqual(data["dynamic_after_shape"], 1)
        self.assertAlmostEqual(data["dynamic_after_x"], 4.25, places=5)
        self.assertFalse(data["old_position_after_valid"])
        self.assertFalse(data["shadow_old"])
        self.assertTrue(data["shadow_new"])

    def test_dynamic_vertex_updates_preserve_gradients(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            verts = ad.Array3f([0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.5, 0.5, 0.5])
            dr.enable_grad(verts)
            scene.update_mesh_vertices(mesh_id, verts)
            scene.sync()

            ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]),
                         ad.Array3f([0.0], [0.0], [1.0]))
            its = scene.intersect(ray)
            dr.backward(its.t)
            grad = dr.grad(verts)

            print(json.dumps({
                "valid": bool(its.is_valid()[0]),
                "grad_z_sum": float(grad[2][0] + grad[2][1] + grad[2][2]),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertGreater(data["grad_z_sum"], 0.0)

    def test_dynamic_commit_invalidates_primary_edge_cache_until_reprepared(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit.cuda as cuda

            mesh = pj.Mesh(cuda.Array3f([-0.5, 0.5, 0.0],
                                       [-0.5, -0.5, 0.5],
                                       [3.0, 3.0, 3.0]),
                          cuda.Array3i([0], [1], [2]))

            scene = pj.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            camera = pj.Camera(45.0, 1e-4, 1e4)
            camera.width = 32
            camera.height = 32
            camera.build()
            camera.prepare_edges(scene)

            scene.update_mesh_vertices(
                mesh_id,
                cuda.Array3f([-0.25, 0.75, 0.25],
                             [-0.5, -0.5, 0.5],
                             [3.0, 3.0, 3.0])
            )
            scene.sync()

            invalidated = False
            try:
                camera.sample_edge(cuda.Float([0.25]))
            except Exception as e:
                invalidated = "not prepared" in str(e)

            camera.prepare_edges(scene)
            sample = camera.sample_edge(cuda.Float([0.25]))

            print(json.dumps({
                "invalidated": invalidated,
                "sample_idx": int(sample.idx[0]),
                "sample_pdf": float(sample.pdf[0]),
            }))
            """
        )

        self.assertTrue(data["invalidated"])
        self.assertGreaterEqual(data["sample_idx"], 0)
        self.assertGreater(data["sample_pdf"], 0.0)

    def test_gradient_benchmark_repeats_without_rebuilding_optix_state(self):
        data = run_json_case(
            """
            import json
            from tests.benchmark_support import RayDBackend, _make_grid_mesh_data, _make_ray_data

            backend = RayDBackend()
            mesh = _make_grid_mesh_data(8)
            rays = _make_ray_data(8)
            result = backend.gradient_performance(
                mesh,
                rays,
                repeats=2,
                warmup=1,
                dynamic_update=False,
            )
            print(json.dumps(result))
            """,
            timeout=180,
        )

        self.assertGreater(data["min_ms"], 0.0)
        self.assertGreater(data["avg_ms"], 0.0)
        self.assertGreater(data["qps_m"], 0.0)

    def test_forward_benchmark_reports_full_and_reduced_modes(self):
        data = run_json_case(
            """
            import json
            from tests.benchmark_support import RayDBackend, _make_grid_mesh_data, _make_ray_data

            backend = RayDBackend()
            mesh = _make_grid_mesh_data(8)
            updated_mesh = _make_grid_mesh_data(8, x_offset=2.0)
            rays = _make_ray_data(8)
            updated_rays = _make_ray_data(8, x_offset=2.0)

            static_result = backend.forward_performance(mesh, rays, repeats=2, warmup=1)
            dynamic_result = backend.dynamic_forward_performance(
                mesh,
                updated_mesh,
                rays,
                updated_rays,
                repeats=2,
                warmup=1,
            )

            print(json.dumps({
                "static_keys": sorted(static_result.keys()),
                "dynamic_keys": sorted(dynamic_result.keys()),
                "static": static_result,
                "dynamic": dynamic_result,
            }))
            """
        )

        self.assertEqual(data["static_keys"], ["full", "reduced"])
        self.assertEqual(data["dynamic_keys"], ["full", "reduced"])
        for bucket in ("static", "dynamic"):
            for mode in ("full", "reduced"):
                self.assertGreater(data[bucket][mode]["min_ms"], 0.0, msg=f"{bucket}:{mode}")
                self.assertGreater(data[bucket][mode]["avg_ms"], 0.0, msg=f"{bucket}:{mode}")
                self.assertGreater(data[bucket][mode]["qps_m"], 0.0, msg=f"{bucket}:{mode}")

    def test_primary_edge_helper_produces_nonzero_visibility_gradient(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda
            import drjit.cuda.ad as ad

            width = 64
            height = 64

            verts_x = [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]
            verts_y = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]
            verts_z = [3.5, 3.5, 3.5, 3.5, 4.5, 4.5, 4.5, 4.5]

            faces = cuda.Array3i(
                [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3],
                [2, 3, 5, 6, 7, 4, 2, 6, 1, 5, 7, 6],
                [1, 2, 6, 7, 3, 7, 6, 5, 5, 4, 6, 2],
            )

            tx = ad.Float([0.0])
            dr.enable_grad(tx)

            mesh = pj.Mesh(cuda.Array3f(verts_x, verts_y, verts_z), faces)
            mesh.to_world_left = ad.Matrix4f([
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            camera = pj.Camera(45.0, 1e-4, 1e4)
            camera.width = width
            camera.height = height
            camera.build()

            image = camera.render_grad(scene, spp=4)
            dr.set_grad(tx, 1.0)
            dr.forward_to(image)
            grad = dr.grad(image)
            grad_values = [float(v) for v in list(grad.array)]

            print(json.dumps({
                "type": type(image).__name__,
                "shape": list(image.shape),
                "abs_max": max(abs(v) for v in grad_values),
                "nonzero": sum(1 for v in grad_values if abs(v) > 1e-8),
            }))
            """,
            timeout=180,
        )

        self.assertEqual(data["type"], "TensorXf")
        self.assertEqual(data["shape"], [64, 64])
        self.assertGreater(data["abs_max"], 0.0)
        self.assertGreater(data["nonzero"], 0)

    def test_camera_render_returns_tensor_depth_image(self):
        data = run_json_case(
            """
            import json
            import rayd as pj
            import drjit as dr
            import drjit.cuda as cuda

            mesh = pj.Mesh(
                cuda.Array3f(
                    [-1.0, 1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0, 1.0],
                    [3.0, 3.0, 3.0, 3.0],
                ),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()

            camera = pj.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 12
            camera.build()

            image = camera.render(scene)
            values = [float(v) for v in list(dr.detach(image.array))]

            print(json.dumps({
                "type": type(image).__name__,
                "shape": list(image.shape),
                "hits": sum(1 for v in values if v > 0.0),
                "min_positive": min(v for v in values if v > 0.0),
            }))
            """,
            timeout=180,
        )

        self.assertEqual(data["type"], "TensorXf")
        self.assertEqual(data["shape"], [12, 16])
        self.assertGreater(data["hits"], 0)
        self.assertGreater(data["min_positive"], 0.0)


if __name__ == "__main__":
    unittest.main()



