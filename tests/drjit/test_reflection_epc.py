import json
import math
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_CWD = Path(sys.executable).resolve().parent


def run_script(script: str, *, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=SUBPROCESS_CWD,
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

    def test_grouped_epc_resolves_backprojected_point_on_expected_surface_group(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            mirror = pj.Mesh(
                cuda.Array3f([-1.0, 1.0, 1.0, -1.0],
                             [-1.0, -1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(mirror)
            scene.build()

            ray = pj.RayDetached(
                cuda.Array3f([0.0], [-0.5], [-1.0]),
                cuda.Array3f([0.0], [0.0], [1.0]),
            )
            rx = cuda.Array3f([0.0], [1.5], [-1.0])

            options = pj.ReflectionEpcOptions()
            options.expected_prim_ids = cuda.Int([1])
            options.surface_group_id = cuda.Int([0, 0])
            options.surface_group_size = cuda.Int([2])
            options.surface_group_members = cuda.Int([0, 1])
            options.surface_max_group_size = 2
            options.visibility_ignore_mode = "surface_group"

            result = scene.trace_reflection_epc(ray, rx, max_bounces=1, options=options)

            bad_options = pj.ReflectionEpcOptions()
            bad_options.expected_prim_ids = cuda.Int([1])
            bad_options.surface_group_id = cuda.Int([0, 1])
            bad_options.surface_group_size = cuda.Int([1, 1])
            bad_options.surface_group_members = cuda.Int([0, 1])
            bad_options.surface_max_group_size = 1
            bad_options.visibility_ignore_mode = "surface_group"
            bad = scene.trace_reflection_epc(ray, rx, max_bounces=1, options=bad_options)

            dr.eval(result.valid,
                    result.path_length,
                    result.reflection_points,
                    result.trace_prim_ids,
                    result.resolved_prim_ids,
                    result.surface_group_ids,
                    result.plane_normals,
                    bad.valid)

            print(json.dumps({
                "valid": bool(result.valid[0]),
                "bad_valid": bool(bad.valid[0]),
                "path_length": float(result.path_length[0]),
                "point": [
                    float(result.reflection_points[0][0]),
                    float(result.reflection_points[1][0]),
                    float(result.reflection_points[2][0]),
                ],
                "trace_prim": int(result.trace_prim_ids[0]),
                "resolved_prim": int(result.resolved_prim_ids[0]),
                "legacy_prim": int(result.prim_ids[0]),
                "group": int(result.surface_group_ids[0]),
                "normal": [
                    float(result.plane_normals[0][0]),
                    float(result.plane_normals[1][0]),
                    float(result.plane_normals[2][0]),
                ],
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertFalse(data["bad_valid"])
        self.assertAlmostEqual(data["path_length"], 2.0 * math.sqrt(2.0), places=4)
        self.assertAlmostEqual(data["point"][0], 0.0, places=4)
        self.assertAlmostEqual(data["point"][1], 0.5, places=4)
        self.assertAlmostEqual(data["point"][2], 0.0, places=4)
        self.assertEqual(data["trace_prim"], 0)
        self.assertEqual(data["resolved_prim"], 1)
        self.assertEqual(data["legacy_prim"], data["trace_prim"])
        self.assertEqual(data["group"], 0)
        self.assertAlmostEqual(data["normal"][0], 0.0, places=4)
        self.assertAlmostEqual(data["normal"][1], 0.0, places=4)
        self.assertAlmostEqual(data["normal"][2], -1.0, places=4)

    def test_grouped_epc_reports_blocked_group_and_final_ignore_group(self):
        data = run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            mirror = pj.Mesh(
                cuda.Array3f([-1.0, 1.0, 1.0, -1.0],
                             [-1.0, -1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )
            blocker = pj.Mesh(
                cuda.Array3f([-0.5, 0.5, 0.5, -0.5],
                             [1.0, 1.0, 1.0, 1.0],
                             [-0.8, -0.8, -0.2, -0.2]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(mirror)
            scene.add_mesh(blocker)
            scene.build()

            ray = pj.RayDetached(
                cuda.Array3f([0.0], [-0.5], [-1.0]),
                cuda.Array3f([0.0], [0.0], [1.0]),
            )
            rx = cuda.Array3f([0.0], [1.5], [-1.0])

            blocked_options = pj.ReflectionEpcOptions()
            blocked_options.expected_prim_ids = cuda.Int([1])
            blocked_options.surface_group_id = cuda.Int([0, 0, 1, 1])
            blocked_options.surface_group_size = cuda.Int([2, 2])
            blocked_options.surface_group_members = cuda.Int([0, 1, 2, 3])
            blocked_options.surface_max_group_size = 2
            blocked_options.visibility_ignore_mode = "surface_group"

            ignored_options = pj.ReflectionEpcOptions()
            ignored_options.expected_prim_ids = cuda.Int([1])
            ignored_options.surface_group_id = cuda.Int([0, 0, 1, 1])
            ignored_options.surface_group_size = cuda.Int([2, 2])
            ignored_options.surface_group_members = cuda.Int([0, 1, 2, 3])
            ignored_options.surface_max_group_size = 2
            ignored_options.visibility_ignore_mode = "surface_group"
            ignored_options.final_ignore_group_ids = cuda.Int([1])

            blocked = scene.trace_reflection_epc(ray, rx, max_bounces=1, options=blocked_options)
            ignored = scene.trace_reflection_epc(ray, rx, max_bounces=1, options=ignored_options)

            dr.eval(blocked.valid,
                    blocked.first_blocked_segment,
                    blocked.first_blocked_prim,
                    blocked.first_blocked_group,
                    ignored.valid,
                    ignored.first_blocked_group)

            print(json.dumps({
                "blocked_valid": bool(blocked.valid[0]),
                "blocked_segment": int(blocked.first_blocked_segment[0]),
                "blocked_prim": int(blocked.first_blocked_prim[0]),
                "blocked_group": int(blocked.first_blocked_group[0]),
                "ignored_valid": bool(ignored.valid[0]),
                "ignored_blocked_group": int(ignored.first_blocked_group[0]),
            }))
            """
        )

        self.assertFalse(data["blocked_valid"])
        self.assertEqual(data["blocked_segment"], 1)
        self.assertGreaterEqual(data["blocked_prim"], 2)
        self.assertEqual(data["blocked_group"], 1)
        self.assertTrue(data["ignored_valid"])
        self.assertEqual(data["ignored_blocked_group"], -1)

    def test_epc_field_one_bounce_matches_reference_and_returns_geometry(self):
        data = run_json(
            """
            import cmath
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            def dot(a, b):
                return sum(x * y for x, y in zip(a, b))

            def cross(a, b):
                return (
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                )

            def norm(a):
                return math.sqrt(max(dot(a, a), 0.0))

            def scale(a, s):
                return tuple(x * s for x in a)

            def sub(a, b):
                return tuple(x - y for x, y in zip(a, b))

            def normalize(a):
                n = max(norm(a), 1e-12)
                return scale(a, 1.0 / n)

            def stable_perpendicular(direction, preferred):
                direction = normalize(direction)
                projected = sub(preferred, scale(direction, dot(preferred, direction)))
                if dot(projected, projected) > 1e-12:
                    return normalize(projected)
                axis = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (0.0, 1.0, 0.0)
                projected = sub(axis, scale(direction, dot(axis, direction)))
                return normalize(projected)

            def reflect_reference(tx, hit, rx, slot_normal, tx_pol,
                                  eta_r, mu_r, sigma, gain, omega, wavelength):
                eps0 = 8.854187817e-12
                incident = normalize(sub(hit, tx))
                normal = normalize(slot_normal)
                if dot(incident, normal) > 0.0:
                    normal = scale(normal, -1.0)
                reflected = normalize(sub(incident, scale(normal, 2.0 * dot(incident, normal))))
                transverse = sub(tx_pol, scale(incident, dot(tx_pol, incident)))
                if dot(transverse, transverse) <= 1e-12:
                    transverse = stable_perpendicular(incident, tx_pol)
                else:
                    transverse = normalize(transverse)
                s_hat = cross(normal, incident)
                if dot(s_hat, s_hat) <= 1e-12:
                    s_hat = stable_perpendicular(incident, normal)
                else:
                    s_hat = normalize(s_hat)
                p_in = cross(s_hat, incident)
                if dot(p_in, p_in) <= 1e-12:
                    p_in = stable_perpendicular(incident, normal)
                else:
                    p_in = normalize(p_in)
                p_out = cross(s_hat, reflected)
                if dot(p_out, p_out) <= 1e-12:
                    p_out = stable_perpendicular(reflected, normal)
                else:
                    p_out = normalize(p_out)

                cos_theta = min(max(abs(dot(incident, normal)), 1e-6), 1.0)
                sin2 = max(0.0, 1.0 - cos_theta * cos_theta)
                eta = complex(max(eta_r, 1e-6), -max(sigma, 0.0) / (max(omega, 1e-6) * eps0))
                mu = complex(max(mu_r, 1e-6), 0.0)
                a = cmath.sqrt(mu * eta - complex(sin2, 0.0))
                mu_cos = complex(mu_r * cos_theta, 0.0)
                eta_cos = eta * cos_theta
                r_te = gain * (mu_cos - a) / (mu_cos + a)
                r_tm = gain * (eta_cos - a) / (eta_cos + a)
                e_s = dot(transverse, s_hat)
                e_p = dot(transverse, p_in)
                field = [
                    s_hat[i] * r_te * e_s + p_out[i] * r_tm * e_p
                    for i in range(3)
                ]
                path_length = norm(sub(hit, tx)) + norm(sub(rx, hit))
                coeff = cmath.exp(-1j * (2.0 * math.pi / wavelength) * path_length)
                coeff *= wavelength / (4.0 * math.pi * max(path_length, 1e-6))
                return [f * coeff for f in field], path_length

            mirror = pj.Mesh(
                cuda.Array3f([-1.0, 1.0, 1.0, -1.0],
                             [-1.0, -1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(mirror)
            scene.build()

            tx = (0.0, -0.5, -1.0)
            hit = (0.0, 0.5, 0.0)
            rx_tuple = (0.0, 1.5, -1.0)
            ray = pj.RayDetached(
                cuda.Array3f([tx[0]], [tx[1]], [tx[2]]),
                cuda.Array3f([0.0], [0.0], [1.0]),
            )
            rx = cuda.Array3f([rx_tuple[0]], [rx_tuple[1]], [rx_tuple[2]])

            options = pj.ReflectionEpcFieldOptions()
            options.expected_prim_ids = cuda.Int([1])
            options.surface_group_id = cuda.Int([0, 0])
            options.surface_group_size = cuda.Int([2])
            options.surface_group_members = cuda.Int([0, 1])
            options.surface_max_group_size = 2
            options.visibility_ignore_mode = "surface_group"
            options.slot_plane_normal = cuda.Array3f([0.0], [0.0], [1.0])
            options.slot_eta_r = cuda.Float([4.0])
            options.slot_mu_r = cuda.Float([1.0])
            options.slot_sigma = cuda.Float([0.0])
            options.slot_gain = cuda.Float([1.0])
            options.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
            options.omega = 2.0 * math.pi * 299792458.0 / 2.0
            options.wavelength = 2.0
            options.return_geometry = True
            options.return_endpoints = True

            expected_field, expected_path = reflect_reference(
                tx, hit, rx_tuple, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                4.0, 1.0, 0.0, 1.0, options.omega, options.wavelength)

            pj.native_launch_audit_clear()
            result = scene.trace_reflection_epc_field(ray, rx, max_bounces=1, options=options)
            dr.eval(result.valid,
                    result.field_x_re,
                    result.field_x_im,
                    result.field_y_re,
                    result.field_y_im,
                    result.field_z_re,
                    result.field_z_im,
                    result.tx_pos,
                    result.first_hit,
                    result.last_hit,
                    result.hit_points,
                    result.normals,
                    result.resolved_prim_ids,
                    result.surface_group_ids)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "ray_count": result.ray_count,
                "max_bounces": result.max_bounces,
                "valid": bool(result.valid[0]),
                "field": [
                    float(result.field_x_re[0]), float(result.field_x_im[0]),
                    float(result.field_y_re[0]), float(result.field_y_im[0]),
                    float(result.field_z_re[0]), float(result.field_z_im[0]),
                ],
                "expected": [
                    expected_field[0].real, expected_field[0].imag,
                    expected_field[1].real, expected_field[1].imag,
                    expected_field[2].real, expected_field[2].imag,
                ],
                "tx": [float(result.tx_pos[0][0]),
                       float(result.tx_pos[1][0]),
                       float(result.tx_pos[2][0])],
                "first_hit": [float(result.first_hit[0][0]),
                              float(result.first_hit[1][0]),
                              float(result.first_hit[2][0])],
                "last_hit": [float(result.last_hit[0][0]),
                             float(result.last_hit[1][0]),
                             float(result.last_hit[2][0])],
                "hit": [float(result.hit_points[0][0]),
                        float(result.hit_points[1][0]),
                        float(result.hit_points[2][0])],
                "normal": [float(result.normals[0][0]),
                           float(result.normals[1][0]),
                           float(result.normals[2][0])],
                "resolved_prim": int(result.resolved_prim_ids[0]),
                "group": int(result.surface_group_ids[0]),
                "path_length": float(result.path_length[0]),
                "expected_path": expected_path,
                "optix_launches": audit["trace_reflections"]["optix_launch"],
                "cuda_kernel_launches": audit["trace_reflections"]["cuda_kernel_launches"],
            }))
            """
        )

        self.assertEqual(data["ray_count"], 1)
        self.assertEqual(data["max_bounces"], 1)
        self.assertTrue(data["valid"])
        for actual, expected in zip(data["field"], data["expected"]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(data["path_length"], data["expected_path"], places=5)
        self.assertEqual(data["tx"], [0.0, -0.5, -1.0])
        for actual, expected in zip(data["first_hit"], [0.0, 0.5, 0.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(data["last_hit"], [0.0, 0.5, 0.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(data["hit"], [0.0, 0.5, 0.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(data["normal"], [0.0, 0.0, -1.0])
        self.assertEqual(data["resolved_prim"], 1)
        self.assertEqual(data["group"], 0)
        self.assertEqual(data["optix_launches"], 1)
        self.assertEqual(data["cuda_kernel_launches"], 1)

    def test_epc_field_can_skip_optional_geometry_outputs(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            mirror = pj.Mesh(
                cuda.Array3f([-1.0, 1.0, 1.0, -1.0],
                             [-1.0, -1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )
            scene = pj.Scene()
            scene.add_mesh(mirror)
            scene.build()

            ray = pj.RayDetached(
                cuda.Array3f([0.0], [-0.5], [-1.0]),
                cuda.Array3f([0.0], [0.0], [1.0]),
            )
            rx = cuda.Array3f([0.0], [1.5], [-1.0])

            options = pj.ReflectionEpcFieldOptions()
            options.expected_prim_ids = cuda.Int([1])
            options.surface_group_id = cuda.Int([0, 0])
            options.surface_group_size = cuda.Int([2])
            options.surface_group_members = cuda.Int([0, 1])
            options.surface_max_group_size = 2
            options.visibility_ignore_mode = "surface_group"
            options.slot_plane_normal = cuda.Array3f([0.0], [0.0], [1.0])
            options.slot_eta_r = cuda.Float([4.0])
            options.slot_mu_r = cuda.Float([1.0])
            options.slot_sigma = cuda.Float([0.0])
            options.slot_gain = cuda.Float([1.0])
            options.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
            options.omega = 2.0 * math.pi * 299792458.0 / 2.0
            options.wavelength = 2.0
            options.return_geometry = False
            options.return_endpoints = False

            result = scene.trace_reflection_epc_field(ray, rx, max_bounces=1, options=options)
            options.slot_gain = cuda.Float([0.0])
            zero_gain = scene.trace_reflection_epc_field(
                ray, rx, max_bounces=1, options=options)
            dr.eval(result.valid,
                    result.field_x_re,
                    result.field_x_im,
                    zero_gain.valid,
                    zero_gain.field_x_re,
                    zero_gain.field_x_im,
                    zero_gain.field_y_re,
                    zero_gain.field_y_im,
                    zero_gain.field_z_re,
                    zero_gain.field_z_im)
            print(json.dumps({
                "valid": bool(result.valid[0]),
                "field_power": float(result.field_x_re[0]) ** 2
                               + float(result.field_x_im[0]) ** 2,
                "zero_gain_valid": bool(zero_gain.valid[0]),
                "zero_gain_power": (
                    float(zero_gain.field_x_re[0]) ** 2
                    + float(zero_gain.field_x_im[0]) ** 2
                    + float(zero_gain.field_y_re[0]) ** 2
                    + float(zero_gain.field_y_im[0]) ** 2
                    + float(zero_gain.field_z_re[0]) ** 2
                    + float(zero_gain.field_z_im[0]) ** 2
                ),
                "hit_width": int(dr.width(result.hit_points[0])),
                "normal_width": int(dr.width(result.normals[0])),
                "resolved_width": int(dr.width(result.resolved_prim_ids)),
                "group_width": int(dr.width(result.surface_group_ids)),
                "tx_width": int(dr.width(result.tx_pos[0])),
                "first_width": int(dr.width(result.first_hit[0])),
                "last_width": int(dr.width(result.last_hit[0])),
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertGreater(data["field_power"], 0.0)
        self.assertTrue(data["zero_gain_valid"])
        self.assertAlmostEqual(data["zero_gain_power"], 0.0, places=12)
        self.assertEqual(data["hit_width"], 0)
        self.assertEqual(data["normal_width"], 0)
        self.assertEqual(data["resolved_width"], 0)
        self.assertEqual(data["group_width"], 0)
        self.assertEqual(data["tx_width"], 0)
        self.assertEqual(data["first_width"], 0)
        self.assertEqual(data["last_width"], 0)

    def test_epc_field_direct_uses_expected_plane_without_seed_trace(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            reflector = pj.Mesh(
                cuda.Array3f([-4.0, 4.0, -4.0, 4.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [-1.0, -1.0, 1.0, 1.0]),
                cuda.Array3i([0, 0], [1, 3], [3, 2]),
            )
            blocker = pj.Mesh(
                cuda.Array3f([0.5, 0.8, 0.5, 0.8],
                             [-1.0, -1.0, -1.0, -1.0],
                             [-0.35, -0.35, -0.05, -0.05]),
                cuda.Array3i([0, 0], [3, 2], [1, 3]),
            )

            scene = pj.Scene()
            scene.add_mesh(reflector)
            scene.add_mesh(blocker)
            scene.build()

            tx = cuda.Array3f([0.0], [-2.0], [0.0])
            rx = cuda.Array3f([6.0], [-2.0], [0.0])

            options = pj.ReflectionEpcFieldOptions()
            options.expected_prim_ids = cuda.Int([0])
            options.slot_plane_point = cuda.Array3f([-4.0], [0.0], [-1.0])
            options.slot_plane_normal = cuda.Array3f([0.0], [-1.0], [0.0])
            options.slot_eta_r = cuda.Float([4.0])
            options.slot_mu_r = cuda.Float([1.0])
            options.slot_sigma = cuda.Float([0.0])
            options.slot_gain = cuda.Float([1.0])
            options.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
            options.omega = 2.0 * math.pi * 3.5e9
            options.wavelength = 299792458.0 / 3.5e9
            options.return_geometry = True
            options.return_endpoints = True

            pj.native_launch_audit_clear()
            result = scene.trace_reflection_epc_field_direct(
                tx, rx, max_bounces=1, options=options)
            dr.eval(result.valid,
                    result.path_length,
                    result.field_x_re,
                    result.field_x_im,
                    result.first_hit,
                    result.hit_points,
                    result.normals,
                    result.resolved_prim_ids)
            audit = pj.native_launch_audit()

            print(json.dumps({
                "valid": bool(result.valid[0]),
                "path_length": float(result.path_length[0]),
                "field_power": float(result.field_x_re[0]) ** 2
                               + float(result.field_x_im[0]) ** 2,
                "first_hit": [float(result.first_hit[0][0]),
                              float(result.first_hit[1][0]),
                              float(result.first_hit[2][0])],
                "hit": [float(result.hit_points[0][0]),
                        float(result.hit_points[1][0]),
                        float(result.hit_points[2][0])],
                "normal": [float(result.normals[0][0]),
                           float(result.normals[1][0]),
                           float(result.normals[2][0])],
                "resolved_prim": int(result.resolved_prim_ids[0]),
                "optix_launches": audit["trace_reflections"]["optix_launch"],
                "cuda_kernel_launches": audit["trace_reflections"]["cuda_kernel_launches"],
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["path_length"], 2.0 * math.sqrt(13.0), places=4)
        self.assertGreater(data["field_power"], 0.0)
        for actual, expected in zip(data["first_hit"], [3.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)
        for actual, expected in zip(data["hit"], [3.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)
        for actual, expected in zip(data["normal"], [0.0, -1.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertEqual(data["resolved_prim"], 0)
        self.assertEqual(data["optix_launches"], 1)
        self.assertEqual(data["cuda_kernel_launches"], 1)

    def test_epc_field_direct_can_skip_resolved_geometry_outputs(self):
        data = run_json(
            """
            import json
            import math
            import drjit as dr
            import drjit.cuda as cuda
            import rayd as pj

            reflector = pj.Mesh(
                cuda.Array3f([-4.0, 4.0, -4.0, 4.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [-1.0, -1.0, 1.0, 1.0]),
                cuda.Array3i([0, 0], [1, 3], [3, 2]),
            )
            scene = pj.Scene()
            scene.add_mesh(reflector)
            scene.build()

            tx = cuda.Array3f([0.0], [-2.0], [0.0])
            rx = cuda.Array3f([6.0], [-2.0], [0.0])

            options = pj.ReflectionEpcFieldOptions()
            options.expected_prim_ids = cuda.Int([0])
            options.slot_plane_point = cuda.Array3f([-4.0], [0.0], [-1.0])
            options.slot_plane_normal = cuda.Array3f([0.0], [-1.0], [0.0])
            options.slot_eta_r = cuda.Float([4.0])
            options.slot_mu_r = cuda.Float([1.0])
            options.slot_sigma = cuda.Float([0.0])
            options.slot_gain = cuda.Float([1.0])
            options.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
            options.omega = 2.0 * math.pi * 3.5e9
            options.wavelength = 299792458.0 / 3.5e9
            options.return_geometry = True
            options.return_endpoints = False
            options.return_hit_points = True
            options.return_normals = True
            options.return_resolved_prim_ids = False
            options.return_surface_group_ids = False

            result = scene.trace_reflection_epc_field_direct(
                tx, rx, max_bounces=1, options=options)
            dr.eval(result.valid,
                    result.hit_points,
                    result.normals,
                    result.resolved_prim_ids,
                    result.surface_group_ids)

            print(json.dumps({
                "valid": bool(result.valid[0]),
                "hit_width": int(dr.width(result.hit_points[0])),
                "normal_width": int(dr.width(result.normals[0])),
                "resolved_width": int(dr.width(result.resolved_prim_ids)),
                "group_width": int(dr.width(result.surface_group_ids)),
                "hit": [float(result.hit_points[0][0]),
                        float(result.hit_points[1][0]),
                        float(result.hit_points[2][0])],
                "normal": [float(result.normals[0][0]),
                           float(result.normals[1][0]),
                           float(result.normals[2][0])],
            }))
            """
        )

        self.assertTrue(data["valid"])
        self.assertEqual(data["hit_width"], 1)
        self.assertEqual(data["normal_width"], 1)
        self.assertEqual(data["resolved_width"], 0)
        self.assertEqual(data["group_width"], 0)
        for actual, expected in zip(data["hit"], [3.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)
        for actual, expected in zip(data["normal"], [0.0, -1.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
