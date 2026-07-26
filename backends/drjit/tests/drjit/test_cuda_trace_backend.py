"""P3 Stage B/C gate: the pure-CUDA triangle TraceBackend.

Every case runs in a fresh subprocess (mirrors ``test_geometry`` /
``test_golden_scenes``) so Dr.Jit/CUDA state never leaks between cases. The suite
covers, all against ``trace_backend='cuda'``:

* Golden cross-backend parity: the declarative golden scenes collected under the
  CUDA backend match the checked-in OptiX baselines (discrete bit-identical,
  continuous within ``operations.json`` tolerances), for every intersect /
  shadow_test / edge query the backend can serve.
* Watertight property: a grid of rays across a shared-edge quad, including rays
  exactly on the diagonal, hits exactly once in closest-hit semantics and is
  occluded, with no gaps; degenerate and large-coordinate scenes behave.
* AD parity: vertex- and transform-gradient cases produce gradients identical to
  the default OptiX backend (the AD recompute depends only on the winner).
* Refit, launch-count / zero-alloc stability, the symbolic-recording guard,
  capability reporting, and the first-blocker self-test.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `tests.golden` lives at the repository root; see backends/drjit/tests/__init__.py
# for why it resolves from here under both documented invocations.
from tests.golden import compare  # noqa: E402


def _run_json(script, timeout=360):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON.\nSTDERR:\n{result.stderr}")
    return json.loads(lines[-1])


class CudaGoldenParityTests(unittest.TestCase):
    """The golden scenes under the CUDA backend match the OptiX baselines."""

    @classmethod
    def setUpClass(cls):
        cls.golden = _run_json(
            """
            import json
            from tests.golden.runner import collect_golden
            print(json.dumps(collect_golden(trace_backend="cuda"), sort_keys=True))
            """
        )

    def test_discrete_and_continuous_match_optix_baselines(self):
        abs_tol, rel_tol = compare.continuous_tolerances()
        compared_queries = 0
        for name in compare.baseline_scene_names():
            baseline = compare.baseline_scene(name)["queries"]
            actual = self.golden[name]["queries"]
            for query_name, actual_record in actual.items():
                baseline_record = baseline[query_name]
                path = f"{name}.{query_name}"
                self.assertEqual(actual_record["kind"], baseline_record["kind"], f"{path}: kind")
                self.assertEqual(
                    actual_record["discrete"], baseline_record["discrete"], f"{path}: discrete"
                )
                compare._assert_continuous(
                    self, actual_record["continuous"], baseline_record["continuous"],
                    f"{path}.continuous", abs_tol, rel_tol,
                )
                compared_queries += 1
        # Guard against the runner silently skipping everything.
        self.assertGreaterEqual(compared_queries, 12, "too few CUDA golden queries compared")

    def test_core_intersect_scenes_are_covered(self):
        # These must all run under the CUDA backend (intersect / shadow only).
        for name in (
            "single_tri", "shared_edge_quad", "degenerate_tri", "large_coordinates",
            "self_intersection", "multi_mesh_ids", "dynamic_refit", "inactive_lanes",
            "batch_sizes",
        ):
            self.assertIn(name, self.golden, f"{name} missing from CUDA golden run")
            self.assertTrue(self.golden[name]["queries"], f"{name} produced no CUDA queries")


class CudaWatertightTests(unittest.TestCase):
    """Watertight closest-hit / occlusion, including on-diagonal rays."""

    def test_shared_edge_quad_grid_hits_exactly_once(self):
        data = _run_json(
            """
            import json
            import drjit.cuda as cuda
            import rayd.drjit as rd

            QUAD_VERTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            QUAD_FACES = [[0, 1, 2], [0, 2, 3]]

            scene = rd.Scene(trace_backend="cuda")
            scene.add_mesh(rd.Mesh(
                cuda.Array3f([v[0] for v in QUAD_VERTS], [v[1] for v in QUAD_VERTS], [v[2] for v in QUAD_VERTS]),
                cuda.Array3i([f[0] for f in QUAD_FACES], [f[1] for f in QUAD_FACES], [f[2] for f in QUAD_FACES]),
            ))
            scene.build()

            res = 40
            xs, ys, on_diagonal = [], [], 0
            for j in range(res):
                for i in range(res):
                    x = (i + 0.5) / res
                    y = (j + 0.5) / res
                    xs.append(x)
                    ys.append(y)
                    if i == j:
                        on_diagonal += 1
            count = res * res
            ray = rd.Ray(cuda.Array3f(xs, ys, [-1.0] * count),
                         cuda.Array3f([0.0] * count, [0.0] * count, [1.0] * count))
            its = scene.intersect(ray)
            valid = [int(bool(v)) for v in list(its.is_valid())]
            occ = [int(bool(v)) for v in list(scene.shadow_test(ray))]
            print(json.dumps({
                "count": count,
                "on_diagonal": on_diagonal,
                "hit_count": sum(valid),
                "occluded_count": sum(occ),
                "all_hit": all(valid),
                "all_occluded": all(occ),
            }))
            """
        )
        # Every in-square ray (all of them here) hits exactly once and is occluded.
        self.assertGreater(data["on_diagonal"], 0)
        self.assertEqual(data["hit_count"], data["count"], "watertight gap in closest-hit")
        self.assertEqual(data["occluded_count"], data["count"], "watertight gap in occlusion")
        self.assertTrue(data["all_hit"])
        self.assertTrue(data["all_occluded"])

    def test_exact_diagonal_rays_hit_once(self):
        data = _run_json(
            """
            import json
            import drjit.cuda as cuda
            import rayd.drjit as rd

            QUAD_VERTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            QUAD_FACES = [[0, 1, 2], [0, 2, 3]]
            scene = rd.Scene(trace_backend="cuda")
            scene.add_mesh(rd.Mesh(
                cuda.Array3f([v[0] for v in QUAD_VERTS], [v[1] for v in QUAD_VERTS], [v[2] for v in QUAD_VERTS]),
                cuda.Array3i([f[0] for f in QUAD_FACES], [f[1] for f in QUAD_FACES], [f[2] for f in QUAD_FACES]),
            ))
            scene.build()
            # Rays exactly on the shared diagonal x == y in (0, 1).
            samples = [0.1, 0.25, 0.5, 0.75, 0.9]
            ray = rd.Ray(cuda.Array3f(samples, samples, [-1.0] * len(samples)),
                         cuda.Array3f([0.0] * len(samples), [0.0] * len(samples), [1.0] * len(samples)))
            its = scene.intersect(ray)
            valid = [int(bool(v)) for v in list(its.is_valid())]
            ts = [float(t) for t in list(its.t)]
            print(json.dumps({"valid": valid, "ts": ts}))
            """
        )
        self.assertTrue(all(data["valid"]), "a diagonal ray fell in a watertight gap")
        for t in data["ts"]:
            self.assertTrue(math.isclose(t, 1.0, abs_tol=1e-5), f"diagonal t drifted: {t}")

    def test_degenerate_and_large_coordinate_scenes(self):
        data = _run_json(
            """
            import json, math
            import drjit.cuda as cuda
            import rayd.drjit as rd

            # Zero-area (collinear) triangle must never be hit and never emit NaN.
            deg = rd.Scene(trace_backend="cuda")
            deg.add_mesh(rd.Mesh(
                cuda.Array3f([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            ))
            deg.build()
            deg_its = deg.intersect(rd.Ray(cuda.Array3f([1.0], [0.0], [-1.0]),
                                           cuda.Array3f([0.0], [0.0], [1.0])))
            deg_valid = int(bool(list(deg_its.is_valid())[0]))
            deg_t = float(list(deg_its.t)[0])

            # 1e6 offset triangle: a translated ray still hits.
            big = rd.Scene(trace_backend="cuda")
            big.add_mesh(rd.Mesh(
                cuda.Array3f([1000000.0, 1000001.0, 1000000.0],
                             [1000000.0, 1000000.0, 1000001.0],
                             [1000000.0, 1000000.0, 1000000.0]),
                cuda.Array3i([0], [1], [2]),
            ))
            big.build()
            big_its = big.intersect(rd.Ray(cuda.Array3f([1000000.25], [1000000.25], [999999.0]),
                                           cuda.Array3f([0.0], [0.0], [1.0])))
            print(json.dumps({
                "deg_valid": deg_valid,
                "deg_t_nan": int(math.isnan(deg_t)),
                "big_valid": int(bool(list(big_its.is_valid())[0])),
            }))
            """
        )
        self.assertEqual(data["deg_valid"], 0, "zero-area triangle was hit")
        self.assertEqual(data["deg_t_nan"], 0, "degenerate miss produced NaN")
        self.assertEqual(data["big_valid"], 1, "large-coordinate hit was lost")


class CudaAdParityTests(unittest.TestCase):
    """Reverse-mode gradients match the OptiX backend (same winner => same AD)."""

    _AD_SCRIPT = """
    import json
    import drjit as dr
    import drjit.cuda as cuda
    import drjit.cuda.ad as ad
    import rayd.drjit as rd

    TB = {trace_backend!r}

    def vertex_gradients():
        mesh = rd.Mesh(
            cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            cuda.Array3i([0], [1], [2]),
        )
        verts = ad.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        dr.enable_grad(verts)
        mesh.vertex_positions = verts
        scene = rd.Scene(trace_backend=TB)
        scene.add_mesh(mesh)
        scene.build()
        ray = rd.RayAD(ad.Array3f([0.25], [0.25], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
        its = scene.intersect(ray)
        dr.backward(its.t)
        grad = dr.grad(verts)
        return {{"t": float(list(its.t)[0]),
                "grad": [[float(grad[c][r]) for c in range(3)] for r in range(dr.width(grad))]}}

    def transform_gradients():
        mesh = rd.Mesh(
            cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            cuda.Array3i([0], [1], [2]),
        )
        tz = ad.Float([0.0])
        dr.enable_grad(tz)
        mesh.to_world_left = ad.Matrix4f([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene = rd.Scene(trace_backend=TB)
        scene.add_mesh(mesh)
        scene.build()
        ray = rd.RayAD(ad.Array3f([0.25], [0.25], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
        its = scene.intersect(ray)
        dr.backward(its.t)
        return {{"t": float(list(its.t)[0]), "grad_tz": float(list(dr.grad(tz))[0])}}

    print(json.dumps({{"vertex": vertex_gradients(), "transform": transform_gradients()}}))
    """

    def test_gradients_match_optix(self):
        cuda_data = _run_json(self._AD_SCRIPT.format(trace_backend="cuda"))
        optix_data = _run_json(self._AD_SCRIPT.format(trace_backend="optix"))
        self.assertTrue(math.isclose(cuda_data["vertex"]["t"], optix_data["vertex"]["t"], abs_tol=1e-6))
        for cuda_row, optix_row in zip(cuda_data["vertex"]["grad"], optix_data["vertex"]["grad"]):
            for cuda_v, optix_v in zip(cuda_row, optix_row):
                self.assertTrue(math.isclose(cuda_v, optix_v, rel_tol=1e-5, abs_tol=1e-6),
                                f"vertex gradient drift: {cuda_v} vs {optix_v}")
        self.assertTrue(math.isclose(cuda_data["transform"]["grad_tz"],
                                     optix_data["transform"]["grad_tz"], rel_tol=1e-5, abs_tol=1e-6),
                        "transform gradient drift")


class CudaRefitTests(unittest.TestCase):
    def test_dynamic_vertex_update_refits(self):
        data = _run_json(
            """
            import json
            import drjit.cuda as cuda
            import rayd.drjit as rd

            scene = rd.Scene(trace_backend="cuda")
            scene.add_mesh(rd.Mesh(
                cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            ), dynamic=True)
            scene.build()

            def hit(ox, oy):
                its = scene.intersect(rd.Ray(cuda.Array3f([ox], [oy], [-1.0]),
                                             cuda.Array3f([0.0], [0.0], [1.0])))
                return int(bool(list(its.is_valid())[0]))

            pre = hit(0.25, 0.25)
            scene.update_mesh_vertices(0, cuda.Array3f([2.0, 3.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]))
            scene.sync()
            post_same = hit(0.25, 0.25)
            post_shifted = hit(2.25, 0.25)
            print(json.dumps({"pre": pre, "post_same": post_same, "post_shifted": post_shifted}))
            """
        )
        self.assertEqual(data["pre"], 1)
        self.assertEqual(data["post_same"], 0, "refit did not move the triangle away")
        self.assertEqual(data["post_shifted"], 1, "refit did not track the moved triangle")


class CudaLaunchAuditTests(unittest.TestCase):
    def test_intersect_is_a_single_kernel_and_stable(self):
        data = _run_json(
            """
            import json
            import drjit.cuda as cuda
            import rayd.drjit as rd

            scene = rd.Scene(trace_backend="cuda")
            scene.add_mesh(rd.Mesh(
                cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            ))
            scene.build()
            ray = rd.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))

            # Warm up (first query may prime the Dr.Jit malloc pool).
            scene.intersect(ray)

            rd.native_launch_audit_clear()
            scene.intersect(ray)
            one = rd.native_launch_audit()["intersect"]
            rd.native_launch_audit_clear()
            for _ in range(50):
                scene.intersect(ray)
            many = rd.native_launch_audit()["intersect"]
            labels = sorted(k["label"] for k in one["kernels"])
            print(json.dumps({
                "one_launches": one["cuda_kernel_launches"],
                "many_launches": many["cuda_kernel_launches"],
                "labels": labels,
            }))
            """
        )
        # Exactly one traversal kernel per query, no overflow-repair launches.
        self.assertEqual(data["one_launches"], 1, "closest-hit query is not a single kernel")
        self.assertEqual(data["many_launches"], 50, "per-query launch count is not stable")
        self.assertIn("triangle_closest_hit_kernel", data["labels"])


class CudaRecordingGuardTests(unittest.TestCase):
    def test_intersect_inside_recording_raises(self):
        data = _run_json(
            """
            import json
            import drjit as dr
            import drjit.cuda as cuda
            import rayd.drjit as rd

            scene = rd.Scene(trace_backend="cuda")
            scene.add_mesh(rd.Mesh(
                cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            ))
            scene.build()
            ray = rd.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
            out = {}
            try:
                with dr.scoped_set_flag(dr.JitFlag.Recording, True):
                    scene.intersect(ray)
                out = {"raised": False}
            except Exception as exc:  # noqa: BLE001
                out = {"raised": True, "msg": str(exc)}
            print(json.dumps(out))
            """
        )
        self.assertTrue(data["raised"], "recording guard did not fire")
        self.assertIn("recording", data["msg"].lower())


class CudaFirstBlockerTests(unittest.TestCase):
    def test_first_blocker_selftest_with_ignore(self):
        data = _run_json(
            """
            import json
            import drjit.cuda as cuda
            import rayd.drjit as rd

            # Two parallel triangles: prim 0 at z=0, prim 1 at z=0.5.
            scene = rd.Scene(trace_backend="cuda")
            scene.add_mesh(rd.Mesh(
                cuda.Array3f([0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]),
                cuda.Array3i([0, 3], [1, 4], [2, 5]),
            ))
            scene.build()

            origin = cuda.Array3f([0.25], [0.25], [-1.0])
            direction = cuda.Array3f([0.0], [0.0], [1.0])
            tmax = cuda.Float([1.0e30])
            closest = scene._cuda_first_blocker_selftest(origin, direction, tmax)
            ignore_first = scene._cuda_first_blocker_selftest(origin, direction, tmax, [0])
            ignore_both = scene._cuda_first_blocker_selftest(origin, direction, tmax, [0, 1])
            print(json.dumps({
                "closest": [int(v) for v in closest],
                "ignore_first": [int(v) for v in ignore_first],
                "ignore_both": [int(v) for v in ignore_both],
            }))
            """
        )
        self.assertEqual(data["closest"], [0], "closest blocker should be the nearer triangle")
        self.assertEqual(data["ignore_first"], [1], "ignoring prim 0 should reveal prim 1")
        self.assertEqual(data["ignore_both"], [-1], "ignoring both should report no blocker")


if __name__ == "__main__":
    unittest.main()
