# Copyright Xingyu Chen.
# Tests the Dr.Jit nearest-edge queries served by the native compact-BVH traversal against brute force.

"""The Dr.Jit ``drjit`` edge backend (the default) runs its detached broad phase with the shared native
compact-BVH CUDA traversal. These cases check its winners against a float64 brute force over every edge for
point, finite-ray, infinite-ray, top-k, and masked queries, and pin that a query is a fixed handful of kernel
launches instead of one evaluated Dr.Jit kernel per traversal step."""

import unittest

from tests.support.subprocess_cases import compose, run_json_case, run_script


def drjit_available() -> bool:
    result = run_script("import rayd.drjit as pj; print(int(pj.device_count()))", check=False)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return result.returncode == 0 and bool(lines) and lines[-1].strip() not in ("", "0")


DRJIT_AVAILABLE = drjit_available()

PRELUDE = """
import json
import numpy as np
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as pj

rng = np.random.default_rng(7)
n = 24
xs = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
gx, gy = np.meshgrid(xs, xs, indexing="xy")
gz = rng.uniform(-0.03, 0.03, size=gx.shape).astype(np.float32)
vertices = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])
index = np.arange((n + 1) * (n + 1), dtype=np.int32).reshape(n + 1, n + 1)
a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel()
faces = np.concatenate([np.stack([a, b, d]), np.stack([a, d, c])], axis=1).astype(np.int32)


def make_scene(backend="auto"):
    scene = pj.Scene(edge_bvh_backend=backend)
    scene.add_mesh(pj.Mesh(cuda.Array3f(*vertices), cuda.Array3i(*faces)))
    scene.build()
    return scene


def edges_of(scene):
    info = scene.edge_info()
    return np.array(info.start, dtype=np.float64).T, np.array(info.edge, dtype=np.float64).T


def point_distances(points, p0, e1):
    w = points[:, None, :] - p0[None, :, :]
    t = np.clip((w * e1[None]).sum(-1) / np.maximum((e1 * e1).sum(-1), 1e-300)[None], 0.0, 1.0)
    r = w - t[..., None] * e1[None]
    return np.sqrt((r * r).sum(-1))


def segment_distances(origin, segment, p0, e1, iterations=80):
    # The distance from p0 + s e1 to a segment is convex in s, so a float64 ternary search over s in [0, 1]
    # converges to the exact segment-segment distance far below the test tolerance.
    o = origin[:, None, :]
    seg = segment[:, None, :]
    seg_sq = np.maximum((seg * seg).sum(-1), 1e-300)

    def distance(s):
        w = p0[None] + s[..., None] * e1[None] - o
        u = np.clip((w * seg).sum(-1) / seg_sq, 0.0, 1.0)
        r = w - u[..., None] * seg
        return np.sqrt((r * r).sum(-1))

    lo = np.zeros((origin.shape[0], p0.shape[0]))
    hi = np.ones_like(lo)
    for _ in range(iterations):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        left = distance(m1) <= distance(m2)
        hi = np.where(left, m2, hi)
        lo = np.where(left, lo, m1)
    return np.minimum(distance(0.5 * (lo + hi)), np.minimum(distance(np.zeros_like(lo)), distance(np.ones_like(lo))))
"""


def run_case(body: str):
    return run_json_case(compose(PRELUDE, body), timeout=600)


@unittest.skipUnless(DRJIT_AVAILABLE, "a Dr.Jit CUDA device and an importable rayd.drjit are required")
class NativeBvhQueryTests(unittest.TestCase):
    def test_auto_backend_is_the_native_bvh(self):
        data = run_case('print(json.dumps({"backend": make_scene().edge_bvh_backend}))')
        self.assertEqual(data["backend"], "drjit")

    def test_point_and_topk_winners_match_brute_force(self):
        data = run_case(
            """
            scene = make_scene("drjit")
            p0, e1 = edges_of(scene)
            q = rng.uniform(-0.1, 1.1, size=(3, 600)).astype(np.float32)
            q[2] = rng.uniform(-0.2, 0.2, size=600).astype(np.float32)
            truth = point_distances(q.T.astype(np.float64), p0, e1)
            result = scene.nearest_edge(cuda.Array3f(*q))
            ids = np.array(result.global_edge_id)
            k = 5
            topk = scene.nearest_edges(cuda.Array3f(*q), k)
            topk_ids = np.array(topk.global_edge_ids).reshape(-1, k)
            topk_d = np.array(topk.distances).reshape(-1, k)
            chosen = truth[np.arange(600), ids]
            expected_k = np.sort(truth, axis=1)[:, :k]
            print(json.dumps({
                "all_valid": bool((ids >= 0).all()),
                "point_excess": float((chosen - truth.min(axis=1)).max()),
                "point_reported_error": float(np.abs(np.array(result.distance) - chosen).max()),
                "topk_excess": float(np.abs(truth[np.arange(600)[:, None], topk_ids] - expected_k).max()),
                "topk_sorted": bool((np.diff(topk_d, axis=1) >= 0).all()),
                "topk_unique": bool(all(len(set(row)) == k for row in topk_ids.tolist())),
            }))
            """
        )
        self.assertTrue(data["all_valid"])
        self.assertLess(data["point_excess"], 1e-6)
        self.assertLess(data["point_reported_error"], 1e-5)
        self.assertLess(data["topk_excess"], 1e-6)
        self.assertTrue(data["topk_sorted"])
        self.assertTrue(data["topk_unique"])

    def test_finite_and_infinite_ray_winners_match_brute_force(self):
        data = run_case(
            """
            scene = make_scene("drjit")
            p0, e1 = edges_of(scene)
            count = 300
            o = rng.uniform(0.0, 1.0, size=(3, count)).astype(np.float32)
            o[2] = rng.uniform(0.2, 0.5, size=count).astype(np.float32)
            d = rng.normal(size=(3, count)).astype(np.float32)
            d[2] = -np.abs(d[2]) - 0.3
            d /= np.linalg.norm(d, axis=0, keepdims=True)
            tmax = np.full(count, np.inf, dtype=np.float32)
            tmax[: count // 2] = rng.uniform(0.05, 0.6, size=count // 2).astype(np.float32)
            ray = pj.Ray(cuda.Array3f(*o), cuda.Array3f(*d))
            ray.tmax = cuda.Float(tmax)
            ids = np.array(scene.nearest_edge(ray).global_edge_id)
            # The far end of an infinite ray is past the mesh, so a 10-unit segment has the same nearest edge.
            length = np.where(np.isfinite(tmax), tmax, 10.0).astype(np.float64)
            truth = segment_distances(o.T.astype(np.float64), (d * length).T.astype(np.float64), p0, e1)
            excess = truth[np.arange(count), ids] - truth.min(axis=1)
            print(json.dumps({"all_valid": bool((ids >= 0).all()), "excess": float(excess.max())}))
            """
        )
        self.assertTrue(data["all_valid"])
        self.assertLess(data["excess"], 1e-5)

    def test_short_edge_ray_distances_match_brute_force(self):
        # The AD recompute reports the winner's distance with the shared edge-distance degeneracy threshold, so a
        # millimetre edge is no longer treated as a pair of endpoints.
        data = run_case(
            """
            count, length = 200, 1.0e-3
            base = rng.uniform(0.0, 1.0, size=(count, 3))
            u = rng.normal(size=(count, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            w = rng.normal(size=(count, 3)); w -= u * (u * w).sum(1, keepdims=True)
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            tri = np.stack([base, base + length * u, base + length * (0.5 * u + 0.8 * w)], 1).astype(np.float32)
            scene = pj.Scene(edge_bvh_backend="drjit")
            scene.add_mesh(pj.Mesh(cuda.Array3f(*tri.reshape(-1, 3).T),
                                   cuda.Array3i(*np.arange(3 * count, dtype=np.int32).reshape(-1, 3).T)))
            scene.build()
            p0, e1 = edges_of(scene)
            o = rng.uniform(-0.5, 1.5, size=(3, count)).astype(np.float32)
            target = tri[:, 0] + 2.0e-4 * rng.normal(size=(count, 3))
            d = (target.T - o).astype(np.float64); d /= np.linalg.norm(d, axis=0, keepdims=True)
            d = d.astype(np.float32)
            ray = pj.Ray(cuda.Array3f(*o), cuda.Array3f(*d))
            ray.tmax = cuda.Float(np.full(count, 4.0, dtype=np.float32))
            result = scene.nearest_edge(ray)
            truth = segment_distances(o.T.astype(np.float64), (4.0 * d).T.astype(np.float64), p0, e1)
            ids = np.array(result.global_edge_id)
            reported = np.array(result.distance)
            print(json.dumps({"all_valid": bool((ids >= 0).all()),
                              "error": float(np.abs(reported - truth.min(axis=1)).max())}))
            """
        )
        self.assertTrue(data["all_valid"])
        self.assertLess(data["error"], 2e-5)

    def test_degenerate_ray_rows_miss_on_every_edge_backend(self):
        # contracts/operations.json nearest_edge_ray.degenerate_rows. Rows: tmax NaN, +Inf, -Inf, -1, 0, 0.5, 2;
        # zero direction; NaN origin; Inf direction.
        data = run_case(
            """
            inf, nan = float("inf"), float("nan")
            tmax = [nan, inf, -inf, -1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 1.0]
            origins = [[0.5, -0.5, 1.0]] * 8 + [[nan, -0.5, 1.0], [0.5, -0.5, 1.0]]
            directions = [[0.0, 0.0, -1.0]] * 7 + [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, inf, -1.0]]
            out = {}
            for backend in ("drjit", "optix", "auto"):
                scene = pj.Scene(edge_bvh_backend=backend)
                scene.add_mesh(pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                                       cuda.Array3i([0], [1], [2])))
                scene.build()
                ray = pj.Ray(cuda.Array3f(*np.array(origins, np.float32).T),
                             cuda.Array3f(*np.array(directions, np.float32).T))
                ray.tmax = cuda.Float(np.array(tmax, np.float32))
                r = scene.nearest_edge(ray)
                out[backend] = [[float(a), int(b), float(c)] for a, b, c in
                                zip(np.array(r.distance), np.array(r.global_edge_id), np.array(r.ray_t))]
            print(json.dumps(out))
            """
        )
        miss = [float("inf"), -1, 0.0]
        half_ray = [0.5, 0, 1.0]
        expected = [miss, half_ray, miss, miss, miss, [0.5**0.5, 0, 0.5], half_ray, miss, miss, miss]
        for backend, rows in data.items():
            for row, (actual, wanted) in enumerate(zip(rows, expected)):
                with self.subTest(backend=backend, row=row):
                    self.assertEqual(actual[1], wanted[1])
                    self.assertAlmostEqual(actual[0], wanted[0], places=6)
                    self.assertAlmostEqual(actual[2], wanted[2], places=6)

    def test_optix_ray_backend_agrees_with_brute_force_and_breaks_ties_by_edge_id(self):
        # The OptiX ray programs keep the running best in payload inside the intersection program and clip infinite
        # rays to the edge bounds; equal distances must keep the lower edge id whatever the traversal order.
        data = run_case(
            """
            if not pj.optix_available():
                print(json.dumps({"skip": True}))
                raise SystemExit
            scene = make_scene("optix")
            p0, e1 = edges_of(scene)
            count = 300
            o = rng.uniform(0.0, 1.0, size=(3, count)).astype(np.float32)
            o[2] = rng.uniform(0.2, 0.5, size=count).astype(np.float32)
            d = rng.normal(size=(3, count)).astype(np.float32)
            d[2] = -np.abs(d[2]) - 0.3
            d /= np.linalg.norm(d, axis=0, keepdims=True)
            tmax = np.full(count, np.inf, dtype=np.float32)
            tmax[: count // 2] = rng.uniform(0.05, 0.6, size=count // 2).astype(np.float32)
            ray = pj.Ray(cuda.Array3f(*o), cuda.Array3f(*d))
            ray.tmax = cuda.Float(tmax)
            ids = np.array(scene.nearest_edge(ray).global_edge_id)
            length = np.where(np.isfinite(tmax), tmax, 10.0).astype(np.float64)
            truth = segment_distances(o.T.astype(np.float64), (d * length).T.astype(np.float64), p0, e1)
            excess = truth[np.arange(count), ids] - truth.min(axis=1)

            # Two identical triangles give every edge an exact duplicate with a higher global id.
            twin = pj.Scene(edge_bvh_backend="optix")
            tri = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
            twin.add_mesh(pj.Mesh(tri, cuda.Array3i([0], [1], [2])))
            twin.add_mesh(pj.Mesh(tri, cuda.Array3i([0], [1], [2])))
            twin.build()
            probe = pj.Ray(cuda.Array3f([0.5, 0.25], [-0.5, -0.5], [1.0, 1.0]),
                           cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0]))
            twin_ids = [int(v) for v in np.array(twin.nearest_edge(probe).global_edge_id)]
            edge_count = int(twin.edge_bvh_stats().primitive_count)
            print(json.dumps({"skip": False, "all_valid": bool((ids >= 0).all()), "excess": float(excess.max()),
                              "twin_ids": twin_ids, "half": edge_count // 2}))
            """
        )
        if data["skip"]:
            self.skipTest("OptiX is unavailable")
        self.assertTrue(data["all_valid"])
        self.assertLess(data["excess"], 1e-5)
        for edge_id in data["twin_ids"]:
            self.assertGreaterEqual(edge_id, 0)
            self.assertLess(edge_id, data["half"])

    def test_masked_edges_are_never_returned(self):
        data = run_case(
            """
            scene = make_scene("drjit")
            p0, e1 = edges_of(scene)
            keep = np.arange(p0.shape[0]) % 3 != 0
            scene.set_edge_mask(cuda.Bool(keep))
            scene.sync()
            q = rng.uniform(0.0, 1.0, size=(3, 500)).astype(np.float32)
            truth = point_distances(q.T.astype(np.float64), p0, e1)
            truth[:, ~keep] = np.inf
            ids = np.array(scene.nearest_edge(cuda.Array3f(*q)).global_edge_id)
            topk_ids = np.array(scene.nearest_edges(cuda.Array3f(*q), 3).global_edge_ids)
            print(json.dumps({
                "masked_hits": int((~keep[ids]).sum() + (~keep[topk_ids]).sum()),
                "excess": float((truth[np.arange(500), ids] - truth.min(axis=1)).max()),
            }))
            """
        )
        self.assertEqual(data["masked_hits"], 0)
        self.assertLess(data["excess"], 1e-6)

    def test_query_is_a_fixed_number_of_kernels(self):
        # The former evaluated wavefront traversal launched one Dr.Jit kernel per traversal step.
        data = run_case(
            """
            scene = make_scene("drjit")
            q = cuda.Array3f(*rng.uniform(0.0, 1.0, size=(3, 4096)).astype(np.float32))
            dr.eval(q)
            dr.eval(scene.nearest_edge(q).global_edge_id)
            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                r = scene.nearest_edge(q)
                dr.eval(r.global_edge_id, r.distance)
                kernels = len(dr.kernel_history())
            print(json.dumps({"kernels": kernels}))
            """
        )
        self.assertLessEqual(data["kernels"], 8)


if __name__ == "__main__":
    unittest.main()
