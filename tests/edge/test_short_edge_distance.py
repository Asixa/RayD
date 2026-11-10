# Copyright Xingyu Chen.
# Checks nearest-edge queries on edges far shorter than the scene against a float64 brute force on every backend.

"""Edges of length 1e-5 to 1e-3 must keep their extent in every nearest-edge backend of both frontends.

The edge-distance kernels and the Dr.Jit closest-point recompute treat a segment as a point only below a 1e-18
squared-length floor and decide whether two lines are parallel from a test relative to their lengths
(``contracts/operations.json`` ``numeric_policy.shared_multipath.edge_degeneracy_rule``). The former absolute
1e-7 threshold on squared lengths turned every edge shorter
than about 3e-4 into its start point and dropped the interior solution of a ray passing within an edge
length of a short edge, which gave errors of the order of the edge length. These cases put queries within
about one edge length of short edges and compare distances with an exact float64 minimisation, then check
the fixed-winner point gradient on a single short edge.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.support.backends import drjit_modules, torch_cuda_available

TORCH_BACKENDS = ("cuda", "optix")
DRJIT_BACKENDS = ("drjit", "optix", "optix_drjit")
LENGTHS = (1.0e-5, 1.0e-4, 1.0e-3)
# float32 rounding of the query geometry at unit coordinates, plus the conditioning of the interior solution for a
# ray that crosses a short edge at a shallow angle; the absolute threshold this replaces gave errors up to
# about half the edge length (8.9e-5 for points and 4.6e-5 for rays at 1e-4).
POINT_TOLERANCE = 2.0e-7
RAY_TOLERANCE = 3.0e-6


def short_edge_soup(length: float, rng: np.random.Generator, count: int = 64):
    centers = rng.uniform(-1.0, 1.0, (count, 3))
    vertices = []
    for center in centers:
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        b = rng.normal(size=3)
        b -= a * (a @ b)
        b /= np.linalg.norm(b)
        vertices += [center, center + length * a, center + length * (0.5 * a + 0.8 * b)]
    return np.asarray(vertices, np.float32), np.arange(3 * count, dtype=np.int32).reshape(count, 3)


def near_queries(vertices: np.ndarray, faces: np.ndarray, length: float, rng: np.random.Generator, count: int = 256):
    """Points and rays that pass within one edge length of an interior point of a random edge."""
    v = vertices.astype(np.float64)
    face = rng.integers(0, len(faces), count)
    corner = rng.integers(0, 3, count)
    start = v[faces[face, corner]]
    end = v[faces[face, (corner + 1) % 3]]
    on_edge = start + rng.uniform(0.1, 0.9, count)[:, None] * (end - start)
    offset = rng.normal(size=(count, 3))
    offset /= np.linalg.norm(offset, axis=1, keepdims=True)
    target = on_edge + length * rng.uniform(0.0, 1.0, count)[:, None] * offset
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origin = target - rng.uniform(0.5, 1.0, count)[:, None] * direction
    tmax = np.where(np.arange(count) % 2 == 0, 2.0, np.inf).astype(np.float32)
    return target.astype(np.float32), origin.astype(np.float32), direction.astype(np.float32), tmax


def mesh_edges(vertices: np.ndarray, faces: np.ndarray):
    pairs = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)
    v = vertices.astype(np.float64)
    return v[pairs[:, 0]], v[pairs[:, 1]] - v[pairs[:, 0]]


def _dot(x, y):
    return (x * y).sum(-1)


def exact_point_distance(points, starts, vectors):
    w = points.astype(np.float64)[:, None, :] - starts[None]
    t = np.clip(_dot(w, vectors[None]) / _dot(vectors, vectors)[None], 0.0, 1.0)
    return np.linalg.norm(w - t[..., None] * vectors[None], axis=-1).min(1)


def exact_ray_distance(origins, directions, tmax, starts, vectors):
    """Minimum over edges of the exact distance between o + s u (s in [0, 1], or [0, inf) for an infinite tmax)
    and p + t e (t in [0, 1]): the interior stationary point and the minimum along every boundary of the box."""
    finite = np.isfinite(tmax)[:, None]
    o = origins.astype(np.float64)[:, None, :]
    u = (directions.astype(np.float64) * np.where(np.isfinite(tmax), tmax, 1.0)[:, None])[:, None, :]
    p, e = starts[None], vectors[None]
    s_max = np.where(finite, 1.0, np.inf)
    a, b, c = _dot(u, u), _dot(u, e), _dot(e, e)
    w = o - p
    d, f = _dot(u, w), _dot(e, w)

    def distance(s, t):
        return np.linalg.norm(o + s[..., None] * u - p - t[..., None] * e, axis=-1)

    zero = np.zeros(a.shape[:1] + c.shape[1:])
    best = distance(zero, np.clip(f / c, 0.0, 1.0))
    best = np.minimum(best, distance(np.minimum(np.maximum(-d / a, 0.0), s_max), zero))
    best = np.minimum(best, distance(np.minimum(np.maximum((b - d) / a, 0.0), s_max), zero + 1.0))
    end_t = np.clip(_dot(o + u - p, e) / c, 0.0, 1.0)
    best = np.where(finite, np.minimum(best, distance(zero + 1.0, end_t)), best)
    det = a * c - b * b
    safe = np.where(det > 1e-30 * a * c, det, 1.0)
    s, t = (b * f - c * d) / safe, (a * f - b * d) / safe
    inside = (det > 1e-30 * a * c) & (s >= 0.0) & (s <= s_max) & (t >= 0.0) & (t <= 1.0)
    return np.where(inside, np.minimum(best, distance(s, t)), best).min(1)


def cases():
    rng = np.random.default_rng(11)
    for length in LENGTHS:
        vertices, faces = short_edge_soup(length, rng)
        points, origins, directions, tmax = near_queries(vertices, faces, length, rng)
        starts, vectors = mesh_edges(vertices, faces)
        yield (
            length,
            vertices,
            faces,
            (points, origins, directions, tmax),
            exact_point_distance(points, starts, vectors),
            exact_ray_distance(origins, directions, tmax, starts, vectors),
        )


# A single edge of length 1e-4 along x, and a point 3e-5 above its midpoint: the nearest point is the midpoint,
# so d(distance)/d(point) is the unit vector +y. Treating the edge as its start point tilts it by about 59 degrees.
SINGLE_EDGE_VERTICES = np.array([[0.3, 0.2, 0.1], [0.3001, 0.2, 0.1], [0.30005, 0.2, 0.10008]], np.float32)
SINGLE_EDGE_POINT = np.array([[0.30005, 0.20003, 0.1]], np.float32)


@unittest.skipUnless(torch_cuda_available(), "requires CUDA Torch and rayd.torch")
class TorchShortEdgeDistanceTests(unittest.TestCase):
    def test_short_edges_match_float64_brute_force(self) -> None:
        import torch

        import rayd.torch as rt

        for length, vertices, faces, (points, origins, directions, tmax), point_ref, ray_ref in cases():
            for backend in TORCH_BACKENDS:
                with self.subTest(length=length, backend=backend):
                    scene = rt.Scene(edge_bvh_backend=backend)
                    scene.add_mesh(rt.Mesh(torch.tensor(vertices, device="cuda"), torch.tensor(faces, device="cuda")))
                    scene.build()
                    point = scene.nearest_edge(torch.tensor(points, device="cuda")).distance.cpu().numpy()
                    ray = rt.Ray(
                        torch.tensor(origins, device="cuda"),
                        torch.tensor(directions, device="cuda"),
                        torch.tensor(tmax, device="cuda"),
                    )
                    ray_distance = scene.nearest_edge(ray).distance.cpu().numpy()
                    self.assertLessEqual(np.abs(point - point_ref).max(), POINT_TOLERANCE)
                    self.assertLessEqual(np.abs(ray_distance - ray_ref).max(), RAY_TOLERANCE)

    def test_point_gradient_on_a_short_edge_is_perpendicular_to_it(self) -> None:
        import torch

        import rayd.torch as rt

        for backend in TORCH_BACKENDS:
            with self.subTest(backend=backend):
                scene = rt.Scene(edge_bvh_backend=backend)
                faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
                scene.add_mesh(rt.Mesh(torch.tensor(SINGLE_EDGE_VERTICES, device="cuda"), faces))
                scene.build()
                point = torch.tensor(SINGLE_EDGE_POINT, device="cuda", requires_grad=True)
                result = scene.nearest_edge(point)
                result.distance.sum().backward()
                self.assertAlmostEqual(float(result.edge_t.detach()[0]), 0.5, delta=1e-3)
                np.testing.assert_allclose(point.grad.cpu().numpy()[0], [0.0, 1.0, 0.0], atol=2e-3)


@unittest.skipUnless(drjit_modules() is not None, "requires a CUDA-capable rayd.drjit")
class DrJitShortEdgeDistanceTests(unittest.TestCase):
    def test_short_edges_match_float64_brute_force(self) -> None:
        mods = drjit_modules()
        pj, cuda = mods.pj, mods.cuda
        for length, vertices, faces, (points, origins, directions, tmax), point_ref, ray_ref in cases():
            for backend in DRJIT_BACKENDS:
                with self.subTest(length=length, backend=backend):
                    scene = pj.Scene(edge_bvh_backend=backend)
                    scene.add_mesh(pj.Mesh(cuda.Array3f(*vertices.T), cuda.Array3i(*faces.T)))
                    scene.build()
                    point = np.array(scene.nearest_edge(cuda.Array3f(*points.T)).distance)
                    ray = pj.Ray(cuda.Array3f(*origins.T), cuda.Array3f(*directions.T))
                    ray.tmax = cuda.Float(tmax)
                    ray_distance = np.array(scene.nearest_edge(ray).distance)
                    self.assertLessEqual(np.abs(point - point_ref).max(), POINT_TOLERANCE)
                    self.assertLessEqual(np.abs(ray_distance - ray_ref).max(), RAY_TOLERANCE)

    def test_point_gradient_on_a_short_edge_is_perpendicular_to_it(self) -> None:
        mods = drjit_modules()
        pj, cuda, ad, dr = mods.pj, mods.cuda, mods.ad, mods.dr
        for backend in DRJIT_BACKENDS:
            with self.subTest(backend=backend):
                scene = pj.Scene(edge_bvh_backend=backend)
                scene.add_mesh(pj.Mesh(cuda.Array3f(*SINGLE_EDGE_VERTICES.T), cuda.Array3i(0, 1, 2)))
                scene.build()
                point = ad.Array3f(*SINGLE_EDGE_POINT.T)
                dr.enable_grad(point)
                result = scene.nearest_edge(point)
                dr.backward(result.distance)
                self.assertAlmostEqual(float(result.edge_t[0]), 0.5, delta=1e-3)
                grad = np.array([float(component[0]) for component in dr.grad(point)])
                np.testing.assert_allclose(grad, [0.0, 1.0, 0.0], atol=2e-3)


if __name__ == "__main__":
    unittest.main()
