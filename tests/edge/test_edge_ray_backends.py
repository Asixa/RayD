# Copyright Xingyu Chen.
# Tests that Torch nearest_edge(ray) answers do not depend on the edge acceleration backend.

import math
import unittest

import numpy as np
import torch

try:
    import rayd.torch as rt
except ImportError:  # pragma: no cover - exercised only without the Torch backend
    rt = None


EDGE_BACKENDS = ("optix", "cuda")
INF = float("inf")


def _grid(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-1.0, 1.0, n + 1, device="cuda")
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    zz = 0.05 * torch.sin(3.0 * xx) * torch.cos(2.0 * yy)
    vertices = torch.stack((xx, yy, zz), -1).reshape(-1, 3).contiguous()
    index = torch.arange((n + 1) * (n + 1), device="cuda", dtype=torch.int32).reshape(n + 1, n + 1)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = torch.cat((torch.stack((a, b, d), -1).reshape(-1, 3), torch.stack((a, d, c), -1).reshape(-1, 3)))
    return vertices, faces.contiguous()


def _scene(edge_backend: str, meshes) -> "rt.Scene":
    scene = rt.Scene(edge_bvh_backend=edge_backend)
    for vertices, faces in meshes:
        scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


def _segment_distances(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Float64 distances between the segment [p1, q1] and each segment [p2[i], q2[i]] (clamped closest points)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = r @ d1
    b = d2 @ d1
    denom = a * e - b * b
    s = np.where(denom > 1e-30, np.clip((b * f - c * e) / np.where(denom > 1e-30, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / np.where(e > 1e-30, e, 1.0)
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    closest1 = p1[None, :] + s[:, None] * d1[None, :]
    closest2 = p2 + t[:, None] * d2
    return np.linalg.norm(closest1 - closest2, axis=1)


@unittest.skipUnless(rt is not None and torch.cuda.is_available(), "CUDA Torch backend is required")
class EdgeRayBackendTests(unittest.TestCase):
    def test_degenerate_rows_miss_on_every_edge_backend(self):
        # Rows: tmax NaN, +Inf, -Inf, -1, 0, 0.5, 2; zero direction; NaN origin; Inf direction.
        vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        tmax = [math.nan, INF, -INF, -1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 1.0]
        origins = [[0.5, -0.5, 1.0]] * 8 + [[math.nan, -0.5, 1.0], [0.5, -0.5, 1.0]]
        directions = [[0.0, 0.0, -1.0]] * 7 + [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, INF, -1.0]]
        miss = (INF, -1, 0.0, 0.0)
        half_ray = (0.5, 0, 1.0, 0.5)
        expected = [miss, half_ray, miss, miss, miss, (math.sqrt(0.5), 0, 0.5, 0.5), half_ray, miss, miss, miss]
        for edge_backend in EDGE_BACKENDS + ("auto",):
            with self.subTest(edge_backend=edge_backend):
                scene = _scene(edge_backend, [(vertices, faces)])
                ray = rt.Ray(
                    torch.tensor(origins, device="cuda"),
                    torch.tensor(directions, device="cuda"),
                    torch.tensor(tmax, device="cuda"),
                )
                result = scene.nearest_edge(ray)
                rows = zip(
                    result.distance.tolist(),
                    result.global_edge_id.tolist(),
                    result.ray_t.tolist(),
                    result.edge_t.tolist(),
                )
                for row, (actual, wanted) in enumerate(zip(rows, expected)):
                    self.assertEqual(actual[1], wanted[1], f"row {row}")
                    if wanted[1] < 0:
                        self.assertEqual(actual, wanted, f"row {row}")
                        self.assertEqual(result.point[row].tolist(), [0.0, 0.0, 0.0])
                        self.assertEqual(result.edge_point[row].tolist(), [0.0, 0.0, 0.0])
                    else:
                        np.testing.assert_allclose(actual, wanted, atol=1e-6, err_msg=f"row {row}")

    def test_equal_distances_keep_the_lowest_global_edge_id(self):
        # Two mirrored triangles; the ray runs along x midway between their facing edges, which are exactly 1 away.
        below = torch.tensor([[0.0, -1.0, 0.0], [2.0, -1.0, 0.0], [1.0, -2.0, 0.0]], device="cuda")
        above = torch.tensor([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]], device="cuda")
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        origins = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], device="cuda")
        directions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")
        for order in ((below, above), (above, below)):
            winners = {}
            for edge_backend in EDGE_BACKENDS:
                scene = _scene(edge_backend, [(order[0], faces), (order[1], faces)])
                result = scene.nearest_edge(rt.Ray(origins, directions, torch.tensor([INF, 3.0], device="cuda")))
                self.assertEqual(result.distance.tolist(), [1.0, 1.0])
                self.assertEqual(result.shape_id.tolist(), [0, 0])
                winners[edge_backend] = result.global_edge_id.tolist()
            self.assertEqual(winners["optix"], winners["cuda"])

    def test_random_rays_agree_with_cuda_backend_and_float64_reference(self):
        vertices, faces = _grid(24)
        generator = torch.Generator(device="cuda").manual_seed(11)
        count = 4096
        scale = torch.tensor([2.4, 2.4, 1.0], device="cuda")
        shift = torch.tensor([1.2, 1.2, -0.2], device="cuda")
        origins = torch.rand((count, 3), device="cuda", generator=generator) * scale - shift
        spread = torch.tensor([0.4, 0.4, 1.0], device="cuda")
        directions = torch.randn((count, 3), device="cuda", generator=generator) * spread
        directions = torch.nn.functional.normalize(directions - torch.tensor([0.0, 0.0, 1.0], device="cuda"), dim=1)
        tmax = torch.rand((count,), device="cuda", generator=generator) * 2.0 + 0.2
        tmax[: count // 2] = INF
        results = {}
        for edge_backend in EDGE_BACKENDS:
            scene = _scene(edge_backend, [(vertices, faces)])
            results[edge_backend] = scene.nearest_edge(rt.Ray(origins, directions, tmax))
        optix, cuda = results["optix"], results["cuda"]
        self.assertTrue(bool((optix.global_edge_id >= 0).all()))
        torch.testing.assert_close(optix.distance, cuda.distance, atol=2e-6, rtol=1e-5)
        same = optix.global_edge_id == cuda.global_edge_id
        # Distinct winners are allowed only for float near-ties, which the distance check above already bounds.
        self.assertGreater(float(same.float().mean()), 0.99)

        host_vertices = vertices.cpu().double().numpy()
        unique = {tuple(sorted((int(f[i]), int(f[(i + 1) % 3])))) for f in faces.cpu().numpy() for i in range(3)}
        edge_array = np.array(sorted(unique))
        starts, ends = host_vertices[edge_array[:, 0]], host_vertices[edge_array[:, 1]]
        host_origins = origins.cpu().double().numpy()
        host_directions = directions.cpu().double().numpy()
        # Every edge lies within 4 of every origin, so a 10-long segment stands in for the half-ray.
        reach = np.where(np.isinf(tmax.cpu().double().numpy()), 10.0, tmax.cpu().double().numpy())
        for row in np.random.default_rng(3).choice(count, 64, replace=False):
            origin = host_origins[row]
            reference = _segment_distances(origin, origin + reach[row] * host_directions[row], starts, ends).min()
            self.assertAlmostEqual(float(optix.distance[row]), float(reference), delta=1e-5, msg=f"row {row}")


if __name__ == "__main__":
    unittest.main()
