# Copyright Xingyu Chen.
# Tests the default nearest-edge backend and the lazily derived edge acceleration structures.

import os
import subprocess
import sys
import textwrap
import unittest

import torch

if torch.cuda.is_available():
    import rayd.torch as rt


def _grid(n, noise=0.0, seed=0):
    xs = torch.linspace(0.0, 1.0, n + 1, device="cuda")
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    vertices = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros((n + 1) * (n + 1), device="cuda")], 1)
    if noise:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        vertices = vertices + noise * torch.randn(vertices.shape, device="cuda", generator=generator)
    i = torch.arange(n, device="cuda")
    ii, jj = torch.meshgrid(i, i, indexing="ij")
    a = (ii * (n + 1) + jj).reshape(-1)
    b, c = a + 1, a + n + 1
    faces = torch.cat([torch.stack([a, c, b], 1), torch.stack([b, c, c + 1], 1)]).int()
    return vertices.contiguous(), faces.contiguous()


def _queries(count, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    points = torch.rand(count, 3, device="cuda", generator=generator)
    points[:, 2] = (points[:, 2] - 0.5) * 0.2
    origin = torch.rand(count, 3, device="cuda", generator=generator)
    origin[:, 2] = 0.3
    direction = torch.randn(count, 3, device="cuda", generator=generator) * 0.3
    direction[:, 2] = -1.0
    direction = direction / direction.norm(dim=1, keepdim=True)
    return points, rt.Ray(origin, direction, torch.ones(count, device="cuda"))


def query_count(query):
    return int(query.o.shape[0]) if isinstance(query, rt.Ray) else int(query.shape[0])


def _scene(vertices, faces, backend, dynamic=False):
    scene = rt.Scene(edge_bvh_backend=backend)
    scene.add_mesh(rt.Mesh(vertices, faces), dynamic=dynamic)
    scene.build()
    return scene


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class EdgeBackendSelectionTests(unittest.TestCase):
    def test_auto_selects_the_cuda_edge_bvh_and_explicit_optix_is_kept(self):
        vertices, faces = _grid(4)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        self.assertEqual(scene.edge_bvh_backend, "cuda")
        if scene.trace_backend == "optix":
            self.assertEqual(_scene(vertices, faces, "optix").edge_bvh_backend, "optix")

    def test_backends_agree_on_point_and_ray_distances(self):
        vertices, faces = _grid(24, noise=0.004)
        points, ray = _queries(4096, seed=3)
        cuda = _scene(vertices, faces, "cuda")
        if cuda.trace_backend != "optix":
            self.skipTest("OptiX is unavailable")
        optix = _scene(vertices, faces, "optix")
        for query in (points, ray):
            expected, actual = optix.nearest_edge(query), cuda.nearest_edge(query)
            torch.testing.assert_close(actual.distance, expected.distance, atol=2e-7, rtol=0.0)
            # Different winners are allowed only for edges that tie near a shared vertex: the two closest points must
            # coincide up to float32 rounding of the noisy grid, and such ties must stay rare.
            differs = actual.global_edge_id != expected.global_edge_id
            torch.testing.assert_close(actual.edge_point[differs], expected.edge_point[differs], atol=1e-5, rtol=0.0)
            self.assertLess(int(differs.sum()), query_count(query) // 20)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class LazyEdgeAccelerationTests(unittest.TestCase):
    def _assert_same(self, actual, expected):
        self.assertTrue(torch.equal(actual.distance, expected.distance))
        self.assertTrue(torch.equal(actual.global_edge_id, expected.global_edge_id))
        self.assertTrue(torch.equal(actual.edge_point, expected.edge_point))

    def test_first_edge_query_after_sync_sees_the_committed_geometry(self):
        vertices, faces = _grid(32)
        moved, _ = _grid(32, noise=0.01, seed=7)
        points, ray = _queries(2048, seed=5)
        backends = ["cuda"]
        if _scene(vertices, faces, "cuda").trace_backend == "optix":
            backends.append("optix")
        for backend in backends:
            with self.subTest(backend=backend):
                scene = _scene(vertices.clone(), faces, backend, dynamic=True)
                scene.nearest_edge(points)
                # Several commits without an edge query in between, then one query.
                for step in range(3):
                    scene.update_mesh_vertices(0, vertices + 0.003 * (step + 1))
                    scene.sync()
                    scene.intersect(ray)
                scene.update_mesh_vertices(0, moved)
                scene.sync()
                reference = _scene(moved.clone(), faces, backend, dynamic=True)
                for query in (points, ray):
                    self._assert_same(scene.nearest_edge(query), reference.nearest_edge(query))
                top, top_reference = scene.nearest_edges(points, 3), reference.nearest_edges(points, 3)
                self.assertTrue(torch.equal(top.distances, top_reference.distances))

    def test_edge_mask_and_empty_queries_before_any_edge_structure_exists(self):
        vertices, faces = _grid(8)
        scene = _scene(vertices, faces, "cuda", dynamic=True)
        mask = torch.zeros_like(scene.edge_mask())
        mask[::2] = True
        scene.set_edge_mask(mask)
        empty = scene.nearest_edge(torch.empty(0, 3, device="cuda"))
        self.assertEqual(empty.distance.numel(), 0)
        hits = scene.nearest_edge(_queries(256, seed=1)[0])
        self.assertTrue(bool(mask[hits.global_edge_id.long()].all()))

    def test_concurrent_readers_build_the_edge_structures_once_per_commit(self):
        script = textwrap.dedent(
            """
            import sys, threading
            import torch
            import rayd.torch as rt
            sys.path.insert(0, sys.argv[1])
            from test_edge_acceleration_lifecycle import _grid, _queries

            backend = sys.argv[2]
            vertices, faces = _grid(64)
            scene = rt.Scene(edge_bvh_backend=backend)
            scene.add_mesh(rt.Mesh(vertices.clone(), faces), dynamic=True)
            scene.build()
            points, _ = _queries(8192, seed=11)
            native = scene._native_scene
            for round_index in range(6):
                moved, _ = _grid(64, noise=0.005, seed=round_index)
                scene.update_mesh_vertices(0, moved)
                scene.sync()
                torch.cuda.synchronize()
                barrier = threading.Barrier(4)
                results = [None] * 4

                def worker(k):
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        barrier.wait()
                        out = torch.ops.rayd_torch.nearest_edge_forward_noad(native, points)
                        stream.synchronize()
                        results[k] = (out[0].clone(), out[5].clone())

                threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                reference = rt.Scene(edge_bvh_backend=backend)
                reference.add_mesh(rt.Mesh(moved.clone(), faces), dynamic=True)
                reference.build()
                expected = reference.nearest_edge(points)
                for distance, edge in results:
                    assert torch.equal(distance, expected.distance), round_index
                    assert torch.equal(edge, expected.global_edge_id), round_index
            print("OK")
            """
        )
        backends = ["cuda"]
        if _scene(*_grid(2), "cuda").trace_backend == "optix":
            backends.append("optix")
        here = os.path.dirname(os.path.abspath(__file__))
        for backend in backends:
            with self.subTest(backend=backend):
                result = subprocess.run(
                    [sys.executable, "-c", script, here, backend], capture_output=True, text=True, timeout=600
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
