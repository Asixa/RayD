# Copyright Xingyu Chen.
# Tests that Torch scenes reject NaN and infinite vertex coordinates at build and at sync().

import unittest

import torch

try:
    import rayd.torch as rt
except ImportError:  # pragma: no cover - exercised only without the Torch backend
    rt = None


INF = float("inf")
NAN = float("nan")
QUAD = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
FACES = [[0, 1, 2], [1, 3, 2]]
TRACE_BACKENDS = ("optix", "cuda")


def _tensor(values, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=dtype)


def _bad_quad(value: float) -> list[list[float]]:
    vertices = [list(row) for row in QUAD]
    vertices[3][1] = value
    return vertices


@unittest.skipUnless(rt is not None and torch.cuda.is_available(), "CUDA Torch backend is required")
class NonFiniteVertexTests(unittest.TestCase):
    def test_build_rejects_non_finite_vertices_naming_mesh_and_vertex(self):
        for trace_backend in TRACE_BACKENDS:
            for value in (NAN, INF, -INF):
                with self.subTest(trace_backend=trace_backend, value=value):
                    scene = rt.Scene(trace_backend=trace_backend)
                    scene.add_mesh(rt.Mesh(_tensor(QUAD), _tensor(FACES, torch.int32)))
                    scene.add_mesh(rt.Mesh(_tensor(_bad_quad(value)), _tensor(FACES, torch.int32)))
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"Scene.build\(\): mesh 1 has a NaN or infinite world-space coordinate at vertex 3 ",
                    ):
                        scene.build()

    def test_sync_rejects_non_finite_updates_and_a_finite_retry_commits(self):
        ray = rt.Ray(_tensor([[0.25, 0.25, -1.0]]), _tensor([[0.0, 0.0, 1.0]]))
        for trace_backend in TRACE_BACKENDS:
            for value in (NAN, INF, -INF):
                with self.subTest(trace_backend=trace_backend, value=value):
                    scene = rt.Scene(trace_backend=trace_backend)
                    scene.add_mesh(rt.Mesh(_tensor(QUAD), _tensor(FACES, torch.int32)), dynamic=True)
                    scene.build()
                    scene.update_mesh_vertices(0, _tensor(_bad_quad(value)))
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"Scene.sync\(\): mesh 0 has a NaN or infinite world-space coordinate at vertex 3 ",
                    ):
                        scene.sync()
                    self.assertTrue(scene.has_pending_updates())
                    # A finite update commits normally afterwards.
                    lifted = [[x, y, 0.5] for x, y, _ in QUAD]
                    scene.update_mesh_vertices(0, _tensor(lifted))
                    scene.sync()
                    self.assertEqual(scene.intersect(ray).t.tolist(), [1.5])

    def test_instance_transform_producing_infinite_vertices_is_rejected(self):
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(_tensor(QUAD), _tensor(FACES, torch.int32)))
        transform = torch.eye(4, device="cuda")
        transform[0, 3] = INF
        scene.add_instance(0, transform)
        with self.assertRaisesRegex(RuntimeError, r"mesh 1 has a NaN or infinite world-space coordinate at vertex 0 "):
            scene.build()


if __name__ == "__main__":
    unittest.main()
