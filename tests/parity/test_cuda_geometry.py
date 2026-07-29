# Copyright Xingyu Chen.
# Tests cuda geometry.

import unittest

import torch
import rayd.torch as rt


def _scene(backend: str) -> rt.Scene:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda", dtype=torch.int32)
    scene = rt.Scene(trace_backend=backend, edge_bvh_backend=backend)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class CudaGeometryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda_scene = _scene("cuda")
        try:
            cls.optix_scene = _scene("optix")
        except RuntimeError as error:
            raise unittest.SkipTest(f"OptiX is unavailable: {error}") from error

    def test_intersection_discrete_and_continuous_parity(self) -> None:
        ray = rt.Ray(
            torch.tensor(
                [[0.2, 0.1, -1.0], [0.8, 0.9, -2.0], [1.5, 0.5, -1.0]],
                device="cuda",
            ),
            torch.tensor([[0.0, 0.0, 1.0]] * 3, device="cuda"),
        )
        cuda = self.cuda_scene.intersect(ray)
        optix = self.optix_scene.intersect(ray)
        for name in ("shape_id", "local_prim_id", "global_prim_id"):
            self.assertTrue(torch.equal(getattr(cuda, name), getattr(optix, name)))
        for name in ("t", "p", "n", "geo_n", "barycentric"):
            torch.testing.assert_close(
                getattr(cuda, name), getattr(optix, name), atol=1e-6, rtol=1e-6
            )

    def test_point_ray_and_topk_edge_parity(self) -> None:
        points = torch.tensor(
            [[0.2, -0.17, 0.03], [1.23, 0.7, -0.04], [0.6, 1.19, 0.02]],
            device="cuda",
        )
        cuda_point = self.cuda_scene.nearest_edge(points)
        optix_point = self.optix_scene.nearest_edge(points)
        for name in ("shape_id", "edge_id", "global_edge_id"):
            self.assertTrue(
                torch.equal(getattr(cuda_point, name), getattr(optix_point, name))
            )
        for name in ("distance", "edge_point", "edge_t"):
            torch.testing.assert_close(
                getattr(cuda_point, name), getattr(optix_point, name), atol=1e-6, rtol=1e-6
            )

        ray = rt.Ray(
            torch.tensor([[-0.3, 0.2, 0.1], [0.7, -0.4, -0.1]], device="cuda"),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda"),
            # Stop before the projected ray reaches a second coplanar edge;
            # otherwise both candidates are exactly the same 3-D distance and
            # the nearest-ray contract intentionally does not freeze tie order.
            torch.tensor([0.4, 0.8], device="cuda"),
        )
        cuda_ray = self.cuda_scene.nearest_edge(ray)
        optix_ray = self.optix_scene.nearest_edge(ray)
        for name in ("shape_id", "edge_id", "global_edge_id"):
            self.assertTrue(torch.equal(getattr(cuda_ray, name), getattr(optix_ray, name)))
        for name in ("distance", "ray_t", "point", "edge_t", "edge_point"):
            torch.testing.assert_close(
                getattr(cuda_ray, name), getattr(optix_ray, name), atol=1e-6, rtol=1e-6
            )

        cuda_topk = self.cuda_scene.nearest_edges(points, 3)
        optix_topk = self.optix_scene.nearest_edges(points, 3)
        self.assertTrue(torch.equal(cuda_topk.global_edge_ids, optix_topk.global_edge_ids))
        torch.testing.assert_close(
            cuda_topk.distances, optix_topk.distances, atol=1e-6, rtol=1e-6
        )

    def test_optix_intersection_and_edge_params_are_stream_isolated(self) -> None:
        hit_ray = rt.Ray(
            torch.tensor([[0.2, 0.1, -1.0]], device="cuda"),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        )
        miss_ray = rt.Ray(
            torch.tensor([[1.5, 1.5, -1.0]], device="cuda"),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        )
        point_a = torch.tensor([[0.2, -0.17, 0.03]], device="cuda")
        point_b = torch.tensor([[1.23, 0.7, -0.04]], device="cuda")
        reference_a = self.optix_scene.nearest_edge(point_a)
        reference_b = self.optix_scene.nearest_edge(point_b)
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        # Exceed the former 128-entry intersect-params cache boundary while
        # alternating distinct launch params on independent streams.
        for _ in range(140):
            with torch.cuda.stream(stream_a):
                result_a = self.optix_scene.intersect(hit_ray)
                edge_a = self.optix_scene.nearest_edge(point_a)
            with torch.cuda.stream(stream_b):
                result_b = self.optix_scene.intersect(miss_ray)
                edge_b = self.optix_scene.nearest_edge(point_b)
        stream_a.synchronize()
        stream_b.synchronize()

        self.assertEqual(int(result_a.global_prim_id.item()), 0)
        self.assertEqual(int(result_b.global_prim_id.item()), -1)
        self.assertTrue(torch.equal(edge_a.global_edge_id, reference_a.global_edge_id))
        self.assertTrue(torch.equal(edge_b.global_edge_id, reference_b.global_edge_id))
        torch.testing.assert_close(edge_a.distance, reference_a.distance, atol=0, rtol=0)
        torch.testing.assert_close(edge_b.distance, reference_b.distance, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
