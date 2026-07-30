# Copyright Xingyu Chen.
# Tests Torch mesh instancing, identifiers, updates, and transform derivatives.

import unittest

import torch
import rayd.torch as rt


def _triangle() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    return vertices, faces


def _transform(x: float = 0.0, z: float = 0.0, *, requires_grad: bool = False) -> torch.Tensor:
    value = torch.tensor(
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    return value.requires_grad_(requires_grad)


def _vertical_rays(xs: list[float]) -> rt.Ray:
    origins = torch.tensor([[x, 0.0, -1.0] for x in xs], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[0.0, 0.0, 1.0] for _ in xs], device="cuda", dtype=torch.float32)
    return rt.Ray(origins, directions)


def _edge_ray() -> rt.Ray:
    return rt.Ray(
        torch.tensor([[3.0, -2.0, -1.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class MeshInstancingTests(unittest.TestCase):
    def test_instance_shares_geometry_and_keeps_independent_ids(self):
        vertices, faces = _triangle()
        scene = rt.Scene()
        geometry_id = scene.add_mesh(rt.Mesh(vertices, faces))
        instance_id = scene.add_instance(geometry_id, _transform(x=3.0))
        scene.build()

        self.assertEqual(geometry_id, 0)
        self.assertEqual(instance_id, 1)
        self.assertEqual(scene.num_meshes, 2)
        self.assertEqual(scene.num_geometries, 1)

        its = scene.intersect(_vertical_rays([0.0, 3.0]))
        torch.testing.assert_close(its.instance_id, its.shape_id)
        torch.testing.assert_close(its.shape_id, torch.tensor([0, 1], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(its.local_prim_id, torch.tensor([0, 0], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(its.global_prim_id, torch.tensor([0, 1], device="cuda", dtype=torch.int32))

    def test_transform_only_update_moves_instance_and_preserves_ids(self):
        vertices, faces = _triangle()
        scene = rt.Scene()
        geometry_id = scene.add_mesh(rt.Mesh(vertices, faces))
        instance_id = scene.add_instance(geometry_id, _transform(x=3.0))
        scene.build()
        version_before = scene.version

        scene.set_instance_transform(instance_id, _transform(x=5.0))
        self.assertTrue(scene.has_pending_updates())
        scene.sync()
        native_scene = scene._require_native_scene()

        its = scene.intersect(_vertical_rays([3.0, 5.0]))
        self.assertTrue(torch.isinf(its.t[0]))
        self.assertEqual(int(its.shape_id[1].item()), instance_id)
        self.assertEqual(int(its.local_prim_id[1].item()), 0)
        self.assertEqual(int(its.global_prim_id[1].item()), 1)
        self.assertEqual(scene.num_geometries, 1)
        self.assertEqual(scene.version, version_before + 1)
        self.assertEqual(native_scene.last_sync_gas_updates(), 0)
        self.assertEqual(native_scene.last_sync_ias_updates(), 1)

    def test_instance_transform_vjp(self):
        vertices, faces = _triangle()
        transform = _transform(x=3.0, z=0.5, requires_grad=True)
        scene = rt.Scene()
        geometry_id = scene.add_mesh(rt.Mesh(vertices, faces))
        scene.add_instance(geometry_id, transform)
        scene.build()

        scene.intersect(_vertical_rays([3.0])).t.sum().backward()

        self.assertIsNotNone(transform.grad)
        torch.testing.assert_close(transform.grad[2, 3], torch.tensor(1.0, device="cuda"), atol=1e-5, rtol=1e-5)

    def test_instance_transform_jvp(self):
        vertices, faces = _triangle()
        transform = _transform(x=3.0, z=0.5)
        tangent = torch.zeros_like(transform)
        tangent[2, 3] = 1.0

        with torch.autograd.forward_ad.dual_level():
            dual_transform = torch.autograd.forward_ad.make_dual(transform, tangent)
            scene = rt.Scene()
            geometry_id = scene.add_mesh(rt.Mesh(vertices, faces))
            scene.add_instance(geometry_id, dual_transform)
            scene.build()
            result = scene.intersect(_vertical_rays([3.0])).t
            primal, jvp = torch.autograd.forward_ad.unpack_dual(result)

        torch.testing.assert_close(primal, torch.tensor([1.5], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(jvp, torch.tensor([1.0], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_instance_transform_nearest_edge_ray_vjp(self):
        vertices, faces = _triangle()
        transform = _transform(x=3.0, requires_grad=True)
        scene = rt.Scene()
        geometry_id = scene.add_mesh(rt.Mesh(vertices, faces))
        scene.add_instance(geometry_id, transform)
        scene.build()

        result = scene.nearest_edge(_edge_ray())
        result.distance.sum().backward()

        self.assertIsNotNone(transform.grad)
        torch.testing.assert_close(result.distance, torch.tensor([1.0], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(transform.grad[1, 3], torch.tensor(1.0, device="cuda"), atol=1e-5, rtol=1e-5)

    def test_instance_transform_nearest_edge_ray_jvp(self):
        vertices, faces = _triangle()
        transform = _transform(x=3.0)
        tangent = torch.zeros_like(transform)
        tangent[1, 3] = 1.0

        with torch.autograd.forward_ad.dual_level():
            dual_transform = torch.autograd.forward_ad.make_dual(transform, tangent)
            scene = rt.Scene()
            geometry_id = scene.add_mesh(rt.Mesh(vertices, faces))
            scene.add_instance(geometry_id, dual_transform)
            scene.build()
            result = scene.nearest_edge(_edge_ray()).distance
            primal, jvp = torch.autograd.forward_ad.unpack_dual(result)

        torch.testing.assert_close(primal, torch.tensor([1.0], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(jvp, torch.tensor([1.0], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_instances_reject_dynamic_or_nested_geometry(self):
        vertices, faces = _triangle()

        dynamic_scene = rt.Scene()
        dynamic_id = dynamic_scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
        with self.assertRaisesRegex(ValueError, "dynamic source geometry"):
            dynamic_scene.add_instance(dynamic_id, _transform())

        nested_scene = rt.Scene()
        owner_id = nested_scene.add_mesh(rt.Mesh(vertices, faces))
        instance_id = nested_scene.add_instance(owner_id, _transform(x=3.0))
        with self.assertRaisesRegex(ValueError, "instance-of-instance"):
            nested_scene.add_instance(instance_id, _transform(x=6.0))

    def test_owner_transform_metadata_is_not_an_instance_transform(self):
        vertices, faces = _triangle()
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces, to_world_left=_transform(x=2.0)))
        scene.build()

        self.assertEqual(scene.num_meshes, 1)
        self.assertEqual(scene.num_geometries, 1)


if __name__ == "__main__":
    unittest.main()
