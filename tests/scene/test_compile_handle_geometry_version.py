# Copyright Xingyu Chen.
# Tests that the torch.compile handle intersect path refuses a backward after sync() changed the scene geometry.

import unittest

import torch

if torch.cuda.is_available():
    import rayd.torch as rt


def _dynamic_scene():
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    vertices = base.clone().requires_grad_()
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
    scene.build()
    return scene, vertices, base


def _ray():
    return torch.tensor([[0.25, 0.25, 1.0]], device="cuda"), torch.tensor([[0.0, 0.0, -1.0]], device="cuda")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class CompileHandleGeometryVersionTests(unittest.TestCase):
    def test_handle_op_backward_after_sync_is_refused(self):
        scene, vertices, base = _dynamic_scene()
        origin, direction = _ray()
        tmax = torch.empty(0, device="cuda")
        t, _prim, version = torch.ops.rayd_torch.intersect_forward_tape_h(
            scene._native_handle, vertices, origin, direction, tmax
        )
        self.assertEqual(version.device.type, "cpu")
        self.assertEqual(int(version), int(scene._native_scene.geometry_version()))
        scene.update_mesh_vertices(0, base + 0.1)
        scene.sync()
        with self.assertRaisesRegex(RuntimeError, "scene geometry changed"):
            t.sum().backward()

    def test_compiled_backward_tracks_the_version_of_each_forward(self):
        scene, vertices, base = _dynamic_scene()
        origin, direction = _ray()

        def depth(o, d):
            return scene.intersect(rt.Ray(o, d), None, rt.RayFlags(0)).t

        torch._dynamo.reset()
        compiled = torch.compile(depth, backend="aot_eager")
        t = compiled(origin, direction)
        self.assertIsNotNone(t.grad_fn)
        t.sum().backward()
        expected = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, -0.25], [0.0, 0.0, -0.25]], device="cuda")
        torch.testing.assert_close(vertices.grad, expected)
        vertices.grad = None

        stale = compiled(origin, direction)
        moved = (base + torch.tensor([0.1, 0.1, 0.0], device="cuda")).requires_grad_()
        scene.update_mesh_vertices(0, moved)
        scene.sync()
        with self.assertRaisesRegex(RuntimeError, "scene geometry changed"):
            stale.sum().backward()

        # A forward recorded after the sync differentiates normally: the version is not baked into the graph.
        fresh = compiled(origin, direction)
        fresh.sum().backward()
        expected_moved = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, -0.15], [0.0, 0.0, -0.15]], device="cuda")
        torch.testing.assert_close(moved.grad, expected_moved)


if __name__ == "__main__":
    unittest.main()
