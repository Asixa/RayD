# Copyright Xingyu Chen.
# Tests f1 acceptance.

"""Grouped F1 acceptance coverage for Torch geometry and visibility helpers.

The cross-backend cases are opt-in because they require both native extensions
to be installed in the same environment. Set ``RAYD_TORCH_RUN_DR_JIT_PARITY=1``
for the grouped F1/F2 acceptance run.
"""

import importlib
import os
import unittest

import torch
import rayd.torch as rt


def _triangle_vertices(*, x_offset=0.0, requires_grad=False):
    return torch.tensor(
        [[-1.0 + x_offset, -1.0, 0.0], [1.0 + x_offset, -1.0, 0.0], [0.0 + x_offset, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def _torch_scene(*, two_meshes=False, requires_grad=False):
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    vertices = []
    if two_meshes:
        far = _triangle_vertices(x_offset=10.0, requires_grad=requires_grad)
        scene.add_mesh(rt.Mesh(far, faces))
        vertices.append(far)
    near = _triangle_vertices(requires_grad=requires_grad)
    scene.add_mesh(rt.Mesh(near, faces))
    vertices.append(near)
    scene.build()
    return scene, vertices


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class F1TorchVisibilityAcceptanceTests(unittest.TestCase):
    def test_visible_pair_empty_inactive_degenerate_and_global_ignore(self):
        scene, _ = _torch_scene(two_meshes=True)
        empty = torch.empty((0, 3), device="cuda")
        empty_result = scene.visible_pair(empty, empty, empty)
        self.assertEqual(empty_result.ray_count, 0)
        self.assertEqual(tuple(empty_result.visible_a.shape), (0,))
        self.assertEqual(tuple(empty_result.visible_b.shape), (0,))

        start = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device="cuda")
        end_a = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda")
        end_b = torch.tensor([[3.0, 3.0, 1.0], [3.0, 3.0, 1.0]], device="cuda")
        active = torch.tensor([True, False], device="cuda")
        result = scene.visible_pair(start, end_a, end_b, active=active)
        self.assertEqual(result.visible_a.tolist(), [False, False])
        self.assertEqual(result.visible_b.tolist(), [True, False])

        # Mesh 0 is far away, so the blocker is scene-global primitive 1.
        ignored = scene.visible_pair(
            start[:1], end_a[:1], end_b[:1], ignore_prim_ids=torch.tensor([[1]], device="cuda", dtype=torch.int32)
        )
        self.assertEqual(ignored.visible_a.tolist(), [True])
        self.assertEqual(ignored.visible_b.tolist(), [True])

        same = torch.tensor([[0.0, 0.0, 0.5]], device="cuda")
        degenerate = scene.visible_pair(same, same, same)
        self.assertEqual(degenerate.visible_a.tolist(), [True])
        self.assertEqual(degenerate.visible_b.tolist(), [True])

    def test_visible_edge_empty_inactive_and_degenerate(self):
        scene, _ = _torch_scene()
        empty_vec = torch.empty((0, 3), device="cuda")
        empty_scalar = torch.empty((0,), device="cuda")
        empty = scene.visible_edge(empty_vec, empty_vec, empty_vec, empty_scalar, empty_scalar, sample_fractions=(0.0,))
        self.assertEqual(empty.state_count, 0)
        self.assertEqual(tuple(empty.any_visible.shape), (0,))

        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
        edge_position = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
        edge_direction = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
        edge_t = torch.tensor([0.0], device="cuda")
        inactive = scene.visible_edge(
            source,
            edge_position,
            edge_direction,
            edge_t,
            edge_t,
            sample_fractions=(0.0,),
            active=torch.tensor([False], device="cuda"),
        )
        self.assertEqual(inactive.any_visible.tolist(), [False])

        same = torch.tensor([[0.25, 0.25, 0.5]], device="cuda")
        degenerate = scene.visible_edge(same, same, torch.zeros_like(same), edge_t, edge_t, sample_fractions=(0.0,))
        self.assertEqual(degenerate.any_visible.tolist(), [True])

    def test_visible_chain_empty_inactive_degenerate_ignore_and_blocker_id(self):
        scene, _ = _torch_scene(two_meshes=True)
        empty_points = torch.empty((0, 2, 3), device="cuda")
        empty_length = torch.empty((0,), device="cuda", dtype=torch.int32)
        empty = scene.visible_chain(empty_points, empty_length)
        self.assertEqual(empty.chain_count, 0)
        self.assertEqual(empty.max_segments, 1)
        self.assertEqual(tuple(empty.all_visible.shape), (0,))

        crossing = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]], device="cuda")
        length = torch.tensor([1], device="cuda", dtype=torch.int32)
        blocked = scene.visible_chain(crossing, length)
        self.assertEqual(blocked.all_visible.tolist(), [False])
        self.assertEqual(blocked.first_blocked_segment.tolist(), [0])
        self.assertEqual(blocked.first_blocked_prim.tolist(), [1])

        ignored = scene.visible_chain(
            crossing, length, ignore_prim_per_segment=torch.tensor([[[1]]], device="cuda", dtype=torch.int32)
        )
        self.assertEqual(ignored.all_visible.tolist(), [True])
        self.assertEqual(ignored.first_blocked_segment.tolist(), [-1])
        self.assertEqual(ignored.first_blocked_prim.tolist(), [-1])

        inactive = scene.visible_chain(crossing, length, active=torch.tensor([False], device="cuda"))
        self.assertEqual(inactive.all_visible.tolist(), [False])
        self.assertEqual(inactive.first_blocked_segment.tolist(), [-1])
        self.assertEqual(inactive.first_blocked_prim.tolist(), [-1])

        same = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]], device="cuda")
        degenerate = scene.visible_chain(same, length)
        self.assertEqual(degenerate.all_visible.tolist(), [True])
        self.assertEqual(degenerate.first_blocked_segment.tolist(), [-1])
        self.assertEqual(degenerate.first_blocked_prim.tolist(), [-1])

    def test_global_geometry_has_six_fields_and_scene_global_id_spaces(self):
        scene, _ = _torch_scene(two_meshes=True)
        geometry = scene.global_geometry()
        self.assertEqual(
            tuple(geometry.__dataclass_fields__),
            ("vertices", "faces", "face_normal", "shape_id", "local_prim_id", "global_prim_id"),
        )
        self.assertEqual(tuple(geometry.vertices.shape), (6, 3))
        self.assertEqual(tuple(geometry.faces.shape), (2, 3))
        self.assertEqual(tuple(geometry.face_normal.shape), (2, 3))
        self.assertEqual(geometry.shape_id.tolist(), [0, 1])
        self.assertEqual(geometry.local_prim_id.tolist(), [0, 0])
        self.assertEqual(geometry.global_prim_id.tolist(), [0, 1])
        self.assertEqual(geometry.faces[0].tolist(), [0, 1, 2])
        self.assertEqual(geometry.faces[1].tolist(), [3, 4, 5])
        torch.testing.assert_close(
            geometry.face_normal, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda")
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
@unittest.skipUnless(os.environ.get("RAYD_TORCH_RUN_DR_JIT_PARITY") == "1", "cross-backend F1 parity is opt-in")
class F1CrossBackendAcceptanceTests(unittest.TestCase):
    def load_backends(self):
        try:
            dr_backend = importlib.import_module("rayd.drjit")
            cuda = importlib.import_module("drjit.cuda")
            ad = importlib.import_module("drjit.cuda.ad")
            dr = importlib.import_module("drjit")
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            self.skipTest(f"Dr.Jit RayD backend is unavailable: {exc}")
        return dr_backend, cuda, ad, dr

    @staticmethod
    def drjit_scene(dr_backend, cuda, *, two_meshes=False):
        faces = cuda.Array3i([0], [1], [2])
        scene = dr_backend.Scene()
        if two_meshes:
            scene.add_mesh(dr_backend.Mesh(cuda.Array3f([9.0, 11.0, 10.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]), faces))
        scene.add_mesh(dr_backend.Mesh(cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]), faces))
        scene.build()
        return scene

    def test_visibility_and_global_geometry_forward_parity(self):
        dr_backend, cuda, _ad, _dr = self.load_backends()
        torch_scene, _ = _torch_scene(two_meshes=True)
        drjit_scene = self.drjit_scene(dr_backend, cuda, two_meshes=True)

        start_t = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
        end_a_t = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
        end_b_t = torch.tensor([[3.0, 3.0, 1.0]], device="cuda")
        pair_t = torch_scene.visible_pair(start_t, end_a_t, end_b_t)
        axial_t = torch_scene.visible_edge(
            start_t,
            torch.tensor([[-2.0, 0.0, 1.0]], device="cuda"),
            torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
            torch.tensor([0.0], device="cuda"),
            torch.tensor([4.0], device="cuda"),
            sample_fractions=(0.0, 0.5, 1.0),
        )
        chain_t = torch_scene.visible_chain(
            torch.stack((start_t, end_a_t), dim=1), torch.tensor([1], device="cuda", dtype=torch.int32)
        )

        start_d = cuda.Array3f([0.0], [0.0], [-1.0])
        end_a_d = cuda.Array3f([0.0], [0.0], [1.0])
        end_b_d = cuda.Array3f([3.0], [3.0], [1.0])
        pair_d = drjit_scene.visible_pair(start_d, end_a_d, end_b_d)
        axial_d = drjit_scene.visible_edge(
            start_d,
            cuda.Array3f([-2.0], [0.0], [1.0]),
            cuda.Array3f([1.0], [0.0], [0.0]),
            cuda.Float([0.0]),
            cuda.Float([4.0]),
            [0.0, 0.5, 1.0],
            cuda.Bool([True]),
        )
        chain_d = drjit_scene.visible_chain(cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]), cuda.Int([1]))

        self.assertEqual(pair_t.visible_a.tolist(), [bool(pair_d.visible_a[0])])
        self.assertEqual(pair_t.visible_b.tolist(), [bool(pair_d.visible_b[0])])
        self.assertEqual(axial_t.any_visible.tolist(), [bool(axial_d.any_visible[0])])
        self.assertEqual(chain_t.all_visible.tolist(), [bool(chain_d.all_visible[0])])
        self.assertEqual(chain_t.first_blocked_prim.tolist(), [int(chain_d.first_blocked_prim[0])])

        geometry_t = torch_scene.global_geometry()
        geometry_d = drjit_scene.global_geometry()
        self.assertEqual(geometry_t.shape_id.tolist(), [int(v) for v in geometry_d.shape_id])
        self.assertEqual(geometry_t.local_prim_id.tolist(), [int(v) for v in geometry_d.local_prim_id])
        self.assertEqual(geometry_t.global_prim_id.tolist(), [int(v) for v in geometry_d.global_prim_id])
        faces_d = [[int(geometry_d.faces[axis][face]) for axis in range(3)] for face in range(2)]
        self.assertEqual(geometry_t.faces.tolist(), faces_d)

    def test_topk_fixed_winner_forward_vjp_jvp_parity(self):
        dr_backend, cuda, ad, dr = self.load_backends()
        torch_scene, _ = _torch_scene()
        drjit_scene = self.drjit_scene(dr_backend, cuda)
        tangent_values = [0.07, -0.03, 0.05]

        point_t = torch.tensor([[0.25, -1.2, 0.3]], device="cuda", dtype=torch.float32, requires_grad=True)
        result_t = torch_scene.nearest_edges(point_t, 2)
        scalar_t = result_t.distances[0, 0] + 0.3 * result_t.distances[0, 1]
        scalar_t.backward()
        grad_t = point_t.grad.detach().flatten().tolist()

        tangent_t = torch.tensor([tangent_values], device="cuda")

        def torch_topk(point):
            result = torch_scene.nearest_edges(point, 2)
            return result.distances[:, 0] + 0.3 * result.distances[:, 1]

        _primal_t, jvp_t = torch.func.jvp(torch_topk, (point_t.detach(),), (tangent_t,))

        point_d = ad.Array3f([0.25], [-1.2], [0.3])
        dr.enable_grad(point_d)
        result_d = drjit_scene.nearest_edges(point_d, 2)
        weighted_d = result_d.distances * ad.Float([1.0, 0.3])
        scalar_d = dr.sum(weighted_d)
        dr.backward(weighted_d)
        grad_d = [float(dr.grad(point_d)[axis][0]) for axis in range(3)]

        point_j = ad.Array3f([0.25], [-1.2], [0.3])
        dr.enable_grad(point_j)
        result_j = drjit_scene.nearest_edges(point_j, 2)
        weighted_j = result_j.distances * ad.Float([1.0, 0.3])
        dr.set_grad(point_j, ad.Array3f([tangent_values[0]], [tangent_values[1]], [tangent_values[2]]))
        dr.forward_from(point_j)
        jvp_d = float(dr.sum(dr.grad(weighted_j))[0])

        self.assertEqual(result_t.global_edge_ids[0].tolist(), [int(v) for v in result_d.global_edge_ids])
        self.assertAlmostEqual(float(scalar_t.detach()), float(scalar_d[0]), delta=1.0e-5)
        for actual, expected in zip(grad_t, grad_d):
            self.assertAlmostEqual(actual, expected, delta=5.0e-4)
        self.assertAlmostEqual(float(jvp_t[0]), jvp_d, delta=5.0e-4)


if __name__ == "__main__":
    unittest.main()
