# Copyright Xingyu Chen.
# Tests that the EPC in-face test is a distance: a large face's reflection ends at its boundary edge.

import unittest

import torch

import rayd.torch as rt

# A square plate in x = 0 with |y| <= HALF and -2 HALF <= z <= 0: one boundary edge along y at z = 0.
HALF = 20.0


def _plate_scene(trace_backend):
    vertices = torch.tensor(
        [[0.0, -HALF, -2.0 * HALF], [0.0, HALF, -2.0 * HALF], [0.0, HALF, 0.0], [0.0, -HALF, 0.0]], device="cuda"
    )
    faces = torch.tensor([[0, 2, 1], [0, 3, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene(trace_backend=trace_backend)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchEpcFaceContainmentTests(unittest.TestCase):
    def test_reflection_ends_at_the_edge_of_a_large_face(self):
        # Source (-1, 0, 0.5): the reflection off x = 0 reaches the edge z = 0 for a receiver at
        # (-1, 0, -0.5). A receiver offset dz moves the reflection point by dz / 2 from the edge; the
        # former barycentric slack (1e-4 of a 40 m triangle) accepted points 4 mm past it.
        offsets = torch.tensor([-2.0e-3, -5.0e-4, -1.0e-4, 1.0e-4, 5.0e-4, 2.0e-3])
        receiver = torch.stack(
            [torch.full_like(offsets, -1.0), torch.zeros_like(offsets), -0.5 + offsets], dim=1
        ).cuda()
        source = torch.tensor([[-1.0, 0.0, 0.5]], device="cuda").expand_as(receiver).contiguous()
        # Each ray hits the plate 1 cm inside the edge; the path correction then moves the
        # reflection point to the exact specular point, which the in-face test accepts or rejects.
        aim = torch.tensor([[0.0, 0.0, -0.01]], device="cuda").expand_as(receiver)
        ray = rt.Ray(source, torch.nn.functional.normalize(aim - source, dim=1).contiguous())
        for trace_backend in ("optix", "cuda"):
            with self.subTest(trace_backend=trace_backend):
                scene = _plate_scene(trace_backend)
                result = scene.trace_refl_epc_field(ray, receiver, 1)
                torch.cuda.synchronize()
                self.assertEqual(result.valid.cpu().tolist(), (offsets < 0).tolist())


if __name__ == "__main__":
    unittest.main()
