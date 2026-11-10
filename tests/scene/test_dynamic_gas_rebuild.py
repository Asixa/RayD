# Copyright Xingyu Chen.
# Tests that dynamic meshes trace exactly like a fresh build across many refit and periodic-rebuild commits.

import unittest

import torch

if torch.cuda.is_available():
    import rayd.torch as rt


def _sheet(n=48):
    xs = torch.linspace(-0.5, 0.5, n + 1, device="cuda")
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    vertices = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros((n + 1) * (n + 1), device="cuda")], 1)
    i = torch.arange(n, device="cuda")
    ii, jj = torch.meshgrid(i, i, indexing="ij")
    a = (ii * (n + 1) + jj).reshape(-1)
    b, c = a + 1, a + n + 1
    faces = torch.cat([torch.stack([a, c, b], 1), torch.stack([b, c, c + 1], 1)]).int().contiguous()
    return vertices.contiguous(), faces


def _rays(count=8192):
    generator = torch.Generator(device="cuda").manual_seed(5)
    origin = torch.randn(count, 3, device="cuda", generator=generator)
    origin = origin / origin.norm(dim=1, keepdim=True) * 3.0
    target = (torch.rand(count, 3, device="cuda", generator=generator) - 0.5) * torch.tensor(
        [1.0, 1.0, 0.2], device="cuda"
    )
    return rt.Ray(origin, target - origin)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class DynamicGasRebuildTests(unittest.TestCase):
    def test_many_commits_trace_like_a_fresh_build(self):
        vertices, faces = _sheet()
        generator = torch.Generator(device="cuda").manual_seed(1)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices.clone(), faces), dynamic=True)
        shift = torch.eye(4, device="cuda")
        shift[0, 3] = 2.0
        scene.add_mesh(rt.Mesh(vertices + torch.tensor([0.0, 0.0, 0.3], device="cuda"), faces))
        scene.add_instance(1, shift)
        scene.build()
        ray = _rays()
        current = vertices
        # More commits than two rebuild periods, so refits, in-place rebuilds and the IAS refit/rebuild all occur.
        for step in range(40):
            current = (current + 0.004 * torch.randn(current.shape, device="cuda", generator=generator)).contiguous()
            scene.update_mesh_vertices(0, current)
            if step % 7 == 0:
                moved = shift.clone()
                moved[1, 3] = 0.01 * step
                scene.set_instance_transform(2, moved)
            scene.sync()
            if step % 8 == 7:
                fresh = rt.Scene()
                fresh.add_mesh(rt.Mesh(current.clone(), faces), dynamic=True)
                fresh.add_mesh(rt.Mesh(vertices + torch.tensor([0.0, 0.0, 0.3], device="cuda"), faces))
                fresh.add_instance(1, moved)
                fresh.build()
                actual, expected = scene.intersect(ray), fresh.intersect(ray)
                same_t = torch.equal(torch.nan_to_num(actual.t, posinf=-1.0), torch.nan_to_num(expected.t, posinf=-1.0))
                self.assertTrue(same_t, step)
                self.assertTrue(torch.equal(actual.shape_id, expected.shape_id), step)


if __name__ == "__main__":
    unittest.main()
