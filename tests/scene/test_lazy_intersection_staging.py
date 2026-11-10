# Copyright Xingyu Chen.
# Tests that a lazily read intersection traces once and completes its other fields exactly like a full trace.

import unittest
from unittest import mock

import torch

if torch.cuda.is_available():
    import rayd.torch as rt

_FIELDS = ("p", "n", "geo_n", "uv", "barycentric", "shape_id", "prim_id", "local_prim_id", "global_prim_id")


def _grid(n=24):
    xs = torch.linspace(0.0, 1.0, n + 1, device="cuda")
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    z = 0.05 * torch.sin(7.0 * gx) * torch.cos(5.0 * gy)
    vertices = torch.stack([gx.reshape(-1), gy.reshape(-1), z.reshape(-1)], 1).contiguous()
    i = torch.arange(n, device="cuda")
    ii, jj = torch.meshgrid(i, i, indexing="ij")
    a = (ii * (n + 1) + jj).reshape(-1)
    b, c = a + 1, a + n + 1
    faces = torch.cat([torch.stack([a, c, b], 1), torch.stack([b, c, c + 1], 1)]).int().contiguous()
    return vertices, faces


def _rays(count=4096, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    origin = torch.rand(count, 3, device="cuda", generator=generator) * 1.4 - 0.2
    origin[:, 2] = 1.0
    direction = torch.randn(count, 3, device="cuda", generator=generator) * 0.2
    direction[:, 2] = -1.0
    return rt.Ray(origin, direction)


def _same(a, b):
    if a.is_floating_point():
        return torch.equal(torch.nan_to_num(a, nan=7.0), torch.nan_to_num(b, nan=7.0))
    return torch.equal(a, b)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class LazyIntersectionStagingTests(unittest.TestCase):
    def _scene(self, vertices, faces):
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        return scene

    def test_reading_t_then_other_fields_traces_once(self):
        vertices, faces = _grid()
        vertices.requires_grad_()
        scene = self._scene(vertices, faces)
        hit = scene.intersect(_rays())
        hit.t
        with mock.patch.object(
            torch.ops.rayd_torch,
            "intersect_ad_flags",
            side_effect=AssertionError("a lazily read intersection must not trace a second time"),
        ):
            for field in _FIELDS:
                getattr(hit, field)
            hit.is_valid()

    def test_completed_fields_and_gradients_match_a_full_trace(self):
        vertices, faces = _grid()
        ray = _rays()
        generator = torch.Generator(device="cuda").manual_seed(4)
        active = torch.rand(ray.o.shape[0], device="cuda", generator=generator) > 0.2
        for flags in (rt.RayFlags.All, rt.RayFlags.Geometric, rt.RayFlags.ShadingN, rt.RayFlags.UV):
            for mask in (None, active):
                with self.subTest(flags=flags, active=mask is not None):
                    leaf = vertices.clone().requires_grad_()
                    scene = self._scene(leaf, faces)
                    staged = scene.intersect(ray, mask, flags)
                    t = staged.t
                    full = scene.intersect(ray, mask, flags)
                    full_p = full.p
                    self.assertTrue(_same(t, full.t))
                    for field in _FIELDS:
                        self.assertTrue(_same(getattr(staged, field), getattr(full, field)), field)
                    self.assertIs(staged.t, t)
                    hit_rows = torch.isfinite(full.t)
                    if bool(hit_rows.any()) and full_p.numel() != 0:
                        (grad_staged,) = torch.autograd.grad((staged.p[hit_rows] ** 2).sum(), leaf)
                        (grad_full,) = torch.autograd.grad((full_p[hit_rows] ** 2).sum(), leaf)
                        torch.testing.assert_close(grad_staged, grad_full, rtol=1e-6, atol=1e-6)
                    if bool(hit_rows.any()):
                        # `.t` keeps the t-only result and its gradient path, exactly as before staging existed.
                        t_only = torch.ops.rayd_torch.intersect_ad_t(
                            scene._native_scene, leaf, ray.o, ray.d, ray.tmax, mask
                        )
                        (grad_t,) = torch.autograd.grad(t[hit_rows].sum(), leaf)
                        (grad_t_only,) = torch.autograd.grad(t_only[hit_rows].sum(), leaf)
                        torch.testing.assert_close(grad_t, grad_t_only, rtol=1e-6, atol=1e-6)

    def test_completion_reports_instance_ids_like_the_trace(self):
        vertices, faces = _grid(8)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        shift = torch.eye(4, device="cuda")
        shift[0, 3] = 2.0
        scene.add_instance(0, shift)
        scene.build()
        origin = torch.rand(2048, 3, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
        origin[:, 0] *= 3.2
        origin[:, 2] = 1.0
        direction = torch.zeros_like(origin)
        direction[:, 2] = -1.0
        native = scene._native_scene
        leaf = vertices.clone().requires_grad_()
        tmax = torch.empty(0, device="cuda")
        flags = int(rt.RayFlags.All)
        t, prim, uv = torch.ops.rayd_torch.intersect_ad_t_staged(native, leaf, origin, direction, tmax, None)
        completed = torch.ops.rayd_torch.intersect_ad_complete(
            native, leaf, origin, direction, tmax, None, flags, int(native.geometry_version()), t, prim, uv
        )
        full = torch.ops.rayd_torch.intersect_ad_flags(native, leaf, origin, direction, tmax, None, flags)
        self.assertTrue(bool((full[6] == 1).any()))
        for index, value in enumerate(completed):
            self.assertTrue(_same(value, full[index + 1]), _FIELDS[index])

    def test_completion_after_sync_is_refused(self):
        vertices, faces = _grid(8)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices.clone().requires_grad_(), faces), dynamic=True)
        scene.build()
        hit = scene.intersect(_rays(256))
        hit.t
        scene.update_mesh_vertices(0, vertices + 0.01)
        scene.sync()
        with self.assertRaisesRegex(RuntimeError, "geometry changed"):
            hit.p
        native = scene._native_scene
        ray = _rays(16)
        tmax = torch.empty(0, device="cuda")
        t, prim, uv = torch.ops.rayd_torch.intersect_ad_t_staged(native, vertices, ray.o, ray.d, tmax, None)
        with self.assertRaisesRegex(RuntimeError, "geometry changed"):
            torch.ops.rayd_torch.intersect_ad_complete(
                native, vertices, ray.o, ray.d, tmax, None, 15, int(native.geometry_version()) - 1, t, prim, uv
            )


if __name__ == "__main__":
    unittest.main()
