# Copyright Xingyu Chen.
# Tests that Torch Intersection.uv interpolates mesh texture coordinates and differentiates through them.

import unittest

import torch

try:
    import rayd.torch as rt
except ImportError:  # pragma: no cover - exercised only without the Torch backend
    rt = None


def _tensor(values, dtype=torch.float32, requires_grad=False) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=dtype, requires_grad=requires_grad)


QUAD_VERTICES = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
QUAD_FACES = [[0, 1, 2], [0, 2, 3]]
# Per-corner UVs: face 0 corners use rows 0, 1, 2 and face 1 corners rows 3, 4, 5, so the map is discontinuous.
UV_ROWS = [[0.1, 0.2], [0.9, 0.3], [0.7, 0.8], [0.0, 0.0], [0.5, 1.0], [0.25, 0.5]]
FACE_UV = [[0, 1, 2], [3, 4, 5]]


@unittest.skipUnless(rt is not None and torch.cuda.is_available(), "CUDA Torch backend is required")
class MeshUvTests(unittest.TestCase):
    def _scene(self, meshes, trace_backend="auto"):
        scene = rt.Scene(trace_backend=trace_backend)
        for mesh in meshes:
            scene.add_mesh(mesh)
        scene.build()
        return scene

    def _expected(self, hit, corners):
        bary = hit.barycentric
        table = _tensor(corners)
        return bary[:, 0:1] * table[0] + bary[:, 1:2] * table[1] + bary[:, 2:3] * table[2]

    def test_face_uv_indexes_per_corner_texture_coordinates(self):
        for trace_backend in ("optix", "cuda"):
            with self.subTest(trace_backend=trace_backend):
                mesh = rt.Mesh(
                    _tensor(QUAD_VERTICES),
                    _tensor(QUAD_FACES, torch.int32),
                    uv=_tensor(UV_ROWS),
                    face_uv=_tensor(FACE_UV, torch.int32),
                )
                scene = self._scene([mesh], trace_backend)
                ray = rt.Ray(_tensor([[0.7, 0.2, -1.0], [0.2, 0.7, -1.0]]), _tensor([[0.0, 0.0, 1.0]] * 2))
                hit = scene.intersect(ray)
                self.assertEqual(hit.global_prim_id.tolist(), [0, 1])
                expected = torch.cat((self._expected(hit, UV_ROWS[0:3])[0:1], self._expected(hit, UV_ROWS[3:6])[1:2]))
                torch.testing.assert_close(hit.uv, expected, atol=1e-6, rtol=1e-6)

    def test_vertex_uv_without_face_uv_uses_the_face_indices(self):
        vertex_uv = [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0]]
        mesh = rt.Mesh(_tensor(QUAD_VERTICES), _tensor(QUAD_FACES, torch.int32), uv=_tensor(vertex_uv))
        hit = self._scene([mesh]).intersect(rt.Ray(_tensor([[0.7, 0.2, -1.0]]), _tensor([[0.0, 0.0, 1.0]])))
        torch.testing.assert_close(hit.uv, _tensor([[1.4, 0.8]]), atol=1e-6, rtol=1e-6)

    def test_mesh_without_uv_keeps_barycentric_uv_next_to_a_textured_mesh(self):
        plain = rt.Mesh(_tensor(QUAD_VERTICES), _tensor(QUAD_FACES, torch.int32))
        lifted = [[x, y, 2.0] for x, y, _ in QUAD_VERTICES]
        textured = rt.Mesh(
            _tensor(lifted),
            _tensor(QUAD_FACES, torch.int32),
            uv=_tensor(UV_ROWS),
            face_uv=_tensor(FACE_UV, torch.int32),
        )
        ray = rt.Ray(_tensor([[0.7, 0.2, -1.0]]), _tensor([[0.0, 0.0, 1.0]]))
        alone = self._scene([plain]).intersect(ray)
        mixed = self._scene([plain, textured]).intersect(ray)
        self.assertEqual(mixed.shape_id.tolist(), [0])
        # Bitwise: a mesh without UVs reports the barycentric (u, v) whether or not another mesh is textured.
        self.assertTrue(torch.equal(mixed.uv, alone.uv))
        torch.testing.assert_close(alone.uv, alone.barycentric[:, 1:3], atol=0.0, rtol=0.0)

    def test_invalid_uv_inputs_are_rejected_at_build(self):
        vertices, faces = _tensor(QUAD_VERTICES), _tensor(QUAD_FACES, torch.int32)
        cases = {
            "face_uv index 6 outside the valid range [0, 6)": dict(
                uv=_tensor(UV_ROWS), face_uv=_tensor([[0, 1, 2], [3, 4, 6]], torch.int32)
            ),
            "face_uv index -1": dict(uv=_tensor(UV_ROWS), face_uv=_tensor([[0, 1, 2], [-1, 4, 5]], torch.int32)),
            "face_uv indices but no uv": dict(face_uv=_tensor(FACE_UV, torch.int32)),
            "uv rows for 4 vertices": dict(uv=_tensor(UV_ROWS)),
            "face_uv rows for 2 faces": dict(uv=_tensor(UV_ROWS), face_uv=_tensor([[0, 1, 2]], torch.int32)),
        }
        for message, kwargs in cases.items():
            with self.subTest(message=message):
                scene = rt.Scene()
                scene.add_mesh(rt.Mesh(vertices, faces, **kwargs))
                with self.assertRaisesRegex(RuntimeError, message.replace("[", r"\[").replace(")", r"\)")):
                    scene.build()

    def test_uv_that_requires_grad_is_rejected(self):
        scene = rt.Scene()
        scene.add_mesh(
            rt.Mesh(
                _tensor(QUAD_VERTICES),
                _tensor(QUAD_FACES, torch.int32),
                uv=_tensor(UV_ROWS, requires_grad=True),
                face_uv=_tensor(FACE_UV, torch.int32),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "uv requires grad"):
            scene.build()

    def _objective(self, vertices, origins):
        mesh = rt.Mesh(
            vertices, _tensor(QUAD_FACES, torch.int32), uv=_tensor(UV_ROWS), face_uv=_tensor(FACE_UV, torch.int32)
        )
        # The scene must outlive the backward pass, which reads its geometry.
        self.live_scene = self._scene([mesh])
        hit = self.live_scene.intersect(rt.Ray(origins, _tensor([[0.1, -0.05, 1.0], [-0.08, 0.06, 1.0]])))
        weights = _tensor([[1.0, -0.7], [0.4, 1.3]])
        return (hit.uv * weights).sum()

    def test_uv_gradients_match_central_differences(self):
        vertices = _tensor([[0.0, 0.0, 0.1], [1.0, 0.05, 0.0], [1.1, 1.0, 0.2], [0.0, 0.9, 0.0]])
        origins = _tensor([[0.7, 0.2, -1.0], [0.2, 0.7, -1.0]])
        generator = torch.Generator(device="cuda").manual_seed(5)
        direction_v = torch.randn(vertices.shape, device="cuda", generator=generator)
        direction_o = torch.randn(origins.shape, device="cuda", generator=generator)
        step = 1.0e-3
        with torch.no_grad():
            plus = self._objective(vertices + step * direction_v, origins + step * direction_o)
            minus = self._objective(vertices - step * direction_v, origins - step * direction_o)
        reference = float((plus - minus) / (2.0 * step))
        self.assertGreater(abs(reference), 1.0e-2)

        v = vertices.clone().requires_grad_(True)
        o = origins.clone().requires_grad_(True)
        self._objective(v, o).backward()
        vjp = float((v.grad * direction_v).sum() + (o.grad * direction_o).sum())
        self.assertAlmostEqual(vjp, reference, delta=2e-3 * abs(reference) + 1e-4)

        with torch.autograd.forward_ad.dual_level():
            dual_v = torch.autograd.forward_ad.make_dual(vertices, direction_v)
            dual_o = torch.autograd.forward_ad.make_dual(origins, direction_o)
            jvp = float(torch.autograd.forward_ad.unpack_dual(self._objective(dual_v, dual_o)).tangent)
        self.assertAlmostEqual(jvp, reference, delta=2e-3 * abs(reference) + 1e-4)


if __name__ == "__main__":
    unittest.main()
