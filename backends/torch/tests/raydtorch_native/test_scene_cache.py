import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class SceneCacheTests(unittest.TestCase):
    def _mesh(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        return rt.Mesh(verts, faces)

    def test_scene_build_creates_native_handle_and_version(self):
        scene = rt.Scene()
        mesh_id = scene.add_mesh(self._mesh())
        self.assertEqual(mesh_id, 0)
        scene.build()
        self.assertTrue(scene.is_ready())
        self.assertEqual(scene.num_meshes, 1)
        self.assertGreaterEqual(scene.version, 1)

    def test_query_before_build_fails(self):
        scene = rt.Scene()
        scene.add_mesh(self._mesh())
        with self.assertRaisesRegex(RuntimeError, "Call build"):
            scene.intersect(
                rt.Ray(
                    torch.zeros((1, 3), device="cuda", dtype=torch.float32),
                    torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
                )
            )


if __name__ == "__main__":
    unittest.main()
