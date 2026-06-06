import sys
import unittest
from pathlib import Path

import torch


RAYDI_ROOT = Path(r"E:\Code\RayDi")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class DrJitParityTests(unittest.TestCase):
    def test_intersect_forward_matches_drjit_baseline_case(self):
        sys.path.insert(0, str(RAYDI_ROOT))
        import drjit.cuda as cuda
        import rayd as dr_backend
        import raydtorch as rt

        verts_t = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces_t = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene_t = rt.Scene()
        scene_t.add_mesh(rt.Mesh(verts_t, faces_t))
        scene_t.build()
        ray_t = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        out_t = scene_t.intersect(ray_t)

        verts_d = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        faces_d = cuda.Array3i([0], [1], [2])
        scene_d = dr_backend.Scene()
        scene_d.add_mesh(dr_backend.Mesh(verts_d, faces_d))
        scene_d.build()
        ray_d = dr_backend.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
        out_d = scene_d.intersect(ray_d)
        self.assertAlmostEqual(float(out_t.t[0].item()), float(out_d.t[0]), places=5)
