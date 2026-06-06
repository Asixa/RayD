import os
import sys
import unittest
import importlib
from pathlib import Path

import torch


RAYDI_ROOT = Path(r"E:\Code\RayDi")


def _load_backends():
    sys.path.insert(0, str(RAYDI_ROOT))
    import rayd as dr_backend
    import raydtorch as rt

    cuda = importlib.import_module("dr" + "jit.cuda")
    return dr_backend, rt, cuda


def _torch_scene(rt):
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()
    return scene


def _rayd_scene(dr_backend, cuda):
    verts = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    faces = cuda.Array3i([0], [1], [2])
    scene = dr_backend.Scene()
    scene.add_mesh(dr_backend.Mesh(verts, faces))
    scene.build()
    return scene


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
@unittest.skipUnless(os.environ.get("RAYDTORCH_RUN_DR_JIT_PARITY") == "1", "external RayDi parity is opt-in")
class DrJitParityTests(unittest.TestCase):
    def test_intersect_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        ray_t = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        out_t = scene_t.intersect(ray_t)

        scene_d = _rayd_scene(dr_backend, cuda)
        ray_d = dr_backend.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
        out_d = scene_d.intersect(ray_d)
        self.assertAlmostEqual(float(out_t.t[0].item()), float(out_d.t[0]), places=5)

    def test_nearest_edge_point_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        out_t = scene_t.nearest_edge(torch.tensor([[0.25, -0.2, 0.0]], device="cuda", dtype=torch.float32))

        scene_d = _rayd_scene(dr_backend, cuda)
        out_d = scene_d.nearest_edge(cuda.Array3f([0.25], [-0.2], [0.0]))
        self.assertAlmostEqual(float(out_t.distance[0].item()), float(out_d.distance[0]), places=5)
        self.assertAlmostEqual(float(out_t.edge_t[0].item()), float(out_d.edge_t[0]), places=5)
        self.assertEqual(int(out_t.edge_id[0].item()), int(out_d.edge_id[0]))

    def test_visibility_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        start_t = torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32)
        end_t = torch.tensor([[0.25, 0.25, 1.0]], device="cuda", dtype=torch.float32)
        out_t = scene_t.visible(start_t, end_t)

        scene_d = _rayd_scene(dr_backend, cuda)
        out_d = scene_d.visible(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.25], [0.25], [1.0]))
        self.assertEqual(bool(out_t[0].item()), bool(out_d.visible[0]))

    def test_reflection_trace_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        ray_t = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        out_t = scene_t.trace_reflections(ray_t, max_bounces=1)

        scene_d = _rayd_scene(dr_backend, cuda)
        ray_d = dr_backend.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
        out_d = scene_d.trace_reflections(ray_d, max_bounces=1, symbolic=False)
        self.assertEqual(bool(out_t.valid[0, 0].item()), bool(out_d.is_valid()[0]))
        self.assertAlmostEqual(float(out_t.t[0, 0].item()), float(out_d.t[0]), places=5)
        self.assertEqual(int(out_t.prim_ids[0, 0].item()), int(out_d.prim_ids[0]))
