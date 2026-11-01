# Copyright Xingyu Chen.
# Opt-in stress tests: adversarial rays, degenerate meshes, deep reflections, and repeated calls on both backends.

import importlib.util
import os
import unittest

from tests.support.subprocess_cases import compose, run_json_case

RUN_STRESS = os.environ.get("RAYD_RUN_STRESS") == "1"
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DRJIT = importlib.util.find_spec("drjit") is not None


def _backend_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


TORCH_PRELUDE = """
import json
import sys

import numpy as np
import torch
import rayd.torch as rt

from tests.stress.adversarial_inputs import SCENES, adversarial_rays

scene_name, trace_backend = sys.argv[1], sys.argv[2]
device = torch.device("cuda")
vertices_np, faces_np = SCENES[scene_name]
vertices = torch.from_numpy(vertices_np).to(device)
faces = torch.from_numpy(faces_np).to(device)
origin_np, direction_np, well_formed_np = adversarial_rays()
origin = torch.from_numpy(origin_np).to(device)
direction = torch.from_numpy(direction_np).to(device)
well_formed = torch.from_numpy(well_formed_np).to(device)


def build(vertex_tensor):
    scene = rt.Scene(trace_backend=trace_backend)
    scene.add_mesh(rt.Mesh(vertex_tensor, faces))
    scene.build()
    return scene


def nonfinite(*tensors):
    return sum(int((~torch.isfinite(tensor.float())).sum()) for tensor in tensors)
"""

TORCH_BODY = """
scene = build(vertices)
report = {}
its = scene.intersect(rt.Ray(origin, direction))
hit = (its.prim_id >= 0) & well_formed
report["intersect"] = nonfinite(its.t[hit], its.p[hit], its.n[hit], its.geo_n[hit], its.uv[hit], its.barycentric[hit])

grad_vertices = vertices.clone().requires_grad_(True)
grad_origin = origin[well_formed].clone().requires_grad_(True)
grad_direction = direction[well_formed].clone().requires_grad_(True)
grad_its = build(grad_vertices).intersect(rt.Ray(grad_origin, grad_direction))
grad_hit = grad_its.prim_id >= 0
(grad_its.t[grad_hit].sum() + grad_its.p[grad_hit].sum() + grad_its.n[grad_hit].sum()).backward()
report["intersect_grad"] = nonfinite(grad_vertices.grad, grad_origin.grad, grad_direction.grad)

point_edge = scene.nearest_edge(origin)
found = (point_edge.edge_id >= 0) & well_formed
report["nearest_edge_point"] = nonfinite(point_edge.distance[found], point_edge.edge_point[found])
ray_edge = scene.nearest_edge(rt.Ray(origin, direction))
found = (ray_edge.edge_id >= 0) & well_formed
report["nearest_edge_ray"] = nonfinite(ray_edge.distance[found], ray_edge.edge_point[found])
report["visible_completes"] = int(scene.visible(origin, origin + direction).numel() != origin.shape[0])

for bounces in (8, 64):
    chain = scene.trace_reflections(rt.Ray(origin, direction), bounces)
    slots = chain.valid & well_formed[:, None]
    report[f"trace_reflections_{bounces}"] = nonfinite(chain.t[slots], chain.image_sources[slots])

source = origin[well_formed]
receiver = source + 100.0 * direction[well_formed]
field = scene.trace_refl_epc_field(source, receiver, 8)
report["trace_refl_epc_field"] = nonfinite(field.field_real[field.valid], field.path_length[field.valid])
torch.cuda.synchronize()
print(json.dumps(report))
"""

TORCH_MEMORY_BODY = """
import gc

def step(index):
    scaled = (vertices * (1.0 + 0.01 * (index % 7))).requires_grad_(True)
    scene = build(scaled)
    ray = rt.Ray(origin[well_formed], direction[well_formed])
    its = scene.intersect(ray)
    chain = scene.trace_reflections(ray, 4)
    (torch.where(its.prim_id >= 0, its.t, 0).sum() + chain.t[chain.valid].sum()).backward()
    scene.visible(ray.o, ray.o + ray.d)
    scene.nearest_edge(ray.o)

samples = []
for index in range(150):
    step(index)
    if index in (30, 149):
        gc.collect()
        torch.cuda.synchronize()
        samples.append(torch.cuda.memory_allocated())
print(json.dumps({"allocated_growth": samples[1] - samples[0]}))
"""

DRJIT_PRELUDE = """
import json
import sys

import numpy as np
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as rd

from tests.stress.adversarial_inputs import SCENES, adversarial_rays

scene_name, trace_backend = sys.argv[1], sys.argv[2]
vertices_np, faces_np = SCENES[scene_name]
mesh = rd.Mesh(cuda.Array3f(*vertices_np.T.tolist()), cuda.Array3i(*faces_np.T.tolist()))
mesh.build()
scene = rd.Scene(trace_backend=trace_backend)
scene.add_mesh(mesh)
scene.build()
origin_np, direction_np, well_formed = adversarial_rays()
ray = rd.Ray(cuda.Array3f(*origin_np.T.tolist()), cuda.Array3f(*direction_np.T.tolist()))


def rows(value):
    array = np.array(value, dtype=np.float64)
    return array.T if array.ndim == 2 and array.shape[0] == 3 else array


def nonfinite(mask, *values):
    return sum(int((~np.isfinite(rows(value)[mask])).sum()) for value in values)
"""

DRJIT_BODY = """
report = {}
its = scene.intersect(ray)
hit = np.array(its.is_valid()) & well_formed
report["intersect"] = nonfinite(hit, its.t, its.p, its.n)
point_edge = scene.nearest_edge(ray.o)
report["nearest_edge_point"] = nonfinite(np.array(point_edge.is_valid()) & well_formed, point_edge.distance)
ray_edge = scene.nearest_edge(ray)
report["nearest_edge_ray"] = nonfinite(np.array(ray_edge.is_valid()) & well_formed, ray_edge.distance)
visibility = scene.visible(ray.o, ray.o + ray.d)
dr.eval(visibility.visible)
for bounces in (8, 64):
    for symbolic in (True, False):
        chain = scene.trace_reflections(ray, bounces, symbolic=symbolic)
        count = 0
        if symbolic:
            for depth in range(min(bounces, 8)):
                bounce = chain.bounce(depth)
                count += nonfinite(np.array(bounce.is_valid()) & well_formed, bounce.t, bounce.hit_points)
        else:
            dr.eval(chain.t, chain.bounce_count)
        report[f"trace_reflections_{bounces}_{'symbolic' if symbolic else 'eager'}"] = count
dr.sync_thread()
print(json.dumps(report))
"""


def _run(prelude: str, body: str, scene: str, trace_backend: str) -> dict:
    arguments = f"import sys\nsys.argv = ['stress', {scene!r}, {trace_backend!r}]"
    return run_json_case(compose(arguments, prelude, body), timeout=600)


@unittest.skipUnless(RUN_STRESS, "set RAYD_RUN_STRESS=1 to run the adversarial stress suite")
class AdversarialRayStressTests(unittest.TestCase):
    def _assert_all_finite(self, report: dict) -> None:
        for operation, count in report.items():
            self.assertEqual(count, 0, f"{operation} returned {count} non-finite values for well-formed rows")

    @unittest.skipUnless(HAS_TORCH and _backend_importable("rayd.torch"), "rayd.torch is not installed")
    def test_torch_operations_survive_adversarial_rays(self):
        for scene in ("box", "degenerate"):
            for trace_backend in ("optix", "cuda"):
                with self.subTest(scene=scene, trace_backend=trace_backend):
                    self._assert_all_finite(_run(TORCH_PRELUDE, TORCH_BODY, scene, trace_backend))

    @unittest.skipUnless(HAS_TORCH and _backend_importable("rayd.torch"), "rayd.torch is not installed")
    def test_torch_repeated_rebuilds_do_not_grow_device_memory(self):
        report = _run(TORCH_PRELUDE, TORCH_MEMORY_BODY, "box", "optix")
        self.assertLessEqual(report["allocated_growth"], 1 << 20)

    @unittest.skipUnless(HAS_DRJIT and _backend_importable("rayd.drjit"), "rayd.drjit is not installed")
    def test_drjit_operations_survive_adversarial_rays(self):
        for scene in ("box", "degenerate"):
            for trace_backend in ("optix", "cuda"):
                with self.subTest(scene=scene, trace_backend=trace_backend):
                    self._assert_all_finite(_run(DRJIT_PRELUDE, DRJIT_BODY, scene, trace_backend))


if __name__ == "__main__":
    unittest.main()
