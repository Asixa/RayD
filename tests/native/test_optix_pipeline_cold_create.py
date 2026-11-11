# Copyright Xingyu Chen.
# Cold-creates every Torch multipath OptiX pipeline in a fresh process and checks the launch completes.

import unittest

import torch

from tests.support.subprocess_cases import compose, run_json_case

PRELUDE = """
import json
import math

import torch
import rayd.torch as rt

from tests.support.geometry import box_mesh

device = torch.device("cuda")
vertices, faces = box_mesh(device)
scene = rt.Scene(trace_backend="optix")
scene.add_mesh(rt.Mesh(vertices, faces))
scene.build()
source = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, 0.0]], device=device)
receiver = torch.tensor([[0.6, 0.2, -0.5], [0.0, -0.3, 0.4]], device=device)
ray = rt.Ray(source, receiver - source)

# Convex box edge along +x at (y=-1, z=1) between the z=+1 and y=-1 faces.
states = rt.DfrStates(
    edge_index=torch.zeros(2, dtype=torch.int32, device=device),
    edge_pos=torch.tensor([[0.0, -1.0, 1.0]] * 2, device=device),
    edge_dir=torch.tensor([[1.0, 0.0, 0.0]] * 2, device=device),
    edge_t_min=torch.full((2,), -1.0, device=device),
    edge_t_max=torch.full((2,), 1.0, device=device),
    n0=torch.tensor([[0.0, 0.0, 1.0]] * 2, device=device),
    n1=torch.tensor([[0.0, -1.0, 0.0]] * 2, device=device),
    prim0=torch.full((2,), 2, dtype=torch.int32, device=device),
    prim1=torch.full((2,), 4, dtype=torch.int32, device=device),
    exterior_angle=torch.full((2,), 1.5 * math.pi, device=device),
    src=torch.tensor([[0.0, -3.0, 2.0], [0.0, -2.0, 3.0]], device=device),
    src_power=torch.ones(2, device=device),
)
grid = rt.DfrGrid(axis=2, position=2.0, coord0_min=-3, coord0_max=3, coord1_min=-3, coord1_max=3, resolution0=8,
                  resolution1=8)
"""

CASES = {
    "trace_reflections": "values = [scene.trace_reflections(ray, 4).t]",
    "visible": "values = [scene.visible(source, receiver).float()]",
    "trace_refl_epc_field": "values = [scene.trace_refl_epc_field(source, receiver + 100.0 * (receiver - source), 4).field_re]",
    "accum_refl": """
        result = scene.accum_refl(
            ray, source, rt.AccumGrid(axis=2, position=0.5, resolution0=4, resolution1=4),
            rt.ReflMaterial.default(12, device=device), max_bounces=3)
        values = [result.reflection_power]
    """,
    "trace_dfr_paths": """
        rx = torch.tensor([[0.0, 2.0, 2.0], [0.0, -3.0, -0.5], [0.0, 1.0, 3.0]], device=device)
        paths = scene.trace_dfr_paths(states.src[:1], rx, states, options=rt.DfrPathOptions(max_paths=6, wavelength=0.1),
                                      active=torch.ones(2, dtype=torch.bool, device=device))
        values = [paths.delay[paths.valid], paths.field_x_re[paths.valid]]
    """,
    "accum_dfr_direct": """
        values = [scene.accum_dfr_direct(states, grid, options=rt.DfrOptions(wavelength=0.1, direct_samples=64, seed=1)).power]
    """,
    "accum_dfr_coherent_direct": """
        values = [scene.accum_dfr_coherent_direct(states, grid, options=rt.DfrCoherentOptions(wavelength=0.1)).direct_field_x_re]
    """,
}

EPILOGUE = """
torch.cuda.synchronize()
print(json.dumps({"finite": all(bool(torch.isfinite(value).all()) for value in values)}))
"""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchOptixPipelineColdCreateTests(unittest.TestCase):
    def test_every_multipath_pipeline_cold_creates_in_a_fresh_process(self):
        for name, body in CASES.items():
            with self.subTest(operation=name):
                data = run_json_case(compose(PRELUDE, body, EPILOGUE), timeout=300)
                self.assertTrue(data["finite"])

    def test_large_multipath_pipelines_alternate_in_one_process(self):
        # On some drivers the diffraction-path and diffraction-accumulation pipelines cannot both live in one
        # OptiX context; whichever was created second used to fail optixPipelineCreate(multipath).
        order = ("accum_dfr_direct", "trace_dfr_paths", "accum_dfr_coherent_direct", "trace_dfr_paths")
        steps = []
        for index, name in enumerate(order):
            steps.append(CASES[name])
            steps.append(f"results_{index} = [value.detach().cpu() for value in values]")
        body = compose(*steps)
        epilogue = """
            torch.cuda.synchronize()
            outputs = [results_0, results_1, results_2, results_3]
            print(json.dumps({
                "finite": all(bool(torch.isfinite(value).all()) for result in outputs for value in result),
                "paths_repeat_equal": all(torch.equal(a, b) for a, b in zip(results_1, results_3)),
            }))
        """
        data = run_json_case(compose(PRELUDE, body, epilogue), timeout=300)
        self.assertTrue(data["finite"])
        self.assertTrue(data["paths_repeat_equal"])


if __name__ == "__main__":
    unittest.main()
