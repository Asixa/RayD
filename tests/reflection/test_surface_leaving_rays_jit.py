# Copyright Xingyu Chen.
# Tests that Dr.Jit rays leaving a surface never re-hit it, and that real close geometry still registers.

import sys
import unittest
from pathlib import Path

from tests.support.subprocess_cases import run_json_case

SUBPROCESS_CWD = Path(sys.executable).resolve().parent

PRELUDE = r"""
import json
import numpy as np
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as rd

COUNT = 20000


def rotation(seed):
    if seed is None:
        return np.eye(3)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def arr(values):
    return cuda.Array3f(*np.asarray(values, dtype=np.float32).T)


def build(vertices, faces, tracer):
    mesh = rd.Mesh(arr(vertices), cuda.Array3i(*np.asarray(faces, dtype=np.int32).T))
    mesh.build()
    scene = rd.Scene(trace_backend=tracer)
    scene.add_mesh(mesh)
    scene.build()
    return scene
"""

SELF_HIT = (
    PRELUDE
    + r"""
report = {}
for tilted in (None, 11):
    rotate = rotation(tilted)
    for size in (1.0, 100.0, 1000.0, 10000.0):
        rng = np.random.default_rng(3)
        vertices = np.array([[-size, -size, 0.0], [3 * size, -size, 0.0], [-size, 3 * size, 0.0]]) @ rotate.T
        points = np.column_stack([rng.uniform(0, size, (COUNT, 2)), np.zeros(COUNT)]) @ rotate.T
        up = rng.standard_normal((COUNT, 3))
        up[:, 2] = np.abs(up[:, 2]) + 0.05
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        up = up @ rotate.T
        above = points + up * size * 0.5
        for tracer in ("optix", "cuda"):
            scene = build(vertices, [[0, 1, 2]], tracer)
            blocked = ~np.array(scene.visible(arr(points), arr(above)).visible)
            eager = scene.trace_reflections(rd.Ray(arr(above), arr(-up)), 8, symbolic=False)
            dr.eval(eager.bounce_count)
            key = f"{tilted}/{size}/{tracer}"
            report[key + "/blocked"] = int(blocked.sum())
            report[key + "/eager_extra"] = int((np.array(eager.bounce_count) >= 2).sum())
            report[key + "/eager_first"] = int((np.array(eager.bounce_count) >= 1).sum()) - COUNT
            if tracer == "optix":
                symbolic = scene.trace_reflections(rd.Ray(arr(above), arr(-up)), 4)
                dr.eval(symbolic.bounce_count)
                report[key + "/symbolic_extra"] = int((np.array(symbolic.bounce_count) >= 2).sum())
print(json.dumps(report))
"""
)

CLOSE_GEOMETRY = (
    PRELUDE
    + r"""
report = {}
for gap, magnitude in ((1e-3, 0.0), (1e-4, 0.0), (1e-3, 1000.0), (1e-3, 3000.0), (1e-4, 100.0), (1e-2, 6000.0)):
    anchor = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0) * magnitude
    corners = np.array([[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]])
    vertices = np.concatenate([corners + anchor, corners + [0.0, 0.0, gap] + anchor])
    count = 2000
    rng = np.random.default_rng(1)
    origin = np.column_stack([rng.uniform(0.2, 0.3, (count, 2)), np.full(count, gap * 0.5)]) + anchor
    direction = np.column_stack([rng.uniform(-1e-3, 1e-3, (count, 2)), -np.ones(count)])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    on_lower = np.column_stack([rng.uniform(0.2, 0.3, (count, 2)), np.zeros(count)]) + anchor
    above = on_lower + [0.0, 0.0, max(10.0 * gap, 1.0)]
    for tracer in ("optix", "cuda"):
        scene = build(vertices, [[0, 1, 2], [3, 4, 5]], tracer)
        chain = scene.trace_reflections(rd.Ray(arr(origin), arr(direction)), 4, symbolic=False)
        dr.eval(chain.bounce_count)
        key = f"{gap}/{magnitude}/{tracer}"
        report[key + "/missing_bounces"] = int((np.array(chain.bounce_count) < 4).sum())
        report[key + "/not_blocked"] = int(np.array(scene.visible(arr(on_lower), arr(above)).visible).sum())

for distance, magnitude in ((1e-3, 0.0), (1e-3, 1000.0), (1e-2, 6000.0)):
    anchor = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0) * magnitude
    floor = np.array([[0.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 3.0, 0.0]])
    wall = np.array([[0.0, -1.0, 0.0], [0.0, 3.0, 0.0], [0.0, -1.0, 2.0]])
    count = 1000
    landing = np.column_stack([np.full(count, distance), np.linspace(0.2, 0.8, count), np.zeros(count)])
    direction = np.tile(np.array([-1.0, 0.0, -1.0]) / np.sqrt(2.0), (count, 1))
    origin = landing - direction * 0.5 + anchor
    for tracer in ("optix", "cuda"):
        scene = build(np.concatenate([floor, wall]) + anchor, [[0, 1, 2], [3, 4, 5]], tracer)
        chain = scene.trace_reflections(rd.Ray(arr(origin), arr(direction)), 3, symbolic=False)
        dr.eval(chain.bounce_count)
        report[f"corner/{distance}/{magnitude}/{tracer}/missing"] = int((np.array(chain.bounce_count) < 2).sum())
print(json.dumps(report))
"""
)

GRADIENT = (
    PRELUDE
    + r"""
import drjit.cuda.ad as cad

report = {}
for shift in ([0.0, 0.0, 0.0], [600.0, -400.0, 200.0]):
    lower = np.array([[-4.0, -4.0, 0.0], [12.0, -4.0, 0.0], [-4.0, 12.0, 0.0]])
    upper = lower[[0, 2, 1]] + [0.0, 0.0, 2.0]
    vertices = np.concatenate([lower, upper]) + shift
    scene = build(vertices, [[0, 1, 2], [3, 4, 5]], "optix")
    direction = np.array([[0.3, 0.1, -1.0]])
    direction /= np.linalg.norm(direction)
    origin = cad.Array3f(*(np.array([[0.5, 0.5, 1.0]]) + shift).astype(np.float32).T)
    dr.enable_grad(origin)
    chain = scene.trace_reflections(rd.RayAD(origin, cad.Array3f(*direction.astype(np.float32).T)), 2, symbolic=False)
    t = dr.gather(type(chain.t), chain.t, cad.UInt32([1]))
    dr.backward(t)
    key = str(max(abs(v) for v in shift))
    report[key + "/t"] = float(t[0])
    report[key + "/expected"] = 2.0 / float(abs(direction[0, 2]))
    report[key + "/grad_o"] = float(np.abs(np.array(dr.grad(origin))).max())
print(json.dumps(report))
"""
)


class DrJitSurfaceLeavingRayTests(unittest.TestCase):
    def test_flat_and_tilted_ground_at_every_scale(self):
        report = run_json_case(SELF_HIT, timeout=600, cwd=SUBPROCESS_CWD)
        for key, count in report.items():
            self.assertEqual(count, 0, key)

    def test_close_geometry_still_registers(self):
        report = run_json_case(CLOSE_GEOMETRY, timeout=600, cwd=SUBPROCESS_CWD)
        for key, count in report.items():
            self.assertEqual(count, 0, key)

    def test_second_bounce_length_and_gradient_follow_the_surface_path(self):
        report = run_json_case(GRADIENT, timeout=600, cwd=SUBPROCESS_CWD)
        for key in ("0.0", "600.0"):
            self.assertAlmostEqual(report[key + "/t"], report[key + "/expected"], delta=3e-3)
            self.assertLess(report[key + "/grad_o"], 1e-2)


if __name__ == "__main__":
    unittest.main()
