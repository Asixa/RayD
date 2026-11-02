# Copyright Xingyu Chen.
# Tests first-order diffraction visibility against a known answer: one edge of a closed box, on both trace backends.

import math
import unittest

import numpy as np
import torch

import rayd.torch as rt

from tests.support.geometry import box_mesh

STATE_COUNT = 400
RX_COUNT = 40


def gpu(values, dtype=torch.float32):
    return torch.as_tensor(np.asarray(values), dtype=dtype, device="cuda")


def crosses_open_box(start, end, half=1.0, margin=1e-6):
    """Slab test of segments against the box's open interior."""
    delta = end - start
    limit = half - margin
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = (-limit - start) / delta, (limit - start) / delta
    enter = np.where(delta == 0, -np.inf, np.minimum(low, high)).max(-1)
    leave = np.where(delta == 0, np.inf, np.maximum(low, high)).min(-1)
    outside_parallel = ((delta == 0) & (np.abs(start) >= limit)).any(-1)
    return (np.maximum(enter, 0.0) < np.minimum(leave, 1.0)) & ~outside_parallel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchWedgeVisibilityTests(unittest.TestCase):
    """The edge along +x at (y=-1, z=1) is shared by the z=+1 and y=-1 faces of a closed box."""

    EDGE_A = np.array([-1.0, -1.0, 1.0])
    EDGE_DIR = np.array([1.0, 0.0, 0.0])

    @classmethod
    def points(cls, rng, count, inside_share=0.3):
        points = rng.uniform(-4.0, 4.0, (count, 3))
        inside = rng.random(count) < inside_share
        points[inside] = rng.uniform(-0.9, 0.9, (int(inside.sum()), 3))
        return points

    def trace(self, tracer, sources, receivers, shift):
        vertices, faces = box_mesh(torch.device("cuda"))
        scene = rt.Scene(trace_backend=tracer)
        scene.add_mesh(rt.Mesh(vertices + gpu(shift), faces))
        scene.build()
        count = len(sources)
        states = rt.DfrStates(
            edge_index=gpu(np.zeros(count), torch.int32),
            edge_pos=gpu(np.tile(self.EDGE_A + self.EDGE_DIR + shift, (count, 1))),
            edge_dir=gpu(np.tile(self.EDGE_DIR, (count, 1))),
            edge_t_min=gpu(np.full(count, -1.0)),
            edge_t_max=gpu(np.full(count, 1.0)),
            n0=gpu(np.tile([0.0, 0.0, 1.0], (count, 1))),
            n1=gpu(np.tile([0.0, -1.0, 0.0], (count, 1))),
            prim0=gpu(np.full(count, 2), torch.int32),
            prim1=gpu(np.full(count, 4), torch.int32),
            exterior_angle=gpu(np.full(count, 1.5 * math.pi)),
            src=gpu(sources + shift),
            src_power=gpu(np.ones(count)),
        )
        paths = scene.trace_dfr_paths(
            tx_positions=gpu(np.array([[0.0, 0.0, 5.0]]) + shift),
            rx_positions=gpu(receivers + shift),
            states=states,
            active=torch.ones(count, dtype=torch.bool, device="cuda"),
            max_paths=count * len(receivers),
            wavelength=0.1,
            layout=rt.DfrPathLayout.SourceLane,
        )
        return paths.valid.cpu().numpy().reshape(len(receivers), count)

    def visible_pairs(self, sources, receivers):
        visible = np.zeros((len(receivers), len(sources)), dtype=bool)
        for row, receiver in enumerate(receivers):
            along_source = (sources - self.EDGE_A) @ self.EDGE_DIR
            along_receiver = (receiver - self.EDGE_A) @ self.EDGE_DIR
            away_source = np.linalg.norm(sources - self.EDGE_A - np.outer(along_source, self.EDGE_DIR), axis=1)
            away_receiver = np.linalg.norm(receiver - self.EDGE_A - along_receiver * self.EDGE_DIR)
            keller = (along_source * away_receiver + along_receiver * away_source) / np.maximum(
                away_source + away_receiver, 1e-12
            )
            point = self.EDGE_A + np.clip(keller, 0.0, 2.0)[:, None] * self.EDGE_DIR
            target = np.tile(receiver, (len(sources), 1))
            visible[row] = ~crosses_open_box(sources, point) & ~crosses_open_box(point, target)
        return visible

    def test_paths_respect_the_solid_and_both_backends_agree(self):
        rng = np.random.default_rng(2)
        sources = self.points(rng, STATE_COUNT)
        receivers = self.points(rng, RX_COUNT)
        visible = self.visible_pairs(sources, receivers)
        for shift in (np.zeros(3), np.array([300.0, 240.0, 150.0])):
            with self.subTest(shift=float(np.abs(shift).max())):
                optix = self.trace("optix", sources, receivers, shift)
                cuda = self.trace("cuda", sources, receivers, shift)
                self.assertGreater(int(optix.sum()), 0)
                self.assertEqual(int((optix != cuda).sum()), 0, "OptiX and CUDA must find the same paths")
                self.assertEqual(int((optix & ~visible).sum()), 0, "a path whose leg crosses the solid is false")

    def test_receiver_inside_the_solid_gets_no_path(self):
        rng = np.random.default_rng(3)
        sources = rng.uniform(1.5, 4.0, (STATE_COUNT, 3)) * np.array([0.1, -1.0, 1.0])
        receivers = rng.uniform(-0.9, 0.9, (RX_COUNT, 3))
        for tracer in ("optix", "cuda"):
            with self.subTest(tracer=tracer):
                self.assertEqual(int(self.trace(tracer, sources, receivers, np.zeros(3)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
