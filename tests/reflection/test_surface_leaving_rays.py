# Copyright Xingyu Chen.
# Tests that Torch rays leaving a surface never re-hit it, and that real close geometry still registers.

import unittest

import numpy as np
import torch

import rayd.torch as rt

COUNT = 20_000
TRACERS = ("optix", "cuda")
# The relative offset a ray leaving a surface starts at, as documented in ADR-0045 (2 * 2^-23 of the magnitude).
SPAWN_RELATIVE = 2.0 / 8388608.0


def rotation(seed):
    if seed is None:
        return np.eye(3)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def gpu(values):
    return torch.from_numpy(values).to("cuda")


def build(vertices, faces, tracer):
    scene = rt.Scene(trace_backend=tracer)
    scene.add_mesh(rt.Mesh(gpu(vertices.astype(np.float32)), gpu(faces.astype(np.int32))))
    scene.build()
    return scene


def ground(size, rotate, shift=0.0):
    corners = np.array([[-size, -size, 0.0], [3 * size, -size, 0.0], [-size, 3 * size, 0.0]])
    return corners @ rotate.T + shift, np.array([[0, 1, 2]])


def upward(rng, count, rotate, minimum=0.05):
    up = rng.standard_normal((count, 3))
    up[:, 2] = np.abs(up[:, 2]) + minimum
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    return up @ rotate.T


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchSurfaceLeavingRayTests(unittest.TestCase):
    def assert_no_self_hit(self, size, seed, shift=0.0):
        rotate = rotation(seed)
        rng = np.random.default_rng(3)
        vertices, faces = ground(size, rotate, shift)
        points = np.column_stack([rng.uniform(0, size, (COUNT, 2)), np.zeros(COUNT)]) @ rotate.T + shift
        up = upward(rng, COUNT, rotate)
        above = (points + up * size * 0.5).astype(np.float32)
        for tracer in TRACERS:
            with self.subTest(tracer=tracer, size=size, tilted=seed is not None, shift=shift):
                scene = build(vertices, faces, tracer)
                seen = scene.visible(gpu(points.astype(np.float32)), gpu(above)).cpu().numpy()
                self.assertEqual(int((~seen).sum()), 0, "a point on the ground must see a point above it")
                seen_back = scene.visible(gpu(above), gpu(points.astype(np.float32))).cpu().numpy()
                self.assertEqual(int((~seen_back).sum()), 0, "the segment end on the ground must not block it")
                chain = scene.trace_reflections(rt.Ray(gpu(above), gpu((-up).astype(np.float32))), 8)
                valid = chain.valid.cpu().numpy()
                self.assertEqual(int(valid[:, 0].sum()), COUNT, "every ray reaches the ground once")
                self.assertEqual(int(valid[:, 1].sum()), 0, "a ray reflected off a flat ground never returns to it")

    def test_flat_ground_at_every_scale(self):
        for size in (1.0, 100.0, 1000.0, 10000.0):
            self.assert_no_self_hit(size, None)

    def test_tilted_ground_at_every_scale(self):
        for size in (1.0, 100.0, 1000.0, 10000.0):
            self.assert_no_self_hit(size, 11)

    def test_small_triangle_far_from_the_origin(self):
        self.assert_no_self_hit(10.0, 11, shift=np.array([3000.0, -2000.0, 1000.0]))

    def test_coincident_faces_do_not_trap_the_ray(self):
        vertices, faces = ground(1000.0, rotation(5))
        faces = np.concatenate([faces, faces, faces[:, ::-1]])
        rotate = rotation(5)
        rng = np.random.default_rng(4)
        points = np.column_stack([rng.uniform(0, 1000.0, (COUNT, 2)), np.zeros(COUNT)]) @ rotate.T
        up = upward(rng, COUNT, rotate)
        above = (points + up * 500.0).astype(np.float32)
        for tracer in TRACERS:
            with self.subTest(tracer=tracer):
                scene = build(vertices, faces, tracer)
                self.assertEqual(
                    int((~scene.visible(gpu(points.astype(np.float32)), gpu(above)).cpu().numpy()).sum()), 0
                )
                valid = (
                    scene.trace_reflections(rt.Ray(gpu(above), gpu((-up).astype(np.float32))), 8).valid.cpu().numpy()
                )
                self.assertEqual(int(valid[:, 1].sum()), 0)

    def test_documented_origin_recipe_avoids_false_intersect_hits(self):
        # Scene.intersect() treats its origin as a free point; ADR-0045 gives callers this recipe for surface points.
        rotate = rotation(11)
        rng = np.random.default_rng(6)
        for size in (1.0, 1000.0, 10000.0):
            vertices, faces = ground(size, rotate)
            magnitude = np.abs(vertices).max()
            points = np.column_stack([rng.uniform(0, size, (COUNT, 2)), np.zeros(COUNT)]) @ rotate.T
            up = upward(rng, COUNT, rotate)
            normal = np.array([0.0, 0.0, 1.0]) @ rotate.T
            origin = (points + normal * SPAWN_RELATIVE * magnitude).astype(np.float32)
            for tracer in TRACERS:
                with self.subTest(tracer=tracer, size=size):
                    its = build(vertices, faces, tracer).intersect(rt.Ray(gpu(origin), gpu(up.astype(np.float32))))
                    self.assertEqual(int((its.prim_id >= 0).sum()), 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchCloseGeometryTests(unittest.TestCase):
    """What an oversized offset would break: real geometry right next to the surface a ray leaves."""

    # (gap, coordinate magnitude). The gap stays above the 2 * 2^-23 relative offset at that magnitude.
    PLATES = ((1e-3, 0.0), (1e-4, 0.0), (1e-3, 1000.0), (1e-3, 3000.0), (1e-4, 100.0), (1e-2, 6000.0))

    @staticmethod
    def plates(gap, magnitude, seed):
        rotate = rotation(seed)
        # Plate faces near the anchor keep the triangle's own magnitude close to the anchor's.
        size = 1.0
        corners = np.array([[-size, -size, 0.0], [3 * size, -size, 0.0], [-size, 3 * size, 0.0]])
        anchor = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0) * magnitude
        lower = corners @ rotate.T + anchor
        upper = (corners + np.array([0.0, 0.0, gap])) @ rotate.T + anchor
        return np.concatenate([lower, upper]), np.array([[0, 1, 2], [3, 4, 5]]), rotate, anchor

    def test_parallel_plates_keep_bouncing_between_each_other(self):
        for gap, magnitude in self.PLATES:
            vertices, faces, rotate, anchor = self.plates(gap, magnitude, None)
            rng = np.random.default_rng(1)
            count = 2000
            start = np.column_stack([rng.uniform(0.2, 0.3, (count, 2)), np.full(count, gap * 0.5)])
            direction = np.column_stack([rng.uniform(-1e-3, 1e-3, (count, 2)), -np.ones(count)])
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            origin = (start @ rotate.T + anchor).astype(np.float32)
            d = (direction @ rotate.T).astype(np.float32)
            for tracer in TRACERS:
                with self.subTest(gap=gap, magnitude=magnitude, tracer=tracer):
                    chain = build(vertices, faces, tracer).trace_reflections(rt.Ray(gpu(origin), gpu(d)), 4)
                    valid = chain.valid.cpu().numpy()
                    prim = chain.prim_ids.cpu().numpy()
                    self.assertEqual(int(valid[:, :4].sum()), count * 4, "every bounce lands on the facing plate")
                    self.assertTrue(np.all(prim[:, 0] == 0), "the first bounce lands on the lower plate")
                    self.assertTrue(np.all(prim[:, 1] == 1), "the next lands on the upper plate")
                    self.assertTrue(np.all(prim[:, 2] == 0))
                    self.assertTrue(np.all(prim[:, 3] == 1))

    def test_visibility_sees_a_plate_right_above_a_surface_point(self):
        for gap, magnitude in self.PLATES:
            vertices, faces, rotate, anchor = self.plates(gap, magnitude, None)
            count = 500
            rng = np.random.default_rng(2)
            on_lower = np.column_stack([rng.uniform(0.2, 0.3, (count, 2)), np.zeros(count)]) @ rotate.T + anchor
            above = on_lower + np.array([0.0, 0.0, 1.0]) @ rotate.T * max(10.0 * gap, 1.0)
            for tracer in TRACERS:
                with self.subTest(gap=gap, magnitude=magnitude, tracer=tracer):
                    scene = build(vertices, faces, tracer)
                    seen = scene.visible(gpu(on_lower.astype(np.float32)), gpu(above.astype(np.float32))).cpu().numpy()
                    self.assertEqual(int(seen.sum()), 0, "the upper plate blocks the segment leaving the lower one")

    def test_wall_receives_the_ray_reflected_next_to_the_corner(self):
        for distance, magnitude in ((1e-3, 0.0), (1e-3, 1000.0), (1e-2, 6000.0)):
            anchor = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0) * magnitude
            floor = np.array([[0.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 3.0, 0.0]])
            wall = np.array([[0.0, -1.0, 0.0], [0.0, 3.0, 0.0], [0.0, -1.0, 2.0]])
            vertices = np.concatenate([floor, wall]) + anchor
            faces = np.array([[0, 1, 2], [3, 4, 5]])
            count = 1000
            landing = np.column_stack([np.full(count, distance), np.linspace(0.2, 0.8, count), np.zeros(count)])
            # Travelling towards -x and -z: the floor reflects it to -x, straight into the wall.
            direction = np.tile(np.array([-1.0, 0.0, -1.0]) / np.sqrt(2.0), (count, 1))
            origin = (landing - direction * 0.5 + anchor).astype(np.float32)
            for tracer in TRACERS:
                with self.subTest(distance=distance, magnitude=magnitude, tracer=tracer):
                    chain = build(vertices, faces, tracer).trace_reflections(
                        rt.Ray(gpu(origin), gpu(direction.astype(np.float32))), 3
                    )
                    valid = chain.valid.cpu().numpy()
                    prim = chain.prim_ids.cpu().numpy()
                    self.assertEqual(int(valid[:, 1].sum()), count, "the wall right after the floor must register")
                    self.assertTrue(np.all(prim[:, 0] == 0))
                    self.assertTrue(np.all(prim[:, 1] == 1))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchSurfaceLeavingGradientTests(unittest.TestCase):
    """The spawn offset is not part of the differentiated path."""

    @staticmethod
    def corridor(height, shift, requires_grad):
        lower = np.array([[-4.0, -4.0, 0.0], [12.0, -4.0, 0.0], [-4.0, 12.0, 0.0]])
        upper = lower[[0, 2, 1]] + np.array([0.0, 0.0, height])
        vertices = torch.tensor(np.concatenate([lower, upper]) + shift, device="cuda", dtype=torch.float32)
        vertices.requires_grad_(requires_grad)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)))
        scene.build()
        return scene, vertices

    def test_second_bounce_length_and_gradients_follow_the_surface_path(self):
        for shift in (np.zeros(3), np.array([600.0, -400.0, 200.0])):
            with self.subTest(shift=float(np.abs(shift).max())):
                scene, vertices = self.corridor(2.0, shift, True)
                direction = torch.tensor([[0.3, 0.1, -1.0]], device="cuda", dtype=torch.float32)
                direction = direction / direction.norm()
                origin = torch.tensor(np.array([[0.5, 0.5, 1.0]]) + shift, device="cuda", dtype=torch.float32)
                origin.requires_grad_(True)
                direction.requires_grad_(True)
                chain = scene.trace_reflections(rt.Ray(origin, direction), 2)
                self.assertTrue(bool(chain.valid[0, 1]))
                # Between plates 2 apart the second leg has length 2 / |cos|, whatever the first leg did.
                expected = 2.0 / float(direction[0, 2].abs())
                tolerance = max(1e-4, 8.0 * SPAWN_RELATIVE * float(np.abs(shift).max()))
                self.assertAlmostEqual(float(chain.t[0, 1]), expected, delta=tolerance)
                chain.t[0, 1].backward()
                # The second leg does not depend on where the ray started, only on its direction and the plates.
                self.assertLess(float(origin.grad.abs().max()), 1e-2)
                self.assertTrue(bool(torch.isfinite(vertices.grad).all()))
                self.assertTrue(bool(torch.isfinite(direction.grad).all()))
                # Moving both plates together does not change the second leg.
                self.assertLess(float(vertices.grad.sum(dim=0)[2].abs()), 1e-2)


if __name__ == "__main__":
    unittest.main()
