"""GPU tests for the public `rayd.torch` SDF intersection entry point.

Phase 3b of `docs/dev/sdf_intersection_plan.md`. The oracle is the Phase 1
pure-PyTorch reference in `_sdf_reference.py`, which `test_sdf_reference.py`
already checks against closed-form intersections of the baked analytic fields,
so these tests compare the CUDA operation against it rather than re-deriving the
geometry. Both run in float32 on identical inputs, so the tolerances here are
float32 round-off tolerances, not discretisation ones.

Forward-mode parity uses the reference with an explicitly frozen tape:
`ref.march` is called on the primals outside the dual level and `ref.reattach`
inside it. That is the ADR-0037 frozen-winner contract; running `ref.intersect`
whole under a dual level instead lets tangents leak through the march and is not
the operation this primitive implements.
"""

from __future__ import annotations

import unittest

import torch
from torch import Tensor
from torch.autograd import forward_ad

from rayd.torch import SdfGrid, SdfIntersection, sdf_intersect

from . import _reference as ref
from .test_reference import IDENTITY_QUAT, bake, quat, sphere_hit_t, sphere_sdf, z_rays


REQUIRES_CUDA = unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
DEVICE = "cuda"
DTYPE = torch.float32
INPUT_NAMES = ("values", "position", "rotation", "scale", "origins", "directions")


# A sphere of radius 0.5 baked on a cubic grid spanning `[-1, 1]^3`.
def sphere_grid(size: int = 65) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    position = torch.zeros(3, device=DEVICE, dtype=DTYPE)
    rotation = torch.tensor(IDENTITY_QUAT, device=DEVICE, dtype=DTYPE)
    scale = torch.full((3,), 2.0, device=DEVICE, dtype=DTYPE)
    values = bake(
        lambda p: sphere_sdf(p, position, 0.5), (size, size, size), position, rotation, scale
    )
    return values, position, rotation, scale


# A rotated, non-uniformly scaled, off-centre box whose six inputs all matter,
# with three rays that hit the baked sphere well inside it.
def oriented_case(size: int = 65) -> tuple[Tensor, ...]:
    position = torch.tensor([0.1, -0.2, 0.05], device=DEVICE, dtype=DTYPE)
    rotation = 1.3 * quat((0.3, 0.5, 0.8), 0.7, position)  # deliberately unnormalized
    scale = torch.tensor([2.0, 1.4, 1.6], device=DEVICE, dtype=DTYPE)
    values = bake(
        lambda p: sphere_sdf(p, position, 0.45), (size, size, size), position, rotation, scale
    )
    origins = position + torch.tensor(
        [[0.05, -0.1, -2.0], [-0.12, 0.08, -2.0], [0.0, 0.15, -2.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    directions = 1.7 * torch.tensor(  # deliberately unnormalized
        [[0.02, 0.01, 1.0], [-0.03, 0.02, 1.0], [0.0, -0.04, 1.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    return values, position, rotation, scale, origins, directions


# Impact parameters sweeping through the tangent point of the radius-0.5 sphere.
def grazing_rays() -> tuple[Tensor, Tensor]:
    offsets = torch.linspace(0.45, 0.55, 21, device=DEVICE, dtype=DTYPE)
    origins = torch.stack(
        [offsets, torch.zeros_like(offsets), torch.full_like(offsets, -3.0)], dim=-1
    )
    directions = torch.zeros_like(origins)
    directions[:, 2] = 1.0
    return origins, directions


# Scalar objective touching every differentiable output of every ray.
def objective(result: SdfIntersection | ref.SdfIntersectionRef, weights: Tensor) -> Tensor:
    finite_t = torch.where(result.hit_mask, result.t, torch.zeros_like(result.t))
    rows = weights.unsqueeze(-1)
    return (
        (finite_t * weights).sum()
        + (result.position * rows).sum()
        + (result.normal * rows).sum()
    )


@REQUIRES_CUDA
class SdfIntersectForwardTests(unittest.TestCase):
    """Forward parity, miss semantics and determinism."""

    def test_forward_matches_the_reference(self) -> None:
        inputs = oriented_case()
        result = sdf_intersect(SdfGrid(*inputs[:4]), inputs[4], inputs[5])
        expected = ref.intersect(ref.SdfGridRef(*inputs[:4]), inputs[4], inputs[5])

        self.assertTrue(bool(result.hit_mask.all()))
        torch.testing.assert_close(result.t, expected.t, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(result.position, expected.position, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(result.normal, expected.normal, atol=1e-4, rtol=1e-5)
        self.assertTrue(torch.equal(result.steps, expected.steps))
        self.assertEqual(result.hit_mask.dtype, torch.bool)
        self.assertEqual(result.steps.dtype, torch.int32)

    def test_forward_matches_the_closed_form_sphere(self) -> None:
        grid = SdfGrid(*sphere_grid())
        origins, directions = z_rays([0.0, 0.1, 0.2, 0.3], DEVICE, DTYPE)

        result = sdf_intersect(grid, origins, directions)
        self.assertTrue(bool(result.hit_mask.all()))
        torch.testing.assert_close(
            result.t, sphere_hit_t(origins, directions, grid.position, 0.5), atol=1e-3, rtol=0.0
        )
        torch.testing.assert_close(
            result.normal,
            result.position / result.position.norm(dim=-1, keepdim=True),
            atol=5e-2,
            rtol=0.0,
        )

    def test_ray_starting_inside_marches_outward(self) -> None:
        grid = SdfGrid(*sphere_grid())
        origins = torch.zeros(1, 3, device=DEVICE, dtype=DTYPE)
        directions = torch.tensor([[0.0, 0.0, 1.0]], device=DEVICE, dtype=DTYPE)

        result = sdf_intersect(grid, origins, directions)
        self.assertTrue(bool(result.hit_mask.all()))
        torch.testing.assert_close(
            result.t, torch.full((1,), 0.5, device=DEVICE), atol=1e-3, rtol=0.0
        )
        # A signed distance field's gradient points outward on both sides.
        torch.testing.assert_close(result.normal, directions, atol=5e-2, rtol=0.0)

    def test_missed_lanes_are_bitwise_inert(self) -> None:
        grid = SdfGrid(*sphere_grid())
        # Ray 0 hits; ray 1 crosses the box but misses the sphere; ray 2 never
        # overlaps the box at all; ray 3 is clipped short of the sphere by tmax.
        origins, directions = z_rays([0.0, 0.7, 5.0, 0.0], DEVICE, DTYPE)

        result = sdf_intersect(grid, origins, directions)
        clipped = sdf_intersect(grid, origins[3:], directions[3:], tmax=1.0)
        torch.testing.assert_close(
            result.hit_mask, torch.tensor([True, False, False, True], device=DEVICE)
        )
        self.assertFalse(bool(clipped.hit_mask.any()))

        missed = torch.cat([result.t[1:3], clipped.t])
        self.assertTrue(bool((missed == float("inf")).all()))
        zero = torch.zeros(3, 3, device=DEVICE)
        rows = torch.cat([result.position[1:3], clipped.position])
        normals = torch.cat([result.normal[1:3], clipped.normal])
        self.assertTrue(torch.equal(rows, zero))
        self.assertTrue(torch.equal(normals, zero))
        self.assertFalse(bool(rows.signbit().any()))
        self.assertFalse(bool(normals.signbit().any()))
        # A lane the box clip rejects evaluates the field zero times.
        self.assertEqual(int(result.steps[2]), 0)

    def test_grazing_rays_leave_no_nan_or_negative_infinity(self) -> None:
        grid = SdfGrid(*sphere_grid())
        origins, directions = grazing_rays()

        result = sdf_intersect(grid, origins, directions)
        self.assertTrue(bool(result.hit_mask.any()))
        self.assertFalse(bool(result.hit_mask.all()))
        self.assertFalse(bool(result.t.isnan().any()))
        self.assertTrue(bool((result.t[result.hit_mask]).isfinite().all()))
        self.assertTrue(bool((result.t[~result.hit_mask] == float("inf")).all()))
        self.assertTrue(bool(result.position.isfinite().all()))
        self.assertTrue(bool(result.normal.isfinite().all()))

    def test_empty_batch_and_no_grad_request_stay_on_the_forward_path(self) -> None:
        tracked = [value.detach().clone().requires_grad_(True) for value in sphere_grid(17)]
        grid = SdfGrid(*tracked)
        empty = torch.zeros((0, 3), device=DEVICE, dtype=DTYPE)

        result = sdf_intersect(grid, empty, empty)
        self.assertEqual(tuple(result.t.shape), (0,))
        self.assertEqual(tuple(result.position.shape), (0, 3))

        origins, directions = z_rays([0.0], DEVICE, DTYPE)
        with torch.no_grad():
            detached = sdf_intersect(grid, origins, directions)
        self.assertFalse(detached.t.requires_grad)
        self.assertTrue(bool(detached.hit_mask.all()))

    def test_forward_is_deterministic_and_path_independent(self) -> None:
        inputs = oriented_case()
        origins, directions = inputs[4], inputs[5]
        first = sdf_intersect(SdfGrid(*inputs[:4]), origins, directions)
        for _ in range(3):
            repeat = sdf_intersect(SdfGrid(*inputs[:4]), origins, directions)
            for name in ("t", "hit_mask", "position", "normal", "steps"):
                self.assertTrue(torch.equal(getattr(first, name), getattr(repeat, name)), name)

        # The autograd path must not change a single bit of the forward result.
        tracked = [value.detach().clone().requires_grad_(True) for value in inputs]
        tracked_result = sdf_intersect(SdfGrid(*tracked[:4]), tracked[4], tracked[5])
        for name in ("t", "hit_mask", "position", "normal", "steps"):
            expected = getattr(first, name)
            actual = getattr(tracked_result, name).detach()
            self.assertTrue(torch.equal(expected, actual), name)


@REQUIRES_CUDA
class SdfIntersectGradientTests(unittest.TestCase):
    """Reverse-mode parity, gradcheck, and missed-lane gradient inertness."""

    # Gradients of `objective` for every one of the six supported inputs. The
    # reference result exposes the same field names, so it feeds the same
    # objective as the native one.
    def gradients(
        self, inputs: tuple[Tensor, ...], weights: Tensor, native: bool
    ) -> list[Tensor]:
        tracked = [value.detach().clone().requires_grad_(True) for value in inputs]
        grid = tracked[:4]
        if native:
            result = sdf_intersect(SdfGrid(*grid), tracked[4], tracked[5])
        else:
            result = ref.intersect(ref.SdfGridRef(*grid), tracked[4], tracked[5])
        objective(result, weights).backward()
        return [value.grad for value in tracked]

    def test_gradients_match_the_reference(self) -> None:
        inputs = oriented_case()
        weights = torch.tensor([1.0, 2.0, 3.0], device=DEVICE, dtype=DTYPE)
        for name, expected, actual in zip(
            INPUT_NAMES,
            self.gradients(inputs, weights, False),
            self.gradients(inputs, weights, True),
        ):
            with self.subTest(input=name):
                self.assertGreater(float(expected.abs().max()), 1e-3)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-4 * float(expected.abs().max()),
                    rtol=1e-4,
                )

    def test_gradcheck_on_a_small_grid(self) -> None:
        """A plane field is reproduced exactly by trilinear interpolation, so the
        numerical jacobian is limited by float32 round-off rather than by the
        baking error, which is what makes float32 gradcheck meaningful here."""
        position = torch.zeros(3, device=DEVICE, dtype=DTYPE)
        rotation = torch.tensor(IDENTITY_QUAT, device=DEVICE, dtype=DTYPE)
        scale = torch.full((3,), 2.0, device=DEVICE, dtype=DTYPE)
        values = bake(lambda p: p[..., 2] - 0.1, (5, 5, 5), position, rotation, scale)
        origins = torch.tensor(
            [[0.05, -0.1, -3.0], [-0.12, 0.08, -3.0]], device=DEVICE, dtype=DTYPE
        )
        directions = torch.tensor(
            [[0.02, 0.01, 1.0], [-0.03, 0.02, 1.0]], device=DEVICE, dtype=DTYPE
        )
        inputs = tuple(
            value.detach().clone().requires_grad_(True)
            for value in (values, position, rotation, scale, origins, directions)
        )

        def evaluate(*tensors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            result = sdf_intersect(
                SdfGrid(*tensors[:4]), tensors[4], tensors[5], eps_hit=1e-6, max_steps=128
            )
            return result.t, result.position, result.normal

        self.assertTrue(
            torch.autograd.gradcheck(
                evaluate, inputs, eps=1e-3, atol=5e-3, rtol=5e-3, nondet_tol=1e-4
            )
        )

    def test_missed_lanes_contribute_no_gradient(self) -> None:
        grid_inputs = sphere_grid()
        origins, directions = z_rays([0.7, 0.0, 5.0], DEVICE, DTYPE)  # miss, hit, miss
        mixed = self.gradients(
            (*grid_inputs, origins, directions), torch.ones(3, device=DEVICE), True
        )
        hit_only = self.gradients(
            (*grid_inputs, origins[1:2], directions[1:2]), torch.ones(1, device=DEVICE), True
        )

        zero = torch.zeros(2, 3, device=DEVICE)
        for name, gradient in zip(("origins", "directions"), mixed[4:]):
            with self.subTest(input=name):
                self.assertTrue(torch.equal(gradient[[0, 2]], zero))
                self.assertFalse(bool(gradient[[0, 2]].signbit().any()))
                self.assertTrue(torch.equal(gradient[1], hit_only[INPUT_NAMES.index(name)][0]))
        # The hitting ray is the only atomic contributor, so every shared
        # gradient must be bit-identical to the batch that contains it alone.
        for name, gradient, expected in zip(INPUT_NAMES[:4], mixed[:4], hit_only[:4]):
            with self.subTest(input=name):
                self.assertTrue(torch.equal(gradient, expected))

    def test_grazing_gradients_stay_finite(self) -> None:
        tracked = [value.detach().clone().requires_grad_(True) for value in sphere_grid()]
        origins, directions = grazing_rays()
        rays = [value.detach().clone().requires_grad_(True) for value in (origins, directions)]

        result = sdf_intersect(SdfGrid(*tracked), rays[0], rays[1])
        objective(result, torch.ones(origins.shape[0], device=DEVICE)).backward()
        for name, tensor in zip(INPUT_NAMES, tracked + rays):
            with self.subTest(input=name):
                self.assertTrue(bool(tensor.grad.isfinite().all()))


@REQUIRES_CUDA
class SdfIntersectForwardModeTests(unittest.TestCase):
    """Forward-mode parity against the frozen tape, and duality with backward."""

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.inputs = oriented_case()
        self.tangents = tuple(torch.randn_like(value) * 0.1 for value in self.inputs)

    # Tangents of the three differentiable outputs under one dual level.
    def push_forward(self, native: bool) -> list[Tensor]:
        tape = ref.march(
            ref.SdfGridRef(*self.inputs[:4]), self.inputs[4], self.inputs[5], ref.TraceConfig()
        )
        with forward_ad.dual_level():
            duals = [
                forward_ad.make_dual(value, tangent)
                for value, tangent in zip(self.inputs, self.tangents)
            ]
            if native:
                result = sdf_intersect(SdfGrid(*duals[:4]), duals[4], duals[5])
            else:
                result = ref.reattach(ref.SdfGridRef(*duals[:4]), duals[4], duals[5], tape)
            return [
                forward_ad.unpack_dual(field).tangent.clone()
                for field in (result.t, result.position, result.normal)
            ]

    def test_forward_mode_matches_the_frozen_tape_reference(self) -> None:
        for name, expected, actual in zip(
            ("t", "position", "normal"), self.push_forward(False), self.push_forward(True)
        ):
            with self.subTest(output=name):
                self.assertGreater(float(expected.abs().max()), 1e-3)
                torch.testing.assert_close(
                    actual, expected, atol=1e-4 * float(expected.abs().max()), rtol=1e-4
                )

    def test_forward_and_reverse_modes_are_duals(self) -> None:
        tangent_t, tangent_position, tangent_normal = self.push_forward(True)
        tracked = [value.detach().clone().requires_grad_(True) for value in self.inputs]
        result = sdf_intersect(SdfGrid(*tracked[:4]), tracked[4], tracked[5])
        cotangents = (
            torch.randn_like(result.t),
            torch.randn_like(result.position),
            torch.randn_like(result.normal),
        )
        (
            (result.t * cotangents[0]).sum()
            + (result.position * cotangents[1]).sum()
            + (result.normal * cotangents[2]).sum()
        ).backward()

        pushed = sum(
            float((cotangent * tangent).sum())
            for cotangent, tangent in zip(
                cotangents, (tangent_t, tangent_position, tangent_normal)
            )
        )
        pulled = sum(
            float((value.grad * tangent).sum())
            for value, tangent in zip(tracked, self.tangents)
        )
        self.assertAlmostEqual(pushed, pulled, delta=1e-4 * max(abs(pushed), 1.0))

    def test_missed_lanes_have_zero_tangents(self) -> None:
        grid_inputs = sphere_grid()
        origins, directions = z_rays([0.0, 0.7, 5.0], DEVICE, DTYPE)
        with forward_ad.dual_level():
            duals = [
                forward_ad.make_dual(value, torch.randn_like(value) * 0.1)
                for value in (*grid_inputs, origins, directions)
            ]
            result = sdf_intersect(SdfGrid(*duals[:4]), duals[4], duals[5])
            tangents = [
                forward_ad.unpack_dual(field).tangent
                for field in (result.t, result.position, result.normal)
            ]
            self.assertTrue(torch.equal(tangents[0][1:], torch.zeros(2, device=DEVICE)))
            for tangent in tangents[1:]:
                self.assertTrue(torch.equal(tangent[1:], torch.zeros(2, 3, device=DEVICE)))


@REQUIRES_CUDA
class SdfIntersectValidationTests(unittest.TestCase):
    """Structural validation, which never reads a device value."""

    def setUp(self) -> None:
        self.values, self.position, self.rotation, self.scale = sphere_grid(size=17)
        self.origins, self.directions = z_rays([0.0], DEVICE, DTYPE)

    def grid(self, **overrides: Tensor) -> SdfGrid:
        fields = {
            "values": self.values,
            "position": self.position,
            "rotation": self.rotation,
            "scale": self.scale,
        }
        fields.update(overrides)
        return SdfGrid(**fields)

    def test_grid_rejects_a_host_tensor(self) -> None:
        with self.assertRaisesRegex(TypeError, "SdfGrid.values must be a CUDA tensor"):
            self.grid(values=self.values.cpu())

    def test_grid_rejects_a_double_tensor(self) -> None:
        with self.assertRaisesRegex(TypeError, "SdfGrid.scale must be torch.float32"):
            self.grid(scale=self.scale.double())

    def test_grid_rejects_a_strided_view(self) -> None:
        with self.assertRaisesRegex(ValueError, "SdfGrid.values must be contiguous"):
            self.grid(values=self.values[::2])

    def test_grid_rejects_a_wrong_rank_or_extent(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \(Nx, Ny, Nz\)"):
            self.grid(values=self.values[0])
        thin = torch.zeros((4, 1, 4), device=DEVICE, dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "at least 2 samples on every axis"):
            self.grid(values=thin)

    def test_grid_rejects_a_wrong_length_quaternion(self) -> None:
        with self.assertRaisesRegex(ValueError, r"SdfGrid.rotation must have shape \(4,\)"):
            self.grid(rotation=self.rotation[:3].contiguous())

    def test_grid_rejects_a_host_placement(self) -> None:
        with self.assertRaisesRegex(TypeError, "SdfGrid.position must be a CUDA tensor"):
            self.grid(position=self.position.cpu())

    def test_rays_must_be_resident_contiguous_and_shaped(self) -> None:
        grid = self.grid()
        with self.assertRaisesRegex(TypeError, "origins must be a CUDA tensor"):
            sdf_intersect(grid, self.origins.cpu(), self.directions)
        with self.assertRaisesRegex(TypeError, "directions must be torch.float32"):
            sdf_intersect(grid, self.origins, self.directions.double())
        with self.assertRaisesRegex(ValueError, "origins must be contiguous"):
            sdf_intersect(grid, self.origins.expand(4, 3), self.directions)
        with self.assertRaisesRegex(ValueError, r"directions must have shape \(N, 3\)"):
            sdf_intersect(grid, self.origins, self.directions.reshape(-1))
        with self.assertRaisesRegex(ValueError, "same ray count"):
            sdf_intersect(grid, self.origins, self.directions.repeat(2, 1))

    def test_march_parameters_are_validated(self) -> None:
        grid = self.grid()
        with self.assertRaisesRegex(ValueError, "tmax must be positive"):
            sdf_intersect(grid, self.origins, self.directions, tmax=0.0)
        with self.assertRaisesRegex(ValueError, "max_steps must be at least 1"):
            sdf_intersect(grid, self.origins, self.directions, max_steps=0)
        with self.assertRaisesRegex(ValueError, r"relaxation must lie in \(0, 1\]"):
            sdf_intersect(grid, self.origins, self.directions, relaxation=1.5)
        with self.assertRaisesRegex(ValueError, "eps_hit must be positive"):
            sdf_intersect(grid, self.origins, self.directions, eps_hit=0.0)


if __name__ == "__main__":
    unittest.main()
