"""Golden tests for the Phase 1 pure-PyTorch SDF intersection reference.

The reference in `_sdf_reference.py` is the oracle the Phase 3 CUDA kernels will
be checked against, so it is checked here against something independent of it:
closed-form ray/sphere and ray/box intersections of the analytic fields that were
baked onto the grids, `torch.nn.functional.grid_sample` for the interpolant,
`torch.autograd.gradcheck` for the frozen-tape derivative expression, and central
finite differences of a re-marched forward pass for every one of the six
gradient inputs ADR-0037 supports.

Forward tolerances are set by the trilinear discretisation error of the baked
field, not by float32 round-off: a sphere baked on a grid of edge `h` is
represented to `O(h^2)` in value and `O(h)` in gradient, so `t` is checked at
`1e-3` and the normal direction at `5e-2` on the `97^3` grids used here. The one
exception is the box face test, where the field is exactly linear in the
traversed region and the interpolant is therefore exact.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from . import _sdf_reference as ref


REQUIRES_CUDA = unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


# Analytic world-space SDF of a sphere.
def sphere_sdf(points: Tensor, centre: Tensor, radius: float) -> Tensor:
    return (points - centre).norm(dim=-1) - radius


# Analytic world-space SDF of an axis-aligned box given by its half extents.
def box_sdf(points: Tensor, centre: Tensor, half: Tensor) -> Tensor:
    q = (points - centre).abs() - half
    return q.clamp_min(0.0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0.0)


# Scalar-first quaternion of an axis/angle rotation.
def quat(axis: tuple[float, float, float], angle: float, like: Tensor) -> Tensor:
    norm = math.sqrt(sum(a * a for a in axis))
    half = 0.5 * angle
    parts = [math.cos(half)] + [math.sin(half) * a / norm for a in axis]
    return like.new_tensor(parts)


# Bake an analytic world-space SDF onto the vertex-centred grid of an oriented box.
def bake(
    field: Callable[[Tensor], Tensor],
    shape: tuple[int, int, int],
    position: Tensor,
    rotation: Tensor,
    scale: Tensor,
) -> Tensor:
    position, rotation, scale = position.detach(), rotation.detach(), scale.detach()
    axes = [
        torch.linspace(-0.5, 0.5, n, dtype=position.dtype, device=position.device) * s
        for n, s in zip(shape, scale)
    ]
    local = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    world = local @ ref.rotation_matrix(rotation).transpose(0, 1) + position
    # Baked values are caller data, never a graph: an analytic field that closes
    # over a placement tensor would otherwise smuggle a second derivative path.
    return field(world).reshape(shape).detach().contiguous()


# Smallest positive ray/sphere root, or `+inf` when the ray misses.
def sphere_hit_t(
    origins: Tensor, directions: Tensor, centre: Tensor, radius: float
) -> Tensor:
    w = directions / directions.norm(dim=-1, keepdim=True)
    oc = origins - centre
    b = (oc * w).sum(-1)
    disc = b * b - (oc * oc).sum(-1) + radius * radius
    root = disc.clamp_min(0.0).sqrt()
    near, far = -b - root, -b + root
    t = torch.where(near > 0, near, far)
    return torch.where((disc > 0) & (far > 0), t, torch.full_like(t, float("inf")))


# A sphere of radius 0.5 baked on a cubic grid spanning `[-1, 1]^3`.
def sphere_case(device: str, dtype: torch.dtype, size: int = 97) -> ref.SdfGridRef:
    position = torch.zeros(3, device=device, dtype=dtype)
    rotation = torch.tensor(IDENTITY_QUAT, device=device, dtype=dtype)
    scale = torch.full((3,), 2.0, device=device, dtype=dtype)
    values = bake(
        lambda p: sphere_sdf(p, position, 0.5), (size, size, size), position, rotation, scale
    )
    return ref.SdfGridRef(values, position, rotation, scale)


# Rays fired along `+z` from `z = -3` at the given transverse offsets.
def z_rays(offsets: list[float], device: str, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    origins = torch.tensor(
        [[b, 0.0, -3.0] for b in offsets], device=device, dtype=dtype
    )
    directions = torch.zeros_like(origins)
    directions[:, 2] = 1.0
    return origins, directions


@REQUIRES_CUDA
class SdfReferenceForwardTests(unittest.TestCase):
    """Forward accuracy against the closed-form solution of the baked field."""

    device = "cuda"
    dtype = torch.float32

    def test_sphere_hits_match_the_closed_form(self) -> None:
        grid = sphere_case(self.device, self.dtype)
        origins, directions = z_rays([0.0, 0.1, 0.2, 0.3], self.device, self.dtype)
        res = ref.intersect(grid, origins, directions)

        self.assertTrue(bool(res.hit_mask.all()))
        expected = sphere_hit_t(origins, directions, grid.position, 0.5)
        torch.testing.assert_close(res.t, expected, atol=1e-3, rtol=0.0)
        torch.testing.assert_close(
            res.position, origins + res.t.unsqueeze(-1) * directions, atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            res.normal,
            res.position / res.position.norm(dim=-1, keepdim=True),
            atol=5e-2,
            rtol=0.0,
        )
        torch.testing.assert_close(
            res.normal.norm(dim=-1), torch.ones(4, device=self.device), atol=1e-5, rtol=0.0
        )
        self.assertEqual(res.steps.dtype, torch.int32)
        self.assertTrue(bool((res.steps > 0).all()))

    def test_box_face_hit_is_exact_for_a_linear_field(self) -> None:
        """The box SDF is linear along `z` where these rays travel, and trilinear
        interpolation reproduces a linear function exactly, so the only error left
        is the march tolerance."""
        position = torch.zeros(3, device=self.device, dtype=self.dtype)
        rotation = torch.tensor(IDENTITY_QUAT, device=self.device, dtype=self.dtype)
        scale = torch.full((3,), 2.0, device=self.device, dtype=self.dtype)
        half = torch.full((3,), 0.4, device=self.device, dtype=self.dtype)
        values = bake(
            lambda p: box_sdf(p, position, half), (65, 65, 65), position, rotation, scale
        )
        grid = ref.SdfGridRef(values, position, rotation, scale)
        origins, directions = z_rays([0.0, 0.1], self.device, self.dtype)

        res = ref.intersect(grid, origins, directions)
        self.assertTrue(bool(res.hit_mask.all()))
        torch.testing.assert_close(
            res.t, torch.full((2,), 2.6, device=self.device), atol=1e-4, rtol=0.0
        )
        torch.testing.assert_close(
            res.normal,
            torch.tensor([[0.0, 0.0, -1.0]] * 2, device=self.device),
            atol=1e-5,
            rtol=0.0,
        )

    def test_ray_starting_inside_marches_outward(self) -> None:
        grid = sphere_case(self.device, self.dtype)
        origins = torch.zeros(1, 3, device=self.device, dtype=self.dtype)
        directions = torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=self.dtype)

        res = ref.intersect(grid, origins, directions)
        self.assertTrue(bool(res.hit_mask.all()))
        torch.testing.assert_close(
            res.t, torch.full((1,), 0.5, device=self.device), atol=1e-3, rtol=0.0
        )
        # The gradient of a signed distance field points outward on both sides.
        torch.testing.assert_close(
            res.normal, directions, atol=5e-2, rtol=0.0
        )

    def test_missed_lanes_are_bitwise_inert(self) -> None:
        grid = sphere_case(self.device, self.dtype)
        values = grid.values.clone().requires_grad_(True)
        grid = ref.SdfGridRef(values, grid.position, grid.rotation, grid.scale)
        # Ray 0 hits; ray 1 stays inside the box but misses the sphere; ray 2
        # never overlaps the box at all.
        origins, directions = z_rays([0.0, 0.7, 5.0], self.device, self.dtype)

        res = ref.intersect(grid, origins, directions)
        torch.testing.assert_close(
            res.hit_mask, torch.tensor([True, False, False], device=self.device)
        )
        missed = res.t[1:].detach()
        self.assertTrue(bool((missed == float("inf")).all()))
        zero = torch.zeros(2, 3, device=self.device)
        self.assertTrue(torch.equal(res.position[1:], zero))
        self.assertTrue(torch.equal(res.normal[1:], zero))
        self.assertFalse(bool(res.position[1:].signbit().any()))
        self.assertFalse(bool(res.normal[1:].signbit().any()))

        res.position.sum().backward()
        mixed = values.grad.clone()

        values.grad = None
        origins_hit, directions_hit = z_rays([0.0], self.device, self.dtype)
        ref.intersect(grid, origins_hit, directions_hit).position.sum().backward()
        self.assertTrue(torch.equal(mixed, values.grad))

    def test_rejected_obb_lanes_perform_no_field_evaluation(self) -> None:
        grid = sphere_case(self.device, self.dtype, size=33)
        origins, directions = z_rays([0.0, 5.0], self.device, self.dtype)
        tape = ref.march(grid, origins, directions, ref.TraceConfig())
        self.assertGreater(int(tape.steps[0]), 0)
        self.assertEqual(int(tape.steps[1]), 0)

    def test_tmax_clips_the_traced_interval(self) -> None:
        grid = sphere_case(self.device, self.dtype, size=33)
        origins, directions = z_rays([0.0], self.device, self.dtype)

        self.assertTrue(bool(ref.intersect(grid, origins, directions).hit_mask.all()))
        clipped = ref.intersect(
            grid, origins, directions, ref.TraceConfig(tmax=2.0)
        )
        self.assertFalse(bool(clipped.hit_mask.any()))
        self.assertEqual(float(clipped.t[0]), float("inf"))

    def test_non_eikonal_field_recovers_through_bisection(self) -> None:
        """A field scaled by two overshoots every relaxed step; the sign-flip
        bracket has to bring the march back onto the same zero level set."""
        grid = sphere_case(self.device, self.dtype)
        scaled = ref.SdfGridRef(
            2.0 * grid.values, grid.position, grid.rotation, grid.scale
        )
        origins, directions = z_rays([0.0, 0.15, 0.3], self.device, self.dtype)

        tape = ref.march(scaled, origins, directions, ref.TraceConfig())
        self.assertTrue(bool(tape.bisected.all()))
        res = ref.reattach(scaled, origins, directions, tape)
        self.assertTrue(bool(res.hit_mask.all()))
        torch.testing.assert_close(
            res.t,
            sphere_hit_t(origins, directions, grid.position, 0.5),
            atol=1e-3,
            rtol=0.0,
        )

    def test_rotated_non_uniform_box_clips_and_maps_axes(self) -> None:
        position = torch.tensor([0.3, -0.2, 0.1], device=self.device, dtype=self.dtype)
        rotation = quat((0.0, 0.0, 1.0), math.radians(40.0), position)
        scale = torch.tensor([2.0, 0.5, 1.2], device=self.device, dtype=self.dtype)
        rot = ref.rotation_matrix(rotation)
        values = bake(
            lambda p: sphere_sdf(p, position, 0.8), (65, 33, 49), position, rotation, scale
        )
        grid = ref.SdfGridRef(values, position, rotation, scale)

        def local_ray(start: tuple[float, float, float], axis: tuple[float, float, float]):
            origin = position + rot @ position.new_tensor(start)
            return origin.unsqueeze(0), (rot @ position.new_tensor(axis)).unsqueeze(0)

        # Along local +x the box is 1.0 deep, so the sphere is reached inside it.
        origins, directions = local_ray((-3.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        # Along local +y the box is only 0.25 deep and the field never reaches zero.
        blocked_o, blocked_d = local_ray((0.0, -3.0, 0.0), (0.0, 1.0, 0.0))
        # A diagonal world ray that crosses the axis-aligned bound of the box but
        # misses the oriented box itself, and would hit the sphere if traced.
        diag = position.new_tensor([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        side = position.new_tensor([-1.0, 1.0, 0.0]) / math.sqrt(2.0)
        outside_o = (position - 3.0 * diag + 0.7 * side).unsqueeze(0)
        outside_d = diag.unsqueeze(0)
        self.assertTrue(self.crosses_world_bound(grid, rot, outside_o, outside_d))

        res = ref.intersect(
            grid,
            torch.cat([origins, blocked_o, outside_o]),
            torch.cat([directions, blocked_d, outside_d]),
        )
        torch.testing.assert_close(
            res.hit_mask, torch.tensor([True, False, False], device=self.device)
        )
        torch.testing.assert_close(
            res.t[:1], torch.full((1,), 2.2, device=self.device), atol=2e-3, rtol=0.0
        )
        self.assertEqual(int(res.steps[2]), 0)

    # Slab test against the world-axis-aligned bound of the oriented box.
    def crosses_world_bound(
        self, grid: ref.SdfGridRef, rot: Tensor, origins: Tensor, directions: Tensor
    ) -> bool:
        half = rot.abs() @ (0.5 * grid.scale)
        t_a = (grid.position - half - origins) / directions
        t_b = (grid.position + half - origins) / directions
        t_lo = torch.minimum(t_a, t_b).max(dim=-1).values.clamp_min(0.0)
        t_hi = torch.maximum(t_a, t_b).min(dim=-1).values
        return bool((t_lo <= t_hi).all())

    def test_manual_gather_matches_grid_sample(self) -> None:
        """`grid_sample(align_corners=True)` is the same interpolant; the manual
        gather is preferred only because it also yields the frozen base index and
        the analytic `dD/du` (see the module docstring)."""
        torch.manual_seed(7)
        values = torch.randn(9, 7, 11, device=self.device, dtype=torch.float64)
        cells = values.new_tensor([8.0, 6.0, 10.0])
        u = torch.rand(64, 3, device=self.device, dtype=torch.float64) * cells
        manual, _ = ref.sample(values, u, ref.base_index(u, cells))

        normalized = 2.0 * u / cells - 1.0
        sampled = F.grid_sample(
            values.reshape(1, 1, *values.shape),
            normalized.flip(-1).reshape(1, 1, 1, -1, 3),
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        ).reshape(-1)
        torch.testing.assert_close(manual, sampled, atol=1e-12, rtol=0.0)


@dataclass
class GradCase:
    """A differentiable configuration whose six inputs all matter."""

    values: Tensor
    position: Tensor
    rotation: Tensor
    scale: Tensor
    origins: Tensor
    directions: Tensor
    cfg: ref.TraceConfig
    weights: Tensor

    @property
    def grid(self) -> ref.SdfGridRef:
        return ref.SdfGridRef(self.values, self.position, self.rotation, self.scale)

    # Scalar objective whose gradient exercises `t` and the hit position.
    def loss(self) -> Tensor:
        res = ref.intersect(self.grid, self.origins, self.directions, self.cfg)
        return (res.t * self.weights).sum() + (res.position * self.weights.unsqueeze(-1)).sum()


@REQUIRES_CUDA
class SdfReferenceGradientTests(unittest.TestCase):
    """Derivative checks for the frozen-winner IFT contract of ADR-0037."""

    device = "cuda"
    dtype = torch.float64

    def make_case(self, requires_grad: bool = True) -> GradCase:
        position = torch.tensor([0.1, -0.2, 0.05], device=self.device, dtype=self.dtype)
        rotation = 1.3 * quat((0.3, 0.5, 0.8), 0.7, position)  # deliberately unnormalized
        scale = torch.tensor([2.0, 1.4, 1.6], device=self.device, dtype=self.dtype)
        values = bake(
            lambda p: sphere_sdf(p, position, 0.45), (33, 33, 33), position, rotation, scale
        )
        origins = position + torch.tensor(
            [[0.05, -0.1, -2.0], [-0.12, 0.08, -2.0], [0.0, 0.15, -2.0]],
            device=self.device,
            dtype=self.dtype,
        )
        directions = 1.7 * torch.tensor(  # deliberately unnormalized
            [[0.02, 0.01, 1.0], [-0.03, 0.02, 1.0], [0.0, -0.04, 1.0]],
            device=self.device,
            dtype=self.dtype,
        )
        tensors = [values, position, rotation, scale, origins, directions]
        for tensor in tensors:
            tensor.requires_grad_(requires_grad)
        return GradCase(
            *tensors,
            cfg=ref.TraceConfig(max_steps=256, eps_hit=1e-11),
            weights=torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=self.dtype),
        )

    def test_gradcheck_on_the_frozen_tape(self) -> None:
        case = self.make_case()
        small = bake(
            lambda p: sphere_sdf(p, case.position, 0.45),
            (9, 8, 7),
            case.position,
            case.rotation,
            case.scale,
        ).requires_grad_(True)
        grid = ref.SdfGridRef(small, case.position, case.rotation, case.scale)
        origins = case.origins[:2].detach().clone().requires_grad_(True)
        directions = case.directions[:2].detach().clone().requires_grad_(True)
        tape = ref.march(grid, origins, directions, case.cfg)
        self.assertTrue(bool(tape.hit.all()))

        def evaluate(*inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            res = ref.reattach(ref.SdfGridRef(*inputs[:4]), inputs[4], inputs[5], tape)
            return res.t, res.position, res.normal

        self.assertTrue(
            torch.autograd.gradcheck(
                evaluate,
                (small, case.position, case.rotation, case.scale, origins, directions),
                eps=1e-6,
                atol=1e-6,
                rtol=1e-4,
            )
        )

    # Central differences of a fully re-marched forward pass on selected entries.
    def check_finite_difference(
        self, field: str, h: float = 1e-4, atol: float = 1e-5
    ) -> None:
        case = self.make_case()
        case.loss().backward()
        tensor: Tensor = getattr(case, field)
        analytic = tensor.grad.reshape(-1).clone()
        flat = tensor.detach().reshape(-1)
        chosen = (
            analytic.nonzero().reshape(-1)
            if field == "values"
            else torch.arange(flat.numel(), device=self.device)
        )
        self.assertGreater(chosen.numel(), 0)
        # A finite-difference match against an all-zero gradient proves nothing.
        self.assertGreater(float(analytic[chosen].abs().max()), 1e-3)

        numeric = torch.zeros_like(analytic)
        with torch.no_grad():
            for index in chosen.tolist():
                original = flat[index].item()
                flat[index] = original + h
                plus = case.loss()
                flat[index] = original - h
                minus = case.loss()
                flat[index] = original
                numeric[index] = (plus - minus) / (2.0 * h)
        torch.testing.assert_close(
            numeric[chosen], analytic[chosen], atol=atol, rtol=1e-4
        )

    def test_finite_difference_values(self) -> None:
        self.check_finite_difference("values")

    def test_finite_difference_position(self) -> None:
        self.check_finite_difference("position")

    def test_finite_difference_rotation(self) -> None:
        self.check_finite_difference("rotation")

    def test_finite_difference_scale(self) -> None:
        self.check_finite_difference("scale")

    def test_finite_difference_origins(self) -> None:
        self.check_finite_difference("origins")

    def test_finite_difference_directions(self) -> None:
        self.check_finite_difference("directions")

    def test_grazing_rays_keep_every_gradient_finite(self) -> None:
        """Impact parameters sweeping through the tangent point of the sphere put
        the IFT denominator arbitrarily close to zero. The clamp has to keep every
        output and every gradient finite on both sides of the tangency."""
        grid = sphere_case(self.device, self.dtype, size=65)
        tensors = [
            grid.values.clone(),
            grid.position.clone(),
            grid.rotation.clone(),
            grid.scale.clone(),
        ]
        offsets = torch.linspace(0.45, 0.55, 21, device=self.device, dtype=self.dtype)
        origins = torch.stack(
            [offsets, torch.zeros_like(offsets), torch.full_like(offsets, -3.0)], dim=-1
        )
        directions = torch.zeros_like(origins)
        directions[:, 2] = 1.0
        rays = [origins, directions]
        for tensor in tensors + rays:
            tensor.requires_grad_(True)

        res = ref.intersect(
            ref.SdfGridRef(*tensors), *rays, ref.TraceConfig(max_steps=256)
        )
        self.assertTrue(bool(res.hit_mask.any()))
        self.assertFalse(bool(res.hit_mask.all()))
        finite_t = torch.where(res.hit_mask, res.t, torch.zeros_like(res.t))
        self.assertTrue(bool(torch.isfinite(finite_t).all()))
        self.assertTrue(bool(torch.isfinite(res.position).all()))
        self.assertTrue(bool(torch.isfinite(res.normal).all()))

        (finite_t.sum() + res.position.sum() + res.normal.sum()).backward()
        for name, tensor in zip(
            ("values", "position", "rotation", "scale", "origins", "directions"),
            tensors + rays,
        ):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()), msg=name)

    def test_zero_denominator_takes_the_signed_grazing_clamp(self) -> None:
        """A constant field has `grad_w D = 0`, so `g` is exactly zero and the
        clamp decides the derivative outright: `sign(0) := +1` makes it
        `-c_m / eps_graze`, and the eight trilinear weights sum to one."""
        position = torch.zeros(3, device=self.device, dtype=self.dtype)
        rotation = torch.tensor(IDENTITY_QUAT, device=self.device, dtype=self.dtype)
        scale = torch.full((3,), 2.0, device=self.device, dtype=self.dtype)
        values = torch.full((5, 5, 5), 0.5, device=self.device, dtype=self.dtype)
        values.requires_grad_(True)
        grid = ref.SdfGridRef(values, position, rotation, scale)
        origins = torch.tensor([[0.0, 0.0, -3.0]], device=self.device, dtype=self.dtype)
        directions = torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=self.dtype)

        frozen_t = origins.new_tensor([2.5])
        hit_point = origins + frozen_t.unsqueeze(-1) * directions
        u = ref.grid_coord(hit_point - position, scale, grid.cells)
        tape = ref.Tape(
            t=frozen_t,
            hit=torch.ones(1, dtype=torch.bool, device=self.device),
            base=ref.base_index(u, grid.cells),
            value=values.new_tensor([0.5]),
            steps=torch.ones(1, dtype=torch.int32, device=self.device),
            bisected=torch.zeros(1, dtype=torch.bool, device=self.device),
        )

        res = ref.reattach(grid, origins, directions, tape)
        torch.testing.assert_close(res.t, frozen_t, atol=1e-9, rtol=0.0)
        self.assertTrue(torch.equal(res.normal, torch.zeros(1, 3, device=self.device)))
        res.t.sum().backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        torch.testing.assert_close(
            values.grad.sum(),
            values.new_tensor(-1.0 / ref.EPS_GRAZE),
            atol=0.0,
            rtol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
