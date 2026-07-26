"""Pure-PyTorch reference for the ADR-0037 differentiable SDF ray intersection.

Phase 1 of `docs/dev/sdf_intersection_plan.md`. This module is the numerical
oracle the Phase 3 CUDA kernels are validated against. It is test-only and must
never be imported from the shipped `rayd.torch` package.

The march (`march`) is fully detached and reproduces ADR-0037 section 4 verbatim:
entry-sign relaxation, a step clamped to `t_hi` before it is sampled, sign-flip
bisection with the section 4 rule order, and a bounded bisection budget that
still reports a hit when it is exhausted. The differentiable part (`reattach`) is
the frozen-winner implicit function theorem of section 6: it consumes the frozen
hit distance, hit mask, base voxel index and field value from the tape, and
carries derivatives to `values`, `position`, `rotation`, `scale`, `origins` and
`directions` through the partials of `F` at that frozen hit.

Interpolation choice: the trilinear gather is written out by hand rather than
delegated to `torch.nn.functional.grid_sample(..., align_corners=True)`. The two
agree to float64 round-off, which `test_sdf_reference.py` pins, but the manual
form is the exact ADR-0037 expression while `grid_sample` is only equivalent to
it. Three differences decide it: ADR-0037 clamps `u` to `[0, N_i - 1]` before the
base/fraction split, whereas `grid_sample` clamps normalized coordinates through
`padding_mode`; ADR-0037 freezes the base voxel index on the tape, whereas
`grid_sample` re-derives a winner from the coordinate it is handed; and the
normal and the IFT denominator need the analytic index-space gradient `dD/du`,
which the manual weights give in closed form and `grid_sample` would only expose
through a double-backward graph. The `[N, C, D, H, W]` layout of `grid_sample`
additionally reverses the axis order of `values`, which is an avoidable way to
get the mapping wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ADR-0037 section 7. `eps_graze`, `eps_norm` and `eps_parallel` are contract
# constants and are deliberately not caller parameters.
EPS_GRAZE = 1e-6
EPS_NORM = 1e-12
EPS_PARALLEL = 1e-7
EPS_HIT_VOXEL_FRACTION = 1e-3
BISECTION_STEPS = 32
DEFAULT_RELAXATION = 0.9
DEFAULT_MAX_STEPS = 64


@dataclass(frozen=True)
class SdfGridRef:
    """Caller-owned dense field placed in the world by an oriented box."""

    values: Tensor  # [Nx, Ny, Nz], vertex-centred, negative inside
    position: Tensor  # [3], world centre of the box
    rotation: Tensor  # [4], scalar-first quaternion, normalized internally
    scale: Tensor  # [3], full side lengths in world units

    @property
    def cells(self) -> Tensor:
        # Per-axis `N_i - 1`, the span of the grid coordinate `u`.
        extent = torch.tensor(
            self.values.shape, device=self.values.device, dtype=self.values.dtype
        )
        return extent - 1


@dataclass(frozen=True)
class TraceConfig:
    """Caller march parameters. None of these is differentiable."""

    tmax: float = float("inf")
    max_steps: int = DEFAULT_MAX_STEPS
    relaxation: float = DEFAULT_RELAXATION
    eps_hit: float | None = None  # None derives `EPS_HIT_VOXEL_FRACTION * h_min`


@dataclass(frozen=True)
class Tape:
    """Frozen discrete decisions of one march.

    `t`, `hit`, `base` and `value` are the ADR-0037 tape (`value` is the field
    sample at the frozen hit, which the reattachment needs as a constant).
    `bisected` is a reference-only diagnostic so the tests can prove the
    bisection branch is exercised; it carries no contract.
    """

    t: Tensor  # [N] frozen hit distance
    hit: Tensor  # [N] bool
    base: Tensor  # [N, 3] int64 frozen base voxel index
    value: Tensor  # [N] frozen field value at the frozen hit
    steps: Tensor  # [N] int32 field evaluations performed
    bisected: Tensor  # [N] bool


@dataclass(frozen=True)
class SdfIntersectionRef:
    """ADR-0037 section 5 result. Missed lanes are bitwise inert."""

    t: Tensor  # [N], `+inf` on miss
    hit_mask: Tensor  # [N] bool
    position: Tensor  # [N, 3], `+0.0` on miss
    normal: Tensor  # [N, 3], `+0.0` on miss
    steps: Tensor  # [N] int32


# Unit vector under the contract normalization floor, finite even at zero length.
def unit(v: Tensor) -> Tensor:
    length = v.pow(2).sum(-1, keepdim=True).clamp_min(EPS_NORM * EPS_NORM).sqrt()
    return v / length


# Local-to-world rotation matrix of a scalar-first quaternion (ADR-0037 section 2).
def rotation_matrix(q: Tensor) -> Tensor:
    w, x, y, z = unit(q).unbind()
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)])
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)])
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return torch.stack([row0, row1, row2])


# Grid coordinate of local points, clamped to the closed sampled domain.
def grid_coord(x_l: Tensor, scale: Tensor, cells: Tensor) -> Tensor:
    u = (x_l / scale + 0.5) * cells
    return torch.minimum(u.clamp_min(0.0), cells)


# Base voxel index of a grid coordinate, kept in range for any input.
def base_index(u: Tensor, cells: Tensor) -> Tensor:
    safe = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).floor()
    return torch.minimum(safe.clamp_min(0.0), cells - 1).long()


# Trilinear value and index-space gradient at a frozen base index.
def sample(values: Tensor, u: Tensor, base: Tensor) -> tuple[Tensor, Tensor]:
    ny, nz = values.shape[1], values.shape[2]
    offsets = values.new_tensor(
        [0, 1, nz, nz + 1, ny * nz, ny * nz + 1, ny * nz + nz, ny * nz + nz + 1],
        dtype=torch.long,
    )
    linear = (base[:, 0] * ny + base[:, 1]) * nz + base[:, 2]
    corners = values.reshape(-1)[linear.unsqueeze(-1) + offsets].reshape(-1, 2, 2, 2)
    f = u - base.to(u.dtype)
    w = torch.stack([1 - f, f], dim=-1)  # [N, 3, 2]
    slope = u.new_tensor([-1.0, 1.0])
    value = torch.einsum("nabc,na,nb,nc->n", corners, w[:, 0], w[:, 1], w[:, 2])
    d_du = torch.stack(
        [
            torch.einsum("nabc,a,nb,nc->n", corners, slope, w[:, 1], w[:, 2]),
            torch.einsum("nabc,na,b,nc->n", corners, w[:, 0], slope, w[:, 2]),
            torch.einsum("nabc,na,nb,c->n", corners, w[:, 0], w[:, 1], slope),
        ],
        dim=-1,
    )
    return value, d_du


# World-space field gradient at a frozen base index (ADR-0037 section 2).
def world_gradient(grid: SdfGridRef, d_du: Tensor, rot: Tensor) -> Tensor:
    return (d_du * grid.cells / grid.scale) @ rot.transpose(0, 1)


# Ray/box overlap in the local frame, plus the lanes that may be traced at all.
def slab_clip(
    o_l: Tensor, w_l: Tensor, scale: Tensor, tmax: float
) -> tuple[Tensor, Tensor, Tensor]:
    half = 0.5 * scale
    parallel = w_l.abs() <= EPS_PARALLEL
    denom = torch.where(parallel, torch.ones_like(w_l), w_l)
    t_a = (-half - o_l) / denom
    t_b = (half - o_l) / denom
    huge = torch.full_like(w_l, float("inf"))
    near = torch.where(parallel, -huge, torch.minimum(t_a, t_b))
    far = torch.where(parallel, huge, torch.maximum(t_a, t_b))
    t_lo = near.max(dim=-1).values.clamp_min(0.0)
    t_hi = far.min(dim=-1).values.clamp_max(tmax)
    outside = parallel & (o_l.abs() > half)
    valid = (t_lo <= t_hi) & ~outside.any(dim=-1)
    return t_lo, t_hi, valid & torch.isfinite(t_lo) & torch.isfinite(t_hi)


# Structural finiteness of the placement, which a lane cannot recover from.
def placement_is_usable(grid: SdfGridRef) -> Tensor:
    finite = (
        torch.isfinite(grid.position).all()
        & torch.isfinite(grid.rotation).all()
        & torch.isfinite(grid.scale).all()
    )
    return finite & (grid.scale > 0).all()


# Detached sphere trace producing the frozen winner of ADR-0037 section 4.
def march(
    grid: SdfGridRef, origins: Tensor, directions: Tensor, cfg: TraceConfig
) -> Tape:
    with torch.no_grad():
        rot = rotation_matrix(grid.rotation)
        wh = unit(directions)
        cells = grid.cells
        o_l = (origins - grid.position) @ rot
        t_lo, t_hi, valid = slab_clip(o_l, wh @ rot, grid.scale, cfg.tmax)
        valid = valid & placement_is_usable(grid)
        eps_hit = (
            (grid.scale / cells).min() * EPS_HIT_VOXEL_FRACTION
            if cfg.eps_hit is None
            else grid.values.new_tensor(cfg.eps_hit)
        )

        def probe(t: Tensor) -> tuple[Tensor, Tensor]:
            x_l = (origins + t.unsqueeze(-1) * wh - grid.position) @ rot
            u = grid_coord(x_l, grid.scale, cells)
            base = base_index(u, cells)
            return sample(grid.values, u, base)[0], base

        t = torch.where(valid, t_lo, torch.zeros_like(t_lo))
        d, base = probe(t)
        steps = valid.to(torch.int32)
        failed = valid & ~torch.isfinite(d)
        active = valid & ~failed
        sigma = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
        hit = torch.zeros_like(active)
        lo, hi = t.clone(), t.clone()
        bisected = torch.zeros_like(active)

        for _ in range(cfg.max_steps):
            if not bool(active.any()):
                break
            reached = active & (d.abs() < eps_hit)
            hit = hit | reached
            active = active & ~reached
            t_raw = t + cfg.relaxation * sigma * d
            t_next = torch.minimum(t_raw, t_hi)
            d_next, base_next = probe(torch.where(active, t_next, t))
            steps = steps + active.to(torch.int32)
            bad = active & ~torch.isfinite(d_next)
            failed, active = failed | bad, active & ~bad
            flip = active & (sigma * d_next < 0)
            lo = torch.where(flip, t, lo)
            hi = torch.where(flip, t_next, hi)
            bisected, active = bisected | flip, active & ~flip & ~(t_raw > t_hi)
            t = torch.where(active, t_next, t)
            d = torch.where(active, d_next, d)
            base = torch.where(active.unsqueeze(-1), base_next, base)

        alive = bisected.clone()
        for _ in range(BISECTION_STEPS):
            if not bool(alive.any()):
                break
            mid = 0.5 * (lo + hi)
            d_mid, base_mid = probe(torch.where(alive, mid, t))
            steps = steps + alive.to(torch.int32)
            t = torch.where(alive, mid, t)
            d = torch.where(alive, d_mid, d)
            base = torch.where(alive.unsqueeze(-1), base_mid, base)
            bad = alive & ~torch.isfinite(d_mid)
            done = alive & (d_mid.abs() < eps_hit)
            moving = alive & ~done & ~bad
            keep_lo = (sigma * d_mid) >= 0
            lo = torch.where(moving & keep_lo, mid, lo)
            hi = torch.where(moving & ~keep_lo, mid, hi)
            failed, alive = failed | bad, moving

        return Tape(
            t=t,
            hit=(hit | bisected) & ~failed,
            base=base,
            value=d,
            steps=steps,
            bisected=bisected,
        )


# Differentiable last step: the frozen-winner IFT of ADR-0037 section 6.
def reattach(
    grid: SdfGridRef, origins: Tensor, directions: Tensor, tape: Tape
) -> SdfIntersectionRef:
    rot = rotation_matrix(grid.rotation)
    wh = unit(directions)
    lane = tape.hit.unsqueeze(-1)
    frozen_t = torch.where(tape.hit, tape.t, torch.zeros_like(tape.t))
    base = torch.where(lane, tape.base, torch.zeros_like(tape.base))

    x_l = (origins + frozen_t.unsqueeze(-1) * wh - grid.position) @ rot
    value, d_du = sample(grid.values, grid_coord(x_l, grid.scale, grid.cells), base)
    g = (world_gradient(grid, d_du, rot) * wh).sum(-1).detach()
    signed = torch.where(g >= 0, torch.ones_like(g), -torch.ones_like(g))
    t = frozen_t - (value - tape.value) / (signed * g.abs().clamp_min(EPS_GRAZE))

    hit_point = origins + t.unsqueeze(-1) * wh
    x_l_hit = (hit_point - grid.position) @ rot
    _, d_du_hit = sample(
        grid.values, grid_coord(x_l_hit, grid.scale, grid.cells), base
    )
    normal = unit(world_gradient(grid, d_du_hit, rot))

    zero = torch.zeros_like(hit_point)
    return SdfIntersectionRef(
        t=torch.where(tape.hit, t, torch.full_like(t, float("inf"))),
        hit_mask=tape.hit,
        position=torch.where(lane, hit_point, zero),
        normal=torch.where(lane, normal, zero),
        steps=tape.steps,
    )


# Public reference entry point: detached march plus differentiable reattachment.
def intersect(
    grid: SdfGridRef,
    origins: Tensor,
    directions: Tensor,
    cfg: TraceConfig = TraceConfig(),
) -> SdfIntersectionRef:
    return reattach(grid, origins, directions, march(grid, origins, directions, cfg))
