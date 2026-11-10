# Copyright Xingyu Chen.
# Shares smooth geometry cases and directional-derivative drivers for the Torch and Dr.Jit gradient tests.

"""Backend-neutral inputs and the directional-derivative driver used by `tests/gradients/test_geometry_fd_*`.

Every case is a dictionary of named inputs given as nested Python lists, so both backends receive bitwise the
same float32 values. A check perturbs one input domain (one or more named inputs) along a single seeded unit
direction `v` and compares three numbers: the central difference of the scalar objective along `v`, the
reverse-mode gradient contracted with `v`, and the forward-mode tangent pushed along `v`. The last two must also
agree with each other much more tightly than either agrees with the finite difference (inner-product identity).

All configurations keep every discrete decision (hit triangle, winning edge, top-k order, SDF cell, surfel
candidate) at least ten steps away from a switch, so the objective is smooth on `[-step, +step]`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import math
import os
import random

from tests.support.finite_difference import FLOAT32_EPS, assert_derivative_close, central_difference, unit_direction


Nested = list
Values = Mapping[str, Nested]

# A bent quad: two non-coplanar triangles sharing the diagonal (0, 2), with per-vertex texture coordinates.
QUAD_VERTICES = [[-1.0, -1.0, 0.1], [1.2, -0.9, -0.1], [1.1, 1.0, 0.3], [-0.9, 1.1, 0.0]]
QUAD_FACES = [[0, 1, 2], [0, 2, 3]]
QUAD_UV = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

# Two slightly oblique rays that hit the interiors of faces 0 and 1 at least 0.3 from every edge.
INTERSECT_ORIGINS = [[0.3, -0.4, -2.0], [-0.4, 0.35, -2.0]]
INTERSECT_DIRECTIONS = [[0.05, 0.02, 1.0], [-0.03, 0.04, 1.0]]

# Query points off the surface whose nearest edges are unique, with the winner at least 0.1 closer than the
# runner-up; the winning closest points are interior to their edges (edge_t in [0.47, 0.61]).
EDGE_POINTS = [[0.1, -1.3, 0.4], [1.5, 0.2, -0.3], [0.05, 0.1, 0.6], [-1.2, 0.3, 0.2]]
# Top-k uses k = 2: for every query the second and third distances differ by at least 0.04.
TOPK_K = 2

# Rays for nearest_edge(ray). The finite ray is clamped by tmax (ray_t == tmax), so its distance is the
# point-to-edge distance of the segment end; the infinite rays pass the scene without approaching any edge.
EDGE_RAY_FINITE_ORIGINS = [[0.1, -1.6, 0.5]]
EDGE_RAY_FINITE_DIRECTIONS = [[0.2, 1.0, -0.1]]
EDGE_RAY_FINITE_TMAX = [0.25]
EDGE_RAY_INFINITE_ORIGINS = [[1.6, 0.1, -0.4], [-1.6, -0.3, 0.9]]
EDGE_RAY_INFINITE_DIRECTIONS = [[-1.0, 0.3, 0.1], [0.1, 1.0, 0.15]]

# An SDF sphere of radius 0.5 sampled on a 17^3 lattice in local [-1, 1]^3, placed off-centre with a rotation.
SDF_SIZE = 17
SDF_POSITION = [0.1, -0.05, 0.08]
SDF_ROTATION = [0.98, 0.1, -0.12, 0.08]  # (w, x, y, z), deliberately not unit length
SDF_SCALE = [2.0, 2.0, 2.0]
# Each hit lies at least 0.2 cell widths from every lattice face, so the frozen cell survives the perturbation.
SDF_ORIGINS = [[-0.06, 0.17, -2.0], [0.0, 0.05, -2.0], [-0.13, -0.2, -2.0]]
SDF_DIRECTIONS = [[0.02, 0.01, 1.0], [0.02, 0.01, 1.0], [0.02, 0.01, 1.0]]
SDF_MIN_CELL_MARGIN = 0.2
SDF_EPS_HIT = 1.0e-6
SDF_MAX_STEPS = 128

# Two overlapping, tilted Gaussian surfels and rays that cross both within one standard deviation.
SURFEL_CENTERS = [[0.0, 0.0, 0.0], [0.1, -0.05, 0.4]]
SURFEL_TANGENT_U = [[0.9, 0.1, 0.05], [0.8, -0.1, 0.0]]
SURFEL_TANGENT_V = [[-0.05, 1.1, 0.1], [0.1, 0.9, -0.05]]
SURFEL_OPACITY = [0.6, 0.5]
SURFEL_VALUE = [2.0, 3.0]
SURFEL_ORIGINS = [[0.15, 0.2, -1.0], [-0.2, 0.1, -1.0]]
SURFEL_DIRECTIONS = [[0.02, -0.03, 1.0], [0.04, 0.01, 1.0]]
SURFEL_MAX_CANDIDATES = 4

# A mixed scene with a triangle, an SDF slab and a surfel side by side; each ray selects one owner.
MIXED_MESH_VERTICES = [[-2.75, -0.75, 0.05], [-1.25, -0.8, -0.05], [-2.0, 0.75, 0.1]]
MIXED_SDF_SIZE = 16
MIXED_SDF_POSITION = [0.0, 0.0, 0.0]
MIXED_SURFEL_CENTER = [[2.0, 0.0, 0.0]]
MIXED_SURFEL_TANGENT_U = [[0.25, 0.0, 0.02]]
MIXED_SURFEL_TANGENT_V = [[0.0, 0.25, -0.01]]
MIXED_SURFEL_OPACITY = [0.5]
MIXED_ORIGINS = [[-2.05, -0.1, -2.0], [0.07, -0.04, -2.0], [2.03, 0.05, -2.0]]
MIXED_DIRECTIONS = [[0.01, 0.02, 1.0], [-0.02, 0.01, 1.0], [0.01, -0.01, 1.0]]


def sphere_sdf_values(size: int, radius: float = 0.5) -> list[float]:
    """Flat `[x][y][z]` (z fastest) samples of `|x| - radius` over local `[-1, 1]^3`."""
    axis = [-1.0 + 2.0 * index / (size - 1) for index in range(size)]
    return [math.sqrt(x * x + y * y + z * z) - radius for x in axis for y in axis for z in axis]


def slab_sdf_values(size: int) -> list[float]:
    """Flat `[x][y][z]` samples of the plane field `z` over local `[-0.5, 0.5]^3`."""
    axis = [-0.5 + index / (size - 1) for index in range(size)]
    return [z for _x in axis for _y in axis for z in axis]


def flatten(value: Nested | float) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [item for element in value for item in flatten(element)]
    return [float(value)]


def unflatten(flat: Sequence[float], like: Nested | float) -> Nested | float:
    iterator = iter(flat)

    def rebuild(shape):
        if isinstance(shape, (list, tuple)):
            return [rebuild(element) for element in shape]
        return next(iterator)

    return rebuild(like)


def domain_direction(values: Values, domain: Sequence[str], seed: int) -> dict[str, Nested]:
    """One unit direction over the concatenation of every input named in `domain`."""
    sizes = [len(flatten(values[name])) for name in domain]
    joint = unit_direction(sum(sizes), seed)
    direction: dict[str, Nested] = {}
    offset = 0
    for name, size in zip(domain, sizes):
        direction[name] = unflatten(joint[offset : offset + size], values[name])
        offset += size
    return direction


def perturbed(values: Values, direction: Mapping[str, Nested], scale: float) -> dict[str, Nested]:
    result = dict(values)
    for name, tangent in direction.items():
        base = flatten(values[name])
        result[name] = unflatten([b + scale * t for b, t in zip(base, flatten(tangent))], values[name])
    return result


def output_weights(count: int, seed: int) -> list[float]:
    """Fixed objective weights in [0.5, 1.5] with random signs, so no output cancels another by symmetry."""
    generator = random.Random(seed)
    return [generator.choice((-1.0, 1.0)) * generator.uniform(0.5, 1.5) for _ in range(count)]


# Deviations observed by `check_directional`, keyed by label; printed by `report_deviations()` on request.
DEVIATIONS: dict[str, dict[str, float]] = {}


def report_deviations(title: str) -> None:
    if os.environ.get("RAYD_FD_REPORT") != "1":
        return
    print(f"\n{title}: label | fd | vjp | jvp | |vjp-fd| | |jvp-fd| | |vjp-jvp| | rel(vjp-fd)")
    for label in sorted(DEVIATIONS):
        row = DEVIATIONS[label]
        print(
            f"  {label} | {row['fd']:.6g} | {row['vjp']:.6g} | {row['jvp']:.6g} | {row['vjp_fd']:.2e} | "
            f"{row['jvp_fd']:.2e} | {row['vjp_jvp']:.2e} | {row['rel']:.2e}"
        )


def check_directional(
    testcase,
    *,
    label: str,
    values: Values,
    domain: Sequence[str],
    seed: int,
    step: float,
    evaluate: Callable[[Values], float],
    vjp: Callable[[Values, Sequence[str]], Mapping[str, Sequence[float]]],
    jvp: Callable[[Values, Mapping[str, Nested]], float] | None,
    objective_scale: float | None = None,
    rel: float = 2.0e-3,
    abs_tol: float = 1.0e-4,
    duality_rel: float = 1.0e-4,
    require_nonzero: bool = True,
) -> None:
    """Check reverse mode, forward mode and the central difference of one objective along one direction.

    Tolerance: the central difference carries a float32 rounding error of about `2 * eps * |f| / step` per
    evaluation pair (|f| is `objective_scale`, defaulting to |f(x)|), which is folded into `abs_tol` with a
    4x margin; `rel` covers the O(step^2) truncation term and float32 differences in the derivative kernels.
    Reverse and forward mode evaluate the same linearisation, so they must agree to `duality_rel`.
    `jvp=None` checks reverse mode only; it is reserved for operations whose forward mode is tested separately.
    """
    direction = domain_direction(values, domain, seed)
    scale = abs(float(evaluate(values))) if objective_scale is None else objective_scale
    rounding = 8.0 * FLOAT32_EPS * scale / step
    numeric = central_difference(lambda offset: evaluate(perturbed(values, direction, offset)), step)
    gradients = vjp(values, domain)
    reverse = sum(
        float(g) * float(t) for name in domain for g, t in zip(gradients[name], flatten(direction[name]), strict=True)
    )
    forward = float("nan") if jvp is None else float(jvp(values, direction))
    DEVIATIONS[label] = {
        "fd": numeric,
        "vjp": reverse,
        "jvp": forward,
        "vjp_fd": abs(reverse - numeric),
        "jvp_fd": abs(forward - numeric),
        "vjp_jvp": abs(reverse - forward),
        "rel": abs(reverse - numeric) / max(abs(numeric), 1.0e-12),
    }
    if require_nonzero:
        testcase.assertGreater(abs(numeric), 10.0 * (abs_tol + rounding), f"{label}: derivative too small to test")
    assert_derivative_close(testcase, reverse, numeric, rel=rel, abs_tol=abs_tol + rounding, label=f"{label} VJP")
    if jvp is None:
        return
    assert_derivative_close(testcase, forward, numeric, rel=rel, abs_tol=abs_tol + rounding, label=f"{label} JVP")
    assert_derivative_close(
        testcase, reverse, forward, rel=duality_rel, abs_tol=1.0e-6, label=f"{label} VJP-JVP duality"
    )


def supported_domains(manifest: Mapping, family: str, variant: str) -> dict[str, dict[str, str]]:
    """The `input_domains` record of one derivative variant in a backend's `api_manifest()["derivatives"]`."""
    return dict(manifest[family][variant]["input_domains"])


def require_domains(testcase, manifest: Mapping, family: str, variant: str, tested: Iterable[str]) -> None:
    """Fail when the manifest names a domain the test does not exercise, or marks a tested one unsupported.

    Every geometry domain is `supported` in both backends today; if one becomes `unsupported` the matching test
    must be replaced by a check that the request raises, so this guard fails instead of silently passing.
    """
    domains = supported_domains(manifest, family, variant)
    testcase.assertEqual(set(domains), set(tested), f"{family}.{variant}: manifest domains differ from tested ones")
    for name, status in domains.items():
        testcase.assertEqual(status, {"vjp": "supported", "jvp": "supported"}, f"{family}.{variant}.{name}")
