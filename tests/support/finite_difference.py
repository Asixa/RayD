# Copyright Xingyu Chen.
# Provides central-difference directional derivatives and tolerance checks for float32 kernels.

"""Finite-difference helpers shared by the gradient tests of both backends.

The kernels run in float32, so every check compares one directional derivative: the analytic gradient
contracted with a fixed direction against a single central-difference pair along that direction. The
objective is reduced to a Python float (float64) on the host so the only float32 error is the kernel's own.
With step `h` and objective magnitude `|f|`, the cancellation error is about `2 * 2^-24 * |f| / h` and the
truncation error is `O(h^2)`; `fd_tolerance` turns those two terms into an absolute bound.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import random


FLOAT32_EPS = 2.0**-24


def central_difference(evaluate: Callable[[float], float], step: float) -> float:
    """Return `(f(+step) - f(-step)) / (2 * step)` for a scalar objective `f(s)` evaluated at offset `s`."""
    return (float(evaluate(step)) - float(evaluate(-step))) / (2.0 * step)


def unit_direction(count: int, seed: int) -> list[float]:
    """A deterministic unit vector with `count` components drawn from a seeded generator."""
    generator = random.Random(seed)
    values = [generator.uniform(-1.0, 1.0) for _ in range(count)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b, strict=True)))


def fd_tolerance(objective_scale: float, step: float, rel: float = 2.0e-3, floor: float = 1.0e-4) -> float:
    """Absolute tolerance for a central difference of a float32 objective of magnitude `objective_scale`.

    The float32 rounding of each evaluation contributes `8 * eps * |f| / step` (a 4x safety factor over
    the two-evaluation bound); `rel` and `floor` are added by the caller's choice for curvature and for
    derivatives that are close to zero.
    """
    return 8.0 * FLOAT32_EPS * abs(objective_scale) / step + floor + rel * abs(objective_scale)


def assert_derivative_close(
    testcase, analytic: float, numeric: float, *, rel: float = 2.0e-3, abs_tol: float = 1.0e-4, label: str = ""
) -> float:
    """Assert `|analytic - numeric| <= abs_tol + rel * max(|analytic|, |numeric|)` and return the deviation."""
    deviation = abs(float(analytic) - float(numeric))
    bound = abs_tol + rel * max(abs(float(analytic)), abs(float(numeric)))
    testcase.assertLessEqual(
        deviation,
        bound,
        f"{label}: analytic directional derivative {analytic!r} differs from central difference {numeric!r} "
        f"by {deviation:.3e} > {bound:.3e}",
    )
    return deviation


def assert_nonzero_derivative(testcase, value: float, minimum: float = 1.0e-3, label: str = "") -> None:
    """Guard a finite-difference check against passing vacuously on a configuration with no sensitivity."""
    testcase.assertGreater(abs(float(value)), minimum, f"{label}: derivative {value!r} is too small to be a test")
