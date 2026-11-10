# Copyright Xingyu Chen.
# Runs directional finite-difference checks of multipath operations for the Torch and Dr.Jit backends.

"""Directional derivative harness for the multipath gradient tests.

An objective maps a dict of named inputs to `(scalar, signature)`: `scalar` is a one-element float32 result of
the operation under test and `signature` is a hashable record of its discrete decisions (hit primitives, sample
counts, validity). A check perturbs a group of inputs along one seeded unit direction `v` and compares

* the reverse-mode product `<VJP(1), v>`,
* the forward-mode tangent `JVP(v)`,
* the central difference `(f(x + h v) - f(x - h v)) / (2 h)`,

after asserting that the discrete signature is identical at `x - h v`, `x` and `x + h v`, so the comparison never
straddles a visibility, primitive or sample-count change. `<VJP(1), v> == JVP(v)` is the inner-product identity
and holds to float32 rounding; the finite difference carries the cancellation error of two float32 evaluations
of magnitude `|f|`, which `fd_tolerance` bounds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math

import numpy as np

from tests.support.finite_difference import assert_derivative_close, fd_tolerance


# Every executed check appends (label, |AD - FD|, |VJP - JVP|) here so a driver can report observed deviations.
DEVIATIONS: list[tuple[str, float, float]] = []


@dataclass(frozen=True)
class DirectionalResult:
    value: float
    vjp: float
    jvp: float
    fd: float
    step: float


def seeded_tangents(base: Mapping[str, np.ndarray], group: Iterable[str], seed: int) -> dict[str, np.ndarray]:
    """One unit direction over the concatenation of the inputs in `group`, drawn from a seeded generator."""
    generator = np.random.default_rng(seed)
    tangents = {name: generator.standard_normal(np.shape(base[name])) for name in group}
    norm = math.sqrt(sum(float(np.sum(value * value)) for value in tangents.values()))
    return {name: (value / norm).astype(np.float32) for name, value in tangents.items()}


def _shifted(base: Mapping[str, np.ndarray], tangents: Mapping[str, np.ndarray], scale: float) -> dict:
    return {
        name: (np.asarray(value, dtype=np.float32) + np.float32(scale) * tangents[name]).astype(np.float32)
        if name in tangents
        else np.asarray(value, dtype=np.float32)
        for name, value in base.items()
    }


def assert_directional(testcase, result: DirectionalResult, *, label: str, rel: float = 3.0e-3, floor: float = 1.0e-5):
    """Assert VJP == JVP to float32 rounding and both match the central difference within the FD error bound."""
    abs_tol = fd_tolerance(result.value, result.step, rel=0.0, floor=floor)
    testcase.assertGreater(
        abs(result.fd), 10.0 * abs_tol, f"{label}: derivative {result.fd!r} is below the FD resolution {abs_tol:.3e}"
    )
    identity = abs(result.vjp - result.jvp)
    testcase.assertLessEqual(
        identity,
        1.0e-4 * max(abs(result.vjp), abs(result.jvp)) + 1.0e-9,
        f"{label}: <VJP, v> = {result.vjp!r} but JVP(v) = {result.jvp!r}",
    )
    deviation = assert_derivative_close(testcase, result.vjp, result.fd, rel=rel, abs_tol=abs_tol, label=label)
    assert_derivative_close(testcase, result.jvp, result.fd, rel=rel, abs_tol=abs_tol, label=label + " (JVP)")
    DEVIATIONS.append((label, deviation / max(abs(result.fd), 1.0e-30), identity))


# ---------------------------------------------------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------------------------------------------------


def torch_directional(
    testcase,
    objective: Callable[[dict], tuple],
    base: Mapping[str, np.ndarray],
    group: Iterable[str],
    *,
    step: float,
    seed: int,
) -> DirectionalResult:
    import torch

    forward_ad = torch.autograd.forward_ad
    group = tuple(group)
    tangents = seeded_tangents(base, group, seed)

    def tensors(values: Mapping[str, np.ndarray]) -> dict:
        return {name: torch.as_tensor(np.asarray(value), device="cuda") for name, value in values.items()}

    inputs = tensors(base)
    leaves = {name: inputs[name].clone().requires_grad_(name in group) for name in inputs}
    value, signature = objective(leaves)
    grads = torch.autograd.grad(value, [leaves[name] for name in group])
    vjp = sum(
        float(torch.sum(grad.double() * torch.as_tensor(tangents[name], device="cuda").double()))
        for name, grad in zip(group, grads)
    )

    with forward_ad.dual_level():
        duals = dict(inputs)
        for name in group:
            duals[name] = forward_ad.make_dual(inputs[name], torch.as_tensor(tangents[name], device="cuda"))
        dual_value, dual_signature = objective(duals)
        tangent = forward_ad.unpack_dual(dual_value).tangent
        testcase.assertIsNotNone(tangent, "JVP produced no tangent for a supported input domain")
        jvp = float(tangent.reshape(()))

    with torch.no_grad():
        plus_value, plus_signature = objective(tensors(_shifted(base, tangents, step)))
        minus_value, minus_signature = objective(tensors(_shifted(base, tangents, -step)))
    testcase.assertEqual(plus_signature, signature, "the +step evaluation changed a discrete decision")
    testcase.assertEqual(minus_signature, signature, "the -step evaluation changed a discrete decision")
    testcase.assertEqual(dual_signature, signature, "the JVP evaluation changed a discrete decision")
    fd = (float(plus_value) - float(minus_value)) / (2.0 * step)
    return DirectionalResult(float(value.detach()), vjp, jvp, fd, step)


def torch_grad_along_direction(
    leaves: Mapping, loss_at: Callable[[dict], object], *, step: float, seed: int, tangents: Mapping | None = None
) -> DirectionalResult:
    """Compare already-populated `leaf.grad` values with a central difference of `loss_at` along one direction.

    `leaves` maps names to the leaf tensors whose `.grad` holds the gradient of the loss under test; `loss_at`
    re-evaluates that loss without autograd from a dict of detached values. `tangents` defaults to a seeded unit
    direction over all leaves. The returned `jvp` repeats the analytic value so callers with only a VJP can use
    `assert_derivative_close` on `vjp` directly.
    """
    import torch

    if tangents is None:
        host = {name: leaf.detach().cpu().numpy() for name, leaf in leaves.items()}
        tangents = {
            name: torch.as_tensor(value, device=leaves[name].device)
            for name, value in seeded_tangents(host, leaves, seed).items()
        }
    analytic = sum(
        float(torch.sum(leaves[name].grad.double() * tangent.double())) for name, tangent in tangents.items()
    )
    with torch.no_grad():
        base = {name: leaf.detach() for name, leaf in leaves.items()}
        value = float(loss_at(base))
        plus = float(
            loss_at({name: base[name] + step * tangents[name] if name in tangents else base[name] for name in base})
        )
        minus = float(
            loss_at({name: base[name] - step * tangents[name] if name in tangents else base[name] for name in base})
        )
    return DirectionalResult(value, analytic, analytic, (plus - minus) / (2.0 * step), step)


# ---------------------------------------------------------------------------------------------------------------
# Dr.Jit
# ---------------------------------------------------------------------------------------------------------------


def drjit_array(ad, value: np.ndarray):
    """Convert an `[n, 3]` or `[n]` float32 host array into the matching Dr.Jit AD array."""
    value = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if value.ndim == 2:
        return ad.Array3f(value[:, 0].copy(), value[:, 1].copy(), value[:, 2].copy())
    return ad.Float(value)


def _drjit_host(gradient, like: np.ndarray) -> np.ndarray:
    if np.ndim(like) == 2:
        return np.stack([np.array(gradient.x), np.array(gradient.y), np.array(gradient.z)], axis=1)
    return np.array(gradient)


def drjit_directional(
    testcase,
    modules,
    objective: Callable[[dict], tuple],
    base: Mapping[str, np.ndarray],
    group: Iterable[str],
    *,
    step: float,
    seed: int,
) -> DirectionalResult:
    dr, ad = modules.dr, modules.ad
    group = tuple(group)
    tangents = seeded_tangents(base, group, seed)
    flags = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad

    def arrays(values: Mapping[str, np.ndarray]) -> dict:
        return {name: drjit_array(ad, value) for name, value in values.items()}

    inputs = arrays(base)
    for name in group:
        dr.enable_grad(inputs[name])
    value, signature = objective(inputs)
    dr.backward(value, flags=flags)
    vjp = sum(
        float(np.sum(_drjit_host(dr.grad(inputs[name]), base[name]).astype(np.float64) * tangents[name]))
        for name in group
    )

    inputs = arrays(base)
    for name in group:
        dr.enable_grad(inputs[name])
        dr.set_grad(inputs[name], drjit_array(ad, tangents[name]))
    dual_value, dual_signature = objective(inputs)
    dr.forward_to(dual_value, flags=flags)
    jvp = float(dr.grad(dual_value)[0])

    plus_value, plus_signature = objective(arrays(_shifted(base, tangents, step)))
    minus_value, minus_signature = objective(arrays(_shifted(base, tangents, -step)))
    testcase.assertEqual(plus_signature, signature, "the +step evaluation changed a discrete decision")
    testcase.assertEqual(minus_signature, signature, "the -step evaluation changed a discrete decision")
    testcase.assertEqual(dual_signature, signature, "the JVP evaluation changed a discrete decision")
    fd = (float(plus_value[0]) - float(minus_value[0])) / (2.0 * step)
    return DirectionalResult(float(value[0]), vjp, jvp, fd, step)


def manifest_domains(manifest: Mapping, families: Iterable[str], status: str) -> set[tuple[str, str]]:
    """(operation variant, input domain) pairs whose VJP and JVP both have `status` in a backend api_manifest."""
    found = set()
    derivatives = manifest["derivatives"]
    for family in families:
        for variant, entry in derivatives[family].items():
            for domain, modes in entry["input_domains"].items():
                if modes["vjp"] == status and modes["jvp"] == status:
                    found.add((variant, domain))
    return found


MULTIPATH_FAMILIES = ("reflection_trace", "reflection_accumulation", "diffraction_direct", "diffraction_chain")
