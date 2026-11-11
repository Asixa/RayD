# Copyright Xingyu Chen.
# Checks that this machine can run rayd-torch and prints the fix for each problem found.

"""`python -m rayd.torch.doctor`: diagnose a rayd-torch installation.

Prints the system, the installed and build-time torch, the build's GPU targets,
the driver and every visible GPU, the OptiX runtime, and the trace and edge
backends that `"auto"` resolves to, with one verdict per check. Exits with
status 1 when any check fails. It works without CUDA and without the native
libraries, and reports what is missing instead.
"""

from __future__ import annotations

import sys

import torch

from rayd._impl import device_support, diagnostics, runtime


def _resolve_backends(report: diagnostics.Report, optix: diagnostics.OptixProbe, disabled: bool) -> None:
    """Build a one-triangle scene with the default backends and report what `auto` chose."""
    from rayd._impl.scene import Mesh, Scene

    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    scene = Scene()
    scene.add_mesh(Mesh(vertices, faces))
    scene.build()
    trace, edge = scene.trace_backend, scene.edge_bvh_backend
    report.fact("auto trace backend", trace)
    report.fact("auto edge backend", edge)
    if trace == "optix":
        report.check("trace backend", True, "OptiX traces triangles")
    elif disabled:
        report.check("trace backend", True, "the CUDA trace backend, because RAYD_DISABLE_OPTIX is set")
    else:
        fix = "update the NVIDIA driver" if optix.expected_supported is False else diagnostics.DOCKER_HINT
        detail = "OptiX is unavailable (see the OptiX checks), so `auto` uses the slower CUDA trace backend"
        report.check("trace backend", False, detail, fix)


def main() -> int:
    """Print the installation report and return the exit status: 0 when every check passes, 1 otherwise."""
    report = diagnostics.Report("rayd-torch")
    diagnostics.add_system(report)

    record = runtime._BUILD_RECORD
    diagnostics.add_installation(report, "rayd-torch", record)
    report.section("Torch")
    report.fact("torch", f"{torch.__version__} (CUDA {torch.version.cuda or 'none: CPU-only build'})")
    mismatch = diagnostics.torch_mismatch(record, torch.__version__, torch.version.cuda)
    report.check(
        "torch build",
        mismatch is None,
        mismatch or "the installed torch matches this build",
        diagnostics.torch_install_command(record),
    )
    if runtime.NATIVE_AVAILABLE and runtime._STABLE_AVAILABLE:
        report.check("native libraries", True, "_legacy_ops and _stable_ops loaded")
    else:
        error = runtime.LOAD_ERROR if not runtime.NATIVE_AVAILABLE else runtime._STABLE_LOAD_ERROR
        first, _, rest = str(error).partition("\n")
        report.check("native libraries", False, first)
        report.lines.extend("         " + line for line in rest.splitlines())

    driver = diagnostics.query_driver()
    diagnostics.add_driver(report, driver, record)
    diagnostics.add_device_support(report, runtime.BUILD_TARGETS, driver, device_support)
    disabled = device_support.optix_disabled("torch")
    optix = diagnostics.add_optix(report, record, disabled)

    report.section("Backends")
    if not runtime.NATIVE_AVAILABLE:
        report.check("auto backends", None, "not resolved: the native libraries did not load")
    elif not torch.cuda.is_available():
        report.check("auto backends", None, "not resolved: torch sees no CUDA device")
    else:
        try:
            _resolve_backends(report, optix, disabled)
        except Exception as error:  # noqa: BLE001 - every failure is reported, none is hidden
            report.check("auto backends", False, f"building a one-triangle scene failed: {error}")

    print(report.render())
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
