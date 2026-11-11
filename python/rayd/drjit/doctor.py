# Copyright Xingyu Chen.
# Checks that this machine can run rayd-drjit and prints the fix for each problem found.

"""`python -m rayd.drjit.doctor`: diagnose a rayd-drjit installation.

Prints the system, the installed and build-time drjit, the build's GPU targets,
the driver and every visible GPU, the OptiX runtime, and the trace and edge
backends that `"auto"` resolves to, with one verdict per check. Exits with
status 1 when any check fails. It runs once `import rayd.drjit` succeeds; when
that import fails, its error already names the cause and the fix.
"""

from __future__ import annotations

import sys

import drjit as dr

import rayd.drjit as rd
from rayd._impl.runtime_jit import _device_support as device_support
from rayd._impl.runtime_jit import _diagnostics as diagnostics


def _resolve_backends(report: diagnostics.Report, disabled: bool) -> None:
    """Build a one-triangle scene with the default backends and report what `auto` chose."""
    mesh = rd.Mesh(dr.cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]), dr.cuda.Array3i([0], [1], [2]))
    scene = rd.Scene()
    scene.add_mesh(mesh)
    scene.build()
    trace = scene.trace_backend
    report.fact("auto trace backend", trace)
    report.fact("auto edge backend", scene.edge_bvh_backend)
    if trace == "optix":
        report.check("trace backend", True, "OptiX traces triangles")
    elif disabled:
        report.check("trace backend", True, "the CUDA trace backend, because RAYD_DISABLE_OPTIX is set")
    else:
        report.check(
            "trace backend",
            False,
            "OptiX is unavailable for this CUDA context, so `auto` uses the slower CUDA trace backend",
            diagnostics.DOCKER_HINT,
        )


def main() -> int:
    report = diagnostics.Report("rayd-drjit")
    diagnostics.add_system(report)

    record = diagnostics.read_build_record("rayd.drjit._C")
    diagnostics.add_installation(report, "rayd-drjit", record)
    report.section("Dr.Jit")
    report.fact("drjit", dr.__version__)
    built = (record or {}).get("framework_version")
    report.check(
        "drjit build",
        built is None or built == dr.__version__,
        f"this build was compiled against drjit {built or 'an unrecorded version'}",
        f'python -m pip install "drjit=={built}"',
    )
    report.check("native extension", True, "rayd.drjit._C imported")

    driver = diagnostics.query_driver()
    diagnostics.add_driver(report, driver, record)
    targets = device_support.load_build_targets("drjit", "rayd.drjit._C")
    diagnostics.add_device_support(report, targets, driver, device_support)
    disabled = device_support.optix_disabled("drjit")
    diagnostics.add_optix(report, record, disabled)

    report.section("Backends")
    if not dr.has_backend(dr.JitBackend.CUDA):
        report.check("auto backends", None, "not resolved: Dr.Jit has no CUDA backend on this machine")
    else:
        try:
            _resolve_backends(report, disabled)
        except Exception as error:  # noqa: BLE001 - every failure is reported, none is hidden
            report.check("auto backends", False, f"building a one-triangle scene failed: {error}")

    print(report.render())
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
