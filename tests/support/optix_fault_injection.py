# Copyright Xingyu Chen.
# Builds the OptiX fault-injection driver wrapper and runs test scripts against it in fresh processes.

"""Fault injection for OptiX pipeline building.

`optix_fault_injection.c` is compiled into a library named `libnvoptix.so.1` and put first on
`LD_LIBRARY_PATH` of a child process, so both backends load it in place of the driver library. It forwards
every call to the real driver but can fail chosen `optixPipelineCreate` calls with a link error (the
compile-class failure RayD retries) and chosen `optixPipelineSetStackSize` calls with an invalid-value error
(a failure RayD must report without retrying). Scripts arm it through `ctypes` with `shim.rayd_fault_*`.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from tests.support.subprocess_cases import compose, decode_json_result, run_script


SOURCE = Path(__file__).with_name("optix_fault_injection.c")

PRELUDE = """
import ctypes
import json
import sys

shim = ctypes.CDLL("libnvoptix.so.1")
if not hasattr(shim, "rayd_fault_arm_pipeline_create"):
    raise SystemExit("the OptiX fault-injection wrapper was not loaded in place of the driver library")


def counters():
    return {
        "create_calls": shim.rayd_fault_pipeline_create_calls(),
        "create_injected": shim.rayd_fault_pipeline_create_injected(),
        "stack_calls": shim.rayd_fault_set_stack_size_calls(),
        "stack_injected": shim.rayd_fault_set_stack_size_injected(),
        "cache_disable": shim.rayd_fault_cache_disable_calls(),
        "cache_enable": shim.rayd_fault_cache_enable_calls(),
        "patched_tables": shim.rayd_fault_patched_tables(),
    }


def disarm():
    shim.rayd_fault_arm_pipeline_create(0, 0)
    shim.rayd_fault_arm_set_stack_size(0, 0)


def error_of(fn):
    try:
        fn()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None
"""


def real_optix_library() -> str | None:
    """Path of the driver's `libnvoptix.so.1`, from RAYD_FAULT_REAL_OPTIX or the dynamic linker cache."""
    override = os.environ.get("RAYD_FAULT_REAL_OPTIX")
    if override:
        return override if os.path.isfile(override) else None
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    try:
        listing = subprocess.run([ldconfig, "-p"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in listing.splitlines():
        name, _, path = line.strip().partition(" => ")
        if name.startswith("libnvoptix.so.1 ") and "x86-64" in name and os.path.isfile(path.strip()):
            return os.path.realpath(path.strip())
    return None


@functools.lru_cache(maxsize=None)
def build_wrapper() -> tuple[str | None, str]:
    """Compile the wrapper once per test process; return (directory holding libnvoptix.so.1, skip reason)."""
    if platform.system() != "Linux":
        return None, "OptiX fault injection replaces a Linux shared library"
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        return None, "a C compiler is required to build the OptiX fault-injection wrapper"
    if real_optix_library() is None:
        return None, "the OptiX driver library libnvoptix.so.1 was not found"
    directory = tempfile.mkdtemp(prefix="rayd-optix-fault-")
    output = os.path.join(directory, "libnvoptix.so.1")
    command = [compiler, "-O1", "-shared", "-fPIC", "-Wl,-soname,libnvoptix.so.1", "-o", output, str(SOURCE), "-ldl"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        return None, f"building the OptiX fault-injection wrapper failed: {result.stderr[-1000:]}"
    return directory, ""


def wrapper_env() -> dict[str, str]:
    directory, reason = build_wrapper()
    if directory is None:
        raise RuntimeError(reason)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(part for part in (directory, env.get("LD_LIBRARY_PATH")) if part)
    env["RAYD_FAULT_REAL_OPTIX"] = str(real_optix_library())
    return env


def run_with_wrapper(body: str, timeout: int = 300):
    """Run PRELUDE + body in a fresh process that loads the wrapper; return (last JSON line of stdout, stderr)."""
    result = run_script(compose(PRELUDE, body), timeout=timeout, check=True, env=wrapper_env())
    return decode_json_result(result), result.stderr
