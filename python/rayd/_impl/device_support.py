# Copyright Xingyu Chen.
# Checks whether a CUDA device can run the GPU code compiled into an installed RayD backend.

"""Backend-neutral GPU compatibility check.

Each native backend build records the CUDA and OptiX targets it compiled in
`_build_config.json` beside its extension module. This module compares that
record with a device's compute capability and, when the device cannot run the
build, raises one error naming the GPU, the supported range, and the command
that builds RayD from source for that GPU.

It imports neither Torch nor Dr.Jit. The Torch backend installs it as
`rayd/_impl/device_support.py` and the Dr.Jit backend installs the same source
as `rayd/_impl/device_support_jit.py`; each passes its own device query in.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# RayD device code uses Volta warp matching (__match_any_sync) and
# __grid_constant__ kernel parameters, so nothing below sm_70 can run it, even
# from source. cmake/RayDCudaArchitectures.cmake holds the same floor.
MIN_COMPUTE_CAPABILITY = 70

SKIP_ENVIRONMENT = "RAYD_SKIP_DEVICE_CHECK"
DISABLE_OPTIX_ENVIRONMENT = "RAYD_DISABLE_OPTIX"
REPOSITORY_URL = "https://github.com/Asixa/RayD"

_DISTRIBUTIONS = {"drjit": "rayd-drjit", "torch": "rayd-torch"}


class UnsupportedDeviceError(RuntimeError):
    """The installed RayD build contains no code this CUDA device can run."""


@dataclass(frozen=True)
class BuildTargets:
    """CUDA and OptiX device code compiled into one backend build.

    `sass` lists the native cubin architectures and is None when the build
    used the compiler default and recorded nothing. `ptx` lists the CUDA PTX
    virtual architectures the driver can JIT forward. `optix_ptx` is the
    virtual architecture of the OptiX programs.
    """

    backend: str
    sass: tuple[int, ...] | None
    ptx: tuple[int, ...]
    optix_ptx: int | None

    @property
    def distribution(self) -> str:
        return _DISTRIBUTIONS.get(self.backend, f"rayd-{self.backend}")

    @property
    def lowest_supported(self) -> int | None:
        """Lowest compute capability that can run every compiled component."""
        floors = []
        if self.sass is not None:
            floors.append(min(self.sass + self.ptx))
        if self.optix_ptx is not None:
            floors.append(self.optix_ptx)
        return max(floors + [MIN_COMPUTE_CAPABILITY]) if floors else None


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    compute_capability: int


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _sm(value: int) -> str:
    return f"sm_{value}"


def _version(value: int) -> str:
    return f"{value // 10}.{value % 10}"


def parse_targets(record: dict) -> BuildTargets:
    """Validate one `_build_config.json` record."""
    sass = record.get("cuda_sass")
    optix = record.get("optix_ptx")
    return BuildTargets(
        backend=str(record["backend"]),
        sass=None if sass is None else tuple(sorted(int(arch) for arch in sass)),
        ptx=tuple(sorted(int(arch) for arch in record.get("cuda_ptx", ()))),
        optix_ptx=None if optix is None else int(optix),
    )


def load_build_targets(backend: str, extension: str) -> BuildTargets | None:
    """Read the build record that sits beside `extension`, or None when absent.

    The record is located through the extension's import spec so editable
    installs, whose Python sources and native files live in different trees,
    resolve it the same way as wheels.
    """
    try:
        spec = importlib.util.find_spec(extension)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    path = Path(spec.origin).resolve().parent / "_build_config.json"
    if not path.is_file():
        return None
    targets = parse_targets(json.loads(path.read_text(encoding="utf-8")))
    if targets.backend != backend:
        raise RuntimeError(f"{path} describes the {targets.backend!r} backend, not {backend!r}.")
    return targets


def problems(targets: BuildTargets, compute_capability: int, *, optix: bool = True) -> list[str]:
    """Reasons `targets` cannot run on `compute_capability`; empty when it can.

    A cubin runs on its own major architecture at the same or a higher minor
    revision, PTX runs on its own architecture and every later one, and OptiX
    rejects programs whose PTX target is newer than the device.
    """
    cc = compute_capability
    found = []
    if cc < MIN_COMPUTE_CAPABILITY:
        found.append(
            f"RayD requires compute capability {_version(MIN_COMPUTE_CAPABILITY)} (Volta) or newer because its "
            f"device code uses Volta-only warp intrinsics; no RayD build, prebuilt or from source, runs on {_sm(cc)}"
        )
        return found
    if targets.sass is not None:
        native = any(arch // 10 == cc // 10 and arch <= cc for arch in targets.sass)
        forward = any(arch <= cc for arch in targets.ptx)
        if not native and not forward:
            images = ", ".join(_sm(arch) for arch in targets.sass) or "none"
            jit = ", ".join(f"compute_{arch}" for arch in targets.ptx) or "none"
            found.append(f"no CUDA kernel image runs on {_sm(cc)} (cubins: {images}; PTX: {jit})")
    if optix and targets.optix_ptx is not None and targets.optix_ptx > cc:
        found.append(
            f"the OptiX programs are compiled for compute_{targets.optix_ptx}, newer than {_sm(cc)} "
            f"(set {DISABLE_OPTIX_ENVIRONMENT}=1 to run only the CUDA paths)"
        )
    return found


def build_command(backend: str, compute_capability: int, platform: str | None = None) -> str:
    """The documented command that builds `backend` from a checkout for one GPU."""
    platform = sys.platform if platform is None else platform
    arch = str(compute_capability)
    if platform == "win32":
        build = f".\\scripts\\build_local.cmd -Backend {backend} -CudaArch {arch}"
    else:
        build = f"RAYD_CUDA_ARCHS={arch} python -m pip install --no-build-isolation ./{backend}"
    return f"git clone {REPOSITORY_URL}\ncd RayD\n{build}"


def message(targets: BuildTargets, device: Device, found: list[str], platform: str | None = None) -> str:
    cc = device.compute_capability
    lines = [
        f"{targets.distribution} cannot run on CUDA device {device.index} '{device.name}' "
        f"(compute capability {_version(cc)}, {_sm(cc)})."
    ]
    lines += [f"  - {item}" for item in found]
    lowest = targets.lowest_supported
    if lowest is not None:
        lines.append(f"This build supports {_sm(lowest)} and newer.")
    if cc < MIN_COMPUTE_CAPABILITY:
        return "\n".join(lines)
    lines.append("Build RayD from source for this GPU (CUDA Toolkit 12.x and OptiX SDK 8.1+ headers):")
    lines += ["    " + line for line in build_command(targets.backend, cc, platform).splitlines()]
    if cc < 75:
        lines.append("CUDA 13 cannot compile for this GPU; use a CUDA 12.x toolkit.")
    lines.append(f"Set {SKIP_ENVIRONMENT}=1 to skip this check.")
    return "\n".join(lines)


def check(targets: BuildTargets | None, device: Device) -> None:
    """Raise UnsupportedDeviceError when `device` cannot run `targets`."""
    if targets is None or _flag(SKIP_ENVIRONMENT):
        return
    found = problems(targets, device.compute_capability, optix=not _flag(DISABLE_OPTIX_ENVIRONMENT))
    if found:
        raise UnsupportedDeviceError(message(targets, device, found))


def check_visible(targets: BuildTargets | None, devices: list[Device] | None) -> None:
    """Raise when no visible device can run `targets`; warn about the ones that cannot.

    For a backend that picks its device after import: one unusable GPU on a
    machine that also has a usable one must not block the import.
    """
    if targets is None or not devices or _flag(SKIP_ENVIRONMENT):
        return
    optix = not _flag(DISABLE_OPTIX_ENVIRONMENT)
    rejected = [(device, problems(targets, device.compute_capability, optix=optix)) for device in devices]
    rejected = [(device, found) for device, found in rejected if found]
    if len(rejected) == len(devices):
        device, found = rejected[0]
        raise UnsupportedDeviceError(message(targets, device, found))
    for device, found in rejected:
        warnings.warn(message(targets, device, found), RuntimeWarning, stacklevel=3)


def driver_devices() -> list[Device] | None:
    """Every visible CUDA device from the driver API, or None when it is unavailable.

    The driver is queried directly so this works without a framework and
    without creating a CUDA context.
    """
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    library = None
    for name in names:
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if library is None or library.cuInit(0) != 0:
        return None
    count = ctypes.c_int()
    if library.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return None
    devices = []
    for ordinal in range(count.value):
        handle, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        buffer = ctypes.create_string_buffer(256)
        if (
            library.cuDeviceGet(ctypes.byref(handle), ordinal) != 0
            # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR.
            or library.cuDeviceGetAttribute(ctypes.byref(major), 75, handle) != 0
            or library.cuDeviceGetAttribute(ctypes.byref(minor), 76, handle) != 0
        ):
            return None
        if library.cuDeviceGetName(buffer, len(buffer), handle) != 0:
            buffer.value = b"unknown"
        devices.append(Device(ordinal, buffer.value.decode(errors="replace"), major.value * 10 + minor.value))
    return devices
