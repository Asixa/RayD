# Copyright Xingyu Chen.
# Diagnoses an installed RayD backend, the NVIDIA driver, the visible GPUs, and the OptiX runtime.

"""Backend-neutral installation and driver diagnostics.

The backend doctors (`python -m rayd.torch.doctor`, `python -m rayd.drjit.doctor`)
assemble their reports from these pieces, and the backend runtimes use them to
explain a native library that does not load and to refuse an editable install
whose copied Python sources no longer match the checkout.

Like `device_support.py`, this module imports neither Torch nor Dr.Jit and
creates no CUDA context: the driver is queried through the CUDA driver API and
NVML directly. The Torch backend installs it as `rayd/_impl/diagnostics.py` and
the Dr.Jit backend installs the same source as `rayd/_impl/diagnostics_jit.py`.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

PYTORCH_INDEX = "https://download.pytorch.org/whl"
DOCKER_HINT = (
    "in a container, start it with NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility (or all): the default "
    "compute,utility does not mount the driver's OptiX library libnvoptix.so.1"
)
NO_GPU_HINT = (
    "install the NVIDIA driver and make a GPU visible (CUDA_VISIBLE_DEVICES); in a container, run it with --gpus all "
    "through the NVIDIA Container Toolkit; under WSL2, install the Windows NVIDIA driver and no Linux driver"
)

# Driver branch each OptiX SDK ABI first shipped in. Only versions RayD builds
# with are listed; an unlisted ABI is judged by the driver probe alone.
OPTIX_ABI_DRIVER_BRANCH = {87: 535, 93: 555}
OPTIX_ABI_SDK = {87: "8.0", 93: "8.1", 105: "9.0", 118: "9.1"}

_OPTIX_ERROR_UNSUPPORTED_ABI_VERSION = 7801
_OPTIX_ABI_SCAN_LIMIT = 512

# Installed names of the private sources each distribution ships under a name
# other than their own (see the backend CMake install lists).
_RENAMED_FOR = {"drjit": {"device_support.py": "device_support_jit.py", "diagnostics.py": "diagnostics_jit.py"}}


# ---- build record ----


def read_build_record(extension: str) -> dict | None:
    """The raw `_build_config.json` beside `extension`, or None when absent."""
    try:
        spec = importlib.util.find_spec(extension)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    path = Path(spec.origin).resolve().parent / "_build_config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _base_version(version: str | None) -> str | None:
    return None if version is None else version.split("+", 1)[0]


def _cuda_tag(cuda: str | None) -> str:
    """`cu128` for CUDA 12.8; the release wheels' tag when the record says nothing."""
    if not cuda:
        return "cu128"
    major, _, minor = cuda.partition(".")
    return f"cu{major}{minor.split('.', 1)[0]}"


def torch_install_command(record: dict | None) -> str:
    """The pip command that installs the CUDA torch build a rayd-torch build was compiled against."""
    version = (record or {}).get("framework_version")
    tag = _cuda_tag((record or {}).get("framework_cuda"))
    requirement = f"torch=={version}" if version and "+" in version else f"torch=={_base_version(version) or '2.10.*'}"
    reinstall = "" if "+" in requirement else " --force-reinstall"
    return f'python -m pip install{reinstall} "{requirement}" --index-url {PYTORCH_INDEX}/{tag}'


def torch_mismatch(record: dict | None, version: str, cuda: str | None) -> str | None:
    """Why the installed torch cannot run a rayd-torch build, or None when nothing says it cannot."""
    if cuda is None:
        return f"the installed torch {version} is a CPU-only build (torch.version.cuda is None); RayD needs CUDA torch"
    built = (record or {}).get("framework_version")
    if built and _base_version(built) != _base_version(version):
        return (
            f"the installed torch {version} differs from torch {built}, which this rayd-torch build was compiled "
            "against; RayD's native libraries use Torch's C++ ABI, so only that release loads them"
        )
    built_cuda = (record or {}).get("framework_cuda")
    if built_cuda and built_cuda.split(".", 1)[0] != cuda.split(".", 1)[0]:
        return f"the installed torch uses CUDA {cuda}, but this rayd-torch build was compiled against CUDA {built_cuda}"
    return None


def torch_load_failure(
    library: str, path: Path | None, error: BaseException, version: str, cuda: str | None, record: dict | None
) -> str:
    """Explain a rayd-torch native library that did not load, with the command that fixes it."""
    where = f" from {path}" if path is not None else ""
    lines = [f"RayD Torch could not load its native library {library}{where}:", f"  {type(error).__name__}: {error}"]
    lines.append(
        f"Installed torch: {version}, " + (f"CUDA {cuda}." if cuda else "CPU-only (torch.version.cuda is None).")
    )
    if record is not None and record.get("framework_version"):
        lines.append(
            f"This rayd-torch build was compiled against torch {record['framework_version']} "
            f"(CUDA {record.get('framework_cuda') or 'unknown'})."
        )
    else:
        lines.append("This rayd-torch build carries no record of the torch it was compiled against.")
    cause = torch_mismatch(record, version, cuda)
    if cause is not None:
        lines.append(f"Cause: {cause}.")
        lines.append("Fix: install the matching CUDA torch build:")
        lines.append(f"    {torch_install_command(record)}")
    elif isinstance(error, FileNotFoundError):
        lines.append(
            "Cause: the library is not installed. This happens when rayd.torch is imported from a source tree that "
            "was never built; install rayd-torch or build it (scripts/build_local.sh torch)."
        )
    else:
        lines.append(
            "Cause: the installed torch matches the build record, so the library itself is damaged or comes from a "
            "different build. Reinstall rayd-torch and its torch together:"
        )
        lines.append(f"    {torch_install_command(record)}")
        lines.append("    python -m pip install --force-reinstall --no-deps rayd-torch")
    lines.append("Run `python -m rayd.torch.doctor` for a full report.")
    return "\n".join(lines)


def drjit_load_failure(error: BaseException, version: str, record: dict | None) -> str:
    """Explain a rayd-drjit extension that did not import, with the command that fixes it."""
    lines = ["RayD Dr.Jit could not import its native extension rayd.drjit._C:", f"  {type(error).__name__}: {error}"]
    lines.append(f"Installed drjit: {version}.")
    built = (record or {}).get("framework_version")
    if built:
        lines.append(f"This rayd-drjit build was compiled against drjit {built}.")
        if _base_version(built) != _base_version(version):
            lines.append("Cause: RayD's extension uses Dr.Jit's C++ ABI, so only that drjit release loads it. Fix:")
            lines.append(f'    python -m pip install "drjit=={built}"')
            return "\n".join(lines)
    lines.append(
        "Reinstall rayd-drjit and its pinned drjit together: python -m pip install --force-reinstall rayd-drjit"
    )
    return "\n".join(lines)


# ---- editable installs ----


def stale_editable_sources(backend: str, implementation_file: str) -> tuple[Path, list[str]] | None:
    """The checkout and stale file names when an editable install runs outdated copies of `rayd/_impl`.

    A backend build copies its `rayd/_impl` sources into site-packages, while an
    editable install serves `rayd.<backend>` itself from the checkout. After a
    `git pull` or a local edit the copies silently differ from the checkout
    until the next build. Returns None for a wheel install or a source tree on
    `sys.path`, where the two locations coincide.
    """
    try:
        spec = importlib.util.find_spec(f"rayd.{backend}")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    source = Path(spec.origin).resolve().parent.parent / "_impl"
    installed = Path(implementation_file).resolve().parent
    if source == installed or not source.is_dir():
        return None
    renamed = _RENAMED_FOR.get(backend, {})
    stale = []
    for path in sorted(source.glob("*.py")):
        suffixed = path.name.endswith("_jit.py")
        if (backend == "drjit") != suffixed and path.name not in renamed:
            continue
        target = installed / renamed.get(path.name, path.name)
        if not target.is_file() or target.read_bytes() != path.read_bytes():
            stale.append(path.name)
    return (source.parent.parent.parent, stale) if stale else None


def require_current_editable_install(backend: str, implementation_file: str) -> None:
    """Raise ImportError when an editable install would run Python sources older than its checkout."""
    found = stale_editable_sources(backend, implementation_file)
    if found is None:
        return
    checkout, stale = found
    installed = Path(implementation_file).resolve().parent
    raise ImportError(
        f"rayd-{backend} is an editable install of {checkout}, but its build copied rayd/_impl into {installed} and "
        f"these copies no longer match the checkout: {', '.join(stale)}. Rebuild the backend so the import runs the "
        f"checked-out code:\n    cd {checkout} && scripts/build_local.sh {backend}\n"
        f"or: python -m pip install --no-build-isolation -e {checkout / backend}"
    )


# ---- driver and GPUs ----


@dataclass
class Gpu:
    index: int
    name: str
    compute_capability: int
    total_memory: int | None = None
    free_memory: int | None = None


@dataclass
class DriverInfo:
    """What the CUDA driver API and NVML report; every field is None when its query failed."""

    cuda_error: str | None = None
    driver_cuda_version: int | None = None
    driver_version: str | None = None
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def branch(self) -> int | None:
        if self.driver_version is None:
            return None
        head = self.driver_version.split(".", 1)[0]
        return int(head) if head.isdigit() else None


def _load_library(names: tuple[str, ...]) -> ctypes.CDLL | None:
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _nvml_by_uuid() -> tuple[str | None, dict[str, tuple[int, int]]]:
    """The NVML driver version and each GPU's (total, free) memory by UUID, without a CUDA context."""
    names = ("nvml.dll",) if sys.platform == "win32" else ("libnvidia-ml.so.1", "libnvidia-ml.so")
    nvml = _load_library(names)
    if nvml is None or nvml.nvmlInit_v2() != 0:
        return None, {}
    try:
        buffer = ctypes.create_string_buffer(96)
        version = buffer.value.decode() if nvml.nvmlSystemGetDriverVersion(buffer, 96) == 0 else None
        count = ctypes.c_uint()
        memory: dict[str, tuple[int, int]] = {}
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0:
            for index in range(count.value):
                handle = ctypes.c_void_p()
                uuid = ctypes.create_string_buffer(96)
                info = (ctypes.c_ulonglong * 3)()  # total, free, used
                if (
                    nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0
                    and nvml.nvmlDeviceGetUUID(handle, uuid, 96) == 0
                    and nvml.nvmlDeviceGetMemoryInfo(handle, info) == 0
                ):
                    memory[uuid.value.decode().removeprefix("GPU-")] = (int(info[0]), int(info[1]))
        return version, memory
    finally:
        nvml.nvmlShutdown()


def query_driver() -> DriverInfo:
    """Query the driver for its version and every GPU CUDA_VISIBLE_DEVICES leaves visible."""
    info = DriverInfo()
    info.driver_version, memory = _nvml_by_uuid()
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    cuda = _load_library(names)
    if cuda is None:
        info.cuda_error = f"the CUDA driver library ({' or '.join(names)}) was not found"
        return info
    version = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(version)) == 0:
        info.driver_cuda_version = version.value
    result = cuda.cuInit(0)
    if result != 0:
        # CUDA_ERROR_NO_DEVICE is 100; CUDA_VISIBLE_DEVICES="" also produces it.
        info.cuda_error = "no CUDA device is visible" if result == 100 else f"cuInit failed with CUresult {result}"
        return info
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        info.cuda_error = "cuDeviceGetCount failed"
        return info
    for ordinal in range(count.value):
        handle, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        total = ctypes.c_size_t()
        uuid = (ctypes.c_ubyte * 16)()
        if cuda.cuDeviceGet(ctypes.byref(handle), ordinal) != 0:
            continue
        cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, handle)
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, handle)
        if cuda.cuDeviceGetName(name, len(name), handle) != 0:
            name.value = b"unknown"
        gpu = Gpu(ordinal, name.value.decode(errors="replace"), major.value * 10 + minor.value)
        if cuda.cuDeviceTotalMem_v2(ctypes.byref(total), handle) == 0:
            gpu.total_memory = total.value
        if cuda.cuDeviceGetUuid(uuid, handle) == 0:
            hexed = bytes(uuid).hex()
            key = f"{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}"
            if key in memory:
                gpu.total_memory, gpu.free_memory = memory[key]
        info.gpus.append(gpu)
    return info


# ---- OptiX ----


@dataclass
class OptixProbe:
    library: str | None = None
    error: str | None = None
    expected_abi: int | None = None
    expected_supported: bool | None = None
    highest_abi: int | None = None


def _optix_library_candidates() -> list[str]:
    if sys.platform != "win32":
        return ["libnvoptix.so.1"]
    # The Windows driver installs nvoptix.dll into its driver store rather than System32.
    root = os.environ.get("SystemRoot", r"C:\Windows")
    store = glob.glob(os.path.join(root, "System32", "DriverStore", "FileRepository", "*", "nvoptix.dll"))
    return ["nvoptix.dll", *sorted(store, key=os.path.getmtime, reverse=True)]


def probe_optix(expected_abi: int | None) -> OptixProbe:
    """Load the driver's OptiX library and ask which OptiX ABIs it implements.

    `optixQueryFunctionTable` rejects an ABI it does not implement before it
    looks at the table, so a zero-sized query answers the question without
    initializing OptiX.
    """
    probe = OptixProbe(expected_abi=expected_abi)
    errors = []
    library = None
    for candidate in _optix_library_candidates():
        try:
            library = ctypes.CDLL(candidate)
            probe.library = candidate
            break
        except OSError as error:
            errors.append(str(error))
    if library is None:
        probe.error = "; ".join(errors) or "not found"
        return probe
    try:
        query = library.optixQueryFunctionTable
    except AttributeError:
        probe.error = f"{probe.library} does not export optixQueryFunctionTable"
        return probe
    query.restype = ctypes.c_int
    query.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    supported = [
        abi
        for abi in range(1, _OPTIX_ABI_SCAN_LIMIT)
        if query(abi, 0, None, None, None, 0) != _OPTIX_ERROR_UNSUPPORTED_ABI_VERSION
    ]
    probe.highest_abi = max(supported) if supported else None
    if expected_abi is not None:
        probe.expected_supported = expected_abi in supported
    return probe


def optix_driver_requirement(abi: int | None) -> str:
    """The driver an OptiX ABI needs, in words."""
    if abi is None:
        return "a driver that supports the OptiX ABI of this build"
    sdk = OPTIX_ABI_SDK.get(abi)
    branch = OPTIX_ABI_DRIVER_BRANCH.get(abi)
    name = f"OptiX {sdk} (ABI {abi})" if sdk else f"OptiX ABI {abi}"
    return f"{name}, which needs driver R{branch} or newer" if branch else f"{name}, which needs a driver that ships it"


# ---- report ----


@dataclass
class Report:
    """Lines of facts plus one verdict per check; any failed check makes the exit status non-zero."""

    title: str
    lines: list[str] = field(default_factory=list)
    failures: int = 0

    def fact(self, name: str, value: object) -> None:
        self.lines.append(f"  {name + ':':<26} {value}")

    def section(self, name: str) -> None:
        self.lines.append("")
        self.lines.append(name)

    def check(self, name: str, ok: bool | None, detail: str, fix: str | None = None) -> None:
        """`ok` None is a check that could not run; it is reported but does not fail the report."""
        verdict = "OK" if ok else ("FAIL" if ok is False else "SKIP")
        self.lines.append(f"  [{verdict:>4}] {name}: {detail}")
        if not ok and fix:
            self.lines.append(f"         fix: {fix}")
        if ok is False:
            self.failures += 1

    def render(self) -> str:
        status = "all checks passed" if self.failures == 0 else f"{self.failures} check(s) failed"
        return "\n".join([f"{self.title} doctor", *self.lines, "", f"Result: {status}."])


def _memory(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 2**30:.1f} GiB"


def add_system(report: Report) -> None:
    report.section("System")
    report.fact("OS", platform.platform())
    report.fact("Python", sys.version.split()[0] + f" ({sys.executable})")


def add_installation(report: Report, distribution: str, record: dict | None) -> None:
    report.section("Installed build")
    try:
        report.fact(distribution, importlib.metadata.version(distribution))
    except importlib.metadata.PackageNotFoundError:
        report.fact(distribution, "not installed as a distribution (source tree)")
    if record is None:
        report.check("build record", None, "no _build_config.json beside the extension; the build is not described")
        return
    sass = record.get("cuda_sass")
    report.fact("CUDA cubins", "compiler default" if sass is None else ", ".join(f"sm_{arch}" for arch in sass))
    report.fact("CUDA PTX", ", ".join(f"compute_{arch}" for arch in record.get("cuda_ptx", ())) or "none")
    report.fact("OptiX PTX", f"compute_{record['optix_ptx']}" if record.get("optix_ptx") else "unknown")
    report.fact(
        "OptiX ABI", optix_driver_requirement(record.get("optix_abi")) if record.get("optix_abi") else "unknown"
    )
    report.fact("CUDA toolkit", record.get("cuda_toolkit") or "unknown")
    framework = record.get("framework", record.get("backend"))
    built = record.get("framework_version") or "unknown"
    if record.get("framework_cuda"):
        built += f" (CUDA {record['framework_cuda']})"
    report.fact(f"built against {framework}", built)


def add_driver(report: Report, driver: DriverInfo, record: dict | None) -> None:
    report.section("Driver and GPUs")
    report.fact("driver", driver.driver_version or "unknown (NVML unavailable)")
    if driver.driver_cuda_version is not None:
        value = driver.driver_cuda_version
        report.fact("driver CUDA version", f"{value // 1000}.{(value % 1000) // 10}")
    for gpu in driver.gpus:
        report.fact(
            f"GPU {gpu.index}",
            f"{gpu.name}, sm_{gpu.compute_capability}, {_memory(gpu.free_memory)} free of {_memory(gpu.total_memory)}",
        )
    if driver.cuda_error is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        hint = f" (CUDA_VISIBLE_DEVICES={visible!r})" if visible is not None else ""
        report.check("CUDA device", False, driver.cuda_error + hint, NO_GPU_HINT)
    else:
        report.check("CUDA device", True, f"{len(driver.gpus)} visible")
    abi = (record or {}).get("optix_abi")
    branch = OPTIX_ABI_DRIVER_BRANCH.get(abi) if abi else None
    if branch is not None and driver.branch is not None:
        report.check(
            "driver version",
            driver.branch >= branch,
            f"R{driver.branch}; {optix_driver_requirement(abi)}",
            f"update the NVIDIA driver to R{branch} or newer",
        )


def add_device_support(report: Report, targets, driver: DriverInfo, device_support) -> None:
    """One check per GPU against the compiled targets, using the backend's own device_support module."""
    for gpu in driver.gpus:
        found = device_support.problems(targets, gpu.compute_capability) if targets is not None else []
        if targets is None:
            report.check(f"GPU {gpu.index} code", None, "no build record to compare against")
            continue
        detail = "; ".join(found) if found else f"the build contains code for sm_{gpu.compute_capability}"
        fix = None
        if found:
            fix = (
                "build from source for this GPU: "
                + device_support.build_command(targets.backend, gpu.compute_capability).splitlines()[-1]
            )
        report.check(f"GPU {gpu.index} code", not found, detail, fix)


def add_optix(report: Report, record: dict | None, disabled: bool) -> OptixProbe:
    report.section("OptiX")
    probe = probe_optix((record or {}).get("optix_abi"))
    if disabled:
        report.check("OptiX", True, "disabled by RAYD_DISABLE_OPTIX; RayD uses its CUDA trace backend")
        return probe
    if probe.library is None:
        report.check("OptiX library", False, f"the driver's OptiX library is not loadable ({probe.error})", DOCKER_HINT)
        return probe
    report.check("OptiX library", True, f"{probe.library} loaded")
    if probe.highest_abi is not None:
        sdk = OPTIX_ABI_SDK.get(probe.highest_abi)
        report.fact("driver offers OptiX ABI", f"up to {probe.highest_abi}" + (f" (OptiX {sdk})" if sdk else ""))
    if probe.expected_supported is not None:
        report.check(
            "OptiX ABI",
            probe.expected_supported,
            f"the build requests {optix_driver_requirement(probe.expected_abi)}",
            "update the NVIDIA driver",
        )
    return probe
