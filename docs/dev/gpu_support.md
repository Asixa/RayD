# GPU Support, Older GPUs, and Source Builds

This document records which NVIDIA GPUs RayD can run on, why, what happens on a
GPU the prebuilt wheels do not cover, and how to build RayD from source for a
specific GPU. The release wheel matrix itself is owned by
[`ci_build_matrix.md`](ci_build_matrix.md).

## Summary

| GPU family | Compute capability | Prebuilt `rayd-drjit` | Prebuilt `rayd-torch` | From source |
| --- | --- | --- | --- | --- |
| Kepler and older | <= 3.7 | no | no | no |
| Maxwell (GTX 900, Tesla M) | 5.0-5.3 | no | no | no |
| Pascal (GTX 10, Tesla P) | 6.0-6.2 | no | no | no |
| Volta (Tesla V100, TITAN V) | 7.0 | yes | yes | yes |
| Jetson Xavier | 7.2 | no wheel (x86-64 only) | no wheel (x86-64 only) | yes, untested |
| Turing through Blackwell, discrete | 7.5-12.x | yes | yes | yes |
| Jetson Orin | 8.7 | no wheel (x86-64 only) | no wheel (x86-64 only) | yes, untested |
| Jetson Thor | 11.0 | no wheel (x86-64 only) | no wheel (x86-64 only) | untested; CUDA 12.8 cannot target `sm_110` |
| Newer than Blackwell | > 12.x | through `compute_120` PTX | through `compute_120` PTX | yes, with a toolkit that targets it |

"Yes" means the build contains code for that GPU family. It does not mean RayD
has run on that GPU; the [verification record](#verification-record-2026-11-01)
says what has.

**The minimum supported compute capability is 7.0 (Volta), for prebuilt wheels
and source builds alike.** RayD's own device code requires it (next section),
so a source build cannot extend support below `sm_70`.

Torch builds made with CUDA 12.9 or newer need a newer driver than the rest of
RayD. See
[Driver requirements](#driver-requirements).

## Verification record, 2026-11-01

This is what one acceptance run measured on one machine. It is evidence, not a
promise for any particular card.

Machine: RTX 4090 Laptop GPU (`sm_89`), driver 572.16, Windows 11, Torch
2.10.0+cu128, CUDA 12.8. The build was a temporary merge of this work at 48d222a
with the crash-hardening work at 6ab8ac4, without the later Jetson device-check
changes. Without the crash-hardening commits, some Torch OptiX pipelines fail on
this driver.

The per-architecture runs used the Torch backend only, with 70 jobs per target:

| Target | What ran on the RTX 4090 | Result |
| --- | --- | --- |
| `sm_89` | Native code, plus the full 280-job set | All pass; baseline for the rows below |
| `sm_80`, `sm_86` | The `sm_80` or `sm_86` cubin itself | All 70 pass; 65 byte-identical to `sm_89` |
| `sm_70`, `sm_75` | PTX translated for the RTX 4090, not real old-GPU machine code | All 70 pass; 65 byte-identical to `sm_89` |
| `sm_87`, `sm_90`, `sm_100`, `sm_101`, `sm_120` | Nothing: compile-checked only | `cuobjdump` finds the image in `_stable_ops`, `_legacy_ops`, and Dr.Jit `_C` |

The 5 jobs that were not byte-identical are diffraction jobs, and those also
differ between two runs of one build.

A one-kernel program with no RayD code, built for `sm_87`, fails on the RTX 4090
with "no kernel image", while `sm_80` and `sm_86` builds run.

Dr.Jit ran on `sm_89` only, and did not cover diffraction, EPC, or accumulated
reflections.

Not run at all in that acceptance run: a real A6000 or any other physical GPU,
any other driver, and Linux.

Since then (2026-11-09) source builds of both backends for `sm_86`, made with a
conda CUDA 12.8 toolkit, ran their test suites on Linux with two RTX A6000 GPUs
and driver 595.99.02. That covers `sm_86` cubins on real `sm_86` hardware and
Linux, but not the release wheels or any other architecture.

## Why sm_70 is the floor

Building the Dr.Jit backend with `RAYD_CUDA_ARCHS=61,89` on CUDA 12.8 fails in
16 of its CUDA and OptiX translation units, for two independent reasons:

- `include/rayd/math.h` implements same-cell warp aggregation for the
  accumulation atomics (`warp_cell_group`, `atomic_add_same_cell`) with
  `__match_any_sync`, which exists only on compute capability 7.0 and newer
  ([CUDA C++ Programming Guide, warp match functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-match-functions)).
  Every OptiX ray-tracing module and most CUDA kernels include it.
- `src/scene/multipath_jit.cu` passes launch parameters as
  `__grid_constant__`, which nvcc rejects below `compute_70`.

**Decided on 2026-11-01: Pascal and Maxwell are not supported, and no pre-Volta
atomics work is planned.** Supporting them would mean a pre-Volta implementation
of the warp aggregation. That changes the order of the accumulation atomics,
which the numerical contracts in `AGENTS.md` (ADR-0026 and ADR-0038) treat as
part of the result.

## What happens today on a GPU the wheels do not cover

Each layer has its own compatibility rule:

| Layer | Rule | Below sm_70 | Volta (sm_70) |
| --- | --- | --- | --- |
| RayD CUDA kernels | A cubin for `sm_X.y` runs on `sm_X.z` with `z >= y`, except that [Jetson and embedded](#jetson-and-embedded-architectures) cubins never cross; PTX JIT-compiles forward to newer GPUs only ([CUDA application compatibility](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#application-compatibility)). Release wheels carry `sm_70`..`sm_120` cubins and `compute_120` PTX. | No kernel image; launches fail with "no kernel image is available for execution on the device". | Runs (`sm_70` cubin). |
| RayD OptiX programs | OptiX rejects a module whose PTX target is newer than the device, so PTX must be built for the oldest GPU to support ([NVIDIA OptiX forum guidance](https://forums.developer.nvidia.com/t/problem-with-running-optix-6-5-program-invalid-value-for-gpu-architecture/157284/2)). Dr.Jit ships committed `compute_70` PTX and Torch compiles `compute_70` PTX ([ADR-0044](../adr/0044-torch-optix-ptx-at-the-volta-floor.md)). | Module creation fails. | Both backends target it. |
| Dr.Jit 1.3.1 | Generates its own PTX for the device at run time; requires compute capability 5.0 or newer ([drjit-core](https://github.com/mitsuba-renderer/drjit-core)). | Dr.Jit itself works; RayD kernels do not. | Runs. |
| PyTorch 2.10 | `cu128` wheels contain `sm_70`..`sm_120`; Maxwell and Pascal need the `cu126` wheels ([PyTorch announcement](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)). | PyTorch `cu128` itself cannot run. | Runs. |
| CUDA toolkit | CUDA 12.x still compiles `sm_50`..`sm_72`; CUDA 13.0 removed offline compilation for Maxwell, Pascal, and Volta ([CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)). | Not relevant: RayD cannot compile. | Build with a 12.x toolkit, not 13.x. |
| OptiX runtime | Both backends target the OptiX 8.1 ABI, which needs an R555 or newer driver ([OptiX 8.1 release](https://github.com/NVIDIA/optix-sdk/releases/tag/v8.1.0)). The R580 branch is the last driver for Maxwell, Pascal, and Volta ([Phoronix](https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW)). | Not relevant. | Use an R555..R580 driver, and see [Driver requirements](#driver-requirements) for builds made with CUDA 12.9 or newer. |

Before this change an unsupported GPU therefore surfaced as a CUDA launch
error or an OptiX module or pipeline creation error from deep inside the first
native call. The runtime check below replaces that with one early error.

### Jetson and embedded architectures

NVIDIA's rule for cubins has an exception, in the Binary Compatibility section of
the [CUDA 12.8 programming guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html):
"It is not supported for Tegra." The same note rules out compatibility between
desktop and Tegra. The device check therefore lets these architectures run only
their own cubin:

| Architecture | Why it is on the list | Confidence |
| --- | --- | --- |
| `sm_87` | NVIDIA lists Jetson Orin at 8.7 ([compute capabilities](https://developer.nvidia.com/cuda-gpus)) | Failure on `sm_89` seen on hardware; the reverse direction was not |
| `sm_72` | NVIDIA lists Jetson Xavier at 7.2 ([legacy list](https://developer.nvidia.com/cuda/gpus/legacy)) | From NVIDIA's rule; not run |
| `sm_110` | NVIDIA lists Jetson T4000 and T5000 at 11.0 ([compute capabilities](https://developer.nvidia.com/cuda-gpus)) | From NVIDIA's rule; not run. CUDA 12.8 `nvcc` cannot target it |
| `sm_101` | A PyTorch warning, [quoted in an NVIDIA forum post](https://forums.developer.nvidia.com/t/jetson-agx-thor-official-pytorch-25-08-container-works-for-conv2d-and-resnet18-but-pip-installed-pytorch-2-12-0-dev-cu128-fails-with-no-kernel-im/364270), excludes 10.1 from `sm_100` hardware | **Unconfirmed.** The [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html) call `SM_101` Blackwell and name no product, and NVIDIA's table has no 10.1 row |

If a discrete GPU does report 10.1, the cost is a request to rebuild for `sm_101`,
which release wheels already include. `RAYD_SKIP_DEVICE_CHECK=1` overrides it.

Other architectures:

- CUDA 12.8 `nvcc` accepts no other Jetson-class target. It lists `sm_70`, `sm_72`,
  `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, and
  `sm_120`, plus the suffixed forms below and the pre-Volta targets.
- CUDA 12.9 adds targets this list was not checked against, because only the 12.8
  compiler was available here.
- NVIDIA lists GB10 (DGX Spark) at 12.1 without saying whether it counts as Tegra.
  It stays outside the list, so an `sm_120` cubin is assumed to run on it; that is
  unconfirmed.
- Suffixed targets such as `sm_90a` run only on their exact architecture, per the
  same guide. `RAYD_CUDA_ARCHS` does not accept them.

Whether OptiX is available on Jetson at all is also unconfirmed, and no RayD build
has run on Jetson.

## Driver requirements

Both backends need an R555 or newer driver for the OptiX 8.1 ABI. CUDA 12.9 or
newer raises that to R575 for what it compiles.

**OptiX code compiled by CUDA 12.9 or newer needs R575 or newer.** That compiler
turns the complex square root in `include/rayd/math.h` into `copysign.f32`, which
older drivers cannot compile. On driver 572.16 the call fails with:

```text
RuntimeError: OptiX error in optixModuleCreate(multipath)
```

and the OptiX log printed before it says:

```text
New backend is missing implementation for PTX intrinsic optix.ptx.copysign.f32
```

The input is not at fault, so update the driver or rebuild with CUDA 12.8 instead
of changing the call.

This reaches Torch builds made with CUDA 12.9 or newer. Prebuilt `rayd-torch` wheels
and Torch builds with CUDA 12.8 emit no such intrinsic: a CUDA 12.8 build here has
none in any of its 11 OptiX modules. No committed Dr.Jit PTX contains it either,
and Dr.Jit `accumulate_reflections` passes on driver 572.16. The modules that use
the square root were generated with CUDA 12.8 (ADR-0045), so regenerating them with
CUDA 12.9 or newer would bring this requirement back to Dr.Jit.

R575 is the driver branch CUDA 12.9 requires
([CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html):
576.02 on Windows, 575.51.03 on Linux). The failure was reproduced on 572.16 with a
Dr.Jit module generated by CUDA 12.9; no R575 or newer driver and no CUDA 12.9 Torch
build has been tested.

RayD checks the driver against the OptiX ABI at run time. When the driver's
OptiX library is missing or older than the ABI the build requests,
`trace_backend="auto"` falls back to the CUDA trace backend and warns once per
process with the reason in both backends (for an old driver, Torch also names
the CUDA version the driver supports and the branch it needs); an explicit
`trace_backend="optix"` raises the same reason. `RAYD_DISABLE_OPTIX=1` selects
the CUDA trace backend without a warning. `python -m rayd.torch.doctor`
and `python -m rayd.drjit.doctor` print the driver version, the OptiX ABI the
build requests (recorded as `optix_abi` in `_build_config.json`), the highest
ABI the driver implements, and fail the check when the driver is too old. The
CUDA 12.9 `copysign.f32` requirement above is not detected ahead of time.

## Runtime compatibility check

Every native backend build writes `_build_config.json` beside its extension
module. It records the cubin architectures, the CUDA PTX architectures, and the
OptiX PTX architecture compiled into that build, plus the OptiX ABI it requests,
the CUDA toolkit, and the framework version it compiled against (for Torch, also
`torch.version.cuda`) (`cmake/RayDCudaArchitectures.cmake`). The Torch runtime
uses the framework fields to explain a native library that does not load. `rayd/_impl/device_support.py`
compares the record with the device's compute capability:

- CUDA kernels run when a cubin has the device's major version and an equal or
  lower minor version, or when a PTX target is at or below the device. Jetson
  and embedded architectures (`sm_72`, `sm_87`, `sm_101`, `sm_110`) run only
  their own cubin, and other GPUs never run those cubins.
- OptiX programs run when their PTX target is at or below the device; this part
  is skipped when `RAYD_DISABLE_OPTIX=1` turns OptiX off.
- Anything below `sm_70` is rejected outright, without a build suggestion.

When a device fails, `UnsupportedDeviceError` (a `RuntimeError`) names the
distribution, the device index and name, its compute capability, each failing
layer, the GPUs the installed build runs on, and the exact command that builds RayD from source for that GPU.

The check runs where each backend first needs a device:

- `rayd.torch` reads the record at import without touching CUDA. It checks the
  current device on the first native call once CUDA is initialized, each mesh
  device in `Scene.build()`, and each replica device of `Scene(devices=...)`.
  A device that passes is remembered; one that fails raises on every call.
- `rayd.drjit` selects its device after import, so the import queries every
  visible GPU through the CUDA driver API. It raises when none can run the
  build and warns about each one that cannot when at least one can.

A build without a record (a hand-assembled tree, or a build from before this
change) is not checked. `RAYD_SKIP_DEVICE_CHECK=1` disables the check.

Example on a Volta GPU with a Torch build made only for `sm_89`:

```text
rayd-torch cannot run on CUDA device 0 'Tesla V100-SXM2-16GB' (compute capability 7.0, sm_70).
  - no CUDA kernel image runs on sm_70 (cubins: sm_89; PTX: none)
This build runs on discrete GPUs sm_8x from sm_89.
Build RayD from source for this GPU (CUDA Toolkit 12.x and OptiX SDK 8.1+ headers):
    git clone https://github.com/Asixa/RayD
    cd RayD
    RAYD_CUDA_ARCHS=70 python -m pip install --no-build-isolation ./torch
CUDA 13 cannot compile for this GPU; use a CUDA 12.x toolkit.
Set RAYD_SKIP_DEVICE_CHECK=1 to skip this check.
```

## Building from source for a specific GPU

Building from a Git checkout is the supported route for GPUs the wheels do not
cover, for a smaller single-GPU build, and for platforms without wheels. The
backend distributions publish no sdist: each backend builds from shared
`include/`, `src/`, `cmake/`, and `python/` trees one directory above its
`pyproject.toml`, which a backend sdist cannot contain, and an sdist would let
`pip` silently start a CUDA, OptiX, and MSVC build on any platform without a
matching wheel. `tests/packaging/test_source_build_route.py` keeps the release
workflow from publishing one.

Requirements:

- CUDA Toolkit 12.x (12.8 is what release CI uses; Volta needs 12.x, not 13.x,
  and 12.9 raises the driver minimum as [described above](#driver-requirements))
- OptiX SDK 8.1 or newer headers for `rayd-torch` (set `OPTIX_INCLUDE_DIR` or
  `OPTIX_PATH`; the `NVIDIA/optix-dev` repository at tag `v8.1.0` is what CI
  uses). `rayd-drjit` embeds committed OptiX PTX and needs no OptiX SDK.
- Visual Studio 2022 C++ tools on Windows, or GCC 11+ on Linux
- the backend's framework already installed in the target environment:
  `drjit==1.3.1`, `nanobind==2.11.0`, `scikit-build-core`, `ninja`, and
  `cmake` for `rayd-drjit`; CUDA `torch==2.10.*`, `scikit-build-core`,
  `ninja`, and `cmake` for `rayd-torch`

Linux:

```bash
git clone https://github.com/Asixa/RayD
cd RayD
RAYD_CUDA_ARCHS=native python -m pip install --no-build-isolation ./drjit
RAYD_CUDA_ARCHS=native python -m pip install --no-build-isolation ./torch
```

Linux, editable, with one build directory per target list (`--python <path>`
selects the environment; the script clears the release-only architecture
variables and stops with instructions when the OptiX headers or a CUDA Torch are
missing):

```bash
scripts/build_local.sh drjit --cuda-arch native
scripts/build_local.sh torch --cuda-arch native
```

Windows (initializes the MSVC environment and installs editable):

```powershell
git clone https://github.com/Asixa/RayD
cd RayD
.\scripts\build_local.cmd -Backend drjit -CudaArch native
.\scripts\build_local.cmd -Backend torch -CudaArch native
```

`--no-build-isolation` builds against the `drjit` or `torch` already in the
environment instead of downloading a pinned copy into a temporary one, which
for Torch would otherwise fetch a CPU-only wheel from PyPI.

### `RAYD_CUDA_ARCHS`

`RAYD_CUDA_ARCHS` (environment variable or CMake cache entry) selects the GPUs
a source build targets, identically for both backends:

| Value | Meaning |
| --- | --- |
| `native` | every distinct GPU `nvidia-smi` reports on this machine |
| `70` or `7.0` | native code for one architecture |
| `70,86,89` | native code for several; commas, semicolons, or spaces separate entries |
| `8.9+PTX` | native code plus forward-compatible PTX for that entry |

Entries below `70` are rejected at configure time with the reason above. When
`RAYD_CUDA_ARCHS` is unset, the Dr.Jit backend targets the first local GPU and
the Torch backend uses CMake's `native`, as before; on a machine where CMake or
Torch sees no GPU, the Torch configure fails and asks for `RAYD_CUDA_ARCHS`.

The value drives every CUDA target: the Dr.Jit explicit `nvcc` commands, the
Torch CMake CUDA targets, and Caffe2's `TORCH_CUDA_ARCH_LIST`. It does not change
the OptiX PTX, which is `compute_70` in every build of both backends. Release
builds never set
`RAYD_CUDA_ARCHS` and keep their explicit `RAYD_CUDA_GENCODE_ARCHES`,
`RAYD_CUDA_PTX_ARCH`, `CMAKE_CUDA_ARCHITECTURES`, and `TORCH_CUDA_ARCH_LIST`
lists; setting both forms at once fails at configure time in both backends and
names the conflicting variable. For Torch that is an explicit
`CMAKE_CUDA_ARCHITECTURES` (environment or cache, other than CMake's default or
the `CUDAARCHS` initializer that conda CUDA packages set) or a
`TORCH_CUDA_ARCH_LIST`.

### Verifying a source build

```bash
python -c "import json, pathlib, rayd.drjit._C as c; print((pathlib.Path(c.__file__).parent / '_build_config.json').read_text())"
python -c "import json, pathlib, rayd.torch._C as c; print((pathlib.Path(c.__file__).parent / '_build_config.json').read_text())"
```

The printed record lists the targets the runtime check accepts. To inspect
the binary itself, run `cuobjdump --list-elf` on the extension (`_C`) or on
`_legacy_ops`/`_stable_ops`; release CI does the same through
`drjit/scripts/verify_cuda_binary_arches.py`.
