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
| Volta (Tesla V100, TITAN V) | 7.0 | yes | CUDA paths only; OptiX needs a source build | yes |
| Xavier | 7.2 | no x86-64 wheel | no x86-64 wheel | yes, untested |
| Turing through Blackwell | 7.5-12.x | yes | yes | yes |
| Newer than Blackwell | > 12.x | through `compute_120` PTX | through `compute_120` PTX | yes, with a toolkit that targets it |

**The minimum supported compute capability is 7.0 (Volta), for prebuilt wheels
and source builds alike.** RayD's own device code requires it (next section),
so a source build cannot extend support below `sm_70`.

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

Supporting Pascal or Maxwell would mean a pre-Volta implementation of the warp
aggregation. That changes the order of the accumulation atomics, which the
numerical contracts in `AGENTS.md` (ADR-0026 and ADR-0038) treat as part of the
result, so it is a product decision and not a build option.

## What happens today on a GPU the wheels do not cover

Each layer has its own compatibility rule:

| Layer | Rule | Below sm_70 | Volta (sm_70) |
| --- | --- | --- | --- |
| RayD CUDA kernels | A cubin for `sm_X.y` runs on `sm_X.z` with `z >= y`; PTX JIT-compiles forward to newer GPUs only ([CUDA application compatibility](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#application-compatibility)). Release wheels carry `sm_70`..`sm_120` cubins and `compute_120` PTX. | No kernel image; launches fail with "no kernel image is available for execution on the device". | Runs (`sm_70` cubin). |
| RayD OptiX programs | OptiX rejects a module whose PTX target is newer than the device, so PTX must be built for the oldest GPU to support ([NVIDIA OptiX forum guidance](https://forums.developer.nvidia.com/t/problem-with-running-optix-6-5-program-invalid-value-for-gpu-architecture/157284/2)). Dr.Jit ships committed `compute_70` PTX; Torch compiles `compute_75` PTX (ADR-0035 divergence D2). | Module creation fails. | Dr.Jit runs; Torch OptiX module creation fails, as ADR-0035 already notes. |
| Dr.Jit 1.3.1 | Generates its own PTX for the device at run time; requires compute capability 5.0 or newer ([drjit-core](https://github.com/mitsuba-renderer/drjit-core)). | Dr.Jit itself works; RayD kernels do not. | Runs. |
| PyTorch 2.10 | `cu128` wheels contain `sm_70`..`sm_120`; Maxwell and Pascal need the `cu126` wheels ([PyTorch announcement](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)). | PyTorch `cu128` itself cannot run. | Runs. |
| CUDA toolkit | CUDA 12.x still compiles `sm_50`..`sm_72`; CUDA 13.0 removed offline compilation for Maxwell, Pascal, and Volta ([CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)). | Not relevant: RayD cannot compile. | Build with a 12.x toolkit, not 13.x. |
| OptiX runtime | Both backends target the OptiX 8.1 ABI, which needs an R555 or newer driver ([OptiX 8.1 release](https://github.com/NVIDIA/optix-sdk/releases/tag/v8.1.0)). The R580 branch is the last driver for Maxwell, Pascal, and Volta ([Phoronix](https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW)). | Not relevant. | Use an R555..R580 driver. |

Before this change an unsupported GPU therefore surfaced as a CUDA launch
error or an OptiX module or pipeline creation error from deep inside the first
native call. The runtime check below replaces that with one early error.

## Runtime compatibility check

Every native backend build writes `_build_config.json` beside its extension
module. It records the cubin architectures, the CUDA PTX architectures, and the
OptiX PTX architecture compiled into that build
(`cmake/RayDCudaArchitectures.cmake`). `rayd/_impl/device_support.py`
compares the record with the device's compute capability:

- CUDA kernels run when a cubin has the device's major version and an equal or
  lower minor version, or when a PTX target is at or below the device.
- OptiX programs run when their PTX target is at or below the device; this part
  is skipped when `RAYD_DISABLE_OPTIX=1` turns OptiX off.
- Anything below `sm_70` is rejected outright, without a build suggestion.

When a device fails, `UnsupportedDeviceError` (a `RuntimeError`) names the
distribution, the device index and name, its compute capability, each failing
layer, the lowest compute capability the installed build supports, and the
exact command that builds RayD from source for that GPU.

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

Example on a Volta GPU with the prebuilt Torch wheel:

```text
rayd-torch cannot run on CUDA device 0 'Tesla V100-SXM2-16GB' (compute capability 7.0, sm_70).
  - the OptiX programs are compiled for compute_75, newer than sm_70 (set RAYD_DISABLE_OPTIX=1 to run only the CUDA paths)
This build supports sm_75 and newer.
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

- CUDA Toolkit 12.x (12.8 is what release CI uses; Volta needs 12.x, not 13.x)
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
the Torch backend uses CMake's `native`, as before.

The value drives every compiled target: the Dr.Jit explicit `nvcc` commands,
the Torch CMake CUDA targets, Caffe2's `TORCH_CUDA_ARCH_LIST`, and the OptiX
PTX architecture. The OptiX PTX keeps its release architecture unless a
requested target is older, then follows the oldest target. In practice this
only lowers the Torch PTX from `compute_75` to `compute_70` for Volta; the
committed Dr.Jit PTX is already `compute_70`. Release builds never set
`RAYD_CUDA_ARCHS` and keep their explicit `RAYD_CUDA_GENCODE_ARCHES`,
`RAYD_CUDA_PTX_ARCH`, `CMAKE_CUDA_ARCHITECTURES`, and `TORCH_CUDA_ARCH_LIST`
lists; setting both forms at once fails at configure time.

### Verifying a source build

```bash
python -c "import json, pathlib, rayd.drjit._C as c; print((pathlib.Path(c.__file__).parent / '_build_config.json').read_text())"
python -c "import json, pathlib, rayd.torch._C as c; print((pathlib.Path(c.__file__).parent / '_build_config.json').read_text())"
```

The printed record lists the targets the runtime check accepts. To inspect
the binary itself, run `cuobjdump --list-elf` on the extension (`_C`) or on
`_legacy_ops`/`_stable_ops`; release CI does the same through
`drjit/scripts/verify_cuda_binary_arches.py`.
