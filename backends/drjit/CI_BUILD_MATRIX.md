# CUDA CI Build Matrix

This document records the release-wheel compatibility contract shared by RayD and the native CUDA packages in `witwin-platform`. It is the source of truth for CI matrix changes.

## Shared release baseline

| Dimension | Release coverage |
| --- | --- |
| Host platforms | Linux x86-64 and Windows x86-64 |
| GitHub runners | `ubuntu-22.04` and `windows-2022` |
| Python | CPython 3.10, 3.11, 3.12, 3.13, and 3.14 |
| CUDA build toolkit | CUDA 12.8 Update 1 on Windows; CUDA 12.8 latest update from the RHEL 8 repository in manylinux |
| Native SASS | `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_120` |
| Forward-compatible PTX | `compute_120` |
| Oldest explicitly covered consumer GPU | Turing / RTX 2080 class (`sm_75`) |
| Newest explicitly covered families | Data-center and GeForce/RTX PRO Blackwell (`sm_100`, `sm_101`, `sm_120`) |

The `sm_70` image is retained for Volta-class accelerators even though the named consumer baseline is `sm_75`.

## Package-specific ABI matrix

The projects share platform and GPU coverage, but they do not share one Python/native ABI model.

| Package | Python wheel model | Framework matrix | Release build shape |
| --- | --- | --- | --- |
| `witwin-maxwell` | one `py3-none-<platform>` wheel per OS | LibTorch Stable ABI baseline 2.10; load-tested with PyTorch 2.10, 2.11, and 2.12 | 2 native wheel builds plus 14 OS/Python/PyTorch compatibility jobs |
| `witwin-radar` | one `py3-none-<platform>` wheel per OS | LibTorch Stable ABI baseline 2.10; load-tested with PyTorch 2.10, 2.11, and 2.12 | 2 native wheel builds plus 14 OS/Python/PyTorch compatibility jobs |
| `witwin-core` | CPython-specific wheels | PyTorch 2.10/cu128 and 2.11/cu128 native variants | 2 OS x 5 Python versions |
| `witwin-channel` | CPython-specific wheels | Dr.Jit native; PyTorch not applicable | 2 OS x 5 Python versions |
| `rayd` | CPython-specific nanobind wheels | `drjit==1.3.1`; PyTorch not applicable | 2 OS x 5 Python versions |

RayD must not add PyTorch merely to make the matrix look identical. Its public ABI is nanobind plus Dr.Jit, so building each CPython version is the correct compatibility model. Dr.Jit 1.3.1 publishes CPython 3.14 wheels for both Windows x86-64 and manylinux x86-64.

## RayD GitHub Actions matrix

The workflow is `.github/workflows/pypi.yml`.

| Job | Matrix | Purpose |
| --- | --- | --- |
| `unit_tests` | Python 3.10-3.14 on Ubuntu | Validate repository metadata and the release configuration on every supported interpreter |
| `build_linux_wheels` | CPython 3.10-3.14 through cibuildwheel | Build and repair `manylinux_2_28_x86_64` wheels inside the CUDA-enabled manylinux image |
| `build_windows_wheels` | Python 3.10-3.14 on `windows-2022` | Build one `win_amd64` nanobind wheel per interpreter |
| `build_sdist` | Python 3.12 on Ubuntu | Build the source distribution |
| `publish` | release events only | Publish all wheels and the sdist using PyPI trusted publishing |

RayD keeps `manylinux_2_28` rather than changing to the witwin `manylinux_2_35` tag. This is a stricter backward-compatibility target and matches the available Dr.Jit Linux wheels.

## RayD CUDA configuration

RayD compiles four CUDA translation units through explicit `nvcc` custom commands. `CMAKE_CUDA_ARCHITECTURES` does not affect those commands, so release CI supplies two dedicated environment variables:

```text
RAYD_CUDA_GENCODE_ARCHES=70,75,80,86,89,90,100,101,120
RAYD_CUDA_PTX_ARCH=120
```

`CMakeLists.txt` translates them to:

```text
-gencode=arch=compute_70,code=sm_70
-gencode=arch=compute_75,code=sm_75
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_86,code=sm_86
-gencode=arch=compute_89,code=sm_89
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_100,code=sm_100
-gencode=arch=compute_101,code=sm_101
-gencode=arch=compute_120,code=sm_120
-gencode=arch=compute_120,code=compute_120
```

Local source builds may omit these variables and use the local toolkit's default architecture. Release wheels must always set both variables.

The committed OptiX program headers contain PTX and are a separate compatibility layer. The fatbin matrix above applies to the CUDA objects linked into the nanobind extension.

## Artifact validation

Every release wheel is checked by `scripts/verify_cuda_binary_arches.py`. The script extracts the RayD extension, runs `cuobjdump --list-elf` to require every SASS image, and runs `cuobjdump --dump-ptx` to require `.target sm_120`. A missing architecture fails the build.

This distinction matters: a matrix entry or an `nvcc` flag in YAML is an intention, while `cuobjdump` verifies the artifact users actually install.

GitHub-hosted runners do not provide the full physical GPU matrix. CI therefore proves compilation and binary contents, not execution on every listed GPU. GPU runtime regressions should additionally be run on available Turing and Blackwell hardware before a high-risk native release.

## Driver and runtime boundary

- CUDA SASS is architecture-specific; PTX provides forward JIT compatibility on later architectures supported by the installed driver.
- CUDA 12.8 is required to compile native Blackwell targets including `sm_100`, `sm_101`, and `sm_120`.
- End users need an NVIDIA driver new enough for the CUDA 12.x runtime and OptiX version used by Dr.Jit/RayD.
- The CUDA toolkit is a build dependency. A toolkit is not required merely to install a prebuilt wheel.

References:

- [CUDA 12.8 nvcc architecture targets](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html)
- [CUDA binary and PTX compatibility](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#application-compatibility)
- [CUDA 12.8 Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-compatibility-guide/contents.html)
- [Dr.Jit 1.3.1 release files](https://pypi.org/project/drjit/1.3.1/)

## Maintenance checklist

When changing CUDA, Python, Dr.Jit, PyTorch, or runner versions:

1. Update this document and the relevant workflow in the same commit.
2. Keep the OS and Python ranges aligned unless a package's ABI model makes an axis inapplicable.
3. Confirm the selected toolkit accepts every requested `sm_*` target.
4. Verify the built wheel with `cuobjdump`; do not rely only on configuration text.
5. Keep a PTX target for forward compatibility.
6. Build an sdist and inspect wheel metadata (`Requires-Python` and framework dependencies).
7. Record any deliberate exception, such as RayD's lower manylinux baseline or framework-specific ABI matrix.
