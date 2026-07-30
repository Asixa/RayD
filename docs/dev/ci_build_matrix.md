# CUDA CI Build Matrix

This document records the release-wheel compatibility contract shared by RayD and the native CUDA packages in `witwin-platform`. It is the source of truth for CI matrix changes.

## Shared release baseline

| Dimension | Release coverage |
| --- | --- |
| Host platforms | Linux x86-64 and Windows x86-64 |
| GitHub runners | `ubuntu-22.04` and `windows-2022` for build/validate jobs; `ubuntu-latest` for the publish jobs |
| Python | CPython 3.10, 3.11, 3.12, 3.13, and 3.14 |
| CUDA build toolkit | CUDA 12.8 Update 1 on Windows; CUDA 12.8 latest update from the RHEL 8 repository in manylinux |
| Native SASS | `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_120` |
| Forward-compatible PTX | `compute_120` |
| Oldest explicitly covered consumer GPU | Turing / RTX 2080 class (`sm_75`) |
| Newest explicitly covered families | Data-center and GeForce/RTX PRO Blackwell (`sm_100`, `sm_101`, `sm_120`) |

The `sm_70` image is retained for Volta-class accelerators even though the named consumer baseline is `sm_75`.

## Package-specific ABI matrix

The projects share platform and GPU coverage, but they do not share one Python/native ABI model.

| Package | Python wheel model | Framework matrix | Release build shape |
| --- | --- | --- | --- |
| `rayd-drjit` | CPython-specific nanobind wheels | `drjit==1.3.1`; PyTorch not applicable | 2 OS x 5 Python versions |
| `rayd-torch` | CPython-specific wheels while `_C` remains | PyTorch 2.10/cu128; `_stable_ops` uses LibTorch Stable ABI but the transitional `_C` module does not | 2 OS x 5 Python versions |
| `rayd` | pure Python `py3-none-any` meta wheel plus sdist | pins both backend distributions to the same RayD version | 1 build |

The Dr.Jit backend must not add PyTorch merely to make the matrix look identical. The Torch backend cannot collapse to one wheel per platform until the CPython-bound `_C` module has been removed; only `_stable_ops` currently has the Stable ABI boundary. Dr.Jit 1.3.1 and PyTorch 2.10/cu128 both publish CPython 3.14 wheels for Windows x86-64 and manylinux x86-64.

## RayD GitHub Actions matrix

Pull requests use `.github/workflows/ci.yml`: GitHub-hosted Linux and Windows
runners build representative CPython 3.12 wheels with native `sm_87` and
`sm_120` SASS plus `compute_120` PTX. This fast profile proves both native
backends compile and package without paying the full release fan-out on every
commit.

The complete distribution workflow is `.github/workflows/pypi.yml`. It runs
by manual dispatch and for published releases. Every run builds and audits the
complete release artifact set; only a published GitHub Release enables trusted
publishing. The existing `rayd` project uses the `pypi` GitHub Environment;
the two backend projects use the unique `pypi-rayd-drjit` and
`pypi-rayd-torch` environments required by PyPI.

| Job | Matrix | Purpose |
| --- | --- | --- |
| `metadata` | Python 3.10-3.14 on Ubuntu | Validate all three distributions and the release configuration on every supported interpreter |
| `build-drjit-linux` | Five parallel CPython 3.10-3.14 cibuildwheel jobs | Build and repair five `rayd-drjit` `manylinux_2_28_x86_64` wheels inside CUDA-enabled manylinux images |
| `build-torch-linux` | Five parallel CPython 3.10-3.14 cibuildwheel jobs | Build and repair five full `rayd-torch` wheels; audit `_legacy_ops`, `_stable_ops`, external framework dependencies, and CUDA images |
| `build-windows-wheels` | 2 backends x Python 3.10-3.14 on `windows-2022` | Build and audit ten `win_amd64` wheels |
| `build-meta` | Python 3.12 on Ubuntu | Build and check the pure Python `rayd` wheel and sdist |
| `validate-wheel-set` | Ubuntu, after all four build jobs | Validate the complete release artifact set via `tests.packaging.test_release_artifact_matrix`; gates every publish job |
| `publish-*` | published GitHub Releases only, on `ubuntu-latest` | Publish backend wheels first, then the meta distribution, using PyPI trusted publishing |

Both native backends keep `manylinux_2_28` rather than changing to the witwin `manylinux_2_35` tag. This is a stricter backward-compatibility target and matches the Dr.Jit and PyTorch 2.10/cu128 Linux wheels used by the builds.

Every Linux matrix entry uses cibuildwheel's isolated test environment to
load the installed public backend and native extension, then uninstalls the
distribution and confirms that the backend import is absent. Every Windows
matrix entry performs the same lifecycle from the runner's temporary
directory. Both routes launch the probe with Python isolated mode, require the
loaded public module and native library to reside under the active
environment's `site-packages`, and print their resolved paths. The Torch probe
also requires both the legacy dispatcher and Stable ABI operator sets to be
registered. This per-wheel check complements the final merged-wheel install
order and namespace-coexistence matrix.

## Runtime GPU acceptance

GitHub-hosted build runners have no CUDA device. Runtime execution is split
between two self-hosted workflows:

| Workflow | Runner labels | Trigger | Scope |
| --- | --- | --- | --- |
| `.github/workflows/single_gpu.yml` | `self-hosted`, `linux`, `x64`, `cuda`, `single-gpu` | PR label `run-gpu-ci`, weekly schedule, or manual dispatch | Builds both backends from the checkout; requires live Torch and Dr.Jit CUDA plus one successful OptiX hit per backend; runs Torch/Dr.Jit core geometry and multipath, the Dr.Jit cold-pipeline matrix, mixed geometry, SDF, and cross-backend parity |
| `.github/workflows/multi_gpu.yml` | `self-hosted`, `linux`, `x64`, `cuda`, `multi-gpu` | PR label `run-multi-gpu-ci`, weekly schedule, or manual dispatch | Requires at least two visible devices and runs replicated-scene sharding, chunking, resilience, distributed, and benchmark acceptance |

The single-GPU preflight is deliberately outside the skip-guarded unit-test
modules. A missing CUDA device, failed Dr.Jit CUDA evaluation, unavailable
native extension, or failed OptiX traversal terminates the job before a
`skipUnless` guard could turn the run green. It also prints the public module,
native extension, and shared implementation paths actually imported after the
current checkout is built.

## Build concurrency and caching

Standard public GitHub-hosted Linux and Windows runners have four vCPUs.
Dr.Jit uses four build-system jobs because its CUDA sources are separate custom
commands. Torch uses two build-system jobs and `nvcc --threads=2`, which lets
each multi-architecture CUDA compilation use two threads without multiplying
four outer jobs by two inner jobs.

Both workflows install `sccache` 0.11.0. Linux Torch routes C, C++, and CUDA
compiler invocations through CMake launchers. Windows Torch routes only C and
C++ through sccache because version 0.11 loses secondary cubin files from
multi-output grouped gencode calls; its CMake CUDA commands run directly.
Dr.Jit's explicit NVCC custom commands use the generated
`RAYD_NVCC_LAUNCHER` wrapper, while Ninja still schedules four independent
translation units. Linux manylinux jobs access the host-installed
portable sccache binary and persistent cache through cibuildwheel's default
`/host` mount; `/project` is a container copy and must not hold persistent
compiler state. Before the container exits, it resets the cache tree ownership
to match the host workspace so the GitHub cache post-step can archive it.
Compiler caches are capped at 1 GiB per
OS/backend/profile namespace, restore by prefix across commits, and report
statistics at job completion. Python matrix jobs share that namespace rather
than multiplying the repository cache footprint. The Linux cibuildwheel tool
cache and the Windows pip download cache are also persisted.
Windows hosted Dr.Jit jobs use NVIDIA's network installer and install only
`nvcc`, `cudart`, `cuobjdump`, and Visual Studio integration. Torch needs the
larger cuBLAS, cuSPARSE, and cuSOLVER runtime/development packages referenced by
its public CUDA headers; it uses the action's GitHub-cached local installer but
still selects only those packages. This avoids parallel matrix jobs competing
for NVIDIA network-installer downloads while excluding profilers, samples, and
unrelated CUDA libraries from installation.
Linux manylinux jobs likewise install only `cuda-compiler`,
`cuda-cudart-devel`, and `cuda-cuobjdump`, plus those three math development
libraries and the CUDA driver stub development package for Torch, rather than
the 5 GiB full toolkit metapackage. Torch also installs the NVRTC development
package required by its Caffe2 CMake targets.

Pull-request cache keys use the PR head commit rather than the synthetic merge
commit. Push, schedule, release, and manual builds use `github.sha`. Restore
prefixes still allow reuse across commits, while a new head can save updated
compiler results instead of being blocked by an immutable empty cache entry.

Cache misses are always supported. Release correctness does not depend on a
warm cache, and the post-build CUDA binary verifier inspects the produced wheel
regardless of whether compilation was cached.

## RayD CUDA configuration

The Dr.Jit backend compiles CUDA translation units through explicit `nvcc` custom commands. `CMAKE_CUDA_ARCHITECTURES` does not affect those commands, so release CI supplies two dedicated environment variables:

```text
RAYD_CUDA_GENCODE_ARCHES=70,75,80,86,87,89,90,100,101,120
RAYD_CUDA_PTX_ARCH=120
```

`CMakeLists.txt` translates them to:

```text
-gencode=arch=compute_70,code=sm_70
-gencode=arch=compute_75,code=sm_75
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_86,code=sm_86
-gencode=arch=compute_87,code=sm_87
-gencode=arch=compute_89,code=sm_89
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_100,code=sm_100
-gencode=arch=compute_101,code=sm_101
-gencode=arch=compute_120,code=sm_120
-gencode=arch=compute_120,code=compute_120
```

Local source builds may omit these variables; when `nvidia-smi` is available,
CMake detects the first local GPU and compiles one native SASS image. Release
wheels must always set both variables explicitly and never depend on local GPU
detection.

The Torch backend uses the equivalent `CMAKE_CUDA_ARCHITECTURES` list for `_C`, `_stable_ops`, and their linked CUDA objects. Its separately embedded OptiX PTX is a distinct compatibility layer.

## Artifact validation

Every native release wheel is checked by `drjit/scripts/verify_cuda_binary_arches.py`. The script extracts `_C` and, for Torch, `_stable_ops`, runs `cuobjdump --list-elf` to require every SASS image, and runs `cuobjdump --dump-ptx` to require `.target sm_120`. The Torch Stable ABI library is additionally checked by `torch/scripts/verify_stable_abi.py` after wheel repair. A missing architecture or forbidden Python/unstable LibTorch dependency fails the build.

Artifact inspection is followed by an install/import/uninstall lifecycle in
every OS/backend/Python matrix entry. This checks that the repaired or native
wheel can be imported independently of the checkout, that its native runtime
is available, and that uninstall removes its namespace portion cleanly.

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
