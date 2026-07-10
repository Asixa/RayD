# Dual-backend migration validation

Date: 2026-07-09
Environment: `witwin2`, Python 3.11.14, Torch 2.10/CUDA 12.8, CUDA toolkit
12.9, RTX 5080

## Source preservation

- RayD started at `b2336ad`.
- RayDTorch was committed at `3f912c6` and tagged
  `raydn-pre-monorepo-2026-07-09` before import.
- The history-rewritten import commit is `a040f41`; the merge commit is
  `1c8c184`. Earlier RayDTorch history remains reachable below that merge.

## Build and test acceptance

- Torch full suite: 110 passed, 12 skipped.
- Dr.Jit full suite: 194 passed.
- Opt-in Dr.Jit/Torch forward parity: 12 passed.
- Namespace isolation: 6 passed.
- Shared-header ownership guard: 2 passed.
- Dr.Jit project metadata: 13 passed.
- Torch project metadata and public source contract: 22 passed.
- Wheel layout: 2 passed.
- Clean-venv wheel install order and independent uninstall matrix: 2 passed.
- Native extension builds and non-editable wheels completed for both backends.

Accepted wheels:

- `rayd_drjit-0.5.0-cp311-cp311-win_amd64.whl`, SHA-256
  `3aa8c12d01686d2b2b8d43e027178b063f9f37614bc7da0c13f0be80a6726a10`.
- `rayd_torch-0.5.0-cp311-cp311-win_amd64.whl`, SHA-256
  `6d213a336189e3f8ef3e359e115add42e11052e28ddd8740f43ee0725867320e`.

Neither wheel contains `rayd/__init__.py`; their `rayd/**` file sets are
non-overlapping.

## Performance acceptance

Shape: grid 64, 4096 queries, 5 warmups, 30 repeats. Values are milliseconds.

| Mode | Backend | Build | Intersect | Nearest edge | Reflection | Diffraction paths | Direct accumulation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | Dr.Jit | 77.716 | 0.127 | 1.187 | 0.198 | 0.267 | 0.381 |
| Static | Torch | 96.118 | 0.064 | 1.031 | 0.087 | 0.114 | 0.190 |
| Dynamic | Dr.Jit | 117.657 | 0.122 | 1.168 | 0.223 | 0.289 | 0.366 |
| Dynamic | Torch | 93.805 | 0.090 | 1.014 | 0.072 | 0.083 | 0.181 |

The covered steady-state operations meet the existing performance gate. Scene
build remains initialization-sensitive and is reported without a broad
superiority claim.

## Release boundary

The repository contains independent CI and release jobs for `rayd-drjit` and
`rayd-torch`. Publishing to a package index and landing the inventoried
downstream changes are intentionally not performed by local acceptance: the
plan requires both pre-release artifacts to exist in the selected release
channel before downstream main branches are changed.
