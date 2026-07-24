# Changelog

All notable changes to RayD are documented in this file.

## [0.7.0] - 2026-07-23

### Added

- Added an automatic pure-CUDA ray-tracing fallback, a shared traverser
  architecture, and CUDA-fused visibility, reflection, diffraction, and
  multipath execution paths.
- Added typed RF layer-stack, transmission, scattering, penetration, and
  diffraction path families, including batched segment penetration and
  source-lane diffraction layouts.
- Added a validated downstream source bundle and a LibTorch Stable ABI slice
  tested across Torch 2.10 through 2.13.

### Changed

- Unified hosted GitHub Actions wheels across Python 3.10 through 3.14 on
  Windows and real manylinux_2_28, using CUDA 12.8.
- Prebuilt CUDA binaries now contain native SASS for sm_70, sm_75, sm_80,
  sm_86, sm_87, sm_89, sm_90, sm_100, sm_101, and sm_120, plus compute_120
  PTX.
- Grouped compatible CUDA front ends on Windows and added focused release
  validation, reducing the measured cold Torch wheel build from 72.27 to
  40.88 minutes.
- Stabilized the typed Torch integration API and retired the legacy integration
  boundary.

### Fixed

- Fixed finite-edge UTD continuation and gating near stationary-point and
  boundary transitions.
- Fixed CUDA eager-query literal materialization races and hardened BVH memory,
  launch, and treelet-build behavior.
- Fixed reflection EPC, diffraction, and geometry-adjoint consistency across
  the Dr.Jit and Torch backends.

## [0.6.0] - 2026-07-10

### Added

- Added explicit `rayd.drjit` and `rayd.torch` backend namespaces.
- Added independently installable `rayd-drjit` and `rayd-torch` distributions,
  plus the `rayd` meta-distribution that pins both backends to the same version.
- Added a Torch-native C++/CUDA/OptiX backend with dispatcher, autograd, forward
  AD, `torch.compile`, geometry, visibility, reflection, diffraction, and EPC
  operations.
- Added shared native integration headers for downstream source-linked CMake
  builds.

### Changed

- The parent `rayd` namespace is now a PEP 420 namespace and no longer selects
  or re-exports a default backend. Applications must import `rayd.drjit` or
  `rayd.torch` explicitly.
- Dr.Jit and Torch backends now share aligned geometry and multipath contracts
  while retaining backend-native arrays, ownership, streams, and AD graphs.

### Fixed

- Preserved finite-edge coherent UTD endpoint continuation instead of
  introducing a zero-field hard cutoff when the stationary point lies outside
  the finite edge segment.
- Hardened OptiX pipeline creation, scene cache updates, reflection and
  diffraction accumulation, EPC bindings, and native AD propagation across
  both backends.

[0.7.0]: https://github.com/Asixa/RayD/releases/tag/v0.7.0
[0.6.0]: https://github.com/Asixa/RayD/releases/tag/v0.6.0
