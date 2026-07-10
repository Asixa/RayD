# Changelog

All notable changes to RayD are documented in this file.

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

[0.6.0]: https://github.com/Asixa/RayD/releases/tag/v0.6.0
