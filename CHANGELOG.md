# Changelog

All notable changes to RayD are documented in this file.

## [Unreleased]

### Behaviour changes

- **Breaking:** non-finite vertices (NaN or +/-Inf) are rejected with an error
  at `build()` and by every vertex update, in both backends. They used to be
  accepted, and one infinite vertex could make every finite face of its mesh
  unhittable on the OptiX trace backend.
- Monte-Carlo diffraction accumulation sums over its S states instead of
  averaging them, so results with S > 1 states grow by a factor of S: Dr.Jit
  `accum_dfr_direct` now matches Torch, and the chained `accum_dfr`
  (`max_order` 2 and 3) of both backends weights each lane's initial state by
  S, in the primal and in its derivatives. Each further diffraction event
  still draws one recursive state per lane with weight 1.
- `nearest_edge(ray)` follows one degenerate-row rule on every edge backend of
  both frontends: a row is searched only with a finite origin, a finite
  non-zero direction and a positive `tmax` (`+inf` or no `tmax` is the
  half-ray); a NaN, zero or negative `tmax` (`-inf` included), a zero or
  non-finite direction, and a non-finite origin return the miss sentinels.
  The Torch OptiX and CUDA edge backends used to disagree on these rows.
- Torch `Mesh(uv=..., face_uv=...)` texture coordinates are now interpolated
  into `Intersection.uv`; meshes without UVs still report barycentric
  `(u, v)`. `build()` rejects `face_uv` without `uv`, `face_uv` indices outside
  the `uv` rows, and `uv` that requires grad.
- Dr.Jit `edge_bvh_backend="auto"` resolves to `drjit`, a native compact-BVH
  traversal, instead of the OptiX edge backend (see Performance). Exact
  distance ties now go to the lower global edge id, and `build()` takes longer
  (about 60 ms instead of 16 ms at 100k edges). `"optix"` and `"optix_drjit"`
  remain available.
- Dr.Jit `nearest_edge(ray)` now returns the true nearest edge where the old
  traversal ranked edges shorter than about 3e-3 with an absolute 1e-5
  determinant cutoff, and the AD recompute of the reported distance uses the
  same degeneracy test as the traversal (next entry).
- Point, ray and top-k nearest-edge queries (`nearest_edge`,
  `nearest_edges`) on every edge backend of both frontends no longer collapse
  edges shorter than about 3.2e-4 to their start point, which gave errors of
  the order of the edge length. For such edges `edge_point`, `edge_t` and
  `distance` change, and so can the winning edge. A segment is a point only
  when its squared length is at most 1e-18, and two lines with squared lengths
  a and c count as parallel only when |a*c - b*b| <= max(1e-7 * min(a*c, 1),
  1e-18) (a relative sin^2 test for short segments) instead of absolute 1e-7
  bounds on the squared lengths and the Gram determinant
  (`contracts/operations.json` `numeric_policy.shared_multipath.edge_degeneracy_rule`).
  Results on ordinary meshes are bitwise unchanged. A `nearest_edge(ray)`
  direction shorter than about 3.2e-4 is now searched as a ray instead of
  being treated as a point at the origin.
- Transmission: `frequency_hz` must be positive, finite and representable in
  float32 (eval, backward and JVP raise otherwise). A layer-stack material with
  an out-of-range CSR run or non-finite layer data is unusable and gives an
  all-zero row or an invalid path, consistently in the forward, backward and
  JVP; `layer_count < 0` now gives a zero row instead of an identity row. A
  transmission-sequence row whose wall material is out of range or unusable is
  fully inert in the forward too: its `path_length`, `delay` and `direction`
  are zero, as its field and derivatives already were.
- Low-loss transmission is computed with a cancellation-free passive square
  root, so lossy materials keep their loss in float32 (worst ITU-R P.2040 error
  0.07 dB to 0.0002 dB). Lossless inputs are bitwise unchanged; every lossy
  material changes in one component.
- Scattering families validate their inputs: an error is raised unless sample
  CDF shapes match the density table, table and height axes are non-empty, and
  `k0` and frequency are positive, finite float32. Out-of-range device ids,
  table extents, patch rows, realization materials, and per-row chain depths
  outside `[0, 8]` give inert rows (zero outputs and gradients) instead of
  illegal memory accesses or wrong gains.
- Segment penetration rejects requests whose `* 3` element index exceeds
  `int32`, and backward/JVP give zero for rows whose forward failed.
- **Breaking (Dr.Jit):** detached-only bindings refuse grad-enabled Dr.Jit
  arrays instead of silently dropping their derivatives: for example
  `rayd.drjit.Ray(o, d)` with a grad-enabled `o` raises `TypeError` (use
  `RayAD`, `drjit.detach`, or `drjit.suspend_grad()`), and `trace_refl_epc`
  raises when the scene geometry or the receiver carries derivatives.
- The typed Torch integration boundary (`rayd::torch::create_scene`) honours
  `MeshInput.uv` and `MeshInput.face_uv`: `intersect_forward` reports the
  interpolated texture coordinates in `uv` (and its derivatives follow them)
  for meshes that carry UVs, where it used to report barycentric `(u, v)`, and
  scene creation applies the same UV validation as `Scene.build()`. The
  header and `kIntegrationApiVersion` (8) are unchanged; meshes without UVs
  are unaffected.
- Torch scene commits are ordered on the GPU against queries on other CUDA
  streams: `sync()`, `set_edge_mask()` and the release of a scene (for
  example when `build()` replaces it) make their stream wait for the previous
  commit and for the queries already issued on the scene, and a query waits
  for the latest commit while it may still run, so no query or commit reads
  memory another stream is still refitting or has released. Nothing blocks
  the host and unrelated device work is not waited for.
- `RAYD_DISABLE_OPTIX` is read the same way by the native libraries and by
  the Python doctor and device check of both backends: only `1` or `true`, in
  any letter case, turn OptiX off. `yes`, `on` or a value with surrounding
  spaces no longer make the doctor report OptiX as disabled while the backend
  still uses it. The Torch tools also honour the former name
  `RAYD_TORCH_DISABLE_OPTIX`, as the Torch library does.

### Added

- Added `python -m rayd.torch.doctor` and `python -m rayd.drjit.doctor`. They print
  the OS, Python, driver, each GPU's name, compute capability and free memory, the
  installed build's GPU targets and the framework it was compiled against, the
  driver's OptiX library and OptiX ABI versus what the build requests, and the
  trace and edge backends `"auto"` resolves to, with a verdict and fix per check,
  and exit non-zero when a check fails. They also run without CUDA or without
  the Torch native libraries.
- Added `scripts/build_local.sh`, the Linux counterpart of `build_local.cmd`.
- Added the opt-in `RAYD_FETCH_OPTIX_HEADERS` CMake option for the Torch
  backend, which downloads the OptiX SDK 8.1 headers (`NVIDIA/optix-dev`
  `v8.1.0`) when none are found. It is off by default.
- `_build_config.json` now also records the OptiX ABI, the CUDA toolkit, and
  the framework version (and Torch's CUDA version) each build compiled against.
- Added a `Governance` workflow that runs the formatting check and every
  GPU-free governance and packaging test on each push and pull request.

### Changed

- Torch `edge_bvh_backend="auto"` now resolves to the compact CUDA edge BVH
  instead of OptiX, measured 20-80x faster. Edge acceleration structures are built on the first edge query after
  `build()` or a geometry-changing `sync()`, and a dynamic OptiX geometry
  acceleration structure is rebuilt every 16th commit instead of refit
  indefinitely.
- When OptiX is unavailable, `trace_backend="auto"` still falls back to the
  CUDA trace backend but now warns once per process with the reason, in both
  backends (`RAYD_DISABLE_OPTIX=1` stays silent). For Torch the reason is a
  missing driver OptiX library (with the Docker
  `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` hint) or a driver older
  than the build's OptiX ABI. An explicit `trace_backend="optix"` raises the
  same reason.
- A Torch native library that does not load now explains itself: `import
  rayd.torch` warns, and the first native call raises an error naming the
  installed torch and `torch.version.cuda`, the torch the build was compiled
  against, whether the installed torch is CPU-only, the loader error, and the
  `pip install ... --index-url https://download.pytorch.org/whl/cu128` command
  that fixes it. A Dr.Jit extension that does not import names the drjit
  versions the same way. `RAYD_TORCH_LEGACY_LIBRARY` and
  `RAYD_TORCH_STABLE_LIBRARY` now load exactly the named library instead of
  falling back to the installed one.
- Editable installs refuse to import when the `rayd/_impl` copies their build
  made no longer match the checkout, with the rebuild command, instead of
  silently running outdated code. The two backends still install disjoint
  files; `diagnostics.py` joins `device_support.py` as a module both wheels
  ship under their own names.
- Torch configure now fails with a clear message when `RAYD_CUDA_ARCHS` is
  combined with an explicit `CMAKE_CUDA_ARCHITECTURES` or
  `TORCH_CUDA_ARCH_LIST` (as the Dr.Jit backend already did), when the default
  native architecture detection finds no GPU (asking for `RAYD_CUDA_ARCHS`),
  and when the build environment's torch is CPU-only.
- Torch CMake seeds Caffe2's legacy FindCUDA from `find_package(CUDAToolkit)`,
  and the Dr.Jit backend uses `find_package(CUDAToolkit)` (CMake 3.17 or
  newer), so conda CUDA toolkits build without `CUDA_TOOLKIT_ROOT_DIR` or
  `CUDA_NVCC_EXECUTABLE` overrides.
- Dr.Jit symbolic `trace_reflections` accepts at most 65,536 bounces and
  evaluates every 8 bounces; `nearest_edge` and `shadow_test` return empty
  results for empty batches, while `intersect` raises for one.
- The OptiX multipath pipeline retry in both backends now fires only for
  compile, link, and disk-cache OptiX errors, and warns once with the first
  failure. The Torch pipeline cache no longer holds its global lock during a
  compile.
- The README install section now leads with the backend-specific installs and
  the CUDA Torch index URL and explains what `pip install rayd` pulls in; it
  gained a Troubleshooting section.

### Performance

- Dr.Jit nearest-edge queries run the native compact-BVH traversal: 65k point
  queries over 100k edges take about 2 ms instead of 70 ms (2 Dr.Jit kernels and
  one native launch instead of about 460 kernels), and small batches pay far
  fewer host synchronizations.
- The OptiX edge ray query of both backends has no any-hit program and clips a
  half-ray once to the scene edge bounds, about 15-20% faster; a ray that
  misses those bounds returns a miss.
- `visible_edge` stops at the first visible sample (CUDA trace backend, 70%
  visible: 71 to 30 ms).
- Dr.Jit removes host synchronizations from diffraction launches, the CUDA
  trace backend, and dynamic-scene `sync()`, and caches the opaque IAS handle.
- The Torch `sync()` check for non-finite updated vertices costs one flag
  kernel per updated mesh and a single small read per `sync()`; its behaviour
  and API are unchanged.
- Torch visibility queries (`visible`, `visible_pair`, `visible_edge`,
  `visible_chain`) test their inputs for NaN and infinity inside the trace
  instead of in separate tensor passes: 1M segments take about 0.63 ms
  instead of 0.96 ms, with unchanged results.
- The Torch compact CUDA edge BVH stores up to four edges per leaf and the
  nearest-edge outputs are written by one kernel instead of a dozen fills:
  65k point queries over 100k edges take about 2.1 ms instead of 3.8 ms, and
  65k ray queries over 1M edges about 354 ms instead of 636 ms, with unchanged
  results.
- The Torch forward queries `intersect`, `nearest_edge`, `nearest_edges`,
  `visible` and `trace_reflections` run their native host work without
  holding the Python GIL, so threads that drive different devices or streams
  launch concurrently.

### Fixed

- Torch `Scene.build()` rejects negative or out-of-range face indices and
  meshes without vertices; Dr.Jit `Mesh` validates face indices when they are
  set and at build.
- Rays, points, and other query tensors on a different CUDA device than the
  scene are rejected in both backends, including the Dr.Jit EPC, diffraction,
  and `SurfelScene` entry points, which used to fault with an illegal address.
- Faces with a repeated vertex index no longer create edges, and an edge shared
  by more than 256 faces fails the build with a message unless edges are
  disabled for that mesh, in both backends.
- Torch queries with pending updates (`update_mesh_vertices()` without
  `sync()`) now raise, as in Dr.Jit. A backward or forward-mode pass, a lazily
  read result, or a `torch.compile` handle recorded before a geometry-changing
  `sync()` raises instead of differentiating against the new geometry.
- Torch `visible()` returns `False` for non-finite endpoints,
  `trace_reflections()` requires `max_bounces >= 1`, double backward raises a
  clear error, and `update_mesh_vertices()` with an unknown mesh id raises
  `ValueError`.
- Torch `Scene.build()` works from a thread that has not used CUDA, and scenes
  are safe against concurrent mutation through a per-scene reader/writer lock.
- Dr.Jit count products are overflow-checked, and the CUDA trace backend
  rejects batches whose traversal stack would exceed the Dr.Jit array limit.
- The common Torch OptiX launch no longer waits on the host for an earlier
  launch: launch parameters are copied asynchronously on the launch stream.
- Torch forward-mode AD of `intersect()` with an `active` mask (and so of
  `MixedScene` intersect and transmittance) no longer raises.
- Torch `SurfelScene.intersect()` no longer skips a valid surfel less than 1e-3
  behind a rejected one.
- Torch `accum_dfr_coherent_direct` with several states no longer over-weights
  them, and OptiX programs take the warp leader of an aggregated accumulation
  from the hardware lane id, so Monte-Carlo lanes of a partial warp are no
  longer dropped.
- Keller-cone diffraction accumulation gradients match the forward
  contribution in both backends (they were about 2.7x too small).
- Dr.Jit reflected-field derivatives have the right sign at exact normal
  incidence, and Dr.Jit vertex normals are finite on non-manifold fans (for
  grad-enabled arrays in detached overloads see Behaviour changes).
- Scattering ensemble row gradients are zero instead of uninitialized when no
  cotangent is given, and the chain-ensemble JVP stays finite and exact for
  very short and very long chains.
- Near-normal wall-frame normalization no longer biases the transmission path
  gain (relative error 2e-4 to 5e-6 at tilts of 1e-6 to 1e-5).

### Since 0.8.0, before this section

- Added `RAYD_CUDA_ARCHS` source-build target selection for both backends, a
  runtime check that fails early with build-from-source guidance on GPUs the
  installed build cannot run (`UnsupportedDeviceError`), and the `sm_70`
  support floor; compiled the Torch OptiX PTX for `compute_70`.
- Kept Jetson and discrete-GPU cubins apart in the device check and described
  what a build runs on instead of "and newer".
- Kept rays off the surface they leave at any coordinate size (ADR-0045) and
  ended diffraction visibility segments off the wedge on every path.
- Gave Torch multipath OptiX pipelines their own exception flags, rebuilt a
  failed multipath pipeline after releasing idle cached ones and then without
  the OptiX disk cache, and kept the disk cache when pipelines take turns.
- Rejected Torch reflection batches whose output index overflows `int32` and
  EPC `max_bounces` beyond the native chain capacity.
- Regenerated the Dr.Jit PTX so it no longer needs R575.
- Added an opt-in adversarial ray stress suite for both backends, a Building
  from Source / older GPUs README section, and GPU-support documentation that
  separates what a build contains from what was verified.

## [0.8.0] - 2026-07-30

### Added

- Added true mesh instancing with shared geometry acceleration structures,
  per-instance transforms and IDs, scene-global primitive IDs, transform
  derivatives, and transform-only acceleration-structure updates.
- Added Torch high-level reflection accumulation and EPC APIs, a valid
  cross-backend EPC-field numerical parity fixture, replicated multi-GPU
  diffraction path, coherent diffraction, and reflection-accumulation
  execution, and conditional packed SDF-grid batching.

### Changed

- Replaced backend-wide AD claims with operation- and input-domain-level
  VJP/JVP capabilities; Torch EPC-field evaluation is now explicitly
  forward-only until its physical Fresnel and polarization derivatives are
  implemented.
- Expanded single- and multi-GPU CUDA/OptiX CI, Stable ABI checks, wheel
  install/uninstall lifecycle validation, and release-artifact verification.
- Bound the complete `rayd-torch` wheel to its PyTorch 2.10 build minor; the
  separately audited `_stable_ops` slice retains cross-version validation.

### Fixed

- Restored the `rayd.drjit._C` package stub layout and repaired editable-source
  validation so tests cannot silently exercise stale installed frontends.
- Fixed Torch reflection material defaults and EPC-field parity coverage so a
  validated reflected path produces and compares a nonzero physical field.

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

[0.8.0]: https://github.com/Asixa/RayD/releases/tag/v0.8.0
[0.7.0]: https://github.com/Asixa/RayD/releases/tag/v0.7.0
[0.6.0]: https://github.com/Asixa/RayD/releases/tag/v0.6.0
