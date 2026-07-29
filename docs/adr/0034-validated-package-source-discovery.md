# ADR-0034: Validated package source discovery

- Status: Accepted; internal source-bundle layout and ABI identity schema amended by ADR-0039, ADR-0040, and ADR-0041
- Date: 2026-07-22
- Decision ID: `validated-package-source-discovery`
- Scope: RayD Torch packaging and same-graph native consumers

> **ADR-0039/0040/0041 amendment.** The passive resource remains
> `rayd/torch/_source`, but its internal canonical source tree is now
> `torch/CMakeLists.txt`, `include/`, `src/`, and `cmake/`. All discovery,
> manifest completeness, validation, relocatability, and same-graph requirements
> below remain in force. Schema 2 records the exact eight-header integration API
> set, normalized per-header SHA-256 values, and its aggregate digest. The
> backend-neutral `path_exchange.h` contract is bundled separately; the older
> single integration-header hash description below is historical.
## Context

Native consumers currently require an explicitly located RayD Git checkout in
order to compile `rayd_torch_native_core` in their own CMake, LibTorch, CUDA,
and architecture graph. Searching a conda prefix or a global CMake registry is
ambiguous and can select a stale installation. Exporting the current static
target would instead bind consumers to the RayD wheel's compiler, Torch, CUDA,
and SM choices.

## Decision

The `rayd-torch` wheel carries a relocatable source bundle at the fixed passive
resource `rayd/torch/_source`. It contains only `LICENSE`,
`backends/torch/{CMakeLists.txt,include,src}`, and `shared/{include,src}`.
`rayd-source.json` records schema, distribution version, repository, commit,
dirty state, stable integration identity/API/header hash, and a relative source
root. `source-files.json` lists every bundled file and SHA-256; its own SHA-256
is recorded in the metadata.

Consumers locate the resource through the active Python distribution metadata,
without importing `rayd.torch` or executing package code. They must pin and
validate repository, commit, distribution version, integration ABI, manifest
digest, every listed file, and the absence of extra source files before adding
the source directory to their build. Missing, duplicate, dirty, malformed,
escaped, or mutated packages fail loudly. An explicit source checkout has
higher priority and retains Git commit, remote, dirty, and ABI validation.

The source bundle remains compiled in the consumer's graph. This decision does
not install or export `rayd_torch_native_core`, add a second Python extension,
select a runtime backend, or authorize prefix/CMake-registry scanning.

## Consequences

- pip and conda packages can provide a deterministic source dependency without
  a machine-specific checkout path.
- The wheel grows by the compressed form of roughly 2.4 MB of source.
- Package builds outside Git must provide explicit commit and repository
  metadata; downstream integrity still comes from a separately pinned full
  manifest digest, not those self-reported strings alone.
- Multi-architecture release compilation remains a consumer/CI concern rather
  than a property of this source bundle.
