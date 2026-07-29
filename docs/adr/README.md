# RayD architecture decision records

Every accepted RayD decision record lives in this directory as
`NNNN-<slug>.md`. The sequence is not contiguous: it runs `0001`, `0002`, then
`0025`, `0026`, then `0028`-`0041`. The gaps are deliberate. This file records
why, so the sequence is not "tidied" by a later reader.

## Index

| ADR | Title | Decision ID | Date | Status |
| --- | --- | --- | --- | --- |
| [0001](0001-surfel-remains-drjit-only.md) | Keep surfel support Dr.Jit-only | `F5-surfel-backend-scope` | 2026-07-11 | Accepted |
| [0002](0002-shared-rf-transmission-ownership.md) | Shared RF primitives and transmission ownership | `shared-rf-transmission-ownership` | 2026-07-19 | Accepted; source path and ownership-namespace clauses superseded by ADR-0039 |
| [0025](0025-diffraction-family-ownership.md) | Diffraction operation-family ownership | `diffraction-operation-family-ownership` | 2026-07-19 | Accepted; source path and ownership-namespace clauses superseded by ADR-0039 |
| [0026](0026-generic-scattering-runtime-ownership.md) | Generic scattering runtime ownership | `generic-scattering-runtime-ownership` | 2026-07-19 | Accepted; source path and ownership-namespace clauses superseded by ADR-0039 |
| [0028](0028-stable-typed-integration-naming.md) | Stable typed integration naming | `stable-typed-integration-naming` | 2026-07-20 | Accepted; public include and API-version clauses superseded by ADR-0041 |
| [0029](0029-typed-axial-edge-visibility.md) | Typed axial-edge visibility | `typed-axial-edge-visibility` | 2026-07-20 | Accepted; public include clause superseded by ADR-0041 |
| [0030](0030-typed-capacity-row-validity.md) | Typed capacity-row validity | `typed-capacity-row-validity` | 2026-07-20 | Accepted |
| [0031](0031-required-diffraction-path-validity.md) | Required diffraction path validity | `required-diffraction-path-validity` | 2026-07-20 | Accepted |
| [0032](0032-source-lane-diffraction-path-layout.md) | Source-lane diffraction path layout | `source-lane-diffraction-path-layout` | 2026-07-20 | Accepted |
| [0033](0033-batched-segment-penetration.md) | Batched segment-penetration geometry | `batched-segment-penetration` | 2026-07-21 | Accepted; public include and API-version clauses superseded by ADR-0041 |
| [0034](0034-validated-package-source-discovery.md) | Validated package source discovery | `validated-package-source-discovery` | 2026-07-22 | Accepted; internal source-bundle layout and ABI identity schema amended by ADR-0039, ADR-0040, and ADR-0041 |
| [0035](0035-cuda-compile-flag-policy.md) | Per-translation-unit CUDA numeric compile-flag policy | `cuda-numeric-compile-flag-policy` | 2026-07-24 | Accepted |
| [0036](0036-backend-mirrored-python-modules.md) | Backend-mirrored Python modules | `backend-mirrored-python-modules` | 2026-07-25 | Superseded by ADR-0039 |
| [0037](0037-differentiable-sdf-intersection.md) | Differentiable SDF ray intersection | `differentiable-sdf-intersection` | 2026-07-26 | Accepted; integration include and API-version clauses superseded by ADR-0041 |
| [0038](0038-replicated-multi-device-execution.md) | Replicated multi-device and chunked execution | `replicated-multi-device-execution` | 2026-07-27 | Accepted |
| [0039](0039-concept-axis-layout-and-backend-thinning.md) | Concept-axis source layout and thin backend frontends | `concept-axis-layout-and-backend-thinning` | 2026-07-28 | Accepted; installed-header clauses superseded by ADR-0041 |
| [0040](0040-internal-header-and-python-frontend-layout.md) | Internal-header ownership and centralized Python frontends | `internal-header-and-python-frontend-layout` | 2026-07-28 | Accepted; installed-header and source-header-set clauses superseded by ADR-0041 |
| [0041](0041-flat-default-and-jit-header-layout.md) | Flat default and JIT header layout | `flat-default-and-jit-header-layout` | 2026-07-28 | Accepted |

Every row is copied from the target file's own five-line header block. Status,
date, and decision ID are owned by the ADR, not by this index.

## Numbering

`0001` and `0002` are RayD-local decisions and use RayD's own sequence.

`0025` and `0026` jump forward because they are one half of a cross-repository
decision that is also recorded in the sibling Channel repository under
`channel/docs/dev/standards/`. The two repositories chose to give one
cross-repository ownership decision one number on both sides:

| RayD | Channel | Subject | Date |
| --- | --- | --- | --- |
| `0025-diffraction-family-ownership.md` | `adr-025-diffraction-operation-family-ownership.md` | Diffraction operation-family ownership | 2026-07-19 |
| `0026-generic-scattering-runtime-ownership.md` | `adr-026-rayd-generic-scattering-runtime-ownership.md` | Generic scattering runtime ownership | 2026-07-19 |

The `0025` pair shares its title verbatim; the `0026` pair covers the same
decision under differently-worded titles (RayD: "Generic scattering runtime
ownership", Channel: "RayD ownership of generic scattering runtime operations").
Both pairs share the `2026-07-19` date, and each RayD record's Scope line names
the phase pair of the Channel Native direct-RayD integration that the paired
Channel record governs from the other side (RayD `0025`: phases 7 and 8;
RayD `0026`: phases 9 and 10). Channel's sequence was already at `adr-024` while
RayD's was at `0002`, so matching the number cost RayD a 22-entry jump and cost
Channel nothing.

Neither repository carries a written cross-repository numbering policy. The
convention above is reconstructed from the records themselves and from the
`git log` commands the next section reproduces; treat it as the observed
practice, not as a rule either repository is contractually bound to.

The alignment is specific to those two records and was never retroactive.
RayD `0002` covers the same subject as Channel `adr-024-shared-rf-transmission-ownership.md`
(near-identical titles — Channel's adds "runtime" — same `2026-07-19` date,
phases 5 and 6) but kept its RayD-local number, which is why the alignment
starts at `0025` and not earlier.

From `0028` onward RayD continues its own sequence, resuming from the highest
number then in use. The alignment is not maintained past `0026`: Channel's
`adr-027-batched-segment-penetration.md` corresponds to RayD's
[0033](0033-batched-segment-penetration.md), and Channel's `adr-028` through
`adr-033` are distinct decisions from RayD's records with those numbers. Do not
read a cross-repository pairing into any number other than `0025` and `0026`.

## Unallocated numbers

`0003`-`0024` and `0027` were never allocated in RayD. They correspond to no
decision, no superseded record, and no deletion:

```bash
# every path ever added under docs/adr/ — the ADR files plus this README;
# no 0003-0024 or 0027 appears anywhere in the list
git log --all --pretty=format: --name-only --diff-filter=A -- docs/adr/ | sort -u
# returns nothing: no ADR has ever been removed from this directory
git log --diff-filter=D --name-only --oneline -- docs/adr/
```

Do not reuse them. A number that never carried a decision is still a number a
reader may search for in commit messages, review threads, or the Channel
sequence; reusing it would make that search return the wrong record. New RayD
decisions take the next number after the highest one in the index above.

## Do not renumber

Renumbering an existing ADR is rejected. ADR filenames are load-bearing in
exact-string form:

- `tests/test_adr0026_scattering_ownership.py` opens
  `docs/adr/0026-generic-scattering-runtime-ownership.md` by path and also
  asserts that filename appears as a link target.
- `tests/test_adr0029_axial_edge_visibility.py` opens
  `docs/adr/0029-typed-axial-edge-visibility.md` by path.
- `tests/test_adr0033_segment_penetration.py` asserts
  `docs/adr/0033-batched-segment-penetration.md` exists.
- `CLAUDE.md` and the byte-identical `AGENTS.md` cite ADR filenames directly, as
  do `README.md`, `torch/README.md`, and
  `docs/torch/torch_gap_analysis.md`.
- The Channel repository refers to RayD decisions by number from outside this
  repository, which this repository cannot update. For example
  `channel/docs/dev/audit/phase13-migration-delta.json` records `RayD ADR-024`
  for the shared-RF-transmission decision that RayD numbers `0002` - the exact
  ambiguity the `0025`/`0026` alignment was introduced to avoid.

Renumbering would also destroy the `0025`/`0026` pairing that is the entire
reason the gap exists. If a record becomes wrong, supersede it with a new number
and mark the old one superseded; never renumber it.
