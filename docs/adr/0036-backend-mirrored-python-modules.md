# ADR-0036: Backend-mirrored Python modules

- Status: Accepted
- Date: 2026-07-25
- Decision ID: `backend-mirrored-python-modules`
- Scope: the pure-Python modules that exist in both `rayd.drjit` and
  `rayd.torch`

## Context

`rayd` is a PEP 420 namespace with no package of its own; the only
distributions are `rayd-drjit`, `rayd-torch`, and a file-free meta wheel that
pins both (`tests/packaging/test_project_metadata.py`). Neither backend
declares the other as a dependency, every `rayd/`-prefixed wheel member must
live under its own backend directory
(`tests/packaging/test_wheel_layout.py::test_backend_files_are_disjoint`), and
uninstalling one backend must leave the other working
(`tests/packaging/test_wheel_install_matrix.py::test_uninstalling_one_backend_preserves_the_other`).
Two modules are duplicated across the backend packages, which a survey of the
Python surface could misread as drift to be deduplicated.

## Decision

`path_exchange.py` is a deliberate byte-identical mirror pair.
`tests/test_f4_path_exchange_contract.py::test_backend_adapters_are_identical_and_conversion_parity_holds`
asserts `read_bytes()` equality between the two copies and `runpy`-executes
each backend's own file, so the mirror is enforced, not assumed. Deduplication
is illegal under the contracts above: a shared `rayd/_shared/` module violates
the wheel-layout ownership rule, a cross-wheel import violates uninstall
independence, and a third `rayd-shared` distribution violates the frozen
distribution set. Any edit lands in both copies in the same change with
identical bytes, including line endings.

`_capabilities.py` diverges on exactly three lines, each required: `_BACKEND`
(`"drjit"` versus `"torch"`), `"surfel"` (`True` versus `False`, per
ADR-0001), and `"torch_compile"` (`False` versus `True`). All three values are
frozen in `shared/contracts/public_api.json`, whose EOL-normalized SHA-256 both
copies pin in the shared `_SCHEMA_SHA256` line, and
`tests/test_public_api_manifest.py::test_runtime_modules_are_validated_copies_of_shared_manifest`
cross-checks every line against that contract. Everything outside those three
lines is identical between the copies.

## Consequences

- The byte-identity assertion is the anti-drift mechanism for
  `path_exchange.py`; a one-sided edit, or a CRLF/LF mismatch between the
  copies, fails the root suite immediately.
- A `_capabilities.py` change outside the three backend-specific lines must be
  applied to both copies; a change to one of the three lines is a public-API
  capability change governed by `shared/contracts/public_api.json` and, for
  `surfel`, ADR-0001.
- Sharing these modules through a common wheel or cross-import would require
  reopening the frozen distribution set and wheel-layout contracts under a new
  decision record; this record decides against it.
