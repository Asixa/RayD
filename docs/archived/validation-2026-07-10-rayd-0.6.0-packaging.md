# RayD 0.6.0 packaging validation

Date: 2026-07-10
Environment: `witwin2`, Python 3.11, Windows x64, RTX 5080

## Distribution model

- `rayd==0.6.0` is a file-free meta-distribution.
- `rayd` depends exactly on `rayd-drjit==0.6.0` and
  `rayd-torch==0.6.0`.
- `rayd-drjit` and `rayd-torch` remain independently installable.
- No distribution installs `rayd/__init__.py`; the parent remains a PEP 420
  namespace and imports stay explicit.

## Acceptance

- Source metadata/version/release-order tests: 4 passed.
- Namespace isolation and shared-header tests: 8 passed.
- Dr.Jit project metadata tests: 13 passed.
- Torch project metadata and public contract tests: 22 passed.
- Final wheel metadata/layout/default-install/uninstall matrix: 10 passed.
- Installing only `rayd` from the local wheel set resolved and imported both
  `rayd.drjit` and `rayd.torch`.
- Uninstalling the meta-distribution left both backend distributions usable.
- `twine check --strict` passed for all wheels and the meta sdist.
- The meta sdist contains only release metadata, README, license, and build
  configuration; it does not carry backend or test source trees.

This release changes packaging metadata only. Native runtime code is unchanged
from the dual-backend migration acceptance, so the 2026-07-09 numeric and
performance results remain the applicable runtime validation.

## Accepted artifacts

- `rayd-0.6.0-py3-none-any.whl`
  - SHA-256: `a5bf6ef9c00acd368be7e3abf8c250a4e51f3921fd3bc2260900ef6d7a853293`
- `rayd-0.6.0.tar.gz`
  - SHA-256: `915c84b7784c54c07593aded814307a8109cc50494b0d3c8df9ce1d652f7e92d`
- `rayd_drjit-0.6.0-cp311-cp311-win_amd64.whl`
  - SHA-256: `6c2d0bfcab7ded159485db00eb72d0c85ce02acde9d05952726beab031120821`
- `rayd_torch-0.6.0-cp311-cp311-win_amd64.whl`
  - SHA-256: `83931a75a890feebfa2a7fffaf130d0fd170b056f708c7bd1d9847a605242019`
