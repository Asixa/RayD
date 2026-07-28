# Committed PTX Source Identity

Date: 2026-07-24

This note records the staleness guard for the eight committed Dr.Jit OptiX PTX
headers, and — just as importantly — what the guard does *not* claim.

## Why committed PTX exists at all

The Dr.Jit backend deliberately builds without an OptiX SDK. Its device programs
are compiled to PTX ahead of time and checked in as C++ string literals under
`drjit/include/rayd/**/*_ptx.h`, so a wheel build needs only nvcc and
CUDA. Regeneration is opt-in through the eight `RAYD_REGENERATE_*_PTX` options,
all `OFF` by default.

The Torch backend is the opposite: `torch/CMakeLists.txt` writes all of
its PTX headers into `${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/`,
regenerates every one of them on each native build, and hard-fails without the
OptiX SDK. Torch therefore cannot go stale, commits no PTX, and is out of scope
for this guard. `tests/test_ptx_source_digest.py` asserts that asymmetry so it
cannot erode.

## The hazard

Because regeneration is off by default, editing `.cu` device code — or any header
it reaches — changes nothing that the build observes. The committed PTX keeps
describing the older device code and the build keeps succeeding. This is not
hypothetical: commit `8d3ba3f` modified
`drjit/src/multipath/diffraction_paths.cu` without touching
`diffraction_paths_ptx.h`, which stayed at its previous content until `d8a064a`.

## The guard

`drjit/ptx_sources.json` records, for each of the eight modules, the
transitive in-repository `#include` closure of its `.cu` file, a SHA-256 over the
contents of that closure, the SHA-256 of the committed header, the exact nvcc PTX
command line, the Dr.Jit pin, and the names of every include that resolves
outside the repository.

`drjit/scripts/audit_ptx_sources.py` generates and re-checks that
record. It follows the same shape as `torch/scripts/audit_abi_boundary.py`
and `torch/abi_audit.json`:

```bash
python drjit/scripts/audit_ptx_sources.py --check      # gate
python drjit/scripts/audit_ptx_sources.py --write      # after a regen
python drjit/scripts/audit_ptx_sources.py --git-drift  # staleness evidence
```

Three consumers:

- `tests/test_ptx_source_digest.py` is the authoritative gate. It recomputes
  everything from source with the standard library alone: no CUDA, no OptiX SDK,
  no GPU, no build.
- `drjit/CMakeLists.txt` replays `--check` at configure time. It warns
  by default and fails under `-DRAYD_STRICT_PTX_SOURCE_CHECK=ON`. It warns rather
  than fails by default because a source build is not required to have the OptiX
  SDK that fixing the warning would need.

  **As of 2026-07-25 nothing in CI enforces this guard.** No workflow sets
  `RAYD_STRICT_PTX_SOURCE_CHECK=ON` and none runs `tests/test_ptx_source_digest.py`
  or `--check` (`.github/workflows/ci.yml` is currently absent; `pypi.yml` and
  `stable-abi-ci.yml` do not invoke the root `tests/` suite). Until someone wires
  one of those in, a stale committed PTX can reach a release with only a
  non-fatal configure warning as its trace. To close the gap: add
  `python drjit/scripts/audit_ptx_sources.py --check` to the release
  workflow's metadata job, and pass `-DRAYD_STRICT_PTX_SOURCE_CHECK=ON` in
  release wheel configures.
- `rayd_embed_ptx()`'s `DEPENDS` lists are checked against the same closure. The
  build graph's rebuild triggers and the digest's input set must be the same set,
  or one of the two is silently missing a file.

## What the record does not claim

**It is source identity, not correctness.** The digests were captured from the
tree as it stood at adoption. Nobody has proven that any committed `*_ptx.h` is a
byte-exact compile of its recorded sources. `--check` answers "did these inputs
change since the record was written", never "is the committed PTX correct".

Two facts make that caveat concrete, and both are recorded in the artifact's
`adoption` block rather than left to memory:

1. Regeneration could not be exercised locally. It fails on an nvcc 12.9 /
   Windows SDK 10.0.26100 ucrt-intrinsics conflict. Verification state lives per
   module (`modules.<name>.regeneration_verified`) and is `false` for every
   module; the `adoption` block itself is an immutable historical record of the
   bootstrap and is never updated by `--write`.
2. Seven of the eight modules already have closure files whose last commit is
   *newer* than the last commit that touched their PTX header, listed per module
   under `sources_committed_after_header`. Only `surfel_trace` has none. Those
   header edits may well be device-code-neutral; the point is that nothing in the
   repository can currently tell you either way.

`tests/test_ptx_source_digest.py::test_adoption_record_does_not_overclaim`
enforces the honesty: every module must carry an explicit boolean
`regeneration_verified`, and the adoption caveat text cannot be quietly dropped.
The flag is set only by `--mark-verified <module>` and is cleared automatically
by `--write` the moment that module's digests (or the drjit pin) change, so an
attestation can never outlive the inputs it was made for.

## Refreshing a committed header

Regeneration never rewrites a committed header. `rayd_embed_ptx()` writes the
regenerated header to `${CMAKE_CURRENT_BINARY_DIR}/rayd/...` and the binary dir
is prepended to the include path, so the fresh header merely *shadows* the
committed one. A regeneration that changes nothing on disk would otherwise
"satisfy" the guard while leaving the stale file in place. The refresh is a
deliberate three-step manual operation:

1. Configure with `-DRAYD_REGENERATE_<MODULE>_PTX=ON` and an OptiX SDK present,
   and build.
2. Copy `<build>/rayd/<header>` over
   `drjit/include/rayd/<header>`.
3. Run `python drjit/scripts/audit_ptx_sources.py --write` and commit
   the header and the record together.
4. If — and only if — step 2 left `git diff` on that header empty, attest the
   byte-equality with
   `python drjit/scripts/audit_ptx_sources.py --mark-verified <module>`.
   That is the claim the bootstrap record could not make.

## Scanner semantics

Includes are collected by regex over raw text, ignoring `#if`/`#ifdef`. The
closure is therefore a superset of the true preprocessed set. That is deliberate:
an over-approximation can only produce a false "stale" report, never miss a real
one, and it keeps the audit free of a preprocessor, and therefore free of the
CUDA/OptiX toolchain the guard exists to work without.

Headers that do not resolve inside the repository — CUDA toolkit, OptiX SDK,
Dr.Jit, standard library — are recorded by name in `external_includes` and never
hashed, so the digest stays machine-independent. A *new* external include still
shows up as drift. The residual gap is a Dr.Jit upgrade that perturbs device code
without changing any tracked file; `drjit_pin` closes it by being *parsed from*
`drjit/pyproject.toml` at audit time (never hardcoded), so a pin bump
changes the rendered record, `--check` fails, and the bumper is forced into a
conscious `--write` plus re-verification.
