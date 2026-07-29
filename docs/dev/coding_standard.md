# RayD coding standard

This document defines the maintained source layout, formatting, comment, math ownership, and duplication rules.

## Formatting

- Native C, C++, CUDA, and header files use `.clang-format`.
- Python and stub files use Ruff's formatter settings in the root `pyproject.toml`.
- Maintained source uses four-space indentation, LF endings, and a 120-column limit.
- Keep a declaration or call on one line when it fits. When it does not fit, pack as many complete parameters as fit on
  each continuation line. Do not force one parameter per line.
- If a long parameter list represents one coherent request, prefer a named request or parameter record. Flat CUDA or
  OptiX launch boundaries may remain flat when the layout is part of the kernel contract.
- Run `python scripts/format_code.py` after editing source. Run `python scripts/format_code.py --check` in verification.
- Generated files are not formatted by hand. Change their inputs and regenerate them.

## Source layout and ownership

- `include/rayd/` is public API only. Its default Torch and shared headers are flat at the root, and its only
  subdirectory is the flat Dr.Jit surface at `include/rayd/jit/`.
- Private native headers live beside their owners in `src/`; public headers must not include anything from `src/`.
- A private header used by only one production file is part of that file unless a native compile test independently
  validates it as a host/device boundary. Otherwise, fold it into the consumer instead of leaving a fragment header.
- Keep a direct `src/<concept>/` directory only for a real multi-file module. One- and two-file concepts stay flat under
  `src/`; do not recreate backend directory trees.
- Native implementation files use backend suffixes for variants. Shared implementation files use a responsibility name
  or `_shared` suffix only when both backends compile the same physical source.
- There is no maximum file length. Split a file only when the pieces have distinct ownership or multiple consumers, not
  to satisfy a line-count target.
- Renames and consolidation are direct. Do not retain forwarding headers, compatibility include paths, or duplicate
  declarations.
- Consolidated native files use current responsibility comments, never `merged from` markers, and must not retain
  adjacent duplicate includes.
- Python frontends live under `python/rayd/torch/` and `python/rayd/drjit/`. Shared private implementation belongs under
  `python/rayd/_impl/` only when both independently installable distributions can package it without conflicting file
  ownership.

## Math ownership

- `include/rayd/math.h` is the only production file whose name contains `math`.
- `include/rayd/utd.h` uniquely owns shared UTD domain types and algorithms; diffraction launch and backend headers are private under `src/diffraction/`.
- Put reusable vector, complex, matrix, quaternion, dual-scalar, CUDA `float3`, and primitive scalar/vector operations
  in `math.h`.
- Concept files may own domain records and algorithms, but must not redeclare simple math types or copy primitive math
  operations. Use an alias when a domain-specific name improves readability.

## Comments

- Every maintained source starts with `Copyright Xingyu Chen.` and one plain-English sentence stating its responsibility.
- File responsibility sentences are concise, end with a period, and do not cite planning phases or ADR identifiers.
- Internal comments explain intent, invariants, numerical constraints, or non-obvious ownership. Do not narrate syntax.
- Keep historical discussion, benchmark stories, migration plans, and decision records in `docs/`, not in source comments.
- Prefer a short comment beside the relevant invariant over a large introductory essay.

## Duplication

- Reuse common production algorithms and test helpers instead of copying them.
- Shared test subprocess, geometry, hashing, and source-inspection helpers belong under `tests/support/`.
- Text similarity alone is not sufficient reason to merge Torch and Dr.Jit adapters. Keep separate adapters when their
  allocation, stream, ABI, packaging, or error ownership differs, and share the backend-neutral algorithm beneath them.
- The two private path-exchange modules are intentionally byte-identical because the independently installable wheels
  require disjoint private file ownership. New exact whole-file duplicates are not allowed.
- `tests/test_source_file_standards.py` enforces layout, math ownership, file headers, formatting configuration, and the
  closed set of intentional exact whole-file duplicates.
