# ADR-0035: Per-translation-unit CUDA numeric compile-flag policy

- Status: Accepted
- Date: 2026-07-24
- Decision ID: `cuda-numeric-compile-flag-policy`
- Scope: nvcc numeric flags for every RayD CUDA translation unit and ray-tracing
  PTX module, in both backends

## Context

`shared/include/rayd/shared/rt/numeric_policy.h` is the backend-neutral numeric
contract. It freezes the epsilon and sentinel divergences between the two
backends behind named legacy profiles and asserts, in the header itself, that
those divergences are still divergent. It says nothing about compiler numeric
flags.

That is a real hole, because the two backends compile the *same* shared
device-math headers under different nvcc numeric flags. A frozen epsilon does
not survive a translation unit that is compiled with `--use_fast_math`: the
constant is unchanged but the divisions, square roots, and transcendentals
around it are not. Before this record, nothing in the repository stated which
translation unit gets which flags, nothing detected a change, and nothing
distinguished a deliberate divergence from an accident.

The concrete evidence that the hole matters is a currently failing test.
`backends/drjit/tests/drjit/test_cuda_multipath.py::test_diffraction_paths_parity`
compares the Dr.Jit OptiX arm against the Dr.Jit CUDA arm of first-order
diffraction path export. On key `fx_re` it reports `optix=-0.002432416193187237`
against `cuda=-0.0016550300642848015`, an absolute difference of `7.77e-4`
against a `5e-5` field tolerance, and a relative difference near 32%. Both arms
run the same `shared/multipath/diffraction_paths_algo.h` over the same
`shared/utd/utd_math.h`. Both are compiled `--use_fast_math`. They differ in
compilation target and finishing pipeline: the OptiX arm is virtual
`compute_70` PTX finished by the driver's just-in-time compiler under an OptiX
module optimization level, the CUDA arm is offline native SASS. That failure is
not repaired by this record and is not caused by it; it is the standing proof
that the compile configuration of shared device math is load-bearing and was
undocumented.

## Decision

The per-translation-unit numeric flag assignment is a contract, recorded in
`shared/contracts/compile_policy.json` and validated against
`shared/contracts/compile_policy.schema.json`.
`tests/test_compile_flag_policy_contract.py` re-derives the assignment from
`backends/drjit/CMakeLists.txt`, `backends/drjit/cmake/rayd_cuda.cmake`, and
`backends/torch/CMakeLists.txt` and fails when the declaration and the build
disagree in either direction.

Compiler flags are deliberately **not** fields of `NumericPolicy`. That struct's
field order is locked by `tests/test_rt_contract_headers.py` and
`tests/test_numeric_policy_contract.py`, it is consumed as a device-side
`constexpr` value, and a compile flag is a property of a translation unit rather
than of a backend profile. The two artifacts point at each other instead.

### Profiles

Four closed numeric profiles cover every translation unit today:

| Profile | Flags | Arithmetic |
| --- | --- | --- |
| `nvcc_default` | none | IEEE division and square root, FMA contraction on, no flush to zero |
| `fast_math` | `--use_fast_math` | approximate division, square root, reciprocal and transcendentals; denormals flushed to zero |
| `no_fmad` | `--fmad=false` | as `nvcc_default` but without multiply-add contraction |
| `precise_no_ftz` | `--ftz=false --prec-div=true --prec-sqrt=true` | identical arithmetic to `nvcc_default` |

`precise_no_ftz` spells out nvcc's own defaults. It exists so the ADR-0033
penetration family cannot inherit a fast-math option added elsewhere, not to
change that family's arithmetic.

### Assignment

- Dr.Jit: 13 object translation units, of which exactly one,
  `backends/drjit/src/trace/cuda_multipath.cu`, is `fast_math`; the other twelve
  are `nvcc_default`. All 8 PTX modules are `fast_math` at `compute_70`, and all
  8 regeneration options default OFF.
- Torch: 46 object translation units, of which one is `fast_math`
  (`rf/diffraction_wedge.cu`), nine are `no_fmad` (the ADR-0026 scattering
  lockstep and AD units), one is `precise_no_ftz`
  (`penetration/segment_penetration.cu`), and the remaining 35 are
  `nvcc_default`. Ten of the 11 PTX modules are `fast_math` at `compute_75`; the
  eleventh, the ADR-0033 penetration module, is `precise_no_ftz`.

All numeric policy is written at one of three places: the `EXTRA_FLAGS` argument
of a `rayd_cuda_object()` call, the shared PTX command shape inside
`rayd_embed_ptx()`, or a `set_source_files_properties(... COMPILE_OPTIONS ...)`
block in the Torch backend. Neither backend sets a global or target-wide CUDA
numeric flag, and the enforcement test fails if one appears.

### Frozen divergences

The contract's `frozen_divergences` array records ten divergences with their
evidence. Each names the units on both sides, and the test asserts each is still
divergent, so silently "fixing" one fails instead of passing.

- `D1` The fused CUDA multipath executor is `fast_math` in Dr.Jit
  (`src/trace/cuda_multipath.cu`) and `nvcc_default` in Torch
  (`src/torch_ext/scene/multipath_cuda.cu`), over the same six shared multipath
  algorithm headers plus the shared UTD math and triangle intersection.
- `D2` Ray-tracing PTX targets `compute_70` in Dr.Jit and `compute_75` in Torch.
- `D3` `shared/utd/utd_math.h` is compiled under three profiles inside the Torch
  backend alone: `fast_math`, `nvcc_default`, and `no_fmad`.
- `D4` `shared/bvh/triangle_intersect.h` is `fast_math` through the Dr.Jit fused
  executor and `nvcc_default` through the shared triangle-query unit.
- `D5` `shared/edge/edge_distance_math.h` is `fast_math` in both backends'
  edge PTX modules and `nvcc_default` in the custom-BVH query units that serve
  the same public query.
- `D6` `shared/field_math.h` is `fast_math` in the reflection-accumulation PTX
  modules and `nvcc_default` in the object units computing the same reduction.
- `D7` Equal profile does not imply equal device code. The Dr.Jit OptiX and CUDA
  multipath arms are both `fast_math` and still diverge; this is the failing
  parity test quoted above. Status `open`, not `frozen`: nobody has decided that
  this divergence is acceptable.
- `D8` `RAYD_TORCH_OPTIX_FAST_MATH` is a user-flippable option that moves ten
  Torch PTX modules between profiles. Dr.Jit has no equivalent switch.
- `D9` The Dr.Jit backend ships committed `*_ptx.h` headers with every
  regeneration option OFF, and `backends/drjit/ptx_sources.json` records
  `regeneration_verified: false`. The declared PTX flags therefore describe how
  those modules are regenerated, not provably how the shipped artifact was
  produced.
- `D10` The ADR-0033 cross-check oracle
  (`tests/cpp/segment_penetration_oracle.cu`) is `nvcc_default` while the family
  it validates is `precise_no_ftz`.

`D1`, `D2`, `D3`, and `D10` are recorded in `uncontracted` as divergences that no
decision record explains. Recording them is not approval of them.

### Shared header exposure

The contract lists the 36 headers under `shared/include/` that are compiled
under more than one numeric profile. That list is recomputed by the test from
the `#include` graph on disk, never asserted by hand, and the include scan
deliberately over-approximates by ignoring preprocessor conditionals: it can
over-report an exposure but cannot miss one. A header that gains or loses a
profile fails the test.

## Consequences

- The declared policy and the build cannot drift apart silently in either
  direction. A new `.cu` file, a new PTX module, a changed `COMPILE_OPTIONS`
  block, or a changed architecture all fail until the contract is updated.
- The contract records flags **as written in CMake**. It proves
  declared-versus-CMake consistency, not declared-versus-effective-command
  consistency: CMake's build-type defaults, CUDA-language handling, and
  generator expressions resolve later. An effective-command proof belongs to the
  per-family activation evidence procedures in ADR-0026 and ADR-0033, not to a
  text test.
- No generated device code changes as a result of this record. It adds a
  contract, a schema, a test, and comments.

## Deliberately not decided here

This record freezes the current assignment and does not align it. Every item
below is a real numerical or performance change and needs its own accepted
record with before-and-after generated-code and numerical evidence.

- `A1` Removing `--use_fast_math` from `backends/drjit/src/trace/cuda_multipath.cu`.
  High impact both ways: it changes SASS for every fused CUDA multipath kernel
  and converts reciprocal, division, square root, and transcendentals in the
  normalize-heavy and UTD-heavy algorithm headers to IEEE forms. Expect a
  measurable throughput loss and changed Dr.Jit multipath numerics.
- `A2` Adding `--use_fast_math` to `src/torch_ext/scene/multipath_cuda.cu`.
  Likely a throughput win, changes every Torch fused-CUDA multipath result, and
  must stay per-source: applying it target-wide would violate ADR-0033, which
  forbids the penetration family from inheriting a global fast-math option.
- `A3` Aligning the PTX architectures. Changes PTX text and SASS for all 19
  ray-tracing modules, and `compute_75` PTX cannot load on `sm_70` hardware.
  High compatibility risk, no numerical benefit.
- `A4` Regenerating the eight committed Dr.Jit `*_ptx.h` headers. Changes shipped
  device code even with identical flags, because the generating toolkit version
  differs from the original. This is why nothing in this change touches a
  `*_ptx.h` file.
- `A5` Removing the `RAYD_TORCH_OPTIX_FAST_MATH` option. No change at the
  default; removes a documented escape hatch.
- `A6` Applying the ADR-0033 precise flags to the penetration oracle. The
  oracle's arithmetic independence is arguably the point of the cross-check, so
  aligning it could mask a real regression.
- `A7` Removing the Windows/POSIX `--extended-lambda` asymmetry in the six Dr.Jit
  edge and BVH units. Not a numeric flag, so it is out of this contract's scope;
  `rayd_cuda_object()`'s `POSIX_NO_EXTENDED_LAMBDA` option documents it. Proving
  it device-code-neutral needs an actual Linux compile.

## Stop conditions

- A numeric profile may not spread across operation families. ADR-0026 forbids
  `rf/scattering.cu` from taking `--fmad=false`; ADR-0033 forbids the
  penetration family from inheriting the global fast-math option. The
  enforcement test checks both as exhaustive family membership.
- A frozen divergence may not be aligned by editing the contract to match a flag
  change. Alignment requires a separate accepted record carrying the numerical
  and generated-code evidence, after which the divergence entry is removed.
- The enforcement test may not be weakened to substring matching, and its
  translation-unit comparison may not be made one-directional. A parser that
  skips an nvcc invocation it does not understand reintroduces exactly the blind
  spot this record closes; the parser raises instead.
- A fifth numeric profile requires a schema change, a profile entry with its
  arithmetic semantics, and a decision record, in that order.
