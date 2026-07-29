# ADR-0039: Concept-axis source layout and thin backend frontends

- Status: Accepted; installed-header clauses superseded by ADR-0041
- Date: 2026-07-28
- Decision ID: `concept-axis-layout-and-backend-thinning`
- Scope: RayD repository layout, source ownership, packaging manifests, and source-level RF ownership names

## Context

The dual-backend repository layout made distribution ownership explicit but split each numerical concept across `backends/drjit`, `backends/torch`, and `shared`. Most implementation, parity, and ownership-transfer work instead follows the domain concept. Exact source paths are also indexed by CUDA compile policy, committed-PTX source identity, the Torch same-graph source bundle, ABI audits, tests, and downstream Channel pins, so changing the axis is a governed migration rather than a mechanical rename.

The implementation plan `docs/dev/plans/concept-axis-layout-and-backend-thinning-plan.md` was approved for execution on 2026-07-28. This record closes its three open decisions.

## Decision

Production implementation is concept-major under root `src/`, `include/`, and the private PEP 420 subtree `python/rayd/_impl`. Unsuffixed implementations are Torch-owned, adjacent `*_jit.*` implementations are Dr.Jit-owned, and `*_shared.*` denotes one backend-neutral numerical owner consumed directly by both backends. Production filenames do not use `_torch` or `_drjit`.

Root `drjit/` and `torch/` are thin distribution/build frontends. They own packaging metadata, backend entry CMake, public Python export and typing layers, backend-specific build tools, and distribution documentation. They do not own complete domain implementation trees. The public distributions and namespaces remain `rayd`, `rayd-drjit`, `rayd-torch`, `rayd.drjit`, and `rayd.torch`; runtime objects do not cross backends.

The Python implementation install path is `rayd/_impl` with no `__init__.py`. The Dr.Jit wheel owns only manifest-listed `*_jit.py` members and the Torch wheel owns only manifest-listed unsuffixed members. Every installed file has exactly one distribution owner and the wheel member sets remain disjoint. The stable direct Python surfaces are the two root packages and the documented `path_exchange` submodules; former implementation submodule paths are not compatibility contracts unless separately documented. Existing installed Dr.Jit C++ include spellings remain unchanged.

The generic `rf/` source umbrella and ownership namespace are retired in the same source-ABI cut:

- passive complex, medium, Fresnel, layer-stack, and transmission sequence code is owned by `rayd/{shared,torch}/transmission` and `rayd::shared::transmission`;
- resident-table, ensemble, patch, and chain scattering code is owned by `rayd/scattering.h`, `rayd/detail/scattering_table.cuh`, `src/scattering`, and the corresponding scattering namespaces;
- UTD and wedge-field code is owned by diffraction, with backend-neutral UTD code under `rayd/detail/diffraction` and `rayd::shared::diffraction`;
- genuinely cross-concept field transport uses `rayd/detail/field_transport.cuh` and `src/field_transport_ad.cuh` owners.

No forwarding `rf/` headers, namespace aliases, copied helpers, runtime owner selection, or compatibility shims are added. Channel must update its direct includes, qualified names, RayD pin, and source-manifest digest atomically before the renamed source ABI is active downstream.

The passive Torch source resource remains exactly `rayd/torch/_source`, but its internal canonical source tree is `torch/CMakeLists.txt`, the allowlisted public/detail headers, `src/`, and `cmake/`, plus its metadata and license. It remains passive, relocatable, fully manifested, and compiled in the consumer graph. ADR-0041 subsequently makes `rayd/integration.h` the flat durable include and advances `kIntegrationApiVersion` to `8`, while preserving identity `rayd.torch.integration`.

CUDA translation-unit consolidation may map several historical logical units to one physical source only when owner, compiler, numeric profile, registration lifetime, ABI, and activation boundary match. `contracts/compile_policy.json` retains all 80 logical TU roles under stable concept names and preserves D1-D10 semantics while recording the final physical source for each role; every logical entry mapped to one physical compile must agree with that compile's profile, target, kind, architecture, and option. No CUDA numeric flag changes are authorized.

The eight Dr.Jit committed PTX modules move only after actual regeneration and byte comparison. Path-only edits to `ptx_sources.json` are not proof. Stable integration identity, exported symbols, launch count, stream behavior, reduction and atomic order, derivative support, failure behavior, and numerical policy remain unchanged.

RayD has no source-file line-count ceiling. Line count is measurement, never an acceptance gate; file boundaries require an ABI, compiler, numeric-profile, PTX, lazy-load, generated-code, or independently owned activation reason.

## Superseded and amended records

- ADR-0036 is superseded. Source-co-located private modules are allowed because file-level single ownership and disjoint wheel manifests, rather than public-subtree location, govern uninstall independence.
- ADR-0034 is amended only for the internal source-bundle tree described above. Its passive discovery, full-manifest, validation, relocatability, and same-graph requirements remain in force.
- ADR-0002, ADR-0025, and ADR-0026 are superseded only where they prescribe `rayd/{shared,torch}/rf/` paths or generic `rf` ownership namespaces. Their numerical ownership, fusion, stream, derivative, failure, activation, rollback, and downstream atomic-switch contracts remain in force.
- ADR-0028 and ADR-0035 remain in force. This decision changes neither the stable integration identity nor any CUDA numeric profile or frozen divergence.

## Historical evidence

Physical paths recorded before ADR-0039 in archived plans, accepted ADR evidence, benchmark baselines, execution logs, and PTX adoption notes describe the historical layout and are not mechanically rewritten to look current. Live manifests, operational commands, current links, and machine-derived path contracts use the canonical layout. When a historical file must remain operationally discoverable, a current locator is added without altering the recorded historical value.

## Consequences

Concept implementations become discoverable together while the two runtimes remain independent. Builds, wheels, source bundles, tests, and downstream pins must use explicit manifests rather than directory-shape assumptions. Layout-only phases cannot hide numerical, launch, ABI, dependency, or fallback changes. Empty `backends/`, `torch_ext/`, shared umbrella, and artifact-category structures are removed only after every live caller and governance record points at the canonical owner.
