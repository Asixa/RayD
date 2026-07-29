# ADR-0040: Concept-owned backend headers and centralized Python frontends

- Status: Accepted; installed-header and source-header-set clauses superseded by ADR-0041
- Date: 2026-07-28
- Decision ID: `internal-header-and-python-frontend-layout`
- Scope: RayD installed source headers, private Torch headers, Python frontend source ownership, and downstream source-bundle activation

## Context

ADR-0039 made production implementations concept-major, but deliberately left two repository-shape compromises in place. Forty Torch implementation headers using `rayd::torch_backend` remained under the public `include/rayd/torch` tree, and the backend frontend sources remained nested under `drjit/python/rayd/drjit` and `torch/python/rayd/torch`. The shared header tree also retained the artifact-category umbrellas `rayd/shared/multipath` and `rayd/shared/optix`, so reflection, diffraction, visibility, scene, and edge contracts were still discovered by mechanism rather than concept.

Those paths are not all equivalent. The Torch typed boundary and its public dependencies are source interfaces. The shared and Dr.Jit multipath headers are installed in the Dr.Jit distribution and are therefore a source API even when RayD's own consumers treat them as implementation details. Moving them is a hard source-path break and cannot be described as a private mechanical rename.

## Decision

The public Torch source interface is concept-major:

- `rayd/integration/torch.h`;
- `rayd/{scene,reflection,diffraction,visibility,penetration,scattering,transmission}/torch.h`;
- `rayd/field_transport/torch_ad.cuh`.

The integration identity remains exactly `rayd.torch.integration`, while `kIntegrationApiVersion` becomes `7` because every public include spelling changes. `include/rayd/torch` is removed completely. No forwarding header, include alias, or compatibility include root preserves the old spellings.

Headers whose declarations live in `rayd::torch_backend` are private build inputs and move beside their concept implementations under `src/<concept>`. Backend targets and OptiX custom commands receive the repository source root only as a private include root. These headers are present in the passive Torch source bundle but are not installed or advertised as a C++ SDK.

The physical directories `include/rayd/shared/multipath` and `include/rayd/shared/optix` are retired. Their files move to the owning reflection, diffraction, visibility, scene, edge, path-exchange, or runtime directory. Mixed scene/edge OptiX contracts are split so that the common SBT record contract has one runtime owner and each payload family has one concept owner. Existing C++ namespaces are retained in this decision: physical ownership changes without a simultaneous symbol or numerical rewrite.

This is also an intentional hard break for the moved installed Dr.Jit shared-header and multipath spellings. `include/rayd/multipath` is removed. Its six public headers move to concept-owned `rayd/{reflection,diffraction,visibility}/drjit` paths, while its ten private headers move beside their implementations under `src/<concept>`. Dr.Jit installs an explicit public header manifest instead of recursively installing `include/rayd`. No forwarding headers, namespace aliases, duplicate headers, or compatibility include roots are added.

The same backend qualification applies to every installed Dr.Jit header, not only multipath. `include/rayd` contains concept directories rather than loose headers. A concept facade is `rayd/<concept>/drjit.h`; subordinate headers are `rayd/<concept>/drjit/<part>.h`. Core, diagnostics, math, ray, runtime, scene, edge, surfel, and trace headers follow this rule, so paths such as `rayd/rayd.h`, `rayd/scene/scene.h`, and `rayd/edge/edge.h` are removed without forwarding copies.

The two public Python frontend source trees are centralized under the existing repository Python root:

```text
python/rayd/drjit
python/rayd/torch
python/rayd/_impl
```

Each backend wheel uses an exact nested package mapping for only its public subtree. The disjoint `_impl` member lists continue to be installed explicitly by the corresponding backend CMake project. Neither wheel may recursively collect the whole `python/rayd` tree, and neither `rayd` nor `rayd/_impl` gains an `__init__.py`.

Torch executable distributed examples live under `examples/torch/distributed`. Backend-specific API documentation lives under `docs/drjit` or `docs/torch`; repository-wide CI documentation lives under `docs/dev`. Shared benchmark data remains shared and is not duplicated merely for visual symmetry.

## Source-bundle identity

The passive Torch bundle remains at the Python resource path `rayd/torch/_source`; that package-resource name is not a C++ include layout. `rayd-source.json` schema version `2` identifies the typed boundary as a `source-header-set-sha256`: it records the exact nine public header paths, their line-ending-normalized SHA-256 values, and an aggregate digest over the ordered `(path, digest)` pairs. Downstreams must validate the complete set, not only the integration entrypoint.

## Preserved contracts

This decision does not change:

- the exact integration identity `rayd.torch.integration`;
- public Torch declarations and namespaces, apart from the include-path and API-version hard break;
- runtime symbols, dispatcher schemas, launch topology, stream behavior, numerical order, derivative support, or failure behavior;
- any CUDA numeric compile profile or ADR-0035 divergence;
- the `rayd`, `rayd-drjit`, and `rayd-torch` distribution names or the `rayd.drjit` and `rayd.torch` import paths;
- committed-PTX correctness or source-identity requirements.

The eight affected Dr.Jit PTX modules must be genuinely regenerated and byte-compared before their source records are marked verified. The Torch source-bundle file manifest and complete public-header-set digest must be recomputed from the final tree.

## Packaging and activation

Wheel validation uses real artifacts and proves that the two distributions own disjoint `rayd` files, including their explicit `_impl` members. Single-backend editable installs must not expose the other backend; dual editable installs and uninstall in either order must preserve the remaining namespace portion.

Downstream activation is atomic. A consumer of the passive Torch source bundle must update its RayD commit, source manifest digest, include roots, stable ABI locator, and any directly included moved source header in one reviewed change. Rollback changes the pin to a prior complete owner; it does not add alternate include roots or runtime owner selection.

## Superseded and amended records

ADR-0039 remains the governing concept-axis decision, except:

- its frozen installed-Dr.Jit-include clause is superseded for the shared headers explicitly moved by this record;
- its backend-local Python frontend source paths are replaced by the centralized paths above.

ADR-0028's durable identity remains in force, but its public include spelling and API-version clauses are superseded by this record. ADR-0034 and ADR-0035 otherwise remain in force.

## Consequences

Public source interfaces, private build headers, and concept-owned shared algorithms become visibly distinct. The repository loses two mechanism-first shared umbrellas and two repeated backend-local Python source roots. The cost is a deliberate source-path break for installed shared headers, new wheel and editable-install gates, eight PTX regenerations, and an atomic downstream source-manifest update.