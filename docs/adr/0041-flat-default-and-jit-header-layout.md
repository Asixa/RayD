# ADR-0041: Flat default and JIT header layout

- Status: Accepted
- Date: 2026-07-28
- Decision ID: `flat-default-and-jit-header-layout`
- Scope: RayD installed C++ headers, Torch source-bundle headers, Dr.Jit header installation, and committed PTX source identity

## Context

ADR-0040 removed backend container trees but left backend qualification below every concept. The resulting `include/rayd/<concept>/torch.h` and `include/rayd/<concept>/drjit/...` layout created many one-file directories and repeated the backend axis at every concept. It also exposed the Torch-only device implementation `field_transport/torch_ad.cuh` as if it were a public typed API header.

## Decision

`include/rayd` has exactly three ownership surfaces:

- the default Torch typed API and the backend-neutral path exchange contract are flat root headers;
- every installed Dr.Jit header is flat under `rayd/jit`;
- backend-neutral implementation headers live under `rayd/detail`.

The root public headers are:

```text
diffraction.h
integration.h
path_exchange.h
penetration.h
reflection.h
scattering.h
scene.h
transmission.h
visibility.h
```

`rayd/jit` contains no subdirectories. Names that would collide after flattening carry their concept prefix, such as `diffraction_paths.h` and `reflection_trace.h`.

`rayd/detail` retains a subdirectory only for a module with at least two headers. Singleton groups are flattened to `contracts.h`, `field_math.h`, `field_transport.cuh`, `scattering_table.cuh`, and `vec3.h`. Existing `rayd::shared` namespaces remain unchanged: this decision changes physical and include ownership, not numerical code or C++ symbol namespaces.

The Torch-only device derivative header moves to `src/field_transport_ad.cuh`. It remains in the passive source bundle through the `src` tree but is not part of the integration header set.

The durable integration include is `rayd/integration.h`. Its identity remains `rayd.torch.integration` and `kIntegrationApiVersion` becomes `8`. The integration header set contains the eight Torch public headers; `path_exchange.h` is bundled as a separate backend-neutral public contract. No forwarding headers, aliases, duplicate trees, or compatibility include roots preserve the retired paths.

## Packaging and PTX

The Torch source bundle copies an allowlist consisting of its nine root public headers, `rayd/detail`, `src`, `cmake`, and the required build files. It must not contain `rayd/jit`.

The Dr.Jit wheel installs `rayd/jit`, `rayd/detail`, and the neutral `rayd/path_exchange.h`; it must not install the default Torch headers.

All eight committed Dr.Jit PTX modules have include-closure path changes. They must be genuinely regenerated, byte-compared with the committed headers, recorded with `audit_ptx_sources.py --write`, marked verified only after comparison, and pass `--check`.

Each of the eleven Torch raw-NVCC PTX commands emits and registers an NVCC
depfile. Transitive `rayd/detail` changes therefore rebuild the affected PTX;
the handwritten `DEPENDS` lists are not treated as complete include closures.

## Superseded clauses

ADR-0040 remains authoritative for centralized Python frontends and private Torch implementation ownership. Its concept/backend-qualified installed-header layout, nine-header integration set, and API version 7 clauses are superseded by this record.

ADR-0039 remains authoritative for concept-owned source implementations. Its installed-header layout clauses are superseded by this record.

ADR-0028's integration identity remains in force; its include spelling and earlier API-version clauses are superseded.

## Consequences

The public default surface is immediately discoverable, JIT ownership has one unambiguous directory, and internal shared headers no longer look like a third public backend. The include tree has no one-file directories. This is an intentional source-path break with no compatibility layer.
