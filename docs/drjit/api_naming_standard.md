# RayD API Naming Standard

RayD public APIs use short, unambiguous stems. New public names must follow these rules.

## Domain Stems

- Use `Dfr` for diffraction: `DfrStates`, `DfrGrid`, `DfrMaterial`, `DfrAccum`, `DfrPaths`, `accum_dfr_direct`, `accum_dfr`, `trace_dfr_paths`.
- Use `Refl` for reflection-specific compact names, and keep `Epc` in equivalent-path-correction APIs: `ReflEpc`, `ReflEpcField`, `ReflEpcOptions`, `trace_refl_epc`, `trace_refl_epc_field`.
- Do not use `Diff` for diffraction. In RayD, `diff` reads as differentiation/autodiff and conflicts with `AD` naming.
- Use `AD` only for automatic differentiation variants: `RayAD`, `DfrStatesAD`, `DfrAccumAD`.
- Keep full words where they avoid ambiguity: `Reflection`, `Segment`, `Material`, `Visibility`.

## Function Names

- Prefer verb plus compact domain stem: `accum_dfr_direct`, `accum_dfr`, `trace_dfr_paths`, `trace_refl_epc_field`, `visible_edge`.
- Avoid repeating the domain inside every word. Use `accum_dfr_direct`, not `accumulate_diffraction_order1`.
- Keep native fast-path methods strict. Do not add compatibility aliases for renamed public methods.
- Use `trace_*` for path or geometry export, and `accum_*` for kernels that reduce contributions into aggregate outputs such as grids: `accum_refl`, `accum_dfr_direct`, `accum_dfr`.

## Parameter Names and Signatures

- Parameters use full words: `source`, `receiver`, `query`, `edge_position`, `edge_direction`, `max_bounces`. Compact
  names such as `src`, `wi`, and `d0` are reserved for fields of hot data structures (see Field Names).
- A shared operation takes the same parameter names in the same order in both backends.
  `tests/test_api_signature_parity.py` enforces this and lists every accepted difference with its reason.
- Backend selectors are keyword-only: `Scene(*, trace_backend="auto", edge_bvh_backend="auto")`. Their values match
  exactly and case-sensitively, with no aliases, and an invalid value raises `ValueError`.
- A configuration of more than about four knobs goes in an options record (`DfrOptions`, `DfrCoherentOptions`,
  `DfrPathOptions`, `AccumOptions`, `SdfTraceOptions`) instead of a flat keyword list.
- Discrete choices are enums, not magic ints or strings: `DfrStrategy`, `DfrSampleSequence`, `DfrReceiverModel`,
  `ReflectionExportMode`, `AccumStrategy`, `TraceBackend`, `EdgeBVHBackend`. A member name never shadows a Python
  keyword, so the empty `RayFlags` member is `RayFlags.Empty`.
- One concept has one name in both backends: `ReflMaterial`/`ReflMaterialAD`, `SceneGeometry`, `accum_refl`,
  `vis_rejects`/`utd_rejects`. Per-row validity is a `valid` field or property plus an `is_valid()` method.

## Field Names

- Remove repeated domain prefixes inside domain-specific structs: `DfrAccum.power`, not `diffraction_power`.
- Prefer compact geometry names in hot data structures: `src`, `wi`, `d0`, `n0`, `n1`, `prim0`, `prim1`, `edge_t_min`, `edge_t_max`.
- For indexed path outputs, use `edge0`, `edge1`, `edge2` and `p0`, `p1`, `p2`.
- Use plural counter names for counters: `vis_rejects`, `edge_vis_rejects`, `utd_rejects`, `edge_uses`.

## Python/C++ Variant Pattern

- The bare Python class is the non-AD type; the AD variant carries an `AD` suffix.
- C++ keeps the `T<Detached>` alias pattern internally, with public concrete aliases matching the Python names where exposed.
