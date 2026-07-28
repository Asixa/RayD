# API Rename / Migration Guide

Naming cleanup applied on 2026-05-21 (branch `api-naming-cleanup`). Pure renames,
no functional changes. This document lists every old name and its replacement.

## 1. Naming convention flip (Python)

The bare class name is now the **non-AD (detached)** variant - the common case.
The autodiff variant carries an **`AD`** suffix.

```python
# before                          # after
ray = rd.RayDetached(...)         ray = rd.Ray(...)          # non-AD
ray = rd.Ray(...)                 ray = rd.RayAD(...)        # autodiff
```

C++ follows the same convention. Each type `X` is defined as a template
`XData<Float_>`; `XT<Detached>` selects the non-AD or AD `Float`, and `X` (non-AD)
/ `XAD` (AD) are the two concrete instantiations (`include/rayd/core/drjit.h:23-30`).
There is no `RayDetached` alias. The foundational `Float` / `Vector*f` /
`Matrix4f` aliases follow the same rule: `Float` is non-AD and `FloatAD` is the
autodiff variant (`include/rayd/core/drjit/types.h:25-26, 78-79, 87-88`).

## 2. Python class renames

### Flipped families (each had an AD and a `Detached` variant)

| Old (AD) | New (AD) | Old (non-AD) | New (non-AD) |
|---|---|---|---|
| `Ray` | `RayAD` | `RayDetached` | `Ray` |
| `Intersection` | `IntersectionAD` | `IntersectionDetached` | `Intersection` |
| `ReflectionChain` | `ReflectionChainAD` | `ReflectionChainDetached` | `ReflectionChain` |
| `ReflectionBounce` | `ReflectionBounceAD` | `ReflectionBounceDetached` | `ReflectionBounce` |
| `ReflectionTrace` | `ReflectionTraceAD` | `ReflectionTraceDetached` | `ReflectionTrace` |
| `ReflectionEpcResult` | `ReflEpcAD` | `ReflectionEpcResultDetached` | `ReflEpc` |
| `ReflectionEpcFieldResult` | `ReflEpcFieldAD` | `ReflectionEpcFieldResultDetached` | `ReflEpcField` |
| `PrimitiveMaterialPayload` | `MaterialAD` | `PrimitiveMaterialPayloadDetached` | `Material` |
| `ReflectionWedgeEventBuffer` | `WedgeEventsAD` | `ReflectionWedgeEventBufferDetached` | `WedgeEvents` |
| `ReflectionAccumulationResult` | `AccumResultAD` | `ReflectionAccumulationResultDetached` | `AccumResult` |
| `NearestPointEdge` | `NearestPointEdgeAD` | `NearestPointEdgeDetached` | `NearestPointEdge` |
| `NearestRayEdge` | `NearestRayEdgeAD` | `NearestRayEdgeDetached` | `NearestRayEdge` |
| `NearestEdgesTopK` | `NearestEdgesTopKAD` | `NearestEdgesTopKDetached` | `NearestEdgesTopK` |
| `SegmentVisibility` | `SegmentVisibilityAD` | `SegmentVisibilityDetached` | `SegmentVisibility` |
| `SegmentPairVisibility` | `SegmentPairVisibilityAD` | `SegmentPairVisibilityDetached` | `SegmentPairVisibility` |
| `AxialEdgeVisibility` | `AxialEdgeVisibilityAD` | `AxialEdgeVisibilityDetached` | `AxialEdgeVisibility` |
| `SegmentChainVisibility` | `SegmentChainVisibilityAD` | `SegmentChainVisibilityDetached` | `SegmentChainVisibility` |

### Trim-only (single class, no AD/detached pair)

| Old | New |
|---|---|
| `ReflectionAccumulationGrid` | `AccumGrid` |
| `ReflectionAccumulationOptions` | `AccumOptions` |
| `ReflectionEpcOptions` | `ReflEpcOptions` |
| `ReflectionEpcFieldOptions` | `ReflEpcFieldOptions` |

Unchanged: `ReflectionTraceOptions`, `RayFlags`, `SecondaryEdgeInfo`, `SceneEdgeInfo`, `SceneEdgeTopology`, `SceneSyncProfile`,
`SceneEdgeBVHStats`, `Mesh`, `Scene`.

## 3. Method renames (`Scene`)

| Old | New |
|---|---|
| `trace_segment_visibility(...)` | `visible(...)` |
| `trace_segment_pair_visibility(...)` | `visible_pair(...)` |
| `trace_segment_chain_visibility(...)` | `visible_chain(...)` |
| `trace_axial_edge_visibility(...)` | `visible_edge(...)` |
| `trace_reflections_accumulating(...)` | `accumulate_reflections(...)` |
| `trace_reflection_epc(...)` | `trace_refl_epc(...)` |
| `trace_reflection_epc_field(...)` | `trace_refl_epc_field(...)` |
| `trace_reflection_epc_field_direct(tx_position, ...)` | `trace_refl_epc_field(tx_position, ...)` |

`trace_reflection_epc_field_direct` was merged into `trace_refl_epc_field`
as an overload: pass a transmitter position (`Vector3f`) as the first argument for
the direct form, or a `Ray` for the original form.

## 4. Audit dict key (`native_launch_audit()`)

| Old key | New key |
|---|---|
| `audit["trace_reflections_accumulating"]` | `audit["accumulate_reflections"]` |

## 5. C++ type / symbol renames (for C++ users)

Stem trims apply to all variants (`*Data`, `*T`, `*Detached`, and the bare alias):

| Old C++ stem | New C++ stem |
|---|---|
| `PrimitiveMaterialPayload` | `Material` |
| `ReflectionWedgeEventBuffer` | `WedgeEvents` |
| `ReflectionAccumulationResult` | `AccumResult` |
| `ReflectionAccumulationGrid` | `AccumGrid` |
| `ReflectionAccumulationOptions` | `AccumOptions` |
| `ReflectionAccumulationParams` | `AccumParams` |
| `SceneGlobalGeometry` | `SceneGeometry` |
| `ReflectionAccumulationRaw` (file-local) | `AccumRaw` |
| `ReflectionEpcResult` | `ReflEpc` |
| `ReflectionEpcFieldResult` | `ReflEpcField` |
| `ReflectionEpcOptions` | `ReflEpcOptions` |
| `ReflectionEpcFieldOptions` | `ReflEpcFieldOptions` |
| enum `NativeLaunchStage::TraceReflectionsAccumulating` | `NativeLaunchStage::AccumulateReflections` |
| `NativeLaunchAuditSnapshot::trace_reflections_accumulating` | `::accumulate_reflections` |

C++ method renames mirror the Python ones (`Scene::visible`, `Scene::visible_pair`,
`Scene::visible_chain`, `Scene::visible_edge`, `Scene::accumulate_reflections`,
`Scene::trace_refl_epc`, and the merged `Scene::trace_refl_epc_field` overload).

## 6. Quick migration (Python)

```python
import re, pathlib

CLASS = {
    # detached -> bare
    "RayDetached": "Ray", "IntersectionDetached": "Intersection",
    "ReflectionChainDetached": "ReflectionChain", "ReflectionBounceDetached": "ReflectionBounce",
    "ReflectionTraceDetached": "ReflectionTrace",
    "ReflectionEpcResultDetached": "ReflEpc",
    "ReflectionEpcFieldResultDetached": "ReflEpcField",
    "PrimitiveMaterialPayloadDetached": "Material",
    "ReflectionWedgeEventBufferDetached": "WedgeEvents",
    "ReflectionAccumulationResultDetached": "AccumResult",
    "NearestPointEdgeDetached": "NearestPointEdge", "NearestRayEdgeDetached": "NearestRayEdge",
    "NearestEdgesTopKDetached": "NearestEdgesTopK",
    "SegmentVisibilityDetached": "SegmentVisibility",
    "SegmentPairVisibilityDetached": "SegmentPairVisibility",
    "AxialEdgeVisibilityDetached": "AxialEdgeVisibility",
    "SegmentChainVisibilityDetached": "SegmentChainVisibility",
    # trims (single)
    "ReflectionAccumulationGrid": "AccumGrid",
    "ReflectionAccumulationOptions": "AccumOptions",
    "ReflectionEpcOptions": "ReflEpcOptions",
    "ReflectionEpcFieldOptions": "ReflEpcFieldOptions",
    # AD bare -> AD suffix  (apply AFTER the detached ones above)
    "PrimitiveMaterialPayload": "MaterialAD",
}
METHOD = {
    "trace_segment_pair_visibility": "visible_pair",
    "trace_segment_chain_visibility": "visible_chain",
    "trace_segment_visibility": "visible",
    "trace_axial_edge_visibility": "visible_edge",
    "trace_reflections_accumulating": "accumulate_reflections",
    "trace_reflection_epc": "trace_refl_epc",
    "trace_reflection_epc_field": "trace_refl_epc_field",
    "trace_reflection_epc_field_direct": "trace_refl_epc_field",
}
# NOTE: the AD bare names that simply gain an "AD" suffix (Ray->RayAD,
# Intersection->IntersectionAD, etc.) are context-dependent - review those by hand.
```

## 7. 2026-05-22 Dfr diffraction API

Diffraction now uses the `Dfr` stem so `diff` remains reserved for differentiation/autodiff discussions. No compatibility aliases are kept.

| Old | New |
|---|---|
| `DiffractionStateTable` | `DfrStates` |
| `DiffractionGrid` | `DfrGrid` |
| `DiffractionMaterial` | `DfrMaterial` |
| `DiffractionAccumOptions` | `DfrOptions` |
| `DiffractionPathOptions` | `DfrPathOptions` |
| `DiffractionAccumResult` | `DfrAccum` |
| `DiffractionPathResult` | `DfrPaths` |
| `Scene.accumulate_diffraction_order1(...)` | `Scene.accum_dfr_direct(...)` |
| `Scene.accumulate_diffraction_chains(...)` | `Scene.accum_dfr(...)` |
| `Scene.trace_diffraction_paths(...)` | `Scene.trace_dfr_paths(...)` |
| `RAYD_DIFF_DIRECT` | `RAYD_DFR_DIRECT` |
| `RAYD_DIFF_KELLER` | `RAYD_DFR_KELLER` |
| `RAYD_DIFF_SUFFIX_REFLECTION` | `RAYD_DFR_SUFFIX_REFL` |
| `RAYD_DIFF_HASH` / `RAYD_DIFF_SOBOL` | `RAYD_DFR_HASH` / `RAYD_DFR_SOBOL` |
| `RAYD_DIFF_MATCHED_ISOTROPIC` | `RAYD_DFR_MATCHED_ISO` |

Key field renames:

| Old | New |
|---|---|
| `edge_line_min` / `edge_line_max` | `edge_t_min` / `edge_t_max` |
| `face0_normal` / `face1_normal` | `n0` / `n1` |
| `face0_prim_id` / `face1_prim_id` | `prim0` / `prim1` |
| `source_pos` / `source_power` | `src` / `src_power` |
| `incident_direction` / `initial_direction` | `wi` / `d0` |
| `prefix_reflection_depth` | `prefix_depth` |
| `diffraction_power` / `diffraction_field_*` | `power` / `field_*` |
| `visibility_reject_count` / `inter_edge_visibility_reject_count` | `vis_rejects` / `edge_vis_rejects` |
| `utd_reject_count` / `edge_use_count` | `utd_rejects` / `edge_uses` |
| `edge_index_0/1/2` | `edge0/1/2` |
| `point_0/1/2` | `p0/1/2` |
| `tx_index` / `rx_index` | `tx_id` / `rx_id` |
| `return_geometry` / `max_receivers` | `return_geom` / `max_rx` |

See [`API_NAMING_STANDARD.md`](API_NAMING_STANDARD.md) for the active naming rules.
