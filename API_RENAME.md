# API Rename / Migration Guide

Naming cleanup applied on 2026-05-21 (branch `api-naming-cleanup`). Pure renames,
no functional changes. This document lists every old name and its replacement.

## 1. Naming convention flip (Python)

The bare class name is now the **non-AD (detached)** variant — the common case.
The autodiff variant carries an **`AD`** suffix.

```python
# before                          # after
ray = rd.RayDetached(...)         ray = rd.Ray(...)          # non-AD
ray = rd.Ray(...)                 ray = rd.RayAD(...)        # autodiff
```

This flip is **Python-only**. In C++, aliases keep their existing convention:
the bare name is still the AD type and the detached type still carries a
`Detached` suffix (e.g. C++ `Ray` = AD, `RayDetached` = non-AD). The foundational
`Float` / `Vector*f` / `Matrix4f` aliases are unchanged.

## 2. Python class renames

### Flipped families (each had an AD and a `Detached` variant)

| Old (AD) | New (AD) | Old (non-AD) | New (non-AD) |
|---|---|---|---|
| `Ray` | `RayAD` | `RayDetached` | `Ray` |
| `Intersection` | `IntersectionAD` | `IntersectionDetached` | `Intersection` |
| `ReflectionChain` | `ReflectionChainAD` | `ReflectionChainDetached` | `ReflectionChain` |
| `ReflectionBounce` | `ReflectionBounceAD` | `ReflectionBounceDetached` | `ReflectionBounce` |
| `ReflectionTrace` | `ReflectionTraceAD` | `ReflectionTraceDetached` | `ReflectionTrace` |
| `ReflectionEpcResult` | `ReflectionEpcResultAD` | `ReflectionEpcResultDetached` | `ReflectionEpcResult` |
| `ReflectionEpcFieldResult` | `ReflectionEpcFieldResultAD` | `ReflectionEpcFieldResultDetached` | `ReflectionEpcFieldResult` |
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

Unchanged: `ReflectionTraceOptions`, `ReflectionEpcOptions`, `ReflectionEpcFieldOptions`,
`RayFlags`, `SecondaryEdgeInfo`, `SceneEdgeInfo`, `SceneEdgeTopology`, `SceneSyncProfile`,
`SceneEdgeBVHStats`, `Mesh`, `Scene`.

## 3. Method renames (`Scene`)

| Old | New |
|---|---|
| `trace_segment_visibility(...)` | `visible(...)` |
| `trace_segment_pair_visibility(...)` | `visible_pair(...)` |
| `trace_segment_chain_visibility(...)` | `visible_chain(...)` |
| `trace_axial_edge_visibility(...)` | `visible_axial_edge(...)` |
| `trace_reflections_accumulating(...)` | `accumulate_reflections(...)` |
| `trace_reflection_epc_field_direct(tx_position, ...)` | `trace_reflection_epc_field(tx_position, ...)` |

`trace_reflection_epc_field_direct` was merged into `trace_reflection_epc_field`
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
| enum `NativeLaunchStage::TraceReflectionsAccumulating` | `NativeLaunchStage::AccumulateReflections` |
| `NativeLaunchAuditSnapshot::trace_reflections_accumulating` | `::accumulate_reflections` |

C++ method renames mirror the Python ones (`Scene::visible`, `Scene::visible_pair`,
`Scene::visible_chain`, `Scene::visible_axial_edge`, `Scene::accumulate_reflections`,
and the merged `Scene::trace_reflection_epc_field` overload).

## 6. Quick migration (Python)

```python
import re, pathlib

CLASS = {
    # detached -> bare
    "RayDetached": "Ray", "IntersectionDetached": "Intersection",
    "ReflectionChainDetached": "ReflectionChain", "ReflectionBounceDetached": "ReflectionBounce",
    "ReflectionTraceDetached": "ReflectionTrace",
    "ReflectionEpcResultDetached": "ReflectionEpcResult",
    "ReflectionEpcFieldResultDetached": "ReflectionEpcFieldResult",
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
    # AD bare -> AD suffix  (apply AFTER the detached ones above)
    "PrimitiveMaterialPayload": "MaterialAD",
}
METHOD = {
    "trace_segment_pair_visibility": "visible_pair",
    "trace_segment_chain_visibility": "visible_chain",
    "trace_segment_visibility": "visible",
    "trace_axial_edge_visibility": "visible_axial_edge",
    "trace_reflections_accumulating": "accumulate_reflections",
    "trace_reflection_epc_field_direct": "trace_reflection_epc_field",
}
# NOTE: the AD bare names that simply gain an "AD" suffix (Ray->RayAD,
# Intersection->IntersectionAD, etc.) are context-dependent — review those by hand.
```
