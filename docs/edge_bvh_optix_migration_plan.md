# Edge BVH OptiX Migration Plan

Plan for migrating RayD's edge-BVH acceleration structure from the
current hand-rolled LBVH + Dr.Jit-vectorized traversal to OptiX custom
AABB primitives backed by RT cores.

This is **Stage 4** in the edge-BVH evolution. Stages 1–3 (current
state, optimization log, LBVH-treelet improvements, Stage 3
preparation) addressed the hand-rolled path within its own
architectural envelope; Stage 4 changes the envelope.

Predecessor docs (read these for context, not re-explained here):

- [`edge_bvh_bottleneck_analysis.md`](edge_bvh_bottleneck_analysis.md) — current state, profiling, bottleneck list
- [`edge_bvh_optimization_log.md`](edge_bvh_optimization_log.md) — history of optimizations applied
- [`edge_bvh_lbvh_treelet_improvement_plan.md`](edge_bvh_lbvh_treelet_improvement_plan.md) — incremental fixes within the LBVH path
- [`edge_bvh_stage3_preparation.md`](edge_bvh_stage3_preparation.md) — Stage 3 status

---

## 1. Motivation and Path Comparison

### 1.1 Why migrate

The bottleneck analysis identifies five query-phase bottlenecks (§2.1–§2.5
of `edge_bvh_bottleneck_analysis.md`). Three of them are **structural
consequences of the current architecture** and cannot be removed by
incremental optimization:

| Bottleneck | Root cause | Removed by RT cores? |
|---|---|---|
| 32-slot software stack with select-based push/pop (§2.1) | Dr.Jit lane-sync requirement on stack depth | **Yes** — hardware stack, no select tax |
| L2 miss-heavy random gather on node access (§2.2) | Software BVH walks the same tree memory as compute, competes for L2 | **Yes** — RT cores have their own BVH cache |
| Dr.Jit `while_loop` lane divergence (§2.5) | Vectorized model forces all lanes to march in lockstep until last one terminates | **Yes** — RT cores schedule rays independently |

The remaining two (§2.3 conservative AABB lower bounds, §2.4 leaf
linear scan) are **geometric / numeric**, not architectural — they
persist regardless of whether traversal is software or hardware.

Build-phase costs (§1.4 CPU treelet, §1.5 CPU compaction) collapse
under `optixAccelBuild`, which is GPU-resident and does not require
GPU→CPU round-trips.

Refit (§3) shrinks from "level-by-level scatter with 17 sync points"
to a single `optixAccelBuild` UPDATE call.

### 1.2 Three OptiX primitive types — only one fits

| Primitive type | Verdict | Reasoning |
|---|---|---|
| **Built-in triangle** | ❌ Does not work | Degenerate triangles are OptiX UB; "ribbon" triangles change the geometric question (ray-surface) from what we actually need (point/ray-to-segment distance) |
| **Built-in linear curve** | ⚠ Partial fit | Linear-curve primitive is a capsule (segment + radius). Hardware ray-capsule intersection covers `nearest_edge(ray)` cleanly, but does not natively support point queries; you would degrade into custom-AABB-style handling anyway for `nearest_edge(point)` |
| **Custom AABB primitive** | ✅ Recommended | Each edge → one AABB. User-supplied `__intersection__` program does point/ray-to-segment distance. Hardware accelerates BVH traversal; leaf intersection is software. This is the established pattern for non-triangle RT-core workloads (Wald 2019; Morrical et al. 2020) |

### 1.3 The lazy-JIT clarification

A prior version of this analysis claimed that an OptiX-backed edge BVH
would "lose Dr.Jit fusion." That framing was inaccurate.

Mitsuba's and RayD's `Scene.intersect` are **lazy-JIT-integrated**: the
OptiX launch registers as a node in Dr.Jit's IR, deferred-evaluated,
and works inside `dr.syntax` symbolic loops. The same integration
applies to any OptiX call, including a custom-AABB edge query.

The actual difference between "Dr.Jit-vectorized edge BVH" and
"OptiX-backed edge BVH" is **PTX kernel count, not laziness**:

| | Dr.Jit BVH | OptiX custom AABB |
|---|---|---|
| Lazy JIT integration | ✅ | ✅ |
| Symbolic-loop compatible | ✅ | ✅ |
| AD replay pattern | ✅ (detached → AD re-gather) | ✅ (same pattern) |
| Fused into one PTX with surrounding Dr.Jit arithmetic | Possible, but… | No — OptiX launch is a separate kernel |

The "possible fusion" of the current Dr.Jit BVH does **not actually
materialize** at the call sites that matter. `nearest_edge` is
followed by `dr.compress` / `dr.gather` / `dr.scatter` (e.g.
`builders.py:bvh_pairs`), which already force evaluation boundaries.
The hypothetical fusion advantage is theoretical, not realized.

So OptiX custom AABB is a single-axis trade-off: gain RT-core
hardware traversal; lose a fusion that is not actually happening. The
"lose Dr.Jit fusion" objection is not a real cost.

### 1.4 Performance projections

End-to-end query speedup, derived from the bottleneck analysis and
published RT-core kNN benchmarks (Wald 2019, Morrical 2020):

| Phase | Current (110K edges, RTX 5080) | OptiX custom AABB (projected) | Ratio |
|---|---|---|---|
| Build | 138 ms (incl. CPU treelet + CPU compaction) | ≈ 15–30 ms (`optixAccelBuild`, GPU-resident) | 4–10× |
| Query (per batch) | software-stack traversal + Dr.Jit lane sync | RT-core traversal, software leaf | 2–4× |
| Refit | 17 level-by-level sync points | single `optixAccelBuild` UPDATE | 2–5× |

For interactive / real-time digital-twin scenarios where refit happens
every frame, the build/refit speedups compound with the query
speedup. Static scenes get only the query speedup.

These are projections; an actual prototype on a 110K-edge fixture is
the gate criterion for the migration (§7).

---

## 2. Architecture

### 2.1 OptiX pipeline layout

Use a **second OptiX pipeline** dedicated to edge primitives, parallel
to the existing triangle pipeline used by `Scene.intersect` and
`Scene.trace_reflections`. Rationale:

- Triangle pipeline uses OptiX's built-in triangle intersector; edge
  pipeline needs a custom `__intersection__` program. Different SBT
  shapes anyway.
- Separation simplifies debugging: a bug in edge traversal cannot
  destabilize triangle queries.
- The marginal memory cost of a second pipeline is small relative to
  scene BVH memory.

| | Triangle pipeline (existing) | Edge pipeline (new) |
|---|---|---|
| GAS build input | `OPTIX_BUILD_INPUT_TYPE_TRIANGLES` | `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES` |
| Intersection program | (built-in) | `__intersection__edge_*` |
| Closesthit programs | per-trace-kernel (`__closesthit__reflection`, etc.) | per-query-kernel (`__closesthit__edge_point`, `__closesthit__edge_ray`, `__anyhit__edge_topk`) |
| Programs share `OptixDeviceContext` | yes | yes |

### 2.2 Per-edge geometry layout

```cpp
struct EdgeGeometryDevice {
    const float3 *p0;          // [E]  start point (world space)
    const float3 *p1;          // [E]  end point   (world space)
    const float  *aabb;        // [E * 6]  packed (xmin, ymin, zmin, xmax, ymax, zmax)
    const uint8_t *mask;       // [E]      visibility bit for set_edge_mask
    int            n_edges;
};
```

`p0`, `p1` are the canonical edge endpoint buffers RayD already
maintains. `aabb` is rebuilt on every geometry update from `p0`/`p1`
by one CUDA kernel (replaces the LBVH `compute_primitive_bounds_kernel`).
`mask` is per-primitive — preserves `Scene.set_edge_mask` semantics
without rebuilding the BVH.

### 2.3 Per-query launch params (shared base)

```cpp
struct EdgeQueryParamsBase {
    OptixTraversableHandle handle;
    EdgeGeometryDevice     edges;

    // Inputs (per launch-index)
    const float *query_x;
    const float *query_y;
    const float *query_z;
    const uint8_t *active;
    int            n_queries;
};
```

Per-query-kind structs extend this with their own input arrays
(ray direction for ray queries, top-K parameter, etc.) and output
buffers, same convention as `reflection_trace_params.h`.

---

## 3. Query Implementations

### 3.1 `nearest_edge(point)` — point-to-edge nearest

This is the trickier query mode because OptiX is fundamentally a
ray-tracer. The established pattern (Wald 2019) co-opts OptiX
machinery for spatial queries:

1. Launch one "ray" per query point. Direction is arbitrary (use
   `(1, 0, 0)`); `tmin = 0`; `tmax = initial_search_radius` (e.g. scene
   diagonal, or `FLT_MAX` if unbounded).
2. **Inflate primitive AABBs** by `initial_search_radius` so OptiX's
   slab-test AABB traversal does not directionally cull primitives the
   point query needs to consider. The inflation happens once at build.
3. The `__intersection__edge_point` program receives each AABB-hit
   primitive, computes the actual point-to-segment distance, and if
   smaller than current `optixGetRayTmax()`, calls
   `optixReportIntersection(distance, 0)`. OptiX automatically
   tightens `tmax`, narrowing the remaining traversal.
4. `__closesthit__edge_point` records the final nearest edge id and
   distance in payload.

```cuda
extern "C" __global__ void __intersection__edge_point()
{
    const unsigned int prim = optixGetPrimitiveIndex();
    if (!params.edges.mask[prim]) return;

    const float3 q  = optixGetWorldRayOrigin();
    const float3 a  = params.edges.p0[prim];
    const float3 b  = params.edges.p1[prim];

    float dist;
    float edge_t;
    point_segment_distance(q, a, b, dist, edge_t);

    if (dist < optixGetRayTmax()) {
        optixReportIntersection(dist, 0,
            __float_as_uint(edge_t),
            prim);
    }
}

extern "C" __global__ void __closesthit__edge_point()
{
    optixSetPayload_0(optixGetAttribute_1());                // prim id
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));   // distance
    optixSetPayload_2(optixGetAttribute_0());                // edge_t bits
    optixSetPayload_3(1u);                                   // valid
}

extern "C" __global__ void __miss__edge_point()
{
    optixSetPayload_3(0u);  // no edge within search radius
}
```

#### Result

```cpp
struct NearestPointEdgeBatch {
    int               query_count = 0;
    IntDetached       edge_ids;     // [N]  global edge id,    -1 if invalid
    FloatDetached     distances;    // [N]  +inf if invalid
    Vector3fDetached  edge_points;  // [N]  closest point on edge
    BoolDetached      is_valid;     // [N]
};
```

Same field layout as the current `NearestPointEdge` so the channel
side can swap implementations without an interface change.

### 3.2 `nearest_edge(ray)` — ray-to-edge nearest

OptiX is built for ray queries; this is the natural fit. The
intersection program runs the same ray-segment narrow-phase as the
current Dr.Jit traversal (`closest_segment_segment` in `utils.h`).

```cuda
extern "C" __global__ void __intersection__edge_ray()
{
    const unsigned int prim = optixGetPrimitiveIndex();
    if (!params.edges.mask[prim]) return;

    const float3 o = optixGetWorldRayOrigin();
    const float3 d = optixGetWorldRayDirection();
    const float3 a = params.edges.p0[prim];
    const float3 b = params.edges.p1[prim];

    float dist, ray_t, edge_t;
    ray_segment_distance(o, d, a, b, dist, ray_t, edge_t);

    if (dist < optixGetRayTmax() && ray_t >= optixGetRayTmin()) {
        optixReportIntersection(dist, 0,
            __float_as_uint(ray_t),
            __float_as_uint(edge_t),
            prim);
    }
}
```

Result struct mirrors current `NearestRayEdge`.

Note: AABB inflation is **not** required for ray queries because the
ray direction is meaningful and OptiX slab test prunes correctly.
Same BVH can be used for both query kinds if we inflate at build (a
small constant cost in over-traversal for ray queries).

### 3.3 `nearest_edges_topk` — top-K extension

Top-K cannot use OptiX's "shrink tmax on report" semantics because it
needs to keep visiting AABBs even after finding K candidates — until
the K-th best stops being beatable.

Use the "any-hit visitor" pattern: every reported intersection
triggers `__anyhit__`, which updates a per-thread heap in payload and
calls `optixIgnoreIntersection()` so traversal continues.

```cuda
constexpr int K_MAX = 16;

struct TopKHeap {
    float    distances[K_MAX];
    uint32_t ids[K_MAX];
    // distances[0] = current K-th worst, [1..K-1] = ordered by max-heap
};

extern "C" __global__ void __intersection__edge_topk()
{
    const unsigned int prim = optixGetPrimitiveIndex();
    if (!params.edges.mask[prim]) return;

    const float3 q = optixGetWorldRayOrigin();
    const float3 a = params.edges.p0[prim];
    const float3 b = params.edges.p1[prim];

    float dist, edge_t;
    point_segment_distance(q, a, b, dist, edge_t);

    // Read current heap-top from payload (the K-th best so far)
    const float kth_worst = __uint_as_float(optixGetPayload_K_WORST());
    if (dist < kth_worst) {
        optixReportIntersection(dist, 0, prim, __float_as_uint(edge_t));
    }
}

extern "C" __global__ void __anyhit__edge_topk()
{
    // Read heap from payload registers, push new candidate, re-heapify,
    // write heap back to payload, then ignore so traversal continues.
    TopKHeap heap;
    load_heap_from_payload(heap);
    heap_push(heap, optixGetAttribute_0() /* prim */,
                    optixGetRayTmax() /* dist */);
    store_heap_to_payload(heap);
    optixIgnoreIntersection();
}
```

#### Payload register budget

Top-K=8: 8 floats + 8 ints = 16 registers. Fits comfortably in OptiX
8's 32-register payload limit, with headroom.

Top-K=16: 16 floats + 16 ints = 32 registers. At the limit. May need to
keep part of the heap in shared memory or local memory; measure before
committing to K=16 support.

For channel's current use case (`nearest_edges_topk` replaces 18-probe
heuristic with K≈8), K=8 is sufficient.

#### Search radius for top-K

Top-K needs a finite search radius (otherwise traversal visits the
whole tree). Choose:

- Caller-supplied `search_radius` (preferred)
- Fall back to scene diagonal if unspecified

Inflate AABBs at build time by the maximum expected `search_radius` —
or rebuild the BVH if radius changes between queries. For static
radius this is build-once.

#### Result

```cpp
struct NearestEdgesTopK {
    int               query_count = 0;
    int               k = 0;
    IntDetached       edge_ids;     // [N, K]   global edge id, -1 if rank > found
    FloatDetached     distances;    // [N, K]   sorted ascending,  +inf if rank > found
    Vector3fDetached  edge_points;  // [N, K]
    BoolDetached      is_valid;     // [N, K]
};
```

### 3.4 Edge mask handling — `set_edge_mask` preservation

The current `Scene.set_edge_mask(mask)` toggles per-edge visibility
without rebuilding the BVH. Preserve this exactly:

- `mask[prim]` lives in launch params (one byte per edge).
- Every `__intersection__edge_*` program does
  `if (!params.edges.mask[prim]) return;` as its first line.
- Updating the mask is `cudaMemcpyAsync` of the new bits — no BVH
  rebuild, no `optixAccelBuild`.

Cost per `__intersection__` invocation: one byte load. Negligible.

---

## 4. AD and JIT Integration

### 4.1 AD strategy — unchanged

The current AD pattern in
[`scene.cpp:877-964`](../src/scene/scene.cpp) is sound and survives
the migration verbatim:

1. **Detached traversal**: OptiX launch runs in detached mode, returns
   `edge_ids` (discrete).
2. **AD replay**: caller does `dr.gather` of `p0[edge_ids]`,
   `p1[edge_ids]` with AD enabled, then recomputes
   `point_segment_distance` / `ray_segment_distance` in Dr.Jit AD.
3. AD-aware outputs (`distance`, `edge_point`) carry gradients
   through the replay; `edge_ids` is discrete and carries none.

The `__intersection__` programs do not need to be AD-aware. AD only
ever sees the Dr.Jit-side replay.

`edge_bvh_bottleneck_analysis.md` §2.6 already confirmed the replay
overhead is ~10–15% and not a primary bottleneck — that finding
carries forward unchanged.

### 4.2 JIT integration — same pattern as `Scene.intersect`

The new edge query methods register as Dr.Jit JIT nodes via the same
lazy-launch mechanism `Scene.intersect` uses. No changes to channel-side
JIT discipline are required.

Hard requirements (mandatory, see `rf_trace_kernel_plan.md` §11):

- Accept Dr.Jit JIT arrays as inputs; no internal `dr.eval`.
- OptiX launch is a deferred IR node, scheduled when output is
  consumed.
- Output buffers allocated inside the launch closure, not at API
  entry.
- Compatible with `dr.syntax` symbolic loops (`mode='symbolic'`).

### 4.3 Symbolic-loop test pattern

The channel-side acceptance pattern is the MC reflection main loop
([`reflection.py:535-552`](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535-L552)).
The new edge queries must satisfy:

```python
while dr.hint(active & (depth < max_depth), mode="symbolic", ...):
    result = scene.nearest_edges_topk(query_point, query_dir, k=8)
    candidate_ids = result.edge_ids
    # ... downstream Dr.Jit math
    depth += 1
```

This is identical to how `Scene.intersect` already works — no new
integration design needed, just ensure the implementation follows the
existing convention.

---

## 5. Build and Refit

### 5.1 Build path

```
build_edge_gas(scene):
    1. Compute per-edge AABB from p0/p1, inflated by search_radius_max.
       One CUDA kernel.  Replaces lbvh::compute_primitive_bounds_kernel.
    2. optixAccelComputeMemoryUsage(...) with OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES
       to size temp + output buffers.
    3. optixAccelBuild(...) with:
           - OPTIX_BUILD_FLAG_ALLOW_UPDATE (always — RayD supports dynamic meshes)
           - OPTIX_BUILD_FLAG_PREFER_FAST_TRACE (for query throughput)
    4. (optional) optixAccelCompact(...) to shrink memory footprint
       for static scenes.
```

Replaces stages 1–7 of the current LBVH pipeline plus the CPU treelet
and CPU compaction. End-to-end build time is dominated by
`optixAccelBuild` itself, which is highly optimized.

### 5.2 Refit path

```
refit_edge_gas(scene, dirty_edge_indices):
    1. Recompute AABBs for dirty edges (one CUDA kernel over the dirty set).
    2. optixAccelBuild(...) with OPTIX_BUILD_OPERATION_UPDATE.
       Reuses the existing GAS allocation, applies in-place update.
```

Compared to the current dirty-range scatter + 17-level synchronous
refit, this is one CUDA kernel + one OptiX call. The `UPDATE` mode is
strictly cheaper than rebuild and avoids the level-by-level sync.

If too many edges become dirty (e.g. > 50% of total), `UPDATE` mode
produces a lower-quality BVH; fall back to full rebuild via
`OPTIX_BUILD_OPERATION_BUILD`. Heuristic threshold tunable, default
to fall back at 50% dirty.

### 5.3 What this removes

From `edge_bvh_bottleneck_analysis.md`:

- §1.2 — kernel-launch overhead from per-stage sync: subsumed by
  OptiX
- §1.3 — 64-bit Morton sort: gone
- §1.4 — CPU treelet optimization: gone (single largest current
  bottleneck)
- §1.5 — CPU preorder compaction: gone (compaction handled by
  OptiX)
- §1.6 — every stage of the LBVH pipeline: gone

The full bottleneck section §1 effectively retires.

---

## 6. Testing and Validation

### 6.1 Correctness reference

For each query kind, the existing Dr.Jit edge-BVH path is the
reference:

| New OptiX query | Reference for correctness |
|---|---|
| `nearest_edge(point)` (new) | current `Scene.nearest_edge(point, ...)` |
| `nearest_edge(ray)` (new) | current `Scene.nearest_edge(ray, ...)` |
| `nearest_edges_topk` | brute-force scan over all edges, sorted top-K |
| Edge mask filter | run query with mask, compare to query on filtered scene with all-ones mask |

Tolerances:

- `edge_ids`: exact match (discrete); allow ties when distances are
  numerically equal within `1e-5` × scene_diagonal.
- `distances`, `edge_points`: relative tolerance `1e-4`, absolute
  `1e-6`.
- `is_valid`: exact.

### 6.2 Test scenes

Reuse `tests/baseline_cases.py` plus add:

- `single_long_edge` — 1 edge, sanity check on the intersection
  program
- `dense_grid_edges` — 192×192 grid mesh (≈ 110K edges), matches
  `edge_bvh_bottleneck_analysis.md`'s benchmark fixture
- `disjoint_clusters` — 4 clusters of edges far apart, tests AABB
  inflation correctness
- `coincident_edges` — 100 edges sharing endpoints, tests heap-ties
  in top-K

### 6.3 Performance benchmarks

Add:

```
tests/benchmark_edge_optix.py
tests/benchmark_edge_optix_topk.py
tests/benchmark_edge_optix_build.py
```

Each writes results to `docs/performance_benchmark.json` alongside
the existing benchmark data. Sweeps:

- Edge count: 10K, 30K, 110K, 300K
- Query count per launch: 1K, 10K, 100K, 1M
- For top-K: K = 1, 4, 8, 16
- For refit: dirty fraction 1%, 10%, 50%, 100%
- Comparison: side-by-side vs current Dr.Jit BVH on same fixture

Decision metrics:

- End-to-end query throughput at the 110K / 65K-query scale (the
  bottleneck doc's benchmark point): target **≥ 2× the current
  Dr.Jit BVH**.
- Refit time at 10% dirty: target **≥ 2× faster than current
  refit**.
- Build time on cold scenes: target **≥ 3× faster than current build**.

If any of these targets is missed, hold the migration and re-evaluate
(see §7).

### 6.4 AD parity

For each query kind, compute gradients of `distance` and `edge_point`
with respect to `query_point` (or ray origin/direction) and to the
underlying mesh `vertex_positions`. Compare to current implementation:

- Forward outputs: numerical tolerance as in §6.1.
- Backward gradients: relative tolerance `1e-3` (Dr.Jit AD has its own
  floating-point characteristics; tight tolerance not always
  reachable).

### 6.5 JIT discipline (mandatory)

Same four tests as `rf_trace_kernel_plan.md` §6.7, instantiated per
edge-query kind:

1. No `dr.eval` on inputs (use `dr.kernel_history` snapshots).
2. Outputs are unevaluated JIT arrays at API return.
3. Callable inside `dr.syntax` `mode="symbolic"` loop without
   raising.
4. Result is consumable by downstream Dr.Jit math that fuses into a
   single PTX kernel (kernel boundary at the OptiX launch only).

### 6.6 Channel-side integration

Once the prototype lands and acceptance metrics are met:

| Channel site to verify | Current backend | New backend |
|---|---|---|
| [`builders.py:425`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L425) `scene.nearest_edge(...)` | Dr.Jit edge BVH | OptiX point query |
| [`builders.py:bvh_pairs`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L402) 18-probe loop | Dr.Jit edge BVH × 18 probes | OptiX `nearest_edges_topk` (one call, K=8) |
| Channel-side `set_edge_mask` usage in `builders.higher` | Dr.Jit BVH mask | OptiX mask in launch params |

Channel acceptance tests to rerun:

- `tests/deterministic/` higher-order diffraction baselines
- Munich-scale radiomap regression

---

## 7. Migration Strategy

### 7.1 Coexistence phase

The new OptiX edge BVH lands as a **parallel implementation**, not a
replacement. Selection is by build-time flag:

```python
scene = rd.Scene(edge_bvh_backend="optix")   # new path
scene = rd.Scene(edge_bvh_backend="hybrid")  # Dr.Jit point/top-K, OptiX ray
scene = rd.Scene(edge_bvh_backend="drjit")   # current path (default initially)
```

All backends share the public Python API; only the C++ implementation
differs. This lets benchmarks and acceptance tests compare them
side-by-side under identical workloads.

### 7.2 Decision gate

After the prototype is functional and passes correctness tests on all
fixtures, run the §6.3 benchmark sweep. The decision criterion is:

| Metric | Threshold | If met |
|---|---|---|
| Query throughput ≥ 2× | required | Proceed to default-swap |
| Refit ≥ 2× | required | Proceed |
| Build ≥ 3× | required | Proceed |
| Channel acceptance tests pass | required | Proceed |
| AD parity within tolerance | required | Proceed |

If any required metric fails, do **not** swap. The decision is to
either (a) continue Stage 3 work on the Dr.Jit path, or (b) iterate
on the OptiX prototype (e.g. better AABB inflation, larger leaf size
in the GAS) and re-benchmark.

### 7.3 Default swap

Once all gate criteria are met:

1. Flip the default `edge_bvh_backend` to `"optix"`.
2. Update RayD docs (`api_reference.md`, `README.md`).
3. Keep the Dr.Jit path as a fallback for one release cycle in case
   regressions surface in downstream users.

### 7.4 Deprecation

After one release with the OptiX backend as default and no regression
reports:

1. Mark Dr.Jit edge BVH as deprecated.
2. Move its source (`src/scene/edge_bvh.cu`, related files) to
   `src/scene/legacy/` and gate behind a build flag.
3. Remove in a subsequent major release.

Stage 3 work (`edge_bvh_stage3_preparation.md`) becomes
**conditionally relevant** during the coexistence phase: if Stage 4
hits a snag, Stage 3 improvements still extend the Dr.Jit path. If
Stage 4 succeeds, Stage 3 is dropped.

---

## 8. Risks and Open Questions

### 8.1 Point-query traversal: AABB inflation strategy

OptiX BVH traversal uses the ray direction for slab-test culling. For
point queries with arbitrary direction, primitives whose AABB does
not overlap the (degenerate) ray segment get culled — wrong answer.

The standard workaround is to inflate primitive AABBs by the maximum
expected search radius. Trade-offs:

- Inflate too much → over-traversal, more `__intersection__` invocations
- Inflate too little → miss valid candidates
- Adaptive (per-query) inflation → no static inflation, but requires
  per-launch AABB rebuild (defeats the point)

**Action**: prototype with `inflation = scene_diagonal × 0.1` as
default. Measure over-traversal cost. If excessive, expose
`search_radius` as a Scene-level configuration to bound inflation
more tightly.

Alternative under investigation: OptiX 8's `optixTraverse` API for
explicit traversal control. Defer until 8.1 prototype is benchmarked.

### 8.2 Top-K payload heap encoding

K=8 fits in 16 payload registers; K=16 saturates the 32-register
payload limit. Options if K=16 becomes needed:

- Spill heap to local memory (slow but correct)
- Reduce per-entry size (use 16-bit edge id if total edges < 64K)
- Use a buffered visit pattern with batched updates

Channel currently needs K ≈ 8 (replaces 18-probe heuristic), so this
is not on the critical path.

### 8.3 SBT layout

Decision: separate pipeline for edge queries (§2.1). Risks:

- Two pipelines double the host-side OptiX setup code paths.
- Shared OptiX context but independent program groups — verify no
  thread-safety surprise during `optixPipelineCreate`.

**Action**: write the host-side bootstrap as a thin wrapper around the
existing triangle-pipeline setup. Reuse `OptixDeviceContext`,
`OptixModuleCompileOptions`. Only program groups and pipeline are
duplicated.

### 8.4 Mask semantics under top-K

`set_edge_mask` filters at intersection-program entry (§3.4). Under
top-K, an entire `__intersection__edge_topk` invocation early-exits
for masked edges. The heap remains correctly populated.

But: if the mask hides > 90% of edges, traversal still visits all the
AABBs (mask check is per-primitive, not per-AABB). For sparse masks,
consider rebuilding the BVH over only the unmasked subset. **Action**:
profile typical channel mask densities (likely > 10% unmasked); if
sparse-mask cost is acceptable, no special handling. If not, add a
"masked rebuild" code path.

### 8.5 Numerical stability of point-segment distance

The current Dr.Jit implementation uses `utils.h:closest_point_on_segment`
with documented robustness. Port this function verbatim to the
intersection program (same math, same edge cases). Do **not** rewrite
the geometry; equality with the current implementation on degenerate
inputs is required for channel acceptance.

### 8.6 OptiX version dependency

Custom AABB primitives are stable from OptiX 7.x. RT-core scheduling
quality improved in OptiX 8. Set minimum OptiX version to **8.0** for
this migration to avoid platform-specific tuning.

### 8.7 Multi-GPU / device selection

Current `rd.set_device(...)` rebinds OptiX context to a chosen GPU.
The new edge pipeline must follow the same rebind path — no scene
state survives a device switch. Mirror the existing triangle pipeline's
behaviour.

---

## 9. Implementation Sequence

Recommended order:

1. **Prototype host-side scaffolding** — second OptiX pipeline,
   `__intersection__` / `__closesthit__` / `__miss__` skeleton that
   always returns "no edge found." Establishes the build/launch
   plumbing.

2. **`nearest_edge(ray)` first** — easiest, no AABB-inflation
   ambiguity, direct mapping from current implementation. Smallest
   scope to verify the new pipeline produces correct results.

3. **`nearest_edge(point)` with AABB inflation** — once ray queries
   are validated, add point queries with the inflation strategy in
   §8.1.

4. **`nearest_edges_topk`** — once both single-best queries work,
   extend with the any-hit visitor + heap pattern.

5. **Build / refit migration** — replace the current LBVH build with
   `optixAccelBuild`. Validate against existing AABB outputs on the
   same fixtures.

6. **Edge mask integration** — wire `set_edge_mask` into the launch
   params path. Trivial after the rest is in.

7. **Benchmark sweep** — run §6.3, capture acceptance metrics, write
   results into `docs/performance_benchmark.json`.

8. **Decision gate** — apply §7.2 criteria; either swap default or
   iterate.

9. **Channel-side integration** — once default-swapped, channel can
   retire the 18-probe heuristic and use `nearest_edges_topk` directly
   (per [channel doc 24](../../witwin-platform/channel/docs/dev/plans/24-realtime-rt-architecture-roadmap.md) §7 D2).

Each numbered step has its own correctness test before moving on. The
prototype phase (steps 1–6) is the bulk of the work; benchmark and
integration are tractable once the kernels exist.

---

## 10. Out of Scope

Not addressed by this plan:

- **`SecondaryEdgeInfo`** (mesh-local edge analysis for primary-edge
  sampling): unchanged; uses a different code path.
- **Camera primary-edge cache** (camera-side image-space edges):
  unchanged; not BVH-backed.
- **Edge topology and adjacency queries** (`Scene.edge_info()`,
  `Scene.edge_topology()`): these return data, not queries, and are
  not affected by the BVH backend swap.
- **Stage 3 LBVH improvements**: superseded by this plan if migration
  succeeds; otherwise continue per
  `edge_bvh_stage3_preparation.md`.

---

## Appendix: References

- `edge_bvh_bottleneck_analysis.md` — current state profiling
- `edge_bvh_optimization_log.md` — incremental optimization history
- `edge_bvh_stage3_preparation.md` — current path's next step
- `multi_bounce_reflection_trace.md` — established `trace_*` kernel
  pattern this plan extends
- `rf_trace_kernel_plan.md` — sibling plan for visibility kernels;
  same JIT discipline requirements apply
- Wald, "Using OptiX for accelerated nearest-neighbor queries," 2019
- Morrical et al., "Accelerating Unstructured Mesh Point Location
  with RT Cores," 2020
