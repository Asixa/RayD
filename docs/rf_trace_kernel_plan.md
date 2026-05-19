# RF / Channel `trace_*` Kernel Plan

Implementation plan for the next round of fused OptiX kernels in RayD,
driven by performance work in `witwin.channel`. Each kernel below is
the natural extension of the pattern established by
[`multi_bounce_reflection_trace.md`](multi_bounce_reflection_trace.md):
one `optixLaunch`, custom raygen + closesthit (+ anyhit where useful),
device-side output buffers wrapped as Dr.Jit / Torch arrays.

The candidate set is the load-bearing subset of `witwin.channel`'s
fusion inventory (`witwin-platform/channel/docs/dev/plans/24-realtime-rt-architecture-roadmap.md`, §7).
This document specifies APIs, payload layouts, file additions, and test
plans for each kernel without prescribing implementation details that
are obvious from the existing `reflection_trace.cu`.

---

## Scope

In scope (new fused kernels):

| # | Name | Replaces in channel | Pri |
|---|---|---|---|
| 1 | `trace_segment_visibility` | `Scene.segment_visible` with `ignore_prim_idx` Python re-fire loop | P0 |
| 2 | `trace_segment_pair_visibility` | shadow-boundary smoothed pair (`a→b`, `a→b_offset`) | P1 |
| 3 | `trace_axial_edge_visibility` | axial source-edge sampling loop in MC postprocessing | P1 |
| 4 | `nearest_edges_topk` (edge-BVH primitive, not a `trace_*`) | 18-probe heuristic in higher-order diffraction BFS | P1 |

Explicitly **out of scope**:

- `trace_diffraction_chain` (channel doc D2) — most of its win comes
  from Dr.Jit-symbolic loop fusion + (4) on the candidate side. Will
  be re-evaluated after P0–P1 land.
- `trace_reflection_chain_accumulating` (D3 / M1) — does **not** need a
  new RayD kernel. The existing `trace_reflections` output already
  contains the per-segment `t`, `hit_p`, `geo_n`, `prim_id` channel
  needs; the speedup is purely a channel-side refactor to flatten the
  bounce loop into one Dr.Jit kernel over the full chain output. See
  §6 for the small RayD-side adjustment that makes this clean.
- User-extensible closesthit / Slang `optix-ir` target. Decision
  recorded in channel doc 24 §3 — keep RayD's surface narrow and add
  first-class `trace_*` kernels for each RF / acoustic pattern,
  rather than exposing a generic shader extension point.

---

## Shared Design Principles

These are non-negotiable conventions for new `trace_*` kernels, codifying
what `reflection_trace.cu` already gets right.

1. **One launch per call.** No host-side loop wrapping `optixLaunch`. If
   a workload needs multiple traces per launch index, fire them inside
   raygen against the same GAS handle.

2. **Thin payload.** Closesthit communicates with raygen through OptiX
   payload registers only (≤ 8 u32 for portability across SMs). No
   global memory scratch space for per-ray transient state.

3. **Geometry only.** No RF / acoustic physics inside the kernel. UTD
   coefficients, polarization projection, reflection coefficients, and
   field accumulation stay in channel-side Dr.Jit kernels operating on
   RayD output buffers. RayD stays a kernel library, not a renderer.

4. **Params struct in `src/<module>/<name>_params.h`**, same pattern as
   `reflection_trace_params.h`: flat scalar pointer fields (`float *x`,
   `float *y`, ...) and integer counts. No `float3*` in the header so
   it stays `#include`-able from both host C++ and CUDA without `optix.h`
   being mandatory.

5. **`_host.cpp` / `_host.h` separation.** Host-side pipeline build,
   buffer allocation, and `optixLaunch` glue live in `<name>_host.cpp`;
   CUDA programs in `<name>.cu`; PTX embedded via
   `<name>_ptx.h` (generated). Match `reflection_trace_*` layout.

6. **Public header in `include/rayd/<name>.h`.** Defines the result
   struct (`SegmentVisibility`, `EdgePairVisibility`, etc.) using the
   `*Detached` Dr.Jit types already used by `Intersection` and
   `ReflectionChain`.

7. **Method on `Scene`.** Each kernel is exposed as a method on
   `rayd::Scene`, with mirror methods on `rayd.torch.Scene` that
   accept and return CUDA `torch.Tensor` with `(N,)` / `(N, 3)`
   layouts.

8. **AD passthrough mirrors `Scene.intersect`.** Detached inputs return
   detached outputs; gradient-bearing inputs propagate analytic
   gradients where they are well-defined. For boolean visibility
   outputs gradients are zero by definition.

9. **`active` mask first-class.** All kernels accept an `active: MaskDetached`
   to skip work on disabled lanes — same convention as `Scene.intersect`.

10. **AD-safe, not AD-required.** Channel currently invokes every
    candidate replaced by these kernels inside `with dr.suspend_grad():`
    or otherwise detached contexts (visibility is a hard step;
    higher-order candidate enumeration uses ids only). So **none of
    these kernels need AD math.** The requirement is the weaker one:

    - kernels must not crash or silently corrupt when given
      `drjit.cuda.ad.*` inputs (forward path runs detached internally
      is fine),
    - kernels must not be the *source* of a permanent gradient block
      on the input tensors — gradients on continuous inputs may flow
      around the kernel through other code paths downstream.

    Continuous outputs that mirror an existing AD-supported API (e.g.
    `nearest_edges_topk.distances` mirrors `nearest_edge.distance`)
    must preserve the existing contract so future callers can opt in.
    Pure-Bool kernels (visibility kernels) have no AD contract to
    preserve.

    If a downstream use case eventually needs gradients through these
    kernels (e.g. shadow-boundary smoothing baked in), it is added
    explicitly later, not retrofitted by removing detach boundaries.

11. **Dr.Jit JIT discipline — never force materialization. Mandatory.**
    Channel relies on `dr.syntax` symbolic loops for the MC reflection
    main path
    ([reflection.py:535](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535))
    and uses Dr.Jit fusion aggressively everywhere else. Every call to
    a new `trace_*` must:

    - accept inputs as JIT nodes — no internal `dr.eval(...)` /
      `.numpy()` / `.torch()` on inputs,
    - register the OptiX launch as a JIT node so it fuses into the
      surrounding Dr.Jit IR — same lazy-launch mechanism used by the
      existing `Scene.intersect` and `Scene.trace_reflections`,
    - allocate output buffers at launch time inside the JIT closure,
      not eagerly at API entry, so constructing the result object does
      not force evaluation,
    - keep auxiliary inputs (`ignore_prim_ids`, `active`, sample
      fractions, ignore counts) as JIT arrays / compile-time ints; do
      not slip a `dr.eval` in to inspect them on the host.

    This is the one hard cross-cutting requirement. Test it explicitly
    — see §6.7. If JIT integration is incomplete, the kernel does not
    ship.

---

## 1. `trace_segment_visibility` (P0)

### Why

`witwin.channel.Scene.segment_visible(start, end, ignore_prim_idx=...)`
currently re-fires the same ray up to 8 times in Python, advancing past
each ignored primitive
([`channel/witwin/channel_scene/scene.py:761`](../../witwin-platform/channel/witwin/channel_scene/scene.py#L761)).
For shadow tests inside the deterministic diffraction BFS and the MC
BDPT MIS path, this is the dominant ray-query cost.

Anyhit `optixIgnoreIntersection()` collapses the re-fire loop into a
single launch.

### Result struct

```cpp
// include/rayd/segment_visibility.h
struct SegmentVisibility {
    int            ray_count = 0;
    BoolDetached   visible;        // [N]  true == segment from start to end is clear
};
```

A single bit per ray. No per-ray hit info is returned; if the caller
wants the blocker, it should use `Scene.intersect()` instead.

### Python API

```python
visibility = scene.trace_segment_visibility(
    start,                # Array3f / ad.Array3f, [N]
    end,                  # Array3f / ad.Array3f, [N]
    ignore_prim_ids=None, # Int32  [N, K] or None;   global prim id space
    active=True,          # Bool  [N] or scalar
)
visibility.visible        # Bool [N]
```

`ignore_prim_ids` is a `(N, K)` Int32 tensor (or None). `K` is the
maximum number of ignored primitives per ray; `-1` entries are treated
as "no ignore." `K` is fixed at launch time; channel currently passes
up to 4 ignores (two adjacent faces per edge, two endpoints' worth
when chaining), so a runtime cap of `K=8` covers the foreseeable use.

### PyTorch wrapper

```python
visible = scene.trace_segment_visibility(
    start,            # torch.Tensor float32 [N, 3] CUDA
    end,              # torch.Tensor float32 [N, 3] CUDA
    ignore_prim_ids=None,  # torch.Tensor int32 [N, K] CUDA, or None
    active=None,      # torch.Tensor bool [N] CUDA, or None
)
# visible: torch.Tensor bool [N]
```

### Payload layout

Single u32 payload slot.

```
p0: 0 == blocked, 1 == clear
```

### Programs

```cuda
extern "C" __global__ void __anyhit__segment_visibility()
{
    const unsigned int ray_index    = optixGetLaunchIndex().x;
    const unsigned int local_prim   = optixGetPrimitiveIndex();
    const unsigned int instance     = optixGetInstanceId();
    const int          shape_id     = (int)instance;
    const int          face_offset  =
        (shape_id >= 0 && shape_id < params.n_meshes)
            ? params.face_offsets[shape_id] : 0;
    const int          global_prim  = face_offset + (int)local_prim;

    for (int i = 0; i < params.ignore_k; ++i) {
        const int ig = params.ignore_prim_ids[ray_index * params.ignore_k + i];
        if (ig == global_prim) {
            optixIgnoreIntersection();
            return;
        }
    }
    // not ignored — fall through to closesthit / terminate
}

extern "C" __global__ void __closesthit__segment_visibility()
{
    optixSetPayload_0(0u);  // blocked
}

extern "C" __global__ void __miss__segment_visibility()
{
    // payload was preinitialized to 1 (clear) by raygen
}

extern "C" __global__ void __raygen__segment_visibility()
{
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= (unsigned int)params.n_rays) return;
    if (params.active_mask && !params.active_mask[ray]) {
        params.out_visible[ray] = 0u;
        return;
    }

    const float3 o = make_vec3(params.start_x[ray], params.start_y[ray], params.start_z[ray]);
    const float3 e = make_vec3(params.end_x[ray],   params.end_y[ray],   params.end_z[ray]);
    float3 d = e - o;
    const float len = sqrtf(dot3(d, d));
    if (len < kMinSegLen) {
        params.out_visible[ray] = 1u;
        return;
    }
    d = (1.0f / len) * d;
    const float3 origin   = o + kRayBias * d;
    const float  tmax     = fmaxf(len - 2.0f * kRayBias, 0.0f);

    unsigned int p0 = 1u;  // assume clear; closesthit will flip to 0 if blocked
    optixTrace(params.handle, origin, d, kRayTMin, tmax, 0.0f,
               255u, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 1, 0,
               p0);
    params.out_visible[ray] = (uint8_t)p0;
}
```

`OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT` short-circuits at the first
non-ignored hit. Anyhit must not call `optixTerminateRay` itself
because ignored intersections need traversal to continue.

### AD behaviour

Channel always invokes `segment_visible` under `with dr.suspend_grad():`
([scene.py:704, 745](../../witwin-platform/channel/witwin/channel_scene/scene.py#L704)),
so AD math is not required. The kernel only needs to be AD-safe:
accept `ad.Array3f` inputs without crashing, return a detached `Bool`,
do not install a gradient block that would later interfere with
unrelated gradient paths through `start` / `end`.

### JIT discipline — mandatory

- Callable inside `dr.syntax` symbolic loops. The exact pattern that
  must work is the MC reflection main loop
  ([reflection.py:535-552](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535-L552))
  where `scene.ray_intersect` is called inside `dr.hint(mode='symbolic')`.
- OptiX launch registers as a JIT node, fuses with surrounding Dr.Jit
  IR.
- Output buffers allocated lazily inside the launch closure.
- No `dr.eval` on inputs.

### Channel call sites — drop-in points

Each of the following currently routes through
`witwin.channel_scene.scene.Scene.segment_visible(..., ignore_prim_idx=...)`,
whose body
([scene.py:686-789](../../witwin-platform/channel/witwin/channel_scene/scene.py#L686-L789))
is the Python re-fire loop being replaced. The drop-in is a single edit
inside `Scene.segment_visible` once `trace_segment_visibility` is
available; all call sites below inherit it automatically.

| Channel file | Line | Pattern |
|---|---|---|
| `deterministic/path/diffraction_impl/builders.py` | 244, 484 | source / prev-edge → next-edge visibility with 2–4 ignores |
| `deterministic/path/diffraction_impl/forward.py` | 261-268 | source / target → diffraction point with adjacent-face ignores |
| `deterministic/path/diffraction_impl/postprocessing.py` | 269, 317 | tx / source → edge_pos with surface-group ignores |
| `deterministic/path/reflection_impl/epc.py` | 656, 666, 695, 705 | reflection-path validation chain (4 segments per path) |
| `deterministic/kernels/utd/native_impl.py` | 393 | UTD native kernel boundary visibility |
| `montecarlo/integrators/bdpt_diffraction.py` | 1683, 1689 | edge → reflection → target chain |

After the swap, run the corresponding channel acceptance tests as
regression — see §6.8.

### Files

```
include/rayd/
    segment_visibility.h          NEW — SegmentVisibility result struct

src/multipath/
    segment_visibility.cu         NEW — OptiX programs
    segment_visibility_host.cpp   NEW — pipeline + launch
    segment_visibility_host.h     NEW
    segment_visibility_params.h   NEW
    segment_visibility_ptx.h      NEW (generated)

src/rayd.cpp                      MOD — nanobind binding
src/scene/scene.cpp               MOD — Scene::trace_segment_visibility()
rayd/torch/scene.py               MOD — torch wrapper
CMakeLists.txt                    MOD — PTX compilation rule
```

---

## 2. `trace_segment_pair_visibility` (P1)

### Why

Diffraction shadow-boundary smoothing (channel
[`montecarlo/path/diffraction.py:309-318`](../../witwin-platform/channel/witwin/montecarlo/path/diffraction.py#L309))
fires four `segment_visible` calls per state — `a → b` and `a → b_offset`
on both source and target sides — to evaluate the smoothed visibility
indicator across a shadow boundary. The CPU launches are paired by
construction; doing both rays in one launch index halves the launch
count.

### Result struct

```cpp
// include/rayd/segment_visibility.h  (same file as kernel 1)
struct SegmentPairVisibility {
    int            ray_count = 0;
    BoolDetached   visible_a;        // [N]
    BoolDetached   visible_b;        // [N]
};
```

### Python API

```python
pair = scene.trace_segment_pair_visibility(
    start,           # Array3f [N]
    end_a,           # Array3f [N]
    end_b,           # Array3f [N]
    ignore_prim_ids=None,  # Int32 [N, K] or None — applied to both rays
    active=True,
)
pair.visible_a       # Bool [N]
pair.visible_b       # Bool [N]
```

### Payload layout

Two u32 slots. Raygen fires two `optixTrace` calls per launch index,
one for each end point, each with its own payload register cleared and
read.

```
trace #1: p0 = visible_a (init 1, blocked -> 0)
trace #2: p0 = visible_b (init 1, blocked -> 0)
```

### Programs

Reuse the closesthit / anyhit / miss programs from kernel 1 unchanged.
Only `__raygen__segment_pair_visibility` is new — it does two
`optixTrace` invocations against the same anyhit/closesthit pair, with
different end points.

### AD / JIT

Same as kernel 1: AD-safe but not AD-required (channel calls this
inside detached context for shadow-boundary smoothing — the smoothing
is finite-difference on the inputs, not autodiff through visibility).
JIT discipline is mandatory.

### Channel call sites — drop-in points

| Channel file | Line | Pattern |
|---|---|---|
| `montecarlo/path/diffraction.py` | 309-310, 317-318 | `(source → diff_point, source → diff_point_offset)` and the same pair on the target side |
| `montecarlo/integrators/bdpt_diffraction.py` | 679-681 | `_segment_pair_visible` — currently falls back to two sequential `scene.segment_visible` when `VisibilityKernel.available()` is False; this kernel becomes the always-available fast path |

The drop-in target is
[`bdpt_diffraction.py:_segment_pair_visible`](../../witwin-platform/channel/witwin/montecarlo/integrators/bdpt_diffraction.py#L664)
— replace the two-call branch with a single
`scene.trace_segment_pair_visibility` invocation; the existing
`VisibilityKernel` CUDA kernel can be retired in the same change.

### Files

Add to the kernel 1 module; no new module needed.

```
src/multipath/segment_visibility.cu        MOD — add __raygen__segment_pair_visibility
src/multipath/segment_visibility_host.cpp  MOD — second pipeline launch entry
include/rayd/segment_visibility.h          MOD — SegmentPairVisibility struct
```

---

## 3. `trace_axial_edge_visibility` (P1)

### Why

MC postprocessing checks source-to-edge visibility at multiple sample
points along the edge axis to approximate "source can see any portion
of the edge"
([`channel/witwin/montecarlo/path/postprocessing.py:140-163`](../../witwin-platform/channel/witwin/montecarlo/path/postprocessing.py#L140)).
The current Python loop fires N (3–5) full segment_visible queries per
state. Folding the N samples into one launch index removes N-1 launches
per state.

### Result struct

```cpp
// include/rayd/axial_edge_visibility.h
struct AxialEdgeVisibility {
    int            state_count = 0;
    BoolDetached   any_visible;        // [N] OR-reduction over samples
};
```

If a future caller needs per-sample bits, add `visible_per_sample` as a
`[N, S]` `BoolDetached`. Defer until a use site appears.

### Python API

```python
result = scene.trace_axial_edge_visibility(
    source_pos,                # Array3f [N]
    edge_pos,                  # Array3f [N]   — base point on edge
    edge_dir,                  # Array3f [N]   — unit edge direction
    edge_line_min,             # Float   [N]   — axial bound (signed)
    edge_line_max,             # Float   [N]   — axial bound (signed)
    sample_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),  # compile-time tuple
    active=True,
)
result.any_visible             # Bool [N]
```

`sample_fractions` is a host-side tuple converted to a small const-mem
array on launch; up to 16 samples supported.

### Payload layout

```
p0: 1 if ANY sample's segment is clear, 0 otherwise
```

Raygen initialises `p0 = 0`, loops over fractions, ORs in the per-sample
result. Each `optixTrace` uses the closesthit/anyhit/miss from kernel 1.

### Programs

```cuda
extern "C" __global__ void __raygen__axial_edge_visibility()
{
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= (unsigned int)params.n_states) return;

    if (params.active_mask && !params.active_mask[ray]) {
        params.out_any_visible[ray] = 0u;
        return;
    }

    const float3 s   = load_vec3(params.source_x, params.source_y, params.source_z, ray);
    const float3 ep  = load_vec3(params.edge_pos_x, params.edge_pos_y, params.edge_pos_z, ray);
    const float3 ed  = load_vec3(params.edge_dir_x, params.edge_dir_y, params.edge_dir_z, ray);
    const float lmin = params.edge_line_min[ray];
    const float lmax = params.edge_line_max[ray];
    const float span = fmaxf(lmax - lmin, 0.0f);

    uint8_t any = 0u;
    for (int i = 0; i < params.n_samples; ++i) {
        const float frac = params.sample_fractions[i];
        const float axial = lmin + frac * span;
        const float3 sp   = ep + axial * ed;

        // segment from s to sp
        float3 d = sp - s;
        const float len = sqrtf(dot3(d, d));
        if (len < kMinSegLen) { any = 1u; continue; }
        d = (1.0f / len) * d;
        const float3 origin = s + kRayBias * d;
        const float  tmax   = fmaxf(len - 2.0f * kRayBias, 0.0f);

        unsigned int p0 = 1u;
        optixTrace(params.handle, origin, d, kRayTMin, tmax, 0.0f,
                   255u, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   0, 1, 0, p0);
        any |= (uint8_t)p0;
    }
    params.out_any_visible[ray] = any;
}
```

If ignore_prim support is needed later, share kernel 1's anyhit. For
v1, no ignore is required (MC postprocessing does not pass ignores
through this path).

### AD / JIT

AD-safe but not AD-required. JIT mandatory. `sample_fractions` is a
host-side compile-time tuple; the rest of the inputs (positions, line
bounds) stay JIT arrays. No `dr.eval` on inputs.

### Channel call site — drop-in point

| Channel file | Line | Pattern |
|---|---|---|
| `montecarlo/path/postprocessing.py` | 140-163 | `_source_visible_edge_mask` — Python loop over `SOURCE_VISIBILITY_SAMPLE_FRACTIONS` issuing one `segment_visible` per fraction, OR-reducing in Python |

The drop-in replaces the whole `_source_visible_edge_mask` body with a
single `scene.trace_axial_edge_visibility(...)` call; the
`SOURCE_VISIBILITY_SAMPLE_FRACTIONS` constant is passed through as the
`sample_fractions` argument.

### Files

```
include/rayd/axial_edge_visibility.h               NEW
src/multipath/axial_edge_visibility.cu             NEW
src/multipath/axial_edge_visibility_host.cpp       NEW
src/multipath/axial_edge_visibility_host.h         NEW
src/multipath/axial_edge_visibility_params.h       NEW
src/multipath/axial_edge_visibility_ptx.h          NEW (generated)
src/rayd.cpp                                       MOD
src/scene/scene.cpp                                MOD
rayd/torch/scene.py                                MOD
CMakeLists.txt                                     MOD
```

---

## 4. `nearest_edges_topk` (P1 — edge BVH primitive, not a `trace_*`)

### Why

Higher-order diffraction in channel currently enumerates next-edge
candidates by firing 18 deterministic probe rays per previous state
through `nearest_edge`
([`channel/.../diffraction_impl/builders.py:402-438`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L402)).
This is a heuristic with bounded recall — RayD's edge BVH already
contains the spatial structure to answer "for each query point /
direction, give me the top-K nearest edges" exactly. Exposing this as
a primitive replaces the probe heuristic and frees channel from
maintaining a probe-radius schedule.

### Python API

```python
result = scene.nearest_edges_topk(
    query_point,            # Array3f [N]
    query_dir=None,         # Array3f [N] or None (for ray queries)
    k=8,                    # int, k <= 16
    active=True,
)
result.edge_ids             # Int32 [N, K]   global edge ids,   -1 if fewer than K hits
result.distances            # Float [N, K]   sorted ascending,  +inf if invalid
result.edge_points          # Array3f [N, K] closest point on each returned edge
result.is_valid             # Bool   [N, K]
```

Behaviour mirrors the existing `nearest_edge` but returns the top-K
matches instead of just the best one. K is a compile-time constant per
call (allocated once at launch).

### Implementation note

This is **not** a new OptiX kernel — it is an extension of the existing
CUDA edge-BVH traversal in `src/scene/edge_bvh.cu`. The traversal stack
already visits internal nodes in distance order; instead of stopping
after the first leaf hit, maintain a top-K heap per thread.

Heap size K ≤ 16 means a fixed-size insertion sort fits in registers
(no spill). Stop-condition: pop heap top once distance exceeds the
worst best-K. Standard pattern.

### AD behaviour

Channel's higher-order BFS uses `edge_ids` only (it gathers candidate
positions from `edge_data['pos']` afterwards) and currently does **not**
differentiate through edge selection. So AD math is again not required
in v1.

However, this kernel is the top-K extension of an API
(`nearest_edge`) that **does** have AD support for `distance` and
`edge_point`. To avoid making the top-K variant a regression for any
future caller, the kernel should preserve the same AD semantics on
`distances[:, i]` and `edge_points[:, i]` for each rank i: run the
discrete top-K selection detached, then replay the continuous quantities
in AD mode against the selected edge IDs. This mirrors what
`nearest_edge` already does and reuses the same replay code path.

`edge_ids` and `is_valid` are discrete, no AD.

### JIT discipline — mandatory

Top-K traversal happens inside one CUDA kernel launch that registers as
a JIT node, same as `nearest_edge`. `query_point`, `query_dir`,
`active` stay JIT arrays. K is a compile-time int.

### Channel call site — drop-in point

| Channel file | Line | Pattern |
|---|---|---|
| `deterministic/path/diffraction_impl/builders.py` | 402-438 | `bvh_pairs` — currently 18-probe heuristic firing `scene.nearest_edge` per state and deduplicating |

The drop-in is a meaningful refactor, not a line swap:

- Today's `bvh_pairs` allocates `n_probes = chunk_n_prev * 18` probe
  rays, fires `nearest_edge` once, and feeds the deduplicated results
  to `compact_index_pairs` ([builders.py:425-438](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L425-L438)).
- With `nearest_edges_topk`, the call becomes one `scene.nearest_edges_topk(edge_pos, basis_k, k=K)` per state — no probe-grid construction, no per-state direction sampling, no dedupe step (the topk output is already distinct edges per state).
- `_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT` and `_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_*` constants in
  [builders.py:23-26](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L23-L26)
  retire in the same change.

The acceptance criterion for the drop-in is: deterministic Munich
diffraction baseline (in
`witwin-platform/channel/tests/deterministic/`) matches within the
existing tolerance, while the probe-heuristic recall concern (channel
doc 24 §7 D2) goes away by construction.

### Files

```
include/rayd/nearest_edge.h        MOD — NearestEdgesTopK struct
src/scene/edge_bvh.cu              MOD — k-NN traversal kernel
src/scene/edge_bvh.h               MOD — declaration
src/scene/scene_edge.cpp           MOD — Scene::nearest_edges_topk
src/rayd.cpp                       MOD — binding
rayd/torch/scene.py                MOD
```

---

## 5. Why D3 / M1 do not need a new kernel

For completeness — kernels listed under "out of scope" but commonly
asked about.

`witwin.channel`'s deterministic suffix-reflection loop
([forward.py:578-619](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/forward.py#L578))
and the MC reflection trace
([reflection.py:535-619](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535))
both do B Python iterations, each issuing one or two `scene.intersect`
calls and a Dr.Jit physics step. The natural fusion idea is "fold the
bounce loop into one OptiX kernel that calls back into channel physics
at each hit."

That is not the right answer. The existing `trace_reflections` output
already contains, for each ray and bounce, `hit_point`, `geo_normal`,
`global_prim_id`, `t`, `bounce_count`. With that, channel can flatten
its B-iteration loop into a single Dr.Jit kernel over the `[N, B]`
output, doing reflection coefficients, polarization, and receiver-grid
accumulation in one symbolic pass. Total launches drop from `~3B` to
`2` (one OptiX `trace_reflections`, one Dr.Jit physics kernel).

The only RayD-side adjustment needed: `trace_reflections` should also
output, for the *last* bounce of each ray, the distance to the next
blocker in the reflected direction (currently dropped because the
internal loop breaks on miss). Without it, channel still has to fire
one extra `intersect` after `trace_reflections` for receiver-grid
accumulation on the trailing segment.

### Small RayD adjustment

Add to `ReflectionChain`:

```cpp
FloatDetached  trailing_t;       // [N]   distance to next blocker from last bounce in reflected dir;
                                  //       +inf if no blocker
IntDetached    trailing_prim;    // [N]   global prim id of trailing blocker;   -1 if none
Vector3fDetached trailing_dir;   // [N]   reflected direction at last bounce
Vector3fDetached trailing_origin;// [N]   last bounce hit_p + bias along trailing_dir
```

Implementation: one extra `optixTrace` after the bounce loop terminates
(either by max_bounces or by miss), inside the same raygen. No
additional launch.

### AD / JIT for trailing fields

Channel uses `trace_reflections` today only inside discovery /
detached contexts
([`deterministic/path/reflection_impl/paths.py:84-115`](../../witwin-platform/channel/witwin/deterministic/path/reflection_impl/paths.py#L84-L115)),
so AD is not required for the trailing fields either — match the
existing `t` / `hit_points` AD semantics if AD ever gets enabled, but
do not block on it for v1.

JIT-wise, `trace_reflections` is already a JIT node; adding outputs
does not change the lazy-launch / fusion story.

### Channel call sites — where the trailing fields unlock fusion

Channel's two B-iteration loops that do `intersect → physics →
next_blocker → physics` per bounce:

| Channel file | Line range | What flattens after trailing fields land |
|---|---|---|
| `deterministic/path/diffraction_impl/forward.py` | 570-619 | Suffix reflection chain; second `intersect_rays_with_prim` per bounce (line 606) reads `next_blocker_t`, which becomes `trace_reflections.t[ray, b+1]` (or `trailing_t[ray]` for the last bounce). |
| `montecarlo/path/reflection.py` | 535-619 | MC reflection main symbolic loop; `scene.ray_intersect` inside the loop is replaced by a single `trace_reflections` upstream, and the symbolic loop becomes a `[N, B]` Dr.Jit kernel over the chain output. |

Both refactors are channel-side and tracked under channel doc 24 §7 D3
/ M1 — they are not RayD work, but the trailing fields above are the
RayD-side prerequisite that makes the refactor clean.

---

## 6. Testing Methodology

All tests follow the existing pattern in `tests/drjit/test_geometry.py`:
subprocess-isolated Python scripts that print a single-line JSON
result, asserted by `unittest`. Subprocess isolation avoids OptiX state
leaking between tests.

### 6.1 Correctness reference: brute-force

Every visibility kernel has a Python brute-force reference that
operates on the same scene without invoking the new kernel:

```python
def brute_force_segment_visible(scene, start, end, ignore_prim_ids):
    """Reference: repeated scene.intersect() advancing past ignores."""
    # ... existing logic, slow but obviously correct
```

The test asserts bit-exact equality between the kernel output and the
brute-force reference on a small batch (N ≤ 256). For sample-randomised
tests, use a fixed seed.

### 6.2 Correctness reference: existing API

For each new kernel, a test compares against the slower but already-
verified existing API:

| New kernel | Reference |
|---|---|
| `trace_segment_visibility` | `Scene.intersect()` in a Python re-fire loop |
| `trace_segment_pair_visibility` | Two separate `trace_segment_visibility` calls |
| `trace_axial_edge_visibility` | Python loop of `trace_segment_visibility` |
| `nearest_edges_topk` | Brute-force scan over all edges via `nearest_edge` filter |

### 6.3 Test scene matrix

Per kernel, the same three scene fixtures used in
`tests/baseline_cases.py`:

- `single_triangle` — one triangle, sanity check
- `two_quads_perpendicular` — multi-mesh, multi-prim, exercise instance ids
- `cornell_box_decimated` — ≥100 triangles, multi-mesh, dense enough for
  meaningful traversal

### 6.4 Per-kernel test list

#### `trace_segment_visibility`

1. **No-ignore correctness vs brute-force `scene.intersect()`** — N=256
   random segments through `cornell_box_decimated`. Must match
   bit-exact.
2. **Single-ignore correctness** — N=128 segments where the closest
   blocker is the one ignored prim. Must return clear.
3. **Multi-ignore correctness** — K=4 ignores per ray, against the
   brute-force re-fire reference.
4. **Degenerate segments** — start == end, very short (< kMinSegLen),
   very long (> 1e6). Must not crash, must return well-defined.
5. **Empty ignore tensor** — `ignore_prim_ids=None` matches
   `ignore_prim_ids=full(-1)` exactly.
6. **`active` mask** — disabled lanes return `False`, do not launch a
   trace.
7. **Performance ratio** — vs current channel `segment_visible` with 4
   ignores, on a Munich-style scene. Target ≥ 3× speedup at N ≥ 64k.
8. **Torch frontend round-trip** — same fixtures, `torch.Tensor` in /
   out, results match Dr.Jit frontend.

#### `trace_segment_pair_visibility`

1. **Equivalence to two separate calls** — must match
   `(trace_segment_visibility(s, a), trace_segment_visibility(s, b))`
   bit-exact under the same scene + ignores.
2. **Performance ratio** — vs two sequential calls. Target ≥ 1.7×
   speedup at N ≥ 64k.
3. **Torch frontend round-trip.**

#### `trace_axial_edge_visibility`

1. **Single-sample equivalence** — `n_samples=1, fraction=0.5` matches
   `trace_segment_visibility` to the midpoint of the edge.
2. **OR-reduce semantics** — fabricate scene where exactly one sample
   along the edge is visible from source; must return `any_visible=True`.
3. **All-blocked case** — source entirely occluded; must return
   `any_visible=False` for all samples.
4. **Sample count scaling** — `n_samples` from 1 to 16. Output stable,
   no crashes.
5. **Performance ratio** — vs Python loop of N samples × N states.
   Target speedup ≥ N_samples × 0.8.
6. **Torch frontend round-trip.**

#### `nearest_edges_topk`

1. **K=1 equivalence** — `nearest_edges_topk(k=1).edge_ids[:,0]` matches
   `nearest_edge(...).edge_id` and `.distances[:,0]` matches `.distance`.
2. **Brute-force top-K** — N=128 query points, K=8, against a
   Python brute-force `argsort` on all edges. Must match `edge_ids`
   exactly when distances are unambiguous; allow tie-breaking
   tolerance.
3. **Distance monotonicity** — `distances[:,i] <= distances[:,i+1]` for
   all i. Asserted across whole output.
4. **Sparse scenes** — fewer than K edges within finite radius; padded
   entries must have `is_valid=False` and `distance=+inf`.
5. **K bounds** — K=1, K=8, K=16. K=0 raises, K>16 raises.
6. **Performance vs 18-probe heuristic** — measure recall (fraction of
   probe-found candidates that also appear in top-K=18) and runtime on
   the deterministic Munich diffraction BFS. Target: recall ≥ 0.99,
   runtime ≤ 1.2× the heuristic (the heuristic is cheap because it
   reuses Scene.intersect; we are buying recall guarantee, not speed).
7. **Torch frontend round-trip.**

### 6.5 Integration tests (downstream witwin)

After each kernel lands, do the channel-side drop-in and run the
relevant channel acceptance tests from
`witwin-platform/channel/tests/`. The drop-in target sites are listed
in each kernel's "Channel call sites" subsection above; the
verification runs are:

| Kernel | Channel drop-in site | Channel test to assert no regression |
|---|---|---|
| `trace_segment_visibility` | `channel_scene/scene.py:686-789` (`Scene.segment_visible` body) | MC BDPT shadow suite; deterministic diffraction segment-visibility coverage; UTD native impl edge cases |
| `trace_segment_pair_visibility` | `bdpt_diffraction.py:_segment_pair_visible` + `montecarlo/path/diffraction.py:309-318` | smoothed shadow-boundary regression; BDPT MIS path-tracing baseline |
| `trace_axial_edge_visibility` | `montecarlo/path/postprocessing.py:_source_visible_edge_mask` | MC postprocessing radiomap baseline |
| `nearest_edges_topk` | `deterministic/path/diffraction_impl/builders.py:bvh_pairs` (retire 18-probe heuristic) | deterministic Munich diffraction baseline; higher-order diffraction reference outputs |
| `trace_reflections` trailing fields | `deterministic/path/diffraction_impl/forward.py:570-619` (D3) and `montecarlo/path/reflection.py:535-619` (M1) flatten refactor | deterministic suffix-reflection regression; MC reflection radiomap baseline |

Pass criterion: numerical outputs match the pre-drop-in baseline within
existing tolerances. For each kernel, capture the channel test run
output as the canonical regression artifact.

### 6.6 Microbenchmarks

Add to `tests/benchmark_*.py`:

- `benchmark_segment_visibility.py` — scene size sweep × N sweep ×
  ignore-K sweep, comparing against brute-force re-fire.
- `benchmark_axial_edge_visibility.py` — N_samples sweep.
- `benchmark_edge_topk.py` — K sweep, integrated into existing
  `benchmark_edge_queries.py`.

Output to `docs/performance_benchmark.json` so we can track regressions
the same way the existing benchmark does.

### 6.7 JIT discipline tests (mandatory per kernel)

For every kernel, the JIT-friendliness requirement (principle 11) needs
explicit, automated coverage. Without these the kernel may "work" at
the API level but force a `dr.eval` under the hood and silently break
channel's symbolic loops.

For each kernel add to `tests/drjit/test_<kernel>_jit.py`:

1. **No-eval-on-inputs.** Construct inputs as Dr.Jit JIT nodes (e.g.
   `dr.arange(...)`, `dr.linspace(...)`, derived but not evaluated).
   Snapshot `dr.kernel_history()` before the call. Call the kernel.
   Assert the kernel call did **not** trigger a Dr.Jit kernel
   evaluation as a side effect on the inputs — only the OptiX launch
   itself appears in the history.
2. **No-eval-on-outputs at API return.** Immediately after the kernel
   returns, assert the result fields are unevaluated JIT arrays
   (`dr.is_evaluated(result.visible) is False`, or whichever Dr.Jit
   API the version supports). Evaluation must only happen on explicit
   `dr.eval` / `.numpy()`.
3. **Symbolic-loop containment.** Replicate the exact pattern from
   channel's MC reflection loop
   ([reflection.py:535-552](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535-L552)):

   ```python
   while dr.hint(active & (depth < max_depth), mode="symbolic",
                 max_iterations=8, label="kernel_jit_test"):
       result = scene.trace_segment_visibility(start, end, ...)
       blocked = ~result.visible
       depth += 1
   ```

   The test must complete without raising "operation cannot be
   recorded in symbolic mode" or equivalent Dr.Jit errors. Compare
   output to an evaluated-mode (`mode="evaluated"`) run for
   bit-equivalence.
4. **Fusion-into-downstream.** After the kernel call, do a small
   chain of Dr.Jit ops on the output and one of the inputs (e.g.
   `dr.select(result.visible, start, end).sum()`). Verify only one
   Dr.Jit kernel is generated for the entire chain (excluding the
   OptiX launch itself).

These four tests are mandatory acceptance criteria for any new
`trace_*` kernel. A kernel that ships without them does not count as
landed.

### 6.8 AD-safety tests (per kernel — light)

Since AD math is not required (principle 10), AD tests are
narrow-scope sanity checks, not correctness validation:

1. **`ad.Array3f` inputs do not crash.** Pass `drjit.cuda.ad.Array3f`
   for positions; assert the kernel returns successfully with the same
   detached outputs as a non-AD run.
2. **No gradient leak on unrelated graph.** Build a small autodiff
   graph where `start`/`end` come from an AD computation, call the
   kernel, then continue the AD graph on a sibling tensor (not
   through the kernel output). `dr.backward()` on the sibling must
   produce the expected gradient — the kernel must not install a
   global detach.
3. **For `nearest_edges_topk` only**: distances/edge_points behave the
   same as `nearest_edge` under AD on the query point. Compare
   `nearest_edges_topk(k=1).distances[:, 0]` gradient to
   `nearest_edge().distance` gradient on a fixed scene — must match
   to numerical tolerance.

No tests are required for AD math through visibility booleans;
gradient is zero by definition and that is correct behaviour.

---

## 7. Implementation Order

Recommended sequence, optimised for unblocking channel-side work as
fast as possible:

1. **Kernel 1 — `trace_segment_visibility`** alone. Lands the anyhit
   pattern, the params/host/PTX module template, and the bindings
   updates. Once merged, channel can refactor `segment_visible` to
   call it and immediately retire the re-fire loop. Unblocks D4 / D5 /
   M4 in the channel inventory.

2. **Small `trace_reflections` adjustment (§5)** — `trailing_t`,
   `trailing_prim`, `trailing_dir`, `trailing_origin` outputs. One
   extra `optixTrace` inside the existing raygen. Unblocks the
   channel-side D3 / M1 flatten-loop refactor.

3. **Kernel 2 — `trace_segment_pair_visibility`** sharing kernel 1's
   anyhit/closesthit. Trivial after kernel 1.

4. **Kernel 4 — `nearest_edges_topk`**. Independent of kernels 1–3
   (different module). Can be done in parallel.

5. **Kernel 3 — `trace_axial_edge_visibility`**. Smaller demand than
   kernels 1–2; do once the others are stable.

6. **(deferred)** Re-evaluate `trace_diffraction_chain` (D2). After
   kernels 1–4 + channel refactor, measure the residual cost of
   higher-order diffraction. If it is still dominated by Python
   orchestration of edge-chain expansion, consider a fused chain
   kernel. If it is dominated by Dr.Jit physics math on flattened
   candidate sets, no new kernel needed.

Each kernel is one or two days of focused work given the
`reflection_trace.cu` template. The CMake / PTX / nanobind glue is the
slowest part and is shared across all four.

---

## 8. Open Questions

- **Payload register budget on older SMs.** Kernels 1–3 fit comfortably
  in ≤ 2 payload registers. If a future kernel needs > 8, check SM 6.x
  compatibility constraints; OptiX 8 raised the cap but old drivers
  matter for some deployment targets.

- **`ignore_prim_ids` global vs. local space.** Channel passes global
  prim ids through `Scene._broadcast_ignore_i32`. Kernel 1 anyhit
  computes `global = face_offsets[instance] + local_prim`, which is
  one extra dependent load per anyhit invocation. If the inner loop
  becomes hot, consider passing per-instance offsets via SBT records
  instead of params indirection.

- **Top-K traversal stack depth.** Edge-BVH traversal currently sized
  for K=1 (single best). K=16 may bump per-thread stack memory above
  the comfort threshold for high occupancy. Measure during kernel 4
  implementation; if it regresses, fall back to traversing twice
  (first pass: bounding distance threshold; second pass: exact top-K).

- **Test fixture reuse.** `tests/baseline_cases.py` was built for
  intersection / nearest_edge baselines. May need a small extension
  for visibility-kernel scenes that intentionally place blockers
  between sampled points. Plan small additions during kernel 1
  implementation.
