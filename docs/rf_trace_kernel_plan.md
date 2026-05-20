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

### Phase 1 (initial planning — all delivered)

| # | Name | Replaces in channel | Pri | Status |
|---|---|---|---|---|
| 1 | `trace_segment_visibility` | `Scene.segment_visible` with `ignore_prim_idx` Python re-fire loop | P0 | ✅ implemented (`src/multipath/segment_visibility.cu`) |
| 2 | `trace_segment_pair_visibility` | shadow-boundary smoothed pair (`a→b`, `a→b_offset`) | P1 | ✅ implemented (same module) |
| 3 | `trace_axial_edge_visibility` | axial source-edge sampling loop in MC postprocessing | P1 | ✅ implemented (same module) |
| 4 | `nearest_edges_topk` (edge-BVH primitive, not a `trace_*`) | 18-probe heuristic in higher-order diffraction BFS | P1 | ✅ implemented as part of [`edge_bvh_optix_migration_plan.md`](edge_bvh_optix_migration_plan.md) (`src/scene/edge_optix.cu`) |

### Phase 2 (implemented)

| # | Name | Replaces in channel | Pri | Status |
|---|---|---|---|---|
| 5 | `trace_segment_chain_visibility` | EPC / BDPT N-segment Python `for` loop of `segment_visible` | P0 | ✅ implemented (`src/multipath/segment_visibility.cu`) |
| — | `trace_reflections` trailing fields | enables channel-side D3 / M1 flatten refactor | P1 | ✅ implemented (`src/multipath/reflection_trace.cu`) |

### Phase 2 RF native accumulation (implemented)

These items intentionally break the earlier geometry-only rule for a
single channel RF fast path. They are kept separate from the default
reflection trace API so callers choose the native RF path explicitly.

| # | Name | Replaces in channel | Pri | Status |
|---|---|---|---|---|
| 6 | `trace_reflections_accumulating` | `montecarlo/trace/reflection.py` Dr.Jit scatter-reduce forward fast path | P0 | ✅ implemented (`include/rayd/reflection_accumulation.h`, `src/multipath/reflection_accumulation.cu`) |
| 7 | reflection wedge event buffer | `diff_state_store` scatter of wedge/diffraction candidates | P0 | ✅ implemented as part of `trace_reflections_accumulating` |
| 8 | native accumulating AD contract | prevents silent AD fallback | P0 | ✅ implemented: native accumulating rejects AD inputs |
| 9 | wider primitive ignore tables | removes previous `K <= 8` host guard for visibility native path | P1 | ✅ implemented for primitive ignore tables |

### Explicitly out of scope

- `trace_diffraction_chain` (channel doc D2) — honest re-assessment
  (see channel doc 24 §7 plus the speedup audit in §8 below) puts BFS-mode
  fusion at 5–20% gain, not the originally claimed 3–10×. State-array
  materialization between BFS orders is the dominant cost, and fusion
  cannot remove it. Re-evaluate only after Phase 2 lands and if BFS is
  still the residual bottleneck.
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
- Structure-level visibility ignore mapping and arbitrary structure
  ignore tables. Primitive ignore tables are now wide, but the native
  visibility control surface is still primitive-id based. Callers that
  need structure-level ignores must receive an explicit unsupported
  error instead of falling back to an approximate path.
- `trace_diffraction_chain_mc` and per-edge MIS / UTD cache evaluation.
  These still need a fixed Channel ABI for field payload, UTD constants,
  edge cache ownership, and target/RX accumulation semantics before RayD
  can implement a real single-launch chain-field kernel.

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

3. **Geometry first, with explicit RF exceptions.** The default RayD
   surface remains geometry-only: UTD coefficients, polarization
   projection, reflection coefficients, and field accumulation stay in
   channel-side Dr.Jit kernels operating on RayD output buffers. The
   RF-native accumulating kernel below is the one explicit exception:
   it is a named non-AD fast path, has separate types, and is not used
   implicitly by `trace_reflections`.

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

11a. **Lazy outside symbolic loops (mandatory).** All callers expect
    `trace_*` to participate in Dr.Jit's deferred-evaluation graph when
    invoked outside `dr.syntax(mode='symbolic')`. Concretely, this
    means:

    - inputs are accepted as Dr.Jit JIT arrays — no internal
      `dr.eval(...)` / `.numpy()` / `.torch()` on inputs,
    - output buffers are allocated inside the JIT closure at launch
      time, so constructing the result object does not force
      evaluation,
    - auxiliary inputs (`ignore_prim_ids`, `active`, sample fractions,
      counts) stay as JIT arrays / compile-time ints; no `dr.eval` to
      inspect them on the host.

    Both the **jit path** (inline `optixTrace` from within Dr.Jit's
    emitted PTX) and the **native path** (standalone `optixLaunch`
    with custom raygen / anyhit / closesthit) satisfy this requirement
    outside symbolic loops. The native path registers as a Dr.Jit IR
    node and is scheduled — same as `Scene.trace_reflections` —
    inserting a kernel boundary at launch time but allowing Dr.Jit to
    sequence it among other ops.

12. **Inside symbolic loops (jit path required, native not supported).**
    Channel relies on `dr.syntax` symbolic loops for the MC reflection
    main path
    ([reflection.py:535](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535))
    and similar hot loops elsewhere. When recording such a loop,
    Dr.Jit emits a single PTX kernel containing the loop body.

    - The **jit path** uses Dr.Jit's `optixTrace` integration: the
      OptiX call is emitted as inline assembly inside the recorded
      PTX. Recordable. ✅
    - The **native path** uses host-side `optixLaunch` with its own
      raygen kernel. Not emittable from within a Dr.Jit symbolic
      recording — would force the loop out of symbolic mode or raise.
      ❌

    Dispatch (see "Backend Dispatch" below) selects the jit path
    automatically when no native-only feature (custom anyhit,
    thin payload, in-kernel state accumulation) is required. Callers
    do not have to think about symbolic-loop compatibility, **except
    in the one corner case** of "in-symbolic-loop call that needs
    anyhit-ignore" — see Backend Dispatch §"Edge case".

    This is the cross-cutting compatibility requirement. Test it
    explicitly — see §7.7.

---

## Backend Dispatch — `jit` vs `native` paths

The visibility kernels ship two implementations and pick one at call
time. This section documents the dispatch mechanism so callers and
contributors share a single mental model.

### Two implementations

| Path | Implementation | Symbolic-loop recordable | Anyhit / thin payload | Kernel boundary |
|---|---|---|---|---|
| **jit** | inline `optixTrace` in Dr.Jit-emitted PTX (`OptixScene::segment_hit<true>(...)`) | ✅ recordable | ❌ no custom programs | none |
| **native** | standalone `optixLaunch` with custom raygen / anyhit / closesthit pipeline (`launch_segment_visibility_detached(...)`) | ❌ not recordable | ✅ full custom-program control | one boundary at launch |

### Dispatch rule

Implemented in `use_jit_trace_visibility_path(ignore_k)` in
[`src/scene/scene.cpp`](../src/scene/scene.cpp). Picks the path based
on **what features the call actually needs**, with an environment-
variable override for testing:

```
RAYD_TRACE_VISIBILITY_BACKEND ∈ {auto, jit, native}

backend == native  → always native (forced; for benchmarking / debugging)
backend == jit     → always jit (errors if ignore_k > 0)
backend == auto    → jit when ignore_k == 0, else native        (default)
```

The rule is **feature-driven**, not context-driven. A caller's path
choice does not change based on whether they happen to be inside a
symbolic loop — it changes only with the presence of an ignore list.
This makes call behaviour predictable: the same call site always
takes the same path.

### Four-quadrant truth table

How dispatch behaves in each combination of (calling context) ×
(feature need), under `backend = auto`:

| Calling context | `ignore_k` | Selected path | Optimal? |
|---|---|---|---|
| Outside symbolic loop | 0 | **jit** | ✅ no kernel boundary, fuses with surrounding Dr.Jit IR |
| Outside symbolic loop | > 0 | **native** | ✅ anyhit needed for ignore list |
| Inside symbolic loop | 0 | **jit** | ✅ recordable, same PTX kernel as loop body |
| Inside symbolic loop | > 0 | **native** | ⚠ see Edge case below |

The jit path is not just a "compatibility fallback" — for the
no-ignore case it is the **better** choice everywhere, because:

- BVH traversal count is the same (1 traversal in both paths)
- jit avoids the kernel boundary that native would insert
- jit avoids native's full-Intersection write-back (although shadow_test
  variants minimize this anyway)

### Edge case: symbolic loop + ignore list

This is the one combination the current implementation does not
gracefully handle:

- `backend = jit` + `ignore_k > 0`: `require(ignore_k == 0, ...)`
  raises with a clear message.
- `backend = auto` + `ignore_k > 0` inside a symbolic loop: dispatch
  picks native, but recording into the symbolic loop will fail at the
  Dr.Jit level (since the native path is a host-side `optixLaunch`).
  The error surfaces as a Dr.Jit recording failure, not as a clean
  application error.

**Channel does not currently hit this combination.** All channel
`segment_visible(..., ignore_prim_idx=...)` call sites operate at
Dr.Jit evaluation boundaries (after `dr.compress` / `dr.scatter` /
batch state-array materialization), not inside symbolic recordings.
See §8.6 for the explicit reality check.

**If a future caller needs this combination**, two extension paths:

1. **Jit path adds ignore support via Dr.Jit re-fire.** Implement the
   ignore loop in Dr.Jit by iteratively re-firing `optixTrace` and
   advancing the origin past each ignored hit. N BVH traversals
   instead of 1 traversal + N anyhit invocations — slower, but
   symbolic-recordable. Pure Dr.Jit code; modest implementation cost.

2. **Explicit error at dispatch.** Add `require(!dr_recording_active()
   || ignore_k == 0, "trace_segment_visibility with ignore_prim_ids
   cannot be called inside a dr.syntax symbolic loop; hoist the call
   out of the loop or evaluate inputs first.")` so the failure mode
   is a clear application-level error rather than a Dr.Jit-internal
   trace.

Recommended: option 2 immediately (zero-cost guard), and option 1
only when a real call site needs it.

### When to override the default

| Override | Use case |
|---|---|
| `RAYD_TRACE_VISIBILITY_BACKEND=native` | Benchmark native vs jit; debug anyhit logic by forcing every call through the native pipeline |
| `RAYD_TRACE_VISIBILITY_BACKEND=jit` | Verify symbolic-loop tests; catch accidental anyhit dependencies |
| `RAYD_TRACE_VISIBILITY_BACKEND=auto` | Production / default |

These environment variables are read once at scene-construction time
(static `const` initializer). Changing them mid-process has no
effect.

---

## 1. `trace_segment_visibility` (P0) — ✅ implemented

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
as "no ignore." `K` is fixed at launch time. The native anyhit path now
accepts arbitrary primitive ignore width subject to device memory and
launch occupancy; the jit path still supports no-ignore only.

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
- Implemented now for `ignore_prim_ids` empty. Non-empty ignore lists
  still use the custom anyhit native fallback.

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
regression — see §7.8.

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

## 2. `trace_segment_pair_visibility` (P1) — ✅ implemented

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
JIT discipline is mandatory. Implemented now for `ignore_prim_ids`
empty; non-empty ignore lists still use the custom anyhit native
fallback.

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

## 3. `trace_axial_edge_visibility` (P1) — ✅ implemented

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

## 4. `nearest_edges_topk` (P1 — edge BVH primitive, not a `trace_*`) — ✅ implemented

Implemented as part of the broader edge-BVH OptiX migration; see
[`edge_bvh_optix_migration_plan.md`](edge_bvh_optix_migration_plan.md)
for the full design and rollout. Files: `src/scene/edge_optix.cu`,
`src/scene/scene_edge_optix.cpp`. Supports `K ≤ 16`.

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

## 5. `trace_segment_chain_visibility` (Phase 2, P0) — ✅ implemented

### Why

EPC reflection-path validation
([`epc.py:642-708`](../../witwin-platform/channel/witwin/deterministic/path/reflection_impl/epc.py#L642))
walks each candidate N-bounce path with a Python `for slot in
range(chain_depth)` loop, issuing one `segment_visible` per segment.
For a chain depth of 4, every path validation is **5 sequential OptiX
launches** (TX→r₁, r₁→r₂, r₂→r₃, r₃→r₄, r₄→RX). Across thousands of
candidate paths × many receivers, this is a real hot spot that survives
Phase 1.

The same chain-of-visibility pattern appears in BDPT diffraction MIS
path validation
([`bdpt_diffraction.py:1683-1694`](../../witwin-platform/channel/witwin/montecarlo/integrators/bdpt_diffraction.py#L1683)).

Folding M-1 segment tests into one OptiX launch per chain removes
(M-2) launch boundaries plus all the Python orchestration per path.

### Result struct

```cpp
// include/rayd/segment_chain_visibility.h
struct SegmentChainVisibility {
    int            chain_count = 0;
    int            max_segments = 0;        // M-1 segments per chain (compile-time bound)
    BoolDetached   all_visible;             // [N] true iff every segment in chain is clear
    IntDetached    first_blocked_segment;   // [N] -1 if all clear, else segment index in [0, M-2]
    IntDetached    first_blocked_prim;      // [N] global prim id of the blocker; -1 if all clear
};
```

`first_blocked_segment` / `first_blocked_prim` are diagnostic outputs.
They are useful for BDPT MIS pruning ("which segment killed this MIS
path") and for shadow-boundary smoothing decisions. Callers that only
need `all_visible` pay two trivial output writes per chain.

### Python API

```python
result = scene.trace_segment_chain_visibility(
    points,                # Array3f [N, M]   chain points (TX, hits..., RX)
    chain_length,          # Int32   [N]      number of segments per chain (≤ M-1)
    ignore_prim_per_segment=None,  # Int32 [N, M-1, K] global prim ids to skip per segment
    active=True,           # Bool [N] or scalar
)
result.all_visible              # Bool  [N]
result.first_blocked_segment    # Int32 [N]
result.first_blocked_prim       # Int32 [N]
```

`M` is the compile-time upper bound on chain length (default 8).
`K` is the maximum ignored prims per segment (default 4 — covers the
prev/next adjacent face pattern already used by channel).
`chain_length` allows ragged chains in one launch: padding entries
beyond `chain_length[n]` are ignored, so candidate paths of varying
depth can share a launch.

### PyTorch wrapper

```python
result = scene.trace_segment_chain_visibility(
    points,                  # torch.Tensor float32 [N, M, 3] CUDA
    chain_length,            # torch.Tensor int32   [N]       CUDA
    ignore_prim_per_segment=None,  # int32 [N, M-1, K] or None
    active=None,
)
# result.all_visible:           torch.Tensor bool   [N]
# result.first_blocked_segment: torch.Tensor int32  [N]
# result.first_blocked_prim:    torch.Tensor int32  [N]
```

### Payload layout

Two u32 payload slots; raygen-local state is held in registers across
the in-raygen segment loop.

```
trace #s: p0 = 0 (clear) → 1 (blocked); p1 = blocker global prim id (when blocked)
```

After each `optixTrace`, raygen reads `p0` and `p1`, updates the
running outputs, and early-exits the chain once any segment is
blocked.

### Programs

Reuses `__anyhit__segment_visibility` and `__miss__segment_visibility`
from §1. Adds a chain-aware closesthit (records blocker prim) and a
new raygen that loops over segments.

```cuda
extern "C" __global__ void __closesthit__segment_chain_visibility()
{
    optixSetPayload_0(1u);  // blocked

    const unsigned int local_prim = optixGetPrimitiveIndex();
    const int instance = (int)optixGetInstanceId();
    const int face_offset = (instance >= 0 && instance < params.n_meshes)
        ? params.face_offsets[instance] : 0;
    optixSetPayload_1((unsigned int)(face_offset + (int)local_prim));
}

extern "C" __global__ void __raygen__segment_chain_visibility()
{
    const unsigned int chain = optixGetLaunchIndex().x;
    if (chain >= (unsigned int)params.n_chains) return;
    if (params.active_mask && !params.active_mask[chain]) {
        params.out_all_visible[chain] = 0u;
        params.out_first_blocked_segment[chain] = -1;
        params.out_first_blocked_prim[chain] = -1;
        return;
    }

    const int n_seg = params.chain_length[chain];
    bool all_clear = true;
    int first_blocked = -1;
    int first_blocked_prim = -1;

    for (int s = 0; s < n_seg; ++s) {
        if (!all_clear) break;  // early exit once chain is blocked

        float3 a, b;
        load_chain_point(chain, s,     a);
        load_chain_point(chain, s + 1, b);

        float3 d = b - a;
        const float len = sqrtf(dot3(d, d));
        if (len < kMinSegLen) continue;  // degenerate, treat as clear
        d = (1.0f / len) * d;
        const float3 origin = a + kRayBias * d;
        const float  tmax   = fmaxf(len - 2.0f * kRayBias, 0.0f);

        // params.ignore_prim_per_segment indexing uses (chain, segment_slot, k);
        // anyhit reads the slot via a per-segment offset injected as ray-payload-input.

        unsigned int hit_flag = 0u;
        unsigned int blocker_prim = 0xFFFFFFFFu;
        optixTrace(params.handle, origin, d, kRayTMin, tmax, 0.0f,
                   255u, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   0, 1, 0,
                   hit_flag, blocker_prim);

        if (hit_flag != 0u) {
            all_clear = false;
            first_blocked = s;
            first_blocked_prim = (int)blocker_prim;
        }
    }

    params.out_all_visible[chain] = all_clear ? 1u : 0u;
    params.out_first_blocked_segment[chain] = first_blocked;
    params.out_first_blocked_prim[chain] = first_blocked_prim;
}
```

Implementation note on per-segment ignore indexing: anyhit needs to
know which segment slot it is in to look up the right ignore subset.
Pass the segment slot index as a payload input register (one extra
slot reserved for the segment index when calling `optixTrace`); anyhit
uses it for indexing `params.ignore_prim_per_segment`.

### AD / JIT

Same as kernel 1: AD-safe but not AD-required (channel calls all
visibility chains in detached contexts). JIT discipline is mandatory —
must be callable inside `dr.syntax` symbolic loops without forcing
materialization. Implemented now for `ignore_prim_per_segment` empty;
non-empty ignore lists still use the custom anyhit native fallback.

### Channel call sites — drop-in points

| Channel file | Line | Current pattern |
|---|---|---|
| `deterministic/path/reflection_impl/epc.py` | 642-708 | Python `for slot in range(chain_depth)` loop, one `segment_visible` per segment, conjunction with `& valid` |
| `montecarlo/integrators/bdpt_diffraction.py` | 1683-1694 | Two sequential `segment_visible` with shared ignore_prim — folds into chain of length 2 |
| `montecarlo/integrators/bdpt_diffraction.py` (`_validate_reflection_chain` helper) | — | similar N-segment validation pattern |

Drop-in rewrite for EPC:

```python
# Before — N+1 OptiX launches per path candidate
prev_point = tx_pos
for slot, hit_p in enumerate(hit_points):
    ignore = [scene.triangle_group_id(prim_indices[slot]), ...]
    valid = valid & scene.segment_visible(prev_point, hit_p,
                                          ignore_surface_group_idx=tuple(ignore))
    prev_point = hit_p
valid = valid & scene.segment_visible(prev_point, target_pos, ...)

# After — 1 OptiX launch per path candidate
chain_points = stack_points(tx_pos, hit_points, target_pos)   # [N, M, 3]
chain_length = ...                                            # [N]
ignore_table = build_ignore_table(prim_indices, target_groups) # [N, M-1, K]
result = scene.trace_segment_chain_visibility(
    chain_points, chain_length,
    ignore_prim_per_segment=ignore_table,
)
valid = valid & result.all_visible
```

For BDPT, replaces the explicit two-segment pattern with a chain of
length 2 (or 3, when the source-side segment is folded in).

### Expected speedup

- Chain length M-1 → (M-2)× launch overhead saved per candidate path
- Channel EPC typical depth 3-5 → **3-5×** on the path-validation phase
- BDPT MIS shorter chains → **1.5-2×**

Combined with D1 (already shipped), the entire EPC validation pipeline
collapses from one Python-loop-per-path to one fused kernel call per
launch.

### Files

```
include/rayd/segment_visibility.h                      MOD — SegmentChainVisibility result struct
src/multipath/segment_visibility.cu                    MOD — adds __raygen__segment_chain_visibility,
                                                          blocker-reporting closesthit, per-segment ignores
src/multipath/segment_visibility_host.cpp              MOD — fourth raygen entry in existing pipeline
src/multipath/segment_visibility_host.h                MOD
src/multipath/segment_visibility_params.h              MOD
src/multipath/segment_visibility_ptx.h                 MOD (generated)
src/rayd.cpp                                           MOD
src/scene/scene.cpp                                    MOD — Scene::trace_segment_chain_visibility
include/rayd/scene/scene.h                             MOD
```

---

## 6. Why D3 / M1 do not need a new kernel

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

JIT-wise, the default symbolic `trace_reflections` / `trace_bounces`
path remains lazy because it is built from `Scene.intersect`. The
non-symbolic optimized chain path still uses the custom native raygen
and forces its broad-phase inputs before `optixLaunch`; the trailing
fields follow that path.

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

## 6A. `trace_reflections_accumulating` RF native path — ✅ implemented

This is the dedicated RayD-side Tier 2 fast path for channel's MC
reflection forward accumulation. Unlike `trace_reflections`, it is not a
geometry-only chain recorder. It is a separate native OptiX launch that
traces reflection bounces and accumulates scalar RF reflection power
directly into a receiver grid.

### Public API

```python
result = scene.trace_reflections_accumulating(
    rays,              # RayDetached, [N]
    tx_position,       # Array3fDetached, scalar or [N]
    grid,              # ReflectionAccumulationGrid
    max_bounces,       # int
    material_payload,  # PrimitiveMaterialPayloadDetached, one entry per global primitive
    active=True,       # BoolDetached scalar or [N]
    options=ReflectionAccumulationOptions(),
)

result.reflection_power   # FloatDetached [grid.resolution0 * grid.resolution1]
result.reflection_count   # IntDetached   [grid cell count]
result.wedge_events       # ReflectionWedgeEventBufferDetached
```

`ReflectionAccumulationGrid` is an axis-aligned 2D descriptor:
`axis`, `position`, `coord0_min/max`, `coord1_min/max`,
`resolution0`, and `resolution1`. The implementation projects each
post-bounce ray to that plane, filters out-of-bounds intersections, and
uses device atomics for per-cell `reflection_power` and
`reflection_count`.

`PrimitiveMaterialPayloadDetached` is indexed in global primitive id
space and currently carries `eta_r`, `sigma`, `mu_r`, `gain`, and
`valid`. The kernel evaluates a scalar normal-incidence-style Fresnel
power factor with conductivity support, multiplies by `gain`, applies
Russian roulette if requested, and stops when the path throughput falls
below `stop_threshold`.

### Wedge event ABI

When `options.collect_wedges` is true, the same reflection launch emits
wedge/diffraction candidate events while tracing bounces:

```cpp
struct ReflectionWedgeEventBuffer {
    int capacity;
    Int count;              // total attempted events, may exceed capacity
    Int ray_index;          // [capacity]
    Vector3f hit_points;    // [capacity]
    Vector3f normals;       // [capacity]
    Int prim_id;            // [capacity], global primitive id
    Vector3f directions;    // [capacity], outgoing reflected direction
    Int bounce_depth;       // [capacity], zero-based reflection depth
};
```

Only slots `0:min(count, capacity)` are valid. Overflow is observable
because `count` is incremented even when the slot is outside capacity.
`collect_wedge_prefixes=false` records only the first-bounce candidate;
`true` records every bounce.

### Native / JIT / AD contract

This API is intentionally native-only. It lives beside, not inside, the
default symbolic/JIT `trace_reflections` path:

- Detached inputs run the standalone `optixLaunch` fast path.
- AD inputs throw immediately with a message naming the non-AD native
  fast path.
- There is no silent fallback to channel's AD tape path. AD callers must
  explicitly choose the existing Dr.Jit/Channel AD implementation, or a
  future RayD VJP/JVP/topology-tape API.

This keeps the two implementations clear in code and at the API level:
`trace_reflections` remains the JIT/symbolic chain API, while
`trace_reflections_accumulating` is the RF-native non-AD accumulation
API.

### Files

```text
include/rayd/reflection_accumulation.h
src/multipath/reflection_accumulation_params.h
src/multipath/reflection_accumulation_host.h
src/multipath/reflection_accumulation_host.cpp
src/multipath/reflection_accumulation.cu
src/multipath/reflection_accumulation_ptx.h
src/scene/scene.cpp
src/rayd.cpp
CMakeLists.txt
tests/drjit/test_reflection_accumulation.py
```

### Tier 3 boundary

`trace_diffraction_chain_mc` is not implemented by reusing this scalar
reflection accumulator. A correct Tier 3 kernel needs a separate ABI for
TX → edge... → RX/target chain visibility, per-edge UTD/MIS cache
tables, field payload layout, and receiver accumulation semantics. Until
that ABI is fixed, unsupported control modes should continue to error
explicitly rather than falling back.

---

## 7. Testing Methodology

All tests follow the existing pattern in `tests/drjit/test_geometry.py`:
subprocess-isolated Python scripts that print a single-line JSON
result, asserted by `unittest`. Subprocess isolation avoids OptiX state
leaking between tests.

### 7.1 Correctness reference: brute-force

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

### 7.2 Correctness reference: existing API

For each new kernel, a test compares against the slower but already-
verified existing API:

| New kernel | Reference |
|---|---|
| `trace_segment_visibility` | `Scene.intersect()` in a Python re-fire loop |
| `trace_segment_pair_visibility` | Two separate `trace_segment_visibility` calls |
| `trace_axial_edge_visibility` | Python loop of `trace_segment_visibility` |
| `nearest_edges_topk` | Brute-force scan over all edges via `nearest_edge` filter |
| `trace_reflections_accumulating` | Existing detached reflection trace plus channel-side reference accumulation for scalar RF tests |

### 7.3 Test scene matrix

Per kernel, the same three scene fixtures used in
`tests/baseline_cases.py`:

- `single_triangle` — one triangle, sanity check
- `two_quads_perpendicular` — multi-mesh, multi-prim, exercise instance ids
- `cornell_box_decimated` — ≥100 triangles, multi-mesh, dense enough for
  meaningful traversal

### 7.4 Per-kernel test list

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

### 7.5 Integration tests (downstream witwin)

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

### 7.6 Microbenchmarks

Add to `tests/benchmark_*.py`:

- `benchmark_segment_visibility.py` — scene size sweep × N sweep ×
  ignore-K sweep, comparing against brute-force re-fire.
- `benchmark_axial_edge_visibility.py` — N_samples sweep.
- `benchmark_edge_topk.py` — K sweep, integrated into existing
  `benchmark_edge_queries.py`.

Output to `docs/performance_benchmark.json` so we can track regressions
the same way the existing benchmark does.

### 7.7 JIT discipline tests (mandatory per kernel)

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

### 7.8 AD-safety tests (per kernel — light)

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

## 8. Implementation Order and Honest Speedup Calibration

### 8.1 Status snapshot

Phase 1 is shipped. The original implementation order (D1 → trace_reflections
adjustment → kernel 2 → kernel 4 → kernel 3 → deferred D2) was followed
roughly as planned, with kernel 4 absorbed into the broader edge-BVH
OptiX migration ([`edge_bvh_optix_migration_plan.md`](edge_bvh_optix_migration_plan.md)).

```
✅ §1  trace_segment_visibility               (src/multipath/segment_visibility.cu)
✅ §2  trace_segment_pair_visibility          (same module)
✅ §3  trace_axial_edge_visibility            (same module)
✅ §4  nearest_edges_topk                     (src/scene/edge_optix.cu, K ≤ 16)
✅ §5  trace_segment_chain_visibility         (src/multipath/segment_visibility.cu)
✅ §6  trace_reflections trailing fields      (src/multipath/reflection_trace.cu)
✅ §6A trace_reflections_accumulating          (src/multipath/reflection_accumulation.cu)
✅ §6A reflection wedge event buffer           (same module)
✅     primitive ignore tables wider than 8    (src/multipath/segment_visibility.cu)
⏸ D2  trace_diffraction_chain_mc              (deferred; field/cache ABI not fixed)
```

### 8.2 Phase 2 sequence

1. **§6 trailing fields** first. Smallest change (one extra `optixTrace`
   inside the existing `trace_reflections` raygen), unblocks the
   channel-side D3 / M1 flatten-loop refactors. No new module.

2. **§5 `trace_segment_chain_visibility`**. Reuses anyhit/miss from
   the already-shipped segment_visibility module; only the new raygen,
   the chain-aware closesthit, and the params struct are new code.
   Lands the EPC and BDPT path-validation drop-ins.

Both items are independent and could land in either order. Recommended
order above optimises for channel-side refactor unblocking
(D3 / M1 → EPC validation).

### 8.3 Honest speedup calibration

A retrospective on the per-kernel speedup estimates given when this
plan was first written:

| Kernel | Original estimate | Honest re-estimate | Why the gap |
|---|---|---|---|
| §1 `trace_segment_visibility` | 3–8× | **3–8× (confirmed)** | Replaces an unfused Python re-fire loop. Real win. |
| §2 `trace_segment_pair_visibility` | ~2× | **1.5–2× (confirmed)** | Two ray casts per launch index — modest by construction. |
| §3 `trace_axial_edge_visibility` | 3–5× | **2–4×** | Speedup capped by `n_samples` (5 in channel default). |
| §4 `nearest_edges_topk` | 3–10× over 18-probe heuristic | **2–4×** plus recall gain | Most win is in **recall guarantee** (exact top-K vs heuristic), not raw speed. |
| §5 `trace_segment_chain_visibility` | n/a (Phase 2) | **3–5×** on EPC, 1.5–2× on BDPT | Removes (M-2) launch boundaries per chain. |
| §6 trailing fields (channel D3 / M1 refactor) | 2–5× | **2–3×** | The win is mostly already captured by Dr.Jit `mode='symbolic'`; this just removes the residual per-bounce Dr.Jit/OptiX boundary. |
| D2 `trace_diffraction_chain` (BFS mode) | 3–10× | **5–20% (much smaller)** | BFS state-array materialization dominates the cost; fusion cannot remove it. |
| D2 (sampled-path mode) | — | 2–4× over Dr.Jit-symbolic | But this is an *algorithm change*, not just fusion. |

End-to-end channel speedup once all of Phase 1 + Phase 2 are deployed
and the channel-side refactors land: **roughly 1.5× wall-clock per
frame** for the typical deterministic + MC workload. Real but not
transformative.

Reaching tier-3 live digital twin (30-100 ms frame budget for moving
endpoints) requires more than fusion — it requires algorithm-level
work that is out of scope here: switching higher-order diffraction to
sampled MC paths, tabulating UTD coefficients, compacting per-state
memory footprint. See channel doc 24 §8.

### 8.4 D2 deferral rationale

Original §7 listed `trace_diffraction_chain` as a "(deferred)" item with
the implication that magnitudes would justify it. Honest analysis (channel
doc 24 §7 follow-up + the §8.3 calibration above) puts it at 5–20% in BFS
mode. Two changes to the deferral framing:

- Do **not** plan D2 as the natural successor to Phase 2. Its expected
  magnitude is smaller than §5.
- Re-evaluate only after Phase 2 + channel-side refactors land and a
  profile run still shows BFS orchestration dominating. The most
  likely correct answer at that point is "change the algorithm" not
  "write another fused kernel."

### 8.5 Effort

Phase 2 items: 1–2 focused days each given the existing
`segment_visibility.cu` + `reflection_trace.cu` templates. The CMake /
PTX / nanobind glue from Phase 1 carries over directly.

### 8.6 Channel symbolic-loop reality check

The dual-backend dispatch (see "Backend Dispatch" earlier in this
doc) raises a question: is the "symbolic loop + ignore_prim_ids" edge
case ever hit by channel today, or is it purely hypothetical?

Audit of every channel call site that goes through
`Scene.segment_visible` / `Scene.trace_segment_*`:

| Call site | Inside `dr.syntax` symbolic loop? | Uses `ignore_prim_idx`? | Dispatch | Edge case? |
|---|---|---|---|---|
| [`builders.py:244`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L244) | ❌ batch eval | ✅ adjacent faces | native | safe |
| [`builders.py:484`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/builders.py#L484) | ❌ batch eval | ✅ 4 ignores | native | safe |
| [`forward.py:261-268`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/forward.py#L261-L268) | ❌ | ✅ | native | safe |
| [`postprocessing.py:269,317`](../../witwin-platform/channel/witwin/deterministic/path/diffraction_impl/postprocessing.py#L269) | ❌ | ✅ | native | safe |
| [`epc.py:656-705`](../../witwin-platform/channel/witwin/deterministic/path/reflection_impl/epc.py#L656) | ❌ Python `for` | ✅ surface groups | native | safe (chain candidate for §5) |
| [`bdpt_diffraction.py:679-681`](../../witwin-platform/channel/witwin/montecarlo/integrators/bdpt_diffraction.py#L679) | ❌ | (varies) | varies | safe |
| [`bdpt_diffraction.py:1683-1689`](../../witwin-platform/channel/witwin/montecarlo/integrators/bdpt_diffraction.py#L1683) | ❌ | ✅ | native | safe |
| [`reflection.py:535-552`](../../witwin-platform/channel/witwin/montecarlo/path/reflection.py#L535) MC reflection main loop | ✅ `mode='symbolic'` | uses `scene.ray_intersect` only (not `segment_visible`) | jit (via Dr.Jit-op `Scene.intersect`) | n/a |
| [`montecarlo/path/los.py:91`](../../witwin-platform/channel/witwin/montecarlo/path/los.py#L91) | ❌ | ❌ | jit | safe |
| [`montecarlo/path/postprocessing.py:159`](../../witwin-platform/channel/witwin/montecarlo/path/postprocessing.py#L159) | ❌ | ❌ | jit | safe |

**Net status**: zero channel call sites hit the edge case ("inside
symbolic loop + ignore_prim_ids"). The dispatch design is correct for
all current usage; the edge case is reserved for a hypothetical future
caller, with two extension paths documented in the Backend Dispatch
section above.

If channel ever adds a symbolic-loop call that needs ignores, this
table will be the first place that needs updating, and option 1
(Dr.Jit re-fire for ignores) becomes the natural action item.

---

## 9. Open Questions

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
