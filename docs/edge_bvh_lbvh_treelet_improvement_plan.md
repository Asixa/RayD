# Edge BVH LBVH + Treelet Improvement Plan

> Updated: 2026-03-30
> Scope: stable default path only

## Current Decision

- Stable default remains `build_algorithm=lbvh` with `post_build_strategy=gpu_treelet`.
- `PLOC` remains available only as an experimental path and its implementation now lives under `src/scene/experimental/`.
- Serial re-evaluation still says the default path is the best overall tradeoff on this repository's authoritative workload.

## Why This Is Still The Default

Current serial measurements already show that `LBVH + gpu_treelet` wins on:

- single-scene build latency
- full-scene point query throughput
- full-scene finite and infinite ray throughput
- dynamic update / restore behavior

The important implication is that the next wins are no longer "replace LBVH with a smarter builder". The remaining headroom is mostly implementation overhead:

- sparse `refit()` is no longer the dominant edge-BVH cost after Workstream 1
- mask changes no longer rebuild, but masked queries now trade rebuild cost for filtering/culling overhead on the full tree
- `build_edge_bvh_gpu()` still performs host topology round-trips for levelization and treelet scheduling
- traversal is still a gather-heavy Dr.Jit path over scalar arrays
- a GPU-side treelet preparation prototype now exists, but it remains experimental because it has not yet beaten the stable host-prepared path in measured build latency

## OptiX Sync Follow-Up

After Workstream 1, the next bottleneck was no longer edge-BVH refit itself. It was `Scene::sync()` spending most of its remaining time in the OptiX scene update path.

Two follow-up changes now exist:

- `OptixScene::sync()` no longer rebuilds or reuploads every instance slot on every update. It patches only dirty instance records and keeps the cached instance buffer alive across syncs.
- Dynamic GAS / IAS build preferences are now configurable with:
  - `RAYD_OPTIX_DYNAMIC_GAS_PREFERENCE={auto,fast_trace,fast_build,relaxed}`
  - `RAYD_OPTIX_DYNAMIC_IAS_PREFERENCE={auto,fast_trace,fast_build,relaxed}`

Measured on the verified `8x8` dynamic-mesh calibration scene (`32x32` grid per mesh, update one mesh per sync), the current tradeoff is:

- `gas=fast_trace, ias=fast_trace`: best batch query throughput among the tested combinations
- `gas=relaxed, ias=relaxed`: best build latency, but about `12%` slower batch query throughput than `fast_trace/fast_trace`
- `gas=fast_trace, ias=relaxed`: smaller query regression than fully relaxed while still keeping the IAS path less conservative for dynamic updates

Artifacts:

- `artifacts/benchmarks/optix_dynamic_flags_benchmark.json`
- `artifacts/benchmarks/optix_query_dynamic_ff.json`
- `artifacts/benchmarks/optix_query_dynamic_fr.json`
- `artifacts/benchmarks/optix_query_dynamic_rr.json`

Current default therefore remains conservative on GAS quality and only relaxes IAS:

- dynamic GAS default: `fast_trace`
- dynamic IAS default: `relaxed`

The code also contains an experimental static/dynamic OptiX split path behind `RAYD_OPTIX_SPLIT_MODE={off,on,auto}`. The mixed-scene calibration (`63` static meshes + `1` dynamic mesh) showed:

- split `on` did not materially reduce `sync()` time on the measured workload
- split `on` increased batch query time by about `2.3x`
- split `on` therefore is not enabled by default

Artifacts:

- `artifacts/benchmarks/optix_split_benchmark.json`

Default behavior now treats an unset `RAYD_OPTIX_SPLIT_MODE` as `off`, and `auto` is also biased to the single-scene path until a measured heuristic exists that pays for the query tax.

## Priority Order

1. Query-path memory layout cleanup
2. Optional masked-query fallback policy for cases where Workstream 2 filtering regresses traversal too far
3. Revisit GPU treelet preparation only if a clearer build-latency win emerges
4. Treelet heuristic retuning only after the first three are done

## Workstream 1: Dirty-Ancestor-Only Refit

### Current bottleneck

- `Scene::ensure_edge_bvh_ready()` already distinguishes rebuild from refit.
- `SceneEdge::refit()` still recomputes internal nodes level-by-level across the whole tree, not just the ancestors touched by dirty leaves.

This makes sparse vertex edits pay for untouched internal nodes and is the most direct reason `sync()` still has obvious room.

### Implementation plan

1. Keep the existing `primitive_leaf_node_` map as the leaf entry point.
2. For each dirty edge range, gather the touched leaf nodes.
3. Walk parent links upward and mark the exact internal ancestors that need recomputation.
4. Build per-level worklists only for marked nodes.
5. Run the existing merge/update kernels on those compacted worklists instead of all nodes at that height.

### Design details

- Use a device-side mark array plus prefix-sum compaction, not host-side `std::vector` accumulation.
- Keep parent links as persistent BVH state after build/compaction so refit does not have to reconstruct them.
- Maintain the current bottom-up level ordering so the actual merge kernels stay simple.
- For packed mode, reuse the same dirty-internal worklists and only repack touched nodes after scalar bounds are updated.

### Expected impact

- Largest remaining `sync()` win for sparse geometry edits.
- No query regression if the touched-node sets are correct.
- Limited risk because the math is unchanged; only the scheduling granularity changes.

### Current prototype status

- A first dirty-ancestor prototype now exists in code, with `RAYD_EDGE_BVH_REFIT_STRATEGY={auto,full,dirty_ancestors}` for calibration.
- The prototype now keeps parent links on-device, performs dirty-ancestor marking on the GPU, and compacts dirty per-level worklists fully on the GPU before refitting internal nodes.
- On the verified sparse-update calibration scenes, the dirty path now beats the old full-level GPU scan on `edge_refit_ms`.
- `auto` therefore uses the dirty path again, but only under a conservative sparsity heuristic; `full` and `dirty_ancestors` remain available for calibration.

### Acceptance criteria

- Full-scene rebuild behavior unchanged.
- Sparse dirty-range refit becomes materially cheaper than current full-level refit.
- Existing geometry, mask, and gradient tests still pass.

## Workstream 2: Separate Mask Changes From Full Rebuild

### Current bottleneck

- `Scene::ensure_edge_bvh_ready()` routes any `mask_dirty_` case to a fresh `build()`.

That is correct but too expensive. Mask flips are not topology creation. They are visibility changes.

### Implementation plan

1. Keep the stable full-scene LBVH topology as the primary structure.
2. Add a query-time active-edge filter so masked primitives can be skipped without rebuilding topology.
3. Add per-leaf or per-node active counts so completely inactive subtrees can be culled cheaply.
4. Only rebuild topology if mask churn is so extreme that the filtered path becomes consistently worse than rebuild.

### Staged rollout

- Stage A: leaf-level active filtering only. Lowest engineering risk, fastest bring-up.
- Stage B: propagate active counts upward so traversal can skip empty subtrees.
- Stage C: optionally add a heuristic that chooses rebuild only when active density drops below a measured threshold.

### Expected impact

- Removes the biggest avoidable cost in "stride to full" style workflows.
- Lets `sync()` mean "update" instead of "quietly rebuild the BVH" for mask-only changes.

### Main risk

- Query throughput can regress if inactive primitives stay physically present but are only filtered at leaf time.

That is why the active-count propagation step should follow quickly after the initial bring-up.

### Current implementation status

- Stage A and Stage B are now implemented in the default path.
- `Scene::ensure_edge_bvh_ready()` no longer routes `mask_dirty_` to `build()`. It now updates the active mask/count state on the existing full-scene BVH and only uses refit when there are actual dirty edge ranges.
- `SceneEdge` now keeps per-primitive active flags plus per-node active primitive counts so queries can skip empty subtrees without topology rebuild.
- Geometry regression coverage still passes, including the mask-specific tests in `tests.drjit.test_geometry`.

### Measured result

Artifact:

- `artifacts/benchmarks/edge_bvh_workstream2_pressure.json`

Compared against the old rebuild-based baseline in `artifacts/benchmarks/edge_bvh_stages/serial_reeval_lbvh_baseline/pressure.json` on the same `32x32 x 8x8` pressure scene:

- `checker_tiles` mask flip: `edge_refit_ms` dropped from `13.30 ms` to `0.42 ms`; restore to full dropped from `25.28 ms` to `0.44 ms`
- `boundary_only` mask flip: `2.04 ms` to `0.45 ms`; restore to full `24.50 ms` to `0.44 ms`
- `stride_sparse` mask flip: `2.62 ms` to `0.45 ms`; restore to full `23.64 ms` to `0.41 ms`
- `empty` restore to full: `24.18 ms` to `0.46 ms`

This means the original Workstream 2 sync goal is met. Mask-only `sync()` no longer quietly rebuilds the BVH.

### Observed tradeoff

- Masked query throughput is now workload-dependent because inactive primitives still contribute to the stored subtree bounds unless a subtree becomes fully empty.
- Some masks regressed after switching from "rebuild a compact masked BVH" to "filter inside the full BVH":
  - `checker_tiles` finite ray: `25.65 ms` to `32.06 ms`
  - `boundary_only` finite ray: `19.79 ms` to `65.96 ms`
  - `interior_only` finite ray: `15.21 ms` to `18.34 ms`
- Some masks improved because the old rebuild path produced a poor masked tree:
  - `stride_sparse` finite ray: `75.25 ms` to `24.59 ms`

### Conclusion

- Workstream 2 is complete for the intended `sync()` objective.
- It should remain enabled because the restore/full-mask path win is too large to ignore.
- A future Stage C should be a selective fallback or alternate masked-query path, not a return to unconditional rebuilds on every mask flip.

## Workstream 3: Remove Host Round-Trip From Treelet Preparation

### Current bottleneck

`build_edge_bvh_gpu()` currently does all of the following on the host when treelets are enabled:

- copy topology back from the GPU
- build node heights / level groups on the host
- build subtree leaf counts on the host
- upload recompute and optimize schedules back to the GPU

This is pure build overhead. It does not improve final tree quality by itself.

### Implementation plan

1. Compute node heights on the device once raw LBVH topology is available.
2. Compute subtree leaf counts on the device.
3. Build two compacted worklists on the device:
   - recompute roots
   - optimize roots
4. Keep the existing flat-level schedule idea, but materialize it entirely on the GPU.
5. Remove host synchronization points before treelet optimization.

### Recommended order

- First move subtree-leaf counting to the GPU.
- Then move per-level root classification to the GPU.
- Only after that replace host levelization if it still matters in measurements.

This reduces risk and preserves a checkpoint after each step.

### Expected impact

- Best improvement for `build()` latency on large scenes.
- No direct effect on query quality.

### Current prototype status

- A GPU-side prototype now computes treelet subtree leaf counts and node heights on-device, then builds flat recompute/optimize schedules on the GPU.
- The prototype is kept behind `RAYD_EDGE_BVH_EXPERIMENTAL_GPU_TREELET_PREP=1`.
- The stable default does not enable it, because the current measurements do not justify replacing the older host-prepared schedule.

### Measured result

Artifacts:

- `artifacts/benchmarks/edge_bvh_workstream3_single_mesh.json`
- `artifacts/benchmarks/edge_bvh_workstream3_pressure.json`

Compared against `artifacts/benchmarks/edge_bvh_stages/serial_reeval_lbvh_baseline/`:

- single `192x192` build: `43.96 ms` baseline vs `50.54 ms` prototype
- single point query: `4.29 ms` vs `4.73 ms`
- single finite ray query: `11.25 ms` vs `11.72 ms`
- pressure build: `139.63 ms` vs `143.37 ms`
- pressure full finite ray query: `15.53 ms` vs `16.57 ms`

The prototype no longer pays the large host topology copy and schedule upload, but the new GPU-side schedule generation still adds enough overhead that the end result is slightly slower overall on the authoritative workload.

### Conclusion

- Workstream 3 is implemented as an experimental calibration path, not an accepted default optimization.
- The default `LBVH + gpu_treelet` path remains on the older host-prepared schedule because it is still faster in measured end-to-end results.
- If this line is revisited, the next iteration should reduce prototype kernel count and temporary allocations before attempting another default-path swap.

## Workstream 4: Query-Path Layout Cleanup

### Current bottleneck

- The default query path still gathers from scalar node arrays.
- The current packed mode is not yet the sole source of truth and still carries duplication / maintenance overhead.
- Finite-ray traversal is still a Dr.Jit while-loop with many gathers and branchy child ordering.

### Implementation plan

1. Finish a packed-only node layout instead of "scalar source + packed mirror".
2. Keep the existing leaf encoding and exact-result re-gather semantics unchanged.
3. Measure packed-only traversal before considering any custom CUDA traversal kernel.
4. Only if packed-only still does not move the needle enough, prototype a dedicated CUDA finite-ray traversal kernel.

### Why this is lower priority

The current serial data suggests the easiest large wins are still in `sync()` and `build()`. Query quality is already good enough that traversal layout tuning should come after those cheaper wins.

## Workstream 5: Treelet Heuristic Tuning

### Current position

This should not be the first lever. On the authoritative full-scene workload, `LBVH + gpu_treelet` is already producing a very strong tree.

### What is still worth testing

- tighten the condition for which subtrees enter treelet optimization
- add a cheap overlap proxy and only optimize roots that score poorly
- reduce wasted recompute work around tiny subtrees

### What is not worth prioritizing

- larger and larger treelets
- SAH-heavy rebuild stages in the default path
- replacing LBVH as the near-term baseline

Those directions add complexity much faster than they add value on the current workload.

## Verification Matrix

Every milestone should be rechecked on the same serial benchmark discipline:

- single build
- single point query
- single finite ray
- single infinite ray
- single sync
- pressure build
- pressure full finite ray
- pressure full infinite ray
- pressure stride finite ray

Additionally, every milestone should re-run:

- geometry correctness tests
- mask switching checks
- refit correctness checks
- gradient-sensitive nearest-edge tests

## Recommended Execution Sequence

1. Implement dirty-ancestor-only refit.
2. Add mask filtering plus empty-subtree skipping.
3. Move subtree counts and treelet root classification fully onto the GPU.
4. Re-measure.
5. Only then decide whether packed-only traversal is still worth doing before any custom traversal kernel work.

## Stop Conditions

Pause optimization if either of these becomes true:

- `sync()` is no longer a meaningful bottleneck on the serial benchmark suite.
- query time stops moving after layout cleanup and the remaining cost is dominated by exact edge-distance math rather than BVH overhead.

At that point, the default path is "good enough" and further builder complexity is unlikely to pay back.
