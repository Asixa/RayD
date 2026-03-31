# Edge BVH Stage 3 Preparation

> Updated: 2026-03-31
> Scope: `Node data packing` and `PLOC replacing LBVH`

## Context

- Numeric benchmark history remains in [docs/edge_bvh_optimization_log.md](E:\Code\RayDi\docs\edge_bvh_optimization_log.md).
- The auto-generated percentages in that log are anchored to the repository-wide earliest baseline; the serial Stage 3 comparison table below is the current authoritative view for `LBVH` vs `PLOC`.
- Only `stage2_clean_*` rows are valid no-fallback ablations.
- Stage 2 conclusions below are measured relative to `stage2_clean_baseline`.

## Benchmark Discipline

- All speed measurements must be collected with serial execution only: one benchmark process at a time on the machine/GPU.
- Do not launch multiple stage, micro, single-mesh, or pressure benchmark processes concurrently, even if they target different artifact files.
- If any benchmark was run under concurrent benchmark load, its build/query/sync timings are invalid and must be rerun serially before being used in analysis or copied into the optimization log.
- This rule applies to all future Stage 3 comparisons and any follow-up PLOC/LBVH ablation work.

## Stage 2 Status

| Stage | Single build | Pressure build | Pressure stride finite ray | Decision |
| --- | ---: | ---: | ---: | --- |
| `stage2_clean_stream_overlap` | `+2.5%` | `-5.6%` | `+1.3%` | Keep available, not the main win |
| `stage2_clean_treelet_schedule` | `-3.2%` | `-6.5%` | `-10.6%` | Keep |
| `stage2_clean_finalize_levels` | `+1.2%` | `-6.7%` | `+3324.2%` | Reject |
| `stage2_clean_gpu_compaction` | `+1.1%` | `-0.1%` | `+4.8%` | Reject |
| `stage2_clean_host_exact` | `+2.1%` | `-2.5%` | `-81.5%` | Keep for baseline quality |
| `stage2_clean_final` | `-2.1%` | `-8.4%` | `-81.4%` | Stage 3 baseline |

## Stage 3 Baseline

The baseline for Stage 3 work is:

- `post_build_strategy = gpu_treelet`
- `build_stream_mode = overlap`
- `finalize_mode = atomic`
- `treelet_schedule_mode = flat_levels`
- `compaction_mode = host_upload_raw`
- `build_algorithm = lbvh`
- `node_layout_mode = scalar_arrays`

This keeps the best verified Stage 2 path while preserving the no-fallback rule.

## Stage 3 Scope

### 1. Node Data Packing

Goal:

- Replace the current scalar-array traversal layout with a packed node representation that improves gather locality.

Constraints:

- No behavioral fallback in benchmark runs.
- Query result fields, leaf encoding semantics, and AD re-gather behavior must remain unchanged.
- Packed layout should be introduced as an explicit runtime mode, not auto-enabled by scene characteristics.

Target benchmark stage:

- `stage3_node_packing`

Current status:

- Implemented and benchmarkable through `node_layout_mode=packed`
- No fallback: selecting `packed` now runs the packed traversal/refit layout, not the scalar-array query path
- Current implementation keeps scalar node arrays as the build/refit source of truth and maintains packed node buffers in parallel for query-time access
- This is correct and benchmark-clean, but not yet a performance winner

### 2. PLOC Replacing LBVH

Goal:

- Add a `PLOC` build algorithm as a first-class alternative to `LBVH`, benchmarked explicitly and compared against the Stage 3 baseline.

Constraints:

- No hidden fallback to `LBVH`.
- If `PLOC` is selected before implementation is complete, the build must fail fast.
- Treelet, compaction, and query-path comparisons must use the same fixed benchmark matrix.
- The current implementation is a real GPU `PLOC` path with GPU matching/finalization and explicit runtime mode selection.
- `PLOC` may be benchmarked with either `post_build_strategy = none` or `post_build_strategy = gpu_treelet`; unsupported combinations must still fail fast.

Target benchmark stages:

- `stage3_ploc`
- `stage3_ploc_packed`

## Benchmark Matrix for Stage 3

Use the existing fixed scenarios from the stage benchmark harness.

Recommended stage labels:

1. `stage3_baseline`
2. `stage3_node_packing`
3. `stage3_ploc`
4. `stage3_ploc_packed`
5. `stage3_final`

Expected mode combinations:

| Stage | Build algorithm | Post-build strategy | Node layout |
| --- | --- | --- | --- |
| `stage3_baseline` | `lbvh` | `gpu_treelet` | `scalar_arrays` |
| `stage3_node_packing` | `lbvh` | `gpu_treelet` | `packed` |
| `stage3_ploc` | `ploc` | `none` | `scalar_arrays` |
| `stage3_ploc_packed` | `ploc` | `none` | `packed` |
| `stage3_final` | winner from above | winner from above | winner from above |

Current serial re-evaluation prioritized scalar-array `PLOC` first and added an extra `serial_reeval_ploc_gpu_treelet` row because raw scalar `PLOC` is still not competitive enough to justify a packed-layout rerun yet.

## Serial Re-evaluation

These rows are the current authoritative comparison because they were collected serially, one benchmark process at a time. Any earlier timings collected under concurrent benchmark load are invalid for speed analysis and should not be used.

### Valid serial stages

| Stage | Build algorithm | Post-build strategy | Compaction | Node layout |
| --- | --- | --- | --- | --- |
| `serial_reeval_lbvh_baseline` | `lbvh` | `gpu_treelet` | `host_upload_raw` | `scalar_arrays` |
| `serial_reeval_ploc_none` | `ploc` | `none` | `gpu_emit` | `scalar_arrays` |
| `serial_reeval_ploc_gpu_treelet` | `ploc` | `gpu_treelet` | `gpu_emit` | `scalar_arrays` |

### Current serial results

Absolute measurements from the serial runs:

| Stage | Single build | Single point | Single finite ray | Single infinite ray | Single sync | Pressure build | Pressure full finite | Pressure full infinite | Pressure stride finite | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `serial_reeval_lbvh_baseline` | `43.96 ms` | `4.29 ms` | `11.25 ms` | `9.59 ms` | `1.74 ms` | `139.63 ms` | `15.53 ms` | `13.84 ms` | `75.25 ms` | Current winner |
| `serial_reeval_ploc_none` | `91.41 ms` | `40.02 ms` | `101.13 ms` | `100.71 ms` | `3.07 ms` | `216.62 ms` | `82.89 ms` | `73.55 ms` | `50.47 ms` | Not competitive |
| `serial_reeval_ploc_gpu_treelet` | `97.73 ms` | `23.86 ms` | `66.33 ms` | `55.14 ms` | `2.59 ms` | `218.64 ms` | `50.84 ms` | `42.72 ms` | `43.09 ms` | Improved over raw PLOC, still not competitive |

Interpretation:

- `LBVH + gpu_treelet` remains the clear default on serial measurements. It is faster than both `PLOC` variants on build, full-scene point/ray query, and mask restore refit.
- `PLOC + none` is still not viable. Serial results show roughly `2.1x` single-build slowdown and `9.0x` single finite-ray slowdown relative to the serial LBVH baseline.
- `PLOC + gpu_treelet` materially improves over raw `PLOC`, but it still trails LBVH by roughly `2.2x` on single build and `5.9x` on single finite-ray query.
- `PLOC` does look better than LBVH on the pressure `stride_sparse` query subset, but that isolated mask case does not overturn the full-scene result because build, full-query throughput, and refit all remain much worse.
- The next optimization target for `PLOC` should continue to be overlap reduction and GPU-side matching quality, not CPU heuristics or tree-height-only tuning.

## Implementation Guardrails

- All benchmarked modes must be explicit runtime selections.
- Unsupported combinations must raise an error; they must not downgrade to another path.
- Build and query acceptance should continue using the same fixed benchmark harness and JSON artifacts.
- `stage3_final` should only be recorded after point/ray correctness, mask switching, refit, and gradient tests pass.

## Immediate Engineering Tasks

1. Introduce packed node storage without changing public query APIs.
2. Keep the current scalar-array layout available as the explicit Stage 3 baseline.
3. Keep `build_algorithm=ploc` fail-fast for unsupported combinations and benchmark both explicit `none` and `gpu_treelet` post-build modes.
4. Preserve explicit compaction-mode selection while bringing up packed nodes and PLOC.
5. Re-run the clean stage matrix after each substantial milestone.
6. Treat any future PLOC work as a second-pass optimization problem, not as a near-term default candidate.
