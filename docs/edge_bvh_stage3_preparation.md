# Edge BVH Stage 3 Preparation

> Updated: 2026-03-30
> Scope: `Node data packing` and `PLOC replacing LBVH`

## Context

- Numeric benchmark history remains in [docs/edge_bvh_optimization_log.md](E:\Code\RayDi\docs\edge_bvh_optimization_log.md).
- Only `stage2_clean_*` rows are valid no-fallback ablations.
- Stage 2 conclusions below are measured relative to `stage2_clean_baseline`.

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
- `compaction_mode = host_upload_exact`
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

### 2. PLOC Replacing LBVH

Goal:

- Add a `PLOC` build algorithm as a first-class alternative to `LBVH`, benchmarked explicitly and compared against the Stage 3 baseline.

Constraints:

- No hidden fallback to `LBVH`.
- If `PLOC` is selected before implementation is complete, the build must fail fast.
- Treelet, compaction, and query-path comparisons must use the same fixed benchmark matrix.

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

| Stage | Build algorithm | Node layout |
| --- | --- | --- |
| `stage3_baseline` | `lbvh` | `scalar_arrays` |
| `stage3_node_packing` | `lbvh` | `packed` |
| `stage3_ploc` | `ploc` | `scalar_arrays` |
| `stage3_ploc_packed` | `ploc` | `packed` |
| `stage3_final` | winner from above | winner from above |

## Implementation Guardrails

- All benchmarked modes must be explicit runtime selections.
- Unsupported combinations must raise an error; they must not downgrade to another path.
- Build and query acceptance should continue using the same fixed benchmark harness and JSON artifacts.
- `stage3_final` should only be recorded after point/ray correctness, mask switching, refit, and gradient tests pass.

## Immediate Engineering Tasks

1. Introduce packed node storage without changing public query APIs.
2. Keep the current scalar-array layout available as the explicit Stage 3 baseline.
3. Add a PLOC build path behind the explicit `build_algorithm=ploc` selector.
4. Preserve `host_upload_exact` compaction semantics while bringing up packed nodes.
5. Re-run the clean stage matrix after each substantial milestone.
