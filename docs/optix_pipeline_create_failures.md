# OptiX Pipeline Create Failures

Date: 2026-05-25

This note records the root cause and guardrails for the `optixPipelineCreate(multipath)` failures fixed around reflection EPC, diffraction accumulation, diffraction path export, and AD reflection tracing.

## Summary

The failures were not caused by bad scene inputs, Python API changes, or missing fallback behavior. They were OptiX pipeline link/compile failures that only surfaced on cold pipeline creation.

There were two related but distinct failure modes:

- RayD's multipath OptiX pipelines were linked with the global production exception flags (`0`). On the verified Windows/RTX 5080 target, cold `optixPipelineCreate()` failed for several otherwise valid multipath program groups unless multipath used dedicated OptiX exception flags (`11`).
- Existing tests often missed the issue because a pipeline had already been warmed, or because they covered one public API at a time instead of fresh subprocess cold creation for every native multipath entry point.

Instruction count, trace-call count, and OptiX stderr pipeline statistics were useful risk signals, but they were not the root cause by themselves. A small segment-visibility pipeline with one trace site and only 191 entry instructions still failed with exception flags `0`, and passed with multipath exception flags `11`.

## Symptoms

Typical Python-side exception:

```text
RuntimeError: OptiX error in optixPipelineCreate(multipath)
```

Typical OptiX stderr:

```text
jit_optix_log(): [COMPILER] COMPILE ERROR: failed to create pipeline
Info: Pipeline statistics
    module(s)                            :     1
    entry function(s)                    :     3
    trace call(s)                        :     2
```

Failures appeared across different pipeline shapes, including `trace_reflections`, `trace_refl_epc`, reflection accumulation, diffraction coherent builders, and native segment visibility. This was not an API-contract change and not bad input data.

## Root Cause

RayD had treated all OptiX module exception flags as one global production setting. That was fine for scene/edge OptiX pipelines, but not for multipath launch pipelines on the verified Windows target. With `RAYD_OPTIX_EXCEPTION_FLAGS=0`, fresh multipath pipeline creation could fail during `optixPipelineCreate()` even when:

- the PTX module compiled successfully
- the selected raygen had only one `optixTrace()` call site
- the same test passed after another API had warmed the pipeline cache
- the public Python call and input tensors were valid

The fix that generalized across all failing APIs was to separate multipath OptiX settings:

- keep scene/edge production settings at `RAYD_OPTIX_MODULE_OPT_LEVEL=0x2343` and `RAYD_OPTIX_EXCEPTION_FLAGS=0`
- keep multipath module optimization at the production level by default
- use `RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=11` for multipath pipeline compile/link

Two experimental fixes were rejected because they were not causal:

- splitting `trace_reflections()` into primary/one-bounce raygen variants did not fix production cold creation
- changing multipath module optimization to level 0 only passed because it was tested together with exception flags `11`; restoring optimization level 3 while keeping multipath exception flags `11` also passed

## Fix Pattern

Use dedicated multipath OptiX compile settings instead of mutating public APIs or falling back to JIT paths:

1. define `RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL` separately from the global scene/edge setting
2. default it to the production module optimization level
3. define `RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=11`
4. use the multipath-specific macros in `OptixLaunchPipeline::build()`
5. keep the shared pipeline cache key keyed by full pipeline shape: context, PTX pointer, PTX size, raygen entries, hit entries, hitgroup capacity, payload count, and params size

For staged diffraction/EPC paths, staged native launches are still acceptable when they are the intended implementation. For reflection tracing, do not introduce staged fallback launches just to avoid pipeline creation: the reason Channel moved reflection discovery to RayD was to keep reflection discovery in one native OptiX launch.

After changing a `.cu` OptiX kernel, regenerate and commit the matching embedded PTX header. A source-only change is incomplete.

## Design Rules

- Treat `optixPipelineCreate(multipath)` first as a multipath OptiX pipeline configuration issue, not as a Python API/input issue.
- Verify `RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS` before rewriting kernels or splitting launches.
- Use trace-call and instruction counts as diagnostics, not as proof of root cause.
- Keep scene/edge OptiX production flags independent from multipath flags.
- Keep reflection discovery as one native OptiX launch; do not use staged fallback launches to work around reflection pipeline creation.
- Do not rely on warmed pipeline cache behavior. Reproduce in a fresh subprocess.
- Do not fallback silently. If a native OptiX path is selected, fix that native path.
- Keep public API stable. The successful fixes changed internal launch params, pipeline members, and PTX entries without requiring downstream caller changes.

## Diagnostic Checklist

1. Reproduce in a fresh Python subprocess using the target installed `.pyd`.
2. Confirm which native extension was loaded:

   ```python
   import sys, rayd.drjit
   print(rayd.drjit.__file__)
   print(rayd.drjit._C.__file__)
   print([(k, getattr(v, "__file__", None)) for k, v in sys.modules.items() if k.startswith("rayd")])
   ```

   Since 0.6.0 `rayd` is a PEP 420 namespace package, so `rayd.__file__` is `None`; inspect `rayd.drjit` and its `_C` extension instead. Editable installs can load the backend package from the source tree while loading `rayd.drjit._C` from `site-packages`.

3. Confirm the effective C++ build defines. For the verified fix, the build log should contain `RAYD_OPTIX_MODULE_OPT_LEVEL=0x2343`, `RAYD_OPTIX_EXCEPTION_FLAGS=0`, `RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL=0x2343`, and `RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=11`.
4. Read the OptiX stderr pipeline statistics. Treat them as risk diagnostics, not the final root cause.
5. Count trace sites in the embedded PTX:

   ```powershell
   $ptx = Get-Content backends\drjit\include\rayd\multipath\diffraction_paths_ptx.h -Raw
   $entry = "__raygen__diffraction_paths_order1_target_export_primary"
   $start = $ptx.IndexOf(".visible .entry $entry")
   $next = $ptx.IndexOf(".visible .entry ", $start + 1)
   if ($next -lt 0) { $next = $ptx.Length }
   $body = $ptx.Substring($start, $next - $start)
   ([regex]::Matches($body, "_optix_trace")).Count
   ```

6. If a failing primary-only path has more than one trace site, consider staged launches only if that matches the operation contract.
7. Reinstall the package and verify the actual `.pyd` size/timestamp under the target conda environment.
8. Run the full cold-create matrix; do not accept a single warmed-process pass as proof.

## Regression Tests

Run RayD tests:

```powershell
python -m unittest backends.drjit.tests.drjit.test_reflection_epc -v
python -m unittest backends.drjit.tests.drjit.test_optix_pipeline_cold_create -v
python -m unittest backends.drjit.tests.drjit.test_geometry.GeometryCoreTests.test_trace_reflections_cold_pipeline_survives_materialized_ad_inputs -v
python -m unittest backends.drjit.tests.test_project_metadata -v
python -m unittest backends.drjit.tests.drjit.test_diffraction_accumulation -v
```

Run channel endpoint repros. These paths are **not in this repository**; run them from
the downstream Channel repo checkout (see `docs/downstream-migration.md`):

```powershell
python -m pytest tests\path\test_endpoint_api_contract.py -q --gpu
python -m pytest tests\path\test_example_path_solver_minimal.py tests\deterministic\test_reflection_rayd_epc_backend.py tests\deterministic\test_example_deterministic_radiomap_three_cubes.py -q --gpu
```

The 2026-05-26 verified state after the multipath exception-flag fix:

- RayD `tests.drjit.test_optix_pipeline_cold_create`: `1 passed`
- RayD targeted trace/reflection EPC cold-create tests: passing
- No downstream caller API changes required
