# OptiX Pipeline Create Failures

Date: 2026-05-25

This note records the root cause and guardrails for the `optixPipelineCreate(multipath)` failures fixed around reflection EPC, diffraction accumulation, and diffraction path export.

## Summary

The failures were not caused by bad scene inputs, Python API changes, or missing fallback behavior. They were OptiX pipeline link/compile failures triggered by overly complex native multipath raygen programs, especially on cold pipeline creation.

The recurring pattern was:

- A single raygen contained multiple `optixTrace()` sites.
- The raygen also carried enough surrounding control flow, strategy branching, visibility logic, and payload handling to make the OptiX pipeline linker fail.
- Existing tests often missed the issue because a pipeline had already been warmed, or the tested path used a smaller/simpler entry point.

The reliable fix was to split large multipath raygens into smaller operation-specific pipelines, particularly for primary-only scenes.

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

`trace call(s): 2` was enough to fail for `trace_dfr_paths()` in the channel endpoint repro. Other generic entries had even more trace sites.

## Root Cause

RayD's native multipath kernels had several generic raygen entries that combined multiple visibility or propagation phases into one OptiX program group:

- reflection EPC field used a generic EPC pipeline for a direct-path field case
- first-order diffraction accumulation combined source visibility and target visibility/accumulation
- diffraction path export combined source-to-edge visibility and edge-to-receiver export in one raygen
- some AD custom-op paths pre-created native OptiX pipelines before all JIT inputs had been evaluated, making cold creation failures show up in paths that later did not need the same generic pipeline shape

This produced pipeline programs with multiple trace sites and large branch bodies. On the verified Windows target, these could fail during `optixPipelineCreate()` even though the CUDA code compiled and the same algorithm was logically valid.

The failure is therefore a pipeline-shape problem, not a numerical or API-contract problem.

## Fix Pattern

Use staged native launches for primary-only multipath paths that need more than one visibility segment.

For example:

1. launch a small source-visibility raygen
2. write a `uint8_t` temporary visibility mask
3. launch a small target/export or target/accumulation raygen
4. keep the temporary Dr.Jit array alive until after the staged launches and `drjit::sync_thread()`

Pipeline configs should be operation-specific:

- generic split-scene fallback entries can remain available for split static/dynamic scenes
- primary-only scenes should use smaller entries such as `*_primary`, `*_source_visibility_primary`, and `*_target_*_primary`
- direct reflection EPC field should use direct/direct-primary EPC entries instead of the generic EPC raygen

After changing a `.cu` OptiX kernel, regenerate and commit the matching embedded PTX header. A source-only change is incomplete.

## Design Rules

- Treat `optixPipelineCreate(multipath)` as a pipeline-shape regression until proven otherwise.
- For cold-created primary-only native multipath pipelines, prefer raygen entries with one `optixTrace()` site.
- Do not merge source visibility, target visibility, suffix reflection, export, and accumulation into a single primary raygen when staged launches are possible.
- Do not rely on warmed pipeline cache behavior. Reproduce in a fresh subprocess.
- Do not fallback silently. If a native OptiX path is selected, fix that native path.
- Keep public API stable. The successful fixes changed internal launch params, pipeline members, and PTX entries without requiring downstream caller changes.

## Diagnostic Checklist

1. Reproduce in a fresh Python subprocess using the target installed `.pyd`.
2. Confirm which native extension was loaded:

   ```python
   import sys, rayd
   print(rayd.__file__)
   print([(k, getattr(v, "__file__", None)) for k, v in sys.modules.items() if k.startswith("rayd")])
   ```

   Editable installs can load `rayd/__init__.py` from the source tree while loading `rayd.rayd` from `site-packages`.

3. Read the OptiX stderr pipeline statistics. Pay special attention to `trace call(s)`.
4. Count trace sites in the embedded PTX:

   ```powershell
   $ptx = Get-Content include\rayd\multipath\diffraction_paths_ptx.h -Raw
   $entry = "__raygen__diffraction_paths_order1_target_export_primary"
   $start = $ptx.IndexOf(".visible .entry $entry")
   $next = $ptx.IndexOf(".visible .entry ", $start + 1)
   if ($next -lt 0) { $next = $ptx.Length }
   $body = $ptx.Substring($start, $next - $start)
   ([regex]::Matches($body, "_optix_trace")).Count
   ```

5. If a failing primary-only path has more than one trace site, split it into staged launches.
6. Reinstall the package and verify the actual `.pyd` size/timestamp under the target conda environment.

## Regression Tests

Run RayD tests:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_reflection_epc -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_diffraction_accumulation -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover -v
```

Run channel endpoint repros:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\path\test_endpoint_api_contract.py -q --gpu
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\path\test_example_path_solver_minimal.py tests\deterministic\test_reflection_rayd_epc_backend.py tests\deterministic\test_example_deterministic_radiomap_three_cubes.py -q --gpu
```

The 2026-05-25 verified state after the staged fixes:

- `tests\path\test_endpoint_api_contract.py -q --gpu`: `21 passed`
- channel minimal + deterministic files: `11 passed`
- RayD diffraction suite: `29 passed`
- RayD reflection EPC suite: `12 passed`
- RayD full unittest discover: `119 passed`

