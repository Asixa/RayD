# ADR-0046: Pure-CUDA trace backend for every typed scene operation

- Status: Accepted; pending review by Xingyu Chen
- Date: 2026-11-10
- Decision ID: `cuda-trace-backend-coverage`
- Scope: the Torch typed integration API (`rayd/integration.h`): the ADR-0029 axial-edge visibility primitive, the
  ADR-0033 segment-penetration family, and the scene trace-backend request and report.

## Context

A Torch `SceneCache` traces triangles either through OptiX or through the pure-CUDA compact BVH. `Auto` picks OptiX
when the driver's OptiX runtime is usable and the CUDA BVH otherwise, which is the normal state in a container started
without the `graphics` driver capability and on drivers older than the OptiX ABI the build requests. Every typed scene
operation supported the CUDA BVH except two: `axial_edge_visibility_forward` and the four segment-penetration entry
points refused a CUDA scene with "unsupported by the CUDA ray-tracing backend". A downstream solver that uses them
(Channel: default diffraction, transmission, the Path solver, and BDPT) therefore failed on such machines, and the
typed API gave the caller no way to learn which tracer a scene used or why OptiX was unusable.

## Decision

1. **Both operations run on the CUDA BVH.** Each keeps one numerical body shared by both tracers; only the closest-hit
   or occlusion query differs.
   - Axial-edge visibility: the per-state lane (input checks, the four exact sample points with inline PTX
     round-to-nearest instructions, the OR over `trace_segment`) moves to `src/visibility/axial_edge_visibility.cuh`.
     The OptiX raygen and a new CUDA kernel in `src/scene/multipath.cu` both call it. The CUDA kernel answers each
     unfiltered occlusion query with the any-hit BVH traversal, which reports the same blocked/visible decision as a
     closest-blocker search over the same triangle test.
   - Segment penetration: the ordered `D + 1` probe march moves to `src/penetration/segment_penetration_march.cuh`,
     templated over the closest-hit query. The OptiX raygen supplies `optixTrace`; a new CUDA kernel in
     `src/penetration/penetration.cu` supplies `CudaBvhTraverser::trace_closest` and maps a miss to the OptiX payload
     sentinels. The kernel lives in the family translation unit, so it compiles with the family `precise_no_ftz`
     profile; no unit changes profile and no flag changes (ADR-0035 is untouched). Backward and JVP already consumed
     only the frozen tape and needed no change.
   - Each structurally active non-empty forward call still submits exactly one trace launch (an OptiX launch or one
     CUDA kernel), keeps the caller's stream, and adds no host synchronization or copy. The CUDA path allocates the
     same per-lane traversal stack the other CUDA-BVH operations use.
2. **The trace backend is requestable and observable through the typed API**, additively in `rayd/scene.h`:
   `SceneTraceBackend {Auto, Optix, Cuda}`, `SceneOptions`, the overload
   `create_scene(meshes, const SceneOptions&)` (the one-argument form equals `Auto`), `scene_trace_backend(scene)`
   returning the resolved tracer, and `optix_availability(device)` returning `{available, reason}` without the
   once-per-process fallback warning. A forced `Optix` request on a machine without usable OptiX fails loudly at
   scene creation. `kIntegrationApiVersion` stays `8`: no existing declaration changed.

3. **Every CUDA lane reads the same inputs as its OptiX twin.** An audit of each OptiX launch against its CUDA twin
   found one family where they diverged: the order-1 diffraction path exporter. Its OptiX program is a separate
   hand-written body, while the CUDA kernel runs the shared `src/diffraction/paths.h` algorithm, which hard-coded a
   +x source polarization, never set the ISB taper width, and read the material tensors as contiguous although the
   typed API accepts strided ones. On the CUDA backend a non-x polarization, a non-zero `isb_taper_width_scale`, or a
   strided material column therefore gave wrong fields silently (a z-polarized source on one wedge differed by about
   100%). `paths.h` now reads `tx_pol` (per transmitter, or one shared row), `isb_taper_width_scale`, and the material
   strides whenever its params type carries them. The Dr.Jit params carry none of these fields, so the Dr.Jit path keeps
   its +x, hard-boundary, contiguous behaviour, and its generated PTX is byte-identical. The other families were
   already consistent: reflection trace, reflection accumulation, EPC, segment visibility, axial-edge visibility,
   penetration, and diffraction accumulation and coherent accumulation instantiate one shared algorithm for both
   tracers, with matching layout policies, from one host-built params struct. Diffraction accumulation and coherent
   accumulation fix their source polarization in the host params (+x and zero respectively) identically for both
   tracers; that is a property of their typed configs, which have no polarization or ISB field. The wedge
   coefficient functions and their AD take no scene and run the same kernels on both backends.

## Parity contract

Same inputs through both tracers give identical discrete outputs: axial `any_visible`; penetration `valid`,
`num_hits`, `reached_target`, `overflow`, `global_primitive_id`, the shared failure transaction, the tape primitive
ids, restart branches, tie masks, and direction-denominator branches; and the intersect winners. Values computed before
the first trace (`distance`, `direction`) are bit-identical. Both normals depend only on the winning primitive but
agree to within `1e-6`, not bit for bit (about 2.4% of normal components differed in the 8,192-segment parity run):
the CUDA kernel and the OptiX module are separate compilations of the same source, and their FMA contraction of the
cross product can differ; ADR-0035 freezes the family's FMA-contraction setting, so this record does not change it.
Values derived from the hit distance or barycentrics (`t`, `position`, restart epsilon, barycentric tape, and the derivatives
that consume them) agree to float32 rounding of the triangle test, because OptiX hardware and the CUDA watertight test
round differently. Inputs that graze a triangle edge or vertex within that rounding can resolve differently on the two
tracers; that is a property of two independent triangle tests, not a contract violation.

Field amplitudes agree to a few `1e-4` relative, not to float32 rounding. The Torch OptiX PTX modules compile with
`--use_fast_math` (`RAYD_TORCH_OPTIX_FAST_MATH`, default ON), while the CUDA kernels in `multipath.cu` compile at nvcc
defaults (ADR-0035 profiles `fast_math` and `nvcc_default`), so approximate division, square root, and
transcendentals in the UTD and field math round differently. Channel measured Path amplitudes that differ by up to
`2.8e-4` relative between the backends. This record does not change either profile; aligning them is an ADR-0035
decision. The penetration family is unaffected because both its PTX and its CUDA unit use `precise_no_ftz`.

`tests/native/trace_backend_parity_test.cpp` (CTest `rayd_torch_trace_backend_parity`) checks the contract on a
two-mesh scene of random and tilted triangles with 8,192 lanes per operation, both penetration policies, masks,
non-finite inputs, overflow, all-inactive requests, VJP and JVP; and every diffraction entry point Channel calls on
a closed box: order-1 paths in both layouts with x, z, and per-transmitter oblique polarizations, ISB taper widths 0
and 0.35 (with receivers placed at each edge's incident shadow boundary), strided material columns, diffraction
accumulation (direct and Keller samples), and coherent accumulation with and without point selection. It also asserts
that a z polarization changes the CUDA field, so the +x defect cannot return unnoticed. It exits with CTest's skip code without CUDA or
without usable OptiX. `RAYD_TRACE_BACKEND_BENCH=1` adds a timing table.

## Consequences

- Machines without an OptiX runtime run every typed scene operation, more slowly (see the report that accompanied
  this change for measured ratios).
- The diffraction-accumulation `lane_offset` restriction (ADR-0038) is unchanged; it is not part of the typed API.
- ADR-0029's and ADR-0033's "one OptiX launch" statements now read "one trace launch" on a CUDA scene.

## Stop conditions

Stop and record a new decision if the parity test reports a discrete mismatch on non-grazing inputs, if keeping one
shared body would require changing either family's compile profile, or if a downstream needs per-call tracer selection
instead of per-scene selection.
