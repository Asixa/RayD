# ADR-0045: Rays leaving a surface, independent of coordinate size

- Status: Accepted
- Date: 2026-11-01
- Decision ID: `surface-leaving-rays`
- Scope: every ray RayD starts on a surface: reflection, EPC, reflection accumulation, segment visibility, and
  diffraction segments, in both backends and on both trace backends. It does not change `scene.intersect`.

## Context

RayD kept rays off the surface they leave with fixed distances: a minimum ray distance of `1e-5` for multipath and
visibility rays, `1e-6` for Torch `intersect`, `1e-3` for Dr.Jit `intersect`, and a bias of `1e-5` along the outgoing
direction. Float32 rounding grows with the numbers in the computation, including the triangle's own vertices: it is
about `1e-4` at kilometre coordinates, which beats the first two distances. An origin moved to zero still fails
against a 3 km ground triangle.

The result was wrong answers with no error. On one flat ground triangle with landing points up to 1000 m, about a
quarter of 100,000 reflected rays bounced off the same triangle again and a quarter of points on the ground could not
see a point above them. In a 550,681-triangle town model with 3 km ground triangles, 41,184 of 200,000 rays bounced
twice in a row on one flat triangle on OptiX and 7,951 on CUDA, and total bounces differed by 93.6% between them.

Raising the minimum distance is not a fix. A throwaway build with `1e-3` everywhere still left 4,248 false hits per
100,000 on a tilted plane at 3000 m, because a grazing ray's distance to the plane is its travel times a small cosine,
and it lost 16 real hits per 200,000 in a room. A distance along the ray cannot cover every angle and every scale.

## Decision

A line crosses a plane once, so both cases below are exact at every angle. Their tolerance is relative to the largest
coordinate magnitude involved, with no absolute floor, so it vanishes with the coordinates and small scenes keep their
close geometry. `src/runtime/rt_device.cuh` owns them; `surface_tolerance_relative` in `contracts/operations.json`
records the value.

- **A ray leaving a known surface** starts at `hit + n * 2 * t`, with `n` the unit normal on the side the ray leaves
  and `t = 2^-23 * M`, where `M` is the largest coordinate magnitude of the triangle's vertices. The offset is
  measured along the normal, not the ray, so the angle does not matter. This covers reflection tracing, reflection
  accumulation and EPC discovery on every path, including Dr.Jit's symbolic and differentiable paths.
- **A segment** has no normal at its ends, so `visible` and EPC visibility trace it first. If a blocker's plane passes
  within `t` of an end (`M` also covers both ends), that blocker cannot lie between the ends. The end moves off the
  plane by `2t` and the segment is traced again: away from the other end, or with both ends on one side when the
  segment lies in the plane. Native paths retry up to four times, one retry per plane at a corner; Dr.Jit's HitObject
  path casts three times, because it is recorded inline and cannot branch on the hit count. A blocker still reported
  after the last attempt stands, so the answer errs towards blocked.
- **Diffraction segments** that end on a wedge's edge end just off both faces instead, along the exterior bisector,
  by `max(1e-4, 2t)`, with no along-segment bias at that end (`rt::wedge_visibility_end`). A segment ending exactly on
  the edge grazes the wedge's own faces, and one running into the solid crosses no face before it stops. Only Torch's
  OptiX path export had this; the shared path export and all accumulation did not, which made the backends disagree.
  A thin screen has opposite normals and no bisector, so its point stays and keeps the bias. A scaled along-segment
  bias was tried first and rejected: at 200 km it reached 9.5 cm and multiplied the paths of a far-moved room tenfold
  against the same room near the origin.
- **Gradients** do not see the offset. It is detached in Dr.Jit, and Torch's backward and JVP replay from the surface
  point. This removes the old `1e-5 * direction` term from those derivatives, which was a tiny artifact, not physics.
- The minimum ray distance stays `1e-5`, and every value in `numeric_policy` stays frozen.

### The value

`2^-23` was chosen by measuring against a known answer: a room near the origin is the truth for the same room and
the same rays moved far away. Larger values remove no more false bounces and jump over real geometry: at 200 km
`2 * 2^-23` gives a 9.5 cm offset, lost 1.1% of bounces, and matched the truth less often than no fix at all. `2^-23`
leaves no false repeats at any distance and keeps every bounce count within noise. On 12.6 million rays over flat,
tilted, far-from-origin, duplicated and tessellated surfaces up to 10 km it left 24 false blocks and 20 false bounces,
all on tessellated grids whose triangles are only a few hundred float steps wide. The offset is `2.4e-7 * M`: 24 um at
100 m, 0.24 mm at 1 km, 1.6 mm at 6.5 km.

### Scene.intersect

`scene.intersect` keeps its contract as a query from a free point: it knows nothing about a surface under the origin.
Its minimum distance stays `1e-6` on Torch and `1e-3` on Dr.Jit. Unifying them would break callers either way: Torch
at `1e-3` loses real hits within a millimetre of the origin, and Dr.Jit at `1e-6` changes every symbolic caller.

A caller whose origin lies on a surface must leave it first:

```python
origin = point + outgoing_normal * (2.0 / 2**23) * largest_coordinate_magnitude_of_the_triangle
```

Rays RayD spawns itself (`trace_reflections`, EPC, `visible`) already do this.

## Limits

- Geometry closer than the offset cannot be told apart from the surface. 1 mm plates work at 3 km, where 1 mm is 4
  float steps, but not at 6.5 km. 0.1 mm at 3 km is below one float step and has no representation at all.
- The tolerance uses the magnitude of the surface the ray leaves. A small triangle near the origin coplanar with a
  3 km triangle can still hit the large one.
- A segment lying within a tessellated tilted surface can truly cross a neighbouring triangle whose rounded vertices
  tilt its plane; that plane need not pass near either end. No answer is meaningful for such a segment. Triangles only
  a few float steps wide are rough, not flat, so their extra bounces are real.
- Dr.Jit's symbolic reflection path traces through `scene.intersect`, so its `1e-3` minimum still hides geometry
  closer than 1 mm along the ray. The native path (`symbolic=False`) does not have this limit.
- Surfel, SDF, mixed-scene and the dormant penetration paths keep their own policies.
- Known open difference: on identical inputs in a town model with coordinates up to about 6500, Torch OptiX and CUDA
  still differ by about 8% in first-order diffraction paths and 14% in diffraction accumulation hits, OptiX higher;
  total power differs by 2-8%. The same scenes near the origin agree within 0.5%. Which backend is right at large
  coordinates is not established. Comparing the two with different random inputs overstates this: diffraction samples
  few edges, so job-to-job sampling noise alone reaches tens of percent.

## Consequences

- Reflection results change wherever a false bounce used to happen, which is the point. Elsewhere a later bounce's `t`
  moves by the size of the old bias.
- Six of the eight committed Dr.Jit PTX modules were regenerated. They were built with CUDA 12.8 (PTX ISA 8.7) where
  the old ones used CUDA 12.9 (ISA 8.8); `drjit/ptx_sources.json` records this. The old `reflection_accumulation`
  module contained a `copysign.f32` instruction that driver 572.16 cannot load; the regenerated one does not.

## Stop conditions

- Do not raise the minimum ray distance to hide a self-hit.
- Do not add an absolute floor to the tolerance: it would break small scenes.
- Changing `surface_tolerance_relative` needs the reproduction at every size and the close-geometry tests in
  `tests/reflection/test_surface_leaving_rays*.py`.
