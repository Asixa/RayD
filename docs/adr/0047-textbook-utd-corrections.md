# ADR-0047: Textbook corrections to the shared UTD coefficient and diffraction source

- Status: Accepted; accepted by the owner, 2026-11-15
- Date: 2026-11-10
- Decision ID: `textbook-utd-corrections`
- Scope: `include/rayd/utd.h` (finite-edge aperture factors, face reflection axes, direct source field),
  `include/rayd/field_transport.cuh` and its dual mirror (slab face reflection axes), and the Torch
  diffraction-accumulation and coherent-accumulation launch parameters in `src/diffraction/diffraction.cpp`.

## Context

Channel's reference-accuracy suite (Channel plan item M0-3, findings A-2, A-3 and A-4) compares the
order-1 diffraction field with float64 Kouyoumjian-Pathak (KP) UTD and with Sommerfeld's exact
half-plane solution. On a perfectly conducting 90 degree wedge and a half-plane with 800 m edges, the
field was 2-3 % low at every observation angle, with a deficit that changed with the distances
(|D / D_KP| = 0.978 at s' = 100 m, s = 10 m; 0.972 at s = 30 m), and the half-plane was off by 36 %
(soft) / 21 % (hard) exactly at phi = 270 degrees for s = s' = 30 m. The same deficit was reproduced
outside Channel by host-compiling `utd.h` and evaluating `compute_pair_contribution` directly
(`tests/diffraction/test_utd_reference.py`), so it belongs to RayD.

Five defects were found, each with its own evidence.

1. **Finite-edge aperture factors used the wrong Fresnel argument.** `fresnel_boersma(x)` evaluates
   `int_0^x exp(j t) / sqrt(2 pi t) dt`, a Fresnel integral over the *phase* `t`. The finite-edge
   factors pass it the standard Fresnel variable `v = sigma (u - u*)`, `sigma = sqrt(k c / pi)`,
   whose aperture integral is `int_0^v exp(j pi tau^2 / 2) d tau = fresnel_boersma(pi v^2 / 2)`. With
   the linear argument the tail past an edge end falls as `1 / sqrt(2 pi v)` instead of `1 / (pi v)`:
   an 800 m edge (`v` about 640) lost 2.2 % of its field, a 4 km edge 0.7 %, a 40 km edge 0.22 %
   (the `1/sqrt(length)` signature), and the loss depended on `s` and `s'` through `sigma`. The
   factors affected are `finite_wedge_truncation_factor_bounds` (Monte Carlo and the non-stationary
   path) and `finite_wedge_monotone_truncation` (the deterministic path).
2. **Face reflection axes came from rounding noise along a face normal.** `face_reflection_operator`
   (and `slab_face_operator`) build the face s axis as `normal x direction`. When the incident or
   diffracted direction lies along the face normal (phi' = pi/2, or phi = n pi - pi/2) the cross
   product is rounding noise; `safe_normalize` accepted it whenever its length exceeded 1e-6, so the
   output s/p basis pointed in an arbitrary direction and the face term was wrong by up to O(1).
3. **The direct source dropped the short-dipole pattern.** `direct_source_vector` normalised the
   transverse projection of the transmitter polarization, so the incident field at the edge had unit
   amplitude in every direction instead of `sin(theta)`. Channel's receiver projection keeps
   `cos` factors, so swapping transmitter and receiver changed the field (Channel A-4). Channel
   ADR-045 had already accepted this correction (unnormalised projection) on 2026-07-29, but the
   commit carrying it (`8effb71`, branch `codex/boundary-physics-channel-pin`) never reached this
   history; the code still normalised.

4. **The face reflection angles were neither reciprocal nor independent of the face labels.** The
   finite-conductivity (Luebbers) coefficient evaluates face 0's reflection coefficient at the incidence
   angle `phi'` and face n's at the observation angle `n pi - phi`. Exchanging source and observer, or
   calling the other face "face 0" (which a re-triangulated mesh does: the edge-geometry kernel orders the
   faces from the vertex order), changes which angle each face sees. On a concrete 90 degree wedge
   (eps_r = 5.24, sigma = 0.123 S/m) the host harness showed up to 1466 % field change from relabelling
   the faces and 914 % from a source/observer swap. This is a root cause of Channel A-4 (reciprocity)
   and of the per-path gain changes left in C-6 after the edge set is merged.
5. **The stationary point was computed by a rotation that loses precision.**
   `first_order_diffraction_parameter` rotated the target into the source half-plane with
   `acos`/`sincos` and intersected lines; in float32 it is off by up to 3.1 mm on random 10 m
   geometries (against the float64 closed form), and its error depends on the tracer's math flags
   (OptiX PTX uses fast math). Near an edge end the corner weight is a steep function of that point,
   so Channel's OptiX/CUDA parity test saw a 56x different corner row.

The CUDA-backend audit also flagged two launch parameters that both tracers share:

6. **Diffraction accumulation used `omega = 2 pi c`.** `fresnel_reflection_face` divides the
   conductivity by `omega eps0`, so the loss tangent of every lossy face was scaled by the wavelength
   in metres (11.7x too lossy at 3.5 GHz). The Dr.Jit backend already sets `2 pi c / lambda`.
7. **Coherent accumulation left `tx_pol` at zero.** With point selection the selected stationary
   point uses `direct_source_vector`; with a zero polarization the old normalised code fell back to
   an axis that switched between `z` and `y` with the ray elevation, and the corrected projection
   would give no field at all. The Dr.Jit coherent default is a +x dipole.

## Decision

1. The two finite-edge factors pass `edge_fresnel_phase(v) = sign(v) (pi/2) v^2` to
   `fresnel_boersma`. The phase helper is templated, so the Dual (JVP/VJP) instantiation follows.
   `corner_mend_gamma` keeps its linear `sigma`-unit exit coordinate: it is a sigmoid on a distance,
   not a Fresnel integral.
2. `face_s_axis(normal, direction, fallback)` returns the cross product only when its length exceeds
   `kFaceAxisTolerance = 1e-4`; otherwise the fallback axis: the incident edge basis' `v` axis for the
   incoming side and the projection of the incoming s axis for the outgoing side. At normal incidence
   on the face `r_TM = -r_TE`, so the reflection is isotropic there and the fallback is the
   continuous limit. The UTD face operator, `slab_face_operator` and `slab_face_operator_dual` use the
   helper alike.
3. `direct_source_vector` uses `project_to_wedge_plane(txPol, rayDir)` (ADR-045's accepted form).
4. `reciprocal_face_cosines` sets the incidence cosine of face 0 to `(|sin phi| + |sin phi'|) / 2` and of
   face n to `(|sin(n pi - phi)| + |sin(n pi - phi')|) / 2`. It is symmetric in `phi, phi'` and maps into
   itself under `phi -> n pi - phi`, so the coefficient is reciprocal and label-free by construction; on
   each face's reflection shadow boundary the two sines are equal, so it equals Luebbers' value where the
   reflection term carries the GO discontinuity. (A geometric mean has the same two properties but goes
   grazing whenever only the observer grazes a face, which drives the effective-conductivity PEC's TM
   coefficient into its pseudo-Brewster roll-off; the arithmetic mean does not.) The UTD vector path and
   Channel's Monte Carlo twin use it; Channel's coupled R-D/D-R legs keep their own 3D incidence angle.
5. `first_order_diffraction_parameter` uses the unfolding closed form
   `(rho_t a_s + rho_s a_t) / (rho_s + rho_t)` (axial offsets `a`, radial distances `rho`), symmetric in
   source and target, and returns NaN for a degenerate edge or when both points lie on the edge line
   (every caller already rejects a non-finite parameter or clamps it). Worst float32 error on the same
   random set: 2.3e-6 m.
6. Torch diffraction accumulation sets `omega = 2 pi c / lambda`; coherent accumulation sets the
   documented +x dipole and keeps `omega = 0`, which selects the face operators stored with its states.

## Evidence

Host harness, `tests/diffraction/test_utd_reference.py`, PEC faces, 800 m edge, float64 KP reference,
ratio `|D / D_KP|` (phase within 0.1 degree everywhere):

| case | before | after |
| --- | ---: | ---: |
| 90 deg wedge, s' = 100, s = 10, all angles | 0.9777 | 0.9993 |
| 90 deg wedge, s' = 100, s = 30 | 0.9719 | 0.9989 |
| 90 deg wedge, s' = s = 30, phi = 180 hard | 0.2947 | 0.9989 |
| half-plane, s' = 100, s = 10, all angles | 0.9777 | 0.9993 |
| half-plane, s' = s = 30, phi = 270 soft / hard | 1.3315 / 0.7688 | 0.9991 / 0.9991 |
| wedge, edge half-length 40 / 400 / 4000 m | 0.859 / 0.978 / 0.993 | about 1 - 1/(pi v) |

Concrete 90 degree wedge, 24 source/observer angle pairs, s' = 40 m, s = 12 m: largest field change from
relabelling the faces 14.66 before, 1.9e-5 after; from swapping source and observer (soft) 9.14 before,
1.4e-7 after.

The remaining 0.07-0.1 % is the physical aperture of an 800 m edge (`1/(pi v)` per end). The test
failed on 58 sub-cases before the change (the label/reciprocity case was added with item 4) and passes after it; it also checks `|E| / |E_scalar| =
sin(theta)` for the direct source.

## Consequences

- Every deterministic, Monte Carlo and coupled diffraction value that uses these factors moves. The
  change is largest for short edges (in Fresnel units) and for receivers whose diffracted direction
  is near a face normal. Downstream bit-level baselines that contain diffraction must be regenerated.
- The committed Dr.Jit `diffraction_paths` and `diffraction_accumulation` PTX include `utd.h` and are
  regenerated with this change.
- The calibrated corner-mend constants (`C_BLEND`, the sigmoid width) were fitted with the old
  aperture factor. They are left unchanged; refitting them needs the full-wave reference, which this
  change does not have.

## Owner acceptance

Accepted by the owner, 2026-11-15: the symmetric face-angle rule (mean of the incidence and
observation sines), the refit of the corner-weight constants (`C_BLEND`, the corner sigmoid width)
against the full-wave reference with the corrected finite-edge factor, and the resulting movement
of every diffraction value.
