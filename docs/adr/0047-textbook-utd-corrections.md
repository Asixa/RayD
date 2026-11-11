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

## Amendment: edge-fixed face operator (2026-11-15)

Item 2's fallback axis was not enough. The face operator mapped `diag(r_TE, r_TM)` through a
face-local s/p frame and aligned the outgoing s axis to the incoming one by the sign of their dot
product. On the Keller cone the incident and diffracted rays are not a specular pair; for a geometry
symmetric about the face (Channel's trace-backend parity fixture: a corner row whose stationary point
lies 0.5 m past an edge end) the two s axes are exactly perpendicular, the sign came from rounding and
flipped the whole face term. A standalone kernel gave 1.4e-6 with precise math and 2.6e-8 with
`--use_fast_math`; Channel's OptiX and CUDA tracers fell on opposite sides (1.0e-6 versus 1.9e-8),
which is why the CUDA value looked like the fast-math one.

Decision: the operator is Luebbers' edge-fixed form, `op = -diag(r_TM, r_TE)` on the (hard u,
soft v) edge-basis components (the pair assembles `-direct + face * op`, so a perfect conductor gives
`D_h = d12 + d34`, `D_s = d12 - d34`), in `face_reflection_operator`, `slab_face_operator` and
`slab_face_operator_dual`. The edge bases already carry the ray geometry, so no face frame is needed
and `face_s_axis` is removed. The host harness (`tests/diffraction/test_utd_reference.py`) passes
unchanged, including the normal-exit, label and reciprocity cases; the corner row is 2.565e-8 on host,
precise GPU and fast-math GPU alike.

## Amendment: corner-weight refit against full-wave references (2026-11-11)

The owner-accepted refit of the corner-mend constants, now done with the corrected finite-edge factor.

**Reference.** Channel's Tidy3D reference set (Channel `tests/data/fullwave/`,
`benchmarks/scenarios/fullwave_reference.v1.json`, recorded in Channel ADR-050's amendment): a PEC beam
(four 90-degree wedges, 8.3-wavelength edges), a zero-thickness PEC half-plane, a PEC cube, the
three-cube layout and a dielectric cube at 5 GHz, z and horizontal dipoles, 13 observation planes. The
maps are calibrated to the exact unit-dipole field (+-0.4 % magnitude, under 4 degrees of FDTD phase
drift); samples are labelled lit, shadow, transition (near an ISB/RSB) and corner (Keller stationary
point within `0.15 sqrt(lambda L)` of an edge end).

**Procedure.** A host build of `utd.h` with `C_BLEND`, the sigmoid width and the locality factor as
runtime values evaluated the order-1 rows of the fit scenes (beam z/y, half-plane z/y, cube z); it
matches Channel's diffraction component to 1e-3 of the peak at the 99th percentile. The total field
was the Channel solve with its diffraction part replaced. Objective: mean per-region RMS dB magnitude
error over the fit planes, plus the p95 excess of adjacent-sample jumps across ISB/RSB support edges
(Channel `analyze_boundaries`). Held out: cube x-polarized, three cubes, dielectric cube.

| `C_BLEND` / width | mean RMS dB (fit) | ISB/RSB p95 excess mean / max |
| --- | ---: | ---: |
| 0.35 / 0.03 (before) | 1.246 | -0.05 / +0.71 dB |
| 1.0 / 0.03 | 0.999 | +0.00 / +0.59 dB |
| 3.0 / 0.03 | 0.920 | +0.07 / +0.65 dB |
| infinity / 0.03 (adopted) | 0.902 | +0.10 / +0.75 dB |
| infinity / 0.2 | 0.775 | +1.19 / +4.15 dB |

- The error falls monotonically as `C_BLEND` grows: the blend that relaxed the odd (GO-step) part to
  `T_mono` deep in the shadow was compensating the old aperture factor and now makes the shadow 1-4 dB
  too dark. It is removed (`t_used = even * truncEven + odd * gammaOdd`).
- A wider sigmoid lowers the magnitude error further but puts 4-10 dB steps on shadow boundaries near
  edge ends (the interior GO step is no longer fully compensated); a narrower one leaves a seam on the
  corner-cone extension. The width stays 0.03.
- The locality factor (4) moved the objective by under 0.01 dB between 4 and 16 and is not refitted.

**Result (Channel full solve, coupled legs included).** Mean per-region RMS dB error, fit planes
1.246 -> 0.963 (shadow 2.09 -> 1.49, corner 1.30 -> 1.06, transition 0.99 -> 0.78, lit 0.69 -> 0.58);
held-out planes 2.126 -> 1.965 (shadow 5.24 -> 4.32). The full-solve gain is smaller than the host
estimate (0.902) because Channel's coupled R-D/D-R legs evaluate the same corner model and move too.
ISB/RSB jump excess stays within +-0.8 dB on every single-object plane. Per-case tables, gaps and the
regression gate are in Channel ADR-050's amendment. The MC path (`gammaOdd = truncEven = 1`) is
bit-identical; the Dr.Jit `diffraction_paths` and `diffraction_accumulation` PTX are regenerated
(byte-reproducible, marked verified). `tests/diffraction/test_utd_reference.py` passes unchanged.

## Amendment: the coefficient exactly on a GO boundary (2026-11-11)

**Context.** Channel's full-wave suite (Channel defect D6) found a receiver exactly on a reflection shadow
boundary off by half the reflected field. `cot_transition_product_value` replaces a term's cotangent-transition
product by its one-sided limit `+-n sqrt(pi kL) (1 + j)` when the transition argument `a` is below
`shadow_a_threshold` (about 4e-6 n rad from the boundary), and takes the side from `sin(po)` or, within 1e-6 rad,
from the sign of the cotangent. Two faults:

1. On a boundary where the plus/minus branch index `N` rounds at its tie, the *other* term of the pair also has
   `a` = 0 while its cotangent is finite (zero on a half-plane). It got the limit too, so a band of about
   +-2e-6 rad around every boundary carried a spurious `+-n sqrt(pi kL)(1+j)`: the host harness showed the
   coefficient at zero or at twice its lit value there instead of either side's value.
2. Within 1e-6 rad the side came from float rounding of the angles, while the GO term on the same sample is
   decided by Channel's in-face test (inclusive at the edge) or segment visibility (a segment through the edge
   is blocked). The two disagreed at random.

**Decision.** Only a term at its pole (`|cot| > 1e3`; inside the threshold the pole term has `|cot| > 5e5`)
takes the limit; any other term is its raw product. Within the float band the pole term takes the side the
geometric tests choose for a point exactly on the boundary: the lit side for the reflection terms (2, 3), the
shadow side for the incident terms (0, 1) (`beta_term_boundary_side`). Under grazing incidence (`phi'` on a
face plane, within 1e-6 rad) the incident and reflection boundaries of that face coincide and their two terms share
one boundary; they keep the side the cotangent's sign gives, as before, so the pair stays consistent (a
double-diffraction leg between two edges of one face is grazing by construction, and a receiver grid on that
face's plane met it on a whole row of samples). The corner even/odd split evaluates the
mirrored term on the other side, so on the boundary the odd part is the full half-step and the even part the
average. Outside +-2e-6 rad nothing changes.

**Evidence.** `tests/diffraction/test_utd_reference.py::...exactly_on_a_boundary...`: a half-plane with the
source at 60 degrees, the coefficient at the face-0 reflection boundary (120 deg) equals its lit-side value and
at the incident boundary (240 deg) its shadow-side value, for both polarizations; on the previous header the
four cases fail. Channel's boundary probe (a 10 m PEC or concrete half-plane, receivers 0 and 1e-6 m from the
reflection boundary) is recorded in Channel's ADR for the full-wave fixes.

## Amendment: the face term reflects with the face's slab or layer stack (2026-11-11)

**Context.** Channel's full-wave suite (Channel defect D3) measured the step left in the total field at a
reflection shadow boundary, as a fraction of the reflected field: 0.05 for a 0.1 m concrete slab, 0.38-0.42 for
6 mm glass and 0.54-0.60 for a three-layer wall, at normal incidence on the edge. The diffraction face term
evaluated the half-space Fresnel coefficient of one (eta, sigma) per face (`face_reflection_operator`), while
Channel's specular reflection of the same face uses the finite slab of its thickness or its whole layer stack.
On the reflection boundary the face term must cancel exactly the reflected field's step, so it must use the
same coefficient.

**Decision.** The pair evaluation takes its face reflection coefficients from a provider:
`compute_pair_vector_contribution_faces` / `compute_pair_contribution_faces` /
`compute_pair_vector_at_angles_faces` take a `FaceCoefficients` object with `evaluates(face, material)` and
`operator()(face, material, cosTheta) -> FaceFresnelT{rTE, rTM}` (s / p = s x k convention, gain included).
The former entry points forward with `HalfSpaceFaceCoefficients{mat.omega}`, the previous coefficient, and
keep the stored-operator path for faces it does not evaluate, so every existing caller is unchanged.
`pair_face_operator` turns the coefficients into the face operator.

The Torch order-1 exporter and the wedge field with its JVP/VJP gain an optional `DiffractionFaceLayers`
table (`DiffractionPathConfig::face_layers`, `DiffractionWedgeRequest::face0_layers`/`face1_layers`): per face
the slab thickness and the row of a layer store in the transmission CSR layout. With it a face reflects as the
slab of its (eta, sigma, mu) and thickness, or, when its row holds two or more usable layers, as the whole stack
(`src/diffraction/face_coefficients.cuh`, the same `slab_fresnel` / `stack_rt` the specular kernels use). The
wedge companion's forward-mode twin uses `slab_fresnel_dual` / `stack_rt_dual`; the slab thickness and the
stack layers carry no tangent. Without the table nothing changes. The Dr.Jit exporter has no table and keeps
the half-space faces.

**Evidence.** `tests/native/integration_test.cpp::test_diffraction_face_layer_reflection`: a 30 m lossy slab
reflects like the half-space (1e-3), a 5 cm slab changes the field by more than 1 %, a stack of two layers equals
the slab of their summed thickness (1e-4), and per-face rows must match the material rows.
`tests/diffraction/diffraction_wedge_test.cpp::test_face_layers_forward_and_ad`: the layered wedge field
differs from the half-space one, its permittivity JVP matches the central difference of the layered forward
(2 %), and JVP/VJP duality holds. The Dr.Jit `diffraction_paths` and `diffraction_accumulation` PTX were
regenerated (the PTX closure gains `face_coefficients.cuh`, `field_transport.cuh` and `transmission_device.cuh`).
Channel's full-wave tables are in its ADR for the full-wave fixes.

## Amendment: the face term under oblique incidence (2026-11-11)

**Context.** Channel's full-wave suite (Channel defect D2): at a dielectric edge's reflection shadow boundary
the total field kept a step of 26-28 % of the reflected field when the incidence on the edge is oblique
(45 degree skew, concrete), 0.2 % at normal incidence. Two parts of the face term disagree with the geometric
reflection it has to cancel there:

1. **Angle.** The coefficients were evaluated at `(|sin phi| + |sin phi'|) / 2`, the edge-plane angle. The
   specular reflection at the edge sees the 3D incidence cosine `sin(beta0) |sin(phi')|`.
2. **Polarization basis.** Luebbers' operator `-diag(r_TM, r_TE)` on the edge-fixed (hard, soft) components
   is the specular reflection only at normal incidence on the edge. Under oblique incidence the face's TE/TM
   axes are rotated against the edge-fixed axes by an angle that depends on `beta0` and `phi'`, so a
   dielectric face couples the two components. (A perfect conductor's dyadic is `diag(-1, +1)` in the
   edge-fixed bases for every incidence, which is why PEC edges were exact.)

A host harness (Channel task scratch, the RSB residual of a half-plane at 45 degree skew) showed that the angle
alone makes the residual larger (0.26 -> 0.55) and that the rotated dyadic at the 3D angle makes it zero.

**Decision.** `pair_face_operator` evaluates the provider's coefficients at
`sin(beta0) (|sin phi| + |sin phi'|) / 2` (`reciprocal_face_cosines` with `sinBeta0`) and returns
`face_reflection_dyadic`: the mean of the specular reflection dyadic of the incident ray, written in its
edge-fixed basis and in the edge-fixed basis of its mirror image, and the reciprocal image of the same
dyadic built from the reversed observation ray (`M = (M_in + S M_rev^T S) / 2`, `S = diag(1, -1)`). On a
reflection shadow boundary the diffracted ray is the incident ray's specular image, both halves equal the
geometric reflection, and the face term cancels its step exactly; elsewhere the mean keeps the coefficient
reciprocal and independent of the face labels, as amendment 4 of this record requires. For a perfect
conductor, and at normal incidence on the edge, the result is the previous `-diag(r_TM, r_TE)`. Faces kept
on the stored-operator path (Monte Carlo accumulation's stored operators) are unchanged.

**Evidence.** `tests/diffraction/test_utd_reference.py::test_dielectric_face_term_cancels_the_reflection_step_under_oblique_incidence`
(a concrete half-plane, three source/receiver height pairs from normal to strongly oblique incidence, three
polarizations; the total (independent 3D Fresnel GO oracle plus the diffraction) steps by less than 2 % of the
reflected field across the boundary; 15-24 % before) and the existing KP, reciprocity and face-label tests,
unchanged. Dr.Jit `diffraction_paths` / `diffraction_accumulation` PTX regenerated. Channel's full-wave
tables are in its ADR for the full-wave fixes.

**Follow-up (same date): one small angle near the pole.** Within about 1e-4 rad of a boundary the cotangent and the
transition argument were computed from the angle separately (the cotangent of an argument near 0 or pi, the cosine
of one near pi/2 after a subtraction from 2 pi n N), so their product, which is finite, lost the small angle to
different roundings; `cot_val`'s guard on `|sin| < 1e-6` also capped the cotangent within `2n * 1e-6` rad. The
coefficient missed its limit by up to 8 % at 2e-6 rad from a half-plane's boundaries (host harness) and Channel's
boundary gradient ledger saw it as a jump. `beta_term_values` now derives both, within 1e-2 rad of the term's
pole, from one small angle `d = po -+ pi`: the cotangent of `-+d / 2n` without the guard and `a = 2 sin^2(d / 2)`.
Outside that window nothing changes. `test_the_coefficient_is_smooth_up_to_the_boundary` (each side within 0.5 %
of its value 1e-3 deg away, at 1e-4 and 3e-4 deg; eight failures on the previous header).

**Follow-up (same date): the grazing pair's side from the observation angle.** The grazing exception above left the
pair's side to each term's cotangent sign. Each term's own angle is `phi -+ phi'`, so with `phi'` in float noise
(up to 1e-6 rad) on either side of a face plane the two terms could take opposite sides; that went unnoticed
while both cotangents came from one rounded `pi + beta`, and the near-pole evaluation above resolves them. A
grazing pair now takes the side of the observation angle `phi`: the cotangent's sign at `phi` on the term's branch.
`test_grazing_incidence_keeps_one_side_for_both_terms` (sources 2e-5 deg either side of either face plane of a
half-plane match the exactly grazing value within 3 %; 18 failures on the previous header, where a source
2e-5 deg below the face plane gave 12x the grazing value).
