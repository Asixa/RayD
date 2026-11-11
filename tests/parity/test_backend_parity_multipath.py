# Copyright Xingyu Chen.
# Compares rayd.torch and rayd.drjit reflection and diffraction operations on identical inputs.

"""Cross-backend parity for multipath operations.

Reflections, EPC paths and fields, reflection accumulation, and every diffraction entry point are run on the
same scenes, sources and states in both backends. Inputs keep paths away from visibility and edge
discontinuities so the comparison is about the computation, not about which side of a boundary a float32
rounding lands. Float tolerances come from `contracts/operations.json`; set `RAYD_PARITY_REPORT=1` to print
the largest deviation seen per operation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from tests.support import backend_scenes as bs
from tests.support.backends import both_backends_available, drjit_modules


CONTRACT = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "operations.json").read_text())
TOL = CONTRACT["tolerances"]
ABS, REL = TOL["default_abs"], TOL["default_rel"]
FIELD_ABS, FIELD_REL = TOL["field_abs"], TOL["field_rel"]
PROFILES = CONTRACT["numeric_policy"]["backend_profiles"]
RECORDER = bs.ParityRecorder()
WAVELENGTH = 0.125


def tearDownModule() -> None:
    if os.environ.get("RAYD_PARITY_REPORT") == "1":
        print("\nmax |Torch - Dr.Jit| per operation (multipath):\n" + RECORDER.report(), file=sys.stderr)


def _two_bounce_mirrors():
    """Two small mirrors that reflect a source at the origin to a receiver at (0, 2, 0) via (0,1,0), (0,1,1)."""
    from tests.parity.test_share2_ad import _rotate_two_bounce_vector, _two_bounce_vertices

    vertices = [_rotate_two_bounce_vector(vertex) for vertex in _two_bounce_vertices()]
    planes = [
        ([0.0, 1.0, 0.0], [0.0, math.sqrt(0.5), -math.sqrt(0.5)]),
        ([0.0, 1.0, 1.0], [0.0, -0.3826834, 0.9238795]),
    ]
    return bs.SceneSpec("two_bounce_mirrors", (bs.MeshSpec(vertices, [[0, 1, 2], [3, 4, 5]]),)), planes


# Source and receiver offsets that keep both two-bounce reflection points well inside their mirrors.
EPC_SOURCES = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.03, 0.0, 0.0], [0.0, 0.0, -0.05], [-0.02, 0.0, 0.02]]
EPC_RECEIVERS = [[0.0, 2.0, 0.0], [0.0, 2.0, 0.05], [0.03, 2.0, 0.0], [0.0, 2.1, 0.0], [-0.02, 1.95, 0.0]]


def _dfr_state_rows(count: int):
    """Order-1 states on the far triangle's edges 0..count-1, each with its own wedge tilt, source and power."""
    rows = []
    for index in range(count):
        tilt = 0.05 * index
        rows.append(
            {
                "edge_index": index,
                "edge_pos": [0.0, 0.0, 0.0],
                "edge_dir": [math.cos(tilt), math.sin(tilt), 0.0],
                "edge_t_min": -0.5,
                "edge_t_max": 0.5,
                "n0": [-math.sin(tilt), math.cos(tilt), 0.0],
                "n1": [math.sin(tilt), -math.cos(tilt), 0.0],
                "prim0": -1,
                "prim1": -1,
                "exterior_angle": 1.5 * math.pi,
                "src": [0.05 * index, 0.1 - 0.04 * index, 1.0 + 0.1 * index],
                "src_power": 1.0 + 0.5 * index,
                "wi": [0.0, 0.0, -1.0],
                "d0": [0.0, 0.0, -1.0],
            }
        )
    return rows


def _dfr_scene_spec():
    """A single far-away triangle: the diffraction states do not depend on mesh edges, only on visibility."""
    return bs.SceneSpec(
        "dfr_far_triangle", (bs.MeshSpec([[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]], [[0, 1, 2]]),)
    )


class _MultipathBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        import rayd.torch as rt

        cls.torch, cls.rt, cls.mods = torch, rt, drjit_modules()

    def t_tensor(self, values, dtype=None):
        return self.torch.tensor(values, device="cuda", dtype=dtype or self.torch.float32)

    def close(self, op, name, a, b, atol=ABS, rtol=REL, rows=None):
        RECORDER.close(self, op, name, a, b, atol, rtol, rows)

    def equal(self, op, name, a, b, rows=None):
        RECORDER.equal(self, op, name, a, b, rows)

    # Diffraction inputs.

    def torch_states(self, rows):
        torch = self.torch
        vec = lambda key: self.t_tensor([row[key] for row in rows])  # noqa: E731
        ints = lambda key: self.t_tensor([row[key] for row in rows], torch.int32)  # noqa: E731
        return self.rt.DfrStates(
            edge_index=ints("edge_index"),
            edge_pos=vec("edge_pos"),
            edge_dir=vec("edge_dir"),
            edge_t_min=vec("edge_t_min"),
            edge_t_max=vec("edge_t_max"),
            n0=vec("n0"),
            n1=vec("n1"),
            prim0=ints("prim0"),
            prim1=ints("prim1"),
            exterior_angle=vec("exterior_angle"),
            src=vec("src"),
            src_power=vec("src_power"),
            wi=vec("wi"),
            d0=vec("d0"),
            count=len(rows),
        )

    def drjit_states(self, rows):
        cuda, pj = self.mods.cuda, self.mods.pj
        states = pj.DfrStates()
        states.count = len(rows)
        for key in ("edge_pos", "edge_dir", "n0", "n1", "src", "wi", "d0"):
            setattr(states, key, bs.drjit_vec3(self.mods, [row[key] for row in rows]))
        for key in ("edge_t_min", "edge_t_max", "exterior_angle", "src_power"):
            setattr(states, key, cuda.Float([float(row[key]) for row in rows]))
        for key in ("edge_index", "prim0", "prim1"):
            setattr(states, key, cuda.Int([int(row[key]) for row in rows]))
        states.prefix_depth = cuda.Int([0] * len(rows))
        return states

    def materials(self, eta_r=4.0, sigma=0.0, faces=1):
        cuda = self.mods.cuda
        material_t = self.rt.DfrMaterial(
            eta_r=self.t_tensor([eta_r] * faces),
            sigma=self.t_tensor([sigma] * faces),
            mu_r=self.t_tensor([1.0] * faces),
            gain=self.t_tensor([1.0] * faces),
            valid=self.t_tensor([True] * faces, self.torch.bool),
        )
        material_d = self.mods.pj.DfrMaterial()
        material_d.eta_r = cuda.Float([eta_r] * faces)
        material_d.sigma = cuda.Float([sigma] * faces)
        material_d.mu_r = cuda.Float([1.0] * faces)
        material_d.gain = cuda.Float([1.0] * faces)
        material_d.valid = cuda.Bool([True] * faces)
        return material_t, material_d

    def grids(self, resolution=4, axis=2, position=-1.0, span=1.0):
        grid_t = self.rt.DfrGrid(
            axis=axis,
            position=position,
            coord0_min=-span,
            coord0_max=span,
            coord1_min=-span,
            coord1_max=span,
            resolution0=resolution,
            resolution1=resolution,
            cell_area=(2.0 * span / resolution) ** 2,
        )
        grid_d = self.mods.pj.DfrGrid()
        grid_d.axis, grid_d.position = axis, position
        grid_d.coord0_min, grid_d.coord0_max, grid_d.coord1_min, grid_d.coord1_max = -span, span, -span, span
        grid_d.resolution0 = grid_d.resolution1 = resolution
        grid_d.cell_area = (2.0 * span / resolution) ** 2
        return grid_t, grid_d

    def dfr_options(self, *, seed, direct=0, keller=0, suffix=0, max_order=1):
        pj = self.mods.pj
        options = pj.DfrOptions()
        options.wavelength = WAVELENGTH
        options.k = 2.0 * math.pi / WAVELENGTH
        options.seed = seed
        options.samples = direct + keller + suffix
        options.max_order = max_order
        options.direct_samples, options.keller_samples, options.suffix_samples = direct, keller, suffix
        options.strategy_mask = (
            (pj.DfrStrategy.DIRECT if direct else pj.DfrStrategy(0))
            | (pj.DfrStrategy.KELLER if keller else pj.DfrStrategy(0))
            | (pj.DfrStrategy.SUFFIX_REFL if suffix else pj.DfrStrategy(0))
        )
        options.sample_sequence = pj.DfrSampleSequence.HASH
        options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
        options.collect_edge_use = True
        options.collect_debug_counts = True
        return options

    def compare_accum(self, op, out_t, out_d, counts=("direct_count", "keller_count", "suffix_count", "edge_uses")):
        self.assertEqual(out_t.grid_cell_count, int(out_d.grid_cell_count))
        self.close(op, "power", out_t.power, out_d.power, FIELD_ABS, FIELD_REL)
        for axis in "xyz":
            field = getattr(out_d, f"field_{axis}")
            self.close(op, f"field_{axis}_re", getattr(out_t, f"field_{axis}_re"), field.real, FIELD_ABS, FIELD_REL)
            self.close(op, f"field_{axis}_im", getattr(out_t, f"field_{axis}_im"), field.imag, FIELD_ABS, FIELD_REL)
        for name in counts:
            self.equal(op, name, getattr(out_t, name), getattr(out_d, name))


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class ReflectionParityTests(_MultipathBase):
    def _box_rays(self, count, seed):
        generator = np.random.default_rng(seed)
        origins = generator.uniform(-0.5, 0.5, size=(count, 3)).tolist()
        return origins, bs.unit_vectors(count, seed + 1)

    def test_trace_reflections_bounce_counts_and_active_mask(self):
        cases = [
            (bs.closed_box(), self._box_rays(128, 80)),
            (bs.grid_sheet(), bs.scene_rays(bs.grid_sheet(), 128, 81)),
            (bs.overlapping_meshes(), bs.scene_rays(bs.overlapping_meshes(), 128, 82)),
        ]
        for spec, (origins, directions) in cases:
            scene_t = bs.torch_scene(self.rt, spec)
            scene_d = bs.drjit_scene(self.mods, spec)
            active = [index % 5 != 4 for index in range(len(origins))]
            for bounces in (1, 2, 4, 8):
                for use_active in (False, True):
                    with self.subTest(scene=spec.name, bounces=bounces, active=use_active):
                        ray_t = bs.torch_ray(self.rt, origins, directions)
                        ray_d = bs.drjit_ray(self.mods, origins, directions)
                        mask_t = self.t_tensor(active, self.torch.bool) if use_active else None
                        mask_d = self.mods.cuda.Bool(active) if use_active else True
                        chain_t = scene_t.trace_reflections(ray_t, bounces, mask_t)
                        chain_d = scene_d.trace_reflections(ray_d, bounces, mask_d, symbolic=False)
                        self.assertEqual((int(chain_d.ray_count), int(chain_d.max_bounces)), (len(origins), bounces))
                        valid = bs.host(chain_t.valid)
                        self.assertGreater(int(valid.sum()), 16)
                        if use_active:
                            self.assertFalse(valid[~np.asarray(active)].any())
                        self.equal("trace_reflections", "valid", chain_t.valid, chain_d.is_valid())
                        self.equal("trace_reflections", "prim_ids", chain_t.prim_ids, chain_d.global_prim_ids)
                        # Invalid slots carry backend-specific miss sentinels (test_reflection_miss_sentinel...).
                        slots = valid.reshape(-1)
                        # Relative tolerance grows with the bounce index: each bounce re-derives its origin.
                        self.close("trace_reflections", "t", chain_t.t, chain_d.t, ABS, REL * bounces, rows=slots)
                        self.close(
                            "trace_reflections",
                            "image_sources",
                            chain_t.image_sources,
                            chain_d.image_sources,
                            ABS * bounces,
                            REL * bounces,
                            rows=slots,
                        )
                        # The symbolic path traces each bounce with the Dr.Jit scene ray_tmin (1e-3), so only
                        # rays whose every segment is far longer than that are comparable; see
                        # test_symbolic_reflections_drop_segments_below_the_drjit_ray_tmin.
                        segments = np.where(valid, bs.host(chain_t.t), np.inf)
                        rows = segments.min(axis=1) > 10.0 * PROFILES["drjit"]["ray_tmin"]
                        self.assertGreater(int(rows.sum()), len(origins) // 2)
                        trace_d = scene_d.trace_reflections(ray_d, bounces, mask_d, symbolic=True)
                        for depth in (0, bounces - 1):
                            bounce = trace_d.bounce(depth)
                            op = "trace_reflections[symbolic]"
                            self.equal(op, "valid", valid[:, depth], bounce.is_valid(), rows)
                            self.close(
                                op,
                                "t",
                                bs.host(chain_t.t)[:, depth],
                                bounce.t,
                                ABS,
                                REL * bounces,
                                rows=rows & valid[:, depth],
                            )
                            self.equal(
                                op, "prim_ids", bs.host(chain_t.prim_ids)[:, depth], bounce.global_prim_ids, rows
                            )

    def test_symbolic_reflections_drop_segments_below_the_drjit_ray_tmin(self):
        # Dr.Jit symbolic=True traces every bounce through the scene intersect (ray_tmin 1e-3); the fused
        # symbolic=False launch and Torch use the shared multipath trace_tmin (1e-5). A bounce that leaves a box
        # face 2e-4 from the adjacent face therefore has a ~2.8e-4 second segment that only symbolic=True drops.
        tmin_d = PROFILES["drjit"]["ray_tmin"]
        tmin_shared = CONTRACT["numeric_policy"]["shared_multipath"]["trace_tmin"]
        spec = bs.closed_box()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        origin = np.asarray([0.0, 0.3, 0.0])
        direction = np.asarray([-1.0 + 2.0e-4, 0.3, -1.0]) - origin
        direction /= np.linalg.norm(direction)
        origins, directions = [origin.tolist()], [direction.tolist()]
        chain_t = scene_t.trace_reflections(bs.torch_ray(self.rt, origins, directions), 2)
        chain_n = scene_d.trace_reflections(bs.drjit_ray(self.mods, origins, directions), 2, symbolic=False)
        trace_s = scene_d.trace_reflections(bs.drjit_ray(self.mods, origins, directions), 2, symbolic=True)
        second = float(bs.host(chain_t.t)[0, 1])
        self.assertTrue(tmin_shared < second < tmin_d / 2.0, second)
        self.assertEqual(bs.host(chain_t.valid).tolist(), [[True, True]])
        self.assertEqual(bs.host(chain_n.is_valid()).tolist(), [True, True])
        self.assertEqual([bool(bs.host(trace_s.bounce(i).is_valid())[0]) for i in range(2)], [True, False])

    def test_reflection_miss_sentinel_follows_the_contract(self):
        # contracts/operations.json records the invalid-bounce ReflectionChain.t per backend: Torch reports the
        # shared reflection-trace kernel's finite tmax (1e8), Dr.Jit reports +inf.
        by_backend = CONTRACT["miss_sentinels"]["reflection_trace_distance_by_backend"]
        self.assertEqual(by_backend["torch"], CONTRACT["miss_sentinels"]["reflection_trace_distance"])
        self.assertEqual(by_backend["drjit"], "inf")
        spec = bs.single_triangle()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        origins, directions = [[0.25, 0.25, -1.0], [5.0, 5.0, -1.0]], [[0.0, 0.0, 1.0]] * 2
        chain_t = scene_t.trace_reflections(bs.torch_ray(self.rt, origins, directions), 2)
        chain_d = scene_d.trace_reflections(bs.drjit_ray(self.mods, origins, directions), 2, symbolic=False)
        invalid = ~bs.host(chain_t.valid)
        self.assertEqual(invalid.tolist(), [[False, True], [True, True]])
        self.assertTrue(np.all(bs.host(chain_t.t)[invalid] == by_backend["torch"]))
        self.assertTrue(
            np.all(bs.host(chain_d.t).reshape(2, 2)[invalid] == np.inf),
            f"Dr.Jit invalid-bounce t = {bs.host(chain_d.t).tolist()}",
        )

    def test_trace_refl_epc_with_expected_primitives(self):
        spec, planes = _two_bounce_mirrors()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        count = len(EPC_SOURCES)
        options_t = self.rt.ReflEpcOptions(
            expected_prim_ids=self.t_tensor([[0, 1]] * count, self.torch.int32),
            direct_plane_points=self.t_tensor([[plane[0] for plane in planes]] * count),
            direct_plane_normals=self.t_tensor([[plane[1] for plane in planes]] * count),
            surface_group_id=self.t_tensor([0, 1], self.torch.int32),
            surface_group_size=self.t_tensor([1, 1], self.torch.int32),
            surface_group_members=self.t_tensor([0, 1], self.torch.int32),
        )
        epc_t = scene_t.trace_refl_epc(bs.torch_rows(EPC_SOURCES), bs.torch_rows(EPC_RECEIVERS), 2, options_t)
        cuda = self.mods.cuda
        options_d = self.mods.pj.ReflEpcOptions()
        options_d.expected_prim_ids = cuda.Int([0, 1] * count)
        options_d.surface_group_id = cuda.Int([0, 1])
        options_d.surface_group_size = cuda.Int([1, 1])
        options_d.surface_group_members = cuda.Int([0, 1])
        options_d.surface_max_group_size = 1
        # Dr.Jit traces the discovery ray; aim it at the receiver like the Torch direct-primary discovery.
        directions = np.asarray(EPC_RECEIVERS) - np.asarray(EPC_SOURCES)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        epc_d = scene_d.trace_refl_epc(
            bs.drjit_ray(self.mods, EPC_SOURCES, directions.tolist()),
            bs.drjit_vec3(self.mods, EPC_RECEIVERS),
            2,
            options_d,
        )
        self.assertTrue(bs.host(epc_t.valid).all())
        self.equal("trace_refl_epc", "valid", epc_t.valid, epc_d.valid)
        self.close("trace_refl_epc", "path_length", epc_t.path_length, epc_d.path_length)
        self.close("trace_refl_epc", "reflection_points", epc_t.hit_points, epc_d.reflection_points)
        self.equal("trace_refl_epc", "resolved_prim_ids", epc_t.resolved_prim_ids, epc_d.resolved_prim_ids)
        self.equal("trace_refl_epc", "surface_group_ids", epc_t.surface_group_ids, epc_d.surface_group_ids)
        # Torch reports the plane normals it was given; Dr.Jit reports the face orientation. Equal up to sign.
        cosine = np.abs(np.sum(bs.host(epc_t.normals).reshape(-1, 3) * bs.host(epc_d.plane_normals), axis=1))
        self.close("trace_refl_epc", "normal_alignment", cosine, np.ones_like(cosine))

    def test_trace_refl_epc_field_two_bounce(self):
        spec, planes = _two_bounce_mirrors()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        count = len(EPC_SOURCES)
        field_t = scene_t.trace_refl_epc_field(bs.torch_rows(EPC_SOURCES), bs.torch_rows(EPC_RECEIVERS), 2)
        cuda = self.mods.cuda
        options = self.mods.pj.ReflEpcFieldOptions()
        options.expected_prim_ids = cuda.Int([0, 1] * count)
        options.slot_plane_point = bs.drjit_vec3(self.mods, [plane[0] for plane in planes] * count)
        options.slot_plane_normal = bs.drjit_vec3(self.mods, [plane[1] for plane in planes] * count)
        # The Torch field fixes eta_r = 4, mu_r = 1, sigma = 0, gain = 1 and an x-polarised source.
        options.slot_eta_r = cuda.Float([4.0] * 2 * count)
        options.slot_mu_r = cuda.Float([1.0] * 2 * count)
        options.slot_sigma = cuda.Float([0.0] * 2 * count)
        options.slot_gain = cuda.Float([1.0] * 2 * count)
        field_d = scene_d.trace_refl_epc_field(
            bs.drjit_vec3(self.mods, EPC_SOURCES),
            bs.drjit_vec3(self.mods, EPC_RECEIVERS),
            2,
            options,
            cuda.Bool([True] * count),
        )
        self.assertTrue(bs.host(field_t.valid).all())
        self.assertGreater(float(np.abs(bs.host(field_t.field_re)).min()), 1.0e-4)
        self.equal("trace_refl_epc_field", "valid", field_t.valid, field_d.valid)
        self.close("trace_refl_epc_field", "path_length", field_t.path_length, field_d.path_length)
        self.close("trace_refl_epc_field", "field_re", field_t.field_re, field_d.field_x_re, FIELD_ABS, FIELD_REL)
        self.close("trace_refl_epc_field", "field_im", field_t.field_im, field_d.field_x_im, FIELD_ABS, FIELD_REL)
        for name in ("field_y_re", "field_y_im", "field_z_re", "field_z_im"):
            self.close("trace_refl_epc_field", name, np.zeros(count), getattr(field_d, name), FIELD_ABS, 0.0)

    # A material, source polarization, frequency and wavelength that differ from the Torch defaults in every input.
    EPC_MATERIAL = {"slot_eta_r": 2.5, "slot_mu_r": 1.3, "slot_sigma": 0.02, "slot_gain": 0.8}
    EPC_POLARIZATION = [0.6, 0.0, 0.8]
    EPC_OMEGA = 2.0 * math.pi * 3.0e9

    def _epc_material_options(self, count, bounces):
        """Matching non-default Torch and Dr.Jit EPC field options; the Dr.Jit one still needs its planes."""
        cuda = self.mods.cuda
        options_t = self.rt.ReflEpcFieldOptions(
            **{name: self.t_tensor([[value] * bounces] * count) for name, value in self.EPC_MATERIAL.items()},
            tx_polarization=self.t_tensor([self.EPC_POLARIZATION]),
            omega=self.EPC_OMEGA,
            wavelength=WAVELENGTH,
        )
        options_d = self.mods.pj.ReflEpcFieldOptions()
        for name, value in self.EPC_MATERIAL.items():
            setattr(options_d, name, cuda.Float([value] * bounces * count))
        options_d.tx_polarization = bs.drjit_vec3(self.mods, [self.EPC_POLARIZATION])
        options_d.omega = self.EPC_OMEGA
        options_d.wavelength = WAVELENGTH
        return options_t, options_d

    def _compare_epc_field(self, op, field_t, field_d, count):
        self.assertTrue(bs.host(field_t.valid).all())
        self.assertGreater(float(np.abs(bs.host(field_t.field_re)).min()), 1.0e-5)
        self.equal(op, "valid", field_t.valid, field_d.valid)
        self.close(op, "path_length", field_t.path_length, field_d.path_length)
        self.close(op, "field_re", field_t.field_re, field_d.field_x_re, FIELD_ABS, FIELD_REL)
        self.close(op, "field_im", field_t.field_im, field_d.field_x_im, FIELD_ABS, FIELD_REL)

    def test_trace_refl_epc_field_non_default_material(self):
        spec, planes = _two_bounce_mirrors()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        count = len(EPC_SOURCES)
        options_t, options_d = self._epc_material_options(count, 2)
        options_d.expected_prim_ids = self.mods.cuda.Int([0, 1] * count)
        options_d.slot_plane_point = bs.drjit_vec3(self.mods, [plane[0] for plane in planes] * count)
        options_d.slot_plane_normal = bs.drjit_vec3(self.mods, [plane[1] for plane in planes] * count)
        field_t = scene_t.trace_refl_epc_field(bs.torch_rows(EPC_SOURCES), bs.torch_rows(EPC_RECEIVERS), 2, options_t)
        field_d = scene_d.trace_refl_epc_field(
            bs.drjit_vec3(self.mods, EPC_SOURCES), bs.drjit_vec3(self.mods, EPC_RECEIVERS), 2, options_d
        )
        self._compare_epc_field("trace_refl_epc_field_material", field_t, field_d, count)
        default_t = scene_t.trace_refl_epc_field(bs.torch_rows(EPC_SOURCES), bs.torch_rows(EPC_RECEIVERS), 2)
        self.assertFalse(np.allclose(bs.host(default_t.field_re), bs.host(field_t.field_re)))

    def test_trace_refl_epc_field_from_ray(self):
        spec, planes = _two_bounce_mirrors()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        count = len(EPC_SOURCES)
        directions = (np.asarray(EPC_RECEIVERS) - np.asarray(EPC_SOURCES)).tolist()
        options_t, options_d = self._epc_material_options(count, 2)
        # The Dr.Jit Ray overload evaluates the field with the slot normals it is given; these are the mirror
        # normals the Torch trace discovers.
        options_d.slot_plane_normal = bs.drjit_vec3(self.mods, [plane[1] for plane in planes] * count)
        field_t = scene_t.trace_refl_epc_field(
            bs.torch_ray(self.rt, EPC_SOURCES, directions), bs.torch_rows(EPC_RECEIVERS), 2, options_t
        )
        field_d = scene_d.trace_refl_epc_field(
            bs.drjit_ray(self.mods, EPC_SOURCES, directions), bs.drjit_vec3(self.mods, EPC_RECEIVERS), 2, options_d
        )
        self._compare_epc_field("trace_refl_epc_field_ray", field_t, field_d, count)

    def test_trace_refl_epc_from_ray(self):
        spec, _planes = _two_bounce_mirrors()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        count = len(EPC_SOURCES)
        directions = (np.asarray(EPC_RECEIVERS) - np.asarray(EPC_SOURCES)).tolist()
        options_t = self.rt.ReflEpcOptions(
            expected_prim_ids=self.t_tensor([[0, 1]] * count, self.torch.int32),
            surface_group_id=self.t_tensor([0, 1], self.torch.int32),
            surface_group_size=self.t_tensor([1, 1], self.torch.int32),
            surface_group_members=self.t_tensor([0, 1], self.torch.int32),
        )
        cuda = self.mods.cuda
        options_d = self.mods.pj.ReflEpcOptions()
        options_d.expected_prim_ids = cuda.Int([0, 1] * count)
        options_d.surface_group_id = cuda.Int([0, 1])
        options_d.surface_group_size = cuda.Int([1, 1])
        options_d.surface_group_members = cuda.Int([0, 1])
        options_d.surface_max_group_size = 1
        for label, opts_t, opts_d in (("constrained", options_t, options_d), ("free", None, None)):
            with self.subTest(options=label):
                epc_t = scene_t.trace_refl_epc(
                    bs.torch_ray(self.rt, EPC_SOURCES, directions), bs.torch_rows(EPC_RECEIVERS), 2, opts_t
                )
                epc_d = scene_d.trace_refl_epc(
                    bs.drjit_ray(self.mods, EPC_SOURCES, directions), bs.drjit_vec3(self.mods, EPC_RECEIVERS), 2, opts_d
                )
                self.assertTrue(bs.host(epc_t.valid).all())
                self.equal("trace_refl_epc_ray", "valid", epc_t.valid, epc_d.valid)
                self.close("trace_refl_epc_ray", "path_length", epc_t.path_length, epc_d.path_length)
                self.close("trace_refl_epc_ray", "reflection_points", epc_t.hit_points, epc_d.reflection_points)
                self.equal("trace_refl_epc_ray", "resolved_prim_ids", epc_t.resolved_prim_ids, epc_d.resolved_prim_ids)
                self.equal("trace_refl_epc_ray", "surface_group_ids", epc_t.surface_group_ids, epc_d.surface_group_ids)
                self.close("trace_refl_epc_ray", "normals", epc_t.normals.reshape(-1, 3), epc_d.plane_normals)

    def test_accumulate_reflections_box_bounce_counts(self):
        spec = bs.closed_box()
        scene_t = bs.torch_scene(self.rt, spec)
        scene_d = bs.drjit_scene(self.mods, spec)
        cuda, pj = self.mods.cuda, self.mods.pj
        count, faces, resolution, low, high = 256, 12, 4, -0.9, 0.9
        tx = [0.1, -0.2, 0.3]
        origins, directions = [tx] * count, bs.unit_vectors(count, 83)
        active = [index % 7 != 6 for index in range(count)]
        polarization = [[1.0, 0.0, 0.0] if index % 2 else [0.0, 0.6, 0.8] for index in range(count)]
        eta = [3.0 + 0.25 * index for index in range(faces)]
        material_t = self.rt.ReflMaterial(
            eta_r=self.t_tensor(eta),
            sigma=self.t_tensor([0.01] * faces),
            mu_r=self.t_tensor([1.0] * faces),
            gain=self.t_tensor([1.0] * faces),
            valid=self.t_tensor([True] * faces, self.torch.bool),
        )
        material_d = pj.ReflMaterial()
        material_d.eta_r, material_d.sigma = cuda.Float(eta), cuda.Float([0.01] * faces)
        material_d.mu_r, material_d.gain = cuda.Float([1.0] * faces), cuda.Float([1.0] * faces)
        material_d.valid = cuda.Bool([True] * faces)
        grid_t = self.rt.AccumGrid(
            axis=2,
            position=-0.5,
            coord0_min=low,
            coord0_max=high,
            coord1_min=low,
            coord1_max=high,
            resolution0=resolution,
            resolution1=resolution,
        )
        grid_d = pj.AccumGrid()
        grid_d.axis, grid_d.position = 2, -0.5
        grid_d.coord0_min, grid_d.coord0_max, grid_d.coord1_min, grid_d.coord1_max = low, high, low, high
        grid_d.resolution0 = grid_d.resolution1 = resolution
        wavelength = 0.37
        options_d = pj.AccumOptions()
        # Torch derives k and the cell area from the wavelength and the grid; Dr.Jit takes them explicitly.
        options_d.wavelength, options_d.k = wavelength, 2.0 * math.pi / wavelength
        options_d.cell_area = ((high - low) / resolution) ** 2
        for bounces in (1, 2, 3, 5):
            with self.subTest(bounces=bounces):
                out_t = scene_t.accum_refl(
                    bs.torch_ray(self.rt, origins, directions, [10.0] * count),
                    bs.torch_rows(origins),
                    grid_t,
                    material_t,
                    bounces,
                    self.rt.AccumOptions(wavelength=wavelength),
                    self.t_tensor(active, self.torch.bool),
                    bs.torch_rows(polarization),
                )
                out_d = scene_d.accum_refl(
                    bs.drjit_ray(self.mods, origins, directions, [10.0] * count),
                    bs.drjit_vec3(self.mods, origins),
                    grid_d,
                    material_d,
                    bounces,
                    options_d,
                    cuda.Bool(active),
                    bs.drjit_vec3(self.mods, polarization),
                )
                self.assertEqual(out_t.grid_cell_count, int(out_d.grid_cell_count))
                self.assertGreater(int(bs.host(out_t.reflection_count).sum()), 0)
                self.equal("accumulate_reflections", "reflection_count", out_t.reflection_count, out_d.reflection_count)
                self.close(
                    "accumulate_reflections",
                    "power",
                    out_t.reflection_power,
                    out_d.reflection_power,
                    FIELD_ABS * 1e-2,
                    FIELD_REL,
                )
                for axis in "xyz":
                    field = getattr(out_d, f"reflection_field_{axis}")
                    for part, value in (("re", field.real), ("im", field.imag)):
                        self.close(
                            "accumulate_reflections",
                            f"field_{axis}_{part}",
                            getattr(out_t, f"reflection_field_{axis}_{part}"),
                            value,
                            FIELD_ABS * 1e-2,
                            FIELD_REL,
                        )


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class DiffractionParityTests(_MultipathBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        spec = _dfr_scene_spec()
        cls.scene_t = bs.torch_scene(cls.rt, spec)
        cls.scene_d = bs.drjit_scene(cls.mods, spec)

    def _sorted_paths(self, paths, count, keys):
        order = np.lexsort(tuple(np.asarray(key)[:count] for key in reversed(keys)))
        return order

    def test_trace_dfr_paths_order1(self):
        rows = _dfr_state_rows(3)
        tx = [[0.0, 0.0, 1.0], [0.1, 0.05, 1.2]]
        rx = [[0.0, 0.0, -1.0], [0.2, -0.1, -0.8], [-0.3, 0.2, -1.1]]
        active = [True, False, True]
        material_t, material_d = self.materials()
        paths_t = self.scene_t.trace_dfr_paths(
            tx_positions=bs.torch_rows(tx),
            rx_positions=bs.torch_rows(rx),
            states=self.torch_states(rows),
            material=material_t,
            options=self.rt.DfrPathOptions(max_paths=64, wavelength=WAVELENGTH),
            active=self.t_tensor(active, self.torch.bool),
        )
        pj, cuda = self.mods.pj, self.mods.cuda
        options = pj.DfrPathOptions()
        options.wavelength, options.k, options.seed = WAVELENGTH, 2.0 * math.pi / WAVELENGTH, 17
        options.max_order, options.max_paths, options.max_rx = 1, 64, len(rx)
        options.strategy_mask, options.sample_count, options.return_geom = pj.DfrStrategy.DIRECT, 1, 1
        options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
        paths_d = self.scene_d.trace_dfr_paths(
            bs.drjit_vec3(self.mods, tx),
            bs.drjit_vec3(self.mods, rx),
            self.drjit_states(rows),
            material_d,
            options,
            cuda.Bool(active),
        )
        self.assertEqual(paths_t.capacity, int(paths_d.capacity))
        count = int(bs.host(paths_t.count).reshape(-1)[0])
        self.assertEqual(count, int(bs.host(paths_d.count).reshape(-1)[0]))
        self.assertEqual(count, len(tx) * len(rx) * sum(active), "every active (tx, rx, state) lane must export")
        # Compact rows are appended atomically, so compare them in (tx, rx, edge, delay) order.
        keys_t = [bs.host(paths_t.tx_id), bs.host(paths_t.rx_id), bs.host(paths_t.edge0), bs.host(paths_t.delay)]
        keys_d = [bs.host(paths_d.tx_id), bs.host(paths_d.rx_id), bs.host(paths_d.edge0), bs.host(paths_d.delay)]
        order_t = self._sorted_paths(paths_t, count, keys_t)
        order_d = self._sorted_paths(paths_d, count, keys_d)
        for name in ("valid", "tx_id", "rx_id", "order", "edge0"):
            self.equal(
                "trace_dfr_paths",
                name,
                bs.host(getattr(paths_t, name))[:count][order_t],
                bs.host(getattr(paths_d, name))[:count][order_d],
            )
        self.close(
            "trace_dfr_paths", "delay", bs.host(paths_t.delay)[:count][order_t], bs.host(paths_d.delay)[:count][order_d]
        )
        self.close("trace_dfr_paths", "p0", bs.host(paths_t.p0)[:count][order_t], bs.host(paths_d.p0)[:count][order_d])
        for axis in "xyz":
            field = getattr(paths_d, f"field_{axis}")
            for part, value in (("re", field.real), ("im", field.imag)):
                self.close(
                    "trace_dfr_paths",
                    f"field_{axis}_{part}",
                    bs.host(getattr(paths_t, f"field_{axis}_{part}"))[:count][order_t],
                    bs.host(value)[:count][order_d],
                    FIELD_ABS,
                    FIELD_REL,
                )

    def _direct_pair(self, rows, samples, active, seed=29):
        material_t, material_d = self.materials(eta_r=4.0, sigma=0.0)
        grid_t, grid_d = self.grids()
        out_t = self.scene_t.accum_dfr_direct(
            states=self.torch_states(rows),
            grid=grid_t,
            material=material_t,
            options=self.rt.DfrOptions(
                wavelength=WAVELENGTH,
                seed=seed,
                direct_samples=samples.get("direct", 0),
                keller_samples=samples.get("keller", 0),
            ),
            active=self.t_tensor(active, self.torch.bool),
        )
        out_d = self.scene_d.accum_dfr_direct(
            self.drjit_states(rows),
            grid_d,
            material_d,
            self.dfr_options(seed=seed, **samples),
            self.mods.cuda.Bool(active),
        )
        return out_t, out_d

    def test_accum_dfr_direct_strategies(self):
        # One active state: the multi-state estimator normalisation differs (see the expected failure below).
        cases = (
            ("direct", dict(direct=64)),
            ("keller", dict(keller=64)),
            ("direct+keller", dict(direct=32, keller=32)),
        )
        for label, samples in cases:
            for state in range(3):
                with self.subTest(strategy=label, state=state):
                    out_t, out_d = self._direct_pair(_dfr_state_rows(3)[state : state + 1], samples, [True])
                    self.assertGreater(float(bs.host(out_t.power).sum()), 0.0)
                    self.compare_accum(f"accum_dfr_direct[{label}]", out_t, out_d)

    def test_accum_dfr_direct_inactive_states_contribute_nothing(self):
        rows = _dfr_state_rows(3)
        out_t, out_d = self._direct_pair(rows, dict(direct=64), [False, False, False])
        self.assertEqual(float(bs.host(out_t.power).sum()), 0.0)
        self.compare_accum("accum_dfr_direct[inactive]", out_t, out_d)

    def test_accum_dfr_direct_multi_state_normalisation(self):
        # With S > 1 states each Monte-Carlo lane draws state `lane % S` and both backends weight it by
        # edge_length * S / samples (sample_edge_weight_for_lane), so the result estimates the sum over states.
        out_t, out_d = self._direct_pair(_dfr_state_rows(3), dict(direct=64), [True, True, True])
        self.compare_accum("accum_dfr_direct[3 states]", out_t, out_d)

    def test_accum_dfr_direct_suffix_reflection(self):
        from tests.parity import test_drjit as legacy

        material_t, material_d = self.materials()
        out_t = legacy._torch_suffix_scene(self.rt).accum_dfr_direct(
            states=legacy._torch_suffix_states(self.rt),
            grid=legacy._torch_dfr_grid(self.rt, axis=1, position=-2.0),
            material=material_t,
            options=self.rt.DfrOptions(wavelength=WAVELENGTH, seed=41, suffix_samples=16),
        )
        out_d = legacy._rayd_suffix_scene(self.mods.pj, self.mods.cuda).accum_dfr_direct(
            legacy._rayd_suffix_states(self.mods.pj, self.mods.cuda),
            legacy._rayd_dfr_grid(self.mods.pj, axis=1, position=-2.0),
            material_d,
            self.dfr_options(seed=41, suffix=16),
            self.mods.cuda.Bool([True]),
        )
        self.assertGreater(int(bs.host(out_t.suffix_count).sum()), 0)
        self.compare_accum("accum_dfr_direct[suffix]", out_t, out_d)

    def test_accum_dfr_chain_orders(self):
        from tests.parity import test_drjit as legacy

        material_t, material_d = self.materials()
        grid_t, grid_d = self.grids(resolution=2)
        for max_order, recursive in ((2, 1), (3, 2)):
            with self.subTest(max_order=max_order):
                out_t = self.scene_t.accum_dfr(
                    initial_states=legacy._torch_dfr_states(self.rt, src_power=2.0),
                    recursive_states=legacy._torch_recursive_dfr_states(self.rt, count=recursive),
                    grid=grid_t,
                    material=material_t,
                    options=self.rt.DfrOptions(
                        wavelength=WAVELENGTH, seed=43, direct_samples=32, keller_samples=128, max_order=max_order
                    ),
                )
                out_d = self.scene_d.accum_dfr(
                    legacy._rayd_dfr_states(self.mods.pj, self.mods.cuda, src_power=2.0),
                    legacy._rayd_recursive_dfr_states(self.mods.pj, self.mods.cuda, count=recursive),
                    grid_d,
                    material_d,
                    self.dfr_options(seed=43, direct=32, keller=128, max_order=max_order),
                    True,
                )
                self.assertGreater(float(bs.host(out_t.power).sum()), 0.0)
                self.compare_accum(f"accum_dfr[order{max_order}]", out_t, out_d)

    def _coherent_pair(self, rows, select=True, prefilter=True):
        material_t, material_d = self.materials()
        grid_t, grid_d = self.grids(resolution=4)
        pj = self.mods.pj
        out_t = self.scene_t.accum_dfr_coherent_direct(
            states=self.torch_states(rows),
            grid=grid_t,
            material=material_t,
            options=self.rt.DfrCoherentOptions(
                wavelength=WAVELENGTH, select_diffraction_point=select, prefilter_visibility=prefilter
            ),
        )
        options = pj.DfrCoherentOptions()
        options.wavelength, options.k, options.max_order = WAVELENGTH, 2.0 * math.pi / WAVELENGTH, 1
        options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
        options.select_diffraction_point, options.prefilter_visibility = select, prefilter
        options.collect_debug_counts = True
        out_d = self.scene_d.accum_dfr_coherent_direct(
            self.drjit_states(rows), grid_d, material_d, options, self.mods.cuda.Bool([True] * len(rows))
        )
        return out_t, out_d

    def compare_coherent(self, op, out_t, out_d):
        self.assertEqual(out_t.grid_cell_count, int(out_d.grid_cell_count))
        for axis in "xyz":
            field = getattr(out_d, f"direct_field_{axis}")
            for part, value in (("re", field.real), ("im", field.imag)):
                self.close(
                    op,
                    f"direct_field_{axis}_{part}",
                    getattr(out_t, f"direct_field_{axis}_{part}"),
                    value,
                    FIELD_ABS,
                    FIELD_REL,
                )
        for name in ("direct_count", "multi_count", "vis_rejects", "utd_rejects"):
            self.equal(op, name, getattr(out_t, name), getattr(out_d, name))

    def test_accum_dfr_coherent_direct(self):
        # One state per call: multi-state coherent accumulation differs (see the expected failure below).
        for state in range(3):
            for select, prefilter in ((True, True), (False, True), (True, False)):
                with self.subTest(state=state, select_diffraction_point=select, prefilter_visibility=prefilter):
                    out_t, out_d = self._coherent_pair(_dfr_state_rows(3)[state : state + 1], select, prefilter)
                    self.assertEqual(int(bs.host(out_t.direct_count).sum()), 16)
                    self.compare_coherent("accum_dfr_coherent_direct", out_t, out_d)

    def test_accum_dfr_coherent_direct_multi_state(self):
        # With three states over a 4x4 grid the coherent field is the sum of the three single-state fields and
        # direct_count is 3 * 16 = 48 on both backends: each (state, cell) lane weighs its whole edge length, and the
        # warp leader of same-cell lanes is picked by the %laneid register (include/rayd/math.h).
        rows = _dfr_state_rows(3)
        singles = [self._coherent_pair(rows[index : index + 1]) for index in range(3)]
        out_t, out_d = self._coherent_pair(rows)
        summed_d = sum(bs.host(single[1].direct_field_x.real) for single in singles)
        self.close(
            "accum_dfr_coherent_direct[3 states]",
            "drjit_superposition",
            out_d.direct_field_x.real,
            summed_d,
            FIELD_ABS,
            FIELD_REL,
        )
        self.assertEqual(int(bs.host(out_t.direct_count).sum()), 48)
        self.compare_coherent("accum_dfr_coherent_direct[3 states]", out_t, out_d)


if __name__ == "__main__":
    unittest.main()
