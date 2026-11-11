# Copyright Xingyu Chen.
# Checks Dr.Jit multipath VJP and JVP against central differences and unsupported domains against loud refusal.

"""Finite-difference gradient checks for the Dr.Jit multipath operations.

Every check is one directional derivative per input domain (see `tests/support/multipath_fd_cases.py`): the
reverse-mode product from `dr.backward`, the forward-mode tangent from `dr.forward_to` and one central-difference
pair must agree. `rayd.drjit.api_manifest()["derivatives"]` decides the domains: each `supported` multipath domain
has a check here and each `unsupported` one must raise when its inputs carry derivatives.

Steps are `h = 2e-3` for O(1) geometry. Reflection and EPC fields use wavelength 0.5 (k ~ 12.6, so the phase moves
k h ~ 0.025 rad per step and the O((k h)^2) truncation stays below 1e-4 relative); diffraction path export uses
wavelength 0.125 and `h = 5e-4` for the same reason. Every configuration keeps hits away from triangle edges, avoids
normal incidence, and the harness asserts identical hit primitives, validity and sample counts at x - h v, x and
x + h v.
"""

from __future__ import annotations

import functools
import math
import unittest

import numpy as np

from tests.support.backends import drjit_modules
from tests.support.multipath_fd_cases import (
    MULTIPATH_FAMILIES,
    assert_directional,
    drjit_array,
    drjit_directional,
    manifest_domains,
)


DJ = drjit_modules()
F32 = np.float32
STEP = 2.0e-3

REFLECTION_VERTICES = np.array(
    [[-3.0, -3.0, 0.0], [5.0, -3.0, 0.0], [-3.0, 5.0, 0.0], [1.5, -3.0, -1.0], [1.5, 3.0, -1.0], [1.8, 0.0, 4.0]], F32
)
REFLECTION_ORIGINS = np.array([[-1.0, 0.1, 1.0], [-0.8, -0.3, 1.2], [-1.2, 0.4, 0.9]], F32)
_directions = np.array([[1.0, 0.05, -1.0], [1.0, 0.1, -1.1], [1.1, -0.05, -0.9]], np.float64)
REFLECTION_DIRECTIONS = (_directions / np.linalg.norm(_directions, axis=1, keepdims=True)).astype(F32)

EPC_BASE = {
    "vertices": np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]], F32),
    "tx": np.array([[0.1, 0.05, 1.0]], F32),
    "rx": np.array([[0.6, -0.1, 1.2]], F32),
    "plane_point": np.array([[0.0, 0.0, 0.0]], F32),
    "plane_normal": np.array([[0.0, 0.0, 1.0]], F32),
    "eta_r": np.array([4.0], F32),
    "sigma": np.array([0.02], F32),
    "gain": np.array([0.9], F32),
}

# Two bounces: the floor of the reflection scene, then the wall plane x = 1.5, one material slot per bounce.
EPC2_BASE = {
    "vertices": np.array(
        [[-3.0, -3.0, 0.0], [5.0, -3.0, 0.0], [-3.0, 5.0, 0.0], [1.5, -3.0, -1.0], [1.5, 3.0, -1.0], [1.5, 0.0, 4.0]],
        F32,
    ),
    "tx": np.array([[-1.0, 0.1, 1.0]], F32),
    "rx": np.array([[-0.5, -0.2, 1.5]], F32),
    "plane_point": np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], F32),
    "plane_normal": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], F32),
    "eta_r": np.array([4.0, 3.0], F32),
    "sigma": np.array([0.02, 0.01], F32),
    "gain": np.array([0.9, 0.8], F32),
}

_accum_directions = np.array([[0.1, 0.05, -1.0], [-0.2, 0.1, -1.0], [0.15, -0.2, -1.0]], np.float64)
ACCUM_BASE = {
    "vertices": np.array([[-4.0, -4.0, 0.0], [6.0, -4.0, 0.0], [-4.0, 6.0, 0.0]], F32),
    "o": np.repeat(np.array([[0.1, 0.05, 1.0]], F32), 3, axis=0),
    "d": (_accum_directions / np.linalg.norm(_accum_directions, axis=1, keepdims=True)).astype(F32),
    "tx": np.array([[0.1, 0.05, 1.0]], F32),
    "eta_r": np.array([4.0], F32),
    "sigma": np.array([0.02], F32),
    "gain": np.array([0.9], F32),
}

DIFFRACTION_BASE = {
    "edge_pos": np.array([[0.0, 0.0, 0.0]], F32),
    "edge_dir": np.array([[1.0, 0.0, 0.0]], F32),
    "edge_t_min": np.array([-0.5], F32),
    "edge_t_max": np.array([0.5], F32),
    "exterior_angle": np.array([1.5 * math.pi], F32),
    "src": np.array([[0.3, 0.2, 1.0]], F32),
    "src_power": np.array([2.0], F32),
    "gain": np.array([0.8], F32),
    "eta_r": np.array([4.0], F32),
    "rec_edge_pos": np.array([[0.0, 0.5, 0.0]], F32),
    "rec_edge_dir": np.array([[1.0, 0.0, 0.0]], F32),
    "rec_exterior_angle": np.array([1.5 * math.pi], F32),
    "tx": np.array([[0.1, 0.05, 1.0], [-0.1, 0.2, 1.2]], F32),
    "rx": np.array([[0.05, -0.1, -1.0], [0.2, 0.1, -0.9]], F32),
}

DIRECT_GEOMETRY_GROUPS = (
    ("edge_pos",),
    ("edge_dir",),
    ("edge_t_min", "edge_t_max"),
    ("exterior_angle",),
    ("src",),
    ("src_power",),
    ("edge_pos", "edge_dir", "edge_t_min", "edge_t_max", "exterior_angle", "src", "src_power"),
)
CHAIN_GEOMETRY_GROUPS = (
    ("edge_pos",),
    ("rec_edge_pos",),
    ("edge_dir", "rec_edge_dir"),
    ("exterior_angle", "rec_exterior_angle"),
    ("src", "src_power"),
)

COVERED_SUPPORTED = {
    ("trace_reflections", "geometry"),
    ("trace_reflections", "ray"),
    ("trace_refl_epc_field", "geometry"),
    ("trace_refl_epc_field", "source"),
    ("trace_refl_epc_field", "receiver"),
    ("trace_refl_epc_field", "material"),
    ("accum_refl", "geometry"),
    ("accum_refl", "source_receiver"),
    ("accum_refl", "material"),
    ("trace_dfr_paths", "continuous_inputs"),
    ("accum_dfr_direct", "geometry"),
    ("accum_dfr_direct", "material"),
    ("accum_dfr", "geometry"),
    ("accum_dfr", "material"),
}


def _signature(*arrays) -> tuple:
    return tuple(tuple(np.array(array).tolist()) for array in arrays)


def _reflection_objective(bounces: int):
    dr, cuda, ad, pj = DJ.dr, DJ.cuda, DJ.ad, DJ.pj
    faces = cuda.Array3i([0, 3], [1, 4], [2, 5])

    def objective(x):
        scene = pj.Scene()
        scene.add_mesh(pj.Mesh(x["vertices"], faces))
        scene.build()
        chain = scene.trace_reflections(pj.RayAD(x["o"], x["d"]), max_bounces=bounces, symbolic=False)
        count = chain.ray_count * chain.max_bounces
        weight = dr.linspace(ad.Float, 0.5, 1.5, count)
        signed = dr.linspace(ad.Float, -1.0, 1.0, count)
        hit, image = chain.hit_points, chain.image_sources
        value = dr.sum(weight * chain.t) + dr.sum(signed * (hit.x + 0.5 * hit.y - 0.3 * hit.z))
        value += dr.sum(signed * (image.x - image.z))
        return value, _signature(chain.prim_ids, chain.is_valid())

    return objective


def _epc_field_objective(x, *, path_weight=1.0, field_weight=30.0):
    dr, cuda, ad, pj = DJ.dr, DJ.cuda, DJ.ad, DJ.pj
    bounces = len(np.array(x["eta_r"]))
    scene = pj.Scene()
    triangles = len(np.array(x["vertices"].x)) // 3
    scene.add_mesh(pj.Mesh(x["vertices"], cuda.Array3i(*[[3 * t + k for t in range(triangles)] for k in range(3)])))
    scene.build()
    options = pj.ReflEpcFieldOptionsAD()
    options.expected_prim_ids = cuda.Int(list(range(bounces)))
    options.slot_plane_point = x["plane_point"]
    options.slot_plane_normal = x["plane_normal"]
    options.slot_eta_r = x["eta_r"]
    options.slot_mu_r = ad.Float([1.0] * bounces)
    options.slot_sigma = x["sigma"]
    options.slot_gain = x["gain"]
    options.tx_polarization = ad.Array3f([1.0], [0.0], [0.0])
    options.wavelength = 0.5
    options.omega = 2.0 * math.pi * 299792458.0 / 0.5
    options.return_geom = True
    result = scene.trace_refl_epc_field(x["tx"], x["rx"], bounces, options, ad.Bool([True]))
    value = path_weight * dr.sum(result.path_length)
    value += field_weight * (dr.sum(result.field_x_re) - 0.7 * dr.sum(result.field_x_im))
    value += field_weight * (0.3 * dr.sum(result.field_y_re) + 1.5 * dr.sum(result.field_z_im))
    return value, _signature(result.valid, result.bounce_count)


def _accum_objective(x):
    dr, cuda, ad, pj = DJ.dr, DJ.cuda, DJ.ad, DJ.pj
    scene = pj.Scene()
    scene.add_mesh(pj.Mesh(x["vertices"], cuda.Array3i([0], [1], [2])))
    scene.build()
    grid = pj.AccumGrid()
    grid.axis, grid.position = 2, 0.6
    grid.coord0_min, grid.coord0_max, grid.coord1_min, grid.coord1_max = -4.0, 4.0, -4.0, 4.0
    grid.resolution0, grid.resolution1 = 1, 1
    material = pj.ReflMaterialAD()
    material.eta_r, material.sigma, material.gain = x["eta_r"], x["sigma"], x["gain"]
    material.mu_r, material.valid = ad.Float([1.0]), ad.Bool([True])
    options = pj.AccumOptions()
    options.wavelength, options.k = 0.5, 2.0 * math.pi / 0.5
    options.solid_angle_per_ray, options.cell_area = 0.01, 1.0
    result = scene.accum_refl(pj.RayAD(x["o"], x["d"]), x["tx"], grid, material, 1, options)
    fx, fy, fz = result.reflection_field_x, result.reflection_field_y, result.reflection_field_z
    value = 1.0e4 * dr.sum(result.reflection_power)
    value += 300.0 * (dr.sum(fx.real) - 0.7 * dr.sum(fx.imag) + 0.3 * dr.sum(fy.real) + 0.5 * dr.sum(fz.imag))
    return value, _signature(result.reflection_count)


def _dfr_scene():
    cuda, pj = DJ.cuda, DJ.pj
    scene = pj.Scene()
    scene.add_mesh(
        pj.Mesh(cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0]), cuda.Array3i([0], [1], [2]))
    )
    scene.build()
    return scene


def _dfr_states(x, recursive=False):
    ad, pj = DJ.ad, DJ.pj
    prefix = "rec_" if recursive else ""
    states = pj.DfrStatesAD()
    states.count = 1
    states.edge_index = ad.Int([1 if recursive else 0])
    states.edge_pos = x[prefix + "edge_pos"]
    states.edge_dir = x[prefix + "edge_dir"]
    states.edge_t_min = x["edge_t_min"]
    states.edge_t_max = x["edge_t_max"]
    states.n0 = ad.Array3f([0.0], [1.0], [0.0])
    states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
    states.prim0 = ad.Int([0])
    states.prim1 = ad.Int([0])
    states.exterior_angle = x[prefix + "exterior_angle"]
    states.src = ad.Array3f([0.0], [0.0], [1.0]) if recursive else x["src"]
    states.src_power = ad.Float([1.0]) if recursive else x["src_power"]
    states.wi = ad.Array3f([0.0], [1.0], [0.0]) if recursive else ad.Array3f([0.0], [0.0], [-1.0])
    states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
    states.prefix_depth = ad.Int([0])
    return states


def _dfr_material(x):
    ad, pj = DJ.ad, DJ.pj
    material = pj.DfrMaterialAD()
    material.eta_r = x["eta_r"] if "eta_r" in x else ad.Float([4.0])
    material.sigma, material.mu_r = ad.Float([0.0]), ad.Float([1.0])
    material.gain, material.valid = x["gain"], ad.Bool([True])
    return material


def _dfr_grid(axis=2, position=-1.0, resolution=2):
    grid = DJ.pj.DfrGrid()
    grid.axis, grid.position = axis, position
    grid.coord0_min, grid.coord0_max, grid.coord1_min, grid.coord1_max = -1.0, 1.0, -1.0, 1.0
    grid.resolution0 = grid.resolution1 = resolution
    grid.cell_area = 4.0 / float(resolution * resolution)
    return grid


def _dfr_options(*, direct=0, keller=0, suffix=0, max_order=1, seed=17):
    pj = DJ.pj
    options = pj.DfrOptions()
    options.wavelength, options.k, options.seed = 0.125, 2.0 * math.pi / 0.125, seed
    options.samples, options.max_order = direct + keller + suffix, max_order
    options.direct_samples, options.keller_samples, options.suffix_samples = direct, keller, suffix
    options.strategy_mask = (
        (pj.DfrStrategy.DIRECT if direct else 0)
        | (pj.DfrStrategy.KELLER if keller else 0)
        | (pj.DfrStrategy.SUFFIX_REFL if suffix else 0)
    )
    options.sample_sequence = pj.DfrSampleSequence.HASH
    options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
    options.collect_edge_use = True
    options.collect_debug_counts = True
    return options


def _grid_objective(result, scale=1.0):
    dr, ad = DJ.dr, DJ.ad
    cells = int(result.grid_cell_count)
    weight = 0.5 + dr.arange(ad.Float, cells) / float(cells)
    value = scale * (100.0 * dr.sum(weight * result.power) + 10.0 * dr.sum(weight * result.field_x.real))
    return value, _signature(result.direct_count, result.keller_count, result.suffix_count, result.edge_uses)


def _direct_objective(scene, *, direct, keller):
    def objective(x):
        result = scene.accum_dfr_direct(
            _dfr_states(x), _dfr_grid(), _dfr_material(x), _dfr_options(direct=direct, keller=keller), True
        )
        return _grid_objective(result)

    return objective


def _chain_objective(scene, *, direct, keller):
    def objective(x):
        result = scene.accum_dfr(
            _dfr_states(x),
            _dfr_states(x, recursive=True),
            _dfr_grid(),
            _dfr_material(x),
            _dfr_options(direct=direct, keller=keller, max_order=2),
            True,
        )
        return _grid_objective(result)

    return objective


def _suffix_objective():
    cuda, pj = DJ.cuda, DJ.pj
    mirror = pj.Scene()
    mirror.add_mesh(
        pj.Mesh(cuda.Array3f([-2.0, 2.0, -2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, 2.0]), cuda.Array3i([0], [1], [2]))
    )
    mirror.build()

    def objective(x):
        states = _dfr_states(
            {
                **x,
                "edge_t_min": DJ.ad.Float([-0.25]),
                "edge_t_max": DJ.ad.Float([0.25]),
                "exterior_angle": DJ.ad.Float([1.5 * math.pi]),
                "src_power": DJ.ad.Float([1.0]),
            }
        )
        result = mirror.accum_dfr_direct(
            states,
            _dfr_grid(axis=1, position=-2.0, resolution=1),
            _dfr_material(x),
            _dfr_options(suffix=16, seed=41),
            True,
        )
        # The single suffix cell receives ~1e-5 of the source power; rescale above the float32 FD resolution.
        return _grid_objective(result, scale=100.0)

    return objective


SUFFIX_BASE = {
    "edge_pos": np.array([[0.0, -1.0, 0.0]], F32),
    "edge_dir": np.array([[1.0, 0.0, 0.0]], F32),
    "src": np.array([[0.05, -1.0, 1.0]], F32),
    "gain": np.array([0.8], F32),
}


def _paths_objective(scene):
    dr, ad, pj = DJ.dr, DJ.ad, DJ.pj

    def objective(x):
        options = pj.DfrPathOptions()
        options.wavelength, options.k, options.seed = 0.125, 2.0 * math.pi / 0.125, 17
        options.max_order, options.max_paths, options.max_rx, options.sample_count = 1, 8, 2, 1
        options.strategy_mask, options.return_geom = pj.DfrStrategy.DIRECT, 1
        options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
        paths = scene.trace_dfr_paths(x["tx"], x["rx"], _dfr_states(x), _dfr_material(x), options, ad.Bool([True]))
        weight = dr.linspace(ad.Float, 0.5, 1.5, int(paths.capacity))
        terms = weight * paths.delay * 1.0e8 + weight * (paths.p0.x + paths.p0.y + paths.p0.z)
        terms += 300.0 * (weight * paths.field_x.real - 0.7 * paths.field_x.imag + 0.4 * paths.field_y.real)
        value = dr.sum(dr.select(paths.valid, terms, 0.0))
        return value, _signature(paths.valid, paths.count, paths.rx_id, paths.tx_id, paths.edge0)

    return objective


@unittest.skipUnless(DJ is not None, "a CUDA-capable rayd.drjit is required")
class DrJitReflectionFiniteDifferenceTests(unittest.TestCase):
    def test_trace_reflections_geometry_and_ray_match_central_differences(self):
        base = {"vertices": REFLECTION_VERTICES, "o": REFLECTION_ORIGINS, "d": REFLECTION_DIRECTIONS}
        for bounces in (1, 2):
            objective = _reflection_objective(bounces)
            for seed, group in enumerate((("vertices",), ("o",), ("d",), ("vertices", "o", "d"))):
                with self.subTest(bounces=bounces, group=group):
                    result = drjit_directional(self, DJ, objective, base, group, step=STEP, seed=seed)
                    assert_directional(self, result, label=f"drjit trace_reflections b={bounces} {group}")

    def test_trace_refl_epc_field_domains_match_central_differences(self):
        groups = {
            "geometry": ("plane_point", "plane_normal"),
            "source": ("tx",),
            "receiver": ("rx",),
            "material": ("eta_r", "sigma", "gain"),
        }
        # The path length dominates the objective but does not depend on the material, so the material check uses
        # the fields alone to stay far above the float32 finite-difference resolution.
        field_only = functools.partial(_epc_field_objective, path_weight=0.0)
        for bounces, base in ((1, EPC_BASE), (2, EPC2_BASE)):
            for seed, (domain, group) in enumerate(groups.items()):
                objective = field_only if domain == "material" else _epc_field_objective
                with self.subTest(bounces=bounces, domain=domain):
                    result = drjit_directional(self, DJ, objective, base, group, step=STEP, seed=seed)
                    assert_directional(self, result, label=f"drjit trace_refl_epc_field b={bounces} {domain}")

    def test_trace_refl_epc_field_ignores_mesh_vertices(self):
        # The field is evaluated on the caller's slot planes, so the mesh only resolves visibility: its vertex
        # derivative is exactly zero, and the central difference must agree.
        result = drjit_directional(self, DJ, _epc_field_objective, EPC_BASE, ("vertices",), step=STEP, seed=7)
        self.assertEqual(result.vjp, 0.0)
        self.assertEqual(result.jvp, 0.0)
        self.assertEqual(result.fd, 0.0)

    def test_accumulate_reflections_domains_match_central_differences(self):
        groups = {
            "geometry": ("vertices",),
            "source_receiver/origin": ("o",),
            "source_receiver/direction": ("d",),
            "source_receiver/tx": ("tx",),
            "source_receiver": ("o", "d", "tx"),
            "material": ("eta_r", "sigma", "gain"),
        }
        for seed, (domain, group) in enumerate(groups.items()):
            with self.subTest(domain=domain):
                result = drjit_directional(self, DJ, _accum_objective, ACCUM_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"drjit accum_refl {domain}")

    # At exact normal incidence the s/p basis is undefined; the field derivative must still equal its limit from
    # tilted rays (and the central difference) in both operations that reflect a polarised field.
    def test_reflected_field_derivative_is_continuous_at_normal_incidence(self):
        normal = {**ACCUM_BASE, "o": ACCUM_BASE["tx"], "d": np.array([[0.0, 0.0, -1.0]], F32)}
        with self.subTest(operation="accum_refl"):
            result = drjit_directional(self, DJ, _accum_objective, normal, ("d",), step=STEP, seed=3)
            assert_directional(self, result, label="drjit accum_refl direction at normal incidence")
        epc = {**EPC_BASE, "tx": np.array([[0.0, 0.0, 1.0]], F32), "rx": np.array([[0.0, 0.0, 2.0]], F32)}
        field_only = functools.partial(_epc_field_objective, path_weight=0.0)
        with self.subTest(operation="trace_refl_epc_field"):
            result = drjit_directional(self, DJ, field_only, epc, ("rx",), step=STEP, seed=3)
            assert_directional(self, result, label="drjit trace_refl_epc_field receiver at normal incidence")


@unittest.skipUnless(DJ is not None, "a CUDA-capable rayd.drjit is required")
class DrJitDiffractionFiniteDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = _dfr_scene()

    def test_trace_dfr_paths_continuous_inputs_match_central_differences(self):
        groups = (
            ("rx",),
            ("edge_pos",),
            ("src",),
            ("edge_t_min", "edge_t_max"),
            ("exterior_angle",),
            ("src_power",),
            ("gain", "eta_r"),
            ("rx", "edge_pos", "src", "edge_t_min", "edge_t_max", "exterior_angle", "src_power", "gain"),
        )
        objective = _paths_objective(self.scene)
        for seed, group in enumerate(groups):
            with self.subTest(group=group):
                result = drjit_directional(self, DJ, objective, DIFFRACTION_BASE, group, step=5.0e-4, seed=seed)
                assert_directional(self, result, label=f"drjit trace_dfr_paths {group}")

    def test_accum_dfr_direct_direct_strategy_matches_central_differences(self):
        objective = _direct_objective(self.scene, direct=64, keller=0)
        for seed, group in enumerate(DIRECT_GEOMETRY_GROUPS + (("gain",),)):
            with self.subTest(group=group):
                result = drjit_directional(self, DJ, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"drjit accum_dfr_direct direct {group}")

    def test_accum_dfr_direct_suffix_strategy_matches_central_differences(self):
        objective = _suffix_objective()
        for seed, group in enumerate((("edge_pos",), ("edge_dir",), ("src",), ("gain",))):
            with self.subTest(group=group):
                result = drjit_directional(self, DJ, objective, SUFFIX_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"drjit accum_dfr_direct suffix {group}")

    def test_accum_dfr_chain_direct_strategy_matches_central_differences(self):
        objective = _chain_objective(self.scene, direct=64, keller=0)
        for seed, group in enumerate(CHAIN_GEOMETRY_GROUPS + (("gain",),)):
            with self.subTest(group=group):
                result = drjit_directional(self, DJ, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"drjit accum_dfr chain direct {group}")

    # Keller-cone lanes: the derivative tape rebuilds the forward Keller contribution, including the cone around
    # the source-to-edge-point direction and the measure-scale jacobian, so every supported domain is checked.
    def test_accum_dfr_direct_keller_strategy_matches_central_differences(self):
        objective = _direct_objective(self.scene, direct=0, keller=64)
        for seed, group in enumerate((("gain",), *DIRECT_GEOMETRY_GROUPS)):
            with self.subTest(group=group):
                result = drjit_directional(self, DJ, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"drjit accum_dfr_direct keller {group}")

    def test_accum_dfr_chain_keller_strategy_matches_central_differences(self):
        objective = _chain_objective(self.scene, direct=0, keller=64)
        for seed, group in enumerate((("gain",), *CHAIN_GEOMETRY_GROUPS)):
            with self.subTest(group=group):
                result = drjit_directional(self, DJ, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"drjit accum_dfr chain keller {group}")


def _epc_mirror(vertices):
    scene = DJ.pj.Scene()
    scene.add_mesh(DJ.pj.Mesh(vertices, DJ.cuda.Array3i([0], [1], [2])))
    scene.build()
    return scene


def _epc_vertices():
    return drjit_array(DJ.ad, EPC_BASE["vertices"])


def _epc_ray(kind):
    origin, direction = [[-0.5], [0.0], [1.0]], [[0.4472136], [0.0], [-0.8944272]]
    if kind == "ad":
        return DJ.pj.RayAD(DJ.ad.Array3f(*origin), DJ.ad.Array3f(*direction))
    return DJ.pj.Ray(DJ.cuda.Array3f(*origin), DJ.cuda.Array3f(*direction))


def _coherent(states, material):
    pj = DJ.pj
    options = pj.DfrCoherentOptions()
    options.wavelength, options.k, options.max_order = 0.125, 2.0 * math.pi / 0.125, 1
    options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
    return _dfr_scene().accum_dfr_coherent_direct(states, _dfr_grid(resolution=1), material, options, True)


def _grad_enabled(array):
    DJ.dr.enable_grad(array)
    return array


@unittest.skipUnless(DJ is not None, "a CUDA-capable rayd.drjit is required")
class DrJitMultipathDerivativeManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = DJ.pj.api_manifest()

    def test_every_supported_multipath_domain_has_a_finite_difference_check(self):
        self.assertEqual(manifest_domains(self.manifest, MULTIPATH_FAMILIES, "supported"), COVERED_SUPPORTED)

    def test_unsupported_multipath_domains_are_the_ones_refused_below(self):
        self.assertEqual(
            manifest_domains(self.manifest, MULTIPATH_FAMILIES, "unsupported"),
            {
                ("trace_refl_epc", "geometry"),
                ("trace_refl_epc", "ray"),
                ("trace_refl_epc", "receiver"),
                ("trace_refl_epc", "plane_geometry"),
                ("accum_dfr_coherent_direct", "continuous_inputs"),
            },
        )

    def test_trace_refl_epc_rejects_differentiable_rays(self):
        # The receiver and the plane geometry of Dr.Jit EPC discovery have no AD overload at all; the ray does.
        receiver = DJ.cuda.Array3f([0.5], [0.0], [1.0])
        with self.assertRaises(TypeError):
            _epc_mirror(_epc_vertices()).trace_refl_epc(_epc_ray("ad"), receiver, 1)

    def test_accum_dfr_coherent_direct_rejects_differentiable_states_and_material(self):
        base = {name: drjit_array(DJ.ad, value) for name, value in DIFFRACTION_BASE.items()}
        for name in ("edge_pos", "src", "gain"):
            with self.subTest(input=name):
                inputs = {**base, name: _grad_enabled(drjit_array(DJ.ad, DIFFRACTION_BASE[name]))}
                with self.assertRaisesRegex(RuntimeError, "AD inputs are not supported"):
                    _coherent(_dfr_states(inputs), _dfr_material(inputs))

    # The manifest marks trace_refl_epc geometry and receiver derivatives unsupported: a grad-enabled receiver or
    # scene geometry must be refused rather than traced detached, and drjit.suspend_grad() opts out explicitly.
    def test_trace_refl_epc_rejects_differentiable_receiver_and_geometry(self):
        receiver = _grad_enabled(DJ.ad.Array3f([0.5], [0.0], [1.0]))
        with self.assertRaisesRegex(RuntimeError, "forward-only.*receiver"):
            _epc_mirror(_epc_vertices()).trace_refl_epc(_epc_ray("plain"), receiver, 1)
        scene = _epc_mirror(_grad_enabled(_epc_vertices()))
        with self.assertRaisesRegex(RuntimeError, "forward-only.*geometry"):
            scene.trace_refl_epc(_epc_ray("plain"), DJ.cuda.Array3f([0.5], [0.0], [1.0]), 1)
        with DJ.dr.suspend_grad():
            result = scene.trace_refl_epc(_epc_ray("plain"), receiver, 1)
        self.assertEqual(DJ.dr.width(result.path_length), 1)

    def test_detached_records_refuse_grad_enabled_arrays(self):
        # Detached records and overloads would drop the derivative of a grad-enabled AD array in an implicit
        # conversion; the conversion is refused and the AD variant (here ReflMaterialAD, RayAD) must be used.
        ad, pj = DJ.ad, DJ.pj
        for record, field, value in (
            (pj.ReflMaterial(), "gain", ad.Float([0.9])),
            (pj.DfrMaterial(), "gain", ad.Float([0.9])),
            (pj.DfrStates(), "src", ad.Array3f([0.0], [0.0], [1.0])),
            (pj.ReflEpcFieldOptions(), "slot_eta_r", ad.Float([4.0])),
            (pj.Ray(), "o", ad.Array3f([0.0], [0.0], [1.0])),
        ):
            with self.subTest(record=type(record).__name__, field=field):
                setattr(record, field, value)  # an AD array without gradients still converts
                with self.assertRaises(TypeError):
                    setattr(record, field, _grad_enabled(value))
        material = pj.ReflMaterialAD()
        material.gain = _grad_enabled(ad.Float([0.9]))
        self.assertTrue(DJ.dr.grad_enabled(material.gain))


if __name__ == "__main__":
    unittest.main()
