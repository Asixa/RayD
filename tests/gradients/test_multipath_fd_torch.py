# Copyright Xingyu Chen.
# Checks Torch multipath VJP and JVP against central differences and unsupported domains against loud refusal.

"""Finite-difference gradient checks for the Torch multipath operations.

Every check is one directional derivative per input domain (see `tests/support/multipath_fd_cases.py`). Which
domains are checked is read from `rayd.torch.api_manifest()["derivatives"]`: each `supported` multipath domain
must have a check here, and each `unsupported` domain must raise for both reverse-mode and forward-mode inputs
instead of returning a detached or zero derivative.

Step sizes: geometry is O(1) in size and the diffraction accumulators deposit `sqrt` and inverse-square terms that
are smooth on the scale of the inputs, so `h = 2e-3` keeps the O(h^2) truncation near 1e-6 relative while the
float32 cancellation error `8 * 2^-24 * |f| / h` stays near 1e-4 * |f|. Configurations keep every hit and every
Monte-Carlo sample well inside its triangle or grid cell, and the harness asserts that hit primitives and sample
counts are identical at x - h v, x and x + h v.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.support.backends import torch_cuda_available
from tests.support.multipath_fd_cases import MULTIPATH_FAMILIES, assert_directional, manifest_domains, torch_directional

if torch_cuda_available():
    import torch
    import rayd.torch as rt


F32 = np.float32
STEP = 2.0e-3

# Floor plus a tilted wall: one large triangle each, so no hit lies near a triangle edge.
REFLECTION_VERTICES = np.array(
    [[-3.0, -3.0, 0.0], [5.0, -3.0, 0.0], [-3.0, 5.0, 0.0], [1.5, -3.0, -1.0], [1.5, 3.0, -1.0], [1.8, 0.0, 4.0]], F32
)
REFLECTION_FACES = [[0, 1, 2], [3, 4, 5]]
REFLECTION_ORIGINS = np.array([[-1.0, 0.1, 1.0], [-0.8, -0.3, 1.2], [-1.2, 0.4, 0.9]], F32)
_directions = np.array([[1.0, 0.05, -1.0], [1.0, 0.1, -1.1], [1.1, -0.05, -0.9]], np.float64)
REFLECTION_DIRECTIONS = (_directions / np.linalg.norm(_directions, axis=1, keepdims=True)).astype(F32)

# A wedge edge along x through the origin with a source above it and the grid below it; the material row is the
# single face of a far-away triangle so that `prim0 = prim1 = 0` selects a material without blocking any path.
DIFFRACTION_BASE = {
    "edge_pos": np.array([[0.0, 0.0, 0.0]], F32),
    "edge_dir": np.array([[1.0, 0.0, 0.0]], F32),
    "edge_t_min": np.array([-0.5], F32),
    "edge_t_max": np.array([0.5], F32),
    "exterior_angle": np.array([1.5 * math.pi], F32),
    "src": np.array([[0.3, 0.2, 1.0]], F32),
    "src_power": np.array([2.0], F32),
    "gain": np.array([0.8], F32),
    "rec_edge_pos": np.array([[0.0, 0.5, 0.0]], F32),
    "rec_edge_dir": np.array([[1.0, 0.0, 0.0]], F32),
    "rec_exterior_angle": np.array([1.5 * math.pi], F32),
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


def _t(value, dtype=None):
    import torch

    return torch.tensor(value, device="cuda", dtype=dtype or torch.float32)


def _reflection_objective(bounces: int):
    import torch

    faces = _t(REFLECTION_FACES, torch.int32)

    def objective(x):
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(x["vertices"], faces))
        scene.build()
        chain = scene.trace_reflections(rt.Ray(x["o"], x["d"]), max_bounces=bounces)
        t_weight = torch.linspace(0.5, 1.5, chain.t.numel(), device="cuda").reshape(chain.t.shape)
        image_weight = torch.linspace(-1.0, 1.0, chain.image_sources.numel(), device="cuda").reshape(
            chain.image_sources.shape
        )
        value = (t_weight * chain.t).sum() + (image_weight * chain.image_sources).sum()
        return value, (chain.valid.cpu().tolist(), chain.prim_ids.cpu().tolist())

    return objective


def _dfr_scene():
    scene = rt.Scene()
    vertices = _t([[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]])
    scene.add_mesh(rt.Mesh(vertices, _t([[0, 1, 2]], torch.int32)))
    scene.build()
    return scene


def _dfr_states(x, recursive: bool = False):
    prefix = "rec_" if recursive else ""
    return rt.DfrStates(
        edge_index=_t([1 if recursive else 0], torch.int32),
        edge_pos=x[prefix + "edge_pos"],
        edge_dir=x[prefix + "edge_dir"],
        edge_t_min=x["edge_t_min"],
        edge_t_max=x["edge_t_max"],
        n0=_t([[0.0, 1.0, 0.0]]),
        n1=_t([[0.0, -1.0, 0.0]]),
        prim0=_t([0], torch.int32),
        prim1=_t([0], torch.int32),
        exterior_angle=x[prefix + "exterior_angle"],
        src=_t([[0.0, 0.0, 1.0]]) if recursive else x["src"],
        src_power=_t([1.0]) if recursive else x["src_power"],
        wi=_t([[0.0, 1.0, 0.0]]) if recursive else _t([[0.0, 0.0, -1.0]]),
        d0=_t([[0.0, 0.0, -1.0]]),
        count=1,
    )


def _dfr_material(x):
    return rt.DfrMaterial(
        eta_r=_t([4.0]), sigma=_t([0.0]), mu_r=_t([1.0]), gain=x["gain"], valid=_t([True], torch.bool)
    )


def _grid_objective(out):
    weight = torch.linspace(0.5, 1.5, out.power.numel(), device="cuda").reshape(out.power.shape)
    value = 100.0 * (weight * out.power).sum() + 10.0 * (weight * out.field_x_re).sum()
    counts = torch.stack((out.direct_count, out.keller_count, out.suffix_count, out.edge_uses)).cpu().tolist()
    return value, counts


def _direct_objective(scene, *, direct: int, keller: int):
    def objective(x):
        out = scene.accum_dfr_direct(
            states=_dfr_states(x),
            grid=rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2),
            material=_dfr_material(x),
            options=rt.DfrOptions(wavelength=0.125, seed=17, direct_samples=direct, keller_samples=keller),
        )
        return _grid_objective(out)

    return objective


def _chain_objective(scene, *, direct: int, keller: int):
    def objective(x):
        out = scene.accum_dfr(
            initial_states=_dfr_states(x),
            recursive_states=_dfr_states(x, recursive=True),
            grid=rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2),
            material=_dfr_material(x),
            options=rt.DfrOptions(wavelength=0.125, seed=17, direct_samples=direct, keller_samples=keller, max_order=2),
        )
        return _grid_objective(out)

    return objective


def _suffix_objective():
    mirror = rt.Scene()
    mirror.add_mesh(rt.Mesh(_t([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [-2.0, 0.0, 2.0]]), _t([[0, 1, 2]], torch.int32)))
    mirror.build()

    def objective(x):
        states = rt.DfrStates(
            edge_index=_t([0], torch.int32),
            edge_pos=x["edge_pos"],
            edge_dir=x["edge_dir"],
            edge_t_min=_t([-0.25]),
            edge_t_max=_t([0.25]),
            n0=_t([[0.0, 1.0, 0.0]]),
            n1=_t([[0.0, -1.0, 0.0]]),
            prim0=_t([0], torch.int32),
            prim1=_t([0], torch.int32),
            exterior_angle=_t([1.5 * math.pi]),
            src=x["src"],
            src_power=_t([1.0]),
            wi=_t([[0.0, 0.0, -1.0]]),
            d0=_t([[0.0, 0.0, -1.0]]),
            count=1,
        )
        out = mirror.accum_dfr_direct(
            states=states,
            grid=rt.DfrGrid(axis=1, position=-2.0, cell_area=4.0),
            material=_dfr_material(x),
            options=rt.DfrOptions(wavelength=0.125, seed=41, suffix_samples=16),
        )
        # The single suffix cell receives ~1e-5 of the source power; rescale so the edge-direction sensitivity
        # sits well above the float32 finite-difference resolution.
        value, counts = _grid_objective(out)
        return 100.0 * value, counts

    return objective


SUFFIX_BASE = {
    "edge_pos": np.array([[0.0, -1.0, 0.0]], F32),
    "edge_dir": np.array([[1.0, 0.0, 0.0]], F32),
    "src": np.array([[0.05, -1.0, 1.0]], F32),
    "gain": np.array([0.8], F32),
}

# (operation, domain) pairs of the Torch manifest that the checks below exercise.
COVERED_SUPPORTED = {
    ("trace_reflections", "geometry"),
    ("trace_reflections", "ray"),
    ("accum_dfr_direct", "geometry"),
    ("accum_dfr_direct", "material"),
    ("accum_dfr", "geometry"),
    ("accum_dfr", "material"),
}


@unittest.skipUnless(torch_cuda_available(), "CUDA torch and rayd.torch are required")
class TorchReflectionTraceFiniteDifferenceTests(unittest.TestCase):
    def test_trace_reflections_geometry_and_ray_match_central_differences(self):
        base = {"vertices": REFLECTION_VERTICES, "o": REFLECTION_ORIGINS, "d": REFLECTION_DIRECTIONS}
        for bounces in (1, 2):
            objective = _reflection_objective(bounces)
            for seed, group in enumerate((("vertices",), ("o",), ("d",), ("vertices", "o", "d"))):
                with self.subTest(bounces=bounces, group=group):
                    result = torch_directional(self, objective, base, group, step=STEP, seed=seed)
                    assert_directional(self, result, label=f"torch trace_reflections b={bounces} {group}")


@unittest.skipUnless(torch_cuda_available(), "CUDA torch and rayd.torch are required")
class TorchDiffractionAccumulationFiniteDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = _dfr_scene()

    def test_accum_dfr_direct_direct_strategy_matches_central_differences(self):
        objective = _direct_objective(self.scene, direct=64, keller=0)
        for seed, group in enumerate(DIRECT_GEOMETRY_GROUPS + (("gain",),)):
            with self.subTest(group=group):
                result = torch_directional(self, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"torch accum_dfr_direct direct {group}")

    def test_accum_dfr_direct_suffix_strategy_matches_central_differences(self):
        objective = _suffix_objective()
        for seed, group in enumerate((("edge_pos",), ("edge_dir",), ("src",), ("gain",))):
            with self.subTest(group=group):
                result = torch_directional(self, objective, SUFFIX_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"torch accum_dfr_direct suffix {group}")

    def test_accum_dfr_chain_direct_strategy_matches_central_differences(self):
        objective = _chain_objective(self.scene, direct=64, keller=0)
        for seed, group in enumerate(CHAIN_GEOMETRY_GROUPS + (("gain",),)):
            with self.subTest(group=group):
                result = torch_directional(self, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"torch accum_dfr chain direct {group}")

    # Keller-cone lanes: the derivative tape rebuilds the forward Keller contribution, including the cone around
    # the source-to-edge-point direction and the measure-scale jacobian, so every supported domain is checked.
    def test_accum_dfr_direct_keller_strategy_matches_central_differences(self):
        objective = _direct_objective(self.scene, direct=0, keller=64)
        for seed, group in enumerate((("gain",), *DIRECT_GEOMETRY_GROUPS)):
            with self.subTest(group=group):
                result = torch_directional(self, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"torch accum_dfr_direct keller {group}")

    def test_accum_dfr_chain_keller_strategy_matches_central_differences(self):
        objective = _chain_objective(self.scene, direct=0, keller=64)
        for seed, group in enumerate((("gain",), *CHAIN_GEOMETRY_GROUPS)):
            with self.subTest(group=group):
                result = torch_directional(self, objective, DIFFRACTION_BASE, group, step=STEP, seed=seed)
                assert_directional(self, result, label=f"torch accum_dfr chain keller {group}")


def _mark(value, mode):
    if mode == "vjp":
        return value.clone().requires_grad_(True)
    return torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))


def _mirror_scene(mode=None):
    vertices = _t([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]])
    if mode is not None:
        vertices = _mark(vertices, mode)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, _t([[0, 1, 2]], torch.int32)))
    scene.build()
    return scene


def _epc_options(plane_points=None):
    return rt.ReflEpcOptions(
        expected_prim_ids=_t([[0]], torch.int32),
        direct_plane_points=_t([[[0.0, 0.0, 0.0]]]) if plane_points is None else plane_points,
        direct_plane_normals=_t([[[0.0, 0.0, 1.0]]]),
        surface_group_id=_t([0], torch.int32),
        surface_group_size=_t([1], torch.int32),
        surface_group_members=_t([0], torch.int32),
    )


def _accumulate(scene, *, tx=None, eta_r=None, origin=None):
    source = _t([[0.0, 0.0, 1.0]])
    material = rt.ReflMaterial(
        eta_r=_t([4.0]) if eta_r is None else eta_r,
        sigma=_t([0.0]),
        mu_r=_t([1.0]),
        gain=_t([1.0]),
        valid=_t([True], torch.bool),
    )
    grid = rt.AccumGrid(
        axis=2, position=0.5, coord0_min=-2, coord0_max=2, coord1_min=-2, coord1_max=2, resolution0=1, resolution1=1
    )
    ray = rt.Ray(source if origin is None else origin, _t([[0.1, 0.0, -1.0]]))
    return scene.accum_refl(ray, source if tx is None else tx, grid, material, 1)


def _paths_states(src=None, edge_pos=None):
    return rt.DfrStates(
        edge_index=_t([0], torch.int32),
        edge_pos=_t([[0.0, 0.0, 0.5]]) if edge_pos is None else edge_pos,
        edge_dir=_t([[1.0, 0.0, 0.0]]),
        edge_t_min=_t([-0.5]),
        edge_t_max=_t([0.5]),
        n0=_t([[0.0, 1.0, 0.0]]),
        n1=_t([[0.0, -1.0, 0.0]]),
        prim0=_t([0], torch.int32),
        prim1=_t([0], torch.int32),
        exterior_angle=_t([1.5 * math.pi]),
        src=_t([[0.0, 0.0, 1.5]]) if src is None else src,
        src_power=_t([1.0]),
        wi=_t([[0.0, 0.0, -1.0]]),
        d0=_t([[0.0, 0.0, -1.0]]),
        count=1,
    )


def _paths_material(gain=None):
    return rt.DfrMaterial(
        eta_r=_t([4.0]),
        sigma=_t([0.0]),
        mu_r=_t([1.0]),
        gain=_t([1.0]) if gain is None else gain,
        valid=_t([True], torch.bool),
    )


def _trace_paths(scene, *, tx=None, states=None, material=None):
    return scene.trace_dfr_paths(
        tx_positions=_t([[0.0, 0.0, 1.5]]) if tx is None else tx,
        rx_positions=_t([[0.0, 0.0, -1.0]]),
        states=_paths_states() if states is None else states,
        material=_paths_material() if material is None else material,
        active=_t([True], torch.bool),
        options=rt.DfrPathOptions(max_paths=4, wavelength=0.125),
    )


def _coherent(scene, *, states=None, material=None):
    return scene.accum_dfr_coherent_direct(
        states=_paths_states() if states is None else states,
        grid=rt.DfrGrid(axis=2, position=-1.0),
        material=_paths_material() if material is None else material,
        options=rt.DfrCoherentOptions(wavelength=0.125),
    )


# Each unsupported (operation, domain) maps to calls that attach a reverse-mode or forward-mode input of that domain.
UNSUPPORTED_CASES = {
    ("trace_refl_epc", "geometry"): (
        lambda m: _mirror_scene(m).trace_refl_epc(_t([[0.0, 0.0, 1.0]]), _t([[0.5, 0.0, 1.0]]), 1, _epc_options()),
    ),
    ("trace_refl_epc", "ray"): (
        lambda m: _mirror_scene().trace_refl_epc(
            _mark(_t([[0.0, 0.0, 1.0]]), m), _t([[0.5, 0.0, 1.0]]), 1, _epc_options()
        ),
    ),
    ("trace_refl_epc", "receiver"): (
        lambda m: _mirror_scene().trace_refl_epc(
            _t([[0.0, 0.0, 1.0]]), _mark(_t([[0.5, 0.0, 1.0]]), m), 1, _epc_options()
        ),
    ),
    ("trace_refl_epc", "plane_geometry"): (
        lambda m: _mirror_scene().trace_refl_epc(
            _t([[0.0, 0.0, 1.0]]), _t([[0.5, 0.0, 1.0]]), 1, _epc_options(_mark(_t([[[0.0, 0.0, 0.0]]]), m))
        ),
    ),
    ("trace_refl_epc_field", "geometry"): (
        lambda m: _mirror_scene(m).trace_refl_epc_field(_t([[0.0, 0.0, 1.0]]), _t([[0.5, 0.0, 1.0]]), 1),
    ),
    ("trace_refl_epc_field", "source"): (
        lambda m: _mirror_scene().trace_refl_epc_field(_mark(_t([[0.0, 0.0, 1.0]]), m), _t([[0.5, 0.0, 1.0]]), 1),
    ),
    ("trace_refl_epc_field", "receiver"): (
        lambda m: _mirror_scene().trace_refl_epc_field(_t([[0.0, 0.0, 1.0]]), _mark(_t([[0.5, 0.0, 1.0]]), m), 1),
    ),
    ("accum_refl", "geometry"): (lambda m: _accumulate(_mirror_scene(m)),),
    ("accum_refl", "source_receiver"): (
        lambda m: _accumulate(_mirror_scene(), tx=_mark(_t([[0.0, 0.0, 1.0]]), m)),
        lambda m: _accumulate(_mirror_scene(), origin=_mark(_t([[0.0, 0.0, 1.0]]), m)),
    ),
    ("accum_refl", "material"): (lambda m: _accumulate(_mirror_scene(), eta_r=_mark(_t([4.0]), m)),),
    ("trace_dfr_paths", "continuous_inputs"): (
        lambda m: _trace_paths(_mirror_scene(m)),
        lambda m: _trace_paths(_mirror_scene(), tx=_mark(_t([[0.0, 0.0, 1.5]]), m)),
        lambda m: _trace_paths(_mirror_scene(), states=_paths_states(src=_mark(_t([[0.0, 0.0, 1.5]]), m))),
        lambda m: _trace_paths(_mirror_scene(), material=_paths_material(gain=_mark(_t([1.0]), m))),
    ),
    ("accum_dfr_coherent_direct", "continuous_inputs"): (
        lambda m: _coherent(_mirror_scene(m)),
        lambda m: _coherent(_mirror_scene(), states=_paths_states(edge_pos=_mark(_t([[0.0, 0.0, 0.5]]), m))),
        lambda m: _coherent(_mirror_scene(), material=_paths_material(gain=_mark(_t([1.0]), m))),
    ),
}


@unittest.skipUnless(torch_cuda_available(), "CUDA torch and rayd.torch are required")
class TorchMultipathDerivativeManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = rt.api_manifest()

    def test_every_supported_multipath_domain_has_a_finite_difference_check(self):
        self.assertEqual(manifest_domains(self.manifest, MULTIPATH_FAMILIES, "supported"), COVERED_SUPPORTED)

    def test_every_unsupported_multipath_domain_has_a_refusal_case(self):
        self.assertEqual(manifest_domains(self.manifest, MULTIPATH_FAMILIES, "unsupported"), set(UNSUPPORTED_CASES))

    def test_unsupported_domains_raise_for_reverse_and_forward_inputs(self):
        forward_ad = torch.autograd.forward_ad
        for (operation, domain), calls in UNSUPPORTED_CASES.items():
            for index, call in enumerate(calls):
                for mode in ("vjp", "jvp"):
                    with self.subTest(operation=operation, domain=domain, case=index, mode=mode):
                        with self.assertRaisesRegex(RuntimeError, "forward-only"):
                            if mode == "jvp":
                                with forward_ad.dual_level():
                                    call(mode)
                            else:
                                call(mode)


if __name__ == "__main__":
    unittest.main()
