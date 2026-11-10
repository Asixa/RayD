# Copyright Xingyu Chen.
# Checks Torch geometry-query VJPs and JVPs against central differences along seeded directions.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import unittest

import torch
from torch.autograd import forward_ad

from tests.support import geometry_fd_cases as cases
from tests.support.backends import torch_cuda_available


if torch_cuda_available():
    import rayd.torch as rt
else:  # pragma: no cover - the module is skipped below
    rt = None

DEVICE = "cuda"
REQUIRES_CUDA = unittest.skipUnless(torch_cuda_available(), "CUDA Torch RayD is required")

# Mesh queries use a 5e-3 step: every discrete winner is at least 0.1 (20 steps) from a switch, and the
# rounding term 2 * 2^-24 * |f| / h stays below 1e-4 for objectives of magnitude <= 4.
MESH_STEP = 5.0e-3
# SDF hits are chosen at least 0.01 from every lattice face (cell size 0.125); a 1e-3 step keeps the frozen
# cell fixed while the bisection tolerance (eps_hit 1e-6) stays three orders below the perturbation.
SDF_STEP = 1.0e-3
# A unit direction over all 17^3 lattice values puts only ~1/40 of its weight on the 8 corners a ray reads, so
# the value domain uses a 5e-2 step: each corner still moves by under 1e-3, inside the frozen cell.
SDF_VALUES_STEP = 5.0e-2
SURFEL_STEP = 2.0e-3

Builder = Callable[[Mapping[str, torch.Tensor]], tuple[torch.Tensor, object]]


def _tensor(value) -> torch.Tensor:
    return torch.tensor(value, device=DEVICE, dtype=torch.float32)


def _weighted(output: torch.Tensor, seed: int) -> torch.Tensor:
    """Float64 objective `sum(w * output)` with fixed weights; reduced in float64 so only kernel error remains."""
    weights = torch.tensor(cases.output_weights(output.numel(), seed), device=output.device, dtype=torch.float64)
    return (output.double().reshape(-1) * weights).sum()


def _evaluate(build: Builder) -> Callable[[Mapping], float]:
    def evaluate(values: Mapping) -> float:
        objective, _keepalive = build({name: _tensor(value) for name, value in values.items()})
        return float(objective)

    return evaluate


def _vjp(build: Builder) -> Callable[[Mapping, Sequence[str]], dict[str, list[float]]]:
    def vjp(values: Mapping, domain: Sequence[str]) -> dict[str, list[float]]:
        tensors = {name: _tensor(value).requires_grad_(name in domain) for name, value in values.items()}
        objective, keepalive = build(tensors)
        objective.backward()
        del keepalive
        result = {}
        for name in domain:
            grad = tensors[name].grad
            result[name] = [0.0] * tensors[name].numel() if grad is None else grad.reshape(-1).tolist()
        return result

    return vjp


def _jvp(build: Builder) -> Callable[[Mapping, Mapping], float]:
    def jvp(values: Mapping, direction: Mapping) -> float:
        with forward_ad.dual_level():
            tensors = {
                name: forward_ad.make_dual(_tensor(value), _tensor(direction[name]))
                if name in direction
                else _tensor(value)
                for name, value in values.items()
            }
            objective, _keepalive = build(tensors)
            tangent = forward_ad.unpack_dual(objective).tangent
            return 0.0 if tangent is None else float(tangent)

    return jvp


class _DirectionalCase(unittest.TestCase):
    def check(
        self,
        label: str,
        build: Builder,
        values: Mapping,
        domain: Sequence[str],
        seed: int,
        step: float,
        forward: bool = True,
        **kw,
    ):
        cases.check_directional(
            self,
            label=f"torch/{label}",
            values=values,
            domain=domain,
            seed=seed,
            step=step,
            evaluate=_evaluate(build),
            vjp=_vjp(build),
            jvp=_jvp(build) if forward else None,
            **kw,
        )


def tearDownModule() -> None:
    cases.report_deviations("Torch geometry finite differences")


# (trace_backend, edge_bvh_backend) pairs: the OptiX product path and the custom CUDA BVH path.
ACCELERATION_BACKENDS = (("optix", "optix"), ("cuda", "cuda"))
INTERSECT_VALUES = {
    "vertices": cases.QUAD_VERTICES,
    "origins": cases.INTERSECT_ORIGINS,
    "directions": cases.INTERSECT_DIRECTIONS,
}
EDGE_POINT_VALUES = {"vertices": cases.QUAD_VERTICES, "points": cases.EDGE_POINTS}


def _mesh_scene(vertices: torch.Tensor, backends: tuple[str, str] = ACCELERATION_BACKENDS[0]) -> "rt.Scene":
    faces = torch.tensor(cases.QUAD_FACES, device=DEVICE, dtype=torch.int32)
    scene = rt.Scene(trace_backend=backends[0], edge_bvh_backend=backends[1])
    scene.add_mesh(rt.Mesh(vertices, faces, uv=_tensor(cases.QUAD_UV), face_uv=faces))
    scene.build()
    return scene


@REQUIRES_CUDA
class IntersectFiniteDifferenceTests(_DirectionalCase):
    DOMAINS = {"geometry": ("vertices",), "ray": ("origins", "directions")}
    # The bent quad has no vertex normals, so the shading normal is the face normal and does not move with a ray
    # that stays on one face; its ray derivative is checked to be zero instead of being required non-zero.
    ZERO_DERIVATIVES = {("n", "ray"), ("geo_n", "ray")}

    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        cases.require_domains(self, rt.api_manifest()["derivatives"], "intersect", "intersect", self.DOMAINS)

    def test_fields_match_central_differences(self) -> None:
        for backends in ACCELERATION_BACKENDS:
            for seed, field in enumerate(("t", "p", "n", "geo_n", "uv", "barycentric")):

                def build(tensors, field=field, seed=seed, backends=backends):
                    scene = _mesh_scene(tensors["vertices"], backends)
                    hit = scene.intersect(rt.Ray(tensors["origins"], tensors["directions"]))
                    return _weighted(getattr(hit, field), 100 + seed), scene

                for domain_name, domain in self.DOMAINS.items():
                    with self.subTest(backend=backends[0], field=field, domain=domain_name):
                        self.check(
                            f"intersect[{backends[0]}].{field}/{domain_name}",
                            build,
                            INTERSECT_VALUES,
                            domain,
                            seed=10 * seed + len(domain),
                            step=MESH_STEP,
                            require_nonzero=(field, domain_name) not in self.ZERO_DERIVATIVES,
                        )

    def test_fields_with_an_all_true_active_mask_reverse_mode(self) -> None:
        domain = ("origins", "directions")
        build = self.active_builder()
        self.check("intersect[active].t/ray", build, INTERSECT_VALUES, domain, 90, MESH_STEP, forward=False)

    # Regression: the intersection autograd Function once returned the caller's `active` tensor as-is and saved it
    # with save_for_forward(), which forward-mode AD refuses ("returned as-is ... saved for backward").
    def test_fields_with_an_all_true_active_mask_forward_mode(self) -> None:
        domain = ("origins", "directions")
        self.check("intersect[active].t/ray[jvp]", self.active_builder(), INTERSECT_VALUES, domain, 91, MESH_STEP)

    @staticmethod
    def active_builder() -> Builder:
        def build(tensors):
            scene = _mesh_scene(tensors["vertices"])
            active = torch.ones(tensors["origins"].shape[0], device=DEVICE, dtype=torch.bool)
            hit = scene.intersect(rt.Ray(tensors["origins"], tensors["directions"]), active)
            return _weighted(hit.t, 109), scene

        return build


@REQUIRES_CUDA
class NearestEdgeFiniteDifferenceTests(_DirectionalCase):
    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        manifest = rt.api_manifest()["derivatives"]
        cases.require_domains(self, manifest, "nearest_edge_point", "nearest_edge", {"geometry", "point"})
        cases.require_domains(self, manifest, "nearest_edge_ray", "nearest_edge", {"geometry", "ray"})
        cases.require_domains(self, manifest, "nearest_edges_topk", "nearest_edges", {"geometry", "point"})

    def test_point_query_distance_and_edge_point(self) -> None:
        for backends in ACCELERATION_BACKENDS:
            for seed, field in enumerate(("distance", "edge_point")):

                def build(tensors, field=field, seed=seed, backends=backends):
                    scene = _mesh_scene(tensors["vertices"], backends)
                    return _weighted(getattr(scene.nearest_edge(tensors["points"]), field), 200 + seed), scene

                for domain_name, domain in (("geometry", ("vertices",)), ("point", ("points",))):
                    with self.subTest(backend=backends[1], field=field, domain=domain_name):
                        self.check(
                            f"nearest_edge_point[{backends[1]}].{field}/{domain_name}",
                            build,
                            EDGE_POINT_VALUES,
                            domain,
                            20 + seed,
                            MESH_STEP,
                        )

    def test_ray_query_distance_finite_and_infinite_tmax(self) -> None:
        rays = {
            "finite": (cases.EDGE_RAY_FINITE_ORIGINS, cases.EDGE_RAY_FINITE_DIRECTIONS, cases.EDGE_RAY_FINITE_TMAX),
            "infinite": (cases.EDGE_RAY_INFINITE_ORIGINS, cases.EDGE_RAY_INFINITE_DIRECTIONS, None),
        }
        for backends in ACCELERATION_BACKENDS:
            for seed, (kind, (origins, directions, tmax)) in enumerate(rays.items()):
                values = {"vertices": cases.QUAD_VERTICES, "origins": origins, "directions": directions}

                def build(tensors, tmax=tmax, seed=seed, backends=backends):
                    scene = _mesh_scene(tensors["vertices"], backends)
                    ray = rt.Ray(tensors["origins"], tensors["directions"], None if tmax is None else _tensor(tmax))
                    return _weighted(scene.nearest_edge(ray).distance, 300 + seed), scene

                for domain_name, domain in (("geometry", ("vertices",)), ("ray", ("origins", "directions"))):
                    with self.subTest(backend=backends[1], tmax=kind, domain=domain_name):
                        self.check(
                            f"nearest_edge_ray[{backends[1]}].{kind}/{domain_name}",
                            build,
                            values,
                            domain,
                            30 + seed,
                            MESH_STEP,
                        )

    def test_ray_query_reports_the_clamped_and_unclamped_segment_parameter(self) -> None:
        scene = _mesh_scene(_tensor(cases.QUAD_VERTICES))
        finite = scene.nearest_edge(
            rt.Ray(
                _tensor(cases.EDGE_RAY_FINITE_ORIGINS),
                _tensor(cases.EDGE_RAY_FINITE_DIRECTIONS),
                _tensor(cases.EDGE_RAY_FINITE_TMAX),
            )
        )
        infinite = scene.nearest_edge(
            rt.Ray(_tensor(cases.EDGE_RAY_INFINITE_ORIGINS), _tensor(cases.EDGE_RAY_INFINITE_DIRECTIONS))
        )
        # The finite case must actually exercise the tmax clamp and the infinite one must not touch an edge.
        torch.testing.assert_close(finite.ray_t, _tensor(cases.EDGE_RAY_FINITE_TMAX))
        self.assertTrue(bool((infinite.distance > 0.1).all()))
        self.assertTrue(bool(((infinite.edge_t > 0.05) & (infinite.edge_t < 0.95)).all()))

    def test_topk_winners_are_separated(self) -> None:
        scene = _mesh_scene(_tensor(cases.QUAD_VERTICES))
        distances = scene.nearest_edges(_tensor(cases.EDGE_POINTS), cases.TOPK_K + 1).distances
        gaps = distances[:, 1:] - distances[:, :-1]
        self.assertGreater(float(gaps.min()), 0.04)

    def test_topk_distances(self) -> None:
        for backends in ACCELERATION_BACKENDS:

            def build(tensors, backends=backends):
                scene = _mesh_scene(tensors["vertices"], backends)
                return _weighted(scene.nearest_edges(tensors["points"], cases.TOPK_K).distances, 400), scene

            for domain_name, domain in (("geometry", ("vertices",)), ("point", ("points",))):
                with self.subTest(backend=backends[1], domain=domain_name):
                    self.check(
                        f"nearest_edges[{backends[1]}].distances/{domain_name}",
                        build,
                        EDGE_POINT_VALUES,
                        domain,
                        40,
                        MESH_STEP,
                    )


def _sdf_grid(tensors: Mapping[str, torch.Tensor]) -> "rt.SdfGrid":
    size = cases.SDF_SIZE
    values = tensors["values"].reshape(size, size, size)
    return rt.SdfGrid(values, tensors["position"], tensors["rotation"], tensors["scale"])


@REQUIRES_CUDA
class SdfFiniteDifferenceTests(_DirectionalCase):
    DOMAINS = {
        "sdf_values": ("values",),
        "placement": ("position", "rotation", "scale"),
        "ray": ("origins", "directions"),
    }

    def values(self) -> dict:
        return {
            "values": cases.sphere_sdf_values(cases.SDF_SIZE),
            "position": cases.SDF_POSITION,
            "rotation": cases.SDF_ROTATION,
            "scale": cases.SDF_SCALE,
            "origins": cases.SDF_ORIGINS,
            "directions": cases.SDF_DIRECTIONS,
        }

    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        cases.require_domains(self, rt.api_manifest()["derivatives"], "sdf_intersect", "intersect", self.DOMAINS)

    def test_hits_are_away_from_lattice_faces(self) -> None:
        from tests.sdf import _reference as ref

        tensors = {name: _tensor(value) for name, value in self.values().items()}
        result = rt.sdf_intersect(
            _sdf_grid(tensors),
            tensors["origins"],
            tensors["directions"],
            eps_hit=cases.SDF_EPS_HIT,
            max_steps=cases.SDF_MAX_STEPS,
        )
        self.assertTrue(bool(result.hit_mask.all()))
        local = (result.position - tensors["position"]) @ ref.rotation_matrix(tensors["rotation"])
        lattice = (local / tensors["scale"] + 0.5) * (cases.SDF_SIZE - 1)
        fraction = lattice - lattice.floor()
        margin = float(torch.minimum(fraction, 1.0 - fraction).min())
        self.assertGreater(margin, cases.SDF_MIN_CELL_MARGIN)

    def test_fields_match_central_differences(self) -> None:
        values = self.values()
        for seed, field in enumerate(("t", "position", "normal")):

            def build(tensors, field=field, seed=seed):
                result = rt.sdf_intersect(
                    _sdf_grid(tensors),
                    tensors["origins"],
                    tensors["directions"],
                    eps_hit=cases.SDF_EPS_HIT,
                    max_steps=cases.SDF_MAX_STEPS,
                )
                return _weighted(getattr(result, field), 500 + seed), result

            for domain_name, domain in self.DOMAINS.items():
                with self.subTest(field=field, domain=domain_name):
                    step = SDF_VALUES_STEP if domain_name == "sdf_values" else SDF_STEP
                    self.check(f"sdf.{field}/{domain_name}", build, values, domain, 50 + seed, step, rel=5.0e-3)


def _surfel_scene(tensors: Mapping[str, torch.Tensor]) -> "rt.SurfelScene":
    cloud = rt.SurfelCloud(
        tensors["center"], tensors["tangent_u"], tensors["tangent_v"], tensors["opacity"], tensors["value"]
    )
    scene = rt.SurfelScene(
        cloud, rt.SurfelTraceOptions(max_candidate_hits=cases.SURFEL_MAX_CANDIDATES, transmittance_min=0.0)
    )
    scene.build()
    return scene


@REQUIRES_CUDA
class SurfelFiniteDifferenceTests(_DirectionalCase):
    VALUES = {
        "center": cases.SURFEL_CENTERS,
        "tangent_u": cases.SURFEL_TANGENT_U,
        "tangent_v": cases.SURFEL_TANGENT_V,
        "opacity": cases.SURFEL_OPACITY,
        "value": cases.SURFEL_VALUE,
        "origins": cases.SURFEL_ORIGINS,
        "directions": cases.SURFEL_DIRECTIONS,
    }
    GEOMETRY = ("center", "tangent_u", "tangent_v")
    FIELDS = ("center", "tangent_u", "tangent_v", "opacity", "value")
    RAY = ("origins", "directions")

    def test_both_surfels_are_composited(self) -> None:
        tensors = {name: _tensor(value) for name, value in self.VALUES.items()}
        scene = _surfel_scene(tensors)
        composite = scene.composite_alpha(rt.Ray(tensors["origins"], tensors["directions"]))
        self.assertEqual(composite.candidate_count.tolist(), [2, 2])
        self.assertTrue(bool(((composite.transmittance > 0.2) & (composite.transmittance < 0.9)).all()))

    def test_intersection_distance(self) -> None:
        def build(tensors):
            scene = _surfel_scene(tensors)
            return _weighted(scene.intersect(rt.Ray(tensors["origins"], tensors["directions"])).t, 600), scene

        for domain_name, domain in (("geometry", self.GEOMETRY), ("ray", self.RAY)):
            with self.subTest(domain=domain_name):
                self.check(f"surfel.intersect.t/{domain_name}", build, self.VALUES, domain, 60, SURFEL_STEP)

    def test_composite_transmittance_and_intensity(self) -> None:
        for seed, field in enumerate(("transmittance", "intensity")):

            def build(tensors, field=field, seed=seed):
                scene = _surfel_scene(tensors)
                composite = scene.composite_alpha(rt.Ray(tensors["origins"], tensors["directions"]))
                return _weighted(getattr(composite, field), 610 + seed), scene

            for domain_name, domain in (("surfel_fields", self.FIELDS), ("ray", self.RAY)):
                with self.subTest(field=field, domain=domain_name):
                    self.check(
                        f"surfel.composite.{field}/{domain_name}", build, self.VALUES, domain, 81 + seed, SURFEL_STEP
                    )


@REQUIRES_CUDA
class MixedSceneFiniteDifferenceTests(_DirectionalCase):
    VALUES = {
        "mesh_vertices": cases.MIXED_MESH_VERTICES,
        "sdf_position": cases.MIXED_SDF_POSITION,
        "surfel_center": cases.MIXED_SURFEL_CENTER,
        "surfel_tangent_u": cases.MIXED_SURFEL_TANGENT_U,
        "surfel_tangent_v": cases.MIXED_SURFEL_TANGENT_V,
        "surfel_opacity": cases.MIXED_SURFEL_OPACITY,
        "origins": cases.MIXED_ORIGINS,
        "directions": cases.MIXED_DIRECTIONS,
    }
    INTERSECT_DOMAINS = {
        "selected_geometry": ("mesh_vertices", "sdf_position", "surfel_center"),
        "ray": ("origins", "directions"),
    }
    TRANSMITTANCE_DOMAINS = {
        "surfel_fields": ("surfel_center", "surfel_tangent_u", "surfel_tangent_v", "surfel_opacity"),
        "ray": ("origins", "directions"),
    }

    @staticmethod
    def scene(tensors: Mapping[str, torch.Tensor]) -> "rt.MixedScene":
        size = cases.MIXED_SDF_SIZE
        faces = torch.tensor([[0, 1, 2]], device=DEVICE, dtype=torch.int32)
        scene = rt.MixedScene()
        scene.add_mesh(rt.Mesh(tensors["mesh_vertices"], faces, edges_enabled=False))
        scene.add_sdf(
            rt.SdfGrid(
                _tensor(cases.slab_sdf_values(size)).reshape(size, size, size),
                tensors["sdf_position"],
                _tensor([1.0, 0.0, 0.0, 0.0]),
                _tensor([1.0, 1.0, 1.0]),
            )
        )
        cloud = rt.SurfelCloud(
            tensors["surfel_center"],
            tensors["surfel_tangent_u"],
            tensors["surfel_tangent_v"],
            tensors["surfel_opacity"],
            _tensor([1.0]),
        )
        scene.add_surfel(cloud, rt.SurfelTraceOptions(max_candidate_hits=1, transmittance_min=0.0))
        scene.build()
        return scene

    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        manifest = rt.api_manifest()["derivatives"]
        cases.require_domains(self, manifest, "mixed_scene", "intersect", self.INTERSECT_DOMAINS)
        cases.require_domains(self, manifest, "mixed_scene", "transmittance", self.TRANSMITTANCE_DOMAINS)

    def test_each_ray_selects_a_different_owner(self) -> None:
        tensors = {name: _tensor(value) for name, value in self.VALUES.items()}
        hit = self.scene(tensors).intersect(rt.Ray(tensors["origins"], tensors["directions"]))
        self.assertEqual(hit.shape_id.tolist(), [0, 1, 2])

    def intersect_builder(self, field: str, seed: int) -> Builder:
        def build(tensors):
            scene = self.scene(tensors)
            hit = scene.intersect(rt.Ray(tensors["origins"], tensors["directions"]))
            return _weighted(getattr(hit, field), 700 + seed), scene

        return build

    def transmittance_builder(self) -> Builder:
        def build(tensors):
            scene = self.scene(tensors)
            return _weighted(scene.transmittance(rt.Ray(tensors["origins"], tensors["directions"])), 720), scene

        return build

    # Reverse mode only: forward mode through a mixed scene that contains a mesh is the known defect pinned by
    # the two expected failures below.
    def test_intersect_t_and_p_reverse_mode(self) -> None:
        for seed, field in enumerate(("t", "p")):
            for domain_name, domain in self.INTERSECT_DOMAINS.items():
                with self.subTest(field=field, domain=domain_name):
                    self.check(
                        f"mixed.intersect.{field}/{domain_name}",
                        self.intersect_builder(field, seed),
                        self.VALUES,
                        domain,
                        70 + seed,
                        MESH_STEP,
                        forward=False,
                    )

    def test_transmittance_reverse_mode(self) -> None:
        for domain_name, domain in self.TRANSMITTANCE_DOMAINS.items():
            with self.subTest(domain=domain_name):
                self.check(
                    f"mixed.transmittance/{domain_name}",
                    self.transmittance_builder(),
                    self.VALUES,
                    domain,
                    72,
                    SURFEL_STEP,
                    forward=False,
                )

    # Regression: MixedScene.intersect() always passes a lane `active` mask to the mesh Scene.intersect(), whose
    # forward rule once refused a mask returned as-is, so every JVP through a mixed scene with a mesh raised.
    def test_intersect_forward_mode(self) -> None:
        self.check(
            "mixed.intersect.t/ray[jvp]",
            self.intersect_builder("t", 0),
            self.VALUES,
            self.INTERSECT_DOMAINS["ray"],
            74,
            MESH_STEP,
        )

    # Surfel-field tangents never reach the mesh query.
    def test_transmittance_surfel_fields_forward_mode(self) -> None:
        self.check(
            "mixed.transmittance/surfel_fields[jvp]",
            self.transmittance_builder(),
            self.VALUES,
            self.TRANSMITTANCE_DOMAINS["surfel_fields"],
            75,
            SURFEL_STEP,
        )

    # Regression: MixedScene.transmittance() tests mesh opacity with Scene.intersect(ray, lane_active), so ray
    # tangents reach the same forward rule.
    def test_transmittance_ray_forward_mode(self) -> None:
        self.check(
            "mixed.transmittance/ray[jvp]",
            self.transmittance_builder(),
            self.VALUES,
            self.TRANSMITTANCE_DOMAINS["ray"],
            76,
            SURFEL_STEP,
        )


if __name__ == "__main__":
    unittest.main()
