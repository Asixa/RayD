# Copyright Xingyu Chen.
# Checks Dr.Jit geometry-query VJPs and JVPs against central differences along seeded directions.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import unittest

from tests.support import geometry_fd_cases as cases
from tests.support.backends import drjit_modules


MODULES = drjit_modules()
REQUIRES_DRJIT = unittest.skipUnless(MODULES is not None, "a CUDA device and an importable rayd.drjit are required")
if MODULES is not None:
    pj, dr, cuda, ad = MODULES.pj, MODULES.dr, MODULES.cuda, MODULES.ad

# Step sizes and their reasoning match the Torch module: every discrete winner is at least 20 steps from a
# switch, and the float32 rounding term of the central difference stays below 1e-4.
MESH_STEP = 5.0e-3
SDF_STEP = 1.0e-3
SDF_VALUES_STEP = 5.0e-2
SURFEL_STEP = 2.0e-3

Builder = Callable[[Mapping[str, object]], tuple[object, object]]


def _array(value, tangent: bool = False):
    """Convert nested lists to a Dr.Jit AD array: rows of 3 or 2 become Array3f/Array2f, flat lists Float."""
    if value and isinstance(value[0], (list, tuple)):
        columns = [[float(row[c]) for row in value] for c in range(len(value[0]))]
        return (ad.Array3f if len(columns) == 3 else ad.Array2f)(*columns)
    return ad.Float([float(item) for item in value])


def _flat(array) -> list[float]:
    """Row-major flattening matching `geometry_fd_cases.flatten` of the nested input."""
    if isinstance(array, (ad.Array3f, ad.Array2f)):
        columns = [list(array[c]) for c in range(len(array))]
        return [float(columns[c][i]) for i in range(len(columns[0])) for c in range(len(columns))]
    return [float(x) for x in array]


def _weighted(output, seed: int):
    """`sum(w * output)` over every element of a Float/Array2f/Array3f output in row-major order."""
    if isinstance(output, (ad.Array3f, ad.Array2f)):
        width = len(output)
        count = dr.width(output[0])
        weights = cases.output_weights(width * count, seed)
        return sum(dr.sum(output[c] * ad.Float(weights[c::width])) for c in range(width))
    weights = cases.output_weights(dr.width(output), seed)
    return dr.sum(output * ad.Float(weights))


def _evaluate(build: Builder) -> Callable[[Mapping], float]:
    def evaluate(values: Mapping) -> float:
        objective, _keepalive = build({name: _array(value) for name, value in values.items()})
        return float(objective[0])

    return evaluate


def _vjp(build: Builder) -> Callable[[Mapping, Sequence[str]], dict[str, list[float]]]:
    def vjp(values: Mapping, domain: Sequence[str]) -> dict[str, list[float]]:
        arrays = {name: _array(value) for name, value in values.items()}
        for name in domain:
            dr.enable_grad(arrays[name])
        objective, keepalive = build(arrays)
        dr.backward(objective, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        del keepalive
        return {name: _flat(dr.grad(arrays[name])) for name in domain}

    return vjp


def _jvp(build: Builder) -> Callable[[Mapping, Mapping], float]:
    def jvp(values: Mapping, direction: Mapping) -> float:
        arrays = {name: _array(value) for name, value in values.items()}
        for name, tangent in direction.items():
            dr.enable_grad(arrays[name])
            dr.set_grad(arrays[name], _array(tangent))
        objective, _keepalive = build(arrays)
        dr.forward_to(objective, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        return float(dr.grad(objective)[0])

    return jvp


class _DirectionalCase(unittest.TestCase):
    def check(self, label: str, build: Builder, values: Mapping, domain: Sequence[str], seed: int, step: float, **kw):
        cases.check_directional(
            self,
            label=f"drjit/{label}",
            values=values,
            domain=domain,
            seed=seed,
            step=step,
            evaluate=_evaluate(build),
            vjp=_vjp(build),
            jvp=_jvp(build),
            **kw,
        )


def tearDownModule() -> None:
    cases.report_deviations("Dr.Jit geometry finite differences")


def _int3(rows):
    return cuda.Array3i(*[[int(row[c]) for row in rows] for c in range(3)])


# (trace_backend, edge_bvh_backend) pairs: the OptiX product path and the CUDA trace with the Dr.Jit edge BVH.
ACCELERATION_BACKENDS = (("optix", "optix"), ("cuda", "drjit"))
INTERSECT_VALUES = {
    "vertices": cases.QUAD_VERTICES,
    "origins": cases.INTERSECT_ORIGINS,
    "directions": cases.INTERSECT_DIRECTIONS,
}
EDGE_POINT_VALUES = {"vertices": cases.QUAD_VERTICES, "points": cases.EDGE_POINTS}


def _mesh_scene(vertices, backends: tuple[str, str] = ACCELERATION_BACKENDS[0]) -> "pj.Scene":
    plain = cuda.Array3f(*[[0.0] * len(cases.QUAD_VERTICES)] * 3)
    uv = cuda.Array2f([uv[0] for uv in cases.QUAD_UV], [uv[1] for uv in cases.QUAD_UV])
    mesh = pj.Mesh(plain, _int3(cases.QUAD_FACES), uv, _int3(cases.QUAD_FACES))
    mesh.vertex_positions = vertices
    scene = pj.Scene(trace_backend=backends[0], edge_bvh_backend=backends[1])
    scene.add_mesh(mesh)
    scene.build()
    return scene


def _ray(origins, directions, tmax=None) -> "pj.RayAD":
    ray = pj.RayAD(origins, directions)
    if tmax is not None:
        ray.tmax = ad.Float([float(value) for value in tmax])
    return ray


@REQUIRES_DRJIT
class IntersectFiniteDifferenceTests(_DirectionalCase):
    DOMAINS = {"geometry": ("vertices",), "ray": ("origins", "directions")}
    # The geometric normal of a face does not move with a ray that stays on it. Unlike Torch, the Dr.Jit shading
    # normal interpolates vertex normals, so it has a non-zero ray derivative.
    ZERO_DERIVATIVES = {("geo_n", "ray")}

    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        cases.require_domains(self, pj.api_manifest()["derivatives"], "intersect", "intersect", self.DOMAINS)

    def test_fields_match_central_differences(self) -> None:
        for backends in ACCELERATION_BACKENDS:
            for seed, field in enumerate(("t", "p", "n", "geo_n", "uv", "barycentric")):

                def build(arrays, field=field, seed=seed, backends=backends):
                    scene = _mesh_scene(arrays["vertices"], backends)
                    hit = scene.intersect(_ray(arrays["origins"], arrays["directions"]))
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

    def test_fields_with_an_all_true_active_mask(self) -> None:
        def build(arrays):
            scene = _mesh_scene(arrays["vertices"])
            active = ad.Bool([True] * len(cases.INTERSECT_ORIGINS))
            return _weighted(scene.intersect(_ray(arrays["origins"], arrays["directions"]), active).t, 109), scene

        self.check("intersect[active].t/ray", build, INTERSECT_VALUES, ("origins", "directions"), 91, MESH_STEP)


@REQUIRES_DRJIT
class NearestEdgeFiniteDifferenceTests(_DirectionalCase):
    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        manifest = pj.api_manifest()["derivatives"]
        cases.require_domains(self, manifest, "nearest_edge_point", "nearest_edge", {"geometry", "point"})
        cases.require_domains(self, manifest, "nearest_edge_ray", "nearest_edge", {"geometry", "ray"})
        cases.require_domains(self, manifest, "nearest_edges_topk", "nearest_edges", {"geometry", "point"})

    def test_point_query_distance_and_edge_point(self) -> None:
        for backends in ACCELERATION_BACKENDS:
            for seed, field in enumerate(("distance", "edge_point")):

                def build(arrays, field=field, seed=seed, backends=backends):
                    scene = _mesh_scene(arrays["vertices"], backends)
                    return _weighted(getattr(scene.nearest_edge(arrays["points"]), field), 200 + seed), scene

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

                def build(arrays, tmax=tmax, seed=seed, backends=backends):
                    scene = _mesh_scene(arrays["vertices"], backends)
                    ray = _ray(arrays["origins"], arrays["directions"], tmax)
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
        scene = _mesh_scene(_array(cases.QUAD_VERTICES))
        finite = scene.nearest_edge(
            _ray(
                _array(cases.EDGE_RAY_FINITE_ORIGINS),
                _array(cases.EDGE_RAY_FINITE_DIRECTIONS),
                cases.EDGE_RAY_FINITE_TMAX,
            )
        )
        infinite = scene.nearest_edge(
            _ray(_array(cases.EDGE_RAY_INFINITE_ORIGINS), _array(cases.EDGE_RAY_INFINITE_DIRECTIONS))
        )
        self.assertAlmostEqual(float(finite.ray_t[0]), cases.EDGE_RAY_FINITE_TMAX[0], places=6)
        for i in range(len(cases.EDGE_RAY_INFINITE_ORIGINS)):
            self.assertGreater(float(infinite.distance[i]), 0.1)
            self.assertTrue(0.05 < float(infinite.edge_t[i]) < 0.95)

    def test_topk_distances(self) -> None:
        for backends in ACCELERATION_BACKENDS:

            def build(arrays, backends=backends):
                scene = _mesh_scene(arrays["vertices"], backends)
                return _weighted(scene.nearest_edges(arrays["points"], cases.TOPK_K).distances, 400), scene

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


def _sdf_options() -> "pj.SdfTraceOptions":
    options = pj.SdfTraceOptions()
    options.eps_hit = cases.SDF_EPS_HIT
    options.max_steps = cases.SDF_MAX_STEPS
    return options


@REQUIRES_DRJIT
class SdfFiniteDifferenceTests(_DirectionalCase):
    DOMAINS = {
        "sdf_values": ("values",),
        "placement": ("position", "rotation", "scale"),
        "ray": ("origins", "directions"),
    }
    VALUES = {
        "values": cases.sphere_sdf_values(cases.SDF_SIZE),
        "position": [cases.SDF_POSITION],
        "rotation": cases.SDF_ROTATION,
        "scale": [cases.SDF_SCALE],
        "origins": cases.SDF_ORIGINS,
        "directions": cases.SDF_DIRECTIONS,
    }

    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        cases.require_domains(self, pj.api_manifest()["derivatives"], "sdf_intersect", "intersect", self.DOMAINS)

    def test_fields_match_central_differences(self) -> None:
        size = cases.SDF_SIZE
        for seed, field in enumerate(("t", "position", "normal")):

            def build(arrays, field=field, seed=seed):
                grid = pj.SdfGrid(
                    arrays["values"], size, size, size, arrays["position"], arrays["rotation"], arrays["scale"]
                )
                result = grid.intersect(_ray(arrays["origins"], arrays["directions"]), _sdf_options())
                return _weighted(getattr(result, field), 500 + seed), (grid, result)

            for domain_name, domain in self.DOMAINS.items():
                with self.subTest(field=field, domain=domain_name):
                    step = SDF_VALUES_STEP if domain_name == "sdf_values" else SDF_STEP
                    self.check(f"sdf.{field}/{domain_name}", build, self.VALUES, domain, 50 + seed, step, rel=5.0e-3)


def _surfel_scene(arrays: Mapping, collect_candidate_stats: bool = False) -> "pj.SurfelScene":
    cloud = pj.SurfelCloud(
        arrays["center"], arrays["tangent_u"], arrays["tangent_v"], arrays["opacity"], arrays["value"]
    )
    options = pj.SurfelTraceOptions()
    options.max_candidate_hits = cases.SURFEL_MAX_CANDIDATES
    options.transmittance_min = 0.0
    options.collect_candidate_stats = collect_candidate_stats
    scene = pj.SurfelScene(cloud, options)
    scene.build()
    return scene


@REQUIRES_DRJIT
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
        arrays = {name: _array(value) for name, value in self.VALUES.items()}
        scene = _surfel_scene(arrays, collect_candidate_stats=True)
        composite = scene.composite_alpha(_ray(arrays["origins"], arrays["directions"]))
        self.assertEqual([int(x) for x in composite.candidate_count], [2, 2])
        for value in composite.transmittance:
            self.assertTrue(0.2 < float(value) < 0.9)

    def test_intersection_distance(self) -> None:
        def build(arrays):
            scene = _surfel_scene(arrays)
            return _weighted(scene.intersect(_ray(arrays["origins"], arrays["directions"])).t, 600), scene

        for domain_name, domain in (("geometry", self.GEOMETRY), ("ray", self.RAY)):
            with self.subTest(domain=domain_name):
                self.check(f"surfel.intersect.t/{domain_name}", build, self.VALUES, domain, 60, SURFEL_STEP)

    def test_composite_transmittance_and_intensity(self) -> None:
        for seed, field in enumerate(("transmittance", "intensity")):

            def build(arrays, field=field, seed=seed):
                scene = _surfel_scene(arrays)
                composite = scene.composite_alpha(_ray(arrays["origins"], arrays["directions"]))
                return _weighted(getattr(composite, field), 610 + seed), scene

            for domain_name, domain in (("surfel_fields", self.FIELDS), ("ray", self.RAY)):
                with self.subTest(field=field, domain=domain_name):
                    self.check(
                        f"surfel.composite.{field}/{domain_name}", build, self.VALUES, domain, 81 + seed, SURFEL_STEP
                    )


@REQUIRES_DRJIT
class MixedSceneFiniteDifferenceTests(_DirectionalCase):
    VALUES = {
        "mesh_vertices": cases.MIXED_MESH_VERTICES,
        "sdf_position": [cases.MIXED_SDF_POSITION],
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
    def scene(arrays: Mapping) -> "pj.MixedScene":
        size = cases.MIXED_SDF_SIZE
        mesh = pj.Mesh(cuda.Array3f(*[[0.0] * 3] * 3), _int3([[0, 1, 2]]))
        mesh.vertex_positions = arrays["mesh_vertices"]
        mesh.edges_enabled = False
        sdf = pj.SdfGrid(
            ad.Float(cases.slab_sdf_values(size)),
            size,
            size,
            size,
            arrays["sdf_position"],
            ad.Float([1.0, 0.0, 0.0, 0.0]),
            ad.Array3f(1.0),
        )
        cloud = pj.SurfelCloud(
            arrays["surfel_center"],
            arrays["surfel_tangent_u"],
            arrays["surfel_tangent_v"],
            arrays["surfel_opacity"],
            ad.Float([1.0]),
        )
        options = pj.SurfelTraceOptions()
        options.max_candidate_hits = 1
        options.transmittance_min = 0.0
        scene = pj.MixedScene()
        scene.add_mesh(mesh)
        scene.add_sdf(sdf)
        scene.add_surfel(cloud, options)
        scene.build()
        return scene

    def test_manifest_lists_exactly_the_tested_domains(self) -> None:
        manifest = pj.api_manifest()["derivatives"]
        cases.require_domains(self, manifest, "mixed_scene", "intersect", self.INTERSECT_DOMAINS)
        cases.require_domains(self, manifest, "mixed_scene", "transmittance", self.TRANSMITTANCE_DOMAINS)

    def test_each_ray_selects_a_different_owner(self) -> None:
        arrays = {name: _array(value) for name, value in self.VALUES.items()}
        hit = self.scene(arrays).intersect(_ray(arrays["origins"], arrays["directions"]))
        self.assertEqual([int(x) for x in hit.shape_id], [0, 1, 2])

    def test_intersect_t_and_p(self) -> None:
        for seed, field in enumerate(("t", "p")):

            def build(arrays, field=field, seed=seed):
                scene = self.scene(arrays)
                hit = scene.intersect(_ray(arrays["origins"], arrays["directions"]))
                return _weighted(getattr(hit, field), 700 + seed), scene

            for domain_name, domain in self.INTERSECT_DOMAINS.items():
                with self.subTest(field=field, domain=domain_name):
                    self.check(
                        f"mixed.intersect.{field}/{domain_name}", build, self.VALUES, domain, 70 + seed, MESH_STEP
                    )

    def test_transmittance(self) -> None:
        def build(arrays):
            scene = self.scene(arrays)
            return _weighted(scene.transmittance(_ray(arrays["origins"], arrays["directions"])), 720), scene

        for domain_name, domain in self.TRANSMITTANCE_DOMAINS.items():
            with self.subTest(domain=domain_name):
                self.check(f"mixed.transmittance/{domain_name}", build, self.VALUES, domain, 72, SURFEL_STEP)


if __name__ == "__main__":
    unittest.main()
