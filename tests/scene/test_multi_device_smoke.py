# Copyright Xingyu Chen.
# Tests multi device smoke.

import unittest

import torch

from tests.support.geometry import grid_mesh as _grid_mesh, tensor_bits as _bits
import rayd.torch as rt


# Query inputs are built once on the host and moved to each device, so the two
# devices receive bit-identical inputs and any difference is device handling.
_RAY_O = torch.tensor(
    [[-0.60, -0.60, -1.0], [0.10, -0.30, -1.0], [0.40, 0.50, -1.0], [-0.20, 0.70, -1.0]], dtype=torch.float32
)
_RAY_D = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32)
_POINTS = torch.tensor(
    [[0.00, 0.00, 0.25], [0.90, -0.90, 0.10], [-0.50, 0.50, -0.30], [0.30, 0.20, 0.50]], dtype=torch.float32
)
_SEGMENT_START = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [4.0, 4.0, -1.0]], dtype=torch.float32)
_SEGMENT_END = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.5, 2.0], [4.0, 4.0, 1.0]], dtype=torch.float32)
_SOURCE = torch.tensor([[0.0, 0.0, -1.0], [0.2, 0.1, -1.0], [-0.2, 0.2, -1.0]], dtype=torch.float32)
_RECEIVER = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.1, 1.0], [-0.2, 0.2, 1.0]], dtype=torch.float32)
# The second pair endpoint clears the mesh, so `visible_pair` reports one
# blocked and one unblocked segment per query instead of a constant answer.
_PAIR_END_B = torch.tensor([[4.0, 4.0, 1.0], [4.0, -4.0, 1.0], [-4.0, 4.0, 1.0]], dtype=torch.float32)
# The first axial edge sits below the mesh (visible from its source), the
# second above it (occluded), so neither sample answer is degenerate.
_EDGE_SOURCE = torch.tensor([[0.0, 0.0, -1.0], [0.5, 0.5, -1.0]], dtype=torch.float32)
_EDGE_POSITION = torch.tensor([[0.0, 0.0, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
_EDGE_DIRECTION = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
_EDGE_T_MIN = torch.tensor([-0.5, -0.5], dtype=torch.float32)
_EDGE_T_MAX = torch.tensor([0.5, 0.5], dtype=torch.float32)
# Chain 0 stays below the mesh; chain 1 crosses it on its first segment.
_CHAIN_POINTS = torch.tensor(
    [[[0.0, 0.0, -1.0], [0.0, 0.0, -0.5], [0.2, 0.2, -0.4]], [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.5]]],
    dtype=torch.float32,
)
_CHAIN_LENGTH = torch.tensor([2, 2], dtype=torch.int32)
# Camera is Torch-only public API and takes no scene, so it is covered here as
# its own family rather than through a scene query.
_CAMERA = rt.Camera(width=16, height=12, fov_x=45.0)
_CAMERA_SAMPLE = torch.tensor([[0.0, 0.0], [0.25, 0.75], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
_CAMERA_POINT = torch.tensor(
    [[0.1, -0.2, 2.0], [0.3, 0.4, 4.0], [-0.5, 0.25, 1.0], [0.0, 0.0, 3.0]], dtype=torch.float32
)


def _sphere_field(resolution: int = 5, radius: float = 0.5) -> torch.Tensor:
    """Vertex-centred signed distance to a sphere over the grid's local box."""
    axis = torch.linspace(-1.0, 1.0, resolution, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    field = torch.stack((gx, gy, gz), dim=-1).norm(dim=-1) - radius
    return field.contiguous()


# The SDF primitive owns no acceleration structure and never joins a `Scene`,
# so its device comes from the caller's grid tensors alone (ADR-0037).
_SDF_VALUES = _sphere_field()
_SDF_POSITION = torch.zeros(3, dtype=torch.float32)
_SDF_ROTATION = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
_SDF_SCALE = torch.full((3,), 2.0, dtype=torch.float32)
_SDF_ORIGINS = torch.tensor(
    [[0.00, 0.00, -2.0], [0.20, 0.10, -2.0], [-0.15, 0.25, -2.0], [0.80, 0.80, -2.0]], dtype=torch.float32
)
_SDF_DIRECTIONS = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32)


def _build_scene(index: int) -> rt.Scene:
    """Scene.build() requires its own device to be current; nothing else does."""
    vertices, faces = _grid_mesh(torch.device("cuda", index), cells=3)
    with torch.cuda.device(index):
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
    return scene


def _dfr_states(device: torch.device) -> rt.DfrStates:
    """Caller-owned order-1 diffraction states; RayD never derives them itself.

    The literals materialize on the host and are copied to `device`, exactly
    like the module-level query inputs, so both devices receive equal bits.
    """

    def f32(values):
        return torch.tensor(values, dtype=torch.float32).to(device)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32).to(device)

    return rt.DfrStates(
        edge_index=i32([0, 1]),
        edge_pos=f32([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        edge_dir=f32([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_t_min=f32([-1.0, -1.0]),
        edge_t_max=f32([1.0, 1.0]),
        n0=f32([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        n1=f32([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        prim0=i32([0, 0]),
        prim1=i32([0, 0]),
        exterior_angle=f32([torch.pi, torch.pi]),
        src=f32([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25]]),
        src_power=f32([1.0, 1.0]),
    )


def _run_scene_query_ops(scene: rt.Scene, index: int) -> dict[str, torch.Tensor]:
    """Scene-coupled `per_ray` queries: shard-invariant, so bitwise comparable."""
    device = torch.device("cuda", index)
    ray = rt.Ray(_RAY_O.to(device), _RAY_D.to(device))
    point = _POINTS.to(device)

    intersection = scene.intersect(ray)
    nearest_point = scene.nearest_edge(point)
    nearest_ray = scene.nearest_edge(ray)
    topk = scene.nearest_edges(point, 4)
    visible = scene.visible(_SEGMENT_START.to(device), _SEGMENT_END.to(device))
    pair = scene.visible_pair(_SOURCE.to(device), _RECEIVER.to(device), _PAIR_END_B.to(device))
    axial = scene.visible_edge(
        _EDGE_SOURCE.to(device),
        _EDGE_POSITION.to(device),
        _EDGE_DIRECTION.to(device),
        _EDGE_T_MIN.to(device),
        _EDGE_T_MAX.to(device),
    )
    chain = scene.visible_chain(_CHAIN_POINTS.to(device), _CHAIN_LENGTH.to(device))
    geometry = scene.global_geometry()

    return {
        "intersect.t": intersection.t,
        "intersect.p": intersection.p,
        "intersect.n": intersection.n,
        "intersect.uv": intersection.uv,
        "intersect.barycentric": intersection.barycentric,
        "intersect.prim_id": intersection.prim_id,
        "intersect.global_prim_id": intersection.global_prim_id,
        "nearest_edge_point.distance": nearest_point.distance,
        "nearest_edge_point.edge_point": nearest_point.edge_point,
        "nearest_edge_point.edge_t": nearest_point.edge_t,
        "nearest_edge_point.global_edge_id": nearest_point.global_edge_id,
        "nearest_edge_ray.distance": nearest_ray.distance,
        "nearest_edge_ray.ray_t": nearest_ray.ray_t,
        "nearest_edge_ray.edge_point": nearest_ray.edge_point,
        "nearest_edge_ray.global_edge_id": nearest_ray.global_edge_id,
        "nearest_edges.is_valid": topk.is_valid,
        "nearest_edges.distances": topk.distances,
        "nearest_edges.edge_points": topk.edge_points,
        "nearest_edges.global_edge_ids": topk.global_edge_ids,
        "visible": visible,
        "visible_pair.visible_a": pair.visible_a,
        "visible_pair.visible_b": pair.visible_b,
        "visible_edge.any_visible": axial.any_visible,
        "visible_chain.all_visible": chain.all_visible,
        "visible_chain.first_blocked_segment": chain.first_blocked_segment,
        "visible_chain.first_blocked_prim": chain.first_blocked_prim,
        "global_geometry.vertices": geometry.vertices,
        "global_geometry.faces": geometry.faces,
        "global_geometry.face_normal": geometry.face_normal,
        "global_geometry.shape_id": geometry.shape_id,
        "global_geometry.global_prim_id": geometry.global_prim_id,
    }


def _run_reflection_ops(scene: rt.Scene, index: int) -> dict[str, torch.Tensor]:
    """Reflection `per_ray` ops, kept apart so a fault here hides no other family."""
    device = torch.device("cuda", index)
    ray = rt.Ray(_RAY_O.to(device), _RAY_D.to(device))

    chain = scene.trace_reflections(ray, max_bounces=1)
    epc = scene.trace_refl_epc_field(_SOURCE.to(device), _RECEIVER.to(device), max_bounces=1)

    return {
        "trace_reflections.valid": chain.valid,
        "trace_reflections.t": chain.t,
        "trace_reflections.prim_ids": chain.prim_ids,
        "trace_refl_epc_field.field_real": epc.field_real,
        "trace_refl_epc_field.field_imag": epc.field_imag,
        "trace_refl_epc_field.path_length": epc.path_length,
        "trace_refl_epc_field.valid": epc.valid,
        "trace_refl_epc_field.resolved_prim_ids": epc.resolved_prim_ids,
    }


def _run_scene_free_ops(index: int) -> dict[str, torch.Tensor]:
    """Public ops that take no scene, so their device comes from their inputs."""
    device = torch.device("cuda", index)
    sample = _CAMERA_SAMPLE.to(device)

    world = _CAMERA.sample_to_world(sample)
    back = _CAMERA.world_to_sample(_CAMERA_POINT.to(device))
    camera_ray = _CAMERA.sample_ray(sample)
    grid = rt.SdfGrid(
        values=_SDF_VALUES.to(device),
        position=_SDF_POSITION.to(device),
        rotation=_SDF_ROTATION.to(device),
        scale=_SDF_SCALE.to(device),
    )
    sdf = rt.sdf_intersect(grid, _SDF_ORIGINS.to(device), _SDF_DIRECTIONS.to(device))

    return {
        "camera.sample_to_world": world,
        "camera.world_to_sample": back,
        "camera.sample_ray.o": camera_ray.o,
        "camera.sample_ray.d": camera_ray.d,
        "sdf_intersect.t": sdf.t,
        "sdf_intersect.hit_mask": sdf.hit_mask,
        "sdf_intersect.position": sdf.position,
        "sdf_intersect.normal": sdf.normal,
        "sdf_intersect.steps": sdf.steps,
    }


def _run_bitwise_ops(scene: rt.Scene, index: int) -> dict[str, torch.Tensor]:
    return {**_run_scene_query_ops(scene, index), **_run_reflection_ops(scene, index), **_run_scene_free_ops(index)}


def _run_diffraction_ops(scene: rt.Scene, index: int) -> dict[str, torch.Tensor]:
    """Diffraction exporter and accumulation families.

    These are the plan's `grid_reduce` and `batch_coupled` classes: they
    accumulate through atomics and place exporter rows through a device
    counter, so only placement and shape are a contract here, never bits.
    """
    device = torch.device("cuda", index)
    states = _dfr_states(device)
    active = torch.ones(states.state_count, dtype=torch.bool).to(device)
    grid = rt.DfrGrid(axis=2, position=0.5, resolution0=2, resolution1=2)

    paths = scene.trace_dfr_paths(
        tx_positions=_SOURCE.to(device),
        rx_positions=_RECEIVER.to(device),
        states=states,
        active=active,
        max_paths=states.state_count,
        wavelength=1.0,
    )
    direct = scene.accum_dfr_direct(states=states, grid=grid, wavelength=1.0, direct_samples=4, seed=7)
    recursive = scene.accum_dfr(
        initial_states=states,
        recursive_states=states,
        grid=grid,
        wavelength=1.0,
        direct_samples=4,
        keller_samples=4,
        seed=7,
        max_order=2,
    )
    coherent = scene.accum_dfr_coherent_direct(states=states, grid=grid, wavelength=1.0)

    return {
        "trace_dfr_paths.count": paths.count,
        "trace_dfr_paths.valid": paths.valid,
        "trace_dfr_paths.tx_id": paths.tx_id,
        "trace_dfr_paths.rx_id": paths.rx_id,
        "trace_dfr_paths.delay": paths.delay,
        "accum_dfr_direct.power": direct.power,
        "accum_dfr_direct.field_x_re": direct.field_x_re,
        "accum_dfr_direct.direct_count": direct.direct_count,
        "accum_dfr.power": recursive.power,
        "accum_dfr.field_x_re": recursive.field_x_re,
        "accum_dfr.keller_count": recursive.keller_count,
        "accum_dfr_coherent_direct.direct_field_x_re": coherent.direct_field_x_re,
        "accum_dfr_coherent_direct.direct_field_x_im": coherent.direct_field_x_im,
        "accum_dfr_coherent_direct.direct_count": coherent.direct_count,
    }


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices are required")
class MultiDeviceSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()
        torch.cuda.set_device(0)

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def assert_same_results(self, left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], context: str) -> None:
        self.assertEqual(sorted(left), sorted(right))
        for name, value in left.items():
            other = right[name]
            self.assertEqual(value.dtype, other.dtype, f"{context}: {name} dtype")
            self.assertEqual(value.shape, other.shape, f"{context}: {name} shape")
            self.assertTrue(torch.equal(_bits(value), _bits(other)), f"{context}: {name} is not bitwise equal")

    def assert_same_shapes(self, left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], context: str) -> None:
        """Placement contract for the families whose float order is not fixed."""
        self.assertEqual(sorted(left), sorted(right))
        for name, value in left.items():
            other = right[name]
            self.assertEqual(value.dtype, other.dtype, f"{context}: {name} dtype")
            self.assertEqual(value.shape, other.shape, f"{context}: {name} shape")

    def assert_on_device(self, results: dict[str, torch.Tensor], index: int, context: str) -> None:
        for name, value in results.items():
            self.assertEqual(value.device.index, index, f"{context}: {name} left the scene device")

    def run_with_ambient_device_zero(self, run):
        """Run `run` with device 0 current and prove no op leaked its guard."""
        torch.cuda.set_device(0)
        self.assertEqual(torch.cuda.current_device(), 0)
        results = run()
        self.assertEqual(torch.cuda.current_device(), 0, "an op leaked its device guard")
        return results

    def test_same_mesh_on_two_devices_intersects_bitwise_equal(self):
        if torch.cuda.get_device_name(0) != torch.cuda.get_device_name(1):
            self.skipTest("bitwise cross-device equality needs identical devices")
        scene0 = _build_scene(0)
        scene1 = _build_scene(1)

        first = scene0.intersect(rt.Ray(_RAY_O.to("cuda:0"), _RAY_D.to("cuda:0")))
        with torch.cuda.device(1):
            second = scene1.intersect(rt.Ray(_RAY_O.to("cuda:1"), _RAY_D.to("cuda:1")))

        self.assertEqual(first.t.device.index, 0)
        self.assertEqual(second.t.device.index, 1)
        for name in ("t", "p", "n", "geo_n", "uv", "barycentric", "prim_id", "shape_id"):
            left = getattr(first, name)
            right = getattr(second, name)
            self.assertEqual(left.dtype, right.dtype, f"intersect.{name} dtype")
            self.assertTrue(
                torch.equal(_bits(left), _bits(right)), f"intersect.{name} differs between cuda:0 and cuda:1"
            )

    def test_scene_query_ops_are_independent_of_the_ambient_device(self):
        scene = _build_scene(1)

        with torch.cuda.device(1):
            reference = _run_scene_query_ops(scene, 1)
        ambient_zero = self.run_with_ambient_device_zero(lambda: _run_scene_query_ops(scene, 1))

        self.assert_on_device(ambient_zero, 1, "ambient device 0")
        self.assert_same_results(reference, ambient_zero, "ambient device 0")

    def test_reflection_ops_are_independent_of_the_ambient_device(self):
        scene = _build_scene(1)

        with torch.cuda.device(1):
            reference = _run_reflection_ops(scene, 1)
        ambient_zero = self.run_with_ambient_device_zero(lambda: _run_reflection_ops(scene, 1))

        self.assert_on_device(ambient_zero, 1, "ambient device 0")
        self.assert_same_results(reference, ambient_zero, "ambient device 0")

    def test_scene_free_ops_are_independent_of_the_ambient_device(self):
        with torch.cuda.device(1):
            reference = _run_scene_free_ops(1)
        ambient_zero = self.run_with_ambient_device_zero(lambda: _run_scene_free_ops(1))

        self.assert_on_device(ambient_zero, 1, "ambient device 0")
        self.assert_same_results(reference, ambient_zero, "ambient device 0")

    def test_diffraction_ops_are_independent_of_the_ambient_device(self):
        scene = _build_scene(1)

        with torch.cuda.device(1):
            reference = _run_diffraction_ops(scene, 1)
        ambient_zero = self.run_with_ambient_device_zero(lambda: _run_diffraction_ops(scene, 1))

        self.assert_on_device(ambient_zero, 1, "ambient device 0")
        self.assert_same_shapes(reference, ambient_zero, "ambient device 0")

    def test_second_device_results_match_the_first_device(self):
        if torch.cuda.get_device_name(0) != torch.cuda.get_device_name(1):
            self.skipTest("bitwise cross-device equality needs identical devices")
        scene0 = _build_scene(0)
        scene1 = _build_scene(1)

        first = _run_bitwise_ops(scene0, 0)
        with torch.cuda.device(1):
            second = _run_bitwise_ops(scene1, 1)

        self.assert_same_results(first, second, "cuda:0 versus cuda:1")

    def test_second_device_diffraction_ops_match_the_first_device_in_shape(self):
        scene0 = _build_scene(0)
        scene1 = _build_scene(1)

        first = _run_diffraction_ops(scene0, 0)
        with torch.cuda.device(1):
            second = _run_diffraction_ops(scene1, 1)

        self.assert_on_device(first, 0, "cuda:0")
        self.assert_on_device(second, 1, "cuda:1")
        self.assert_same_shapes(first, second, "cuda:0 versus cuda:1")

    def test_cross_device_inputs_are_rejected(self):
        scene = _build_scene(0)
        foreign_point = _POINTS.to("cuda:1")

        with self.assertRaises(RuntimeError):
            scene.nearest_edges(foreign_point, 4)
        with self.assertRaises(RuntimeError):
            scene.set_edge_mask(torch.ones(scene.edge_mask().numel(), device="cuda:1", dtype=torch.bool))

        # The scene stays usable on its own device after a rejected query.
        local = scene.nearest_edges(_POINTS.to("cuda:0"), 4)
        self.assertEqual(local.distances.device.index, 0)

    def test_cross_device_mesh_build_is_rejected(self):
        vertices, faces = _grid_mesh(torch.device("cuda", 0), cells=3)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces.to("cuda:1")))
        with self.assertRaises(RuntimeError):
            scene.build()

    def test_cross_device_vertex_update_is_rejected(self):
        vertices, faces = _grid_mesh(torch.device("cuda", 0), cells=3)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
        scene.build()
        with self.assertRaises(RuntimeError):
            scene.update_mesh_vertices(0, vertices.to("cuda:1"))


if __name__ == "__main__":
    unittest.main()
