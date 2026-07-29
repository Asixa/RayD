# Copyright Xingyu Chen.
# Tests lane offset.

"""Monte-Carlo lane-window (`lane_offset` / `lane_count`) contract.

The diffraction accumulation entry points describe a global Monte-Carlo lane
space of `direct_samples + keller_samples + suffix_samples` lanes. A launch may
execute any sub-window of it: local lane `l` runs global lane
`lane_offset + l`, so a K-way split into `(offset_i, count_i)` windows draws
exactly the samples the single launch would draw, and the default window
`(0, -1)` is the unsharded launch it has always been.
"""

import unittest

import torch

import rayd.torch as rt


def _tri_scene():
    verts = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()
    return scene


def _states(requires_grad: bool = False):
    def vec3(rows):
        return torch.tensor(rows, device="cuda", dtype=torch.float32, requires_grad=requires_grad)

    def scalars(values):
        return torch.tensor(values, device="cuda", dtype=torch.float32, requires_grad=requires_grad)

    return rt.DfrStates(
        edge_index=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        edge_pos=vec3([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        edge_dir=vec3([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_t_min=scalars([-1.0, -1.0]),
        edge_t_max=scalars([1.0, 1.0]),
        n0=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        n1=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        prim0=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        prim1=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        exterior_angle=scalars([torch.pi, torch.pi]),
        src=vec3([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25]]),
        src_power=scalars([1.0, 1.0]),
    )


def _chain_scene_and_states(requires_grad: bool = False):
    """Order-2 chain fixture: one initial and one recursive diffraction state."""
    verts = torch.tensor(
        [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()

    def leaf(rows):
        return torch.tensor(rows, device="cuda", dtype=torch.float32, requires_grad=requires_grad)

    def fixed(rows):
        return torch.tensor(rows, device="cuda", dtype=torch.float32)

    def states(index, edge_pos, src, wi):
        return rt.DfrStates(
            edge_index=torch.tensor([index], device="cuda", dtype=torch.int32),
            edge_pos=leaf(edge_pos),
            edge_dir=leaf([[1.0, 0.0, 0.0]]),
            edge_t_min=leaf([-0.5]),
            edge_t_max=leaf([0.5]),
            n0=fixed([[0.0, 1.0, 0.0]]),
            n1=fixed([[0.0, -1.0, 0.0]]),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=leaf([1.5 * torch.pi]),
            src=leaf(src),
            src_power=leaf([2.0]),
            wi=fixed(wi),
            d0=fixed([[0.0, 0.0, -1.0]]),
            count=1,
        )

    initial = states(0, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]])
    recursive = states(1, [[0.0, 0.5, 0.0]], [[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]])
    return scene, initial, recursive


def _material():
    return rt.DfrMaterial(
        eta_r=torch.ones((1,), device="cuda", dtype=torch.float32),
        sigma=torch.zeros((1,), device="cuda", dtype=torch.float32),
        mu_r=torch.ones((1,), device="cuda", dtype=torch.float32),
        gain=torch.ones((1,), device="cuda", dtype=torch.float32),
        valid=torch.ones((1,), device="cuda", dtype=torch.bool),
    )


def _grid():
    return rt.DfrGrid(axis=2, position=0.0, resolution0=2, resolution1=2)


def _forward(
    scene,
    states,
    grid,
    material,
    *,
    direct_samples,
    keller_samples=0,
    suffix_samples=0,
    seed=0,
    export_tape=1,
    lane_window=None,
):
    """Call the native accumulation op, optionally on a `(offset, count)` window.

    `lane_window=None` omits the trailing lane arguments entirely, which is the
    pre-lane-window call shape.
    """
    args = [
        scene._require_native_scene(),
        None,
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        states.state_count,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        1.0,
        int(direct_samples),
        int(keller_samples),
        int(suffix_samples),
        int(seed),
        1,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        int(export_tape),
    ]
    if lane_window is not None:
        args += [None, None, int(lane_window[0]), int(lane_window[1])]
    return torch.ops.rayd_torch.diffraction_accumulation_forward(*args)


def _tape_rows(values):
    """Sorted `(state_idx, cell, edge_u bits)` rows of the active tape lanes."""
    active = values[14].to(torch.bool)
    state_idx = values[15][active].tolist()
    cell = values[16][active].tolist()
    edge_u = values[18][active].view(torch.int32).tolist()
    return sorted(zip(state_idx, cell, edge_u))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class LaneOffsetTests(unittest.TestCase):
    def setUp(self):
        self.scene = _tri_scene()
        self.states = _states()
        self.grid = _grid()
        self.material = _material()

    def _single(self, samples, seed=17):
        return _forward(
            self.scene,
            self.states,
            self.grid,
            self.material,
            direct_samples=samples,
            seed=seed,
        )

    def _windowed(self, samples, windows, seed=17):
        return [
            _forward(
                self.scene,
                self.states,
                self.grid,
                self.material,
                direct_samples=samples,
                seed=seed,
                lane_window=window,
            )
            for window in windows
        ]

    def test_default_window_is_bitwise_identical_to_omitting_the_arguments(self):
        omitted = self._single(64)
        explicit = _forward(
            self.scene,
            self.states,
            self.grid,
            self.material,
            direct_samples=64,
            seed=17,
            lane_window=(0, -1),
        )
        self.assertTrue(bool(omitted[14].any()), "the fixture must produce active lanes")
        for index, (a, b) in enumerate(zip(omitted, explicit)):
            if a is None:
                self.assertIsNone(b)
                continue
            self.assertTrue(torch.equal(a, b), f"output {index} differs")

    def test_full_window_by_explicit_count_is_bitwise_identical(self):
        omitted = self._single(64)
        explicit = _forward(
            self.scene,
            self.states,
            self.grid,
            self.material,
            direct_samples=64,
            seed=17,
            lane_window=(0, 64),
        )
        for index, (a, b) in enumerate(zip(omitted, explicit)):
            if a is None:
                self.assertIsNone(b)
                continue
            self.assertTrue(torch.equal(a, b), f"output {index} differs")

    def test_public_accum_dfr_direct_lane_offset_zero_matches_the_default(self):
        default = self.scene.accum_dfr_direct(
            states=self.states,
            grid=self.grid,
            material=self.material,
            wavelength=1.0,
            direct_samples=64,
            seed=17,
        )
        explicit = self.scene.accum_dfr_direct(
            states=self.states,
            grid=self.grid,
            material=self.material,
            wavelength=1.0,
            direct_samples=64,
            seed=17,
            lane_offset=0,
        )
        self.assertTrue(torch.equal(default.power, explicit.power))
        self.assertTrue(torch.equal(default.field_x_re, explicit.field_x_re))
        self.assertTrue(torch.equal(default.direct_count, explicit.direct_count))

    def _assert_sample_partition(self, samples, windows, seed=17):
        """The windows draw exactly the single launch's samples, once each."""
        single = self._single(samples, seed=seed)
        shards = self._windowed(samples, windows, seed=seed)
        expected_rows = _tape_rows(single)
        self.assertTrue(expected_rows, "the fixture must produce active tape rows")
        actual_rows = sorted(row for shard in shards for row in _tape_rows(shard))
        self.assertEqual(expected_rows, actual_rows)
        return single, shards

    def _assert_grid_partition(self, samples, windows, seed=17):
        """Additionally: the merged grid reproduces the single-launch grid.

        Grid accumulation aggregates within a warp before the atomic, and that
        aggregation already drops contributions for a partially filled warp
        (reproducible without any lane window: a plain `direct_samples=20`
        launch accumulates fewer samples than it tapes). That defect predates
        the lane window and is not ours to change here, so the merged-grid
        comparison uses warp-multiple window widths.
        """
        for _, count in windows:
            self.assertEqual(count % 32, 0, "grid merging needs warp-multiple windows")
        single, shards = self._assert_sample_partition(samples, windows, seed=seed)

        for name, index in (("power", 0), ("field_x_re", 1)):
            merged = torch.zeros_like(single[index])
            for shard in shards:
                merged = merged + shard[index]
            torch.testing.assert_close(
                merged, single[index], rtol=1e-5, atol=1e-6, msg=f"{name} mismatch"
            )

        merged_count = sum(int(shard[7].item()) for shard in shards)
        self.assertEqual(int(single[7].item()), merged_count)

    def test_two_way_split_partitions_the_lane_space(self):
        self._assert_grid_partition(64, ((0, 32), (32, 32)))

    def test_three_way_uneven_split_partitions_the_lane_space(self):
        self._assert_grid_partition(256, ((0, 32), (32, 96), (128, 128)))

    def test_arbitrary_window_widths_partition_the_sample_set(self):
        self._assert_sample_partition(64, ((0, 7), (7, 20), (27, 37)))
        self._assert_sample_partition(64, ((0, 1), (1, 62), (63, 1)))

    def test_degenerate_windows_are_inert(self):
        self._assert_grid_partition(64, ((0, 0), (0, 64), (64, 0)))

    def test_staged_accumulation_path_splits_the_lane_space(self):
        """The sort/reduce staging route (no tape, many samples) also shards."""
        samples = 4096
        kwargs = dict(direct_samples=samples, seed=17, export_tape=0)
        single = _forward(self.scene, self.states, self.grid, self.material, **kwargs)
        merged = torch.zeros_like(single[0])
        for window in ((0, 2048), (2048, 2048)):
            shard = _forward(
                self.scene,
                self.states,
                self.grid,
                self.material,
                lane_window=window,
                **kwargs,
            )
            merged = merged + shard[0]
        self.assertGreater(float(single[0].sum().item()), 0.0)
        torch.testing.assert_close(merged, single[0], rtol=1e-4, atol=1e-7)

    def test_window_must_stay_inside_the_lane_space(self):
        with self.assertRaises(RuntimeError):
            self._windowed(32, ((16, 24),))
        with self.assertRaises(RuntimeError):
            self._windowed(32, ((33, -1),))
        with self.assertRaises(RuntimeError):
            self._windowed(32, ((-1, 8),))

    def test_split_backward_matches_the_single_launch_gradient(self):
        samples = 128
        windows = ((0, 32), (32, 96))

        single_states = _states(requires_grad=True)
        single = self.scene.accum_dfr_direct(
            states=single_states,
            grid=self.grid,
            material=self.material,
            wavelength=1.0,
            direct_samples=samples,
            seed=17,
        )
        single.power.sum().backward()
        expected = single_states.edge_pos.grad

        split_states = _states(requires_grad=True)
        total = None
        for offset, count in windows:
            shard = self.scene.accum_dfr_direct(
                states=split_states,
                grid=self.grid,
                material=self.material,
                wavelength=1.0,
                direct_samples=samples,
                seed=17,
                lane_offset=offset,
                lane_count=count,
            )
            total = shard.power if total is None else total + shard.power
        torch.testing.assert_close(total, single.power.detach(), rtol=1e-5, atol=1e-6)
        total.sum().backward()

        self.assertIsNotNone(expected)
        self.assertGreater(float(expected.abs().sum().item()), 0.0)
        torch.testing.assert_close(
            split_states.edge_pos.grad, expected, rtol=1e-4, atol=1e-6
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class ChainLaneOffsetTests(unittest.TestCase):
    """`accum_dfr` (order-2 chain) carries the same lane window."""

    def _accum(self, requires_grad, **window):
        scene, initial, recursive = self._fixture
        return scene.accum_dfr(
            initial_states=initial,
            recursive_states=recursive,
            grid=rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2),
            material=_material(),
            wavelength=0.125,
            seed=17,
            direct_samples=64,
            keller_samples=64,
            max_order=2,
            **window,
        )

    def setUp(self):
        self._fixture = _chain_scene_and_states()

    def test_default_window_matches_lane_offset_zero(self):
        # Chain accumulation reduces through plain atomics, so repeated runs
        # already differ in the last ULP with identical arguments; the default
        # window is compared at that noise level rather than bitwise.
        default = self._accum(False)
        explicit = self._accum(False, lane_offset=0, lane_count=-1)
        torch.testing.assert_close(explicit.power, default.power, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(
            explicit.field_x_re, default.field_x_re, rtol=1e-6, atol=0.0
        )

    def test_split_merges_back_to_the_single_launch(self):
        single = self._accum(False)
        merged = None
        for offset, count in ((0, 32), (32, 96)):
            shard = self._accum(False, lane_offset=offset, lane_count=count)
            merged = shard.power if merged is None else merged + shard.power
        self.assertGreater(float(single.power.sum().item()), 0.0)
        torch.testing.assert_close(merged, single.power, rtol=1e-4, atol=1e-7)

    def test_split_backward_matches_the_single_launch_gradient(self):
        self._fixture = _chain_scene_and_states(requires_grad=True)
        single_edge_pos = self._fixture[1].edge_pos
        self._accum(True).power.sum().backward()
        expected = single_edge_pos.grad
        self.assertIsNotNone(expected)
        self.assertGreater(float(expected.abs().sum().item()), 0.0)

        self._fixture = _chain_scene_and_states(requires_grad=True)
        split_edge_pos = self._fixture[1].edge_pos
        total = None
        for offset, count in ((0, 32), (32, 96)):
            shard = self._accum(True, lane_offset=offset, lane_count=count)
            total = shard.power if total is None else total + shard.power
        total.sum().backward()
        torch.testing.assert_close(split_edge_pos.grad, expected, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
