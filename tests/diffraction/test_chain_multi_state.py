# Copyright Xingyu Chen.
# Checks that chained diffraction accumulation sums over its initial states in both backends.

"""`accum_dfr` (order 2 and 3) sums over initial states, as `accum_dfr_direct` does.

Monte-Carlo lane l draws initial state l % S and weighs it by S / samples, so S copies of one state must give S times
the single-state power, in the primal and in the fixed-path derivatives, on direct and Keller lanes. Each lane adds the
square root of its power contribution to `field_x`, so the field scales by sqrt(S). A one-cell grid keeps the lane to
cell assignment independent of S, so every lane sees the same geometry in both calls and only the weight differs.
"""

from __future__ import annotations

import math
import unittest

from tests.support.backends import drjit_modules, torch_cuda_available

COPIES = 3
REL = 1.0e-5
INITIAL = dict(
    edge_pos=[0.0, 0.0, 0.0],
    edge_dir=[1.0, 0.0, 0.0],
    n0=[0.0, 1.0, 0.0],
    n1=[0.0, -1.0, 0.0],
    src=[0.0, 0.0, 1.0],
    wi=[0.0, 0.0, -1.0],
    d0=[0.0, 0.0, -1.0],
)
RECURSIVE_Y = (0.5, 1.0)
STRATEGIES = (("direct", dict(direct=64, keller=0)), ("keller", dict(direct=0, keller=64)))


def _assert_scaled(testcase, label, single, copies, factor):
    for index, (a, b) in enumerate(zip(single, copies)):
        tolerance = REL * max(abs(factor * a), 1.0e-30)
        testcase.assertLessEqual(abs(b - factor * a), tolerance, f"{label}[{index}]: {b} != {factor} * {a}")


def _flat(grads):
    return [float(value) for grad in grads for value in grad.flatten().tolist()]


def _assert_state_sum(testcase, single, copies):
    """`single`/`copies`: (power, field_x_re, power gradients, field gradients) for 1 and COPIES states."""
    testcase.assertGreater(single[0], 0.0)
    _assert_scaled(testcase, "power", [single[0]], [copies[0]], COPIES)
    _assert_scaled(testcase, "field_x_re", [single[1]], [copies[1]], math.sqrt(COPIES))
    _assert_scaled(testcase, "power gradient", single[2], copies[2], COPIES)
    _assert_scaled(testcase, "field_x_re gradient", single[3], copies[3], math.sqrt(COPIES))


@unittest.skipUnless(torch_cuda_available(), "CUDA Torch and rayd.torch are required")
class TorchChainStateSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        import rayd.torch as rt

        cls.torch, cls.rt = torch, rt
        scene = rt.Scene()
        verts = torch.tensor([[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]], device="cuda")
        scene.add_mesh(rt.Mesh(verts, torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)))
        scene.build()
        cls.scene = scene

    def tensor(self, rows, dtype=None):
        return self.torch.tensor(rows, device="cuda", dtype=dtype or self.torch.float32)

    def initial(self, count):
        torch = self.torch
        return self.rt.DfrStates(
            edge_index=self.tensor([0] * count, torch.int32),
            **{name: self.tensor([value] * count) for name, value in INITIAL.items()},
            edge_t_min=self.tensor([-0.5] * count),
            edge_t_max=self.tensor([0.5] * count),
            prim0=self.tensor([-1] * count, torch.int32),
            prim1=self.tensor([-1] * count, torch.int32),
            exterior_angle=self.tensor([1.5 * math.pi] * count),
            src_power=self.tensor([2.0] * count),
            count=count,
        )

    def recursive(self, count, edge_pos):
        torch = self.torch
        return self.rt.DfrStates(
            edge_index=self.tensor([1, 2][:count], torch.int32),
            edge_pos=edge_pos,
            edge_dir=self.tensor([[1.0, 0.0, 0.0]] * count),
            edge_t_min=self.tensor([-0.5] * count),
            edge_t_max=self.tensor([0.5] * count),
            n0=self.tensor([[0.0, 1.0, 0.0]] * count),
            n1=self.tensor([[0.0, -1.0, 0.0]] * count),
            prim0=self.tensor([-1] * count, torch.int32),
            prim1=self.tensor([-1] * count, torch.int32),
            exterior_angle=self.tensor([1.5 * math.pi] * count),
            src=self.tensor([[0.0, 0.0, 1.0]] * count),
            src_power=self.tensor([1.0] * count),
            wi=self.tensor([[0.0, 1.0, 0.0]] * count),
            d0=self.tensor([[0.0, 0.0, -1.0]] * count),
            count=count,
        )

    def run_chain(self, initial_count, max_order, samples, gain, rec_edge_pos):
        material = self.rt.DfrMaterial(
            eta_r=self.tensor([4.0]),
            sigma=self.tensor([0.0]),
            mu_r=self.tensor([1.0]),
            gain=gain,
            valid=self.tensor([True], self.torch.bool),
        )
        return self.scene.accum_dfr(
            initial_states=self.initial(initial_count),
            recursive_states=self.recursive(max_order - 1, rec_edge_pos),
            grid=self.rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1),
            material=material,
            options=self.rt.DfrOptions(
                wavelength=0.125,
                seed=17,
                direct_samples=samples["direct"],
                keller_samples=samples["keller"],
                max_order=max_order,
            ),
        )

    def test_initial_state_copies_scale_primal_and_gradients(self):
        for max_order in (2, 3):
            for label, samples in STRATEGIES:
                with self.subTest(max_order=max_order, strategy=label):
                    runs = []
                    for count in (1, COPIES):
                        gain = self.tensor([1.0]).requires_grad_()
                        rec_edge_pos = self.tensor([[0.0, y, 0.0] for y in RECURSIVE_Y[: max_order - 1]])
                        rec_edge_pos.requires_grad_()
                        out = self.run_chain(count, max_order, samples, gain, rec_edge_pos)
                        power, field = out.power.sum(), out.field_x_re.sum()
                        leaves = (gain, rec_edge_pos)
                        power_grads = self.torch.autograd.grad(power, leaves, retain_graph=True)
                        field_grads = self.torch.autograd.grad(field, leaves)
                        runs.append((float(power), float(field), _flat(power_grads), _flat(field_grads)))
                    _assert_state_sum(self, *runs)

    def test_multi_state_gradient_matches_central_differences(self):
        from tests.support.multipath_fd_cases import assert_directional, torch_grad_along_direction

        for label, samples in STRATEGIES:
            with self.subTest(strategy=label):
                leaves = {
                    "gain": self.tensor([1.0]).requires_grad_(),
                    "rec_edge_pos": self.tensor([[0.0, 0.5, 0.0]]).requires_grad_(),
                }

                def loss_at(values, samples=samples):
                    out = self.run_chain(COPIES, 2, samples, values["gain"], values["rec_edge_pos"])
                    return out.power.sum() + out.field_x_re.sum()

                loss_at(leaves).backward()
                result = torch_grad_along_direction(leaves, loss_at, step=2e-3, seed=0)
                assert_directional(self, result, label=f"accum_dfr {COPIES} states {label}")


class DrJitChainStateSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mods = drjit_modules()
        if cls.mods is None:
            raise unittest.SkipTest("CUDA rayd.drjit is required")
        cuda, pj = cls.mods.cuda, cls.mods.pj
        scene = pj.Scene()
        scene.add_mesh(
            pj.Mesh(cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0]), cuda.Array3i([0], [1], [2]))
        )
        scene.build()
        cls.scene = scene

    def vec(self, rows):
        ad = self.mods.ad
        return ad.Array3f([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])

    def initial(self, count):
        ad, pj = self.mods.ad, self.mods.pj
        states = pj.DfrStatesAD()
        states.count = count
        states.edge_index = ad.Int([0] * count)
        for name, value in INITIAL.items():
            setattr(states, name, self.vec([value] * count))
        states.edge_t_min, states.edge_t_max = ad.Float([-0.5] * count), ad.Float([0.5] * count)
        states.prim0, states.prim1 = ad.Int([-1] * count), ad.Int([-1] * count)
        states.exterior_angle = ad.Float([1.5 * math.pi] * count)
        states.src_power = ad.Float([2.0] * count)
        states.prefix_depth = ad.Int([0] * count)
        return states

    def recursive(self, count):
        ad, pj = self.mods.ad, self.mods.pj
        states = pj.DfrStatesAD()
        states.count = count
        states.edge_index = ad.Int([1, 2][:count])
        states.edge_pos = self.vec([[0.0, y, 0.0] for y in RECURSIVE_Y[:count]])
        states.edge_dir = self.vec([[1.0, 0.0, 0.0]] * count)
        states.edge_t_min, states.edge_t_max = ad.Float([-0.5] * count), ad.Float([0.5] * count)
        states.n0, states.n1 = self.vec([[0.0, 1.0, 0.0]] * count), self.vec([[0.0, -1.0, 0.0]] * count)
        states.prim0, states.prim1 = ad.Int([-1] * count), ad.Int([-1] * count)
        states.exterior_angle = ad.Float([1.5 * math.pi] * count)
        states.src, states.src_power = self.vec([[0.0, 0.0, 1.0]] * count), ad.Float([1.0] * count)
        states.wi, states.d0 = self.vec([[0.0, 1.0, 0.0]] * count), self.vec([[0.0, 0.0, -1.0]] * count)
        states.prefix_depth = ad.Int([0] * count)
        return states

    def options(self, max_order, direct, keller):
        pj = self.mods.pj
        options = pj.DfrOptions()
        options.wavelength, options.k, options.seed = 0.125, 2.0 * math.pi / 0.125, 17
        options.samples, options.max_order = direct + keller, max_order
        options.direct_samples, options.keller_samples, options.suffix_samples = direct, keller, 0
        options.strategy_mask = (pj.DfrStrategy.DIRECT if direct else pj.DfrStrategy(0)) | (
            pj.DfrStrategy.KELLER if keller else pj.DfrStrategy(0)
        )
        options.sample_sequence = pj.DfrSampleSequence.HASH
        options.receiver_model = pj.DfrReceiverModel.MATCHED_ISO
        return options

    def grid(self):
        grid = self.mods.pj.DfrGrid()
        grid.axis, grid.position = 2, -1.0
        grid.coord0_min, grid.coord0_max, grid.coord1_min, grid.coord1_max = -1.0, 1.0, -1.0, 1.0
        grid.resolution0 = grid.resolution1 = 1
        grid.cell_area = 4.0
        return grid

    def test_initial_state_copies_scale_primal_and_gradients(self):
        dr, ad, pj = self.mods.dr, self.mods.ad, self.mods.pj
        for max_order in (2, 3):
            for label, samples in STRATEGIES:
                with self.subTest(max_order=max_order, strategy=label):
                    runs = []
                    for count in (1, COPIES):
                        values, grads = [], []
                        for pick in ("power", "field"):
                            gain = ad.Float([1.0])
                            dr.enable_grad(gain)
                            material = pj.DfrMaterialAD()
                            material.eta_r, material.sigma, material.mu_r = (
                                ad.Float([4.0]),
                                ad.Float([0.0]),
                                ad.Float([1.0]),
                            )
                            material.gain, material.valid = gain, ad.Bool([True])
                            out = self.scene.accum_dfr(
                                self.initial(count),
                                self.recursive(max_order - 1),
                                self.grid(),
                                material,
                                self.options(max_order, **samples),
                                True,
                            )
                            loss = dr.sum(out.power) if pick == "power" else dr.sum(out.field_x.real)
                            dr.backward(loss)
                            values.append(float(loss[0]))
                            grads.append([float(dr.grad(gain)[0])])
                        runs.append((values[0], values[1], grads[0], grads[1]))
                    _assert_state_sum(self, *runs)


if __name__ == "__main__":
    unittest.main()
