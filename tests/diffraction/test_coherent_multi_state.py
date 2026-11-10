# Copyright Xingyu Chen.
# Tests Torch diffraction accumulation lane bookkeeping: coherent superposition and per-backend lane counts.

import math
import unittest

import torch

try:
    import rayd.torch as rt
except ImportError:  # pragma: no cover - exercised only without the Torch backend
    rt = None


WAVELENGTH = 0.125
STATES = 3


def _rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        angle = 0.05 * index
        rows.append(
            dict(
                edge_dir=[math.cos(angle), math.sin(angle), 0.0],
                n0=[-math.sin(angle), math.cos(angle), 0.0],
                n1=[math.sin(angle), -math.cos(angle), 0.0],
                src=[0.05 * index, 0.1 - 0.04 * index, 1.0 + 0.1 * index],
                power=1.0 + 0.5 * index,
            )
        )
    return rows


def _tensor(values, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=dtype)


def _states(rows: list[dict]):
    count = len(rows)
    return rt.DfrStates(
        edge_index=_tensor(list(range(count)), torch.int32),
        edge_pos=_tensor([[0.0, 0.0, 0.0]] * count),
        edge_dir=_tensor([row["edge_dir"] for row in rows]),
        edge_t_min=_tensor([-0.5] * count),
        edge_t_max=_tensor([0.5] * count),
        n0=_tensor([row["n0"] for row in rows]),
        n1=_tensor([row["n1"] for row in rows]),
        prim0=_tensor([-1] * count, torch.int32),
        prim1=_tensor([-1] * count, torch.int32),
        exterior_angle=_tensor([1.5 * math.pi] * count),
        src=_tensor([row["src"] for row in rows]),
        src_power=_tensor([row["power"] for row in rows]),
        wi=_tensor([[0.0, 0.0, -1.0]] * count),
        d0=_tensor([[0.0, 0.0, -1.0]] * count),
        count=count,
    )


@unittest.skipUnless(rt is not None and torch.cuda.is_available(), "CUDA Torch backend is required")
class CoherentMultiStateTests(unittest.TestCase):
    def _accumulate(self, scene, rows):
        material = rt.DfrMaterial(
            eta_r=_tensor([4.0]),
            sigma=_tensor([0.0]),
            mu_r=_tensor([1.0]),
            gain=_tensor([1.0]),
            valid=_tensor([True], torch.bool),
        )
        grid = rt.DfrGrid(
            axis=2,
            position=-1.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=4,
            resolution1=4,
            cell_area=0.25,
        )
        return scene.accum_dfr_coherent_direct(
            states=_states(rows), grid=grid, material=material, wavelength=WAVELENGTH
        )

    def _scene(self, trace_backend: str):
        scene = rt.Scene(trace_backend=trace_backend)
        scene.add_mesh(
            rt.Mesh(
                _tensor([[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]]), _tensor([[0, 1, 2]], torch.int32)
            )
        )
        scene.build()
        return scene

    def _check_superposition(self, trace_backend: str) -> None:
        scene = self._scene(trace_backend)
        rows = _rows(STATES)
        singles = [self._accumulate(scene, rows[index : index + 1]) for index in range(STATES)]
        combined = self._accumulate(scene, rows)
        self.assertEqual(int(combined.direct_count.sum()), STATES * 16)
        for axis in "xyz":
            for part in ("re", "im"):
                name = f"direct_field_{axis}_{part}"
                expected = sum(getattr(single, name) for single in singles)
                torch.testing.assert_close(getattr(combined, name), expected, atol=1e-6, rtol=1e-5, msg=name)

    def test_cuda_trace_backend_field_is_the_sum_of_single_state_fields(self):
        # Each (state, cell) pair owns one lane, so a lane weighs its whole edge length; scaling it by the number
        # of states made the multi-state field sqrt(S) times each state's share.
        self._check_superposition("cuda")

    # The warp leader is picked by the %laneid register: an OptiX program has no meaningful threadIdx.x, and a wrong
    # leader dropped the warp sum of some same-cell lane groups (direct_count 32 instead of 48).
    def test_optix_trace_backend_field_is_the_sum_of_single_state_fields(self):
        self._check_superposition("optix")

    # Same warp-leader rule on the Monte-Carlo path: with 37 lanes (one full and one partial warp) every lane is
    # accumulated on the OptiX trace backend, as on the CUDA trace backend.
    def test_monte_carlo_lanes_of_a_partial_warp_are_all_accumulated_on_optix(self):
        rows = _rows(STATES)
        material = rt.DfrMaterial(
            eta_r=_tensor([4.0]),
            sigma=_tensor([0.0]),
            mu_r=_tensor([1.0]),
            gain=_tensor([1.0]),
            valid=_tensor([True], torch.bool),
        )
        grid = rt.DfrGrid(
            axis=2,
            position=-1.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=4,
            resolution1=4,
            cell_area=0.25,
        )
        results = {}
        for trace_backend in ("optix", "cuda"):
            scene = self._scene(trace_backend)
            results[trace_backend] = scene.accum_dfr_direct(
                states=_states(rows), grid=grid, material=material, wavelength=WAVELENGTH, seed=29, direct_samples=37
            )
        self.assertEqual(int(results["cuda"].direct_count.sum()), 37)
        self.assertEqual(int(results["optix"].direct_count.sum()), 37)
        torch.testing.assert_close(results["optix"].power, results["cuda"].power, atol=1e-9, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
