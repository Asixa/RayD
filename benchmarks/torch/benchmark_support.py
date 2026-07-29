# Copyright Xingyu Chen.
# Benchmarks benchmark support.

from __future__ import annotations

import time
from typing import Callable

import torch

import rayd.torch as rt


def synchronize() -> None:
    torch.cuda.synchronize()


def time_ms(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def dfr_case():
    states = rt.DfrStates(
        edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
        edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        edge_t_min=torch.tensor([-0.5], device="cuda", dtype=torch.float32),
        edge_t_max=torch.tensor([0.5], device="cuda", dtype=torch.float32),
        n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
        n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
        prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
        prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
        exterior_angle=torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32),
        src=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        src_power=torch.tensor([2.0], device="cuda", dtype=torch.float32),
        wi=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        count=1,
    )
    grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1, cell_area=4.0)
    material = rt.DfrMaterial.default(1, device=torch.device("cuda"), dtype=torch.float32)
    return states, grid, material
