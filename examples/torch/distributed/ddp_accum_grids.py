# Copyright Xingyu Chen.
# Demonstrates ddp accum grids.

"""One rank per GPU: a Monte-Carlo accumulation split by lane window.

Run it with `torchrun`, one process per GPU:

```bash
torchrun --nnodes=1 --nproc_per_node=2 ddp_accum_grids.py
```

`accum_dfr_direct` has no batch axis to shard: its cost is the sample count, and
its result is a grid. So the split is over the Monte-Carlo *lane space*. A
launch describes a global space of `direct + keller + suffix` lanes and may
execute any sub-window of it: local lane `l` runs global lane
`lane_offset + l`. Handing rank `k` a disjoint window therefore draws exactly
the samples the single launch would have drawn -- the multiset of samples is a
property of the window partition, not of who ran it -- and the ranks' grids are
partial sums that add up to the single-launch grid. Only float summation order
differs, which the accumulation's own atomics already leave unpinned on one
device.

Traffic is one `all_reduce(SUM)` per grid, sized by the grid and not by the
sample count: doubling the samples costs nothing extra on the wire.

`rayd.drjit` follows the same recipe with `CUDA_VISIBLE_DEVICES` pinning; see
`README.md` in this directory.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys

import torch
import torch.distributed as dist

# Interior lane boundaries are kept on a whole number of warps so that each
# rank inherits the warp grouping the single launch has. Any exact partition of
# the lane space draws the same samples; this one also keeps the launch shapes
# comparable.
LANE_ALIGNMENT = 32

# The grids that make up the accumulation result. Reduced field by field so the
# script says exactly what crosses the interconnect.
FLOAT_GRIDS = ("power", "field_x_re", "field_x_im", "field_y_re", "field_y_im", "field_z_re", "field_z_im")
COUNT_GRIDS = (
    "direct_count",
    "keller_count",
    "suffix_count",
    "vis_rejects",
    "edge_vis_rejects",
    "utd_rejects",
    "edge_uses",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples",
        type=int,
        default=1 << 20,
        help="global Monte-Carlo sample count, rounded up to a whole number of warps per rank",
    )
    parser.add_argument("--resolution", type=int, default=8, help="grid resolution")
    parser.add_argument("--seed", type=int, default=7, help="Monte-Carlo seed")
    parser.add_argument("--out", type=str, default=None, help="rank 0 writes the merged grids here")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="collective timeout in seconds; a dead peer fails the run instead of hanging it",
    )
    return parser.parse_args(argv)


def resolve_total_samples(requested: int, world_size: int) -> int:
    """Round `requested` up so the ranks get equal, warp-aligned windows.

    A remainder would work too -- `lane_window()` handles it -- but an exactly
    divisible total keeps the example's arithmetic visible.
    """
    block = LANE_ALIGNMENT * world_size
    return max(1, math.ceil(max(int(requested), 1) / block)) * block


def lane_window(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Rank `rank`'s `(lane_offset, lane_count)` out of `[0, total)`.

    The windows are contiguous, disjoint, and cover the space exactly, which is
    the whole contract: the union of the ranks' launches is the single launch.
    """
    per_rank = (total // world_size) - (total // world_size) % LANE_ALIGNMENT
    begin = rank * per_rank
    count = total - begin if rank + 1 == world_size else per_rank
    return begin, count


def build_fixture(device: torch.device, resolution: int):
    """A wedge-lit two-state diffraction fixture: scene, states, material, grid.

    Small on purpose. The sample count, not the state count, is what the lane
    split divides, so a two-state fixture exercises the contract exactly as a
    large one would.
    """
    import rayd.torch as rt

    vertices = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], device=device, dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()

    def vec3(rows):
        return torch.tensor(rows, device=device, dtype=torch.float32)

    def scalars(values):
        return torch.tensor(values, device=device, dtype=torch.float32)

    states = rt.DfrStates(
        edge_index=torch.tensor([0, 1], device=device, dtype=torch.int32),
        edge_pos=vec3([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        edge_dir=vec3([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_t_min=scalars([-1.0, -1.0]),
        edge_t_max=scalars([1.0, 1.0]),
        n0=vec3([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        n1=vec3([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        prim0=torch.tensor([0, 0], device=device, dtype=torch.int32),
        prim1=torch.tensor([0, 0], device=device, dtype=torch.int32),
        exterior_angle=scalars([math.pi, math.pi]),
        src=vec3([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25]]),
        src_power=scalars([1.0, 1.0]),
    )
    material = rt.DfrMaterial(
        eta_r=torch.ones((1,), device=device, dtype=torch.float32),
        sigma=torch.zeros((1,), device=device, dtype=torch.float32),
        mu_r=torch.ones((1,), device=device, dtype=torch.float32),
        gain=torch.ones((1,), device=device, dtype=torch.float32),
        valid=torch.ones((1,), device=device, dtype=torch.bool),
    )
    grid = rt.DfrGrid(axis=2, position=0.0, resolution0=resolution, resolution1=resolution)
    return scene, states, material, grid


def accumulate(scene, states, material, grid, *, total, seed, begin, count):
    """One rank's partial grids over its lane window."""
    return scene.accum_dfr_direct(
        states=states,
        grid=grid,
        material=material,
        wavelength=1.0,
        direct_samples=total,
        seed=seed,
        lane_offset=begin,
        lane_count=count,
    )


def merged_grids(accum) -> dict[str, torch.Tensor]:
    """All-reduce every grid in place and return the merged tensors.

    Nothing else crosses the interconnect: no samples, no per-lane state, no
    scene data. The volume is `len(FLOAT_GRIDS) + len(COUNT_GRIDS)` grids
    however many samples were drawn.
    """
    merged: dict[str, torch.Tensor] = {}
    for name in FLOAT_GRIDS + COUNT_GRIDS:
        tensor = getattr(accum, name).detach().clone().contiguous()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        merged[name] = tensor
    return merged


def checksum(grids: dict[str, torch.Tensor]) -> float:
    """A float64 sum over every merged grid, printed as the run's fingerprint."""
    total = 0.0
    for name in FLOAT_GRIDS:
        total += float(grids[name].double().sum())
    return total


def run(args: argparse.Namespace, rank: int, world_size: int, device) -> None:
    total = resolve_total_samples(args.samples, world_size)
    begin, count = lane_window(total, rank, world_size)
    scene, states, material, grid = build_fixture(device, args.resolution)
    print(f"rank={rank} device={device} total_samples={total} lane_offset={begin} lane_count={count}", flush=True)

    accum = accumulate(scene, states, material, grid, total=total, seed=args.seed, begin=begin, count=count)
    local = float(accum.power.double().sum())
    grids = merged_grids(accum)

    print(
        f"rank={rank} local_power_sum={local!r} merged_power_sum={float(grids['power'].double().sum())!r}", flush=True
    )
    print(f"rank={rank} merged_direct_count={int(grids['direct_count'].sum())}", flush=True)
    print(f"rank={rank} merged_grid_checksum={checksum(grids)!r}", flush=True)

    if args.out is not None and rank == 0:
        payload = {name: tensor.to("cpu") for name, tensor in grids.items()}
        payload["total_samples"] = total
        payload["world_size"] = world_size
        payload["seed"] = int(args.seed)
        payload["resolution"] = int(args.resolution)
        torch.save(payload, args.out)
        print(f"rank={rank} wrote {args.out}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if "RANK" not in os.environ:
        print(
            "This example must be launched with torchrun, e.g.\n"
            "  torchrun --nnodes=1 --nproc_per_node=2 ddp_accum_grids.py",
            file=sys.stderr,
        )
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # A private OptiX disk cache per process; see `ddp_intersect_train.py` and
    # the README for why the shared default cache is a hazard here.
    cache_root = os.environ.get("RAYD_EXAMPLE_OPTIX_CACHE_ROOT")
    if cache_root and "OPTIX_CACHE_PATH" not in os.environ:
        path = os.path.join(cache_root, f"rank-{rank}")
        os.makedirs(path, exist_ok=True)
        os.environ["OPTIX_CACHE_PATH"] = path

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=args.timeout), device_id=device)
    try:
        run(args, rank, world_size, device)
        dist.barrier()
    except Exception as error:  # noqa: BLE001 - an example's top-level handler
        print(f"rank={rank} FAILED: {error!r}", file=sys.stderr, flush=True)
        return 1
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
