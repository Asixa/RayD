# Copyright Xingyu Chen.
# Demonstrates ddp intersect train.

"""One rank per GPU: sharded differentiable `intersect`, all-reduced gradients.

Run it with `torchrun`, one process per GPU:

```bash
torchrun --nnodes=1 --nproc_per_node=2 ddp_intersect_train.py
```

Every rank builds its *own* single-device `Scene` from the same mesh -- a scene
owns device-resident acceleration structures and is never shared -- and then
owns one slice of a global ray batch. Only the vertex gradient crosses the
interconnect: one `all_reduce(SUM)` of a `[V, 3]` tensor per step, whose size is
a property of the scene and not of the ray count. The optimizer then applies the
same update to the same replicated parameter on every rank, so the replicas stay
bitwise identical without any parameter traffic at all; the script asserts that
every `--check-every` steps and prints a hash of the final parameter so an
outside harness can check it too.

The loss is a height fit: rays fall straight down onto a grid mesh whose vertex
z coordinates are the parameter, and the target is the hit distance of a bump
surface. It is deliberately the plainest differentiable objective that exercises
`scene.intersect` -- the point of the example is the distributed shape around it.

`rayd.drjit` follows the same recipe with `CUDA_VISIBLE_DEVICES` pinning; see
`README.md` in this directory.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys

import torch
import torch.distributed as dist


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=24, help="optimizer steps")
    parser.add_argument("--rays", type=int, default=1 << 18, help="global ray batch size")
    parser.add_argument("--cells", type=int, default=24, help="grid mesh resolution")
    parser.add_argument("--lr", type=float, default=0.02, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="ray sampling seed")
    parser.add_argument(
        "--check-every", type=int, default=4, help="assert zero cross-rank parameter drift every N steps"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="collective timeout in seconds; a dead peer fails the run instead of hanging it",
    )
    return parser.parse_args(argv)


def grid_mesh(cells: int, span: float, device: torch.device):
    """Flat `z = 0` triangle grid over `[-span/2, span/2]^2`, identical everywhere.

    Built on the host from exact binary values and moved to the device, so every
    rank starts from bit-identical geometry.
    """
    axis = torch.linspace(-0.5 * span, 0.5 * span, cells + 1, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    flat_x = x.reshape(-1)
    vertices = torch.stack((flat_x, y.reshape(-1), torch.zeros_like(flat_x)), dim=1).contiguous()
    index = torch.arange((cells + 1) * (cells + 1), dtype=torch.int32).reshape(cells + 1, cells + 1)
    a = index[:-1, :-1].reshape(-1)
    b = index[:-1, 1:].reshape(-1)
    c = index[1:, :-1].reshape(-1)
    d = index[1:, 1:].reshape(-1)
    faces = torch.cat((torch.stack((a, b, c), dim=1), torch.stack((b, d, c), dim=1))).contiguous()
    return vertices.to(device), faces.to(device)


def global_rays(count: int, seed: int, extent: float, height: float, device):
    """The whole batch, sampled identically in every process.

    Sampling on the CPU from a seeded generator is what makes the shards a
    partition of one batch rather than `world_size` unrelated batches.
    """
    generator = torch.Generator().manual_seed(seed)
    xy = (torch.rand((count, 2), generator=generator, dtype=torch.float32) - 0.5) * (2.0 * extent)
    origins = torch.cat((xy, torch.full((count, 1), height, dtype=torch.float32)), dim=1)
    directions = torch.zeros((count, 3), dtype=torch.float32)
    directions[:, 2] = -1.0
    return origins.contiguous().to(device), directions.contiguous().to(device)


def shard_bounds(count: int, rank: int, world_size: int) -> tuple[int, int]:
    """`[begin, end)` of rank `rank`; the shards partition `[0, count)` exactly."""
    base, remainder = divmod(count, world_size)
    begin = rank * base + min(rank, remainder)
    return begin, begin + base + (1 if rank < remainder else 0)


def target_distance(origins: torch.Tensor, height: float) -> torch.Tensor:
    """Hit distance of the surface we are fitting, evaluated per ray."""
    radius_sq = origins[:, 0] ** 2 + origins[:, 1] ** 2
    return height - 0.3 * torch.exp(-2.0 * radius_sq)


def parameter_hash(vertices: torch.Tensor) -> str:
    """SHA-256 over the parameter's exact bytes -- equality here is bitwise."""
    return hashlib.sha256(vertices.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()


def train(args: argparse.Namespace, rank: int, world_size: int, device) -> None:
    import rayd.torch as rt

    span = 2.0
    ray_height = 2.0
    vertices, faces = grid_mesh(args.cells, span, device)
    vertices = vertices.requires_grad_(True)

    scene = rt.Scene()
    # Dynamic: the optimizer moves these vertices, and the scene has to be told.
    scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
    scene.build()

    origins, directions = global_rays(args.rays, args.seed, 0.45 * span, ray_height, device)
    begin, end = shard_bounds(args.rays, rank, world_size)
    shard_o = origins[begin:end].contiguous()
    shard_d = directions[begin:end].contiguous()
    target = target_distance(shard_o, ray_height)
    ray = rt.Ray(shard_o, shard_d)

    # Adam rather than SGD only because its step size does not depend on how
    # many rays happen to land on a vertex, so the example converges under any
    # `--rays` / `--cells` pair. Every rank runs the identical optimizer over
    # the identical reduced gradient, so the optimizer state is replicated too:
    # no optimizer state crosses the interconnect either.
    optimizer = torch.optim.Adam([vertices], lr=args.lr)
    print(
        f"rank={rank} device={device} vertices={vertices.shape[0]} rays={args.rays} shard=[{begin},{end})", flush=True
    )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        hit = scene.intersect(ray)
        # Rays fall vertically inside the grid's footprint, so every one of them
        # hits. Asserting it keeps a miss (which would carry a non-finite `t`
        # into the loss) from silently poisoning the gradient.
        if not bool(hit.is_valid().all()):
            raise RuntimeError("a shard ray missed the mesh; shrink --rays extent")
        # Normalizing by the *global* ray count, not the shard's, is what makes
        # the summed gradient the gradient of the global mean loss.
        shard_loss = ((hit.t - target) ** 2).sum() / args.rays
        shard_loss.backward()

        # A rank whose shard is empty (fewer rays than ranks) contributes no
        # gradient at all. It still has to join the all-reduce, or the ranks
        # that do have work would wait for it until the timeout.
        if vertices.grad is None:
            vertices.grad = torch.zeros_like(vertices)

        # This objective is a height fit, so only the z coordinates are free.
        # Letting x and y move would shrink the mesh's footprint until rays
        # fall off its edge -- a property of the toy objective, not of RayD,
        # and not worth confusing the distributed part of the example with.
        vertices.grad[:, :2] = 0.0

        # The only cross-rank traffic in the step: one scene-sized gradient.
        dist.all_reduce(vertices.grad, op=dist.ReduceOp.SUM)
        loss = shard_loss.detach().clone()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        optimizer.step()
        # The parameter is also the mesh, so push it back into the acceleration
        # structure before the next query.
        scene.update_mesh_vertices(0, vertices)
        scene.sync()

        if (step + 1) % args.check_every == 0 or step + 1 == args.steps:
            # `all_reduce` returns the same bits on every rank and the
            # optimizer is elementwise, so the replicas must be bitwise equal.
            # Compare against rank 0's copy rather than trusting that.
            reference = vertices.detach().clone()
            dist.broadcast(reference, src=0)
            drift = (vertices.detach() - reference).abs().max()
            if float(drift) != 0.0:
                raise RuntimeError(f"rank={rank} step={step + 1} parameter drift {float(drift)!r}")
            print(f"rank={rank} step={step + 1} loss={float(loss):.8e} drift=0", flush=True)

    digest = parameter_hash(vertices)
    print(f"rank={rank} final_param_sha256={digest}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if "RANK" not in os.environ:
        print(
            "This example must be launched with torchrun, e.g.\n"
            "  torchrun --nnodes=1 --nproc_per_node=2 ddp_intersect_train.py",
            file=sys.stderr,
        )
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # A private OptiX disk cache per process. Several processes compiling into
    # the shared default cache is the documented corruption hazard; the variable
    # is read when the OptiX context is created, i.e. on this rank's first RayD
    # query, so setting it here is early enough.
    cache_root = os.environ.get("RAYD_EXAMPLE_OPTIX_CACHE_ROOT")
    if cache_root and "OPTIX_CACHE_PATH" not in os.environ:
        path = os.path.join(cache_root, f"rank-{rank}")
        os.makedirs(path, exist_ok=True)
        os.environ["OPTIX_CACHE_PATH"] = path

    # Turn a dead peer into an error instead of an indefinite wait. Read when
    # the process group is created, so it has to be set before `init_*`.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=args.timeout), device_id=device)
    try:
        train(args, rank, world_size, device)
        # Everyone reaches the end together, so a rank that failed early cannot
        # be mistaken for a rank that finished.
        dist.barrier()
    except Exception as error:  # noqa: BLE001 - an example's top-level handler
        print(f"rank={rank} FAILED: {error!r}", file=sys.stderr, flush=True)
        return 1
    finally:
        # Without this a surviving rank blocks in NCCL until its own timeout.
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
