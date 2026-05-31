import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
TESTS_DIR = os.path.normcase(str(THIS_FILE.parent))
REPO_ROOT = THIS_FILE.parent.parent
CWD = os.path.normcase(os.path.abspath(os.getcwd()))
sys.path = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or CWD)) != TESTS_DIR
]
sys.path.insert(0, str(REPO_ROOT))

import drjit as dr
import drjit.cuda
import rayd as rd


def make_wall_mesh() -> rd.Mesh:
    return rd.Mesh(
        dr.cuda.Array3f([1.0, 1.0, 1.0, 1.0],
                        [-1.0, 1.0, 1.0, -1.0],
                        [0.0, 0.0, 2.0, 2.0]),
        dr.cuda.Array3i([0, 0], [1, 2], [2, 3]),
    )


def make_ceiling_mesh() -> rd.Mesh:
    return rd.Mesh(
        dr.cuda.Array3f([-2.0, 2.0, 2.0, -2.0],
                        [-2.0, -2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0]),
        dr.cuda.Array3i([0, 0], [1, 2], [2, 3]),
    )


def make_rays(ray_count: int) -> rd.Ray:
    side = max(1, int(ray_count ** 0.5))
    count = side * side
    inv_sqrt2 = 2.0 ** -0.5
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for iy in range(side):
        y = -0.2 + 0.4 * ((iy + 0.5) / side)
        for ix in range(side):
            x = -0.2 + 0.4 * ((ix + 0.5) / side)
            xs.append(x)
            ys.append(y)
            zs.append(0.5)
    return rd.Ray(
        dr.cuda.Array3f(xs, ys, zs),
        dr.cuda.Array3f([inv_sqrt2] * count, [0.0] * count, [inv_sqrt2] * count),
    )


def summarize(samples_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples_ms)
    return {
        "samples_ms": samples_ms,
        "min_ms": min(samples_ms),
        "avg_ms": statistics.fmean(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
    }


def materialize(chain: rd.ReflectionChain) -> None:
    dr.eval(
        chain.bounce_count,
        chain.t,
        chain.shape_ids,
        chain.prim_ids,
        chain.global_prim_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full reflection export while materializing global primitive ids."
    )
    parser.add_argument("--ray-count", type=int, default=65536)
    parser.add_argument("--max-bounces", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene = rd.Scene()
    scene.add_mesh(make_wall_mesh())
    scene.add_mesh(make_ceiling_mesh())
    scene.build()
    rays = make_rays(args.ray_count)
    actual_ray_count = len(rays.o[0])

    def run() -> rd.ReflectionChain:
        return scene.trace_reflections(rays, max_bounces=args.max_bounces, symbolic=False)

    for _ in range(args.warmup):
        materialize(run())
        dr.sync_thread()

    samples_ms: list[float] = []
    audits: list[dict[str, Any]] = []
    last = None
    for _ in range(args.repeats):
        rd.native_launch_audit_clear()
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            last = run()
            materialize(last)
            dr.sync_thread()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        audits.append(rd.native_launch_audit().get("trace_reflections", {}))

    assert last is not None
    sanity = {
        "bounce_sum": int(dr.sum(last.bounce_count)[0]),
        "global_prim_sum": int(dr.sum(dr.select(last.global_prim_ids >= 0, last.global_prim_ids, 0))[0]),
        "global_prim_width": int(dr.width(last.global_prim_ids)),
    }
    payload = {
        "benchmark": "rayd_reflection_global_ids",
        "config": {
            "requested_ray_count": args.ray_count,
            "actual_ray_count": actual_ray_count,
            "max_bounces": args.max_bounces,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "sanity": sanity,
        "performance": summarize(samples_ms),
        "native_audit": audits,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
