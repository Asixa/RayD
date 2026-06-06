from __future__ import annotations

import argparse
import json
import time

import torch
import raydtorch as rt

from .benchmark_support import synchronize, time_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--queries", type=int, default=65536)
    args = parser.parse_args()

    n = args.grid
    xs, ys = torch.meshgrid(
        torch.linspace(0, 1, n, device="cuda"),
        torch.linspace(0, 1, n, device="cuda"),
        indexing="ij",
    )
    verts = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(n * n, device="cuda")], dim=1).contiguous()
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = torch.tensor(faces, device="cuda", dtype=torch.int32)

    scene = rt.Scene()
    t0 = time.perf_counter()
    scene.add_mesh(rt.Mesh(verts, faces, edges_enabled=True), dynamic=True)
    scene.build()
    synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0

    updated = verts.clone()
    updated[:, 2] = updated[:, 2] + 0.001
    sync_start = time.perf_counter()
    scene.update_mesh_vertices(0, updated)
    scene.sync()
    synchronize()
    dynamic_sync_ms = (time.perf_counter() - sync_start) * 1000.0

    ray = rt.Ray(
        torch.rand((args.queries, 3), device="cuda", dtype=torch.float32),
        torch.randn((args.queries, 3), device="cuda", dtype=torch.float32),
    )
    points = torch.rand((args.queries, 3), device="cuda", dtype=torch.float32)

    result = {
        "grid": n,
        "queries": args.queries,
        "build_ms": build_ms,
        "dynamic_sync_ms": dynamic_sync_ms,
        "intersect_ms": time_ms(lambda: scene.intersect(ray), 3, 10),
        "nearest_edge_ms": time_ms(lambda: scene.nearest_edge(points), 3, 10),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
