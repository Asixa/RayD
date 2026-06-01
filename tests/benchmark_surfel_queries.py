import argparse
import json
import math
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
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import rayd as rd


def summarize(samples_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples_ms)
    return {
        "samples_ms": samples_ms,
        "min_ms": min(samples_ms),
        "avg_ms": statistics.fmean(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
    }


def make_surfel_grid(side: int, spacing: float) -> rd.SurfelCloud:
    centers_x: list[float] = []
    centers_y: list[float] = []
    centers_z: list[float] = []
    half = 0.5 * (side - 1)
    for iy in range(side):
        for ix in range(side):
            centers_x.append((ix - half) * spacing)
            centers_y.append((iy - half) * spacing)
            centers_z.append(0.0)

    count = side * side
    radius = spacing * 0.48
    return rd.SurfelCloud(
        cuda.Array3f(centers_x, centers_y, centers_z),
        cuda.Array3f([radius] * count, [0.0] * count, [0.0] * count),
        cuda.Array3f([0.0] * count, [radius] * count, [0.0] * count),
        cuda.Float([1.0] * count),
    )


def make_ortho_rays(width: int, height: int, extent: float, z: float = 2.0) -> rd.Ray:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for iy in range(height):
        y = -extent + 2.0 * extent * ((iy + 0.5) / height)
        for ix in range(width):
            x = -extent + 2.0 * extent * ((ix + 0.5) / width)
            xs.append(x)
            ys.append(y)
            zs.append(z)
    return rd.Ray(
        cuda.Array3f(xs, ys, zs),
        cuda.Array3f([0.0] * len(xs), [0.0] * len(xs), [-1.0] * len(xs)),
    )


def materialize(its: rd.SurfelIntersection) -> None:
    dr.eval(its.t, its.surfel_id, its.triangle_id, its.gaussian_weight)


def benchmark_mode(
    mode: rd.SurfelPrimitiveMode,
    cloud: rd.SurfelCloud,
    rays: rd.Ray,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = mode

    dr.sync_thread()
    build_start = time.perf_counter()
    scene = rd.SurfelScene(cloud, opts)
    scene.build()
    dr.sync_thread()
    build_ms = (time.perf_counter() - build_start) * 1000.0

    for _ in range(warmup):
        materialize(scene.intersect(rays))
        dr.sync_thread()

    samples_ms: list[float] = []
    last = None
    for _ in range(repeats):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            last = scene.intersect(rays)
            materialize(last)
            dr.sync_thread()
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    valid_count = 0
    if last is not None:
        valid = last.is_valid()
        dr.eval(valid)
        valid_count = sum(1 for i in range(len(valid)) if bool(valid[i]))

    return {
        "mode": str(mode).split(".")[-1],
        "surfel_count": scene.surfel_count,
        "triangle_count": scene.triangle_count,
        "valid_count": valid_count,
        "build_ms": build_ms,
        "trace": summarize(samples_ms),
    }


def prewarm_surfel_optix() -> None:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    scene = rd.SurfelScene(
        rd.SurfelCloud(
            cuda.Array3f([0.0], [0.0], [0.0]),
            cuda.Array3f([1.0], [0.0], [0.0]),
            cuda.Array3f([0.0], [1.0], [0.0]),
            cuda.Float([1.0]),
        ),
        opts,
    )
    scene.build()
    ray = rd.Ray(
        cuda.Array3f([0.0], [0.0], [1.0]),
        cuda.Array3f([0.0], [0.0], [-1.0]),
    )
    materialize(scene.intersect(ray))
    dr.sync_thread()


def write_pgm(path: Path, values: list[float], width: int, height: int) -> None:
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    lo = min(finite) if finite else 0.0
    hi = max(finite) if finite else 1.0
    if abs(hi - lo) < 1e-8:
        hi = lo + 1.0

    pixels: list[int] = []
    for value in values:
        if value != value or abs(value) == float("inf"):
            pixels.append(0)
        else:
            pixels.append(max(0, min(255, int(round(255.0 * (value - lo) / (hi - lo))))))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write(f"P2\n{width} {height}\n255\n")
        for y in range(height):
            row = pixels[y * width:(y + 1) * width]
            file.write(" ".join(str(p) for p in row))
            file.write("\n")


def array_to_list(values, count: int) -> list[float]:
    dr.eval(values)
    return [float(values[i]) for i in range(count)]


def fit_depth_image(width: int, height: int, output_dir: Path) -> dict[str, Any]:
    count = width * height
    xs: list[float] = []
    ys: list[float] = []
    for iy in range(height):
        y = -0.7 + 1.4 * ((iy + 0.5) / height)
        for ix in range(width):
            x = -0.7 + 1.4 * ((ix + 0.5) / width)
            xs.append(x)
            ys.append(y)

    opts = rd.SurfelTraceOptions()
    opts.alpha_min = math.exp(-0.5 * 2.0 * 2.0)
    target_z = 0.3
    target_scene = rd.SurfelScene(
        rd.SurfelCloud(
            cuda.Array3f([0.0], [0.0], [target_z]),
            cuda.Array3f([0.8], [0.0], [0.0]),
            cuda.Array3f([0.0], [0.8], [0.0]),
            cuda.Float([1.0]),
        ),
        opts,
    )
    target_scene.build()
    target_ray = rd.Ray(
        cuda.Array3f(xs, ys, [1.0] * count),
        cuda.Array3f([0.0] * count, [0.0] * count, [-1.0] * count),
    )
    target_its = target_scene.intersect(target_ray)
    target_depth = array_to_list(target_its.t, count)

    z_value = -0.25
    initial_depth: list[float] | None = None
    final_depth: list[float] | None = None
    initial_rms = None
    final_rms = None

    for iteration in range(24):
        z = ad.Float([z_value])
        dr.enable_grad(z)
        scene = rd.SurfelScene(
            rd.SurfelCloud(
                ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), z),
                ad.Array3f(ad.Float([0.8]), ad.Float([0.0]), ad.Float([0.0])),
                ad.Array3f(ad.Float([0.0]), ad.Float([0.8]), ad.Float([0.0])),
                ad.Float([1.0]),
            ),
            opts,
        )
        scene.build()
        ray = rd.RayAD(
            ad.Array3f(ad.Float(xs), ad.Float(ys), ad.Float([1.0] * count)),
            ad.Array3f(ad.Float([0.0] * count),
                       ad.Float([0.0] * count),
                       ad.Float([-1.0] * count)),
        )
        its = scene.intersect(ray)
        residual = its.t - ad.Float(target_depth)
        loss = dr.sum(residual * residual) / count
        dr.backward(loss)

        rms = float(loss[0]) ** 0.5
        if initial_rms is None:
            initial_rms = rms
            initial_depth = array_to_list(its.t, count)
        final_rms = rms
        final_depth = array_to_list(its.t, count)
        z_value -= 0.45 * float(dr.grad(z)[0])

    write_pgm(output_dir / "target_depth.pgm", target_depth, width, height)
    write_pgm(output_dir / "initial_depth.pgm", initial_depth or target_depth, width, height)
    write_pgm(output_dir / "final_depth.pgm", final_depth or target_depth, width, height)

    return {
        "width": width,
        "height": height,
        "target_z": target_z,
        "fitted_z": z_value,
        "initial_rms": initial_rms,
        "final_rms": final_rms,
        "images": {
            "target_depth": str(output_dir / "target_depth.pgm"),
            "initial_depth": str(output_dir / "initial_depth.pgm"),
            "final_depth": str(output_dir / "final_depth.pgm"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RayD surfel ray tracing and run a synthetic depth-image fitting demo."
    )
    parser.add_argument("--grid-side", type=int, default=64)
    parser.add_argument("--ray-side", type=int, default=256)
    parser.add_argument("--spacing", type=float, default=0.08)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--fit-image-size", type=int, default=24)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-output-dir", type=Path, default=Path("artifacts/surfel_fit"))
    args = parser.parse_args()

    cloud = make_surfel_grid(args.grid_side, args.spacing)
    extent = max(1.0, args.grid_side * args.spacing * 0.55)
    rays = make_ortho_rays(args.ray_side, args.ray_side, extent)
    prewarm_surfel_optix()

    result: dict[str, Any] = {
        "prewarmed": True,
        "ray_count": args.ray_side * args.ray_side,
        "surfel_count": args.grid_side * args.grid_side,
        "modes": [
            benchmark_mode(rd.SurfelPrimitiveMode.Icosahedron20, cloud, rays, args.repeats, args.warmup),
            benchmark_mode(rd.SurfelPrimitiveMode.QuadTriangles, cloud, rays, args.repeats, args.warmup),
            benchmark_mode(rd.SurfelPrimitiveMode.SingleTriangle, cloud, rays, args.repeats, args.warmup),
        ],
    }

    if not args.skip_fit:
        result["fit"] = fit_depth_image(args.fit_image_size, args.fit_image_size, args.image_output_dir)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
