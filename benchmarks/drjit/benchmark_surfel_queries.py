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
REPO_ROOT = THIS_FILE.parents[2]
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
import rayd.drjit as rd


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


def make_layered_surfel_field(
    width: int,
    height: int,
    layers: int,
    layer_spacing: float,
    scale: float,
    opacity: float,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[float]]:
    centers: list[list[float]] = []
    tangent_u: list[list[float]] = []
    tangent_v: list[list[float]] = []
    values: list[float] = []
    for layer in range(layers):
        z = layer * layer_spacing
        for y in range(height):
            for x in range(width):
                centers.append([(x - width * 0.5) * scale, (y - height * 0.5) * scale, z])
                tangent_u.append([scale, 0.0, 0.0])
                tangent_v.append([0.0, scale, 0.0])
                values.append(1.0)
    return centers, tangent_u, tangent_v, values


def make_layered_surfel_cloud(
    width: int,
    height: int,
    layers: int,
    layer_spacing: float,
    scale: float,
    opacity: float,
) -> rd.SurfelCloud:
    centers, tangent_u, tangent_v, values = make_layered_surfel_field(
        width,
        height,
        layers,
        layer_spacing,
        scale,
        opacity,
    )
    count = len(centers)
    return rd.SurfelCloud(
        cuda.Array3f(
            [point[0] for point in centers],
            [point[1] for point in centers],
            [point[2] for point in centers],
        ),
        cuda.Array3f(
            [basis[0] for basis in tangent_u],
            [basis[1] for basis in tangent_u],
            [basis[2] for basis in tangent_u],
        ),
        cuda.Array3f(
            [basis[0] for basis in tangent_v],
            [basis[1] for basis in tangent_v],
            [basis[2] for basis in tangent_v],
        ),
        cuda.Float([opacity] * count),
        cuda.Float(values),
    )


def make_layered_surfel_geometry(
    width: int,
    height: int,
    layers: int,
    layer_spacing: float,
    scale: float,
    z_offset: float = 0.0,
) -> rd.SurfelGeometry:
    centers, tangent_u, tangent_v, _ = make_layered_surfel_field(
        width,
        height,
        layers,
        layer_spacing,
        scale,
        1.0,
    )
    return rd.SurfelGeometry(
        cuda.Array3f(
            [point[0] for point in centers],
            [point[1] for point in centers],
            [point[2] + z_offset for point in centers],
        ),
        cuda.Array3f(
            [basis[0] for basis in tangent_u],
            [basis[1] for basis in tangent_u],
            [basis[2] for basis in tangent_u],
        ),
        cuda.Array3f(
            [basis[0] for basis in tangent_v],
            [basis[1] for basis in tangent_v],
            [basis[2] for basis in tangent_v],
        ),
    )


def make_surfel_geometry(side: int, spacing: float, ad_mode: bool = False) -> rd.SurfelGeometry:
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
    array3 = ad.Array3f if ad_mode else cuda.Array3f
    scalar = ad.Float if ad_mode else cuda.Float
    return rd.SurfelGeometry(
        array3(scalar(centers_x), scalar(centers_y), scalar(centers_z)),
        array3(scalar([radius] * count), scalar([0.0] * count), scalar([0.0] * count)),
        array3(scalar([0.0] * count), scalar([radius] * count), scalar([0.0] * count)),
    )


def make_rgb_appearance(count: int, ad_mode: bool = False) -> rd.SurfelAppearance:
    scalar = ad.Float if ad_mode else cuda.Float
    array3 = ad.Array3f if ad_mode else cuda.Array3f
    return rd.SurfelAppearance.rgb(
        scalar([0.75] * count),
        array3(scalar([0.2] * count), scalar([0.4] * count), scalar([0.6] * count)),
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


def materialize_render(out, include_normal: bool = False) -> None:
    values = [
        out.rgb,
        out.alpha,
        out.transmittance,
        out.depth,
        out.candidate_count,
        out.candidate_buffer_full,
    ]
    if include_normal:
        values.append(out.normal)
    dr.eval(*values)


def mean_float_array(values, count: int) -> float:
    dr.eval(values)
    if count <= 0:
        return 0.0
    return statistics.fmean(float(values[i]) for i in range(count))


def mean_int_array(values, count: int) -> float:
    dr.eval(values)
    if count <= 0:
        return 0.0
    return statistics.fmean(int(values[i]) for i in range(count))


def true_fraction(mask, count: int) -> float:
    dr.eval(mask)
    if count <= 0:
        return 0.0
    return statistics.fmean(1.0 if bool(mask[i]) else 0.0 for i in range(count))


def benchmark_mode(
    mode: rd.SurfelPrimitiveMode,
    cloud: rd.SurfelCloud,
    rays: rd.Ray,
    repeats: int,
    warmup: int,
    single_launch: bool = True,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = mode
    opts.single_launch = single_launch

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
        "trace_backend": "single_launch" if single_launch else "legacy_retrace",
        "surfel_count": scene.surfel_count,
        "triangle_count": scene.triangle_count,
        "valid_count": valid_count,
        "build_ms": build_ms,
        "trace": summarize(samples_ms),
    }


def benchmark_appearance_pipeline(
    side: int,
    spacing: float,
    rays: rd.Ray,
    ray_count: int,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True
    surfel_count = side * side

    dr.sync_thread()
    build_start = time.perf_counter()
    scene = rd.SurfelScene(make_surfel_geometry(side, spacing), opts)
    scene.build()
    dr.sync_thread()
    build_ms = (time.perf_counter() - build_start) * 1000.0

    appearance = make_rgb_appearance(surfel_count)
    update_samples: list[float] = []
    render_samples: list[float] = []
    for iteration in range(warmup + repeats):
        dr.sync_thread()
        start = time.perf_counter()
        scene.update_appearance(appearance)
        dr.sync_thread()
        update_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        out = scene.render(rays, rd.SurfelRenderOptions.rgb())
        materialize_render(out)
        dr.sync_thread()
        render_ms = (time.perf_counter() - start) * 1000.0

        if iteration >= warmup:
            update_samples.append(update_ms)
            render_samples.append(render_ms)

    rays_ad = rd.RayAD(
        ad.Array3f(
            ad.Float([float(rays.o[0][i]) for i in range(ray_count)]),
            ad.Float([float(rays.o[1][i]) for i in range(ray_count)]),
            ad.Float([float(rays.o[2][i]) for i in range(ray_count)]),
        ),
        ad.Array3f(
            ad.Float([float(rays.d[0][i]) for i in range(ray_count)]),
            ad.Float([float(rays.d[1][i]) for i in range(ray_count)]),
            ad.Float([float(rays.d[2][i]) for i in range(ray_count)]),
        ),
    )
    ad_scene = rd.SurfelScene(make_surfel_geometry(side, spacing, ad_mode=True), opts)
    ad_scene.build()
    rgb = ad.Array3f(
        ad.Float([0.2] * surfel_count),
        ad.Float([0.4] * surfel_count),
        ad.Float([0.6] * surfel_count),
    )
    dr.enable_grad(rgb)
    ad_scene.update_appearance(rd.SurfelAppearance.rgb(ad.Float([0.75] * surfel_count), rgb))
    dr.sync_thread()
    ad_start = time.perf_counter()
    out_ad = ad_scene.render(rays_ad, rd.SurfelRenderOptions.rgb())
    loss = dr.sum(out_ad.rgb[0])
    dr.backward(loss)
    dr.eval(dr.grad(rgb))
    dr.sync_thread()
    ad_ms = (time.perf_counter() - ad_start) * 1000.0

    return {
        "surfel_count": surfel_count,
        "ray_count": ray_count,
        "build_ms": build_ms,
        "appearance_update": summarize(update_samples),
        "render": summarize(render_samples),
        "ad_replay_backward_ms": ad_ms,
    }


def benchmark_candidate_render(
    cloud: rd.SurfelCloud,
    rays: rd.Ray,
    ray_count: int,
    candidate_hits: int,
    collect_candidate_stats: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.max_candidate_hits = candidate_hits
    opts.collect_candidate_stats = collect_candidate_stats
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True

    dr.sync_thread()
    build_start = time.perf_counter()
    scene = rd.SurfelScene(cloud, opts)
    scene.build()
    dr.sync_thread()
    build_ms = (time.perf_counter() - build_start) * 1000.0

    samples_ms: list[float] = []
    last = None
    for iteration in range(warmup + repeats):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            last = scene.render(rays, rd.SurfelRenderOptions.rgb())
            materialize_render(last)
            dr.sync_thread()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= warmup:
            samples_ms.append(elapsed_ms)

    alpha_mean = 0.0
    candidate_count_mean = 0.0
    candidate_buffer_full_fraction = 0.0
    if last is not None:
        alpha_mean = mean_float_array(last.alpha, ray_count)
        candidate_count_mean = mean_int_array(last.candidate_count, ray_count)
        candidate_buffer_full_fraction = true_fraction(last.candidate_buffer_full, ray_count)

    return {
        "candidate_hits": candidate_hits,
        "surfel_count": int(scene.surfel_count),
        "ray_count": int(ray_count),
        "build_ms": build_ms,
        "render": summarize(samples_ms),
        "render_ms_avg": statistics.fmean(samples_ms),
        "alpha_mean": alpha_mean,
        "candidate_count_mean": candidate_count_mean,
        "candidate_buffer_full_fraction": candidate_buffer_full_fraction,
    }


def benchmark_normal_output(
    cloud: rd.SurfelCloud,
    rays: rd.Ray,
    ray_count: int,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True

    scene = rd.SurfelScene(cloud, opts)
    scene.build()
    dr.sync_thread()

    rgb_only_samples: list[float] = []
    normal_samples: list[float] = []
    last_normal = None
    for iteration in range(warmup + repeats):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            out_rgb = scene.render(rays, rd.SurfelRenderOptions.rgb())
            materialize_render(out_rgb)
            dr.sync_thread()
            rgb_only_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            last_normal = scene.render(rays, rd.SurfelRenderOptions.rgb(normal=True))
            materialize_render(last_normal, include_normal=True)
            dr.sync_thread()
            normal_ms = (time.perf_counter() - start) * 1000.0

        if iteration >= warmup:
            rgb_only_samples.append(rgb_only_ms)
            normal_samples.append(normal_ms)

    normal_z_mean = 0.0
    if last_normal is not None:
        normal_z_mean = mean_float_array(last_normal.normal.z, ray_count)

    rgb_only_avg = statistics.fmean(rgb_only_samples)
    normal_avg = statistics.fmean(normal_samples)
    return {
        "surfel_count": int(scene.surfel_count),
        "ray_count": ray_count,
        "render_rgb_only": summarize(rgb_only_samples),
        "render_with_normal": summarize(normal_samples),
        "overhead_vs_rgb_only_ms": normal_avg - rgb_only_avg,
        "overhead_vs_rgb_only_fraction": (
            (normal_avg - rgb_only_avg) / rgb_only_avg if rgb_only_avg > 0.0 else 0.0
        ),
        "normal_z_mean": normal_z_mean,
    }


def benchmark_geometry_update(
    side: int,
    layers: int,
    layer_spacing: float,
    spacing: float,
    rays: rd.Ray,
    ray_count: int,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True

    scale = spacing * 0.48
    surfel_count = side * side * max(1, layers)
    geometry_a = make_layered_surfel_geometry(side, side, max(1, layers), layer_spacing, scale, 0.0)
    geometry_b = make_layered_surfel_geometry(side, side, max(1, layers), layer_spacing, scale, 0.25 * scale)
    appearance = make_rgb_appearance(surfel_count)

    scene = rd.SurfelScene(geometry_a, opts)
    scene.build()
    scene.update_appearance(appearance)
    dr.sync_thread()

    update_samples: list[float] = []
    render_samples: list[float] = []
    for iteration in range(warmup + repeats):
        geometry = geometry_b if iteration % 2 == 0 else geometry_a
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            scene.update_geometry(geometry)
            dr.sync_thread()
            update_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            out = scene.render(rays, rd.SurfelRenderOptions.rgb())
            materialize_render(out)
            dr.sync_thread()
            render_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= warmup:
            update_samples.append(update_ms)
            render_samples.append(render_ms)

    rebuild_samples: list[float] = []
    for iteration in range(warmup + repeats):
        geometry = geometry_b if iteration % 2 == 0 else geometry_a
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            rebuild_scene = rd.SurfelScene(geometry, opts)
            rebuild_scene.build()
            rebuild_scene.update_appearance(appearance)
            dr.sync_thread()
            rebuild_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= warmup:
            rebuild_samples.append(rebuild_ms)

    return {
        "surfel_count": surfel_count,
        "ray_count": ray_count,
        "semantics": "topology_stable_geometry_update_rebuilds_proxy_gas",
        "update_geometry": summarize(update_samples),
        "rebuild_scene": summarize(rebuild_samples),
        "post_update_render": summarize(render_samples),
        "scene_build_count": scene.build_count,
    }


def benchmark_miss_prepass(
    cloud: rd.SurfelCloud,
    rays: rd.Ray,
    ray_count: int,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True

    scene = rd.SurfelScene(cloud, opts)
    scene.build()
    dr.sync_thread()

    render_only_samples: list[float] = []
    prepass_total_samples: list[float] = []
    prepass_intersect_samples: list[float] = []
    last_intersection = None
    for iteration in range(warmup + repeats):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            dr.sync_thread()
            start = time.perf_counter()
            out = scene.render(rays, rd.SurfelRenderOptions.rgb())
            materialize_render(out)
            dr.sync_thread()
            render_only_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            last_intersection = scene.intersect(rays)
            materialize(last_intersection)
            dr.sync_thread()
            intersect_ms = (time.perf_counter() - start) * 1000.0

        if iteration >= warmup:
            render_only_samples.append(render_only_ms)
            prepass_intersect_samples.append(intersect_ms)
            prepass_total_samples.append(intersect_ms + render_only_ms)

    hit_fraction = 0.0
    if last_intersection is not None:
        hit_fraction = true_fraction(last_intersection.is_valid(), ray_count)

    return {
        "surfel_count": int(scene.surfel_count),
        "ray_count": ray_count,
        "hit_fraction": hit_fraction,
        "decision": "disabled_by_default",
        "render_only": summarize(render_only_samples),
        "prepass_intersect": summarize(prepass_intersect_samples),
        "prepass_total_no_compaction": summarize(prepass_total_samples),
        "overhead_vs_render_only_ms": (
            statistics.fmean(prepass_total_samples) - statistics.fmean(render_only_samples)
        ),
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
    parser.add_argument("--candidate-hits", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--collect-candidate-stats", action="store_true")
    parser.add_argument("--surfel-layers", type=int, default=1)
    parser.add_argument("--layer-spacing", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trace-backend", choices=["single-launch", "legacy-retrace"], default="single-launch")
    parser.add_argument("--fit-image-size", type=int, default=24)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-appearance", action="store_true")
    parser.add_argument("--skip-normal-output", action="store_true")
    parser.add_argument("--skip-geometry-update", action="store_true")
    parser.add_argument("--skip-miss-prepass", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-output-dir", type=Path, default=Path("artifacts/surfel_fit"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cloud = make_layered_surfel_cloud(
        args.grid_side,
        args.grid_side,
        max(1, args.surfel_layers),
        args.layer_spacing,
        args.spacing * 0.48,
        1.0,
    )
    extent = max(1.0, args.grid_side * args.spacing * 0.55)
    rays = make_ortho_rays(args.ray_side, args.ray_side, extent)
    prewarm_surfel_optix()

    result: dict[str, Any] = {
        "prewarmed": True,
        "ray_count": args.ray_side * args.ray_side,
        "surfel_count": args.grid_side * args.grid_side * max(1, args.surfel_layers),
        "surfel_layers": max(1, args.surfel_layers),
        "layer_spacing": args.layer_spacing,
        "trace_backend": args.trace_backend,
        "candidate_render": [
            benchmark_candidate_render(
                cloud,
                rays,
                args.ray_side * args.ray_side,
                candidate_hits,
                args.collect_candidate_stats,
                args.repeats,
                args.warmup,
            )
            for candidate_hits in args.candidate_hits
        ],
        "modes": [
            benchmark_mode(rd.SurfelPrimitiveMode.Icosahedron20,
                           cloud,
                           rays,
                           args.repeats,
                           args.warmup,
                           args.trace_backend == "single-launch"),
            benchmark_mode(rd.SurfelPrimitiveMode.QuadTriangles,
                           cloud,
                           rays,
                           args.repeats,
                           args.warmup,
                           args.trace_backend == "single-launch"),
            benchmark_mode(rd.SurfelPrimitiveMode.SingleTriangle,
                           cloud,
                           rays,
                           args.repeats,
                           args.warmup,
                           args.trace_backend == "single-launch"),
        ],
    }

    if not args.skip_fit:
        result["fit"] = fit_depth_image(args.fit_image_size, args.fit_image_size, args.image_output_dir)
    if not args.skip_appearance:
        result["appearance"] = benchmark_appearance_pipeline(
            args.grid_side,
            args.spacing,
            rays,
            args.ray_side * args.ray_side,
            args.repeats,
            args.warmup,
        )
    if not args.skip_normal_output:
        result["normal_output"] = benchmark_normal_output(
            cloud,
            rays,
            args.ray_side * args.ray_side,
            args.repeats,
            args.warmup,
        )
    if not args.skip_geometry_update:
        result["geometry_update"] = benchmark_geometry_update(
            args.grid_side,
            max(1, args.surfel_layers),
            args.layer_spacing,
            args.spacing,
            rays,
            args.ray_side * args.ray_side,
            args.repeats,
            args.warmup,
        )
    if not args.skip_miss_prepass:
        result["miss_prepass"] = benchmark_miss_prepass(
            cloud,
            rays,
            args.ray_side * args.ray_side,
            args.repeats,
            args.warmup,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=None if args.json else 2))


if __name__ == "__main__":
    main()
