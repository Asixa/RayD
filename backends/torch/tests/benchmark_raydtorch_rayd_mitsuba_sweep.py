from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .benchmark_raydtorch_rayd_mitsuba_stress import (
    RAYDI_ROOT,
    _cleanup_drjit,
    _cleanup_torch,
    _load_rayd,
    _make_grid_mesh_data,
    _make_ray_data,
    _mitsuba_forward_performance,
    _rayd_forward_performance,
    _speedups,
    _torch_forward_performance,
    _try_import_mitsuba,
)


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "mesh_resolutions": [16, 32],
        "total_rays": [4_096, 16_384],
        "ray_batch_side": 64,
        "repeats": 2,
        "warmup": 1,
    },
    "standard": {
        "mesh_resolutions": [64, 128, 256, 512, 768],
        "total_rays": [16_384, 65_536, 1_048_576],
        "ray_batch_side": 512,
        "repeats": 5,
        "warmup": 2,
    },
    "large": {
        "mesh_resolutions": [64, 128, 256, 512, 768, 1024],
        "total_rays": [65_536, 1_048_576, 10_485_760],
        "ray_batch_side": 1024,
        "repeats": 5,
        "warmup": 2,
    },
    "extreme": {
        "mesh_resolutions": [128, 256, 512, 768, 1024],
        "total_rays": [1_048_576, 10_485_760, 100_663_296],
        "ray_batch_side": 1024,
        "repeats": 5,
        "warmup": 2,
    },
}


def _parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip().replace("_", "")
            if part:
                out.append(int(part))
    if not out:
        raise ValueError("empty integer list")
    return out


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def _ray_batch_side_for_target(target_total_rays: int, configured_side: int) -> int:
    if target_total_rays <= 0:
        raise ValueError("total rays must be positive")
    configured_count = configured_side * configured_side
    if target_total_rays >= configured_count:
        return configured_side
    return max(1, int(math.ceil(math.sqrt(target_total_rays))))


def _augment_perf(
    result: dict[str, Any],
    *,
    requested_total_rays: int,
    ray_batch_size: int,
    batch_count: int,
    execute_total_rays: bool,
) -> dict[str, Any]:
    effective_total_rays = ray_batch_size * batch_count
    for phase_name in ("forward_static", "forward_dynamic"):
        phase = result.get(phase_name)
        if not phase:
            continue
        phase["ray_batch_size"] = ray_batch_size
        phase["requested_total_rays"] = requested_total_rays
        phase["effective_total_rays"] = effective_total_rays
        phase["batch_count"] = batch_count
        phase["execute_total_rays"] = execute_total_rays
        for stats in phase.get("performance", {}).values():
            total_ms = float(stats["avg_ms"]) * float(batch_count)
            stats["projected_total_ms"] = total_ms
            stats["effective_total_rays"] = effective_total_rays
            stats["ray_batch_size"] = ray_batch_size
            stats["batch_count"] = batch_count
            stats["executed_total_rays"] = execute_total_rays
    return result


def _run_case(args: argparse.Namespace, mesh_resolution: int, requested_total_rays: int) -> dict[str, Any]:
    ray_batch_side = _ray_batch_side_for_target(requested_total_rays, args.ray_batch_side)
    ray_batch_size = ray_batch_side * ray_batch_side
    batch_count = int(math.ceil(requested_total_rays / ray_batch_size))
    repeats = batch_count if args.execute_total_rays else args.repeats

    mesh_data = _make_grid_mesh_data(mesh_resolution)
    updated_mesh_data = _make_grid_mesh_data(mesh_resolution, x_offset=args.dynamic_x_offset)
    ray_data = _make_ray_data(ray_batch_side)
    updated_ray_data = _make_ray_data(ray_batch_side, x_offset=args.dynamic_x_offset)
    triangle_count = mesh_resolution * mesh_resolution * 2
    vertex_count = (mesh_resolution + 1) * (mesh_resolution + 1)

    backends: dict[str, Any] = {}

    if "raydtorch" in args.backends:
        backends["raydtorch"] = {
            "forward_static": _torch_forward_performance(
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=False,
                edges_enabled=args.edges,
                repeats=repeats,
                warmup=args.warmup,
            ),
            "forward_dynamic": _torch_forward_performance(
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=True,
                edges_enabled=args.edges,
                repeats=repeats,
                warmup=args.warmup,
            ),
        }
        _augment_perf(
            backends["raydtorch"],
            requested_total_rays=requested_total_rays,
            ray_batch_size=ray_batch_size,
            batch_count=batch_count,
            execute_total_rays=args.execute_total_rays,
        )
        _cleanup_torch()

    rayd = cuda = dr = None
    if "rayd" in args.backends:
        rayd, cuda, dr = _load_rayd(args.rayd_source, args.rayd_root)
        backends["rayd"] = {
            "forward_static": _rayd_forward_performance(
                rayd,
                cuda,
                dr,
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=False,
                repeats=repeats,
                warmup=args.warmup,
            ),
            "forward_dynamic": _rayd_forward_performance(
                rayd,
                cuda,
                dr,
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=True,
                repeats=repeats,
                warmup=args.warmup,
            ),
        }
        _augment_perf(
            backends["rayd"],
            requested_total_rays=requested_total_rays,
            ray_batch_size=ray_batch_size,
            batch_count=batch_count,
            execute_total_rays=args.execute_total_rays,
        )
        _cleanup_drjit(dr)

    if "mitsuba" in args.backends:
        if dr is None:
            dr = importlib.import_module("drjit")
        mi = _try_import_mitsuba(args.mitsuba_variant)
        if mi is None:
            if args.require_mitsuba:
                raise RuntimeError("Mitsuba is not installed in the current environment.")
            backends["mitsuba"] = {"error": "Mitsuba is not installed in the current environment."}
        else:
            backends["mitsuba"] = {
                "forward_static": _mitsuba_forward_performance(
                    mi,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=False,
                    include_preliminary=args.mitsuba_preliminary,
                    repeats=repeats,
                    warmup=args.warmup,
                ),
                "forward_dynamic": _mitsuba_forward_performance(
                    mi,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=True,
                    include_preliminary=args.mitsuba_preliminary,
                    repeats=repeats,
                    warmup=args.warmup,
                ),
            }
            _augment_perf(
                backends["mitsuba"],
                requested_total_rays=requested_total_rays,
                ray_batch_size=ray_batch_size,
                batch_count=batch_count,
                execute_total_rays=args.execute_total_rays,
            )
            _cleanup_drjit(dr)

    return {
        "config": {
            "label": f"tri={_format_count(triangle_count)} rays={_format_count(requested_total_rays)}",
            "mesh_resolution": mesh_resolution,
            "triangle_count": triangle_count,
            "vertex_count": vertex_count,
            "requested_total_rays": requested_total_rays,
            "effective_total_rays": ray_batch_size * batch_count,
            "ray_batch_side": ray_batch_side,
            "ray_batch_size": ray_batch_size,
            "batch_count": batch_count,
            "timed_repeats": repeats,
            "warmup": args.warmup,
            "execute_total_rays": args.execute_total_rays,
            "dynamic_x_offset": args.dynamic_x_offset,
            "edges_enabled_for_raydtorch": args.edges,
        },
        "backends": backends,
        "speedups": _speedups(backends),
    }


def _rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in results["cases"]:
        cfg = case["config"]
        for backend, backend_result in case["backends"].items():
            if "error" in backend_result:
                continue
            for phase in ("forward_static", "forward_dynamic"):
                phase_result = backend_result[phase]
                for mode, stats in phase_result["performance"].items():
                    rows.append(
                        {
                            "backend": backend,
                            "phase": phase,
                            "mode": mode,
                            "mesh_resolution": cfg["mesh_resolution"],
                            "triangle_count": cfg["triangle_count"],
                            "vertex_count": cfg["vertex_count"],
                            "requested_total_rays": cfg["requested_total_rays"],
                            "effective_total_rays": cfg["effective_total_rays"],
                            "ray_batch_size": cfg["ray_batch_size"],
                            "batch_count": cfg["batch_count"],
                            "build_ms": phase_result["build_ms"],
                            "avg_ms": stats["avg_ms"],
                            "min_ms": stats["min_ms"],
                            "qps_m": stats["qps_m"],
                            "projected_total_ms": stats["projected_total_ms"],
                            "execute_total_rays": cfg["execute_total_rays"],
                        }
                    )
    return rows


def _write_csv(path: Path, results: dict[str, Any]) -> None:
    rows = _rows(results)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(results: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(results)
    if not rows:
        return []

    written: list[str] = []
    backends = sorted({row["backend"] for row in rows})
    phases = ["forward_static", "forward_dynamic"]
    modes = ["full", "reduced"]
    colors = {"raydtorch": "#2563eb", "rayd": "#16a34a", "mitsuba": "#c2410c"}
    markers = {"raydtorch": "o", "rayd": "s", "mitsuba": "^"}

    def save(fig: Any, name: str) -> None:
        png = output_dir / f"{name}.png"
        svg = output_dir / f"{name}.svg"
        fig.tight_layout()
        fig.savefig(png, dpi=180)
        fig.savefig(svg)
        plt.close(fig)
        written.extend([str(png), str(svg)])

    build_seen: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["backend"], int(row["triangle_count"]))
        if key not in build_seen:
            build_seen[key] = row
    fig, ax = plt.subplots(figsize=(8, 5))
    for backend in backends:
        pts = sorted(
            [row for (row_backend, _tri), row in build_seen.items() if row_backend == backend],
            key=lambda item: item["triangle_count"],
        )
        if not pts:
            continue
        ax.plot(
            [p["triangle_count"] for p in pts],
            [p["build_ms"] for p in pts],
            label=backend,
            color=colors.get(backend),
            marker=markers.get(backend, "o"),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Triangles")
    ax.set_ylabel("Build time (ms)")
    ax.set_title("Build scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save(fig, "build_vs_triangles")

    for phase in phases:
        for mode in modes:
            fig, ax = plt.subplots(figsize=(8, 5))
            for backend in backends:
                for requested_total_rays in sorted({row["requested_total_rays"] for row in rows}):
                    pts = sorted(
                        [
                            row
                            for row in rows
                            if row["backend"] == backend
                            and row["phase"] == phase
                            and row["mode"] == mode
                            and row["requested_total_rays"] == requested_total_rays
                        ],
                        key=lambda item: item["triangle_count"],
                    )
                    if len(pts) < 2:
                        continue
                    ax.plot(
                        [p["triangle_count"] for p in pts],
                        [p["qps_m"] for p in pts],
                        label=f"{backend} rays={_format_count(int(requested_total_rays))}",
                        color=colors.get(backend),
                        marker=markers.get(backend, "o"),
                        alpha=0.82,
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Triangles")
            ax.set_ylabel("Throughput (M rays/s)")
            ax.set_title(f"Throughput vs triangles: {phase} {mode}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
            save(fig, f"throughput_vs_triangles_{phase}_{mode}")

            fig, ax = plt.subplots(figsize=(8, 5))
            for backend in backends:
                for triangle_count in sorted({row["triangle_count"] for row in rows}):
                    pts = sorted(
                        [
                            row
                            for row in rows
                            if row["backend"] == backend
                            and row["phase"] == phase
                            and row["mode"] == mode
                            and row["triangle_count"] == triangle_count
                        ],
                        key=lambda item: item["requested_total_rays"],
                    )
                    if len(pts) < 2:
                        continue
                    ax.plot(
                        [p["requested_total_rays"] for p in pts],
                        [p["qps_m"] for p in pts],
                        label=f"{backend} tri={_format_count(int(triangle_count))}",
                        color=colors.get(backend),
                        marker=markers.get(backend, "o"),
                        alpha=0.82,
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Requested total rays")
            ax.set_ylabel("Throughput (M rays/s)")
            ax.set_title(f"Throughput vs total rays: {phase} {mode}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
            save(fig, f"throughput_vs_total_rays_{phase}_{mode}")

    speedup_rows = [row for row in rows if row["backend"] != "raydtorch" and row["mode"] in modes]
    for phase in phases:
        for mode in modes:
            fig, ax = plt.subplots(figsize=(8, 5))
            for backend in sorted({row["backend"] for row in speedup_rows}):
                for requested_total_rays in sorted({row["requested_total_rays"] for row in speedup_rows}):
                    pts: list[tuple[int, float]] = []
                    for row in speedup_rows:
                        if row["backend"] != backend or row["phase"] != phase or row["mode"] != mode:
                            continue
                        if row["requested_total_rays"] != requested_total_rays:
                            continue
                        torch_match = next(
                            (
                                base
                                for base in rows
                                if base["backend"] == "raydtorch"
                                and base["phase"] == phase
                                and base["mode"] == mode
                                and base["triangle_count"] == row["triangle_count"]
                                and base["requested_total_rays"] == row["requested_total_rays"]
                            ),
                            None,
                        )
                        if torch_match is not None:
                            pts.append((int(row["triangle_count"]), float(row["avg_ms"]) / float(torch_match["avg_ms"])))
                    pts.sort()
                    if len(pts) < 2:
                        continue
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        label=f"{backend}/RayDTorch rays={_format_count(int(requested_total_rays))}",
                        color=colors.get(backend),
                        marker=markers.get(backend, "o"),
                    )
            ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel("Triangles")
            ax.set_ylabel("Speedup over RayDTorch avg ms ratio")
            ax.set_title(f"RayDTorch speedup vs scale: {phase} {mode}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
            save(fig, f"speedup_vs_triangles_{phase}_{mode}")

    return written


def _default_output_dir(preset: str) -> Path:
    return Path("artifacts") / "benchmarks" / "scaling" / preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep RayDTorch/RayD/Mitsuba intersection scaling and plot results.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    parser.add_argument("--mesh-resolution", action="append", default=None)
    parser.add_argument("--total-rays", action="append", default=None)
    parser.add_argument("--ray-batch-side", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--execute-total-rays", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["raydtorch", "rayd", "mitsuba"],
        choices=["raydtorch", "rayd", "mitsuba"],
    )
    parser.add_argument("--edges", action="store_true", help="Enable RayDTorch edge cache during scene build.")
    parser.add_argument("--rayd-source", choices=("package", "local"), default="package")
    parser.add_argument("--rayd-root", type=Path, default=RAYDI_ROOT)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    parser.add_argument("--mitsuba-preliminary", action="store_true")
    parser.add_argument("--require-mitsuba", action="store_true")
    parser.add_argument("--dynamic-x-offset", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    args.mesh_resolutions = _parse_int_list(args.mesh_resolution, preset["mesh_resolutions"])
    args.total_rays_values = _parse_int_list(args.total_rays, preset["total_rays"])
    args.ray_batch_side = int(args.ray_batch_side or preset["ray_batch_side"])
    args.repeats = int(args.repeats or preset["repeats"])
    args.warmup = int(args.warmup or preset["warmup"])
    output_dir = args.output_dir or _default_output_dir(args.preset)
    json_output = args.json_output or (output_dir / "sweep.json")
    csv_output = args.csv_output or (output_dir / "sweep.csv")

    if "raydtorch" in args.backends and not torch.cuda.is_available():
        raise SystemExit("RayDTorch backend requires CUDA torch.")

    cases: list[dict[str, Any]] = []
    for mesh_resolution in args.mesh_resolutions:
        for total_rays in args.total_rays_values:
            cases.append(_run_case(args, mesh_resolution, total_rays))

    results = {
        "benchmark": "raydtorch_rayd_mitsuba_intersection_scaling_sweep",
        "suite_config": {
            "preset": args.preset,
            "mesh_resolutions": args.mesh_resolutions,
            "total_rays": args.total_rays_values,
            "ray_batch_side": args.ray_batch_side,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "execute_total_rays": args.execute_total_rays,
            "backends": args.backends,
            "rayd_source": args.rayd_source,
            "rayd_root": str(args.rayd_root) if args.rayd_source == "local" else None,
            "mitsuba_variant": args.mitsuba_variant if "mitsuba" in args.backends else None,
            "mitsuba_preliminary": args.mitsuba_preliminary,
        },
        "cases": cases,
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_output, results)
    plot_outputs: list[str] = []
    if not args.no_plots:
        plot_outputs = _plot_results(results, output_dir)
    results["outputs"] = {
        "json": str(json_output),
        "csv": str(csv_output),
        "plots": plot_outputs,
    }
    json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
