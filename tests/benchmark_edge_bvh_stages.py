import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts" / "benchmarks" / "edge_bvh_stages"
LOG_PATH = WORKSPACE_ROOT / "docs" / "edge_bvh_optimization_log.md"

SINGLE_MESH_ARGS = (
    "--scenario",
    "192:256",
    "--repeats",
    "5",
    "--warmup",
    "2",
)

PRESSURE_ARGS = (
    "--mesh-resolution",
    "32",
    "--tiles-x",
    "8",
    "--tiles-y",
    "8",
    "--tile-spacing",
    "1.25",
    "--row-z-stride",
    "0.02",
    "--query-grid-side",
    "128",
    "--point-z",
    "0.05",
    "--ray-z-origin",
    "1.5",
    "--finite-ray-tmax",
    "3.0",
    "--mask-keep-stride",
    "8",
    "--repeats",
    "5",
    "--warmup",
    "2",
)


def _run_json_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Benchmark command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(source_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _gpu_name() -> str | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        cwd=WORKSPACE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0]


def _relative(path: Path) -> str:
    return os.fspath(path.relative_to(WORKSPACE_ROOT))


def _extract_pressure_summary(pressure_payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = {
        scenario["name"]: scenario
        for scenario in pressure_payload["mask_scenarios"]
    }
    full = scenarios["full"]
    stride_sparse = scenarios["stride_sparse"]
    return {
        "scene": pressure_payload["scene"],
        "queries": pressure_payload["queries"],
        "build": pressure_payload["build"],
        "full": {
            "point_query": full["queries"]["point_query"],
            "finite_ray_query": full["queries"]["finite_ray_query"],
            "infinite_ray_query": full["queries"]["infinite_ray_query"],
        },
        "stride_sparse": {
            "point_query": stride_sparse["queries"]["point_query"],
            "finite_ray_query": stride_sparse["queries"]["finite_ray_query"],
            "infinite_ray_query": stride_sparse["queries"]["infinite_ray_query"],
            "sync_to_mask_edge_refit_ms": stride_sparse["sync_to_mask"]["profile_avg"]["edge_refit_ms"],
            "restore_full_edge_refit_ms": stride_sparse["restore_full"]["profile_avg"]["edge_refit_ms"],
        },
    }


def _build_summary(
    stage: str,
    include_gradients: bool,
    source_root: Path,
    single_mesh_payload: dict[str, Any],
    pressure_payload: dict[str, Any],
    single_mesh_raw: Path,
    pressure_raw: Path,
) -> dict[str, Any]:
    scenario = single_mesh_payload["scenarios"][0]
    return {
        "benchmark": "rayd_edge_bvh_stage",
        "stage": stage,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": _git_head(source_root),
            "gpu_name": _gpu_name(),
            "drjit_version": single_mesh_payload["environment"].get("drjit_version"),
            "single_mesh_include_gradients": include_gradients,
            "source_root": os.fspath(source_root),
        },
        "paths": {
            "single_mesh_raw": _relative(single_mesh_raw),
            "pressure_raw": _relative(pressure_raw),
        },
        "single_mesh": {
            "config": scenario["config"],
            "sanity": scenario["sanity"],
            "performance": scenario["performance"],
        },
        "pressure": _extract_pressure_summary(pressure_payload),
    }


def _metric(summary: dict[str, Any], key: str) -> float | None:
    lookup = {
        "single_build": ("single_mesh", "performance", "build", "avg_ms"),
        "single_point": ("single_mesh", "performance", "point_query", "avg_ms"),
        "single_finite": ("single_mesh", "performance", "finite_ray_query", "avg_ms"),
        "single_infinite": ("single_mesh", "performance", "infinite_ray_query", "avg_ms"),
        "single_sync": ("single_mesh", "performance", "sync", "avg_ms"),
        "pressure_build": ("pressure", "build", "avg_ms"),
        "pressure_full_finite": ("pressure", "full", "finite_ray_query", "avg_ms"),
        "pressure_full_infinite": ("pressure", "full", "infinite_ray_query", "avg_ms"),
        "pressure_stride_finite": ("pressure", "stride_sparse", "finite_ray_query", "avg_ms"),
        "pressure_stride_restore": ("pressure", "stride_sparse", "restore_full_edge_refit_ms"),
        "pressure_stride_sync": ("pressure", "stride_sparse", "sync_to_mask_edge_refit_ms"),
        "grad_point": ("single_mesh", "performance", "point_gradient", "avg_ms"),
        "grad_finite": ("single_mesh", "performance", "finite_ray_gradient", "avg_ms"),
    }
    value: Any = summary
    for part in lookup[key]:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return float(value)


def _format_metric(value: float | None, baseline: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.2f}"
    if baseline is None or baseline == 0.0 or value == baseline:
        return text
    delta_pct = ((value / baseline) - 1.0) * 100.0
    return f"{text} ({delta_pct:+.1f}%)"


def _render_log(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return (
            "# Edge BVH Optimization Log\n\n"
            "No stage benchmark data has been recorded yet.\n"
        )

    baseline = summaries[0]
    lines = [
        "# Edge BVH Optimization Log",
        "",
        "Auto-generated by `tests/benchmark_edge_bvh_stages.py`.",
        "",
        "固定场景：",
        "",
        "- Single-mesh: `192x192` mesh, `256x256` queries, `repeats=5`, `warmup=2`.",
        "- Pressure: `32x32` mesh, `8x8` tiles, `128x128` queries, `mask_keep_stride=8`, `repeats=5`, `warmup=2`.",
        "",
        "## Single-Mesh Metrics",
        "",
        "| Stage | Recorded At (UTC) | Build ms | Point ms | Finite ray ms | Infinite ray ms | Sync ms | Point grad ms | Finite grad ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        lines.append(
            "| {stage} | {recorded_at} | {build} | {point} | {finite} | {infinite} | {sync} | {grad_point} | {grad_finite} |".format(
                stage=summary["stage"],
                recorded_at=summary["recorded_at"],
                build=_format_metric(_metric(summary, "single_build"), _metric(baseline, "single_build")),
                point=_format_metric(_metric(summary, "single_point"), _metric(baseline, "single_point")),
                finite=_format_metric(_metric(summary, "single_finite"), _metric(baseline, "single_finite")),
                infinite=_format_metric(_metric(summary, "single_infinite"), _metric(baseline, "single_infinite")),
                sync=_format_metric(_metric(summary, "single_sync"), _metric(baseline, "single_sync")),
                grad_point=_format_metric(_metric(summary, "grad_point"), _metric(baseline, "grad_point")),
                grad_finite=_format_metric(_metric(summary, "grad_finite"), _metric(baseline, "grad_finite")),
            )
        )

    lines.extend(
        [
            "",
            "## Pressure Metrics",
            "",
            "| Stage | Pressure build ms | Full finite ray ms | Full infinite ray ms | Stride-sparse finite ray ms | Stride->mask refit ms | Stride->full refit ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for summary in summaries:
        lines.append(
            "| {stage} | {build} | {full_finite} | {full_infinite} | {stride_finite} | {stride_sync} | {stride_restore} |".format(
                stage=summary["stage"],
                build=_format_metric(_metric(summary, "pressure_build"), _metric(baseline, "pressure_build")),
                full_finite=_format_metric(_metric(summary, "pressure_full_finite"), _metric(baseline, "pressure_full_finite")),
                full_infinite=_format_metric(_metric(summary, "pressure_full_infinite"), _metric(baseline, "pressure_full_infinite")),
                stride_finite=_format_metric(_metric(summary, "pressure_stride_finite"), _metric(baseline, "pressure_stride_finite")),
                stride_sync=_format_metric(_metric(summary, "pressure_stride_sync"), _metric(baseline, "pressure_stride_sync")),
                stride_restore=_format_metric(_metric(summary, "pressure_stride_restore"), _metric(baseline, "pressure_stride_restore")),
            )
        )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{baseline['environment']['python_version']}`",
            f"- Platform: `{baseline['environment']['platform']}`",
            f"- Dr.Jit: `{baseline['environment'].get('drjit_version')}`",
        ]
    )

    git_head = baseline["environment"].get("git_head")
    if git_head:
        lines.append(f"- Baseline git head: `{git_head}`")
    gpu_name = baseline["environment"].get("gpu_name")
    if gpu_name:
        lines.append(f"- GPU: `{gpu_name}`")
    lines.append("")
    return "\n".join(lines)


def _update_log() -> None:
    summary_paths = sorted(
        ARTIFACT_ROOT.glob("*/summary.json"),
        key=lambda path: _load_json(path)["recorded_at"],
    )
    summaries = [_load_json(path) for path in summary_paths]
    LOG_PATH.write_text(_render_log(summaries), encoding="utf-8")


def run_stage(stage: str, include_gradients: bool, source_root: Path) -> Path:
    stage_dir = ARTIFACT_ROOT / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    single_mesh_raw = stage_dir / "single_mesh.json"
    pressure_raw = stage_dir / "pressure.json"
    summary_path = stage_dir / "summary.json"

    single_mesh_command = [
        sys.executable,
        os.fspath(source_root / "tests" / "benchmark_edge_queries.py"),
        *SINGLE_MESH_ARGS,
        "--json-output",
        os.fspath(single_mesh_raw),
    ]
    if include_gradients:
        single_mesh_command.append("--include-gradients")

    pressure_command = [
        sys.executable,
        os.fspath(source_root / "tests" / "benchmark_edge_bvh_pressure.py"),
        *PRESSURE_ARGS,
        "--json-output",
        os.fspath(pressure_raw),
    ]

    _run_json_command(single_mesh_command, source_root)
    _run_json_command(pressure_command, source_root)

    single_mesh_payload = _load_json(single_mesh_raw)
    pressure_payload = _load_json(pressure_raw)
    summary = _build_summary(
        stage,
        include_gradients,
        source_root,
        single_mesh_payload,
        pressure_payload,
        single_mesh_raw,
        pressure_raw,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _update_log()
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed Edge BVH stage benchmark matrix, save raw/summary JSON, "
            "and refresh docs/edge_bvh_optimization_log.md."
        )
    )
    parser.add_argument("--stage", required=True, help="Stage label, e.g. baseline, stage1_build, final.")
    parser.add_argument(
        "--include-gradients",
        action="store_true",
        help="Include gradient benchmarks in the fixed single-mesh scenario.",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=os.fspath(WORKSPACE_ROOT),
        help="Repository root to benchmark. Defaults to the current workspace.",
    )
    args = parser.parse_args()

    summary_path = run_stage(args.stage, args.include_gradients, Path(args.source_root).resolve())
    print(json.dumps(_load_json(summary_path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
