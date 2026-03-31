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

MICRO_ARGS = (
    "--scenario",
    "48:128",
    "--repeats",
    "8",
    "--warmup",
    "3",
)

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

MODE_ENV_VARS = {
    "post_build_strategy": "RAYD_EDGE_BVH_POST_BUILD_STRATEGY",
    "build_stream_mode": "RAYD_EDGE_BVH_BUILD_STREAM_MODE",
    "finalize_mode": "RAYD_EDGE_BVH_FINALIZE_MODE",
    "treelet_schedule_mode": "RAYD_EDGE_BVH_TREELET_SCHEDULE_MODE",
    "compaction_mode": "RAYD_EDGE_BVH_COMPACTION_MODE",
    "build_algorithm": "RAYD_EDGE_BVH_BUILD_ALGORITHM",
    "node_layout_mode": "RAYD_EDGE_BVH_NODE_LAYOUT_MODE",
}

DEFAULT_EDGE_BVH_MODES = {
    "post_build_strategy": "gpu_treelet",
    "build_stream_mode": "overlap",
    "finalize_mode": "atomic",
    "treelet_schedule_mode": "flat_levels",
    "compaction_mode": "host_upload_raw",
    "build_algorithm": "lbvh",
    "node_layout_mode": "scalar_arrays",
}


def _run_json_command(command: list[str], cwd: Path, mode_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(mode_env)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
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
    mode_overrides: dict[str, str],
    micro_payload: dict[str, Any],
    single_mesh_payload: dict[str, Any],
    pressure_payload: dict[str, Any],
    micro_raw: Path,
    single_mesh_raw: Path,
    pressure_raw: Path,
) -> dict[str, Any]:
    micro_scenario = micro_payload["scenarios"][0]
    single_scenario = single_mesh_payload["scenarios"][0]
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
        "edge_bvh_modes": mode_overrides,
        "paths": {
            "micro_raw": _relative(micro_raw),
            "single_mesh_raw": _relative(single_mesh_raw),
            "pressure_raw": _relative(pressure_raw),
        },
        "micro": {
            "config": micro_scenario["config"],
            "sanity": micro_scenario["sanity"],
            "performance": micro_scenario["performance"],
        },
        "single_mesh": {
            "config": single_scenario["config"],
            "sanity": single_scenario["sanity"],
            "performance": single_scenario["performance"],
        },
        "pressure": _extract_pressure_summary(pressure_payload),
    }


def _metric(summary: dict[str, Any], key: str) -> float | None:
    lookup = {
        "micro_build": ("micro", "performance", "build", "avg_ms"),
        "micro_point": ("micro", "performance", "point_query", "avg_ms"),
        "micro_finite": ("micro", "performance", "finite_ray_query", "avg_ms"),
        "micro_infinite": ("micro", "performance", "infinite_ray_query", "avg_ms"),
        "micro_sync": ("micro", "performance", "sync", "avg_ms"),
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
        "Benchmark timing validity rule:",
        "",
        "- Every benchmark command must be run strictly serially, one process at a time.",
        "- Do not run multiple benchmark Python processes concurrently on the same machine/GPU, even for different stages or scenarios.",
        "- Any build/query/sync timing captured under concurrent benchmark load is invalid and must be discarded and rerun serially.",
        "",
        "Fixed scenarios:",
        "",
        "- Micro: `48x48` mesh, `128x128` queries, `repeats=8`, `warmup=3`.",
        "- Single-mesh: `192x192` mesh, `256x256` queries, `repeats=5`, `warmup=2`.",
        "- Pressure: `32x32` mesh, `8x8` tiles, `128x128` queries, `mask_keep_stride=8`, `repeats=5`, `warmup=2`.",
        "- Benchmark cleanliness: `stage2_clean_*` rows use explicit runtime modes with no fallback. Older `stage2_*` rows predate this harness change and are not directly comparable.",
        "",
        "## Micro Metrics",
        "",
        "| Stage | Micro build ms | Micro point ms | Micro finite ray ms | Micro infinite ray ms | Micro sync ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        lines.append(
            "| {stage} | {build} | {point} | {finite} | {infinite} | {sync} |".format(
                stage=summary["stage"],
                build=_format_metric(_metric(summary, "micro_build"), _metric(baseline, "micro_build")),
                point=_format_metric(_metric(summary, "micro_point"), _metric(baseline, "micro_point")),
                finite=_format_metric(_metric(summary, "micro_finite"), _metric(baseline, "micro_finite")),
                infinite=_format_metric(_metric(summary, "micro_infinite"), _metric(baseline, "micro_infinite")),
                sync=_format_metric(_metric(summary, "micro_sync"), _metric(baseline, "micro_sync")),
            )
        )

    lines.extend(
        [
            "",
            "## Single-Mesh Metrics",
            "",
            "| Stage | Recorded At (UTC) | Build ms | Point ms | Finite ray ms | Infinite ray ms | Sync ms | Point grad ms | Finite grad ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

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


def _effective_edge_bvh_modes(args: argparse.Namespace) -> dict[str, str]:
    modes = dict(DEFAULT_EDGE_BVH_MODES)
    for arg_name in MODE_ENV_VARS:
        value = getattr(args, arg_name)
        if value:
            modes[arg_name] = value
    return modes


def _mode_env_from_effective_modes(effective_modes: dict[str, str]) -> dict[str, str]:
    return {
        env_name: effective_modes[arg_name]
        for arg_name, env_name in MODE_ENV_VARS.items()
    }


def run_stage(
    stage: str,
    include_gradients: bool,
    source_root: Path,
    effective_modes: dict[str, str],
) -> Path:
    stage_dir = ARTIFACT_ROOT / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    micro_raw = stage_dir / "micro.json"
    single_mesh_raw = stage_dir / "single_mesh.json"
    pressure_raw = stage_dir / "pressure.json"
    summary_path = stage_dir / "summary.json"

    micro_command = [
        sys.executable,
        os.fspath(source_root / "tests" / "benchmark_edge_queries.py"),
        *MICRO_ARGS,
        "--json-output",
        os.fspath(micro_raw),
    ]
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

    mode_env = _mode_env_from_effective_modes(effective_modes)
    _run_json_command(micro_command, source_root, mode_env)
    _run_json_command(single_mesh_command, source_root, mode_env)
    _run_json_command(pressure_command, source_root, mode_env)

    micro_payload = _load_json(micro_raw)
    single_mesh_payload = _load_json(single_mesh_raw)
    pressure_payload = _load_json(pressure_raw)
    summary = _build_summary(
        stage,
        include_gradients,
        source_root,
        effective_modes,
        micro_payload,
        single_mesh_payload,
        pressure_payload,
        micro_raw,
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
    parser.add_argument(
        "--post-build-strategy",
        choices=("none", "hybrid_top_level_sah", "gpu_treelet"),
        help="Explicit Edge BVH post-build strategy for this stage run.",
    )
    parser.add_argument(
        "--build-stream-mode",
        choices=("serial", "overlap"),
        help="Explicit Edge BVH build stream mode for this stage run.",
    )
    parser.add_argument(
        "--finalize-mode",
        choices=("atomic", "level_by_level"),
        help="Explicit Edge BVH finalize mode for this stage run.",
    )
    parser.add_argument(
        "--treelet-schedule-mode",
        choices=("per_level_uploads", "flat_levels"),
        help="Explicit Edge BVH treelet schedule mode for this stage run.",
    )
    parser.add_argument(
        "--compaction-mode",
        choices=("host_upload_raw", "host_upload_exact", "gpu_emit"),
        help="Explicit Edge BVH compaction mode for this stage run.",
    )
    parser.add_argument(
        "--build-algorithm",
        choices=("lbvh", "ploc"),
        help="Explicit Edge BVH build algorithm for this stage run.",
    )
    parser.add_argument(
        "--node-layout-mode",
        choices=("scalar_arrays", "packed"),
        help="Explicit Edge BVH node layout mode for this stage run.",
    )
    args = parser.parse_args()

    summary_path = run_stage(
        args.stage,
        args.include_gradients,
        Path(args.source_root).resolve(),
        _effective_edge_bvh_modes(args),
    )
    print(json.dumps(_load_json(summary_path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
