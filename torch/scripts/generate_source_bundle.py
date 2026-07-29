# Copyright Xingyu Chen.
# Supports the Torch package's generate source bundle workflow.

"""Build the relocatable, integrity-described RayD Torch source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


INTEGRATION_ABI_PATH = "include/rayd/integration.h"
INTEGRATION_ABI_PATHS = (
    INTEGRATION_ABI_PATH,
    "include/rayd/diffraction.h",
    "include/rayd/penetration.h",
    "include/rayd/reflection.h",
    "include/rayd/scattering.h",
    "include/rayd/scene.h",
    "include/rayd/transmission.h",
    "include/rayd/visibility.h",
)
BUNDLED_PUBLIC_HEADERS = (
    *INTEGRATION_ABI_PATHS,
    "include/rayd/path_exchange.h",
)
SOURCE_INPUTS = (
    "LICENSE",
    "torch/CMakeLists.txt",
    "torch/scripts/embed_ptx.py",
    *BUNDLED_PUBLIC_HEADERS,
    "include/rayd/contracts.h",
    "include/rayd/field_transport.cuh",
    "include/rayd/math.h",
    "include/rayd/scattering_table.cuh",
    "include/rayd/bvh",
    "include/rayd/diffraction",
    "include/rayd/edge",
    "include/rayd/reflection",
    "include/rayd/rt",
    "include/rayd/scene",
    "include/rayd/sdf",
    "include/rayd/transmission",
    "include/rayd/visibility",
    "src",
    "cmake",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _header_set(source_root: Path) -> tuple[list[dict[str, str]], str]:
    headers = [
        {"path": relative, "sha256": _normalized_text_sha256(source_root / relative)}
        for relative in sorted(INTEGRATION_ABI_PATHS)
    ]
    digest = hashlib.sha256()
    for header in headers:
        digest.update(header["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(header["sha256"].encode("ascii"))
        digest.update(b"\n")
    return headers, digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_value(workspace: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for relative in SOURCE_INPUTS:
        candidate = workspace / relative
        if not candidate.exists():
            raise RuntimeError(f"required source input is missing: {relative}")
        if candidate.is_file():
            files.append(candidate)
            continue
        files.extend(path for path in candidate.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(workspace).as_posix())


def generate(
    workspace: Path,
    output: Path,
    *,
    distribution_version: str,
    commit: str | None,
    repository_url: str | None,
) -> None:
    workspace = workspace.resolve(strict=True)
    git_commit = _git_value(workspace, "rev-parse", "HEAD")
    git_repository_url = _git_value(workspace, "remote", "get-url", "origin")
    git_status = _git_value(workspace, "status", "--porcelain", "--untracked-files=normal")
    resolved_commit = commit or git_commit
    resolved_repository_url = repository_url or git_repository_url
    if not resolved_commit or len(resolved_commit) != 40:
        raise RuntimeError(
            "RayD source commit is unavailable; pass --commit when building outside a Git checkout"
        )
    if not resolved_repository_url:
        raise RuntimeError(
            "RayD repository URL is unavailable; pass --repository-url when building outside a Git checkout"
        )

    if output.exists():
        shutil.rmtree(output)
    source_root = output / "source"
    source_root.mkdir(parents=True)

    manifest_files: list[dict[str, str]] = []
    for source in _source_files(workspace):
        relative = source.relative_to(workspace)
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest_files.append(
            {"path": relative.as_posix(), "sha256": _sha256(destination)}
        )

    manifest_path = output / "source-files.json"
    _write_json(manifest_path, {"schema_version": 1, "files": manifest_files})
    integration_header = source_root / INTEGRATION_ABI_PATH
    header_text = integration_header.read_text(encoding="utf-8")
    if "kIntegrationApiVersion = 8;" not in header_text:
        raise RuntimeError("RayD integration API version is not the expected stable value 8")
    if '"rayd.torch.integration"' not in header_text:
        raise RuntimeError("RayD integration identity is not rayd.torch.integration")

    integration_headers, integration_digest = _header_set(source_root)
    _write_json(
        output / "rayd-source.json",
        {
            "schema_version": 2,
            "distribution": {
                "name": "rayd-torch",
                "version": distribution_version,
            },
            "repository_url": resolved_repository_url,
            "commit": resolved_commit,
            "dirty": git_status not in (None, ""),
            "source_root": "source",
            "source_manifest": {
                "path": "source-files.json",
                "sha256": _sha256(manifest_path),
            },
            "integration_abi": {
                "kind": "source-header-set-sha256",
                "entrypoint": INTEGRATION_ABI_PATH,
                "headers": integration_headers,
                "sha256": integration_digest,
                "api_version": 8,
                "identity": "rayd.torch.integration",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distribution-version", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--repository-url")
    arguments = parser.parse_args()
    generate(
        arguments.workspace,
        arguments.output,
        distribution_version=arguments.distribution_version,
        commit=arguments.commit or None,
        repository_url=arguments.repository_url or None,
    )


if __name__ == "__main__":
    main()
