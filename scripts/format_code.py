# Copyright Xingyu Chen.
# Formats or checks every maintained native and Python source file.

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh", ".cu"}
PYTHON_SUFFIXES = {".py", ".pyi"}
EXCLUDED_ROOTS = {"generated"}


def tracked_sources() -> tuple[list[str], list[str]]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    native: list[str] = []
    python: list[str] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        if relative.parts and relative.parts[0] in EXCLUDED_ROOTS:
            continue
        if not (ROOT / relative).is_file():
            continue
        normalized = relative.as_posix()
        if relative.suffix.lower() in NATIVE_SUFFIXES:
            native.append(normalized)
        elif relative.suffix.lower() in PYTHON_SUFFIXES:
            python.append(normalized)
    return native, python


def batches(paths: list[str], size: int = 64) -> Iterable[list[str]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]


def resolve_tool(environment_name: str, executable: str) -> str:
    configured = os.environ.get(environment_name)
    resolved = shutil.which(configured or executable)
    if resolved is None:
        setting = f" or set {environment_name}" if not configured else ""
        raise RuntimeError(f"Could not find {executable}{setting}.")
    return resolved


def run_formatter(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Report formatting drift without editing files.")
    parser.add_argument("--list", action="store_true", help="Print the governed files without running tools.")
    args = parser.parse_args()

    native, python = tracked_sources()
    if args.list:
        print(f"native={len(native)} python={len(python)} total={len(native) + len(python)}")
        for path in (*native, *python):
            print(path)
        return 0

    clang_format = resolve_tool("RAYD_CLANG_FORMAT", "clang-format")
    ruff = resolve_tool("RAYD_RUFF", "ruff")

    for batch in batches(native):
        command = [clang_format, "--style=file"]
        command.extend(["--dry-run", "--Werror"] if args.check else ["-i"])
        run_formatter([*command, *batch])

    for batch in batches(python):
        command = [ruff, "format", "--config", str(ROOT / "pyproject.toml")]
        if args.check:
            command.append("--check")
        run_formatter([*command, *batch])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
