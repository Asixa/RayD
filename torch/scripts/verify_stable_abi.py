# Copyright Xingyu Chen.
# Supports the Torch package's verify stable abi workflow.

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


TORCH_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TORCH_ROOT.parent


FORBIDDEN_SOURCE_MARKERS = ("at::", "c10::", "py::", "torch/extension.h", "torch/library.h")
FORBIDDEN_DEPENDENCIES = ("torch_python", "c10.dll", "c10_cuda.dll", "libc10.so", "libc10_cuda.so", "python3")
FORBIDDEN_DYNAMIC_SYMBOLS = ("at::", "c10::", "@at@@", "@c10@@")


def verify_sources(source_root: Path) -> None:
    sources = sorted(
        path
        for path in source_root.rglob("*")
        if path.suffix in {".h", ".hpp", ".cuh", ".cc", ".cpp", ".cu"} and "_stable." in path.name
    )
    if not sources:
        raise SystemExit(f"No Stable ABI sources found under {source_root}")
    violations: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for marker in FORBIDDEN_SOURCE_MARKERS:
            if marker in text:
                violations.append(f"{source}: forbidden marker {marker!r}")
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"Verified Stable ABI source boundary across {len(sources)} files")


def dependency_listing(binary: Path) -> str:
    if os.name == "nt":
        tool = shutil.which("dumpbin")
        if tool is None:
            program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            candidates = sorted(
                program_files.glob("Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"),
                reverse=True,
            )
            tool = str(candidates[0]) if candidates else None
        if tool is None:
            raise SystemExit("dumpbin is required for Stable ABI dependency auditing")
        command = [tool, "/dependents", str(binary)]
    else:
        tool = shutil.which("readelf")
        if tool is None:
            raise SystemExit("readelf is required for Stable ABI dependency auditing")
        command = [tool, "-d", str(binary)]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    if os.name != "nt":
        # RUNPATH commonly contains the Python installation directory used by
        # the build. Only DT_NEEDED entries are direct binary dependencies.
        output = "\n".join(line for line in output.splitlines() if "(NEEDED)" in line)
    return output


def dynamic_symbol_listing(binary: Path) -> str:
    if os.name == "nt":
        tool = shutil.which("dumpbin")
        if tool is None:
            program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            candidates = sorted(
                program_files.glob("Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"),
                reverse=True,
            )
            tool = str(candidates[0]) if candidates else None
        if tool is None:
            raise SystemExit("dumpbin is required for Stable ABI symbol auditing")
        command = [tool, "/imports", str(binary)]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout

    tool = shutil.which("readelf")
    if tool is None:
        raise SystemExit("readelf is required for Stable ABI symbol auditing")
    command = [tool, "--dyn-syms", "--wide", "--demangle", str(binary)]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return "\n".join(line for line in output.splitlines() if " UND " in line)


def verify_binary(binary: Path) -> None:
    if not binary.is_file():
        raise SystemExit(f"Stable ABI library does not exist: {binary}")
    listing = dependency_listing(binary).lower()
    violations = [name for name in FORBIDDEN_DEPENDENCIES if name in listing]
    if violations:
        raise SystemExit(f"{binary} has forbidden direct dependencies: {', '.join(violations)}")
    symbols = dynamic_symbol_listing(binary).lower()
    symbol_violations = [name for name in FORBIDDEN_DYNAMIC_SYMBOLS if name in symbols]
    if symbol_violations:
        raise SystemExit(f"{binary} imports unstable LibTorch symbols: {', '.join(symbol_violations)}")
    if any(tag in binary.name.lower() for tag in ("cp310", "cp311", "cp312", "cp313", "cp314")):
        raise SystemExit(f"Stable ABI library must not carry a CPython ABI tag: {binary.name}")
    print(f"Verified Stable ABI dependencies for {binary}")


def verify_input(path: Path) -> None:
    if path.suffix != ".whl":
        verify_binary(path)
        return

    with tempfile.TemporaryDirectory(prefix="rayd_stable_abi_verify_") as temp_dir:
        extract_root = Path(temp_dir)
        with zipfile.ZipFile(path) as wheel:
            members = [
                name
                for name in wheel.namelist()
                if Path(name).name.startswith("_stable_ops") and Path(name).suffix in {".dll", ".so", ".dylib"}
            ]
            if len(members) != 1:
                raise SystemExit(f"{path} must contain exactly one Stable ABI library; found {members}")
            wheel.extract(members[0], extract_root)
        verify_binary(extract_root / members[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RayD's LibTorch Stable ABI boundary")
    parser.add_argument("--source-root", type=Path, default=WORKSPACE_ROOT / "src")
    parser.add_argument("binary", nargs="*", type=Path)
    args = parser.parse_args()
    verify_sources(args.source_root)
    for binary in args.binary:
        verify_input(binary)


if __name__ == "__main__":
    main()
