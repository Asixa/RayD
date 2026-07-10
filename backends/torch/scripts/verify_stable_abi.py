from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


FORBIDDEN_SOURCE_MARKERS = (
    "at::",
    "c10::",
    "py::",
    "torch/extension.h",
    "torch/library.h",
)
FORBIDDEN_DEPENDENCIES = (
    "torch_python",
    "c10.dll",
    "c10_cuda.dll",
    "libc10.so",
    "libc10_cuda.so",
    "python3",
)


def verify_sources(source_root: Path) -> None:
    sources = sorted(path for path in source_root.rglob("*") if path.suffix in {".h", ".hpp", ".cuh", ".cc", ".cpp", ".cu"})
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
                program_files.glob(
                    "Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"
                ),
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


def verify_binary(binary: Path) -> None:
    if not binary.is_file():
        raise SystemExit(f"Stable ABI library does not exist: {binary}")
    listing = dependency_listing(binary).lower()
    violations = [name for name in FORBIDDEN_DEPENDENCIES if name in listing]
    if violations:
        raise SystemExit(f"{binary} has forbidden direct dependencies: {', '.join(violations)}")
    if any(tag in binary.name.lower() for tag in ("cp310", "cp311", "cp312", "cp313", "cp314")):
        raise SystemExit(f"Stable ABI library must not carry a CPython ABI tag: {binary.name}")
    print(f"Verified Stable ABI dependencies for {binary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RayD's LibTorch Stable ABI boundary")
    parser.add_argument("--source-root", type=Path, default=Path("src/stable"))
    parser.add_argument("binary", nargs="?", type=Path)
    args = parser.parse_args()
    verify_sources(args.source_root)
    if args.binary is not None:
        verify_binary(args.binary)


if __name__ == "__main__":
    main()
