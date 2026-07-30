# Copyright Xingyu Chen.
# Verifies that the Torch legacy library does not directly depend on the CUDA driver.

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
import zipfile

from verify_stable_abi import dependency_listing, dynamic_symbol_listing


FORBIDDEN_DRIVER_DEPENDENCIES = ("libcuda.so", "nvcuda.dll")
UNRESOLVED_OPTIX_SYMBOL = re.compile(r"\boptix[A-Z][A-Za-z0-9_]*\b")


def verify_binary(binary: Path) -> None:
    listing = dependency_listing(binary).lower()
    violations = [name for name in FORBIDDEN_DRIVER_DEPENDENCIES if name in listing]
    if violations:
        raise SystemExit(f"{binary} has direct CUDA driver dependencies: {', '.join(violations)}")
    if os.name != "nt":
        symbols = dynamic_symbol_listing(binary)
        unresolved_optix = sorted(
            {
                symbol
                for line in symbols.splitlines()
                if " UND " in line
                for symbol in UNRESOLVED_OPTIX_SYMBOL.findall(line)
            }
        )
        if unresolved_optix:
            raise SystemExit(f"{binary} has unresolved OptiX symbols: {', '.join(unresolved_optix)}")
    print(f"Verified CUDA driver independence for {binary}")


def verify_input(path: Path) -> None:
    if path.suffix != ".whl":
        verify_binary(path)
        return

    with tempfile.TemporaryDirectory(prefix="rayd_driver_independence_") as temp_dir:
        extract_root = Path(temp_dir)
        with zipfile.ZipFile(path) as wheel:
            members = [
                name
                for name in wheel.namelist()
                if Path(name).name.startswith("_legacy_ops") and Path(name).suffix in {".dll", ".so", ".dylib"}
            ]
            if len(members) != 1:
                raise SystemExit(f"{path} must contain exactly one legacy library; found {members}")
            wheel.extract(members[0], extract_root)
        verify_binary(extract_root / members[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RayD's CUDA driver dependency boundary")
    parser.add_argument("binary", nargs="+", type=Path)
    args = parser.parse_args()
    for binary in args.binary:
        verify_input(binary)


if __name__ == "__main__":
    main()
