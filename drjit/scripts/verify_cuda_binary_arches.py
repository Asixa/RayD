# Copyright Xingyu Chen.
# Checks Dr.Jit metadata for verify cuda binary arches.

from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path


EXPECTED_SASS = ("70", "75", "80", "86", "87", "89", "90", "100", "101", "120")
EXPECTED_PTX_TARGET = "sm_120"


def _matches(path: Path, stems: tuple[str, ...]) -> bool:
    return path.suffix in {".dll", ".pyd", ".so"} and any(path.name.startswith(stem) for stem in stems)


def _collect_binaries(inputs: list[Path], stems: tuple[str, ...], extract_root: Path) -> list[Path]:
    binaries: list[Path] = []
    for input_path in inputs:
        if input_path.suffix == ".whl":
            with zipfile.ZipFile(input_path) as wheel:
                for name in wheel.namelist():
                    member = Path(name)
                    if not _matches(member, stems):
                        continue
                    wheel.extract(name, extract_root)
                    binaries.append(extract_root / member)
            continue
        if input_path.is_dir():
            binaries.extend(path for path in input_path.rglob("*") if _matches(path, stems))
            continue
        if _matches(input_path, stems):
            binaries.append(input_path)
    return sorted(set(path.resolve() for path in binaries))


def _cuobjdump(flag: str, binary: Path) -> str:
    result = subprocess.run(
        ["cuobjdump", flag, str(binary)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return f"{result.stdout}\n{result.stderr}"

    errors = [
        f"cuobjdump {flag} failed for {binary} with exit code {result.returncode}:",
        result.stderr.strip() or result.stdout.strip() or "<no output>",
    ]
    if binary.suffix == ".so":
        with tempfile.TemporaryDirectory(prefix="rayd_cuda_fatbin_") as temp_dir:
            fatbin = Path(temp_dir) / f"{binary.stem}.fatbin"
            extraction = subprocess.run(
                ["objcopy", "--dump-section", f".nv_fatbin={fatbin}", str(binary)],
                capture_output=True,
                text=True,
            )
            if extraction.returncode == 0 and fatbin.is_file() and fatbin.stat().st_size:
                retry = subprocess.run(
                    ["cuobjdump", flag, str(fatbin)],
                    capture_output=True,
                    text=True,
                )
                if retry.returncode == 0:
                    return f"{retry.stdout}\n{retry.stderr}"
                errors.extend(
                    [
                        f"cuobjdump {flag} failed for extracted {fatbin.name} "
                        f"with exit code {retry.returncode}:",
                        retry.stderr.strip() or retry.stdout.strip() or "<no output>",
                    ]
                )
            else:
                errors.extend(
                    [
                        f"Could not extract .nv_fatbin from {binary}:",
                        extraction.stderr.strip()
                        or extraction.stdout.strip()
                        or "<no output>",
                    ]
                )
    raise SystemExit("\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA SASS and PTX targets in RayD release binaries.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--stem", action="append", required=True)
    parser.add_argument(
        "--expected-sass",
        default=",".join(EXPECTED_SASS),
        help="Comma-separated native SASS targets. Defaults to the release matrix.",
    )
    parser.add_argument(
        "--expected-ptx",
        default=EXPECTED_PTX_TARGET.removeprefix("sm_"),
        help="PTX target without the sm_ prefix. Defaults to the release matrix.",
    )
    args = parser.parse_args()
    expected_sass = tuple(arch.strip() for arch in args.expected_sass.split(",") if arch.strip())
    if not expected_sass:
        raise SystemExit("--expected-sass must contain at least one architecture.")
    expected_ptx_target = f"sm_{args.expected_ptx.strip().removeprefix('sm_')}"

    with tempfile.TemporaryDirectory(prefix="rayd_cuda_arch_verify_") as temp_dir:
        binaries = _collect_binaries(args.inputs, tuple(args.stem), Path(temp_dir))
        if not binaries:
            raise SystemExit(f"No native binaries matching {args.stem!r} were found.")

        for binary in binaries:
            elf_listing = _cuobjdump("--list-elf", binary)
            missing_sass = [arch for arch in expected_sass if f"sm_{arch}" not in elf_listing]
            if missing_sass:
                raise SystemExit(f"{binary} is missing SASS targets: {', '.join(missing_sass)}")

            ptx_dump = _cuobjdump("--dump-ptx", binary)
            if f".target {expected_ptx_target}" not in ptx_dump:
                raise SystemExit(f"{binary} is missing PTX target {expected_ptx_target}.")

            print(f"Verified CUDA architectures in {binary}")


if __name__ == "__main__":
    main()
