# Copyright Xingyu Chen.
# Validates that a GitHub Release tag exactly matches the project version.

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def expected_release_tag(pyproject: Path) -> str:
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = metadata.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"{pyproject} does not declare a non-empty project.version")
    return f"v{version}"


def validate_release_tag(release_tag: str, pyproject: Path) -> str:
    expected = expected_release_tag(pyproject)
    if release_tag != expected:
        raise RuntimeError(f"release tag {release_tag!r} does not match project version tag {expected!r}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="GitHub Release tag name")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    arguments = parser.parse_args()
    validated = validate_release_tag(arguments.tag, arguments.pyproject)
    print(f"Validated release tag {validated}")


if __name__ == "__main__":
    main()
