# Copyright Xingyu Chen.
# Tests source file ownership and opening header standards.

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_SUFFIXES = {".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh", ".cu"}
GENERATED_ROOT = Path("generated")
NATIVE_SUFFIXES = MAINTAINED_SUFFIXES - {".py", ".pyi"}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        path = ROOT / relative
        if path.is_file() and path.suffix.lower() in MAINTAINED_SUFFIXES:
            paths.append(path)
    return paths


class SourceFileStandardsTests(unittest.TestCase):
    def test_maintained_sources_have_concise_ownership_headers(self):
        offenders: list[str] = []
        for path in tracked_files():
            relative = path.relative_to(ROOT)
            if relative.parts and relative.parts[0] == GENERATED_ROOT.name:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            prefix = "#" if path.suffix.lower() in {".py", ".pyi"} else "//"
            expected_owner = f"{prefix} Copyright Xingyu Chen."
            if len(lines) < 3 or lines[0] != expected_owner:
                offenders.append(f"{relative.as_posix()}: missing copyright header")
                continue
            description = lines[1]
            if not description.startswith(f"{prefix} ") or not description.endswith("."):
                offenders.append(f"{relative.as_posix()}: malformed responsibility sentence")
            elif len(description) > 123:
                offenders.append(f"{relative.as_posix()}: responsibility sentence is too long")
            elif re.search(r"\bADR(?:[- _]?\d+)?\b", description, re.IGNORECASE):
                offenders.append(f"{relative.as_posix()}: responsibility sentence cites an ADR")
            if lines[2] != "":
                offenders.append(f"{relative.as_posix()}: header must be followed by one blank line")
        self.assertEqual(offenders, [])

    def test_math_has_one_production_file_owner(self):
        math_named = []
        for root in (ROOT / "include", ROOT / "src"):
            for path in root.rglob("*"):
                if path.is_file() and "math" in path.name.lower():
                    math_named.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(math_named, ["include/rayd/math.h"])
        self.assertFalse((ROOT / "include" / "rayd" / "detail").exists())

    def test_simple_math_types_are_declared_only_in_math_h(self):
        declaration = re.compile(
            r"\b(?:struct|class)\s+"
            r"(?:Vec3|Vec3f|Complex|Complex3|Mat3|Quat|Quaternion|V3|C2|"
            r"BvhFloat3|PathVec3f|PathComplex3f|LegacySlabComplex)\b"
        )
        offenders = []
        for root in (ROOT / "include", ROOT / "src"):
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in NATIVE_SUFFIXES:
                    continue
                if path == ROOT / "include" / "rayd" / "math.h":
                    continue
                if declaration.search(path.read_text(encoding="utf-8")):
                    offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_retired_math_headers_are_not_referenced(self):
        retired = (
            "rayd/detail/",
            "field_math.h",
            "vec3.h",
            "edge_distance_math.h",
            "utd_math.h",
            "device_math.cuh",
            "src/runtime/math.cuh",
        )
        offenders = []
        for path in tracked_files():
            relative = path.relative_to(ROOT)
            if relative.parts and relative.parts[0] == GENERATED_ROOT.name:
                continue
            if relative.as_posix() == "tests/test_source_file_standards.py":
                continue
            text = path.read_text(encoding="utf-8")
            for token in retired:
                if token in text:
                    offenders.append(f"{relative.as_posix()}: {token}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()