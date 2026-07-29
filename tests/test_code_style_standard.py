# Copyright Xingyu Chen.
# Tests formatting, duplication, and development-standard governance.

from __future__ import annotations

from collections import defaultdict
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh", ".cu"}
INTENTIONAL_EXACT_DUPLICATES = {("python/rayd/_impl/path_exchange.py", "python/rayd/_impl/path_exchange_jit.py")}


def tracked_sources() -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    paths: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        path = ROOT / relative
        if path.is_file() and relative.parts[0] != "generated" and relative.suffix.lower() in SOURCE_SUFFIXES:
            paths.append(relative)
    return paths


class CodeStyleStandardTests(unittest.TestCase):
    def test_native_and_python_formatting_contracts_are_pinned(self):
        clang = (ROOT / ".clang-format").read_text(encoding="utf-8")
        for setting in (
            "ColumnLimit: 120",
            "IndentWidth: 4",
            "BinPackArguments: true",
            "BinPackParameters: true",
            "AllowAllParametersOfDeclarationOnNextLine: true",
            "SortIncludes: Never",
        ):
            self.assertIn(setting, clang)

        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.ruff]", project)
        self.assertIn("line-length = 120", project)
        self.assertIn('line-ending = "lf"', project)
        self.assertIn("skip-magic-trailing-comma = true", project)

        editor = (ROOT / ".editorconfig").read_text(encoding="utf-8")
        self.assertIn("end_of_line = lf", editor)
        self.assertIn("max_line_length = 120", editor)

    def test_project_memory_points_to_the_complete_standard(self):
        agents = (ROOT / "AGENTS.md").read_bytes()
        self.assertEqual(agents, (ROOT / "CLAUDE.md").read_bytes())
        text = agents.decode("utf-8")
        self.assertIn("## Code Formatting and Duplication Control", text)
        self.assertIn("docs/dev/coding_standard.md", text)
        self.assertTrue((ROOT / "docs" / "dev" / "coding_standard.md").is_file())
        self.assertTrue((ROOT / "scripts" / "format_code.py").is_file())

    def test_exact_whole_file_duplication_is_a_closed_set(self):
        bodies: dict[str, list[str]] = defaultdict(list)
        for relative in tracked_sources():
            lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
            body = re.sub(r"\s+", " ", "\n".join(lines[3:])).strip()
            if len(body) >= 120:
                bodies[body].append(relative.as_posix())
        duplicates = {tuple(sorted(paths)) for paths in bodies.values() if len(paths) > 1}
        self.assertEqual(duplicates, INTENTIONAL_EXACT_DUPLICATES)


if __name__ == "__main__":
    unittest.main()
