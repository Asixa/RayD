# Copyright Xingyu Chen.
# Extracts small structural regions from source text for governance tests.

from __future__ import annotations

import re
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    start = text.index(name)
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise AssertionError(f"unterminated function {name}")


def struct_body(text: str, name: str) -> str:
    match = re.search(rf"struct {name}\s*\{{(?P<body>.*?)\n\}};", text, re.S)
    if match is None:
        raise AssertionError(f"missing struct {name}")
    return match.group("body")
