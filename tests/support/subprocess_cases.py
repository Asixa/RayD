# Copyright Xingyu Chen.
# Runs isolated Python test cases and decodes their JSON results.

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]


def run_script(
    script: str,
    timeout: int = 300,
    check: bool = True,
    cwd: str | os.PathLike[str] = ROOT,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Subprocess failed.\nReturn code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def decode_json_result(result: subprocess.CompletedProcess[str]):
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON output.\nSTDERR:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"Failed to parse JSON from subprocess.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from error


def run_json_case(
    script: str, timeout: int = 300, cwd: str | os.PathLike[str] = ROOT, env: Mapping[str, str] | None = None
):
    return decode_json_result(run_script(script, timeout=timeout, check=True, cwd=cwd, env=env))


def compose(*parts: str) -> str:
    return "\n".join(textwrap.dedent(part).strip("\n") for part in parts)
