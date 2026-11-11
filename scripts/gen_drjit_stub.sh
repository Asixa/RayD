#!/usr/bin/env bash
# Copyright Xingyu Chen.
# Regenerates the committed Dr.Jit extension stub from the built rayd.drjit._C module.

set -euo pipefail

usage() {
    cat <<'USAGE'
usage: scripts/gen_drjit_stub.sh [--python PYTHON] [--format-python PYTHON]

  --python PYTHON         interpreter of the environment that holds the built rayd-drjit (default: python)
  --format-python PYTHON  interpreter that runs scripts/format_code.py with the pinned ruff and clang-format on
                          PATH (default: the --python interpreter)

Rebuild the Dr.Jit backend first (scripts/build_local.sh drjit) so the stub describes the current bindings.
Importing the extension needs a CUDA device; select one with CUDA_VISIBLE_DEVICES.
USAGE
}

python_bin="python"
format_python=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_bin="$2"; shift 2 ;;
        --format-python) format_python="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
format_python="${format_python:-$python_bin}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stub="$root/python/rayd/drjit/_C/__init__.pyi"
patterns="$root/scripts/drjit_stub_patterns.txt"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Run from an empty directory: the checkout root holds torch/ and drjit/ project folders that would shadow the
# installed framework packages on sys.path.
(cd "$work" && "$python_bin" -m nanobind.stubgen -q -m rayd.drjit._C -p "$patterns" -o "$work/stub.pyi")

{
    printf '# Copyright Xingyu Chen.\n# Exposes the Dr.Jit C Python API.\n\n'
    cat "$work/stub.pyi"
} > "$stub"

(cd "$root" && "$format_python" scripts/format_code.py)
echo "wrote $stub"
