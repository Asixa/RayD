"""Audit the source identity of the committed Dr.Jit OptiX PTX headers.

The Dr.Jit backend is the only backend that commits generated PTX: the eight
``*_ptx.h`` headers under ``generated/drjit/ptx`` are checked in so a wheel
build needs no OptiX SDK. Regeneration is opt-in
(``RAYD_REGENERATE_*_PTX``, all OFF by default), so editing a ``.cu`` file or any
header it reaches silently leaves the committed PTX describing older device code.

This script records, for every PTX module, the exact set of in-repository source
files that reach its ``.cu`` through ``#include`` plus a digest over their
contents, and re-checks that record with ``--check``. A drifted digest means the
committed PTX no longer corresponds to the sources in the tree.

What the record is NOT
----------------------
It is a *source identity* record, not a correctness certificate. The digests were
captured from the tree as it stood at adoption; nobody has proven that the
committed PTX is a byte-exact compile of those sources (see ``ADOPTION`` below).
``--check`` answers "did the inputs change since the record was written", never
"is the committed PTX correct".

Regenerating never rewrites a committed header
----------------------------------------------
``rayd_embed_ptx()`` writes regenerated headers to
``${CMAKE_CURRENT_BINARY_DIR}/generated/drjit/ptx`` and that directory is
prepended to the include path, so a fresh header only shadows the committed
one. Refreshing the committed PTX is a four-step manual operation:

1. configure with ``-DRAYD_REGENERATE_<MODULE>_PTX=ON`` and an OptiX SDK present,
2. copy ``<build>/generated/drjit/ptx/<header>`` over
   ``generated/drjit/ptx/<header>``,
3. re-run ``python drjit/scripts/audit_ptx_sources.py --write``,
4. after byte-comparing the regenerated header against the committed one,
   attest it: ``... --mark-verified <module>``. The flag lives per module and
   is cleared automatically the moment that module's digests drift again.

Include-closure scanning
------------------------
Includes are collected with a regex over the raw text, ignoring ``#if``/``#ifdef``
conditionals. The closure is therefore a superset of the true preprocessed set.
That is deliberate: the over-approximation can only produce a false "stale"
report, never miss a real one, and it keeps this audit free of CUDA, OptiX and a
GPU. Headers that do not resolve inside this repository (CUDA toolkit, OptiX SDK,
Dr.Jit, libstdc++) are recorded by name in ``external_includes`` and never
hashed, so the digest stays machine-independent while a *new* external include
still shows up as drift. The Dr.Jit pin is recorded separately.

Usage::

    python drjit/scripts/audit_ptx_sources.py --check
    python drjit/scripts/audit_ptx_sources.py --write
    python drjit/scripts/audit_ptx_sources.py --git-drift
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "drjit"
DEFAULT_OUTPUT = BACKEND / "ptx_sources.json"
CMAKELISTS = BACKEND / "CMakeLists.txt"
PYPROJECT = BACKEND / "pyproject.toml"


def drjit_pin() -> str:
    """The drjit pin, read from pyproject.toml so it can never go stale here.

    The Dr.Jit headers are genuine PTX inputs that are recorded by name only
    (external_includes) and never hashed. The pin is the record's proxy for
    their content, so it must be the REAL pin: parsing it from pyproject.toml
    means a version bump changes render(), --check fails, and the bumper is
    forced into a conscious --write plus re-verification.
    """
    pins = set(re.findall(r'"(drjit==[^"]+)"', PYPROJECT.read_text(encoding="utf-8")))
    if len(pins) != 1:
        raise SystemExit(
            f"Expected exactly one distinct drjit pin in {PYPROJECT}, found: "
            f"{sorted(pins) or 'none'}")
    return pins.pop()

# The nvcc command line rayd_embed_ptx() uses for every PTX blob. Recorded so an
# architecture or flag change invalidates the record along with the sources.
NVCC_PTX_FLAGS = "-ptx --use_fast_math -std=c++17 -arch=compute_70"

# CMake variables used inside the rayd_embed_ptx() call sites, resolved to
# repository-relative paths. Mirrors drjit/CMakeLists.txt.
CMAKE_VARIABLES = {
    "CMAKE_CURRENT_SOURCE_DIR": "drjit",
    "RAYD_SOURCE_DIR": "src",
    "RAYD_INCLUDE_DIR": "include/rayd",
    "RAYD_SHARED_INCLUDE_DIR": "include",
}

# Directories searched for an #include, after the including file's own directory.
# This is RAYD_CUDA_INCLUDE_DIRS minus the binary dir, the CUDA toolkit dirs and
# the Dr.Jit dirs, i.e. exactly the in-repository part of the PTX include path.
INCLUDE_DIRS = (
    ROOT / "include",
    ROOT / "generated" / "drjit" / "ptx",
    ROOT / "src" / "scene",
    ROOT / "src" / "runtime",
    ROOT,
)

INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^">]+)[">]', re.MULTILINE)
EMBED_CALL_RE = re.compile(r"^rayd_embed_ptx\(\n(.*?)^\)$", re.MULTILINE | re.DOTALL)
TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')
SINGLE_VALUE_KEYWORDS = ("NAME", "SOURCE", "HEADER", "OPTION", "OUT_SOURCES")

# Recorded once, at adoption. These are historical facts about the tree the
# digests were taken from; they are not recomputed, so --check stays a pure
# string comparison that needs no git, no network and no toolchain.
ADOPTION = {
    "baseline_commit": "4f0e953c28aedf6ae3ffd519169bd59f3bc2c155",
    "worktree_clean": False,
    "worktree_note": (
        "The adoption worktree carried unrelated uncommitted edits (CI workflows, "
        "Torch python package, root tests). No file in any PTX include closure was "
        "modified relative to the baseline commit, so the recorded digests describe "
        "committed content."
    ),
    "regeneration_note": (
        "At adoption, no committed *_ptx.h had been proven byte-identical to a "
        "fresh compile of its recorded sources (local regeneration fails on an "
        "nvcc 12.9 / Windows SDK 10.0.26100 ucrt-intrinsics conflict). The LIVING "
        "verification state is per module: modules.<name>.regeneration_verified, "
        "flipped only by --mark-verified and cleared automatically when that "
        "module's digests change. Treat digests as 'inputs as of the last "
        "--write', never as evidence that the committed PTX is current."
    ),
    "toolchain_observed_in_headers": "cuda 12.9 V12.9.41, ptx isa 8.8, target sm_70",
    # Per module, in-repository closure files whose last commit is newer than the
    # last commit that touched the module's committed PTX header. Each entry is
    # positive evidence that the header may already be stale; an empty list means
    # only that git history shows no such drift, not that the header is current.
    # Paths below use their current canonical spelling; short commits retain the
    # original pre-migration evidence. Reproduce current drift with --git-drift.
    "sources_committed_after_header": {
        "diffraction_accumulation": [
            "src/diffraction/accumulation_params_jit.h @2634aa1",
            "include/rayd/shared/diffraction/accumulation_algo.h @2634aa1",
            "include/rayd/shared/diffraction/utd_math.h @346416f",
            "include/rayd/shared/diffraction/utd_types.h @346416f",
        ],
        "diffraction_paths": [
            "src/diffraction/paths_params_jit.h @2634aa1",
            "include/rayd/shared/diffraction/paths_algo.h @cf51e4c",
            "include/rayd/shared/diffraction/utd_math.h @346416f",
            "include/rayd/shared/diffraction/utd_types.h @346416f",
        ],
        "edge_optix": [
            "include/rayd/shared/edge/edge_distance_math.h @3cf3fb1",
            "include/rayd/shared/math/vec3.h @a139d93",
            "include/rayd/shared/rt/numeric_policy.h @3cf3fb1",
            "include/rayd/shared/rt/qualifiers.h @a139d93",
        ],
        "reflection_accumulation": [
            "src/reflection/accumulation_params_jit.h @2634aa1",
            "include/rayd/shared/reflection/accumulation_algo.h @2634aa1",
        ],
        "reflection_epc": [
            "include/rayd/shared/reflection/epc_algo.h @2634aa1",
        ],
        "reflection_trace": [
            "include/rayd/shared/reflection/trace_algo.h @2634aa1",
        ],
        "segment_visibility": [
            "include/rayd/shared/visibility/segment_algo.h @2634aa1",
        ],
        "surfel_trace": [],
    },
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _substitute(value: str) -> Path:
    """Resolve a CMake path argument to an absolute path under ROOT."""
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in CMAKE_VARIABLES:
            raise SystemExit(f"Unknown CMake variable '${{{name}}}' in: {value}")
        return CMAKE_VARIABLES[name]

    resolved = re.sub(r"\$\{([A-Za-z0-9_]+)\}", replace, value)
    if "${" in resolved:
        raise SystemExit(f"Unresolved CMake variable in: {value}")
    return (ROOT / resolved).resolve()


def parse_embed_calls() -> list[dict[str, object]]:
    """Read the rayd_embed_ptx() call sites out of the backend CMakeLists."""
    text = CMAKELISTS.read_text(encoding="utf-8")
    calls = []
    for body in EMBED_CALL_RE.findall(text):
        tokens = [quoted or bare for quoted, bare in TOKEN_RE.findall(body)]
        call: dict[str, object] = {"depends": []}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in SINGLE_VALUE_KEYWORDS:
                call[token.lower()] = tokens[index + 1]
                index += 2
            elif token == "DEPENDS":
                call["depends"] = tokens[index + 1:]
                index = len(tokens)
            else:
                raise SystemExit(f"Unexpected token '{token}' in rayd_embed_ptx call")
        calls.append(call)
    if not calls:
        raise SystemExit(f"No rayd_embed_ptx() calls found in {CMAKELISTS}")
    return calls


def _resolve_include(name: str, including: Path) -> Path | None:
    for directory in (including.parent, *INCLUDE_DIRS):
        candidate = (directory / name).resolve()
        if candidate.is_file():
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                return None
            return candidate
    return None


def include_closure(source: Path) -> tuple[list[Path], list[str]]:
    """Return the in-repo transitive include closure of `source` plus externals."""
    seen: set[Path] = set()
    external: set[str] = set()
    pending = [source]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        text = current.read_text(encoding="utf-8", errors="strict")
        for name in INCLUDE_RE.findall(text):
            resolved = _resolve_include(name, current)
            if resolved is None:
                external.add(name)
            elif resolved not in seen:
                pending.append(resolved)
    # Sort by the repository-relative posix string, not by Path: Path ordering is
    # case- and separator-normalized on Windows, which would make the digest
    # depend on the host platform.
    return sorted(seen, key=_relative), sorted(external)


def _digest(paths: list[Path]) -> str:
    """Content digest over an ordered file set, insensitive to line endings."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_relative(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def module_record(call: dict[str, object]) -> dict[str, object]:
    source = _substitute(str(call["source"]))
    header = (
        ROOT / "generated" / "drjit" / "ptx" / str(call["header"])
    ).resolve()
    if not source.is_file():
        raise SystemExit(f"Missing PTX source: {source}")
    if not header.is_file():
        raise SystemExit(f"Missing committed PTX header: {header}")
    closure, external = include_closure(source)
    return {
        "cu": _relative(source),
        "header": _relative(header),
        "option": call["option"],
        "variable": f"{call['name']}_ptx",
        "header_sha256": _file_sha256(header),
        "source_sha256": _digest(closure),
        "sources": [_relative(path) for path in closure],
        "external_includes": external,
    }


def depends_drift() -> dict[str, dict[str, list[str]]]:
    """Compare each rayd_embed_ptx() DEPENDS list against the include closure.

    The DEPENDS lists are the build graph's freshness inputs and the closure is
    the digest's input set; they must be the same set or one of the two is lying.
    `SOURCE` is passed to add_custom_command separately, so it is excluded here.
    """
    drift = {}
    for call in parse_embed_calls():
        record = module_record(call)
        closure = set(record["sources"]) - {record["cu"]}
        declared = {_relative(_substitute(item)) for item in call["depends"]}
        if closure != declared:
            drift[str(call["name"])] = {
                "missing": sorted(closure - declared),
                "extra": sorted(declared - closure),
            }
    return drift


def audit(existing: dict | None = None) -> dict[str, object]:
    """Build the record, carrying per-module verification forward from `existing`.

    ``regeneration_verified`` is human-attested state (--mark-verified), not
    derivable from the tree, so --write preserves it -- but only while the
    module's digests and the drjit pin are unchanged. Any input drift clears the
    flag: a verification claim must never outlive the inputs it was made for.
    """
    pin = drjit_pin()
    prior = (existing or {}).get("modules", {})
    pin_unchanged = (existing or {}).get("drjit_pin") == pin
    modules = {}
    for call in parse_embed_calls():
        name = str(call["name"])
        record = module_record(call)
        before = prior.get(name, {})
        record["regeneration_verified"] = bool(
            pin_unchanged
            and before.get("regeneration_verified")
            and before.get("source_sha256") == record["source_sha256"]
            and before.get("header_sha256") == record["header_sha256"])
        modules[name] = record
    return {
        "version": 1,
        "backend": "drjit",
        "nvcc_ptx_flags": NVCC_PTX_FLAGS,
        "drjit_pin": pin,
        "adoption": ADOPTION,
        "modules": modules,
    }


def _load_existing(output: Path) -> dict | None:
    if not output.is_file():
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def render(existing: dict | None = None) -> str:
    return json.dumps(audit(existing), indent=2, sort_keys=True) + "\n"


def _last_commit(relative: str) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H %cI", "--", relative],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def git_drift() -> None:
    """Report closure files committed after their module's PTX header.

    Positive evidence that a committed header may already be stale. Used to fill
    ADOPTION["sources_committed_after_header"] and to decide, per module, whether
    regeneration_verified can ever be flipped without a rebuild.
    """
    for name, record in audit()["modules"].items():
        header_commit = _last_commit(str(record["header"]))
        header_time = header_commit.split(" ", 1)[1] if header_commit else ""
        newer = []
        for relative in record["sources"]:
            entry = _last_commit(relative)
            if not entry or not header_time:
                continue
            # Compare as datetimes: %cI carries the committer's UTC offset, and
            # lexical comparison misorders timestamps across differing offsets.
            if datetime.fromisoformat(entry.split(" ", 1)[1]) \
                    > datetime.fromisoformat(header_time):
                newer.append(f"{relative} @{entry.split(' ', 1)[0][:7]}")
        print(f"{name}: header @{header_commit[:7]} {header_time}")
        for line in newer:
            print(f"    newer: {line}")
        if not newer:
            print("    no source drift since the header commit")


def mark_verified(output: Path, name: str) -> None:
    """Attest that `name`'s committed header was regenerated and byte-compared.

    Only valid against a current record: attesting on top of a stale record
    would bind the claim to inputs that no longer exist in the tree.
    """
    existing = _load_existing(output)
    if existing is None:
        raise SystemExit(f"Missing record {output}; run --write first")
    if render(existing) != output.read_text(encoding="utf-8"):
        raise SystemExit(
            f"{output} is stale; run --check to see why, then --write, "
            "then re-verify and --mark-verified again")
    if name not in existing["modules"]:
        raise SystemExit(
            f"Unknown module '{name}'; known: {', '.join(sorted(existing['modules']))}")
    existing["modules"][name]["regeneration_verified"] = True
    output.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"Marked {name} regeneration_verified=true in {output}")


def check(output: Path) -> None:
    expected = render(_load_existing(output))
    if not output.is_file():
        raise SystemExit(
            f"Missing PTX source-identity record: {output}\n"
            "Run: python drjit/scripts/audit_ptx_sources.py --write"
        )
    drift = depends_drift()
    if drift:
        lines = [f"rayd_embed_ptx() DEPENDS no longer matches the include closure "
                 f"in {_relative(CMAKELISTS)}:"]
        for name, delta in drift.items():
            for item in delta["missing"]:
                lines.append(f"  {name}: add    {item}")
            for item in delta["extra"]:
                lines.append(f"  {name}: remove {item}")
        raise SystemExit("\n".join(lines))
    actual = output.read_text(encoding="utf-8")
    if actual == expected:
        print(f"PTX source identity is current: {output}")
        return
    recorded = json.loads(actual).get("modules", {})
    current = json.loads(expected)["modules"]
    drifted = [
        name for name, record in current.items()
        if recorded.get(name, {}).get("source_sha256") != record["source_sha256"]
        or recorded.get(name, {}).get("header_sha256") != record["header_sha256"]
    ]
    lines = [f"PTX source identity is stale: {output}"]
    for name in drifted:
        lines.append(
            f"  {name}: regenerate with -D{current[name]['option']}=ON, copy the "
            f"build-tree header over {current[name]['header']}"
        )
    if not drifted:
        lines.append("  record metadata differs; rerun with --write")
    lines.append("Then: python drjit/scripts/audit_ptx_sources.py --write")
    raise SystemExit("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit committed Dr.Jit PTX source identity")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="fail if the record no longer matches the sources")
    mode.add_argument("--write", action="store_true",
                      help="rewrite the record from the current sources")
    mode.add_argument("--git-drift", action="store_true",
                      help="list closure files committed after their PTX header")
    mode.add_argument("--mark-verified", metavar="MODULE",
                      help="attest MODULE's committed header was regenerated and "
                           "byte-compared against its recorded sources")
    args = parser.parse_args()
    if args.git_drift:
        git_drift()
        return
    if args.check:
        check(args.output)
        return
    if args.mark_verified:
        mark_verified(args.output, args.mark_verified)
        return
    args.output.write_text(render(_load_existing(args.output)),
                           encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
