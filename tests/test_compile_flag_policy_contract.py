"""Enforcement gate for the per-translation-unit CUDA numeric flag contract.

`shared/include/rayd/shared/rt/numeric_policy.h` freezes the two backends'
epsilon and sentinel divergences. It says nothing about the *compiler* numeric
flags, and the two backends compile the same shared device-math headers under
four different nvcc numeric profiles. `shared/contracts/compile_policy.json`
declares that assignment; `docs/adr/0035-cuda-compile-flag-policy.md` records why
each divergence is frozen. This test re-derives the assignment from both
CMakeLists and fails when the declaration and the build disagree in either
direction, so neither can drift alone.

It also recomputes, from the `#include` graph on disk, which shared headers are
compiled under more than one profile. That set is the actual hazard the contract
exists to make visible, and it is derived, never asserted.

What this test deliberately does not own:

* the exact `--fmad=false` source list and the single-block invariant, owned by
  `tests/test_adr0026_scattering_ownership.py::test_phase10b_source_local_fmad_policy`;
* the penetration family's precise-flag command shape, owned by
  `tests/test_adr0033_segment_penetration.py::test_cmake_owns_precise_ptx_native_and_direct_test`;
* the committed Dr.Jit PTX source identity and its pinned nvcc flag string, owned
  by `tests/test_ptx_source_digest.py`.

Those tests check their own family from the inside. This one checks that the
whole flag matrix is declared, complete, and still divergent where the contract
says it is.
"""

import json
import re
import unittest
from pathlib import Path

from tests._schema_validate import validate


ROOT = Path(__file__).resolve().parents[1]
DRJIT = ROOT / "backends" / "drjit"
TORCH = ROOT / "backends" / "torch"
CONTRACT_PATH = ROOT / "shared" / "contracts" / "compile_policy.json"
SCHEMA_PATH = ROOT / "shared" / "contracts" / "compile_policy.schema.json"
ADR_PATH = ROOT / "docs" / "adr" / "0035-cuda-compile-flag-policy.md"
NUMERIC_POLICY = ROOT / "shared" / "include" / "rayd" / "shared" / "rt" / "numeric_policy.h"

# Only flags that change device arithmetic. Anything else (-std, include dirs,
# --extended-lambda, -Xcompiler, gencode) is out of scope by contract.
NUMERIC_RE = re.compile(r"^--?(use_fast_math|fmad=\w+|ftz=\w+|prec-div=\w+|prec-sqrt=\w+)$")
# A token that looks numeric but does not parse is kept verbatim so an unknown
# spelling produces an unknown profile and a loud failure, never a silent drop.
SUSPECT_RE = re.compile(r"fast_math|fmad|ftz|prec-div|prec-sqrt", re.IGNORECASE)
ARCH_RE = re.compile(r"^(?:-arch=|--gpu-architecture=)(\w+)$")
COMPILE_OPTION_RE = re.compile(r"^\$<\$<COMPILE_LANGUAGE:CUDA>:([^>]*)>$")
TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')
INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^">]+)[">]', re.MULTILINE)

PROFILE_BY_FLAGS = {
    (): "nvcc_default",
    ("--use_fast_math",): "fast_math",
    ("--fmad=false",): "no_fmad",
    ("--ftz=false", "--prec-div=true", "--prec-sqrt=true"): "precise_no_ftz",
}

INCLUDE_DIRS = {
    "drjit": (DRJIT / "include", ROOT / "shared" / "include", DRJIT),
    "torch": (TORCH / "include", ROOT / "shared" / "include", TORCH),
}


def strip_comments(text: str) -> str:
    return re.sub(r"#[^\n]*", "", text)


def call_bodies(text: str, opener: str):
    """Yield the body of every balanced `opener(...)` call, quote-aware.

    Callers pass comment-stripped text: a helper name written inside a comment
    would otherwise be parsed as an empty call.
    """
    for match in re.finditer(re.escape(opener) + r"\(", text):
        index, depth, in_quote = match.end(), 1, False
        start = index
        while index < len(text):
            char = text[index]
            if char == '"':
                in_quote = not in_quote
            elif not in_quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        break
            index += 1
        else:
            raise AssertionError(f"unbalanced {opener}( in CMake text")
        yield text[start:index]


def tokens(body: str) -> list[str]:
    return [quoted if quoted else bare
            for quoted, bare in TOKEN_RE.findall(strip_comments(body))]


def numeric_flags(items) -> set[str]:
    found = set()
    for token in items:
        if NUMERIC_RE.match(token):
            found.add("--" + token.lstrip("-"))
        elif SUSPECT_RE.search(token):
            found.add(token)
    return found


def profile_of(flags) -> str:
    return PROFILE_BY_FLAGS.get(tuple(sorted(flags)), "unknown:" + ",".join(sorted(flags)))


def cmake_options(text: str) -> dict[str, bool]:
    """Every `option(NAME "doc" ON|OFF)` and its default.

    Comments are stripped first: a commented-out option() must read as absent,
    not as a live default (otherwise disabling e.g. RAYD_TORCH_OPTIX_FAST_MATH
    by commenting it out would leave the contract green while ten PTX modules
    silently change profile).
    """
    return {
        name: value == "ON"
        for name, value in re.findall(
            r"option\(\s*(\w+)\s+\"[^\"]*\"\s+(ON|OFF)\s*\)",
            strip_comments(text), re.DOTALL)
    }


def resolve(raw: str, base: Path, extra: dict[str, str] | None = None) -> Path:
    substitutions = {"CMAKE_CURRENT_SOURCE_DIR": str(base)}
    substitutions.update(extra or {})

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in substitutions:
            raise AssertionError(f"unresolved CMake variable ${{{name}}} in {raw!r}")
        return substitutions[name]

    resolved = re.sub(r"\$\{(\w+)\}", replace, raw)
    path = Path(resolved)
    if not path.is_absolute():
        path = base / resolved
    return path.resolve()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------
# Dr.Jit backend: explicit nvcc command lines built by cmake/rayd_cuda.cmake.
# --------------------------------------------------------------------------
DRJIT_SINGLE = {"NAME", "SOURCE", "OUT_SOURCES", "HEADER", "OPTION"}
DRJIT_MULTI = {"EXTRA_FLAGS", "DEPENDS"}
DRJIT_FLAG = {"POSIX_NO_EXTENDED_LAMBDA"}
DRJIT_PATH_VARS = {"RAYD_SOURCE_DIR": "src", "RAYD_INCLUDE_DIR": "include/rayd"}


def _split_win32_branch(body: str) -> dict[str, str]:
    """Split a helper body into its WIN32 branch and its POSIX branch.

    Each helper emits the same nvcc invocation twice: once inside a generated
    .bat wrapper for Windows and once as a direct add_custom_command for POSIX.
    Collecting flags over the whole body would union the two, so a numeric flag
    present on only one platform would read as present on both -- exactly the
    divergence that would ship different arithmetic on Linux than on Windows.
    """
    match = re.search(r"^([ \t]*)if\(WIN32\)$", body, re.MULTILINE)
    if not match:
        # A body may legitimately have no platform split, but a present-yet-
        # unmatched split (e.g. `if (WIN32)` after a reformat) must fail loudly:
        # falling back to the whole body would silently restore the
        # union-over-branches blind spot this splitter exists to close.
        assert "WIN32" not in body, (
            "helper body mentions WIN32 but the if(WIN32)/else()/endif() "
            "anchor regex did not match; update _split_win32_branch for the "
            "new formatting instead of letting the branches merge")
        return {"win32": body, "posix": body}
    indent = match.group(1)
    rest = body[match.end():]
    else_match = re.search(rf"^{re.escape(indent)}else\(\)$", rest, re.MULTILINE)
    endif_match = re.search(rf"^{re.escape(indent)}endif\(\)$", rest, re.MULTILINE)
    assert else_match and endif_match, "if(WIN32)/else()/endif() not matched"
    # Text outside the branch applies to both platforms.
    common = body[:match.start()] + rest[endif_match.end():]
    return {
        "win32": common + rest[:else_match.start()],
        "posix": common + rest[else_match.end():endif_match.start()],
    }


def drjit_helper_flags() -> dict[str, object]:
    """Numeric flags and arch baked into the two shared nvcc command shapes.

    Reported per platform branch, so the WIN32 and POSIX invocations of the same
    helper are compared against each other rather than merged.
    """
    text = (DRJIT / "cmake" / "rayd_cuda.cmake").read_text(encoding="utf-8")
    result: dict[str, object] = {}
    for name in ("rayd_cuda_object", "rayd_embed_ptx"):
        match = re.search(
            rf"function\({name}\)(.*?)^endfunction\(\)", text, re.DOTALL | re.MULTILINE)
        assert match, f"{name}() not found in cmake/rayd_cuda.cmake"
        branches = _split_win32_branch(strip_comments(match.group(1)))
        per_branch = {}
        per_branch_arch = {}
        for platform_name, branch in branches.items():
            per_branch[platform_name] = numeric_flags(branch.split())
            per_branch_arch[platform_name] = {
                ARCH_RE.match(t).group(1) for t in branch.split() if ARCH_RE.match(t)}
        # Branch agreement is asserted by
        # test_helper_command_shapes_agree_across_platform_branches, not here:
        # this helper runs at import time, and failing here would abort
        # collection instead of producing a named, diagnosable failure. The
        # WIN32 branch is the representative answer for the other tests.
        result[name] = per_branch["win32"]
        result[name + ":arch"] = per_branch_arch["win32"]
        result[name + ":per_branch"] = per_branch
        result[name + ":per_branch_arch"] = per_branch_arch
    return result


def drjit_units() -> dict[tuple[str, str], dict]:
    raw = (DRJIT / "CMakeLists.txt").read_text(encoding="utf-8")
    text = strip_comments(raw)
    options = cmake_options(raw)
    helper = drjit_helper_flags()
    ptx_flags = helper["rayd_embed_ptx"]
    ptx_arch = helper["rayd_embed_ptx:arch"]
    assert len(ptx_arch) == 1, f"rayd_embed_ptx() declares several arches: {ptx_arch}"

    units: dict[tuple[str, str], dict] = {}
    for opener, kind in (("rayd_cuda_object", "object"), ("rayd_embed_ptx", "ptx")):
        for body in call_bodies(text, opener):
            call: dict[str, object] = {"EXTRA_FLAGS": [], "DEPENDS": []}
            items, index = tokens(body), 0
            while index < len(items):
                token = items[index]
                if token in DRJIT_SINGLE:
                    call[token] = items[index + 1]
                    index += 2
                elif token in DRJIT_MULTI:
                    values, cursor = [], index + 1
                    while cursor < len(items) and items[cursor] not in (
                            DRJIT_SINGLE | DRJIT_MULTI | DRJIT_FLAG):
                        values.append(items[cursor])
                        cursor += 1
                    call[token] = values
                    index = cursor
                elif token in DRJIT_FLAG:
                    index += 1
                else:
                    raise AssertionError(f"unexpected token {token!r} in {opener}()")
            source = resolve(str(call["SOURCE"]), DRJIT, DRJIT_PATH_VARS)
            entry = {
                "backend": "drjit",
                "unit": str(call["NAME"]),
                "source": relative(source),
                "kind": kind,
                "target": "rayd_core",
            }
            if kind == "object":
                entry["profile"] = profile_of(numeric_flags(call["EXTRA_FLAGS"]))
                entry["default_enabled"] = True
            else:
                option = str(call["OPTION"])
                entry["profile"] = profile_of(ptx_flags)
                entry["arch"] = next(iter(ptx_arch))
                entry["option"] = option
                entry["default_enabled"] = options[option]
            units[("drjit", entry["unit"])] = entry
    return units


# --------------------------------------------------------------------------
# Torch backend: CMake CUDA language plus standalone PTX custom commands.
# --------------------------------------------------------------------------
def torch_units() -> dict[tuple[str, str], dict]:
    raw = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
    text = strip_comments(raw)
    options = cmake_options(raw)

    optix_flags = None
    core_sources = None
    for body in call_bodies(text, "set"):
        items = tokens(body)
        if not items:
            continue
        if items[0] == "RAYD_TORCH_OPTIX_NVCC_FLAGS":
            optix_flags = list(items[1:])
        elif items[0] == "RAYD_TORCH_NATIVE_CORE_SOURCES":
            core_sources = list(items[1:])
    assert optix_flags is not None, "RAYD_TORCH_OPTIX_NVCC_FLAGS not found"
    assert core_sources, "RAYD_TORCH_NATIVE_CORE_SOURCES not found"
    if options["RAYD_TORCH_OPTIX_FAST_MATH"]:
        optix_flags.append("--use_fast_math")

    overrides: dict[str, list[str]] = {}
    for body in call_bodies(text, "set_source_files_properties"):
        items = tokens(body)
        split = items.index("PROPERTIES")
        assert items[split + 1] == "COMPILE_OPTIONS", (
            "only COMPILE_OPTIONS overrides are contracted")
        match = COMPILE_OPTION_RE.match(items[split + 2])
        assert match, f"unrecognized COMPILE_OPTIONS shape: {items[split + 2]!r}"
        for source in items[:split]:
            overrides[source] = match.group(1).split(";")

    units: dict[tuple[str, str], dict] = {}

    for body in call_bodies(text, "add_custom_command"):
        items = tokens(body)
        if "--ptx" not in items:
            continue
        command = items[items.index("COMMAND"):items.index("DEPENDS")]
        expanded: list[str] = []
        for token in command:
            if token == "${RAYD_TORCH_OPTIX_NVCC_FLAGS}":
                expanded.extend(optix_flags)
            else:
                expanded.append(token)
        variable = re.match(r"^\$\{(\w+)\}$", items[items.index("OUTPUT") + 1])
        assert variable, "PTX custom command output is not a plain CMake variable"
        source = resolve(command[command.index("--ptx") + 1], TORCH)
        arches = {ARCH_RE.match(t).group(1) for t in expanded if ARCH_RE.match(t)}
        assert len(arches) == 1, f"{variable.group(1)} declares arches {arches}"
        units[("torch", variable.group(1).lower())] = {
            "backend": "torch",
            "unit": variable.group(1).lower(),
            "source": relative(source),
            "kind": "ptx",
            "target": "rayd_torch_native_core",
            "profile": profile_of(numeric_flags(expanded)),
            "arch": next(iter(arches)),
            "default_enabled": True,
        }

    def add_object(raw: str, target: str) -> None:
        source = resolve(raw, TORCH)
        unit = relative(source)
        # One source compiled into two targets would carry two profiles under one
        # contract entry. Surface it instead of letting the last writer win.
        assert ("torch", unit) not in units, f"{unit} is compiled twice"
        units[("torch", unit)] = {
            "backend": "torch",
            "unit": unit,
            "source": unit,
            "kind": "object",
            "target": target,
            "profile": profile_of(numeric_flags(overrides.get(raw, []))),
            "default_enabled": True,
        }

    for raw in core_sources:
        if raw.endswith(".cu"):
            add_object(raw, "rayd_torch_native_core")
    for opener in ("add_library", "add_executable"):
        for body in call_bodies(text, opener):
            items = tokens(body)
            for raw in items[1:]:
                if raw.endswith(".cu"):
                    add_object(raw, items[0])
    return units


# --------------------------------------------------------------------------
# Include closure: which shared headers each translation unit actually reaches.
# --------------------------------------------------------------------------
_CLOSURE_CACHE: dict[tuple[str, str], frozenset[str]] = {}


def include_closure(source: Path, backend: str) -> frozenset[str]:
    """In-repository transitive `#include` closure, conditionals ignored.

    Over-approximates (an `#if`-guarded include still counts), which can only
    over-report an exposure, never miss one.
    """
    key = (str(source), backend)
    cached = _CLOSURE_CACHE.get(key)
    if cached is not None:
        return cached
    seen: set[Path] = set()
    pending = [source]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            text = current.read_text(encoding="utf-8")
        except OSError:
            continue
        for name in INCLUDE_RE.findall(text):
            for directory in (current.parent, *INCLUDE_DIRS[backend]):
                candidate = (directory / name).resolve()
                if not candidate.is_file():
                    continue
                try:
                    candidate.relative_to(ROOT)
                except ValueError:
                    break
                if candidate not in seen:
                    pending.append(candidate)
                break
    result = frozenset(relative(path) for path in seen)
    _CLOSURE_CACHE[key] = result
    return result


def computed_header_exposure(units) -> dict[str, list[str]]:
    exposure: dict[str, set[str]] = {}
    for entry in units.values():
        for header in include_closure(ROOT / entry["source"], entry["backend"]):
            if header.startswith("shared/include/"):
                exposure.setdefault(header, set()).add(entry["profile"])
    return {name: sorted(profiles) for name, profiles in sorted(exposure.items())}


CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
DERIVED = {**drjit_units(), **torch_units()}
DECLARED = {(entry["backend"], entry["unit"]): entry
            for entry in CONTRACT["translation_units"]}


class CompileFlagPolicyContractTests(unittest.TestCase):
    def test_contract_matches_its_schema(self):
        validate(CONTRACT, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))

    def test_declared_translation_units_match_the_build(self):
        # Both directions. An undeclared unit is the blind spot this contract
        # exists to close; a stale declaration is the same blind spot mirrored.
        undeclared = sorted(f"{b}/{u}" for b, u in set(DERIVED) - set(DECLARED))
        stale = sorted(f"{b}/{u}" for b, u in set(DECLARED) - set(DERIVED))
        self.assertEqual(
            (undeclared, stale), ([], []),
            f"compile_policy.json is out of sync with CMake. "
            f"Undeclared in the contract: {undeclared}. "
            f"Declared but not built: {stale}.")
        self.assertEqual(len(DECLARED), len(CONTRACT["translation_units"]),
                         "duplicate (backend, unit) key in the contract")

    def test_declared_profile_and_arch_match_the_build(self):
        for key in sorted(DERIVED):
            derived, declared = DERIVED[key], DECLARED[key]
            with self.subTest(unit="/".join(key)):
                for field in ("source", "kind", "target", "profile",
                              "default_enabled"):
                    self.assertEqual(
                        declared[field], derived[field],
                        f"{key}: {field} declared {declared[field]!r} but CMake "
                        f"says {derived[field]!r}")
                self.assertEqual(declared.get("arch"), derived.get("arch"))
                self.assertEqual(declared.get("option"), derived.get("option"))

    def test_every_declared_profile_is_a_known_profile(self):
        for entry in CONTRACT["translation_units"]:
            with self.subTest(unit=entry["unit"]):
                self.assertIn(entry["profile"], CONTRACT["profiles"])

    def test_profile_flag_sets_are_the_ones_the_parser_recognizes(self):
        # The contract names the flags; the parser maps flags back to a name.
        # If they disagree, every derived profile above is meaningless.
        declared = {tuple(sorted(profile["flags"])): name
                    for name, profile in CONTRACT["profiles"].items()}
        self.assertEqual(declared, PROFILE_BY_FLAGS)

    def test_helper_command_shapes_agree_across_platform_branches(self):
        # cmake/rayd_cuda.cmake emits each nvcc invocation twice, once per
        # platform. A numeric flag or PTX arch present in only one branch would
        # make Linux and Windows compute different arithmetic from the same
        # shared device-math headers, which no other test can see.
        text = (DRJIT / "cmake" / "rayd_cuda.cmake").read_text(encoding="utf-8")
        for name in ("rayd_cuda_object", "rayd_embed_ptx"):
            match = re.search(
                rf"function\({name}\)(.*?)^endfunction\(\)", text, re.DOTALL | re.MULTILINE)
            self.assertIsNotNone(match, name)
            branches = _split_win32_branch(strip_comments(match.group(1)))
            with self.subTest(helper=name, aspect="numeric_flags"):
                self.assertEqual(
                    numeric_flags(branches["win32"].split()),
                    numeric_flags(branches["posix"].split()),
                    f"{name}() numeric flags differ between the WIN32 and POSIX branches")
            with self.subTest(helper=name, aspect="ptx_arch"):
                self.assertEqual(
                    {ARCH_RE.match(t).group(1)
                     for t in branches["win32"].split() if ARCH_RE.match(t)},
                    {ARCH_RE.match(t).group(1)
                     for t in branches["posix"].split() if ARCH_RE.match(t)},
                    f"{name}() PTX arch differs between the WIN32 and POSIX branches")

    def test_numeric_flags_live_only_at_the_declared_places(self):
        # Dr.Jit: the shared object command shape carries no numeric flag at all,
        # so every object unit's profile comes from its own EXTRA_FLAGS. The PTX
        # command shape carries fast-math for all eight units at once.
        helper = drjit_helper_flags()
        self.assertEqual(helper["rayd_cuda_object"], set())
        self.assertEqual(helper["rayd_embed_ptx"], {"--use_fast_math"})

        # Torch: no target-wide or global CUDA numeric flag may exist, otherwise
        # every per-source profile in the contract silently gains it.
        #
        # CMAKE_CUDA_FLAGS is global, so it is the most dangerous place a numeric
        # flag could appear. It is written only to give CODE GENERATION exactly
        # one owner -- Caffe2's duplicate per-architecture `-gencode` pairs are
        # stripped, and grouped `--generate-code` families are appended in their
        # place. Pin those writes and require every one of them to stay free of
        # numeric flags; a fourth write, or a numeric flag in any of them, fails
        # here. ADR-0035 governs numeric flags, not code-generation targets.
        torch_text = strip_comments((TORCH / "CMakeLists.txt").read_text(encoding="utf-8"))
        cuda_flag_writes = [body for body in call_bodies(torch_text, "string")
                            if "CMAKE_CUDA_FLAGS" in body]
        self.assertEqual(len(cuda_flag_writes), 3, cuda_flag_writes)
        for body in cuda_flag_writes:
            with self.subTest(write=" ".join(tokens(body))[:80]):
                self.assertEqual(numeric_flags(tokens(body)), set())
        appends = [body for body in cuda_flag_writes if tokens(body)[:1] == ["APPEND"]]
        self.assertEqual(len(appends), 1, appends)
        # The appended value is the grouped family list, whose every entry the
        # backend validates as `--generate-code=arch=compute_<N>,code=...`.
        self.assertIn("RAYD_TORCH_CUDA_GENCODE_FLAGS", appends[0])
        # set(CMAKE_CUDA_FLAGS ...) and add_compile_options() would bypass the
        # pinned writes above.
        for opener in ("set", "add_compile_options"):
            for body in call_bodies(torch_text, opener):
                self.assertNotIn("CMAKE_CUDA_FLAGS", body)
        for body in call_bodies(torch_text, "target_compile_options"):
            self.assertEqual(numeric_flags(tokens(body)), set())

        # Dr.Jit call sites: the only numeric flag written in the backend
        # CMakeLists is the one cuda_multipath declares.
        drjit_text = strip_comments((DRJIT / "CMakeLists.txt").read_text(encoding="utf-8"))
        self.assertEqual(numeric_flags(drjit_text.split()), {"--use_fast_math"})

    def test_no_numeric_flag_outside_the_contracted_constructs(self):
        # Backstop against constructs the per-shape parsers do not model
        # (add_compile_options, set_property, string(APPEND ...), a resurrected
        # hand-written nvcc block, ...): excise every contracted construct body
        # from each CMakeLists and require the remainder to carry no numeric
        # flag at all. A numeric flag introduced through ANY other spelling then
        # fails here instead of being silently ignored.
        # A flag "occurrence" is a dash-prefixed numeric-flag spelling ANYWHERE
        # inside a token, so generator expressions
        # ($<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>) and quoted strings count,
        # while bare mentions in option names (RAYD_TORCH_OPTIX_FAST_MATH) do
        # not. Doc strings that legitimately quote a flag live in option()
        # bodies, which are excised below.
        embedded_flag = re.compile(
            r"--?(use_fast_math|fmad=|ftz=|prec-div=|prec-sqrt=)", re.IGNORECASE)

        def flag_shaped(items) -> set[str]:
            return {t for t in items if embedded_flag.search(t)}

        def excise(text: str, opener: str, prefix: tuple[str, ...] = ()) -> str:
            for body in call_bodies(text, opener):
                if prefix and tuple(tokens(body)[:len(prefix)]) != prefix:
                    continue
                text = text.replace(body, "", 1)
            return text

        # Torch: flags may live only in the OPTIX nvcc flag variable, PTX custom
        # commands, per-source COMPILE_OPTIONS overrides, target options already
        # asserted empty above, and the option() doc strings. set()/list() are
        # excised ONLY for the contracted variable, so a numeric flag smuggled
        # into any other variable stays visible to this test.
        remainder = strip_comments((TORCH / "CMakeLists.txt").read_text(encoding="utf-8"))
        for opener in ("add_custom_command", "set_source_files_properties",
                       "target_compile_options", "option"):
            remainder = excise(remainder, opener)
        remainder = excise(remainder, "set", prefix=("RAYD_TORCH_OPTIX_NVCC_FLAGS",))
        remainder = excise(remainder, "list", prefix=("APPEND", "RAYD_TORCH_OPTIX_NVCC_FLAGS"))
        with self.subTest(file="torch/CMakeLists.txt"):
            self.assertEqual(
                flag_shaped(tokens(remainder)), set(),
                "numeric flag outside the contracted constructs; either move it "
                "into a contracted construct or extend the contract, never leave "
                "it unmodeled")

        # Dr.Jit: every flag lives in a helper call site.
        remainder = strip_comments((DRJIT / "CMakeLists.txt").read_text(encoding="utf-8"))
        for opener in ("rayd_cuda_object", "rayd_embed_ptx", "option"):
            remainder = excise(remainder, opener)
        with self.subTest(file="drjit/CMakeLists.txt"):
            self.assertEqual(flag_shaped(tokens(remainder)), set())
        # And the helper file itself may only carry numeric flags inside the two
        # function bodies the contract models.
        helper_remainder = strip_comments(
            (DRJIT / "cmake" / "rayd_cuda.cmake").read_text(encoding="utf-8"))
        for name in ("rayd_cuda_object", "rayd_embed_ptx"):
            match = re.search(
                rf"function\({name}\)(.*?)^endfunction\(\)",
                helper_remainder, re.DOTALL | re.MULTILINE)
            self.assertIsNotNone(match, name)
            helper_remainder = helper_remainder.replace(match.group(1), "", 1)
        self.assertEqual(flag_shaped(tokens(helper_remainder)), set())

    def test_shared_header_exposure_is_recomputed_from_the_include_graph(self):
        computed = computed_header_exposure(DERIVED)
        multi = {name: profiles for name, profiles in computed.items()
                 if len(profiles) > 1}
        declared = CONTRACT["shared_header_exposure"]
        self.assertEqual(
            sorted(multi), sorted(declared),
            "a shared header changed how many numeric profiles it is compiled "
            "under. Undeclared: "
            f"{sorted(set(multi) - set(declared))}. No longer multi-profile: "
            f"{sorted(set(declared) - set(multi))}.")
        for name in sorted(multi):
            with self.subTest(header=name):
                self.assertEqual(declared[name], multi[name])

    def test_frozen_divergences_are_still_divergent(self):
        # Mirrors numeric_policy.h's `static_assert(drjit.ray_tmin !=
        # torch.ray_tmin)`: if someone aligns a divergence without updating the
        # record, this fails loudly instead of passing quietly.
        seen_ids = set()
        for divergence in CONTRACT["frozen_divergences"]:
            with self.subTest(divergence=divergence["id"]):
                self.assertNotIn(divergence["id"], seen_ids)
                seen_ids.add(divergence["id"])
                self.assertTrue(divergence["evidence"])
                field = "arch" if divergence["aspect"] == "ptx_arch" else "profile"
                values = []
                for side in divergence["sides"]:
                    key = (side["backend"], side["unit"])
                    self.assertIn(key, DECLARED, f"{divergence['id']} names an "
                                                 f"unknown unit {key}")
                    self.assertEqual(
                        DECLARED[key].get(field), side["value"],
                        f"{divergence['id']} claims {key} is {side['value']!r}")
                    values.append(side["value"])
                if divergence["aspect"] in ("numeric_profile", "ptx_arch"):
                    self.assertGreater(
                        len(set(values)), 1,
                        f"{divergence['id']} is no longer a divergence; if it was "
                        f"deliberately aligned, remove it from the record")
                for header in divergence["shared_headers"]:
                    self.assertTrue((ROOT / header).is_file(), header)

    def test_structural_divergences_still_describe_the_build(self):
        torch_text = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        drjit_text = (DRJIT / "CMakeLists.txt").read_text(encoding="utf-8")

        # D8: the Torch fast-math switch exists, defaults ON, and the Dr.Jit
        # backend has no equivalent.
        self.assertTrue(cmake_options(torch_text)["RAYD_TORCH_OPTIX_FAST_MATH"])
        self.assertNotIn("FAST_MATH", "".join(cmake_options(drjit_text)))

        # D9: every Dr.Jit PTX regeneration option is OFF by default, so the
        # shipped device code is the committed header, not a fresh compile.
        regenerate = {name: default for name, default in cmake_options(drjit_text).items()
                      if name.startswith("RAYD_REGENERATE_")}
        self.assertEqual(len(regenerate), 8)
        self.assertEqual(set(regenerate.values()), {False})
        record = json.loads((DRJIT / "ptx_sources.json").read_text(encoding="utf-8"))
        # Verification is attested per module (--mark-verified); the divergence
        # holds while ANY module ships an unverified committed header.
        self.assertIn(False, {m["regeneration_verified"]
                              for m in record["modules"].values()})

        # D7: equal profile, unequal device code. The cited parity test is the
        # evidence, so it must still exist under that name.
        parity = (DRJIT / "tests" / "drjit" / "test_cuda_multipath.py").read_text(
            encoding="utf-8")
        self.assertIn("def test_diffraction_paths_parity", parity)

    def test_adr_mandated_profiles_hold(self):
        by_source: dict[str, set[str]] = {}
        for entry in CONTRACT["translation_units"]:
            by_source.setdefault(entry["source"], set()).add(entry["profile"])
        for adr, mandate in sorted(CONTRACT["adr_mandates"].items()):
            with self.subTest(adr=adr):
                self.assertTrue((ROOT / mandate["record"]).is_file(),
                                mandate["record"])
                for source in mandate["sources"]:
                    self.assertEqual(
                        by_source.get(source), {mandate["profile"]},
                        f"ADR-{adr} requires {source} to be {mandate['profile']}")
                for source in mandate["excluded_sources"]:
                    self.assertNotIn(
                        mandate["profile"], by_source.get(source, set()),
                        f"ADR-{adr} forbids {source} from taking "
                        f"{mandate['profile']}")
                if mandate["exhaustive"]:
                    carriers = {source for source, profiles in by_source.items()
                                if mandate["profile"] in profiles}
                    self.assertEqual(
                        carriers, set(mandate["sources"]),
                        f"the {mandate['profile']} profile spread beyond the "
                        f"ADR-{adr} family")

    def test_record_and_pointer_are_wired(self):
        self.assertTrue(ADR_PATH.is_file())
        self.assertEqual(CONTRACT["record"], relative(ADR_PATH))
        adr = ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("shared/contracts/compile_policy.json", adr)
        for divergence in CONTRACT["frozen_divergences"]:
            with self.subTest(divergence=divergence["id"]):
                self.assertIn(divergence["id"], adr)
        index = (ROOT / "docs" / "adr" / "README.md").read_text(encoding="utf-8")
        self.assertIn(ADR_PATH.name, index)
        # numeric_policy.h owns the constants and must point at this contract for
        # the flags, so a reader of either artifact finds the other.
        header = NUMERIC_POLICY.read_text(encoding="utf-8")
        self.assertIn("shared/contracts/compile_policy.json", header)


if __name__ == "__main__":
    unittest.main()
