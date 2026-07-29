# Copyright Xingyu Chen.
# Tests ptx source digest.

"""Staleness guard for the committed Dr.Jit OptiX PTX headers.

The Dr.Jit backend commits eight generated `*_ptx.h` headers so that building a
wheel needs no OptiX SDK. Regeneration is opt-in and OFF by default, so an edit
to a `.cu` file or to any header it reaches leaves the committed PTX describing
older device code with nothing in the build to notice.

`drjit/ptx_sources.json` records, per module, the transitive
in-repository include closure of the `.cu` and a digest over its contents. This
test recomputes that record from source and fails when it drifts. It needs no
CUDA, no OptiX SDK, no GPU and no build, so it runs anywhere the repository is
checked out.

The record states source identity, not correctness: it says "these inputs are
unchanged since the record was written", never "the committed PTX is a correct
compile of them". The `adoption` block carries that caveat and this test asserts
the caveat has not been quietly dropped.
"""

import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "drjit"
RECORD_PATH = BACKEND / "ptx_sources.json"
SCRIPT_PATH = BACKEND / "scripts" / "audit_ptx_sources.py"
CMAKELISTS = BACKEND / "CMakeLists.txt"
RAYD_CUDA_CMAKE = ROOT / "cmake" / "RayDOptix.cmake"
TORCH_CMAKELISTS = ROOT / "torch" / "CMakeLists.txt"


def _load_audit_script():
    spec = importlib.util.spec_from_file_location("rayd_audit_ptx_sources", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_ptx_headers(root):
    """Every checked-in `*_ptx.h` under `root`, ignoring build output trees."""
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in root.rglob("*_ptx.h")
        if "build" not in path.relative_to(ROOT).parts
    )


AUDIT = _load_audit_script()
RECORD = json.loads(RECORD_PATH.read_text(encoding="utf-8"))


class PtxSourceDigestTest(unittest.TestCase):
    def test_record_covers_every_committed_ptx_header_and_option(self):
        # A ninth PTX module, or a header dropped from the record, must not be
        # able to slip past the guard by simply not being mentioned.
        recorded_headers = sorted(
            module["header"] for module in RECORD["modules"].values())
        self.assertEqual(
            _committed_ptx_headers(ROOT / "generated" / "drjit" / "ptx"), recorded_headers)

        cmake = CMAKELISTS.read_text(encoding="utf-8")
        declared_options = set(
            re.findall(r"^option\((RAYD_REGENERATE_\w+_PTX)", cmake, re.MULTILINE))
        recorded_options = {m["option"] for m in RECORD["modules"].values()}
        self.assertEqual(declared_options, recorded_options)
        self.assertEqual(len(recorded_options), len(RECORD["modules"]))

        for name, module in RECORD["modules"].items():
            self.assertEqual(module["variable"], f"{name}_ptx")
            self.assertTrue((ROOT / module["cu"]).is_file(), module["cu"])

    def test_recorded_digests_match_the_current_sources(self):
        # The whole guard. A changed .cu, a changed header anywhere in a closure,
        # or a new #include all land here.
        current = AUDIT.audit()["modules"]
        for name in sorted(set(current) | set(RECORD["modules"])):
            with self.subTest(module=name):
                self.assertIn(name, RECORD["modules"])
                self.assertIn(name, current)
                recorded, computed = RECORD["modules"][name], current[name]
                added = sorted(set(computed["sources"]) - set(recorded["sources"]))
                removed = sorted(set(recorded["sources"]) - set(computed["sources"]))
                self.assertEqual(
                    computed["source_sha256"], recorded["source_sha256"],
                    f"{name} PTX sources changed since the record was written "
                    f"(added={added}, removed={removed}). Regenerate the PTX with "
                    f"-D{recorded['option']}=ON, copy the build-tree header over "
                    f"{recorded['header']}, then run "
                    f"'python drjit/scripts/audit_ptx_sources.py --write'.")
                self.assertEqual(
                    computed["external_includes"], recorded["external_includes"],
                    f"{name} gained or lost an out-of-repository include; its "
                    f"device code may have changed without any tracked file "
                    f"changing.")

    def test_committed_ptx_headers_match_their_recorded_digest(self):
        for name, module in RECORD["modules"].items():
            with self.subTest(module=name):
                header = ROOT / module["header"]
                digest = hashlib.sha256(
                    header.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                self.assertEqual(digest, module["header_sha256"])

    def test_cmake_depends_equals_the_include_closure(self):
        # The DEPENDS lists are what makes the build rebuild a PTX blob, and the
        # closure is what the digest hashes. If they disagree, one of the two is
        # missing a file and the guard is leaky on exactly that file.
        drift = AUDIT.depends_drift()
        self.assertEqual(
            drift, {},
            "rayd_embed_ptx() DEPENDS is out of sync with the include closure in "
            f"{CMAKELISTS.relative_to(ROOT).as_posix()}: {json.dumps(drift, indent=2)}")

    def test_nvcc_ptx_flags_are_recorded_verbatim(self):
        # The digest covers sources only, so the compile flags are pinned here.
        # rayd_embed_ptx() carries them twice: the Windows .bat body and the
        # POSIX add_custom_command.
        flags = RECORD["nvcc_ptx_flags"]
        self.assertEqual(flags, AUDIT.NVCC_PTX_FLAGS)
        cmake_text = RAYD_CUDA_CMAKE.read_text(encoding="utf-8")
        self.assertEqual(
            cmake_text.count(flags), 2,
            f"expected the PTX nvcc flags '{flags}' exactly twice in "
            f"{RAYD_CUDA_CMAKE.relative_to(ROOT).as_posix()} (Windows and POSIX "
            f"branches of rayd_embed_ptx)")
        self.assertNotIn(
            "-ptx ", CMAKELISTS.read_text(encoding="utf-8"),
            "PTX nvcc flags belong to rayd_embed_ptx(), not to a call site")

    def test_configure_time_check_is_wired(self):
        # Secondary to this test file, but it is what tells a developer mid-edit.
        # Deleting it should not be silent.
        cmake = CMAKELISTS.read_text(encoding="utf-8")
        self.assertIn("scripts/audit_ptx_sources.py", cmake)
        self.assertIn(
            'option(RAYD_STRICT_PTX_SOURCE_CHECK "Fail configuration when the '
            'committed PTX source-identity record is stale." OFF)', cmake)
        self.assertIn("RAYD_STRICT_PTX_SOURCE_CHECK)\n        message(FATAL_ERROR", cmake)

    def test_adoption_record_does_not_overclaim(self):
        # The record was bootstrapped from the tree as it stood, on a machine
        # that cannot regenerate PTX. It must keep saying so until someone
        # actually reproduces a header byte-for-byte -- and the attestation is
        # per module (--mark-verified), never a blanket claim.
        adoption = RECORD["adoption"]
        self.assertRegex(adoption["baseline_commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(adoption["regeneration_note"].strip())
        drift = adoption["sources_committed_after_header"]
        self.assertEqual(set(drift), set(RECORD["modules"]))
        for name, module in RECORD["modules"].items():
            with self.subTest(module=name):
                self.assertIn("regeneration_verified", module)
                self.assertIsInstance(module["regeneration_verified"], bool)

    def test_drjit_pin_is_the_real_pyproject_pin(self):
        # The Dr.Jit headers are genuine PTX inputs recorded by name only, so
        # the pin is the record's proxy for their content. Three uncorrelated
        # literals (pyproject, script, record) would let a version bump slide
        # through with every check green; anchoring all of them to
        # pyproject.toml makes a bump force a conscious --write.
        pyproject = (BACKEND / "pyproject.toml").read_text(encoding="utf-8")
        pins = set(re.findall(r'"(drjit==[^"]+)"', pyproject))
        self.assertEqual(len(pins), 1, f"ambiguous drjit pins in pyproject: {pins}")
        self.assertEqual(RECORD["drjit_pin"], pins.pop())
        self.assertEqual(RECORD["drjit_pin"], AUDIT.drjit_pin())

    def test_torch_backend_commits_no_ptx(self):
        # The guard is Dr.Jit-scoped because only Dr.Jit commits PTX. Torch
        # regenerates every blob into the binary dir on each native build, so it
        # cannot go stale. Keep it that way.
        self.assertEqual(_committed_ptx_headers(ROOT / "torch"), [])
        torch_cmake = TORCH_CMAKELISTS.read_text(encoding="utf-8")
        headers = re.findall(
            r"^\s*set\(RAYD_TORCH_\w*PTX_HEADER\s+\"([^\"]+)\"\)",
            torch_cmake, re.MULTILINE)
        self.assertTrue(headers)
        for header in headers:
            self.assertTrue(
                header.startswith("${CMAKE_CURRENT_BINARY_DIR}/"),
                f"Torch PTX header escapes the binary dir: {header}")


if __name__ == "__main__":
    unittest.main()
