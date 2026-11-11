# Copyright Xingyu Chen.
# Checks that the Scene and MixedScene methods both backends expose take the same parameters in the same order.

"""Cross-backend signature parity for the shared scene surface.

Torch signatures come from the `rayd._impl` sources (and, when the built
`rayd.torch` imports, from `inspect.signature`). Dr.Jit signatures come from the
committed nanobind stub `python/rayd/drjit/_C/__init__.pyi`, which
`scripts/gen_drjit_stub.sh` regenerates from the built extension, so the
comparison needs neither a GPU nor a built backend and runs in hosted
governance. Every difference the two backends are allowed to keep is listed
below with its reason; any other difference fails.
"""

from __future__ import annotations

import ast
import inspect
import json
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "python" / "rayd" / "_impl"
TORCH_SOURCES = {"Scene": IMPL / "scene.py", "MixedScene": IMPL / "mixed.py"}
DRJIT_STUB = ROOT / "python" / "rayd" / "drjit" / "_C" / "__init__.pyi"
OPERATIONS = ROOT / "contracts" / "operations.json"

POSITIONAL = "positional"
KEYWORD_ONLY = "keyword_only"
PROPERTY = "property"
BACKEND_SELECTORS = ("trace_backend", "edge_bvh_backend")

# Parameters only the Torch method takes, keyed by (class, method, parameter).
TORCH_ONLY = {
    # ADR-0038 in-process replication; the Dr.Jit backend is process-per-GPU.
    ("Scene", "__init__", "devices"): "Torch-only replicated multi-device scene (ADR-0038).",
    ("Scene", "__init__", "options"): "Torch-only MultiDeviceOptions for the replicated layer (ADR-0038).",
    # The Dr.Jit recursive states carry their own validity.
    ("Scene", "accum_dfr", "recursive_active"): "Torch-only validity mask for the recursive state set.",
}

# Parameters only the Dr.Jit method takes, keyed by (class, method, parameter).
DRJIT_ONLY = {
    ("Scene", "visible", "ignore_prim_ids"): "Dr.Jit-only keyword that excludes one primitive per segment.",
    ("Scene", "trace_reflections", "symbolic"): "Selects the Dr.Jit symbolic bounce-level path; Torch is eager only.",
    ("Scene", "trace_reflections", "options"): "Dr.Jit-only ReflectionTraceOptions overload.",
    # The flat overload spells out the ReflectionTraceOptions knobs.
    ("Scene", "trace_reflections", "deduplicate"): "Dr.Jit-only flat ReflectionTraceOptions overload.",
    ("Scene", "trace_reflections", "canonical_prim_table"): "Dr.Jit-only flat ReflectionTraceOptions overload.",
    ("Scene", "trace_reflections", "image_source_tolerance"): "Dr.Jit-only flat ReflectionTraceOptions overload.",
    # Adding an options record to the Torch call changes its dispatcher schema, which this API round leaves alone.
    ("Scene", "trace_refl_epc_field", "options"): "Torch trace_refl_epc_field has no options record yet.",
}

# Parameters whose names differ by position, keyed by (class, method, Dr.Jit name) -> Torch name.
RENAMED = {
    # Documented semantic difference: Torch traces from source positions, Dr.Jit from a Ray.
    ("Scene", "trace_refl_epc", "ray"): "source",
    # Dr.Jit's Ray overloads start from a Ray; its position overloads use `source` like Torch.
    ("Scene", "trace_refl_epc_field", "ray"): "source",
}

# Parameters whose positional/keyword-only status differs, keyed by (class, method, parameter).
KIND_DIFFERENCES = {
    # ADR-0031: the Torch path exporter requires its validity mask, so it is keyword-only with no default.
    ("Scene", "trace_dfr_paths", "active"): "Torch requires the validity mask as a keyword; Dr.Jit defaults it."
}

# Operations whose `inputs` describe something other than one Scene method's parameters.
UNMAPPED_OPERATIONS = {
    "sdf_intersect": "Inputs are the SdfGrid fields plus the rayd.torch.sdf_intersect arguments, not a Scene method.",
    "mixed_scene": "Inputs include the MixedScene contents (meshes, SDF grids, surfel clouds) besides the query.",
}


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str


def _parameters(function: ast.FunctionDef) -> tuple[Parameter, ...]:
    arguments = function.args
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise AssertionError(f"{function.name} takes *args or **kwargs")
    positional = [*arguments.posonlyargs, *arguments.args][1:]
    return (
        *(Parameter(argument.arg, POSITIONAL) for argument in positional),
        *(Parameter(argument.arg, KEYWORD_ONLY) for argument in arguments.kwonlyargs),
    )


def _decorators(function: ast.FunctionDef) -> set[str]:
    return {ast.unparse(decorator) for decorator in function.decorator_list}


def _public(name: str) -> bool:
    return name == "__init__" or not name.startswith("_")


def _class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} defines no class {name}")


def torch_signatures_from_source(class_name: str) -> dict[str, object]:
    """Map each public Torch member to its parameters, or PROPERTY. Overload stubs give way to the implementation."""
    result: dict[str, object] = {}
    for node in _class_node(TORCH_SOURCES[class_name], class_name).body:
        if not isinstance(node, ast.FunctionDef) or not _public(node.name):
            continue
        decorators = _decorators(node)
        if "overload" in decorators:
            continue
        if decorators & {"property", "functools.cached_property", "cached_property"}:
            result[node.name] = PROPERTY
        elif not any(decorator.endswith(".setter") for decorator in decorators):
            result[node.name] = _parameters(node)
    return result


def torch_signatures_from_runtime(class_name: str) -> dict[str, object]:
    import rayd.torch as backend

    kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD: POSITIONAL, inspect.Parameter.POSITIONAL_ONLY: POSITIONAL}
    kinds[inspect.Parameter.KEYWORD_ONLY] = KEYWORD_ONLY
    cls = getattr(backend, class_name)
    result: dict[str, object] = {}
    for name, member in vars(cls).items():
        if not _public(name):
            continue
        if isinstance(member, property):
            result[name] = PROPERTY
        elif inspect.isfunction(member):
            parameters = list(inspect.signature(member).parameters.values())[1:]
            result[name] = tuple(Parameter(parameter.name, kinds[parameter.kind]) for parameter in parameters)
    return result


def drjit_signatures(class_name: str) -> dict[str, object]:
    """Map each public Dr.Jit stub member to the list of its overload parameter tuples, or PROPERTY."""
    result: dict[str, object] = {}
    for node in _class_node(DRJIT_STUB, class_name).body:
        if not isinstance(node, ast.FunctionDef) or not _public(node.name):
            continue
        decorators = _decorators(node)
        if "property" in decorators:
            result[node.name] = PROPERTY
        elif not any(decorator.endswith(".setter") for decorator in decorators):
            result.setdefault(node.name, []).append(_parameters(node))
    return result


def _normalized(class_name: str, method: str, parameters, only: dict, renamed: dict | None = None):
    """Drop the allowlisted backend-only parameters and apply the allowlisted renames."""
    used = set()
    names = []
    for parameter in parameters:
        key = (class_name, method, parameter.name)
        if key in only:
            used.add(("only", key))
            continue
        if renamed is not None and key in renamed:
            used.add(("renamed", key))
            parameter = Parameter(renamed[key], parameter.kind)
        names.append(parameter)
    return tuple(names), used


def signature_mismatches(torch_members: dict[str, object]) -> tuple[list[str], set]:
    """Return every unallowlisted difference plus the allowlist entries the comparison relied on."""
    problems: list[str] = []
    used: set = set()
    for class_name in TORCH_SOURCES:
        drjit_members = drjit_signatures(class_name)
        torch_class = torch_members[class_name]
        for method in sorted(set(torch_class) & set(drjit_members)):
            torch_entry, drjit_entry = torch_class[method], drjit_members[method]
            label = f"{class_name}.{method}"
            if (torch_entry == PROPERTY) != (drjit_entry == PROPERTY):
                problems.append(f"{label}: property in only one backend")
                continue
            if torch_entry == PROPERTY:
                continue
            expected, torch_used = _normalized(class_name, method, torch_entry, TORCH_ONLY)
            used |= torch_used
            for overload in drjit_entry:
                actual, drjit_used = _normalized(class_name, method, overload, DRJIT_ONLY, RENAMED)
                used |= drjit_used
                if [p.name for p in actual] != [p.name for p in expected]:
                    problems.append(
                        f"{label}: Torch {[p.name for p in expected]} != Dr.Jit {[p.name for p in actual]} "
                        f"(Dr.Jit overload {[p.name for p in overload]})"
                    )
                    continue
                for torch_parameter, drjit_parameter in zip(expected, actual):
                    if torch_parameter.kind == drjit_parameter.kind:
                        continue
                    key = (class_name, method, torch_parameter.name)
                    if key in KIND_DIFFERENCES:
                        used.add(("kind", key))
                    else:
                        problems.append(
                            f"{label}.{torch_parameter.name}: Torch {torch_parameter.kind} != "
                            f"Dr.Jit {drjit_parameter.kind}"
                        )
    return problems, used


def torch_members_from_source() -> dict[str, dict[str, object]]:
    return {name: torch_signatures_from_source(name) for name in TORCH_SOURCES}


class ApiSignatureParityTests(unittest.TestCase):
    def test_shared_scene_methods_take_the_same_parameters(self) -> None:
        problems, _ = signature_mismatches(torch_members_from_source())
        self.assertEqual(problems, [], "\n".join(problems))

    def test_every_allowlist_entry_is_still_needed(self) -> None:
        _, used = signature_mismatches(torch_members_from_source())
        declared = {
            *(("only", key) for key in TORCH_ONLY),
            *(("only", key) for key in DRJIT_ONLY),
            *(("renamed", key) for key in RENAMED),
            *(("kind", key) for key in KIND_DIFFERENCES),
        }
        self.assertEqual(sorted(declared - used), [], "allowlisted differences that no longer occur")

    def test_the_comparison_covers_the_shared_scene_surface(self) -> None:
        torch_scene = torch_signatures_from_source("Scene")
        shared = set(torch_scene) & set(drjit_signatures("Scene"))
        expected = {
            "__init__",
            "add_mesh",
            "add_instance",
            "build",
            "sync",
            "intersect",
            "nearest_edge",
            "nearest_edges",
            "visible",
            "visible_pair",
            "visible_edge",
            "visible_chain",
            "trace_reflections",
            "trace_refl_epc",
            "trace_refl_epc_field",
            "accum_refl",
            "trace_dfr_paths",
            "accum_dfr_direct",
            "accum_dfr",
            "accum_dfr_coherent_direct",
            "set_edge_mask",
            "global_geometry",
            "trace_backend",
            "edge_bvh_backend",
        }
        self.assertLessEqual(expected, shared)
        self.assertIn("__init__", set(torch_signatures_from_source("MixedScene")) & set(drjit_signatures("MixedScene")))

    def test_backend_selectors_are_keyword_only_in_both_backends(self) -> None:
        for class_name in TORCH_SOURCES:
            torch_init = torch_signatures_from_source(class_name)["__init__"]
            drjit_inits = drjit_signatures(class_name)["__init__"]
            for name in BACKEND_SELECTORS:
                with self.subTest(class_name=class_name, selector=name):
                    self.assertIn(Parameter(name, KEYWORD_ONLY), torch_init)
                    for overload in drjit_inits:
                        self.assertIn(Parameter(name, KEYWORD_ONLY), overload)
                    self.assertTrue(all(parameter.kind == KEYWORD_ONLY for parameter in torch_init))

    def test_source_signatures_match_the_imported_torch_backend(self) -> None:
        try:
            import rayd.torch  # noqa: F401
        except Exception as error:  # The hosted run has no built backend.
            self.skipTest(f"rayd.torch is not importable: {error}")
        for class_name in TORCH_SOURCES:
            with self.subTest(class_name=class_name):
                runtime = torch_signatures_from_runtime(class_name)
                source = torch_signatures_from_source(class_name)
                self.assertEqual({name: runtime[name] for name in source}, source)
        problems, _ = signature_mismatches({name: torch_signatures_from_runtime(name) for name in TORCH_SOURCES})
        self.assertEqual(problems, [], "\n".join(problems))

    def test_operation_inputs_are_real_method_parameters(self) -> None:
        operations = json.loads(OPERATIONS.read_text(encoding="utf-8"))["operations"]
        torch_scene = torch_signatures_from_source("Scene")
        drjit_scene = drjit_signatures("Scene")
        self.assertLessEqual(set(UNMAPPED_OPERATIONS), set(operations))
        for operation, entry in operations.items():
            if operation in UNMAPPED_OPERATIONS:
                continue
            known: set[str] = set()
            for method in entry["derivative_variants"]:
                self.assertIn(method, torch_scene, f"{operation}: Torch Scene has no {method}")
                self.assertIn(method, drjit_scene, f"{operation}: Dr.Jit Scene has no {method}")
                known.update(parameter.name for parameter in torch_scene[method])
                for overload in drjit_scene[method]:
                    known.update(parameter.name for parameter in overload)
            with self.subTest(operation=operation):
                self.assertTrue(entry["inputs"])
                self.assertEqual(sorted(set(entry["inputs"]) - known), [])


if __name__ == "__main__":
    unittest.main()
