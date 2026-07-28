import ast
import hashlib
import json
import runpy
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "shared" / "contracts"
MANIFEST_PATH = CONTRACT_DIR / "public_api.json"
SCHEMA_PATH = CONTRACT_DIR / "public_api.schema.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
DRJIT_BINDING_SOURCE = ROOT / "backends" / "drjit" / "src" / "rayd.cpp"

DRJIT_PACKAGE = ROOT / "backends" / "drjit" / "python" / "rayd" / "drjit"
TORCH_PACKAGE = ROOT / "backends" / "torch" / "python" / "rayd" / "torch"
DRJIT_NATIVE_STUB = DRJIT_PACKAGE / "_C.pyi"
# `rayd/drjit/__init__.pyi` is the one shadow stub that has to stay. It carries
# no annotations of its own -- it re-exports `_C` and the two `_capabilities`
# helpers -- but by shadowing `__init__.py` it also keeps a type checker from
# following that module's runtime `import drjit as _drjit`. Dr.Jit 1.3.1 ships
# a syntactically invalid `drjit/__init__.pyi` (line 1176 is `def \1(...)`),
# and mypy answers a syntax error in a followed file by abandoning the whole
# run, which drops every `rayd.drjit` public symbol to untyped. `rayd-drjit`
# pins `drjit==1.3.1` exactly, so that is not an incidental environment, and
# the import is required at runtime for the win32 DLL-directory branch, so
# there is no annotation-only way to hide it. Drop this stub once the pin moves
# to a Dr.Jit release with a parsable stub.
DRJIT_TOP_LEVEL_STUB = DRJIT_PACKAGE / "__init__.pyi"

# The public modules of each backend package. Every one of them is typed
# inline: the two stubs above are the only ones in the repository, because the
# nanobind extension has no Python source to annotate and the top-level stub
# shields the checker from a broken third-party stub.
DRJIT_PUBLIC_MODULES = ("__init__", "path_exchange")
TORCH_PUBLIC_MODULES = (
    "__init__",
    "autograd",
    "camera",
    "mesh",
    "path_exchange",
    "scene",
    "sdf",
    "types",
    # Private module, public surface: `MultiDeviceOptions` is re-exported from
    # `rayd.torch` and `DeviceCalibration` is what `Scene.calibrate_devices()`
    # hands back, so both are as reachable for a downstream caller as anything
    # in the modules above.
    "_multi",
)

# Dunder methods that are part of a public class's callable surface and were
# typed by the removed stubs; every other dunder stays out of scope.
PUBLIC_DUNDERS = frozenset(
    {"__init__", "__call__", "__len__", "__iter__", "__getitem__", "__contains__"}
)


def _is_public(name):
    return not name.startswith("_") or name in PUBLIC_DUNDERS


def _type_checking_polarity(node):
    """Which branch of an `if` a type checker takes, or `None` if it is not one.

    `True` for `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:`, `False` for
    the negated form. Polarity matters: `rayd.torch.__init__` hides its
    `__getattr__` hook under `if not TYPE_CHECKING:` precisely so no checker
    reads it, and folding that body into the declared type surface would
    demand annotations nothing will ever consult. Only the two bare forms are
    recognised, so a compound test (`if TYPE_CHECKING and x:`) is left alone
    rather than guessed at.
    """
    test = node.test
    negated = False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
        negated = True
    if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    ):
        return not negated
    return None


def _statements(body):
    """Module- or class-body statements as a type checker reads them.

    Inline annotations may keep forward-reference imports and `@overload`
    declarations under `if TYPE_CHECKING:` so runtime import order and cost do
    not change. Those declarations are still part of the declared type surface,
    so the checks below have to see through the guard -- and, by the same
    token, have to stay out of the branch a checker discards.
    """
    for node in body:
        polarity = _type_checking_polarity(node) if isinstance(node, ast.If) else None
        if polarity is True:
            yield from _statements(node.body)
        elif polarity is False:
            yield from _statements(node.orelse)
        else:
            yield node


def _decorator_names(node):
    names = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        while isinstance(target, ast.Attribute):
            names.add(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _overloaded_names(statements):
    """Names carrying at least one `@overload` declaration in this scope.

    The implementation behind a set of `@overload` declarations is not part of
    the public type surface, so only the declarations are checked.
    """
    return {
        node.name
        for node in statements
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "overload" in _decorator_names(node)
    }


def _signature_problems(node, qualname, skip_first):
    arguments = node.args
    positional = list(arguments.posonlyargs) + list(arguments.args)
    if skip_first and positional:
        positional = positional[1:]
    declared = positional + list(arguments.kwonlyargs)
    if arguments.vararg is not None:
        declared.append(arguments.vararg)
    if arguments.kwarg is not None:
        declared.append(arguments.kwarg)
    problems = [
        f"{qualname} (line {node.lineno}): parameter '{argument.arg}' has no annotation"
        for argument in declared
        if argument.annotation is None
    ]
    if node.returns is None:
        problems.append(f"{qualname} (line {node.lineno}): no return annotation")
    return problems


def _unannotated_public_surface(path):
    """Every public declaration in `path` that a type checker cannot resolve.

    This is the inline-annotation form of the guarantee the shadow stubs used
    to carry: a public function, method, or dataclass field added without a
    type is reported here instead of silently widening to `Any`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_statements = list(_statements(tree.body))
    module_overloads = _overloaded_names(module_statements)
    problems = []
    for node in module_statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_public(node.name):
                continue
            decorators = _decorator_names(node)
            if node.name in module_overloads and "overload" not in decorators:
                continue
            problems.extend(_signature_problems(node, node.name, False))
        elif isinstance(node, ast.ClassDef):
            if not _is_public(node.name):
                continue
            class_statements = list(_statements(node.body))
            class_overloads = _overloaded_names(class_statements)
            is_dataclass = "dataclass" in _decorator_names(node)
            for child in class_statements:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _is_public(child.name):
                        continue
                    decorators = _decorator_names(child)
                    if child.name in class_overloads and "overload" not in decorators:
                        continue
                    problems.extend(
                        _signature_problems(
                            child,
                            f"{node.name}.{child.name}",
                            "staticmethod" not in decorators,
                        )
                    )
                elif is_dataclass and isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and _is_public(target.id):
                            problems.append(
                                f"{node.name}.{target.id} (line {child.lineno}): "
                                "dataclass field has no annotation"
                            )
    return problems


def _declared_all(statements):
    node = next(
        node for node in statements
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    return node, {
        element.value for element in node.value.elts if isinstance(element, ast.Constant)
    }


def _reexported_names(statements):
    """Public names an `__init__` binds by importing them at module scope.

    Only intra-package imports count: `from __future__ import annotations` and
    any typing import bind names too, but they are compiler and type-checker
    plumbing rather than part of the backend's public surface.
    """
    return {
        alias.asname or alias.name
        for node in statements
        if isinstance(node, ast.ImportFrom) and node.level > 0
        for alias in node.names
        if alias.name != "*" and not (alias.asname or alias.name).startswith("_")
    }


def _lazily_bound_names(tree):
    """Public names an `__init__` binds on first attribute access.

    `rayd.torch.MultiDeviceOptions` is a real re-export -- `__all__` lists it
    and `from rayd.torch import *` resolves it -- but it is deliberately not
    imported at module scope so a single-device program never imports
    `rayd.torch._multi`.

    This one reads the *runtime* module rather than the checker's view of it:
    the hook lives under `if not TYPE_CHECKING:`, which `_statements` drops on
    purpose, so the whole tree is walked instead.
    """
    hook = next(
        (
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__getattr__"
        ),
        None,
    )
    if hook is None:
        return set()
    return {
        child.value
        for child in ast.walk(hook)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and child.value.isidentifier()
    }


def _module_tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_statements(path):
    return list(_statements(_module_tree(path).body))


def _star_imported_modules(statements):
    """Modules a `from .X import *` pulls every public name of."""
    return {
        node.module
        for node in statements
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and any(alias.name == "*" for alias in node.names)
    }


def _drjit_bound_names():
    """Every public name bound by NB_MODULE(_C, m) in the Dr.Jit extension source.

    `rayd.cpp` holds the single NB_MODULE and binds everything on the module
    object `m`, so scanning it is authoritative. `re.S` is required: several
    `nb::class_<...>` template argument lists wrap before `(m, "Name")`.
    """
    source = DRJIT_BINDING_SOURCE.read_text(encoding="utf-8")
    names = set(
        re.findall(
            r'nb::(?:class_|enum_)<.*?>\s*\(\s*m,\s*"([A-Za-z0-9_]+)"', source, re.S
        )
    )
    names.update(re.findall(r'\bm\.(?:def|attr)\(\s*"([A-Za-z0-9_]+)"', source))
    names.discard("__name__")
    return names


class PublicApiManifestTests(unittest.TestCase):
    def test_manifest_matches_schema_enums_and_required_fields(self):
        self.assertEqual(MANIFEST["version"], 2)
        self.assertEqual(
            set(SCHEMA["required"]),
            {
                "version",
                "capability_keys",
                "stability_levels",
                "naming_conventions",
                "apis",
                "aliases",
                "backends",
                "trace",
            },
        )
        categories = {"core", "multipath", "surfel", "experimental"}
        stability = set(MANIFEST["stability_levels"])
        self.assertEqual(stability, {"stable", "provisional", "experimental", "deprecated"})
        self.assertEqual(set(MANIFEST["apis"]), set(MANIFEST["capability_keys"]))
        for name, metadata in MANIFEST["apis"].items():
            with self.subTest(api=name):
                self.assertIn(metadata["category"], categories)
                self.assertIn(metadata["stability"], stability)
            self.assertTrue(metadata["summary"])

        naming = MANIFEST["naming_conventions"]
        self.assertEqual(set(naming), {"options", "results", "fields"})
        self.assertIn("<Operation>Options", naming["options"])
        self.assertIn("PascalCase", naming["results"])
        self.assertIn("global_", naming["fields"])

    def test_backend_capabilities_are_complete_and_boolean(self):
        required = set(MANIFEST["capability_keys"])
        operations = json.loads(
            (CONTRACT_DIR / "operations.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(operations["required_capability_keys"]), {"backend"} | required)
        for backend in ("drjit", "torch"):
            entry = MANIFEST["backends"][backend]
            self.assertEqual(set(entry["capabilities"]), required)
            self.assertTrue(all(type(value) is bool for value in entry["capabilities"].values()))
            self.assertEqual(entry["typing"], "complete")

    def test_runtime_modules_are_validated_copies_of_shared_manifest(self):
        schema_hash = hashlib.sha256(
            MANIFEST_PATH.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for backend in ("drjit", "torch"):
            module_path = (
                ROOT / "backends" / backend / "python" / "rayd" / backend / "_capabilities.py"
            )
            namespace = runpy.run_path(str(module_path))
            flat = namespace["backend_capabilities"]()
            rich = namespace["api_manifest"]()
            self.assertEqual(flat["backend"], backend)
            self.assertEqual(
                {key: value for key, value in flat.items() if key != "backend"},
                MANIFEST["backends"][backend]["capabilities"],
            )
            self.assertEqual(rich["version"], MANIFEST["version"])
            self.assertEqual(rich["schema_sha256"], schema_hash)
            self.assertEqual(rich["typing"], MANIFEST["backends"][backend]["typing"])
            self.assertEqual(rich["naming_conventions"], MANIFEST["naming_conventions"])
            for name, metadata in rich["apis"].items():
                self.assertEqual(metadata["category"], MANIFEST["apis"][name]["category"])
                self.assertEqual(metadata["stability"], MANIFEST["apis"][name]["stability"])
            self.assertEqual(rich["aliases"], MANIFEST["aliases"])
            self.assertEqual(rich["trace"], MANIFEST["trace"])

    def test_trace_axis_records_the_optix_and_cuda_backends(self):
        trace = MANIFEST["trace"]
        self.assertEqual(set(trace), {"backends", "integration_modes", "frontend_support"})
        self.assertEqual(set(trace["backends"]), {"optix", "cuda"})
        self.assertEqual(trace["backends"]["optix"]["stability"], "stable")
        self.assertTrue(trace["backends"]["optix"]["summary"])
        self.assertEqual(trace["backends"]["cuda"]["stability"], "provisional")
        self.assertTrue(trace["backends"]["cuda"]["summary"])
        self.assertEqual(trace["integration_modes"], ["jit_symbolic", "eager_native"])
        # The CUDA backend is eager-native only: it never folds into a Dr.Jit
        # symbolic megakernel; Torch exposes the eager-native CUDA executor.
        self.assertEqual(trace["frontend_support"], {
            "drjit": {"optix": ["jit_symbolic", "eager_native"], "cuda": ["eager_native"]},
            "torch": {"optix": ["eager_native"], "cuda": ["eager_native"]},
        })

    def test_hybrid_is_only_a_deprecated_compatibility_alias(self):
        aliases = MANIFEST["aliases"]["edge_bvh_backend"]
        self.assertEqual(aliases["hybrid"]["canonical"], "optix_drjit")
        self.assertEqual(aliases["hybrid"]["stability"], "deprecated")
        self.assertIn("unrelated", aliases["hybrid"]["summary"])

    def test_complete_typing_markers_are_shipped(self):
        """Both packages stay PEP 561 typed distributions.

        `py.typed` is what makes the inline annotations of a package visible to
        a downstream type checker at all, so it is load-bearing rather than
        decorative now that the shadow stubs are gone.
        """
        for package in (DRJIT_PACKAGE, TORCH_PACKAGE):
            with self.subTest(package=package.name):
                self.assertEqual((package / "py.typed").read_text(encoding="utf-8"), "")
                capabilities = {
                    node.name: node
                    for node in _module_statements(package / "_capabilities.py")
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                for name in ("backend_capabilities", "api_manifest"):
                    with self.subTest(package=package.name, function=name):
                        self.assertIn(name, capabilities)
                        node = capabilities[name]
                        self.assertEqual(_signature_problems(node, name, False), [])
                        arguments = node.args
                        self.assertEqual(
                            arguments.posonlyargs
                            + arguments.args
                            + arguments.kwonlyargs,
                            [],
                        )
                        self.assertIsNone(arguments.vararg)
                        self.assertIsNone(arguments.kwarg)

    def test_drjit_native_extension_stub_is_the_only_type_source_for_it(self):
        """`_C.pyi` is not a shadow stub: `_C` has no Python source at all.

        `rayd.drjit` re-exports the whole nanobind extension, so the stub is
        the only place its bound names can be typed. Both re-exports are
        asserted, because the two files answer different questions: the `.py`
        is what `import rayd.drjit` actually runs, while the `.pyi` shadows it
        for a type checker and is therefore the only one a checker reads. Drop
        `from ._C import *` from the shield and mypy answers `rd.Scene` with
        `Module has no attribute "Scene"` and `Any` -- the whole public surface
        of the backend at once -- while every runtime test stays green.
        """
        ast.parse(
            DRJIT_NATIVE_STUB.read_text(encoding="utf-8"), filename=str(DRJIT_NATIVE_STUB)
        )
        runtime = _module_statements(DRJIT_PACKAGE / "__init__.py")
        shield = _module_statements(DRJIT_TOP_LEVEL_STUB)
        self.assertIn("_C", _star_imported_modules(runtime))
        self.assertIn("_C", _star_imported_modules(shield))
        # `__all__` names the extension does not bind have to be re-exported by
        # the shield explicitly, or they are unresolvable for the same reason.
        _, declared = _declared_all(runtime)
        self.assertLessEqual(
            declared - _drjit_bound_names(),
            _reexported_names(shield),
            "rayd/drjit/__init__.pyi shadows __init__.py, so a name it does not "
            "re-export is untyped for every downstream type checker",
        )

    def test_no_shadow_stub_shadows_an_inline_annotated_module(self):
        """A stub next to a `.py` silently wins over its inline annotations.

        Re-introducing one leaves the repository in a mixed state where a stale
        stub overrides corrected annotations, which is worse than either pure
        option. `_C.pyi` is exempt because it shadows nothing, and
        `drjit/__init__.pyi` is exempt for the measured third-party reason
        recorded next to `DRJIT_TOP_LEVEL_STUB`; every other module is typed
        inline.
        """
        found = sorted(
            path.relative_to(ROOT).as_posix()
            for package in (DRJIT_PACKAGE, TORCH_PACKAGE)
            for path in package.rglob("*.pyi")
        )
        self.assertEqual(
            found,
            sorted(
                path.relative_to(ROOT).as_posix()
                for path in (DRJIT_NATIVE_STUB, DRJIT_TOP_LEVEL_STUB)
            ),
            "the backend packages are typed inline; the nanobind extension stub "
            "and the Dr.Jit top-level shield are the only stubs allowed to ship",
        )
        # The shield must stay a pure re-export: anything else in it would be a
        # second, silently authoritative copy of an inline-annotated surface.
        shield = ast.parse(
            DRJIT_TOP_LEVEL_STUB.read_text(encoding="utf-8"),
            filename=str(DRJIT_TOP_LEVEL_STUB),
        )
        self.assertTrue(
            all(isinstance(node, ast.ImportFrom) for node in shield.body),
            "rayd/drjit/__init__.pyi may only re-export; it must not declare types",
        )

    def test_public_python_modules_are_annotated_inline(self):
        """Every public declaration carries its own type.

        This is the inline replacement for the removed "the stub covers this
        module" assertion, and it is strictly stronger: the stub test could
        only notice a public addition the stub did not mention, while this one
        also fails on a public addition that is present but untyped.
        """
        for package, modules in (
            (DRJIT_PACKAGE, DRJIT_PUBLIC_MODULES),
            (TORCH_PACKAGE, TORCH_PUBLIC_MODULES),
        ):
            for stem in modules:
                path = package / f"{stem}.py"
                with self.subTest(module=f"rayd.{package.name}.{stem}"):
                    self.assertEqual(
                        _unannotated_public_surface(path),
                        [],
                        f"{path.relative_to(ROOT).as_posix()} has untyped public "
                        "declarations; downstream type checkers would widen them to Any",
                    )

    def test_torch_top_level_reexports_match_runtime_all(self):
        tree = _module_tree(TORCH_PACKAGE / "__init__.py")
        statements = list(_statements(tree.body))
        _, declared = _declared_all(statements)
        resolved = _reexported_names(statements) | _lazily_bound_names(tree)
        self.assertEqual(
            declared,
            resolved,
            "rayd.torch.__all__ and the names the module actually re-exports "
            f"disagree; missing={sorted(declared - resolved)} "
            f"extra={sorted(resolved - declared)}",
        )

    def test_torch_top_level_exports_the_named_result_types(self):
        statements = _module_statements(TORCH_PACKAGE / "__init__.py")
        _, declared = _declared_all(statements)
        reexported = _reexported_names(statements)
        defined = {
            node.name
            for node in _module_statements(TORCH_PACKAGE / "types.py")
            if isinstance(node, ast.ClassDef) and _is_public(node.name)
        }
        for name in (
            "NearestEdgesTopK",
            "SegmentPairVisibility",
            "AxialEdgeVisibility",
            "SegmentChainVisibility",
        ):
            with self.subTest(result_type=name):
                self.assertIn(name, defined)
                self.assertIn(name, reexported)
                self.assertIn(name, declared)

    def test_drjit_top_level_all_matches_native_bindings(self):
        runtime = ast.parse((DRJIT_PACKAGE / "__init__.py").read_text(encoding="utf-8"))
        all_node = next(
            node for node in runtime.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        declared = {
            element.value for element in all_node.value.elts if isinstance(element, ast.Constant)
        }
        self.assertEqual(len(declared), len(all_node.value.elts))
        expected = _drjit_bound_names() | {"api_manifest", "backend_capabilities"}
        self.assertEqual(
            declared,
            expected,
            "rayd.drjit.__all__ is out of sync with backends/drjit/src/rayd.cpp; "
            f"missing={sorted(expected - declared)} stale={sorted(declared - expected)}",
        )

    def test_drjit_native_stub_covers_bound_public_symbols(self):
        stub_path = DRJIT_NATIVE_STUB
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        stub_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        stub_names.update(
            node.target.id
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        )
        bound_names = _drjit_bound_names()
        self.assertFalse(bound_names - stub_names, sorted(bound_names - stub_names))

    def test_drjit_key_classes_have_typed_members(self):
        stub_path = DRJIT_NATIVE_STUB
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }

        def members(name):
            node = classes[name]
            result = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            result.update(
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
            )
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    result.update(members(base.id))
            return result

        required = {
            "Mesh": {
                "vertex_positions", "face_indices", "to_world", "build",
                "set_transform", "append_transform", "secondary_edges",
            },
            "Scene": {
                "intersect", "nearest_edge", "nearest_edges", "set_edge_mask",
                "visible", "visible_pair", "visible_edge", "visible_chain",
                "trace_reflections", "trace_refl_epc_field", "trace_dfr_paths",
                "accumulate_reflections", "accum_dfr_direct", "accum_dfr",
            },
            "ReflectionTraceOptions": {
                "deduplicate", "canonical_prim_table", "export_mode", "return_trailing",
            },
            "DfrOptions": {
                "strategy_mask", "sample_sequence", "receiver_model", "max_order",
            },
            "Intersection": {"is_valid", "t", "p", "global_prim_id"},
            "NearestEdgesTopK": {"query_count", "k", "distances", "global_edge_ids"},
            "ReflectionChain": {"is_valid", "bounce_count", "global_prim_ids"},
        }
        for class_name, expected in required.items():
            with self.subTest(class_name=class_name):
                self.assertLessEqual(expected, members(class_name))


if __name__ == "__main__":
    unittest.main()
