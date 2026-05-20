import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


RAYD_LOAD_PREAMBLE = f"""
import importlib.util as _rayd_importlib_util
import os as _rayd_os
import pathlib as _rayd_pathlib
import sys as _rayd_sys
import types as _rayd_types
import drjit as _rayd_drjit

_rayd_root = _rayd_pathlib.Path({str(ROOT)!r})
_rayd_candidates = list((_rayd_root / "rayd").glob("rayd*.pyd"))
_rayd_candidates += list((_rayd_root / "rayd").glob("rayd*.so"))
for _rayd_path_entry in list(_rayd_sys.path):
    try:
        _rayd_base = _rayd_pathlib.Path(_rayd_path_entry)
    except Exception:
        continue
    _rayd_candidates += list((_rayd_base / "rayd").glob("rayd*.pyd"))
    _rayd_candidates += list((_rayd_base / "rayd").glob("rayd*.so"))
if not _rayd_candidates:
    raise RuntimeError("Could not find a rayd extension module for tests.")
_rayd_ext = _rayd_candidates[0]
if _rayd_sys.platform == "win32" and hasattr(_rayd_os, "add_dll_directory"):
    _rayd_os.add_dll_directory(str(_rayd_pathlib.Path(_rayd_drjit.__file__).resolve().parent))
    _rayd_os.add_dll_directory(str(_rayd_ext.parent))
    if (_rayd_ext.parent / "lib").exists():
        _rayd_os.add_dll_directory(str(_rayd_ext.parent / "lib"))
_rayd_spec = _rayd_importlib_util.spec_from_file_location("rayd.rayd", _rayd_ext)
if _rayd_spec is None or _rayd_spec.loader is None:
    raise RuntimeError(f"Could not load rayd extension spec from {{_rayd_ext}}")
_rayd_ext_mod = _rayd_importlib_util.module_from_spec(_rayd_spec)
_rayd_pkg = _rayd_types.ModuleType("rayd")
_rayd_pkg.__file__ = str(_rayd_root / "rayd" / "__init__.py")
_rayd_pkg.__path__ = [str(_rayd_root / "rayd"), str(_rayd_ext.parent)]
_rayd_pkg.__package__ = "rayd"
_rayd_sys.modules["rayd"] = _rayd_pkg
_rayd_sys.modules["rayd.rayd"] = _rayd_ext_mod
_rayd_spec.loader.exec_module(_rayd_ext_mod)
_rayd_pkg.rayd = _rayd_ext_mod
for _rayd_name, _rayd_value in _rayd_ext_mod.__dict__.items():
    if _rayd_name.startswith("__") and _rayd_name not in {{"__doc__", "__name__"}}:
        continue
    setattr(_rayd_pkg, _rayd_name, _rayd_value)
"""


def run_script(script: str, timeout: int = 180, check: bool = True):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", RAYD_LOAD_PREAMBLE + "\n" + textwrap.dedent(script)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "Subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def run_json_case(script: str, timeout: int = 180):
    result = run_script(script, timeout=timeout, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON output.\nSTDERR:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Failed to parse JSON from subprocess.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc


class VisibilityAndTopKTests(unittest.TestCase):
    def test_segment_visibility_ignore_pair_and_axial(self):
        data = run_json_case(
            """
            import json
            import rayd as rd
            import drjit as dr
            import drjit.cuda as cuda

            mesh = rd.Mesh(
                cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            )
            scene = rd.Scene()
            scene.add_mesh(mesh)
            scene.build()

            start = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [-1.0, -1.0, 0.1])
            end = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.1])
            ignore = cuda.Int([-1, 0, -1])
            active = cuda.Bool([True, True, False])

            vis = scene.trace_segment_visibility(start, end, ignore, active)
            pair = scene.trace_segment_pair_visibility(
                start,
                end,
                cuda.Array3f([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),
                ignore,
                active,
            )
            axial = scene.trace_axial_edge_visibility(
                cuda.Array3f([0.0], [0.0], [-1.0]),
                cuda.Array3f([-2.0], [0.0], [1.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Float([0.0]),
                cuda.Float([4.0]),
                [0.0, 0.5, 1.0],
                cuda.Bool([True]),
            )

            rd.native_launch_audit_clear()
            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                lazy = scene.trace_segment_visibility(start, end, ignore, active)
                lazy_hist = dr.kernel_history()
                values = [bool(v) for v in list(lazy.visible)]
                consume_hist = dr.kernel_history()
            native_audit = rd.native_launch_audit()

            print(json.dumps({
                "visible": [bool(v) for v in list(vis.visible)],
                "pair_a": [bool(v) for v in list(pair.visible_a)],
                "pair_b": [bool(v) for v in list(pair.visible_b)],
                "axial_any": [bool(v) for v in list(axial.any_visible)],
                "lazy_optix_before_consume": sum(1 for h in lazy_hist if bool(h.get("uses_optix", False))),
                "consume_values": values,
                "consume_optix": sum(1 for h in consume_hist if bool(h.get("uses_optix", False))),
                "native_optix_launches": int(native_audit["unknown"]["optix_launch"]),
            }))
            """
        )

        self.assertEqual(data["visible"], [False, True, False])
        self.assertEqual(data["pair_a"], [False, True, False])
        self.assertEqual(data["pair_b"], [True, True, False])
        self.assertEqual(data["axial_any"], [True])
        self.assertEqual(data["lazy_optix_before_consume"], 0)
        self.assertEqual(data["consume_values"], [False, True, False])
        self.assertEqual(data["consume_optix"], 0)
        self.assertGreaterEqual(data["native_optix_launches"], 1)

    def test_no_ignore_segment_visibility_variants_are_lazy_jit_optix_queries(self):
        data = run_json_case(
            """
            import json
            import rayd as rd
            import drjit as dr
            import drjit.cuda as cuda

            mesh = rd.Mesh(
                cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            )
            scene = rd.Scene()
            scene.add_mesh(mesh)
            scene.build()

            start = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [-1.0, -1.0, 0.1])
            end = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.1])
            end_b = cuda.Array3f([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
            active = cuda.Bool([True, True, False])

            chain_points = cuda.Array3f(
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [-1.0, 1.0, -1.0, 1.0],
            )
            chain_length = cuda.Int([1, 1])

            rd.native_launch_audit_clear()
            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                vis = scene.trace_segment_visibility(start, end, active=active)
                pair = scene.trace_segment_pair_visibility(start, end, end_b, active=active)
                axial = scene.trace_axial_edge_visibility(
                    cuda.Array3f([0.0], [0.0], [-1.0]),
                    cuda.Array3f([-2.0], [0.0], [1.0]),
                    cuda.Array3f([1.0], [0.0], [0.0]),
                    cuda.Float([0.0]),
                    cuda.Float([4.0]),
                    [0.0, 0.5, 1.0],
                    cuda.Bool([True]),
                )
                chain = scene.trace_segment_chain_visibility(chain_points, chain_length)
                call_hist = dr.kernel_history()

                values = {
                    "visible": [bool(v) for v in list(vis.visible)],
                    "pair_a": [bool(v) for v in list(pair.visible_a)],
                    "pair_b": [bool(v) for v in list(pair.visible_b)],
                    "axial_any": [bool(v) for v in list(axial.any_visible)],
                    "chain_all": [bool(v) for v in list(chain.all_visible)],
                    "chain_blocked_segment": [int(v) for v in list(chain.first_blocked_segment)],
                    "chain_blocked_prim": [int(v) for v in list(chain.first_blocked_prim)],
                }
                consume_hist = dr.kernel_history()
            native_audit = rd.native_launch_audit()

            print(json.dumps({
                **values,
                "call_jit": sum(1 for h in call_hist if str(h.get("type")) == "KernelType.JIT"),
                "call_optix": sum(1 for h in call_hist if bool(h.get("uses_optix", False))),
                "consume_jit": sum(1 for h in consume_hist if str(h.get("type")) == "KernelType.JIT"),
                "consume_optix": sum(1 for h in consume_hist if bool(h.get("uses_optix", False))),
                "native_optix_launches": int(native_audit["unknown"]["optix_launch"]),
            }))
            """
        )

        self.assertEqual(data["visible"], [False, True, False])
        self.assertEqual(data["pair_a"], [False, True, False])
        self.assertEqual(data["pair_b"], [True, True, False])
        self.assertEqual(data["axial_any"], [True])
        self.assertEqual(data["chain_all"], [False, True])
        self.assertEqual(data["chain_blocked_segment"], [0, -1])
        self.assertGreaterEqual(data["chain_blocked_prim"][0], 0)
        self.assertEqual(data["chain_blocked_prim"][1], -1)
        self.assertEqual(data["call_jit"], 0)
        self.assertEqual(data["call_optix"], 0)
        self.assertGreaterEqual(data["consume_jit"], 1)
        self.assertGreaterEqual(data["consume_optix"], 1)
        self.assertEqual(data["native_optix_launches"], 0)

    def test_trace_visibility_native_backend_keeps_optixlaunch_path_available(self):
        data = run_json_case(
            """
            import os
            os.environ["RAYD_TRACE_VISIBILITY_BACKEND"] = "native"

            import json
            import rayd as rd
            import drjit.cuda as cuda

            mesh = rd.Mesh(
                cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            )
            scene = rd.Scene()
            scene.add_mesh(mesh)
            scene.build()

            start = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [-1.0, -1.0, 0.1])
            end = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.1])
            end_b = cuda.Array3f([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
            active = cuda.Bool([True, True, False])

            chain_points = cuda.Array3f(
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [-1.0, 1.0, -1.0, 1.0],
            )
            chain_length = cuda.Int([1, 1])

            rd.native_launch_audit_clear()
            vis = scene.trace_segment_visibility(start, end, active=active)
            pair = scene.trace_segment_pair_visibility(start, end, end_b, active=active)
            axial = scene.trace_axial_edge_visibility(
                cuda.Array3f([0.0], [0.0], [-1.0]),
                cuda.Array3f([-2.0], [0.0], [1.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
                cuda.Float([0.0]),
                cuda.Float([4.0]),
                [0.0, 0.5, 1.0],
                cuda.Bool([True]),
            )
            chain = scene.trace_segment_chain_visibility(chain_points, chain_length)
            native_audit = rd.native_launch_audit()

            print(json.dumps({
                "visible": [bool(v) for v in list(vis.visible)],
                "pair_a": [bool(v) for v in list(pair.visible_a)],
                "pair_b": [bool(v) for v in list(pair.visible_b)],
                "axial_any": [bool(v) for v in list(axial.any_visible)],
                "chain_all": [bool(v) for v in list(chain.all_visible)],
                "native_optix_launches": int(native_audit["unknown"]["optix_launch"]),
            }))
            """
        )

        self.assertEqual(data["visible"], [False, True, False])
        self.assertEqual(data["pair_a"], [False, True, False])
        self.assertEqual(data["pair_b"], [True, True, False])
        self.assertEqual(data["axial_any"], [True])
        self.assertEqual(data["chain_all"], [False, True])
        self.assertGreaterEqual(data["native_optix_launches"], 4)

    def test_segment_chain_visibility_reports_first_blocker_and_uses_segment_ignores(self):
        data = run_json_case(
            """
            import json
            import rayd as rd
            import drjit.cuda as cuda

            mesh = rd.Mesh(
                cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]),
                cuda.Array3i([0], [1], [2]),
            )
            scene = rd.Scene()
            scene.add_mesh(mesh)
            scene.build()

            points = cuda.Array3f(
                [0.0, 0.0, 2.0, -2.0, -1.0, 0.0],
                [0.0, 0.0, 0.0,  0.0,  0.0, 0.0],
                [-1.0, 1.0, 1.0, 1.0,  1.0, 1.0],
            )
            chain_length = cuda.Int([2, 2])

            blocked = scene.trace_segment_chain_visibility(points, chain_length)
            ignore = cuda.Int([0, -1, -1, -1])
            ignored = scene.trace_segment_chain_visibility(points, chain_length, ignore)

            print(json.dumps({
                "chain_count": int(blocked.chain_count),
                "max_segments": int(blocked.max_segments),
                "visible": [bool(v) for v in list(blocked.all_visible)],
                "first_segment": [int(v) for v in list(blocked.first_blocked_segment)],
                "first_prim": [int(v) for v in list(blocked.first_blocked_prim)],
                "ignored_visible": [bool(v) for v in list(ignored.all_visible)],
                "ignored_first_segment": [int(v) for v in list(ignored.first_blocked_segment)],
                "ignored_first_prim": [int(v) for v in list(ignored.first_blocked_prim)],
            }))
            """
        )

        self.assertEqual(data["chain_count"], 2)
        self.assertEqual(data["max_segments"], 2)
        self.assertEqual(data["visible"], [False, True])
        self.assertEqual(data["first_segment"], [0, -1])
        self.assertEqual(data["first_prim"], [0, -1])
        self.assertEqual(data["ignored_visible"], [True, True])
        self.assertEqual(data["ignored_first_segment"], [-1, -1])
        self.assertEqual(data["ignored_first_prim"], [-1, -1])

    def test_nearest_edges_topk_point_k2(self):
        data = run_json_case(
            """
            import json
            import rayd as rd
            import drjit.cuda as cuda

            mesh = rd.Mesh(
                cuda.Array3f([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
                cuda.Array3i([0, 0], [1, 2], [2, 3]),
            )
            scene = rd.Scene()
            scene.add_mesh(mesh)
            scene.build()

            query = cuda.Array3f([0.5, 0.5], [-0.2, 0.5], [0.0, 0.0])
            active = cuda.Bool([True, False])
            result = scene.nearest_edges_topk(query, 2, active)

            print(json.dumps({
                "query_count": int(result.query_count),
                "k": int(result.k),
                "valid": [bool(v) for v in list(result.is_valid)],
                "global_edge_ids": [int(v) for v in list(result.global_edge_ids)],
                "distances": [float(v) for v in list(result.distances)],
            }))
            """
        )

        self.assertEqual(data["query_count"], 2)
        self.assertEqual(data["k"], 2)
        self.assertEqual(data["valid"][2:], [False, False])
        self.assertTrue(all(edge_id >= 0 for edge_id in data["global_edge_ids"][:2]))
        self.assertLessEqual(data["distances"][0], data["distances"][1])


if __name__ == "__main__":
    unittest.main()
