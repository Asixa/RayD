import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, timeout: int = 300, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
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


def run_json_case(script: str, timeout: int = 300):
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


def compose(*parts: str) -> str:
    """Join independently indented script fragments into one flat script."""
    return "\n".join(textwrap.dedent(part).strip("\n") for part in parts)


def drjit_device_count() -> int:
    """Dr.Jit CUDA device count, or 0 when rayd.drjit cannot be imported."""
    result = run_script(
        """
        import json

        try:
            import rayd.drjit as pj

            devices = int(pj.device_count())
        except Exception:
            devices = 0
        print(json.dumps({"devices": devices}))
        """,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return 0
    try:
        return int(json.loads(lines[-1])["devices"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0


IMPORTS = """
    import json
    import rayd.drjit as pj
    import drjit.cuda as cuda
"""

# One triangle in the z=0 plane: the probe ray hits it at t=1 and the probe
# point sits a quarter unit from the edge between (0,0,0) and (1,0,0).
SCENE = """
    mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0]),
                   cuda.Array3i([0], [1], [2]))
    scene = pj.Scene()
    scene.add_mesh(mesh)
    scene.build()
    ray = pj.Ray(cuda.Array3f([0.25], [0.25], [-1.0]),
                 cuda.Array3f([0.0], [0.0], [1.0]))
    point = cuda.Array3f([0.5], [-0.25], [0.0])
"""


class DeviceBindingTests(unittest.TestCase):
    """A scene is bound to the device that ran build(); queries elsewhere fail."""

    @classmethod
    def setUpClass(cls) -> None:
        if drjit_device_count() < 2:
            raise unittest.SkipTest(
                "two Dr.Jit CUDA devices and an importable rayd.drjit are required"
            )

    def test_intersect_after_set_device_raises_and_restores(self):
        data = run_json_case(
            compose(
                IMPORTS,
                SCENE,
                """
                baseline = float(scene.intersect(ray).t[0])

                pj.set_device(1)
                error = ""
                try:
                    scene.intersect(pj.Ray(cuda.Array3f([0.25], [0.25], [-1.0]),
                                           cuda.Array3f([0.0], [0.0], [1.0])))
                except Exception as exc:
                    error = str(exc)

                pj.set_device(0)
                restored = float(scene.intersect(ray).t[0])

                print(json.dumps({
                    "baseline": baseline,
                    "error": error,
                    "restored": restored,
                }))
                """,
            )
        )

        self.assertAlmostEqual(data["baseline"], 1.0, places=5)
        self.assertIn("Dr.Jit CUDA device", data["error"])
        self.assertIn("device 0", data["error"])
        self.assertIn("rayd.drjit.set_device(0)", data["error"])
        self.assertAlmostEqual(data["restored"], 1.0, places=5)

    def test_nearest_edge_after_set_device_raises_and_restores(self):
        data = run_json_case(
            compose(
                IMPORTS,
                SCENE,
                """
                baseline = float(scene.nearest_edge(point).distance[0])

                pj.set_device(1)
                error = ""
                try:
                    scene.nearest_edge(cuda.Array3f([0.5], [-0.25], [0.0]))
                except Exception as exc:
                    error = str(exc)

                pj.set_device(0)
                restored = float(scene.nearest_edge(point).distance[0])

                print(json.dumps({
                    "baseline": baseline,
                    "error": error,
                    "restored": restored,
                }))
                """,
            )
        )

        self.assertAlmostEqual(data["baseline"], 0.25, places=5)
        self.assertIn("Dr.Jit CUDA device", data["error"])
        self.assertIn("rayd.drjit.set_device(0)", data["error"])
        self.assertAlmostEqual(data["restored"], 0.25, places=5)

    def test_scene_built_on_the_second_device_answers_on_that_device(self):
        data = run_json_case(
            compose(
                IMPORTS,
                "pj.set_device(1)",
                SCENE,
                """
                its = scene.intersect(ray)

                print(json.dumps({
                    "current": int(pj.current_device()),
                    "valid": bool(its.is_valid()[0]),
                    "t": float(its.t[0]),
                }))
                """,
            )
        )

        self.assertEqual(data["current"], 1)
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)

    def test_second_device_scene_rejects_queries_from_the_first_device(self):
        data = run_json_case(
            compose(
                IMPORTS,
                "pj.set_device(1)",
                SCENE,
                """
                baseline = float(scene.intersect(ray).t[0])

                pj.set_device(0)
                error = ""
                try:
                    scene.intersect(pj.Ray(cuda.Array3f([0.25], [0.25], [-1.0]),
                                           cuda.Array3f([0.0], [0.0], [1.0])))
                except Exception as exc:
                    error = str(exc)

                print(json.dumps({"baseline": baseline, "error": error}))
                """,
            )
        )

        self.assertAlmostEqual(data["baseline"], 1.0, places=5)
        self.assertIn("Dr.Jit CUDA device", data["error"])
        self.assertIn("rayd.drjit.set_device(1)", data["error"])


if __name__ == "__main__":
    unittest.main()
