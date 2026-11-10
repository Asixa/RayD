# Copyright Xingyu Chen.
# Runs one invalid-input probe per backend in a fresh interpreter and classifies its outcome.

"""Backend-neutral validation probes.

Each backend gets a prelude that defines the same small adapter vocabulary (`V`, `F`, `scene_from`,
`intersect`, `nearest_point`, ...), so one case body feeds identical inputs to `rayd.torch` and `rayd.drjit`.
A body defines `case()`, which returns a JSON-able summary of what the backend answered. The epilogue
classifies the call as `accepted` (with that summary) or `rejected` (with the exception type and message),
then proves the CUDA context still works by running an ordinary query in the same process. The runner adds
the `crash` (non-zero exit or signal) and `timeout` classes.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import subprocess

from tests.support.subprocess_cases import compose, run_script


_COMMON = """
import json
import math

NAN = float("nan")
INF = float("inf")
TRIANGLE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
# Two triangles sharing edge (1, 2); a ray through (0.25, 0.25) hits face 0 and one through (0.75, 0.75) face 1.
QUAD = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
QUAD_FACES = [[0, 1, 2], [1, 3, 2]]


def message_of(error):
    return {"outcome": "rejected", "type": type(error).__name__, "message": str(error)[:600]}


def attempt(fn):
    # One sub-probe inside a case: {"ok": summary} or {"error": "Type: message"}.
    try:
        value = fn()
        finish()
        return {"ok": value}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:600]}
"""

TORCH_PRELUDE = (
    _COMMON
    + """
import gc
import torch
import rayd.torch as rt

DEVICE = "cuda:0"


def V(rows, device=None):
    return torch.tensor(rows, device=device or DEVICE, dtype=torch.float32).reshape(-1, 3)


def F(rows):
    return torch.tensor(rows, device=DEVICE, dtype=torch.int32).reshape(-1, 3)


def scene_from(meshes, build=True, dynamic=False, edges=True, trace_backend="auto", edge_backend="auto"):
    scene = rt.Scene(trace_backend=trace_backend, edge_bvh_backend=edge_backend)
    for vertices, faces in meshes:
        scene.add_mesh(rt.Mesh(V(vertices), F(faces), edges_enabled=edges), dynamic=dynamic)
    if build:
        scene.build()
    return scene


def make_ray(o, d, tmax=None):
    if tmax is None:
        return rt.Ray(V(o), V(d))
    return rt.Ray(V(o), V(d), torch.tensor(tmax, device=DEVICE, dtype=torch.float32))


def floats(x):
    return [float(v) for v in x.detach().reshape(-1).cpu().tolist()]


def ints(x):
    return [int(v) for v in x.reshape(-1).cpu().tolist()]


def intersect(scene, o, d, tmax=None):
    hit = scene.intersect(make_ray(o, d, tmax))
    return {"t": floats(hit.t), "prim": ints(hit.global_prim_id)}


def nearest_point(scene, points):
    edge = scene.nearest_edge(V(points))
    return {"distance": floats(edge.distance), "edge": ints(edge.global_edge_id)}


def nearest_ray(scene, o, d, tmax=None):
    edge = scene.nearest_edge(make_ray(o, d, tmax))
    return {"distance": floats(edge.distance), "edge": ints(edge.global_edge_id), "ray_t": floats(edge.ray_t)}


def nearest_k(scene, points, k):
    top = scene.nearest_edges(V(points), k)
    return {"k": int(top.k), "distances": floats(top.distances), "edges": ints(top.global_edge_ids)}


def visible(scene, start, end):
    return [bool(v) for v in scene.visible(V(start), V(end)).reshape(-1).cpu().tolist()]


def reflections(scene, o, d, bounces):
    chain = scene.trace_reflections(make_ray(o, d), bounces)
    return {"bounce_count": ints(chain.valid.sum(dim=1)), "t0": floats(chain.t[:, 0])}


def update(scene, mesh_id, vertices):
    scene.update_mesh_vertices(mesh_id, V(vertices))


def set_mask(scene, mask):
    scene.set_edge_mask(torch.tensor(mask, device=DEVICE, dtype=torch.bool))


def finish():
    torch.cuda.synchronize()


def healthy():
    try:
        scene = scene_from([(TRIANGLE, [[0, 1, 2]])])
        t = float(scene.intersect(make_ray([[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])).t[0])
        torch.cuda.synchronize()
        return t
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:400]
"""
)

DRJIT_PRELUDE = (
    _COMMON
    + """
import gc
import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import rayd.drjit as pj


def V(rows):
    rows = [list(map(float, row)) for row in rows]
    return cuda.Array3f([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


def F(rows):
    rows = [list(map(int, row)) for row in rows]
    return cuda.Array3i([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


def scene_from(meshes, build=True, dynamic=False, edges=True, trace_backend="auto", edge_backend="auto"):
    scene = pj.Scene(edge_bvh_backend=edge_backend, trace_backend=trace_backend)
    for vertices, faces in meshes:
        mesh = pj.Mesh(V(vertices), F(faces))
        mesh.edges_enabled = edges
        scene.add_mesh(mesh, dynamic=dynamic)
    if build:
        scene.build()
    return scene


def make_ray(o, d, tmax=None):
    ray = pj.Ray(V(o), V(d))
    if tmax is not None:
        ray.tmax = cuda.Float(list(map(float, tmax)))
    return ray


def floats(x):
    dr.eval(x)
    return [float(v) for v in x]


def ints(x):
    dr.eval(x)
    return [int(v) for v in x]


def intersect(scene, o, d, tmax=None):
    hit = scene.intersect(make_ray(o, d, tmax))
    return {"t": floats(hit.t), "prim": ints(hit.global_prim_id)}


def nearest_point(scene, points):
    edge = scene.nearest_edge(V(points))
    return {"distance": floats(edge.distance), "edge": ints(edge.global_edge_id)}


def nearest_ray(scene, o, d, tmax=None):
    edge = scene.nearest_edge(make_ray(o, d, tmax))
    return {"distance": floats(edge.distance), "edge": ints(edge.global_edge_id), "ray_t": floats(edge.ray_t)}


def nearest_k(scene, points, k):
    top = scene.nearest_edges(V(points), k)
    return {"k": int(top.k), "distances": floats(top.distances), "edges": ints(top.global_edge_ids)}


def visible(scene, start, end):
    result = scene.visible(V(start), V(end))
    dr.eval(result.visible)
    return [bool(v) for v in result.visible]


def reflections(scene, o, d, bounces):
    chain = scene.trace_reflections(make_ray(o, d), bounces)
    # The symbolic trace keeps one record per bounce; an empty batch has none.
    first = floats(chain.bounces[0].t) if len(chain.bounces) else []
    return {"bounce_count": ints(chain.bounce_count), "t0": first}


def update(scene, mesh_id, vertices):
    scene.update_mesh_vertices(mesh_id, ad.Array3f(V(vertices)))


def set_mask(scene, mask):
    scene.set_edge_mask(cuda.Bool(list(map(bool, mask))))


def finish():
    dr.sync_thread()


def healthy():
    try:
        scene = scene_from([(TRIANGLE, [[0, 1, 2]])])
        t = float(scene.intersect(make_ray([[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])).t[0])
        dr.sync_thread()
        return t
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:400]
"""
)

PRELUDES = {"torch": TORCH_PRELUDE, "drjit": DRJIT_PRELUDE}

_EPILOGUE = """
try:
    _value = case()
    finish()
    _result = {"outcome": "accepted", "value": _value}
except Exception as _error:
    _result = message_of(_error)
_result["healthy"] = healthy()
print(json.dumps(_result))
"""


def run_backend_case(
    backend: str, body: str, timeout: float = 240.0, env: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Run `body` (which defines `case()`) on `backend` in a fresh interpreter and classify the outcome."""
    script = compose(PRELUDES[backend], body, _EPILOGUE)
    try:
        result = run_script(script, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout", "timeout": timeout}
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return {"outcome": "crash", "returncode": result.returncode, "stderr": result.stderr[-1500:]}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"outcome": "crash", "returncode": result.returncode, "stdout": result.stdout[-1500:]}


def outcome_summary(record: Mapping[str, object], width: int = 110) -> str:
    """One table cell: the outcome class plus the exception or returned summary, truncated to `width`."""
    outcome = record.get("outcome")
    if outcome == "rejected":
        text = f"rejected {record.get('type')}: {record.get('message')}"
    elif outcome == "accepted":
        text = f"accepted {json.dumps(record.get('value'))}"
    elif outcome == "crash":
        text = f"crash rc={record.get('returncode')} {str(record.get('stderr', ''))[-200:]}"
    else:
        text = str(outcome)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."
