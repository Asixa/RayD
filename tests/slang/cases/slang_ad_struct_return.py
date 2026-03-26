"""Test: m.traceAD() returns IntersectionAD struct with forward + AD gradients."""
import json
import rayd as rd, rayd.slang as rs
import drjit.cuda as cuda
from pathlib import Path

SHADER = str(Path(__file__).resolve().parent.parent / "shaders" / "rayd_query_ad.slang")
M = rs.load_module(SHADER, link_rayd=True)

mesh = rd.Mesh(
    cuda.Array3f([0, 1, 0], [0, 0, 1], [0, 0, 0]),
    cuda.Array3i([0], [1], [2]),
)
scene = rd.Scene(); scene.add_mesh(mesh); scene.build()

r = M.traceAD(scene.slang_handle, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)

tol = 1e-4
print(json.dumps({
    "t": r.t,
    "valid": r.valid,
    "p_z": r.p.z,
    "dt_do_z": r.dt_do.z,
    "dt_dd_z": r.dt_dd.z,
    "t_correct": abs(r.t - 1.0) < tol,
    "dt_do_correct": abs(r.dt_do.x) < tol and abs(r.dt_do.y) < tol and abs(r.dt_do.z - (-1.0)) < tol,
    "dt_dd_correct": abs(r.dt_dd.x) < tol and abs(r.dt_dd.y) < tol and abs(r.dt_dd.z - (-1.0)) < tol,
}))
