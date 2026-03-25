"""Verify dt/doz from Dr.Jit AD (not FD) via Slang→rayd interop.

Triangle at z=0, ray at (0.25, 0.25, -1) going +z.
t = -oz = 1,  dt/doz = -1  (analytically).
"""
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
scene = rd.Scene(); scene.add_mesh(mesh); scene.configure()
H = scene.slang_handle

t   = M.traceT(H, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)
g_z = M.dtdoz(H, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)

print(json.dumps({
    "t": t,
    "dtdoz": g_z,
    "t_correct": abs(t - 1.0) < 1e-5,
    "grad_correct": abs(g_z - (-1.0)) < 1e-5,
}))
