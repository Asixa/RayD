"""Verify all 6 gradient components of dt/d(ray) via Dr.Jit AD.

Triangle vertices: (0,0,0), (1,0,0), (0,1,0)  — lies in z=0 plane.
Ray: origin=(0.25, 0.25, -1), direction=(0, 0, 1).

Analytically for a z-directed ray hitting a z=0 triangle:
  t = -oz / dz = 1.0
  dt/dox = 0, dt/doy = 0, dt/doz = -1/dz = -1
  dt/ddx = 0, dt/ddy = 0, dt/ddz = oz/dz^2 = -1
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

ox, oy, oz = 0.25, 0.25, -1.0
dx, dy, dz = 0.0, 0.0, 1.0

g = {
    "dtdox": M.dtdox(H, ox, oy, oz, dx, dy, dz),
    "dtdoy": M.dtdoy(H, ox, oy, oz, dx, dy, dz),
    "dtdoz": M.dtdoz(H, ox, oy, oz, dx, dy, dz),
    "dtddx": M.dtddx(H, ox, oy, oz, dx, dy, dz),
    "dtddy": M.dtddy(H, ox, oy, oz, dx, dy, dz),
    "dtddz": M.dtddz(H, ox, oy, oz, dx, dy, dz),
}

tol = 1e-4
print(json.dumps({
    **g,
    "origin_ok": abs(g["dtdox"]) < tol and abs(g["dtdoy"]) < tol and abs(g["dtdoz"] - (-1.0)) < tol,
    "dir_ok": abs(g["dtddx"]) < tol and abs(g["dtddy"]) < tol and abs(g["dtddz"] - (-1.0)) < tol,
}))
