"""Test: Slang code calls sceneIntersect on a real rayd scene."""
import json
import rayd as rd
import rayd.slang as rs
import drjit.cuda as cuda

# Load Slang module that calls rayd interop functions
m = rs.load_module(
    str(__import__("pathlib").Path(__file__).resolve().parent.parent / "shaders" / "rayd_query.slang"),
    link_rayd=True,
)

# Build a triangle scene
mesh = rd.Mesh(
    cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
    cuda.Array3i([0], [1], [2]),
)
scene = rd.Scene()
scene.add_mesh(mesh)
scene.build()
handle = scene.slang_handle

# Hit
t_hit = m.traceRayT(handle, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)
valid_hit = m.traceRayHit(handle, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)
shadow_hit = m.shadowTest(handle, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)

# Miss
t_miss = m.traceRayT(handle, 5.0, 5.0, -1.0, 0.0, 0.0, 1.0)
valid_miss = m.traceRayHit(handle, 5.0, 5.0, -1.0, 0.0, 0.0, 1.0)

print(json.dumps({
    "t_hit": t_hit,
    "valid_hit": valid_hit,
    "shadow_hit": shadow_hit,
    "t_miss_inf": t_miss == float("inf"),
    "valid_miss": valid_miss,
}))
