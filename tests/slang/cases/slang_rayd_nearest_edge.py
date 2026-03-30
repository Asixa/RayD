"""Test: Slang nearest-edge interop exposes global IDs and scene edge masks."""
import ctypes
import json
from pathlib import Path

import drjit.cuda as cuda
import rayd as rd
import rayd.slang as rs


SHADER = str(Path(__file__).resolve().parent.parent / "shaders" / "rayd_query.slang")
M = rs.load_module(SHADER, link_rayd=True)

mesh = rd.Mesh(
    cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
    cuda.Array3i([0], [1], [2]),
)
scene = rd.Scene()
scene.add_mesh(mesh)
scene.build()

point_query = cuda.Array3f([0.25], [0.1], [0.0])
ray_query = rd.RayDetached(
    cuda.Array3f([0.25], [-0.5], [0.0]),
    cuda.Array3f([0.0], [1.0], [0.0]),
)
ray_query.tmax = cuda.Float([1.0])

native_point = scene.nearest_edge(point_query)
native_ray = scene.nearest_edge(ray_query)
expected_point_global = int(native_point.global_edge_id[0])
expected_ray_global = int(native_ray.global_edge_id[0])

handle = scene.slang_handle
point_struct = M.nearestPointEdgeStruct(handle, 0.25, 0.1, 0.0)
ray_struct = M.nearestRayEdgeStruct(handle, 0.25, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0)
edge_count = M.getSceneEdgeCount(handle)
default_mask = [bool(M.getSceneEdgeMaskValue(handle, i)) for i in range(edge_count)]

all_false_mask = (ctypes.c_uint8 * edge_count)(*([0] * edge_count))
M.setSceneEdgeMask(handle, ctypes.addressof(all_false_mask), edge_count)
pending_after_false_mask = bool(scene.has_pending_updates())
M.syncScene(handle)
invalid_after_false_mask = M.nearestPointEdgeStruct(handle, 0.25, 0.1, 0.0)
mask_after_false = [bool(M.getSceneEdgeMaskValue(handle, i)) for i in range(edge_count)]

single_mask_values = [0] * edge_count
single_mask_values[expected_point_global] = 1
single_mask = (ctypes.c_uint8 * edge_count)(*single_mask_values)
M.setSceneEdgeMask(handle, ctypes.addressof(single_mask), edge_count)
M.syncScene(handle)
masked_point = M.nearestPointEdgeStruct(handle, 0.25, 0.1, 0.0)
mask_after_single = [bool(M.getSceneEdgeMaskValue(handle, i)) for i in range(edge_count)]

print(json.dumps({
    "point_valid": bool(point_struct.valid),
    "point_global_edge_id": int(point_struct.global_edge_id),
    "point_matches_native": int(point_struct.global_edge_id) == expected_point_global,
    "ray_valid": bool(ray_struct.valid),
    "ray_global_edge_id": int(ray_struct.global_edge_id),
    "ray_matches_native": int(ray_struct.global_edge_id) == expected_ray_global,
    "edge_count": int(edge_count),
    "default_mask": default_mask,
    "pending_after_false_mask": pending_after_false_mask,
    "invalid_after_false_mask": bool(invalid_after_false_mask.valid),
    "invalid_global_after_false_mask": int(invalid_after_false_mask.global_edge_id),
    "mask_after_false": mask_after_false,
    "masked_point_valid": bool(masked_point.valid),
    "masked_point_global_edge_id": int(masked_point.global_edge_id),
    "mask_after_single": mask_after_single,
}))
