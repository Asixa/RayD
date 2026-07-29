# Copyright Xingyu Chen.
# Exercises cuda multipath Dr.Jit in a native smoke test.

"""P4 Stage D gate: the CUDA fused multipath executor matches OptiX.

Every multipath pipeline the CUDA fused executor serves is run under both
``trace_backend='optix'`` and ``trace_backend='cuda'`` in fresh subprocesses
(so Dr.Jit/CUDA state never leaks between the two), and the results are compared:

* Discrete fields (visibility bools, blocker / prim / group ids, bounce and path
  counts, path topology) are BIT-IDENTICAL cuda vs optix.
* Continuous fields (t, hit points, path lengths, fields, grid power) match
  within the ``contracts/operations.json`` tolerances.

Pipelines covered: reflection trace (``trace_reflections`` symbolic=False),
segment visibility (``visible`` / ``visible_pair`` / ``visible_edge`` /
``visible_chain``), reflection EPC (``trace_refl_epc`` primitive + surface-group,
``trace_refl_epc_field``), reflection accumulation (``accumulate_reflections``),
and first-order diffraction path export (``trace_dfr_paths``).
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# operations.json tolerances (default 1e-5, field 5e-5).
DEFAULT_ABS = 1e-5
DEFAULT_REL = 1e-5
FIELD_ABS = 5e-5
FIELD_REL = 5e-5

SCENE_SCRIPT = r'''
import json, math, sys
import numpy as np
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as rd

TB = sys.argv[1]

def build():
    floor = rd.Mesh(
        cuda.Array3f([-5.0, 5.0, 5.0, -5.0], [-5.0, -5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]),
        cuda.Array3i([0, 0], [1, 2], [2, 3]))
    wall = rd.Mesh(
        cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [1.0, 1.0, 1.5]),
        cuda.Array3i([0], [1], [2]))
    scene = rd.Scene(trace_backend=TB)
    scene.add_mesh(floor); scene.add_mesh(wall); scene.build()
    return scene

def flt(a):
    return [float(x) for x in a.numpy().tolist()]
def i(a):
    return [int(x) for x in a.numpy().tolist()]
def b(a):
    return [bool(x) for x in a.numpy().tolist()]

out = {}
scene = build()

# reflection trace (symbolic=False -> native pipeline / CUDA fused arm)
o = cuda.Array3f([-3.0, 0.0, 2.0], [0.0, 1.0, -2.0], [3.0, 3.0, 3.0])
d = cuda.Array3f([1.0, 0.0, 0.5], [0.0, 0.2, 0.6], [-1.0, -1.0, -1.0])
d = d / dr.norm(d)
refl = scene.trace_reflections(rd.Ray(o, d), 3, symbolic=False)
out["refl_bounce_count"] = i(refl.bounce_count)
out["refl_shape_ids"] = i(refl.shape_ids)
out["refl_global_prim_ids"] = i(refl.global_prim_ids)
out["refl_t"] = flt(refl.t)
out["refl_hit"] = flt(dr.ravel(refl.hit_points))
out["refl_img"] = flt(dr.ravel(refl.image_sources))

# segment visibility family
start = cuda.Array3f([0.0, 0.0, -2.0], [0.0, 2.0, 0.0], [3.0, 3.0, 3.0])
end = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [-1.0, -1.0, -1.0])
ignore = cuda.Int([-1, -1, -1]); active = cuda.Bool([True, True, True])
out["visible"] = b(scene.visible(start, end, ignore, active).visible)
pair = scene.visible_pair(start, end,
    cuda.Array3f([3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]), ignore, active)
out["visible_a"] = b(pair.visible_a); out["visible_b"] = b(pair.visible_b)
edge = scene.visible_edge(cuda.Array3f([0.0], [0.0], [3.0]), cuda.Array3f([-2.0], [0.0], [0.0]),
    cuda.Array3f([1.0], [0.0], [0.0]), cuda.Float([0.0]), cuda.Float([4.0]),
    [0.0, 0.25, 0.5, 0.75, 1.0], cuda.Bool([True]))
out["edge_visible"] = b(edge.any_visible)
pts = cuda.Array3f([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                   [3.0, 1.0, -1.0, 3.0, 3.0, 3.0])
chain = scene.visible_chain(pts, cuda.Int([2, 2]), cuda.Int([]), cuda.Bool([True, True]))
out["chain_all_visible"] = b(chain.all_visible)
out["chain_first_blocked_segment"] = i(chain.first_blocked_segment)
out["chain_first_blocked_prim"] = i(chain.first_blocked_prim)

# accumulate reflections
n = 64
rng = np.random.default_rng(7)
th = rng.uniform(0.2, 0.9, n); ph = rng.uniform(0, 2 * math.pi, n)
aray = rd.Ray(
    cuda.Array3f([0.0] * n, [0.0] * n, [3.0] * n),
    cuda.Array3f((np.sin(th) * np.cos(ph)).astype(np.float32).tolist(),
                 (np.sin(th) * np.sin(ph)).astype(np.float32).tolist(),
                 (-np.cos(th)).astype(np.float32).tolist()))
mat = rd.Material()
mat.eta_r = cuda.Float([5.0, 5.0, 5.0]); mat.sigma = cuda.Float([0.01, 0.01, 0.01])
mat.gain = cuda.Float([1.0, 1.0, 1.0]); mat.mu_r = cuda.Float([1.0, 1.0, 1.0])
mat.valid = cuda.Bool([True, True, True])
grid = rd.AccumGrid()
grid.axis = 2; grid.position = 2.0
grid.coord0_min = -5.0; grid.coord0_max = 5.0; grid.coord1_min = -5.0; grid.coord1_max = 5.0
grid.resolution0 = 8; grid.resolution1 = 8
opts = rd.AccumOptions()
opts.wavelength = 0.1; opts.k = 0.5; opts.solid_angle_per_ray = 1.0; opts.cell_area = 1.0
opts.seed = 17; opts.rr_depth = 0; opts.rr_prob = 1.0; opts.stop_threshold = 0.0
acc = scene.accumulate_reflections(aray, cuda.Array3f([0.0], [0.0], [3.0]), grid, mat, 2, opts,
                                   cuda.Bool([True] * n), cuda.Array3f([1.0], [0.0], [0.0]))
out["acc_count"] = int(acc.reflection_count.numpy().tolist()[0])
out["acc_power"] = flt(acc.reflection_power)
out["acc_field_x_re"] = flt(acc.reflection_field_x.real)
out["acc_field_x_im"] = flt(acc.reflection_field_x.imag)

print(json.dumps(out, sort_keys=True))
'''

EPC_SCRIPT = r'''
import json, math, sys
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as rd
TB = sys.argv[1]
mirror = rd.Mesh(cuda.Array3f([-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
                 cuda.Array3i([0, 0], [1, 2], [2, 3]))
scene = rd.Scene(trace_backend=TB); scene.add_mesh(mirror); scene.build()
ray = rd.Ray(cuda.Array3f([0.0], [-0.5], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
rx = cuda.Array3f([0.0], [1.5], [-1.0])
out = {}
opt = rd.ReflEpcOptions()
opt.expected_prim_ids = cuda.Int([1]); opt.surface_group_id = cuda.Int([0, 0])
opt.surface_group_size = cuda.Int([2]); opt.surface_group_members = cuda.Int([0, 1])
opt.surface_max_group_size = 2; opt.visibility_ignore_mode = "surface_group"
g = scene.trace_refl_epc(ray, rx, max_bounces=1, options=opt)
out["g_valid"] = [bool(x) for x in g.valid.numpy().tolist()]
out["g_path"] = [float(x) for x in g.path_length.numpy().tolist()]
out["g_point"] = [float(x) for x in dr.ravel(g.reflection_points).numpy().tolist()]
out["g_group"] = [int(x) for x in g.surface_group_ids.numpy().tolist()]
out["g_resolved"] = [int(x) for x in g.resolved_prim_ids.numpy().tolist()]
# EPC field (direct-primary discovery + backend-agnostic field kernel)
fo = rd.ReflEpcFieldOptions()
fo.slot_plane_normal = cuda.Array3f([0.0], [0.0], [1.0]); fo.slot_eta_r = cuda.Float([4.0])
fo.slot_mu_r = cuda.Float([1.0]); fo.slot_sigma = cuda.Float([0.0]); fo.slot_gain = cuda.Float([1.0])
fo.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
fo.omega = 2.0 * math.pi * 3e8 / 0.1; fo.wavelength = 0.1
fr = scene.trace_refl_epc_field(ray, rx, max_bounces=1, options=fo)
out["f_valid"] = [bool(x) for x in fr.valid.numpy().tolist()]
out["f_path"] = [float(x) for x in fr.path_length.numpy().tolist()]
out["f_fx_re"] = [float(x) for x in fr.field_x_re.numpy().tolist()]
out["f_fx_im"] = [float(x) for x in fr.field_x_im.numpy().tolist()]
print(json.dumps(out, sort_keys=True))
'''

DFR_PATHS_SCRIPT = r'''
import json, math, sys
import numpy as np
import drjit.cuda as cuda
import rayd.drjit as pj
TB = sys.argv[1]
scene = pj.Scene(trace_backend=TB)
scene.add_mesh(pj.Mesh(cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0]),
                       cuda.Array3i([0], [1], [2])))
scene.build()
states = pj.DfrStates()
states.count = 1; states.edge_index = cuda.Int([0])
states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0]); states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
states.edge_t_min = cuda.Float([-0.5]); states.edge_t_max = cuda.Float([0.5])
states.n0 = cuda.Array3f([0.0], [1.0], [0.0]); states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
states.prim0 = cuda.Int([-1]); states.prim1 = cuda.Int([-1])
states.exterior_angle = cuda.Float([1.5 * math.pi])
states.src = cuda.Array3f([0.0], [0.0], [1.0]); states.src_power = cuda.Float([1.0])
states.wi = cuda.Array3f([0.0], [0.0], [-1.0]); states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
states.prefix_depth = cuda.Int([0])
mat = pj.DfrMaterial()
mat.eta_r = cuda.Float([4.0]); mat.sigma = cuda.Float([0.0]); mat.mu_r = cuda.Float([1.0])
mat.gain = cuda.Float([1.0]); mat.valid = cuda.Bool([True])
opt = pj.DfrPathOptions()
opt.wavelength = 0.125; opt.k = 50.26548245743669; opt.seed = 17; opt.max_order = 1
opt.max_paths = 4; opt.max_rx = 1; opt.strategy_mask = pj.RAYD_DFR_DIRECT
opt.sample_count = 4; opt.return_geom = 1; opt.receiver_model = pj.RAYD_DFR_MATCHED_ISO
r = scene.trace_dfr_paths(cuda.Array3f([0.0], [0.0], [1.0]), cuda.Array3f([0.0], [0.0], [-1.0]),
                          states, mat, opt, cuda.Bool([True]))
print(json.dumps({
    "capacity": int(r.capacity),
    "count": int(np.asarray(r.count, dtype=np.int32)[0]),
    "valid0": bool(np.asarray(r.valid, dtype=np.bool_)[0]),
    "rx0": int(np.asarray(r.rx_id, dtype=np.int32)[0]),
    "edge0": int(np.asarray(r.edge0, dtype=np.int32)[0]),
    "delay0": float(np.asarray(r.delay, dtype=np.float32)[0]),
    "fx_re": float(np.asarray(r.field_x.real, dtype=np.float32)[0]),
    "fx_im": float(np.asarray(r.field_x.imag, dtype=np.float32)[0]),
    "p0_x": float(np.asarray(r.p0.x, dtype=np.float32)[0]),
}, sort_keys=True))
'''


DFR_ACCUM_SCRIPT = r'''
import json, math, sys
import numpy as np
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as pj
TB = sys.argv[1]


def wedge_scene():
    scene = pj.Scene(trace_backend=TB)
    scene.add_mesh(pj.Mesh(cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0]),
                           cuda.Array3i([0], [1], [2])))
    scene.build()
    return scene


def suffix_scene():
    scene = pj.Scene(trace_backend=TB)
    scene.add_mesh(pj.Mesh(cuda.Array3f([-2.0, 2.0, -2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, 2.0]),
                           cuda.Array3i([0], [1], [2])))
    scene.build()
    return scene


def edge_state(edge_pos, edge_dir, t_min, t_max, src, prim0=-1, prim1=-1, src_power=2.0):
    st = pj.DfrStates(); st.count = 1; st.edge_index = cuda.Int([0])
    st.edge_pos = edge_pos; st.edge_dir = edge_dir
    st.edge_t_min = cuda.Float([t_min]); st.edge_t_max = cuda.Float([t_max])
    st.n0 = cuda.Array3f([0.0], [1.0], [0.0]); st.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
    st.prim0 = cuda.Int([prim0]); st.prim1 = cuda.Int([prim1])
    st.exterior_angle = cuda.Float([1.5 * math.pi])
    st.src = src; st.src_power = cuda.Float([src_power])
    st.wi = cuda.Array3f([0.0], [0.0], [-1.0]); st.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
    st.prefix_depth = cuda.Int([0])
    return st


def grid(axis=2, position=-1.0):
    g = pj.DfrGrid(); g.axis = axis; g.position = position
    g.coord0_min = -1.0; g.coord0_max = 1.0; g.coord1_min = -1.0; g.coord1_max = 1.0
    g.resolution0 = 1; g.resolution1 = 1; g.cell_area = 4.0
    return g


def material():
    m = pj.DfrMaterial(); m.eta_r = cuda.Float([4.0]); m.sigma = cuda.Float([0.0])
    m.mu_r = cuda.Float([1.0]); m.gain = cuda.Float([1.0]); m.valid = cuda.Bool([True])
    return m


def fi(a):
    return int(np.asarray(a, dtype=np.int32)[0])


def ff(a):
    return float(np.asarray(a, dtype=np.float32)[0])


out = {}

# --- accum_dfr_direct: direct + keller (source visibility + no-suffix target) ---
scene = wedge_scene()
st = edge_state(cuda.Array3f([0.0], [0.0], [0.0]), cuda.Array3f([1.0], [0.0], [0.0]),
                -0.5, 0.5, cuda.Array3f([0.0], [0.0], [1.0]))
opt = pj.DfrOptions()
opt.wavelength = 0.125; opt.k = 50.26548245743669; opt.seed = 17; opt.samples = 128
opt.max_order = 1; opt.direct_samples = 64; opt.keller_samples = 64; opt.suffix_samples = 0
opt.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
opt.sample_sequence = pj.RAYD_DFR_HASH; opt.receiver_model = pj.RAYD_DFR_MATCHED_ISO
opt.collect_edge_use = True; opt.collect_debug_counts = True
r = scene.accum_dfr_direct(st, grid(), material(), opt, cuda.Bool([True]))
dr.eval(r.power, r.field_x.real, r.direct_count, r.keller_count, r.vis_rejects,
        r.utd_rejects, r.edge_uses)
out["direct_cells"] = int(r.grid_cell_count)
out["direct_direct_count"] = fi(r.direct_count)
out["direct_keller_count"] = fi(r.keller_count)
out["direct_vis_rejects"] = fi(r.vis_rejects)
out["direct_utd_rejects"] = fi(r.utd_rejects)
out["direct_edge_uses"] = fi(r.edge_uses)
out["direct_power"] = ff(r.power)
out["direct_field_re"] = ff(r.field_x.real)

# --- accum_dfr_direct: suffix (source vis + suffix-first vis + suffix target) ---
scene = suffix_scene()
st = edge_state(cuda.Array3f([0.0], [-1.0], [0.0]), cuda.Array3f([1.0], [0.0], [0.0]),
                -0.25, 0.25, cuda.Array3f([0.0], [-1.0], [1.0]), prim0=0, prim1=0, src_power=1.0)
sopt = pj.DfrOptions()
sopt.wavelength = 0.125; sopt.k = 50.26548245743669; sopt.seed = 41; sopt.samples = 16
sopt.max_order = 1; sopt.direct_samples = 0; sopt.keller_samples = 0; sopt.suffix_samples = 16
sopt.strategy_mask = pj.RAYD_DFR_SUFFIX_REFL
sopt.sample_sequence = pj.RAYD_DFR_HASH; sopt.receiver_model = pj.RAYD_DFR_MATCHED_ISO
sopt.collect_edge_use = True; sopt.collect_debug_counts = True
r = scene.accum_dfr_direct(st, grid(axis=1, position=-2.0), material(), sopt, cuda.Bool([True]))
dr.eval(r.power, r.suffix_count, r.vis_rejects, r.utd_rejects, r.edge_uses)
out["suffix_suffix_count"] = fi(r.suffix_count)
out["suffix_vis_rejects"] = fi(r.vis_rejects)
out["suffix_utd_rejects"] = fi(r.utd_rejects)
out["suffix_edge_uses"] = fi(r.edge_uses)
out["suffix_power"] = ff(r.power)

# --- accum_dfr: chain order 2 (direct + keller) ---
scene = wedge_scene()
initial = edge_state(cuda.Array3f([0.0], [0.0], [0.0]), cuda.Array3f([1.0], [0.0], [0.0]),
                     -0.5, 0.5, cuda.Array3f([0.0], [0.0], [1.0]))
recursive = pj.DfrStates(); recursive.count = 1; recursive.edge_index = cuda.Int([1])
recursive.edge_pos = cuda.Array3f([0.0], [0.5], [0.0]); recursive.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
recursive.edge_t_min = cuda.Float([-0.5]); recursive.edge_t_max = cuda.Float([0.5])
recursive.n0 = cuda.Array3f([0.0], [1.0], [0.0]); recursive.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
recursive.prim0 = cuda.Int([-1]); recursive.prim1 = cuda.Int([-1])
recursive.exterior_angle = cuda.Float([1.5 * math.pi])
recursive.src = cuda.Array3f([0.0], [0.0], [1.0]); recursive.src_power = cuda.Float([1.0])
recursive.wi = cuda.Array3f([0.0], [1.0], [0.0]); recursive.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
recursive.prefix_depth = cuda.Int([0])
copt = pj.DfrOptions()
copt.wavelength = 0.125; copt.k = 50.26548245743669; copt.seed = 41; copt.samples = 288
copt.max_order = 2; copt.direct_samples = 32; copt.keller_samples = 256; copt.suffix_samples = 0
copt.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER
copt.sample_sequence = pj.RAYD_DFR_HASH; copt.receiver_model = pj.RAYD_DFR_MATCHED_ISO
copt.collect_edge_use = True; copt.collect_debug_counts = True
r = scene.accum_dfr(initial, recursive, grid(), material(), copt, cuda.Bool([True]))
dr.eval(r.power, r.direct_count, r.keller_count, r.edge_vis_rejects, r.edge_uses)
out["chain_direct_count"] = fi(r.direct_count)
out["chain_keller_count"] = fi(r.keller_count)
out["chain_edge_vis_rejects"] = fi(r.edge_vis_rejects)
out["chain_edge_uses"] = fi(r.edge_uses)
out["chain_power"] = ff(r.power)

# --- accum_dfr_coherent_direct: simple-state overload ---
scene = wedge_scene()
st = edge_state(cuda.Array3f([0.0], [0.0], [0.0]), cuda.Array3f([1.0], [0.0], [0.0]),
                -0.5, 0.5, cuda.Array3f([0.0], [0.0], [1.0]))
hopt = pj.DfrCoherentOptions()
hopt.wavelength = 0.125; hopt.k = 50.26548245743669; hopt.max_order = 1
hopt.receiver_model = pj.RAYD_DFR_MATCHED_ISO
hopt.select_diffraction_point = True; hopt.prefilter_visibility = True
hopt.collect_debug_counts = True
r = scene.accum_dfr_coherent_direct(st, grid(), material(), hopt, cuda.Bool([True]))
dr.eval(r.direct_field_x.real, r.direct_field_x.imag, r.direct_count,
        r.visibility_reject_count, r.utd_reject_count)
out["coherent_cells"] = int(r.grid_cell_count)
out["coherent_direct_count"] = fi(r.direct_count)
out["coherent_visibility_reject_count"] = fi(r.visibility_reject_count)
out["coherent_utd_reject_count"] = fi(r.utd_reject_count)
out["coherent_field_re"] = ff(r.direct_field_x.real)
out["coherent_field_im"] = ff(r.direct_field_x.imag)

print(json.dumps(out, sort_keys=True))
'''

DFR_ACCUM_AD_SCRIPT = r'''
import json, sys
import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import rayd.drjit as pj
TB = sys.argv[1]

scene = pj.Scene(trace_backend=TB)
scene.add_mesh(pj.Mesh(cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0]),
                       cuda.Array3i([0], [1], [2])))
scene.build()

src = ad.Array3f([0.0], [0.0], [1.0])
dr.enable_grad(src)
states = pj.DfrStatesAD()
states.count = 1; states.edge_index = ad.Int([0])
states.edge_pos = ad.Array3f([0.0], [0.0], [0.0]); states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
states.edge_t_min = ad.Float([-0.5]); states.edge_t_max = ad.Float([0.5])
states.n0 = ad.Array3f([0.0], [1.0], [0.0]); states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
states.prim0 = ad.Int([-1]); states.prim1 = ad.Int([-1])
states.exterior_angle = ad.Float([1.5 * 3.141592653589793])
states.src = src; states.src_power = ad.Float([2.0])
states.wi = ad.Array3f([0.0], [0.0], [-1.0]); states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
states.prefix_depth = ad.Int([0])

grid = pj.DfrGrid()
grid.axis = 2; grid.position = -1.0
grid.coord0_min = -1.0; grid.coord0_max = 1.0; grid.coord1_min = -1.0; grid.coord1_max = 1.0
grid.resolution0 = 1; grid.resolution1 = 1; grid.cell_area = 4.0

mat = pj.DfrMaterialAD()
mat.eta_r = ad.Float([4.0]); mat.sigma = ad.Float([0.0]); mat.mu_r = ad.Float([1.0])
mat.gain = ad.Float([1.0]); mat.valid = ad.Bool([True])

opt = pj.DfrOptions()
opt.wavelength = 0.125; opt.k = 50.26548245743669; opt.seed = 17; opt.samples = 64
opt.max_order = 1; opt.direct_samples = 64; opt.keller_samples = 0
opt.strategy_mask = pj.RAYD_DFR_DIRECT; opt.sample_sequence = pj.RAYD_DFR_HASH
opt.receiver_model = pj.RAYD_DFR_MATCHED_ISO; opt.collect_edge_use = True

result = scene.accum_dfr_direct(states, grid, mat, opt, True)
loss = dr.sum(result.power)
dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
grad_src = dr.grad(src)
dr.eval(result.power, result.direct_count, grad_src)
out = {
    "direct_power": float(result.power[0]),
    "direct_count": int(result.direct_count[0]),
    "direct_grad_src_x": float(grad_src.x[0]),
    "direct_grad_src_y": float(grad_src.y[0]),
    "direct_grad_src_z": float(grad_src.z[0]),
}

# --- chain accum_dfr (order 2) backward ---
csrc = ad.Array3f([0.0], [0.0], [1.0])
dr.enable_grad(csrc)
initial = pj.DfrStatesAD()
initial.count = 1; initial.edge_index = ad.Int([0])
initial.edge_pos = ad.Array3f([0.0], [0.0], [0.0]); initial.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
initial.edge_t_min = ad.Float([-0.5]); initial.edge_t_max = ad.Float([0.5])
initial.n0 = ad.Array3f([0.0], [1.0], [0.0]); initial.n1 = ad.Array3f([0.0], [-1.0], [0.0])
initial.prim0 = ad.Int([-1]); initial.prim1 = ad.Int([-1])
initial.exterior_angle = ad.Float([1.5 * 3.141592653589793])
initial.src = csrc; initial.src_power = ad.Float([2.0])
initial.wi = ad.Array3f([0.0], [0.0], [-1.0]); initial.d0 = ad.Array3f([0.0], [0.0], [-1.0])
initial.prefix_depth = ad.Int([0])
recursive = pj.DfrStatesAD()
recursive.count = 1; recursive.edge_index = ad.Int([1])
recursive.edge_pos = ad.Array3f([0.0], [0.0], [-0.75]); recursive.edge_dir = ad.Array3f([0.0], [1.0], [0.0])
recursive.edge_t_min = ad.Float([-0.5]); recursive.edge_t_max = ad.Float([0.5])
recursive.n0 = ad.Array3f([1.0], [0.0], [0.0]); recursive.n1 = ad.Array3f([-1.0], [0.0], [0.0])
recursive.prim0 = ad.Int([-1]); recursive.prim1 = ad.Int([-1])
recursive.exterior_angle = ad.Float([1.5 * 3.141592653589793])
recursive.src = ad.Array3f([0.0], [0.0], [0.0]); recursive.src_power = ad.Float([1.0])
recursive.wi = ad.Array3f([0.0], [0.0], [-1.0]); recursive.d0 = ad.Array3f([0.0], [0.0], [-1.0])
recursive.prefix_depth = ad.Int([0])
cgrid = pj.DfrGrid()
cgrid.axis = 2; cgrid.position = -1.5
cgrid.coord0_min = -1.0; cgrid.coord0_max = 1.0; cgrid.coord1_min = -1.0; cgrid.coord1_max = 1.0
cgrid.resolution0 = 1; cgrid.resolution1 = 1; cgrid.cell_area = 4.0
copt = pj.DfrOptions()
copt.wavelength = 0.125; copt.k = 50.26548245743669; copt.seed = 17; copt.samples = 64
copt.max_order = 2; copt.direct_samples = 32; copt.keller_samples = 32
copt.strategy_mask = pj.RAYD_DFR_DIRECT | pj.RAYD_DFR_KELLER; copt.sample_sequence = pj.RAYD_DFR_HASH
copt.receiver_model = pj.RAYD_DFR_MATCHED_ISO; copt.collect_edge_use = True; copt.collect_debug_counts = True
cresult = scene.accum_dfr(initial, recursive, cgrid, mat, copt, True)
closs = dr.sum(cresult.power)
dr.backward(closs, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
cgrad = dr.grad(csrc)
dr.eval(cresult.power, cgrad)
out["chain_power"] = float(cresult.power[0])
out["chain_grad_src_x"] = float(cgrad.x[0])
out["chain_grad_src_z"] = float(cgrad.z[0])

print(json.dumps(out, sort_keys=True))
'''


def _run(script, backend, timeout=240):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), backend],
        cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess ({backend}) failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


class CudaMultipathParityTests(unittest.TestCase):
    DISCRETE_PREFIXES = ("refl_bounce_count", "refl_shape_ids", "refl_global_prim_ids",
                         "visible", "edge_visible", "chain_", "acc_count", "g_valid", "g_group",
                         "g_resolved", "f_valid", "capacity", "count", "valid0", "rx0", "edge0")
    FIELD_KEYS = ("acc_field", "acc_power", "f_fx", "fx_re", "fx_im", "g_path", "f_path", "delay0")

    def _compare(self, optix, cuda):
        for key in sorted(optix):
            with self.subTest(key=key):
                ov, cv = optix[key], cuda[key]
                is_discrete = any(key.startswith(p) or key == p for p in self.DISCRETE_PREFIXES)
                is_field = any(key.startswith(p) for p in self.FIELD_KEYS)
                if isinstance(ov, list) and ov and isinstance(ov[0], (bool, int)) or is_discrete:
                    self.assertEqual(ov, cv, f"discrete mismatch for {key}")
                else:
                    abs_tol = FIELD_ABS if is_field else DEFAULT_ABS
                    rel_tol = FIELD_REL if is_field else DEFAULT_REL
                    ovl = ov if isinstance(ov, list) else [ov]
                    cvl = cv if isinstance(cv, list) else [cv]
                    for a, b in zip(ovl, cvl):
                        if math.isinf(a) or math.isnan(a):
                            self.assertEqual(repr(a), repr(b), f"sentinel mismatch {key}")
                            continue
                        ad = abs(a - b)
                        self.assertTrue(ad <= abs_tol or ad <= rel_tol * abs(a),
                                        f"{key}: optix={a} cuda={b} abs={ad}")

    def test_reflection_visibility_accumulation_parity(self):
        self._compare(_run(SCENE_SCRIPT, "optix"), _run(SCENE_SCRIPT, "cuda"))

    def test_reflection_epc_and_field_parity(self):
        optix = _run(EPC_SCRIPT, "optix")
        cuda = _run(EPC_SCRIPT, "cuda")
        # Surface-group EPC path must be valid and bit-identical.
        self.assertTrue(optix["g_valid"][0])
        self._compare(optix, cuda)

    def test_diffraction_paths_parity(self):
        optix = _run(DFR_PATHS_SCRIPT, "optix")
        cuda = _run(DFR_PATHS_SCRIPT, "cuda")
        self.assertTrue(optix["valid0"])
        self._compare(optix, cuda)

    def _compare_dfr_accum(self, optix, cuda):
        # Discrete keys (sample/reject/edge counts and grid-cell counts) must be
        # BIT-IDENTICAL; power/field keys match within operations.json field tol.
        for key in sorted(optix):
            with self.subTest(key=key):
                ov, cv = optix[key], cuda[key]
                if key.endswith("_power") or "_field" in key:
                    ad = abs(ov - cv)
                    self.assertTrue(ad <= FIELD_ABS or ad <= FIELD_REL * abs(ov),
                                    f"{key}: optix={ov} cuda={cv} abs={ad}")
                else:
                    self.assertEqual(ov, cv, f"discrete mismatch for {key}")

    def test_diffraction_accumulation_parity(self):
        optix = _run(DFR_ACCUM_SCRIPT, "optix")
        cuda = _run(DFR_ACCUM_SCRIPT, "cuda")
        # Every entry point produced work (guards against a silently-empty A/B).
        self.assertGreater(optix["direct_direct_count"], 0)
        self.assertGreater(optix["direct_keller_count"], 0)
        self.assertGreater(optix["suffix_suffix_count"], 0)
        self.assertGreater(optix["chain_direct_count"] + optix["chain_keller_count"], 0)
        self.assertGreater(optix["coherent_direct_count"], 0)
        self._compare_dfr_accum(optix, cuda)

    def test_diffraction_accumulation_ad_parity(self):
        # dr.backward through accum_dfr_direct and accum_dfr (chain) on both
        # backends; the AD custom op re-runs the eager forward (CUDA arm) and its
        # own backend-agnostic backward tape, so the source gradients must match
        # within the gradient tolerance (5e-4).
        optix = _run(DFR_ACCUM_AD_SCRIPT, "optix")
        cuda = _run(DFR_ACCUM_AD_SCRIPT, "cuda")
        self.assertEqual(optix["direct_count"], cuda["direct_count"])
        self.assertNotEqual(optix["direct_grad_src_z"], 0.0)
        self.assertNotEqual(optix["chain_grad_src_z"], 0.0)
        grad_keys = ("direct_grad_src_x", "direct_grad_src_y", "direct_grad_src_z",
                     "chain_grad_src_x", "chain_grad_src_z")
        for key in grad_keys:
            ov, cv = optix[key], cuda[key]
            with self.subTest(key=key):
                ad = abs(ov - cv)
                self.assertTrue(ad <= 5e-4 or ad <= 5e-4 * abs(ov),
                                f"{key}: optix={ov} cuda={cv} abs={ad}")

    def test_diffraction_accumulation_stress(self):
        # 10 consecutive CUDA runs must be deterministic (no literal-materialization
        # race across the staged phases / __constant__ staging).
        first = _run(DFR_ACCUM_SCRIPT, "cuda")
        for _ in range(9):
            self._compare_dfr_accum(first, _run(DFR_ACCUM_SCRIPT, "cuda"))


if __name__ == "__main__":
    unittest.main()
